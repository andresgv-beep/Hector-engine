// kernels/matmul_hqs_compact.cu
// ============================================================================
// MATMUL HQS COMPACT — GEMV for HQ4.1K and HQ5.1K (compact 40-byte header)
// ============================================================================
//
// Separated from matmul_hqs.cu to prevent nvcc register pressure spillover
// that degrades HQ4K/HQ5K kernel performance.
//

#include "hqs_common.cuh"
#include <cuda_fp16.h>
#include <unordered_map>

namespace helios {
namespace kernels {

// Same configs as matmul_hqs.cu — must match
constexpr int COMPACT_MAX_K_SHARED = 16384;

// Batch (M>1): a partir de este M compensa dequant completo + cuBLAS GEMM
// frente al bucle de M GEMVs (el dequant lee el peso UNA vez; el bucle M veces)
constexpr int COMPACT_GEMM_THRESHOLD = 4;

void launch_matmul_hq41k_cublas(const half*, const uint8_t*, half*, int, int, int, cudaStream_t);
void launch_matmul_hq51k_cublas(const half*, const uint8_t*, half*, int, int, int, cudaStream_t);

constexpr int CCA_WPR = 4, CCA_RPB = 4, CCA_BLOCK = CCA_WPR * CCA_RPB * 32;
constexpr int CCB_WPR = 2, CCB_RPB = 8, CCB_BLOCK = CCB_WPR * CCB_RPB * 32;
constexpr int CCC_WPR = 4, CCC_RPB = 1, CCC_BLOCK = CCC_WPR * CCC_RPB * 32;

// ============================================================================
// GEMV HQ4.1K KERNEL
// ============================================================================

// Lecturas globales directas y coalescibles; camino rápido sin bounds-check y
// lecturas half2 del input.
template<int WARPS_PER_ROW, int ROWS_PER_BLOCK>
__global__ void gemv_hq41k_kernel(
    const half* __restrict__ input,
    const uint8_t* __restrict__ weights,
    half* __restrict__ output,
    int K, int N
) {
    using namespace hqs;
    extern __shared__ half s_input[];
    {
        const int BS = WARPS_PER_ROW * ROWS_PER_BLOCK * 32;
        const float4* src = reinterpret_cast<const float4*>(input);
        float4* dst = reinterpret_cast<float4*>(s_input);
        const int n_vec = K / 8;
        for (int i = threadIdx.x; i < n_vec; i += BS) dst[i] = src[i];
        for (int i = n_vec * 8 + threadIdx.x; i < K; i += BS) s_input[i] = input[i];
    }
    __syncthreads();

    const int threads_per_row = WARPS_PER_ROW * 32;
    const int row_group = threadIdx.x / threads_per_row;
    const int local_tid = threadIdx.x % threads_per_row;
    const int warp_in_group = local_tid / 32;
    const int lane_id = local_tid % 32;
    const int row = blockIdx.x * ROWS_PER_BLOCK + row_group;
    if (row >= N) return;

    float* s_partial = reinterpret_cast<float*>(s_input + K);
    const int total_sb = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    const uint8_t* row_weights = weights + (size_t)row * total_sb * HQ41K_BLOCK_SIZE;
    float acc = 0.0f;

    for (int sb = warp_in_group; sb < total_sb; sb += WARPS_PER_ROW) {
        const int sb_base_k = sb * SUPER_BLOCK_SIZE;
        const uint32_t* blk32 = reinterpret_cast<const uint32_t*>(
            row_weights + (size_t)sb * HQ41K_BLOCK_SIZE);

        uint32_t h0 = blk32[0];
        uint32_t h1 = blk32[1];
        float d_scale  = __half2float(__ushort_as_half(h0 & 0xFFFF));
        float d_min    = __half2float(__ushort_as_half(h0 >> 16));
        float min_base = __half2float(__ushort_as_half(h1 & 0xFFFF));

        const uint8_t* sb8 = reinterpret_cast<const uint8_t*>(blk32);
        uint8_t s_packed = sb8[8 + (lane_id >> 1)];
        uint8_t q_s = (lane_id & 1) ? (s_packed & 0x0F) : (s_packed >> 4);
        uint8_t m_packed = sb8[24 + (lane_id >> 1)];
        uint8_t q_m = (lane_id & 1) ? (m_packed & 0x0F) : (m_packed >> 4);

        float scoeff = d_scale * float(q_s) * (1.0f / 15.0f) * (1.0f / HQ4K_Q_MAX);
        float min_f  = fmaf(d_min, float(q_m) * (1.0f / 15.0f), min_base);

        // Payload: word 10 + lane (bytes 40..168)
        uint32_t packed = blk32[10 + lane_id];
        const int k_base = sb_base_k + lane_id * GROUP_SIZE;
        if (k_base + GROUP_SIZE <= K) {
            const half2* in2 = reinterpret_cast<const half2*>(s_input + k_base);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint8_t byte_val = (packed >> (i * 8)) & 0xFF;
                float w0 = fmaf(float((byte_val >> 4) & 0x0F), scoeff, min_f);
                float w1 = fmaf(float(byte_val & 0x0F), scoeff, min_f);
                float2 x = __half22float2(in2[i]);
                acc = fmaf(w0, x.x, acc);
                acc = fmaf(w1, x.y, acc);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint8_t byte_val = (packed >> (i * 8)) & 0xFF;
                float w0 = fmaf(float((byte_val >> 4) & 0x0F), scoeff, min_f);
                float w1 = fmaf(float(byte_val & 0x0F), scoeff, min_f);
                const int k0 = k_base + i * 2;
                const int k1 = k0 + 1;
                if (k0 < K) acc = fmaf(w0, __half2float(s_input[k0]), acc);
                if (k1 < K) acc = fmaf(w1, __half2float(s_input[k1]), acc);
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (lane_id == 0) s_partial[row_group * WARPS_PER_ROW + warp_in_group] = acc;
    __syncthreads();
    if (warp_in_group == 0 && lane_id < WARPS_PER_ROW) {
        float val = s_partial[row_group * WARPS_PER_ROW + lane_id];
        #pragma unroll
        for (int offset = WARPS_PER_ROW / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane_id == 0) output[row] = __float2half(val);
    }
}

// ============================================================================
// GEMV HQ5.1K KERNEL
// ============================================================================

// Lecturas directas y coalescibles desde global. El staging completo del bloque
// añadía sincronizaciones y en lm_head quedaba por debajo del ancho de banda que
// alcanza esta ruta. El decode usa __funnelshift_r para evitar shifts de 64 bit.
template<int WARPS_PER_ROW, int ROWS_PER_BLOCK>
__global__ void gemv_hq51k_kernel(
    const half* __restrict__ input,
    const uint8_t* __restrict__ weights,
    half* __restrict__ output,
    int K, int N
) {
    using namespace hqs;
    extern __shared__ half s_input[];
    {
        const int BS = WARPS_PER_ROW * ROWS_PER_BLOCK * 32;
        const float4* src = reinterpret_cast<const float4*>(input);
        float4* dst = reinterpret_cast<float4*>(s_input);
        const int n_vec = K / 8;
        for (int i = threadIdx.x; i < n_vec; i += BS) dst[i] = src[i];
        for (int i = n_vec * 8 + threadIdx.x; i < K; i += BS) s_input[i] = input[i];
    }
    __syncthreads();

    const int threads_per_row = WARPS_PER_ROW * 32;
    const int row_group = threadIdx.x / threads_per_row;
    const int local_tid = threadIdx.x % threads_per_row;
    const int warp_in_group = local_tid / 32;
    const int lane_id = local_tid % 32;
    const int row = blockIdx.x * ROWS_PER_BLOCK + row_group;
    if (row >= N) return;

    float* s_partial = reinterpret_cast<float*>(s_input + K);
    const int total_sb = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    const uint8_t* row_weights = weights + (size_t)row * total_sb * HQ51K_BLOCK_SIZE;
    float acc = 0.0f;

    for (int sb = warp_in_group; sb < total_sb; sb += WARPS_PER_ROW) {
        const int sb_base_k = sb * SUPER_BLOCK_SIZE;
        const uint32_t* blk32 = reinterpret_cast<const uint32_t*>(
            row_weights + (size_t)sb * HQ51K_BLOCK_SIZE);

        uint32_t h0 = blk32[0];
        uint32_t h1 = blk32[1];
        float d_scale  = __half2float(__ushort_as_half(h0 & 0xFFFF));
        float d_min    = __half2float(__ushort_as_half(h0 >> 16));
        float min_base = __half2float(__ushort_as_half(h1 & 0xFFFF));

        const uint8_t* sb8 = reinterpret_cast<const uint8_t*>(blk32);
        uint8_t s_packed = sb8[8 + (lane_id >> 1)];
        uint8_t q_s = (lane_id & 1) ? (s_packed & 0x0F) : (s_packed >> 4);
        uint8_t m_packed = sb8[24 + (lane_id >> 1)];
        uint8_t q_m = (lane_id & 1) ? (m_packed & 0x0F) : (m_packed >> 4);

        float scoeff = d_scale * float(q_s) * (1.0f / 15.0f) * (1.0f / HQ5K_Q_MAX);
        float min_f  = fmaf(d_min, float(q_m) * (1.0f / 15.0f), min_base);

        // 5 bytes del lane: payload empieza en byte 40 → offset 40 + 5*lane
        const int byte_off = COMPACT_HEADER_SIZE + lane_id * 5;
        const int w0 = byte_off >> 2;
        const int sh = (byte_off & 3) * 8;
        uint32_t pw0 = blk32[w0];
        uint32_t pw1 = blk32[w0 + 1];
        uint32_t lo = __funnelshift_r(pw0, pw1, sh);  // bits 0..31 del paquete
        uint32_t hi = pw1 >> sh;                      // bits 32..39

        const int k_base = sb_base_k + lane_id * GROUP_SIZE;
        if (k_base + GROUP_SIZE <= K) {
            #pragma unroll
            for (int i = 0; i < 7; i++) {
                uint32_t q = __funnelshift_r(lo, hi, i * 5) & 0x1F;
                acc = fmaf(fmaf(float(q), scoeff, min_f),
                           __half2float(s_input[k_base + i]), acc);
            }
            uint32_t q7 = (hi >> 3) & 0x1F;
            acc = fmaf(fmaf(float(q7), scoeff, min_f),
                       __half2float(s_input[k_base + 7]), acc);
        } else {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint32_t q = (i < 7) ? (__funnelshift_r(lo, hi, i * 5) & 0x1F)
                                     : ((hi >> 3) & 0x1F);
                int k_idx = k_base + i;
                if (k_idx < K) {
                    acc = fmaf(fmaf(float(q), scoeff, min_f),
                               __half2float(s_input[k_idx]), acc);
                }
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (lane_id == 0) s_partial[row_group * WARPS_PER_ROW + warp_in_group] = acc;
    __syncthreads();
    if (warp_in_group == 0 && lane_id < WARPS_PER_ROW) {
        float val = s_partial[row_group * WARPS_PER_ROW + lane_id];
        #pragma unroll
        for (int offset = WARPS_PER_ROW / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane_id == 0) output[row] = __float2half(val);
    }
}

// ============================================================================
// LAUNCHERS
// ============================================================================

// smem: input y parciales de la reducción entre warps.
static void launch_hq41k_A(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    int nb = (N + CCA_RPB - 1) / CCA_RPB;
    size_t smem = K * sizeof(half) + CCA_RPB * CCA_WPR * sizeof(float);
    gemv_hq41k_kernel<CCA_WPR, CCA_RPB><<<nb, CCA_BLOCK, smem, s>>>(in, w, out, K, N);
}
static void launch_hq41k_B(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    int nb = (N + CCB_RPB - 1) / CCB_RPB;
    size_t smem = K * sizeof(half) + CCB_RPB * CCB_WPR * sizeof(float);
    gemv_hq41k_kernel<CCB_WPR, CCB_RPB><<<nb, CCB_BLOCK, smem, s>>>(in, w, out, K, N);
}
static void launch_hq41k_C(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    int nb = (N + CCC_RPB - 1) / CCC_RPB;
    size_t smem = K * sizeof(half) + CCC_RPB * CCC_WPR * sizeof(float);
    gemv_hq41k_kernel<CCC_WPR, CCC_RPB><<<nb, CCC_BLOCK, smem, s>>>(in, w, out, K, N);
}
static void launch_hq51k_A(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    int nb = (N + CCA_RPB - 1) / CCA_RPB;
    size_t smem = K * sizeof(half) + CCA_RPB * CCA_WPR * sizeof(float);
    gemv_hq51k_kernel<CCA_WPR, CCA_RPB><<<nb, CCA_BLOCK, smem, s>>>(in, w, out, K, N);
}
static void launch_hq51k_B(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    int nb = (N + CCB_RPB - 1) / CCB_RPB;
    size_t smem = K * sizeof(half) + CCB_RPB * CCB_WPR * sizeof(float);
    gemv_hq51k_kernel<CCB_WPR, CCB_RPB><<<nb, CCB_BLOCK, smem, s>>>(in, w, out, K, N);
}
static void launch_hq51k_C(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    int nb = (N + CCC_RPB - 1) / CCC_RPB;
    size_t smem = K * sizeof(half) + CCC_RPB * CCC_WPR * sizeof(float);
    gemv_hq51k_kernel<CCC_WPR, CCC_RPB><<<nb, CCC_BLOCK, smem, s>>>(in, w, out, K, N);
}

// ============================================================================
// AUTO-TUNE BENCHMARK (local copy to keep TU independent)
// ============================================================================

static float benchmark_compact_kernel(
    void(*launcher)(const half*, const uint8_t*, half*, int, int, cudaStream_t),
    const half* input, const uint8_t* weights, half* output,
    int K, int N, cudaStream_t stream
) {
    for (int i = 0; i < 3; i++) launcher(input, weights, output, K, N, stream);
    cudaStreamSynchronize(stream);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    for (int i = 0; i < 10; i++) launcher(input, weights, output, K, N, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 10.0f;
}

// ============================================================================
// DISPATCH
// ============================================================================

static std::unordered_map<uint64_t, int> s_tune_cache_hq41k;
static std::unordered_map<uint64_t, int> s_tune_cache_hq51k;

void launch_matmul_hq41k(
    const half* input, const uint8_t* weights, half* output,
    int M, int K, int N, cudaStream_t stream
) {
    if (M == 1 && K <= COMPACT_MAX_K_SHARED) {
        uint64_t key = ((uint64_t)K << 32) | (uint64_t)N;
        auto it = s_tune_cache_hq41k.find(key);
        if (it == s_tune_cache_hq41k.end()) {
            float ms_a = benchmark_compact_kernel(launch_hq41k_A, input, weights, output, K, N, stream);
            float ms_b = benchmark_compact_kernel(launch_hq41k_B, input, weights, output, K, N, stream);
            float ms_c = benchmark_compact_kernel(launch_hq41k_C, input, weights, output, K, N, stream);
            int best = 0;
            float best_ms = ms_a;
            if (ms_b < best_ms) { best = 1; best_ms = ms_b; }
            if (ms_c < best_ms) { best = 2; best_ms = ms_c; }
            s_tune_cache_hq41k[key] = best;
            it = s_tune_cache_hq41k.find(key);
        }
        switch (it->second) {
            case 0: launch_hq41k_A(input, weights, output, K, N, stream); break;
            case 1: launch_hq41k_B(input, weights, output, K, N, stream); break;
            case 2: launch_hq41k_C(input, weights, output, K, N, stream); break;
        }
    } else if (M >= COMPACT_GEMM_THRESHOLD) {
        // Batch real: dequant una vez + tensor cores
        launch_matmul_hq41k_cublas(input, weights, output, M, K, N, stream);
    } else {
        for (int m = 0; m < M; m++) {
            launch_matmul_hq41k(input + m * K, weights, output + m * N, 1, K, N, stream);
        }
    }
}

void launch_matmul_hq51k(
    const half* input, const uint8_t* weights, half* output,
    int M, int K, int N, cudaStream_t stream
) {
    if (M == 1 && K <= COMPACT_MAX_K_SHARED) {
        uint64_t key = ((uint64_t)K << 32) | (uint64_t)N;
        auto it = s_tune_cache_hq51k.find(key);
        if (it == s_tune_cache_hq51k.end()) {
            float ms_a = benchmark_compact_kernel(launch_hq51k_A, input, weights, output, K, N, stream);
            float ms_b = benchmark_compact_kernel(launch_hq51k_B, input, weights, output, K, N, stream);
            float ms_c = benchmark_compact_kernel(launch_hq51k_C, input, weights, output, K, N, stream);
            int best = 0;
            float best_ms = ms_a;
            if (ms_b < best_ms) { best = 1; best_ms = ms_b; }
            if (ms_c < best_ms) { best = 2; best_ms = ms_c; }
            s_tune_cache_hq51k[key] = best;
            it = s_tune_cache_hq51k.find(key);
        }
        switch (it->second) {
            case 0: launch_hq51k_A(input, weights, output, K, N, stream); break;
            case 1: launch_hq51k_B(input, weights, output, K, N, stream); break;
            case 2: launch_hq51k_C(input, weights, output, K, N, stream); break;
        }
    } else if (M >= COMPACT_GEMM_THRESHOLD) {
        // Batch real: dequant una vez + tensor cores
        launch_matmul_hq51k_cublas(input, weights, output, M, K, N, stream);
    } else {
        for (int m = 0; m < M; m++) {
            launch_matmul_hq51k(input + m * K, weights, output + m * N, 1, K, N, stream);
        }
    }
}

} // namespace kernels
} // namespace helios
