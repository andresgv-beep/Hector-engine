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

constexpr int CCA_WPR = 4, CCA_RPB = 4, CCA_BLOCK = CCA_WPR * CCA_RPB * 32;
constexpr int CCB_WPR = 2, CCB_RPB = 8, CCB_BLOCK = CCB_WPR * CCB_RPB * 32;
constexpr int CCC_WPR = 4, CCC_RPB = 1, CCC_BLOCK = CCC_WPR * CCC_RPB * 32;

// ============================================================================
// GEMV HQ4.1K KERNEL
// ============================================================================

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
    
    const int total_sb = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    const uint8_t* row_weights = weights + (size_t)row * total_sb * HQ41K_BLOCK_SIZE;
    float acc = 0.0f;
    
    for (int sb = warp_in_group; sb < total_sb; sb += WARPS_PER_ROW) {
        const int sb_base_k = sb * SUPER_BLOCK_SIZE;
        const uint8_t* block_ptr = row_weights + (size_t)sb * HQ41K_BLOCK_SIZE;
        if (lane_id < NUM_GROUPS) {
            float min_f, scoeff;
            decode_compact_group(block_ptr, lane_id, min_f, scoeff, 1.0f / HQ4K_Q_MAX);
            
            const uint32_t* payload32 = reinterpret_cast<const uint32_t*>(
                block_ptr + COMPACT_HEADER_SIZE);
            uint32_t packed = payload32[lane_id];
            const int k_base = sb_base_k + lane_id * GROUP_SIZE;
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
    float* s_partial = reinterpret_cast<float*>(s_input + K);
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
    
    const int total_sb = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    const uint8_t* row_weights = weights + (size_t)row * total_sb * HQ51K_BLOCK_SIZE;
    float acc = 0.0f;
    
    for (int sb = warp_in_group; sb < total_sb; sb += WARPS_PER_ROW) {
        const int sb_base_k = sb * SUPER_BLOCK_SIZE;
        const uint8_t* block_ptr = row_weights + (size_t)sb * HQ51K_BLOCK_SIZE;
        if (lane_id < NUM_GROUPS) {
            float min_f, scoeff;
            decode_compact_group(block_ptr, lane_id, min_f, scoeff, 1.0f / HQ5K_Q_MAX);
            
            const uint8_t* payload = block_ptr + COMPACT_HEADER_SIZE + lane_id * 5;
            uint64_t bits = uint64_t(payload[0])
                          | (uint64_t(payload[1]) << 8)
                          | (uint64_t(payload[2]) << 16)
                          | (uint64_t(payload[3]) << 24)
                          | (uint64_t(payload[4]) << 32);
            const int k_base = sb_base_k + lane_id * GROUP_SIZE;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float w = fmaf(float((bits >> (i * 5)) & 0x1F), scoeff, min_f);
                int k_idx = k_base + i;
                if (k_idx < K) acc = fmaf(w, __half2float(s_input[k_idx]), acc);
            }
        }
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    float* s_partial = reinterpret_cast<float*>(s_input + K);
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
    } else {
        for (int m = 0; m < M; m++) {
            launch_matmul_hq51k(input + m * K, weights, output + m * N, 1, K, N, stream);
        }
    }
}

} // namespace kernels
} // namespace helios
