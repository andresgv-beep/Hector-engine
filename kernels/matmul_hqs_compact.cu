// kernels/matmul_hqs_compact.cu
// ============================================================================
// MATMUL HQS COMPACT — GEMV for HQ4.1K and HQ5.1K (compact 40-byte header)
// ============================================================================
//
// Separated from matmul_hqs.cu to prevent nvcc register pressure spillover
// that degrades HQ4.1K/HQ5.1K kernel performance.
//

#include "hqs_common.cuh"
#include <cuda_fp16.h>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <sys/stat.h>

namespace helios {
namespace kernels {

// Same configs as matmul_hqs.cu — must match
// Ventana maxima del vector de entrada que se aloja en shared. Las matrices
// mas anchas se recorren en varias ventanas; el acumulador permanece en fp32.
// Debe ser multiplo de SUPER_BLOCK_SIZE y de 8 (copia vectorizada float4).
constexpr int COMPACT_INPUT_CHUNK = 16384;
static_assert((COMPACT_INPUT_CHUNK / hqs::SUPER_BLOCK_SIZE) % 2 == 0,
              "GEMV variant A alternates accumulators per superblock pair within a chunk");

// Batch (M>1): a partir de este M compensa dequant completo + cuBLAS GEMM
// frente al bucle de M GEMVs (el dequant lee el peso UNA vez; el bucle M veces)
constexpr int COMPACT_GEMM_THRESHOLD = 9;  // medido: el dequant cuesta ~420us fijos, el GEMV 48,8us cada uno -> equilibrio en 9, no en 4

void launch_matmul_hq41k_cublas(const half*, const uint8_t*, half*, int, int, int, cudaStream_t);
void launch_matmul_hq51k_cublas(const half*, const uint8_t*, half*, int, int, int, cudaStream_t);
void launch_matmul_hq42k_cublas(const half*, const uint8_t*, half*, int, int, int, cudaStream_t);
void launch_matmul_hq52k_cublas(const half*, const uint8_t*, half*, int, int, int, cudaStream_t);

// Candidatos del autoajuste (warps por fila x filas por bloque).
//
// Elegidos barriendo las 14 combinaciones legales sobre ocho formas reales de
// Qwen3-4B, Qwen3-8B y Gemma 4. Los anteriores eran 4x4, 2x8 y 4x1: los dos
// primeros no ganaban en NINGUNA forma y el 4x1 llegaba a tardar el doble que
// el mejor en down_proj (211,9 frente a 95,7 us en el 8B). Todo lo bueno vive
// en warps_por_fila de 1 o 2 con 4 u 8 filas por bloque; 1x1 y 8x4 son
// desastrosos en todas.
constexpr int CCA_WPR = 1, CCA_RPB = 4, CCA_BLOCK = CCA_WPR * CCA_RPB * 32;
constexpr int CCB_WPR = 2, CCB_RPB = 8, CCB_BLOCK = CCB_WPR * CCB_RPB * 32;
constexpr int CCC_WPR = 2, CCC_RPB = 4, CCC_BLOCK = CCC_WPR * CCC_RPB * 32;

// ============================================================================
// GEMV HQ4.1K KERNEL
// ============================================================================

// Lecturas globales directas y coalescibles; camino rápido sin bounds-check y
// lecturas half2 del input.
// One warp per row (variant A) keeps two accumulators and alternates them per
// superblock, reducing each like the two warps of B/C and adding them in the
// same order: the tuned variant then changes speed, never the result.
template<int WARPS_PER_ROW>
__device__ __forceinline__ void rotate_single_warp(float& a, float& b) {
    if constexpr (WARPS_PER_ROW == 1) { const float t = a; a = b; b = t; }
}

template<int WARPS_PER_ROW, int ROWS_PER_BLOCK>
__global__ void gemv_hq41k_kernel(
    const half* __restrict__ input,
    const uint8_t* __restrict__ weights,
    half* __restrict__ output,
    int K, int N
) {
    using namespace hqs;
    extern __shared__ half s_input[];
    const int staged_k = K < COMPACT_INPUT_CHUNK ? K : COMPACT_INPUT_CHUNK;
    float* s_partial = reinterpret_cast<float*>(s_input + staged_k);

    const int threads_per_row = WARPS_PER_ROW * 32;
    const int row_group = threadIdx.x / threads_per_row;
    const int local_tid = threadIdx.x % threads_per_row;
    const int warp_in_group = local_tid / 32;
    const int lane_id = local_tid % 32;
    const int row = blockIdx.x * ROWS_PER_BLOCK + row_group;
    const int total_sb = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    const uint8_t* row_weights = row < N
        ? weights + (size_t)row * total_sb * HQ41K_BLOCK_SIZE
        : weights;
    float acc = 0.0f;
    float acc_odd = 0.0f;

    // K puede superar la capacidad de shared (Qwen2.5-Coder: 18944). Cada
    // ventana se copia una vez por bloque y conserva el mismo orden de suma
    // por warp que el camino de una sola ventana.
    for (int chunk_base = 0; chunk_base < K; chunk_base += COMPACT_INPUT_CHUNK) {
        const int chunk_len = min(COMPACT_INPUT_CHUNK, K - chunk_base);
        const int BS = WARPS_PER_ROW * ROWS_PER_BLOCK * 32;
        const float4* src = reinterpret_cast<const float4*>(input + chunk_base);
        float4* dst = reinterpret_cast<float4*>(s_input);
        const int n_vec = chunk_len / 8;
        for (int i = threadIdx.x; i < n_vec; i += BS) dst[i] = src[i];
        for (int i = n_vec * 8 + threadIdx.x; i < chunk_len; i += BS)
            s_input[i] = input[chunk_base + i];
        __syncthreads();

        const int sb_begin = chunk_base / SUPER_BLOCK_SIZE;
        const int sb_end = (chunk_base + chunk_len + SUPER_BLOCK_SIZE - 1) /
                           SUPER_BLOCK_SIZE;
        for (int sb = sb_begin + warp_in_group;
             row < N && sb < sb_end; sb += WARPS_PER_ROW, rotate_single_warp<WARPS_PER_ROW>(acc, acc_odd)) {
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

        float scoeff = d_scale * float(q_s) * (1.0f / 15.0f) * (1.0f / Q_MAX_4BIT);
        float min_f  = fmaf(d_min, float(q_m) * (1.0f / 15.0f), min_base);

        // Payload: word 10 + lane (bytes 40..168)
        uint32_t packed = blk32[10 + lane_id];
        const int global_k_base = sb_base_k + lane_id * GROUP_SIZE;
        const int local_k_base = global_k_base - chunk_base;
        if (global_k_base + GROUP_SIZE <= K) {
            const half2* in2 = reinterpret_cast<const half2*>(s_input + local_k_base);
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
                const int k0 = global_k_base + i * 2;
                const int k1 = k0 + 1;
                if (k0 < K) acc = fmaf(w0, __half2float(s_input[k0 - chunk_base]), acc);
                if (k1 < K) acc = fmaf(w1, __half2float(s_input[k1 - chunk_base]), acc);
            }
        }
        }
        __syncthreads();
    }

    // Chunks hold an even number of superblocks; an odd total leaves them swapped.
    if (((K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE) & 1) rotate_single_warp<WARPS_PER_ROW>(acc, acc_odd);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        if constexpr (WARPS_PER_ROW == 1) acc_odd += __shfl_down_sync(0xFFFFFFFF, acc_odd, offset);
    }
    if constexpr (WARPS_PER_ROW == 1) acc += acc_odd;
    if (lane_id == 0) s_partial[row_group * WARPS_PER_ROW + warp_in_group] = acc;
    __syncthreads();
    if (warp_in_group == 0 && lane_id < WARPS_PER_ROW) {
        float val = s_partial[row_group * WARPS_PER_ROW + lane_id];
        // La mascara debe nombrar SOLO los lanes que entran en la rama. Con
        // 0xFFFFFFFF el shfl espera a 32 lanes de los que solo llegan
        // WARPS_PER_ROW: es UB en Volta+ y hoy funciona de milagro, porque
        // justo despues el kernel termina y el warp se retira. Con cualquier
        // __syncthreads() detras se cuelga en seco (comprobado).
        constexpr unsigned RED_MASK =
            (WARPS_PER_ROW >= 32) ? 0xFFFFFFFFu : ((1u << WARPS_PER_ROW) - 1u);
        #pragma unroll
        for (int offset = WARPS_PER_ROW / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(RED_MASK, val, offset);
        if (lane_id == 0 && row < N) output[row] = __float2half(val);
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
    const int staged_k = K < COMPACT_INPUT_CHUNK ? K : COMPACT_INPUT_CHUNK;
    float* s_partial = reinterpret_cast<float*>(s_input + staged_k);

    const int threads_per_row = WARPS_PER_ROW * 32;
    const int row_group = threadIdx.x / threads_per_row;
    const int local_tid = threadIdx.x % threads_per_row;
    const int warp_in_group = local_tid / 32;
    const int lane_id = local_tid % 32;
    const int row = blockIdx.x * ROWS_PER_BLOCK + row_group;
    const int total_sb = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    const uint8_t* row_weights = row < N
        ? weights + (size_t)row * total_sb * HQ51K_BLOCK_SIZE
        : weights;
    float acc = 0.0f;
    float acc_odd = 0.0f;

    for (int chunk_base = 0; chunk_base < K; chunk_base += COMPACT_INPUT_CHUNK) {
        const int chunk_len = min(COMPACT_INPUT_CHUNK, K - chunk_base);
        const int BS = WARPS_PER_ROW * ROWS_PER_BLOCK * 32;
        const float4* src = reinterpret_cast<const float4*>(input + chunk_base);
        float4* dst = reinterpret_cast<float4*>(s_input);
        const int n_vec = chunk_len / 8;
        for (int i = threadIdx.x; i < n_vec; i += BS) dst[i] = src[i];
        for (int i = n_vec * 8 + threadIdx.x; i < chunk_len; i += BS)
            s_input[i] = input[chunk_base + i];
        __syncthreads();

        const int sb_begin = chunk_base / SUPER_BLOCK_SIZE;
        const int sb_end = (chunk_base + chunk_len + SUPER_BLOCK_SIZE - 1) /
                           SUPER_BLOCK_SIZE;
        for (int sb = sb_begin + warp_in_group;
             row < N && sb < sb_end; sb += WARPS_PER_ROW, rotate_single_warp<WARPS_PER_ROW>(acc, acc_odd)) {
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

        float scoeff = d_scale * float(q_s) * (1.0f / 15.0f) * (1.0f / Q_MAX_5BIT);
        float min_f  = fmaf(d_min, float(q_m) * (1.0f / 15.0f), min_base);

        // 5 bytes del lane: payload empieza en byte 40 → offset 40 + 5*lane
        const int byte_off = COMPACT_HEADER_SIZE + lane_id * 5;
        const int w0 = byte_off >> 2;
        const int sh = (byte_off & 3) * 8;
        uint32_t pw0 = blk32[w0];
        uint32_t pw1 = blk32[w0 + 1];
        uint32_t lo = __funnelshift_r(pw0, pw1, sh);  // bits 0..31 del paquete
        uint32_t hi = pw1 >> sh;                      // bits 32..39

        const int global_k_base = sb_base_k + lane_id * GROUP_SIZE;
        const int local_k_base = global_k_base - chunk_base;
        if (global_k_base + GROUP_SIZE <= K) {
            #pragma unroll
            for (int i = 0; i < 7; i++) {
                uint32_t q = __funnelshift_r(lo, hi, i * 5) & 0x1F;
                acc = fmaf(fmaf(float(q), scoeff, min_f),
                           __half2float(s_input[local_k_base + i]), acc);
            }
            uint32_t q7 = (hi >> 3) & 0x1F;
            acc = fmaf(fmaf(float(q7), scoeff, min_f),
                       __half2float(s_input[local_k_base + 7]), acc);
        } else {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint32_t q = (i < 7) ? (__funnelshift_r(lo, hi, i * 5) & 0x1F)
                                     : ((hi >> 3) & 0x1F);
                int k_idx = global_k_base + i;
                if (k_idx < K) {
                    acc = fmaf(fmaf(float(q), scoeff, min_f),
                               __half2float(s_input[k_idx - chunk_base]), acc);
                }
            }
        }
        }
        __syncthreads();
    }

    // Chunks hold an even number of superblocks; an odd total leaves them swapped.
    if (((K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE) & 1) rotate_single_warp<WARPS_PER_ROW>(acc, acc_odd);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        if constexpr (WARPS_PER_ROW == 1) acc_odd += __shfl_down_sync(0xFFFFFFFF, acc_odd, offset);
    }
    if constexpr (WARPS_PER_ROW == 1) acc += acc_odd;
    if (lane_id == 0) s_partial[row_group * WARPS_PER_ROW + warp_in_group] = acc;
    __syncthreads();
    if (warp_in_group == 0 && lane_id < WARPS_PER_ROW) {
        float val = s_partial[row_group * WARPS_PER_ROW + lane_id];
        // La mascara debe nombrar SOLO los lanes que entran en la rama. Con
        // 0xFFFFFFFF el shfl espera a 32 lanes de los que solo llegan
        // WARPS_PER_ROW: es UB en Volta+ y hoy funciona de milagro, porque
        // justo despues el kernel termina y el warp se retira. Con cualquier
        // __syncthreads() detras se cuelga en seco (comprobado).
        constexpr unsigned RED_MASK =
            (WARPS_PER_ROW >= 32) ? 0xFFFFFFFFu : ((1u << WARPS_PER_ROW) - 1u);
        #pragma unroll
        for (int offset = WARPS_PER_ROW / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(RED_MASK, val, offset);
        if (lane_id == 0 && row < N) output[row] = __float2half(val);
    }
}

// ============================================================================
// GEMV HQ4.2K / HQ5.2K — header simétrico de 24 bytes
// ============================================================================

template<int BITS, int BLOCK_BYTES, int WARPS_PER_ROW, int ROWS_PER_BLOCK>
__global__ void gemv_hq_symmetric_kernel(
    const half* __restrict__ input,
    const uint8_t* __restrict__ weights,
    half* __restrict__ output,
    int K, int N
) {
    using namespace hqs;
    extern __shared__ half s_input[];
    const int staged_k = K < COMPACT_INPUT_CHUNK ? K : COMPACT_INPUT_CHUNK;
    float* s_partial = reinterpret_cast<float*>(s_input + staged_k);
    const int threads_per_row = WARPS_PER_ROW * 32;
    const int row_group = threadIdx.x / threads_per_row;
    const int local_tid = threadIdx.x % threads_per_row;
    const int warp_in_group = local_tid / 32;
    const int lane = local_tid % 32;
    const int row = blockIdx.x * ROWS_PER_BLOCK + row_group;
    const int total_sb = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    const uint8_t* row_weights = row < N
        ? weights + size_t(row) * total_sb * BLOCK_BYTES : weights;
    float acc = 0.0f;
    float acc_odd = 0.0f;

    for (int chunk_base = 0; chunk_base < K; chunk_base += COMPACT_INPUT_CHUNK) {
        const int chunk_len = min(COMPACT_INPUT_CHUNK, K - chunk_base);
        const int BS = WARPS_PER_ROW * ROWS_PER_BLOCK * 32;
        const float4* src = reinterpret_cast<const float4*>(input + chunk_base);
        float4* dst = reinterpret_cast<float4*>(s_input);
        const int n_vec = chunk_len / 8;
        for (int i = threadIdx.x; i < n_vec; i += BS) dst[i] = src[i];
        for (int i = n_vec * 8 + threadIdx.x; i < chunk_len; i += BS)
            s_input[i] = input[chunk_base + i];
        __syncthreads();

        const int sb_begin = chunk_base / SUPER_BLOCK_SIZE;
        const int sb_end = (chunk_base + chunk_len + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
        for (int sb = sb_begin + warp_in_group;
             row < N && sb < sb_end; sb += WARPS_PER_ROW, rotate_single_warp<WARPS_PER_ROW>(acc, acc_odd)) {
            const int sb_base = sb * SUPER_BLOCK_SIZE;
            const uint8_t* block = row_weights + size_t(sb) * BLOCK_BYTES;
            const float step = decode_symmetric_step(block, lane);
            const int global_k = sb_base + lane * GROUP_SIZE;
            const int local_k = global_k - chunk_base;
            if constexpr (BITS == 4) {
                const uint32_t packed = *reinterpret_cast<const uint32_t*>(
                    block + SYMMETRIC_HEADER_SIZE + lane * 4);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint8_t byte = (packed >> (i * 8)) & 0xff;
                    const float w0 = float(int(byte >> 4) - 8) * step;
                    const float w1 = float(int(byte & 0x0f) - 8) * step;
                    const int k0 = global_k + i * 2;
                    if (k0 < K) acc = fmaf(w0, __half2float(s_input[local_k + i * 2]), acc);
                    if (k0 + 1 < K) acc = fmaf(w1, __half2float(s_input[local_k + i * 2 + 1]), acc);
                }
            } else {
                const int offset = SYMMETRIC_HEADER_SIZE + lane * 5;
                uint64_t packed = 0;
                #pragma unroll
                for (int byte = 0; byte < 5; ++byte)
                    packed |= uint64_t(block[offset + byte]) << (byte * 8);
                #pragma unroll
                for (int i = 0; i < GROUP_SIZE; ++i) {
                    const float w = float(int((packed >> (i * 5)) & 0x1f) - 16) * step;
                    const int k = global_k + i;
                    if (k < K) acc = fmaf(w, __half2float(s_input[local_k + i]), acc);
                }
            }
        }
        __syncthreads();
    }

    // Chunks hold an even number of superblocks; an odd total leaves them swapped.
    if (((K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE) & 1) rotate_single_warp<WARPS_PER_ROW>(acc, acc_odd);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
        if constexpr (WARPS_PER_ROW == 1) acc_odd += __shfl_down_sync(0xffffffff, acc_odd, offset);
    }
    if constexpr (WARPS_PER_ROW == 1) acc += acc_odd;
    if (lane == 0) s_partial[row_group * WARPS_PER_ROW + warp_in_group] = acc;
    __syncthreads();
    if (warp_in_group == 0 && lane < WARPS_PER_ROW) {
        float value = s_partial[row_group * WARPS_PER_ROW + lane];
        constexpr unsigned mask = WARPS_PER_ROW >= 32 ? 0xffffffffu : ((1u << WARPS_PER_ROW) - 1u);
        #pragma unroll
        for (int offset = WARPS_PER_ROW / 2; offset > 0; offset >>= 1)
            value += __shfl_down_sync(mask, value, offset);
        if (lane == 0 && row < N) output[row] = __float2half(value);
    }
}

template<int BITS, int BLOCK_BYTES, int WPR, int RPB>
static void launch_symmetric(const half* in, const uint8_t* weights, half* out,
                             int K, int N, cudaStream_t stream) {
    const int threads = WPR * RPB * 32;
    const int blocks = (N + RPB - 1) / RPB;
    const size_t shared = std::min(K, COMPACT_INPUT_CHUNK) * sizeof(half) +
                          RPB * WPR * sizeof(float);
    gemv_hq_symmetric_kernel<BITS, BLOCK_BYTES, WPR, RPB>
        <<<blocks, threads, shared, stream>>>(in, weights, out, K, N);
}

// ============================================================================
// LAUNCHERS
// ============================================================================

// smem: input y parciales de la reducción entre warps.
static void launch_hq41k_A(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    int nb = (N + CCA_RPB - 1) / CCA_RPB;
    size_t smem = std::min(K, COMPACT_INPUT_CHUNK) * sizeof(half) + CCA_RPB * CCA_WPR * sizeof(float);
    gemv_hq41k_kernel<CCA_WPR, CCA_RPB><<<nb, CCA_BLOCK, smem, s>>>(in, w, out, K, N);
}
static void launch_hq41k_B(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    int nb = (N + CCB_RPB - 1) / CCB_RPB;
    size_t smem = std::min(K, COMPACT_INPUT_CHUNK) * sizeof(half) + CCB_RPB * CCB_WPR * sizeof(float);
    gemv_hq41k_kernel<CCB_WPR, CCB_RPB><<<nb, CCB_BLOCK, smem, s>>>(in, w, out, K, N);
}
static void launch_hq41k_C(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    int nb = (N + CCC_RPB - 1) / CCC_RPB;
    size_t smem = std::min(K, COMPACT_INPUT_CHUNK) * sizeof(half) + CCC_RPB * CCC_WPR * sizeof(float);
    gemv_hq41k_kernel<CCC_WPR, CCC_RPB><<<nb, CCC_BLOCK, smem, s>>>(in, w, out, K, N);
}
static void launch_hq51k_A(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    int nb = (N + CCA_RPB - 1) / CCA_RPB;
    size_t smem = std::min(K, COMPACT_INPUT_CHUNK) * sizeof(half) + CCA_RPB * CCA_WPR * sizeof(float);
    gemv_hq51k_kernel<CCA_WPR, CCA_RPB><<<nb, CCA_BLOCK, smem, s>>>(in, w, out, K, N);
}
static void launch_hq51k_B(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    int nb = (N + CCB_RPB - 1) / CCB_RPB;
    size_t smem = std::min(K, COMPACT_INPUT_CHUNK) * sizeof(half) + CCB_RPB * CCB_WPR * sizeof(float);
    gemv_hq51k_kernel<CCB_WPR, CCB_RPB><<<nb, CCB_BLOCK, smem, s>>>(in, w, out, K, N);
}
static void launch_hq51k_C(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    int nb = (N + CCC_RPB - 1) / CCC_RPB;
    size_t smem = std::min(K, COMPACT_INPUT_CHUNK) * sizeof(half) + CCC_RPB * CCC_WPR * sizeof(float);
    gemv_hq51k_kernel<CCC_WPR, CCC_RPB><<<nb, CCC_BLOCK, smem, s>>>(in, w, out, K, N);
}
static void launch_hq42k_A(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    launch_symmetric<4, hqs::HQ42K_BLOCK_SIZE, CCA_WPR, CCA_RPB>(in, w, out, K, N, s);
}
static void launch_hq42k_B(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    launch_symmetric<4, hqs::HQ42K_BLOCK_SIZE, CCB_WPR, CCB_RPB>(in, w, out, K, N, s);
}
static void launch_hq42k_C(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    launch_symmetric<4, hqs::HQ42K_BLOCK_SIZE, CCC_WPR, CCC_RPB>(in, w, out, K, N, s);
}
static void launch_hq52k_A(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    launch_symmetric<5, hqs::HQ52K_BLOCK_SIZE, CCA_WPR, CCA_RPB>(in, w, out, K, N, s);
}
static void launch_hq52k_B(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    launch_symmetric<5, hqs::HQ52K_BLOCK_SIZE, CCB_WPR, CCB_RPB>(in, w, out, K, N, s);
}
static void launch_hq52k_C(const half* in, const uint8_t* w, half* out, int K, int N, cudaStream_t s) {
    launch_symmetric<5, hqs::HQ52K_BLOCK_SIZE, CCC_WPR, CCC_RPB>(in, w, out, K, N, s);
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
static std::unordered_map<uint64_t, int> s_tune_cache_hq42k;
static std::unordered_map<uint64_t, int> s_tune_cache_hq52k;

// ----------------------------------------------------------------------------
// Sidecar del auto-tune: ~/.helios/tune.cache (o $HELIOS_HOME/.helios)
// ----------------------------------------------------------------------------
// El benchmark del primer uso decide A/B/C por forma, y entre candidatas
// empatadas el ganador depende del ruido del cronómetro: el mismo binario
// podía elegir distinto en cada arranque. B y C comparten WARPS_PER_ROW=2
// (mismo árbol de reducción, bits idénticos), pero A parte la suma distinto:
// un cruce A<->B/C cambia la trayectoria greedy certificada. El sidecar
// persiste la primera decisión por forma y por GPU; borrar el fichero fuerza
// re-tune. HELIOS_TUNE_NOCACHE=1 desactiva lectura y escritura.

static bool s_tune_sidecar_loaded = false;
static bool s_tune_sidecar_has_header = false;
static std::string s_tune_sidecar_path;

static std::string tune_device_name() {
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return "unknown";
    return prop.name;
}

static void tune_sidecar_load_once() {
    if (s_tune_sidecar_loaded) return;
    s_tune_sidecar_loaded = true;
    if (getenv("HELIOS_TUNE_NOCACHE")) return;
    // Misma semántica que helios_chat: HELIOS_HOME es el directorio de perfil
    // completo; sin él, ~/.helios.
    const char* custom = getenv("HELIOS_HOME");
    std::string dir;
    if (custom && *custom) {
        dir = custom;
    } else {
        const char* home = getenv("HOME");
        if (!home) return;
        dir = std::string(home) + "/.helios";
    }
    mkdir(dir.c_str(), 0700);  // EEXIST es lo normal
    s_tune_sidecar_path = dir + "/tune.cache";
    FILE* f = fopen(s_tune_sidecar_path.c_str(), "r");
    if (!f) return;
    const std::string device = tune_device_name();
    bool device_ok = false;
    int loaded = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            // Cabecera "# helios tune.cache v1 <GPU>": otra GPU invalida el
            // contenido (las decisiones se recronometran y el fichero se
            // reescribe en el primer append).
            device_ok = strstr(line, device.c_str()) != nullptr;
            continue;
        }
        if (!device_ok) continue;
        char fmt[16]; int K, N; char choice;
        if (sscanf(line, "%15s %d %d %c", fmt, &K, &N, &choice) != 4) continue;
        int best = choice - 'A';
        if (best < 0 || best > 2) continue;
        uint64_t key = ((uint64_t)K << 32) | (uint64_t)N;
        if      (strcmp(fmt, "hq41k") == 0) s_tune_cache_hq41k[key] = best;
        else if (strcmp(fmt, "hq51k") == 0) s_tune_cache_hq51k[key] = best;
        else if (strcmp(fmt, "hq42k") == 0) s_tune_cache_hq42k[key] = best;
        else if (strcmp(fmt, "hq52k") == 0) s_tune_cache_hq52k[key] = best;
        loaded++;
    }
    fclose(f);
    s_tune_sidecar_has_header = device_ok;
    if (getenv("HELIOS_TUNE_DEBUG"))
        fprintf(stderr, "[tune] sidecar: %d decisiones cargadas de %s\n",
                loaded, s_tune_sidecar_path.c_str());
}

static void tune_sidecar_append(const char* fmt, int K, int N, int best) {
    if (s_tune_sidecar_path.empty() || getenv("HELIOS_TUNE_NOCACHE")) return;
    // Sin cabecera válida (fichero nuevo o de otra GPU) se reescribe entero.
    FILE* f = fopen(s_tune_sidecar_path.c_str(), s_tune_sidecar_has_header ? "a" : "w");
    if (!f) return;
    if (!s_tune_sidecar_has_header) {
        fprintf(f, "# helios tune.cache v1 %s\n", tune_device_name().c_str());
        s_tune_sidecar_has_header = true;
    }
    fprintf(f, "%s %d %d %c\n", fmt, K, N, (char)('A' + best));
    fclose(f);
}

void launch_matmul_hq41k(
    const half* input, const uint8_t* weights, half* output,
    int M, int K, int N, cudaStream_t stream
) {
    if (M == 1) {
        uint64_t key = ((uint64_t)K << 32) | (uint64_t)N;
        tune_sidecar_load_once();
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
            tune_sidecar_append("hq41k", K, N, best);
            if (getenv("HELIOS_TUNE_DEBUG")) {
                // ¿Gana algo el autoajuste? Si A/B/C quedan a menos de un 2%,
                // no compensa el coste del primer run.
                float peor = fmaxf(ms_a, fmaxf(ms_b, ms_c));
                fprintf(stderr, "[tune] hq41k K=%d N=%d  A=%.1fus B=%.1fus C=%.1fus"
                        "  -> %c  (mejor vs peor: %.1f%%)\n",
                        K, N, ms_a*1000, ms_b*1000, ms_c*1000, 'A'+best,
                        100.0f*(peor/best_ms - 1.0f));
            }
            it = s_tune_cache_hq41k.find(key);
        }
        switch (it->second) {
            case 0: launch_hq41k_A(input, weights, output, K, N, stream); break;
            case 1: launch_hq41k_B(input, weights, output, K, N, stream); break;
            case 2: launch_hq41k_C(input, weights, output, K, N, stream); break;
        }
    } else if (M >= COMPACT_GEMM_THRESHOLD) {
        // Batch real: dequant una vez + tensor cores. Decode (M=1), incluso
        // con K ancho, nunca materializa la matriz completa en fp16.
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
    if (M == 1) {
        uint64_t key = ((uint64_t)K << 32) | (uint64_t)N;
        tune_sidecar_load_once();
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
            tune_sidecar_append("hq51k", K, N, best);
            it = s_tune_cache_hq51k.find(key);
        }
        switch (it->second) {
            case 0: launch_hq51k_A(input, weights, output, K, N, stream); break;
            case 1: launch_hq51k_B(input, weights, output, K, N, stream); break;
            case 2: launch_hq51k_C(input, weights, output, K, N, stream); break;
        }
    } else if (M >= COMPACT_GEMM_THRESHOLD) {
        launch_matmul_hq51k_cublas(input, weights, output, M, K, N, stream);
    } else {
        for (int m = 0; m < M; m++) {
            launch_matmul_hq51k(input + m * K, weights, output + m * N, 1, K, N, stream);
        }
    }
}

void launch_matmul_hq42k(
    const half* input, const uint8_t* weights, half* output,
    int M, int K, int N, cudaStream_t stream
) {
    if (M == 1) {
        const uint64_t key = (uint64_t(K) << 32) | uint64_t(N);
        tune_sidecar_load_once();
        auto it = s_tune_cache_hq42k.find(key);
        if (it == s_tune_cache_hq42k.end()) {
            const float ms_a = benchmark_compact_kernel(launch_hq42k_A, input, weights, output, K, N, stream);
            const float ms_b = benchmark_compact_kernel(launch_hq42k_B, input, weights, output, K, N, stream);
            const float ms_c = benchmark_compact_kernel(launch_hq42k_C, input, weights, output, K, N, stream);
            int best = 0;
            float best_ms = ms_a;
            if (ms_b < best_ms) { best = 1; best_ms = ms_b; }
            if (ms_c < best_ms) { best = 2; }
            s_tune_cache_hq42k[key] = best;
            tune_sidecar_append("hq42k", K, N, best);
            if (getenv("HELIOS_TUNE_DEBUG"))
                fprintf(stderr, "[tune] hq42k K=%d N=%d A=%.1fus B=%.1fus C=%.1fus -> %c\n",
                        K, N, ms_a * 1000, ms_b * 1000, ms_c * 1000, 'A' + best);
            it = s_tune_cache_hq42k.find(key);
        }
        switch (it->second) {
            case 0: launch_hq42k_A(input, weights, output, K, N, stream); break;
            case 1: launch_hq42k_B(input, weights, output, K, N, stream); break;
            case 2: launch_hq42k_C(input, weights, output, K, N, stream); break;
        }
    } else if (M >= COMPACT_GEMM_THRESHOLD) {
        launch_matmul_hq42k_cublas(input, weights, output, M, K, N, stream);
    } else {
        for (int m = 0; m < M; ++m)
            launch_matmul_hq42k(input + m * K, weights, output + m * N, 1, K, N, stream);
    }
}

void launch_matmul_hq52k(
    const half* input, const uint8_t* weights, half* output,
    int M, int K, int N, cudaStream_t stream
) {
    if (M == 1) {
        const uint64_t key = (uint64_t(K) << 32) | uint64_t(N);
        tune_sidecar_load_once();
        auto it = s_tune_cache_hq52k.find(key);
        if (it == s_tune_cache_hq52k.end()) {
            const float ms_a = benchmark_compact_kernel(launch_hq52k_A, input, weights, output, K, N, stream);
            const float ms_b = benchmark_compact_kernel(launch_hq52k_B, input, weights, output, K, N, stream);
            const float ms_c = benchmark_compact_kernel(launch_hq52k_C, input, weights, output, K, N, stream);
            int best = 0;
            float best_ms = ms_a;
            if (ms_b < best_ms) { best = 1; best_ms = ms_b; }
            if (ms_c < best_ms) { best = 2; }
            s_tune_cache_hq52k[key] = best;
            tune_sidecar_append("hq52k", K, N, best);
            if (getenv("HELIOS_TUNE_DEBUG"))
                fprintf(stderr, "[tune] hq52k K=%d N=%d A=%.1fus B=%.1fus C=%.1fus -> %c\n",
                        K, N, ms_a * 1000, ms_b * 1000, ms_c * 1000, 'A' + best);
            it = s_tune_cache_hq52k.find(key);
        }
        switch (it->second) {
            case 0: launch_hq52k_A(input, weights, output, K, N, stream); break;
            case 1: launch_hq52k_B(input, weights, output, K, N, stream); break;
            case 2: launch_hq52k_C(input, weights, output, K, N, stream); break;
        }
    } else if (M >= COMPACT_GEMM_THRESHOLD) {
        launch_matmul_hq52k_cublas(input, weights, output, M, K, N, stream);
    } else {
        for (int m = 0; m < M; ++m)
            launch_matmul_hq52k(input + m * K, weights, output + m * N, 1, K, N, stream);
    }
}

} // namespace kernels
} // namespace helios
