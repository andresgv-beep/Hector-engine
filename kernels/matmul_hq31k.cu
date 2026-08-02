// MATMUL HQ3.1K — formato compacto de 136 bytes por 256 pesos.
// Ruta aditiva: no comparte dispatch ni modifica los kernels HQ4.1K/HQ5.1K.

#include "hqs_common.cuh"
#include <cuda_fp16.h>
#include <stdexcept>
#include <unordered_map>

namespace helios {
namespace kernels {

constexpr int HQ31K_MAX_K_SHARED = 16384;
constexpr int H31A_WPR = 4, H31A_RPB = 4, H31A_BLOCK = H31A_WPR * H31A_RPB * 32;
constexpr int H31B_WPR = 2, H31B_RPB = 8, H31B_BLOCK = H31B_WPR * H31B_RPB * 32;
constexpr int H31C_WPR = 4, H31C_RPB = 1, H31C_BLOCK = H31C_WPR * H31C_RPB * 32;

template<int WARPS_PER_ROW, int ROWS_PER_BLOCK>
__global__ void gemv_hq31k_kernel(
    const half* __restrict__ input,
    const uint8_t* __restrict__ weights,
    half* __restrict__ output,
    int K, int N
) {
    using namespace hqs;
    extern __shared__ half s_input[];
    {
        const int block_size = WARPS_PER_ROW * ROWS_PER_BLOCK * 32;
        const float4* src = reinterpret_cast<const float4*>(input);
        float4* dst = reinterpret_cast<float4*>(s_input);
        const int n_vec = K / 8;
        for (int i = threadIdx.x; i < n_vec; i += block_size) dst[i] = src[i];
        for (int i = n_vec * 8 + threadIdx.x; i < K; i += block_size)
            s_input[i] = input[i];
    }
    __syncthreads();

    const int threads_per_row = WARPS_PER_ROW * 32;
    const int row_group = threadIdx.x / threads_per_row;
    const int local_tid = threadIdx.x % threads_per_row;
    const int warp_in_group = local_tid / 32;
    const int lane = local_tid % 32;
    const int row = blockIdx.x * ROWS_PER_BLOCK + row_group;
    if (row >= N) return;

    float* s_partial = reinterpret_cast<float*>(s_input + K);
    const int total_sb = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    const uint8_t* row_weights =
        weights + (size_t)row * total_sb * HQ31K_BLOCK_SIZE;
    float acc = 0.0f;

    for (int sb = warp_in_group; sb < total_sb; sb += WARPS_PER_ROW) {
        const uint8_t* block = row_weights + (size_t)sb * HQ31K_BLOCK_SIZE;
        const uint32_t* words = reinterpret_cast<const uint32_t*>(block);

        uint32_t h0 = words[0];
        uint32_t h1 = words[1];
        float d_scale = __half2float(__ushort_as_half(h0 & 0xFFFF));
        float d_min = __half2float(__ushort_as_half(h0 >> 16));
        float min_base = __half2float(__ushort_as_half(h1 & 0xFFFF));
        uint8_t sp = block[8 + (lane >> 1)];
        uint8_t mp = block[24 + (lane >> 1)];
        uint8_t q_s = (lane & 1) ? (sp & 0x0F) : (sp >> 4);
        uint8_t q_m = (lane & 1) ? (mp & 0x0F) : (mp >> 4);
        float scoeff = d_scale * float(q_s) * (1.0f / 15.0f) *
                       (1.0f / Q_MAX_3BIT);
        float min_f = fmaf(d_min, float(q_m) * (1.0f / 15.0f), min_base);

        // Un grupo natural: 8 índices de 3 bits = 24 bits, LSB-first.
        const int byte_off = COMPACT_HEADER_SIZE + lane * 3;
        uint32_t packed = uint32_t(block[byte_off])
                        | (uint32_t(block[byte_off + 1]) << 8)
                        | (uint32_t(block[byte_off + 2]) << 16);
        const int k_base = sb * SUPER_BLOCK_SIZE + lane * GROUP_SIZE;
        if (k_base + GROUP_SIZE <= K) {
            const half2* in2 = reinterpret_cast<const half2*>(s_input + k_base);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint32_t q0 = (packed >> ((i * 2) * 3)) & 0x07;
                uint32_t q1 = (packed >> ((i * 2 + 1) * 3)) & 0x07;
                float2 x = __half22float2(in2[i]);
                acc = fmaf(fmaf(float(q0), scoeff, min_f), x.x, acc);
                acc = fmaf(fmaf(float(q1), scoeff, min_f), x.y, acc);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < GROUP_SIZE; i++) {
                int k_idx = k_base + i;
                if (k_idx < K) {
                    uint32_t q = (packed >> (i * 3)) & 0x07;
                    acc = fmaf(fmaf(float(q), scoeff, min_f),
                               __half2float(s_input[k_idx]), acc);
                }
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (lane == 0)
        s_partial[row_group * WARPS_PER_ROW + warp_in_group] = acc;
    __syncthreads();
    if (warp_in_group == 0 && lane < WARPS_PER_ROW) {
        float value = s_partial[row_group * WARPS_PER_ROW + lane];
        #pragma unroll
        for (int offset = WARPS_PER_ROW / 2; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xFFFFFFFF, value, offset);
        if (lane == 0) output[row] = __float2half(value);
    }
}

using Hq31Launcher = void(*)(const half*, const uint8_t*, half*, int, int, cudaStream_t);

#define HQ31_LAUNCHER(NAME, WPR, RPB, BLOCK)                                   \
static void NAME(const half* input, const uint8_t* weights, half* output,      \
                 int K, int N, cudaStream_t stream) {                          \
    int blocks = (N + RPB - 1) / RPB;                                          \
    size_t smem = K * sizeof(half) + RPB * WPR * sizeof(float);                 \
    gemv_hq31k_kernel<WPR, RPB><<<blocks, BLOCK, smem, stream>>>(               \
        input, weights, output, K, N);                                          \
}

HQ31_LAUNCHER(launch_hq31k_A, H31A_WPR, H31A_RPB, H31A_BLOCK)
HQ31_LAUNCHER(launch_hq31k_B, H31B_WPR, H31B_RPB, H31B_BLOCK)
HQ31_LAUNCHER(launch_hq31k_C, H31C_WPR, H31C_RPB, H31C_BLOCK)

static float benchmark_hq31k(
    Hq31Launcher launcher, const half* input, const uint8_t* weights,
    half* output, int K, int N, cudaStream_t stream
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
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 10.0f;
}

static std::unordered_map<uint64_t, int> s_hq31k_tune_cache;

static void launch_hq31k_gemv(
    const half* input, const uint8_t* weights, half* output,
    int K, int N, cudaStream_t stream
) {
    uint64_t key = (uint64_t(uint32_t(K)) << 32) | uint32_t(N);
    auto it = s_hq31k_tune_cache.find(key);
    if (it == s_hq31k_tune_cache.end()) {
        float a = benchmark_hq31k(launch_hq31k_A, input, weights, output, K, N, stream);
        float b = benchmark_hq31k(launch_hq31k_B, input, weights, output, K, N, stream);
        float c = benchmark_hq31k(launch_hq31k_C, input, weights, output, K, N, stream);
        int best = 0;
        float best_ms = a;
        if (b < best_ms) { best = 1; best_ms = b; }
        if (c < best_ms) { best = 2; }
        it = s_hq31k_tune_cache.emplace(key, best).first;
    }
    switch (it->second) {
        case 0: launch_hq31k_A(input, weights, output, K, N, stream); break;
        case 1: launch_hq31k_B(input, weights, output, K, N, stream); break;
        default: launch_hq31k_C(input, weights, output, K, N, stream); break;
    }
}

void launch_matmul_hq31k(
    const half* input, const uint8_t* weights, half* output,
    int M, int K, int N, cudaStream_t stream
) {
    // Primera integración conservadora: GEMV correcto para decode y un bucle
    // de GEMV para prefill. El camino GEMM se añadirá después de certificar el
    // formato contra un HNF real.
    if (K > HQ31K_MAX_K_SHARED) {
        throw std::runtime_error("HQ3.1K GEMV K exceeds shared-memory limit");
    }
    for (int m = 0; m < M; m++) {
        launch_hq31k_gemv(input + (size_t)m * K, weights,
                          output + (size_t)m * N, K, N, stream);
    }
}

} // namespace kernels
} // namespace helios
