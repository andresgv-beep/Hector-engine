// Provisional HQS x.3: grouped 5-bit scales and 3/4/5-bit payload. Lab only.
// Derived from frozen v2 to preserve GEMV accumulation order.
#pragma once
#include "hqs_common.cuh"
namespace helios { namespace kernels { namespace hqs_v3 {
namespace hqs = helios::hqs;
constexpr int COMPACT_INPUT_CHUNK = 16384;
template<int WARPS_PER_ROW>
__device__ __forceinline__ void rotate_single_warp(float& a, float& b) {
    if constexpr (WARPS_PER_ROW == 1) { const float t = a; a = b; b = t; }
}

template<int BITS, int GROUP, int BLOCK_BYTES, int WARPS_PER_ROW, int ROWS_PER_BLOCK>
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
            constexpr int HEADER = BLOCK_BYTES - BITS * 32;
            const int group = lane / (GROUP / 8);
            const int bit = group * 5;
            const int byte = 2 + bit / 8;
            const uint16_t packed_step = uint16_t(block[byte]) | (uint16_t(block[byte+1]) << 8);
            const float d = __half2float(__ushort_as_half(uint16_t(block[0]) | (uint16_t(block[1]) << 8)));
            const float step = d * float((packed_step >> (bit & 7)) & 31) * (1.f / 31.f);
            const int global_k = sb_base + lane * GROUP_SIZE;
            const int local_k = global_k - chunk_base;
            if constexpr (BITS == 4) {
                const uint32_t packed = *reinterpret_cast<const uint32_t*>(
                    block + HEADER + lane * 4);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint8_t byte = (packed >> (i * 8)) & 0xff;
                    const float w0 = float(int(byte >> 4) - 8) * step;
                    const float w1 = float(int(byte & 0x0f) - 8) * step;
                    const int k0 = global_k + i * 2;
                    if (k0 < K) acc = fmaf(w0, __half2float(s_input[local_k + i * 2]), acc);
                    if (k0 + 1 < K) acc = fmaf(w1, __half2float(s_input[local_k + i * 2 + 1]), acc);
                }
            } else if constexpr (BITS == 5) {
                const int offset = HEADER + lane * 5;
                const uint32_t* words = reinterpret_cast<const uint32_t*>(block);
                const int word = offset >> 2, shift = (offset & 3) * 8;
                const uint32_t p0 = words[word], p1 = words[word + 1];
                const uint32_t lo = __funnelshift_r(p0, p1, shift), hi = p1 >> shift;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t code = i == 7 ? ((hi >> 3) & 31u)
                        : (__funnelshift_r(lo, hi, i * 5) & 31u);
                    const float w = float(int(code) - 16) * step;
                    if (global_k + i < K) acc = fmaf(w, __half2float(s_input[local_k + i]), acc);
                }
            } else {
                static_assert(BITS == 3);
                const uint8_t* payload = block + HEADER + lane * 3;
                const uint32_t packed = uint32_t(payload[0]) | (uint32_t(payload[1]) << 8) | (uint32_t(payload[2]) << 16);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float w = float(int((packed >> (i * 3)) & 7) - 4) * step;
                    if (global_k + i < K) acc = fmaf(w, __half2float(s_input[local_k + i]), acc);
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

template<int BITS, int GROUP, int WPR, int RPB>
static void launch_symmetric(const half* in, const uint8_t* weights, half* out,
                             int K, int N, cudaStream_t stream) {
    constexpr int BLOCK_BYTES = ((2 + 256/GROUP*5/8 + 3)/4)*4 + BITS*32;
    const int threads = WPR * RPB * 32;
    const int blocks = (N + RPB - 1) / RPB;
    const size_t shared = std::min(K, COMPACT_INPUT_CHUNK) * sizeof(half) +
                          RPB * WPR * sizeof(float);
    gemv_hq_symmetric_kernel<BITS, GROUP, BLOCK_BYTES, WPR, RPB>
        <<<blocks, threads, shared, stream>>>(in, weights, out, K, N);
}


}

} } // helios::kernels
