// HQ8K: symmetric 8-bit HQS blocks (helios_convert src/hqs/hq8.rs).
// Superblock of 256 weights, 272 bytes, 16-byte aligned:
//   [0..1]    d (FP16), largest step of the superblock
//   [2..9]    q[8] (u8), step of group g = d * q[g] / 255, 32 weights per group
//   [10..15]  reserved, zero
//   [16..271] 256 int8 codes; weight = code * step
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>
namespace helios { namespace kernels { namespace hq8 {

constexpr int BLOCK_BYTES = 272;
constexpr int HEADER_BYTES = 16;
constexpr int INPUT_CHUNK = 16384;

__device__ __forceinline__ float step_of(const uint8_t* block, int group) {
    const float d = __half2float(__ushort_as_half(uint16_t(block[0]) | (uint16_t(block[1]) << 8)));
    return d * float(block[2 + group]) * (1.0f / 255.0f);
}

// Dot of 8 int8 codes (two packed words) with 8 FP16 inputs.
__device__ __forceinline__ float dot8(uint2 codes, float4 raw) {
    const half2* x = reinterpret_cast<const half2*>(&raw);
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t word = i < 2 ? codes.x : codes.y;
        const int shift = (i & 1) * 16;
        const float2 v = __half22float2(x[i]);
        sum = fmaf(float(int8_t(word >> shift)), v.x, sum);
        sum = fmaf(float(int8_t(word >> (shift + 8))), v.y, sum);
    }
    return sum;
}

// U superblocks per warp iteration, loads issued before any math. Each lane covers 8
// weights; group g = lane / 4. The 16-byte headers are read cooperatively (lane 4u+w
// holds word w of superblock u) and the step comes back through shuffles.
template<int U>
__device__ __forceinline__ float hq8_superblocks(const uint8_t* row_weights, const half* s_block0,
                                                 int sb, int stride, int lane) {
    uint32_t header = 0;
    {
        const int u = lane >> 2, w = lane & 3;
        if (u < U && w < 3)
            header = *reinterpret_cast<const uint32_t*>(row_weights + size_t(sb + u * stride) * BLOCK_BYTES + w * 4);
    }
    uint2 codes[U];
    #pragma unroll
    for (int u = 0; u < U; ++u)
        codes[u] = *reinterpret_cast<const uint2*>(
            row_weights + size_t(sb + u * stride) * BLOCK_BYTES + HEADER_BYTES + lane * 8);
    const int group = lane >> 2, qbyte = 2 + group;
    float acc = 0.0f;
    #pragma unroll
    for (int u = 0; u < U; ++u) {
        const uint32_t w0 = __shfl_sync(0xffffffffu, header, u * 4);
        const uint32_t wq = __shfl_sync(0xffffffffu, header, u * 4 + (qbyte >> 2));
        const float d = __half2float(__ushort_as_half(uint16_t(w0 & 0xffffu)));
        const float step = d * float((wq >> ((qbyte & 3) * 8)) & 0xffu) * (1.0f / 255.0f);
        const float4 raw = *reinterpret_cast<const float4*>(s_block0 + size_t(u * stride) * 256 + lane * 8);
        acc = fmaf(step, dot8(codes[u], raw), acc);
    }
    return acc;
}

template<int WARPS_PER_ROW, int ROWS_PER_BLOCK, int U>
__global__ void gemv_hq8_kernel(const half* __restrict__ input, const uint8_t* __restrict__ weights,
                                half* __restrict__ output, int K, int N) {
    static_assert(U == 1 || U == 2 || U == 4 || U == 8, "header lanes hold at most eight superblocks");
    extern __shared__ half s_input[];
    const int staged_k = K < INPUT_CHUNK ? K : INPUT_CHUNK;
    float* s_partial = reinterpret_cast<float*>(s_input + staged_k);
    const int threads_per_row = WARPS_PER_ROW * 32;
    const int row_group = threadIdx.x / threads_per_row;
    const int local_tid = threadIdx.x % threads_per_row;
    const int warp_in_group = local_tid / 32;
    const int lane = local_tid % 32;
    const int row = blockIdx.x * ROWS_PER_BLOCK + row_group;
    const uint8_t* row_weights = weights + size_t(row < N ? row : 0) * (K / 256) * BLOCK_BYTES;
    float acc = 0.0f;

    for (int chunk_base = 0; chunk_base < K; chunk_base += INPUT_CHUNK) {
        const int chunk_len = min(INPUT_CHUNK, K - chunk_base);
        constexpr int BS = WARPS_PER_ROW * ROWS_PER_BLOCK * 32;
        const float4* src = reinterpret_cast<const float4*>(input + chunk_base);
        float4* dst = reinterpret_cast<float4*>(s_input);
        for (int i = threadIdx.x; i < chunk_len / 8; i += BS) dst[i] = src[i];
        __syncthreads();

        const int sb_begin = chunk_base / 256, sb_end = (chunk_base + chunk_len) / 256;
        int sb = sb_begin + warp_in_group;
        if (row < N) {
            for (; sb + (U - 1) * WARPS_PER_ROW < sb_end; sb += U * WARPS_PER_ROW)
                acc += hq8_superblocks<U>(row_weights, s_input + (sb * 256 - chunk_base), sb, WARPS_PER_ROW, lane);
            for (; sb < sb_end; sb += WARPS_PER_ROW)
                acc += hq8_superblocks<1>(row_weights, s_input + (sb * 256 - chunk_base), sb, WARPS_PER_ROW, lane);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down_sync(0xffffffff, acc, offset);
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

template<int WPR, int RPB, int U>
static void launch_gemv(const half* in, const uint8_t* weights, half* out, int K, int N, cudaStream_t stream) {
    const int threads = WPR * RPB * 32;
    const int blocks = (N + RPB - 1) / RPB;
    const size_t shared = std::min(K, INPUT_CHUNK) * sizeof(half) + RPB * WPR * sizeof(float);
    gemv_hq8_kernel<WPR, RPB, U><<<blocks, threads, shared, stream>>>(in, weights, out, K, N);
}

} } } // helios::kernels::hq8
