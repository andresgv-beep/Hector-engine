// HQS v4 (hq44k_g16): affine 4-bit blocks with 5-bit group scales and mins.
// Superblock of 256 weights, 152 bytes:
//   [0..1] d (FP16)  [2..3] dmin (FP16)
//   [4..23] little-endian bitstream: scale g at bit 5g, min g at bit 80+5g
//   [24..151] 4-bit codes, weight k in byte k/2 (low nibble for even k)
// weight = d*scale[g]*code - dmin*min[g], group g = k/16.
#pragma once
#include "hqs_common.cuh"
namespace helios { namespace kernels { namespace hqs_v4 {

constexpr int BLOCK_BYTES = 152;
constexpr int HEADER_BYTES = 24;
constexpr int INPUT_CHUNK = 16384;

__device__ __forceinline__ unsigned field5(const uint8_t* block, int bit) {
    const unsigned word = unsigned(block[4 + bit / 8]) | (unsigned(block[5 + bit / 8]) << 8);
    return (word >> (bit & 7)) & 31u;
}

__device__ __forceinline__ float half_at(const uint8_t* p) {
    return __half2float(__ushort_as_half(uint16_t(p[0]) | (uint16_t(p[1]) << 8)));
}

// Each lane covers 8 consecutive weights (half a group); a warp covers one superblock.
template<int WARPS_PER_ROW, int ROWS_PER_BLOCK>
__global__ void gemv_hq44_kernel(const half* __restrict__ input, const uint8_t* __restrict__ weights,
                                 half* __restrict__ output, int K, int N) {
    extern __shared__ half s_input[];
    const int staged_k = K < INPUT_CHUNK ? K : INPUT_CHUNK;
    float* s_partial = reinterpret_cast<float*>(s_input + staged_k);
    const int threads_per_row = WARPS_PER_ROW * 32;
    const int row_group = threadIdx.x / threads_per_row;
    const int local_tid = threadIdx.x % threads_per_row;
    const int warp_in_group = local_tid / 32;
    const int lane = local_tid % 32;
    const int row = blockIdx.x * ROWS_PER_BLOCK + row_group;
    const int total_sb = K / 256;
    const uint8_t* row_weights = weights + size_t(row < N ? row : 0) * total_sb * BLOCK_BYTES;
    const int group = lane >> 1;
    float acc = 0.0f;

    for (int chunk_base = 0; chunk_base < K; chunk_base += INPUT_CHUNK) {
        const int chunk_len = min(INPUT_CHUNK, K - chunk_base);
        constexpr int BS = WARPS_PER_ROW * ROWS_PER_BLOCK * 32;
        const float4* src = reinterpret_cast<const float4*>(input + chunk_base);
        float4* dst = reinterpret_cast<float4*>(s_input);
        for (int i = threadIdx.x; i < chunk_len / 8; i += BS) dst[i] = src[i];
        __syncthreads();

        const int sb_begin = chunk_base / 256, sb_end = (chunk_base + chunk_len) / 256;
        for (int sb = sb_begin + warp_in_group; row < N && sb < sb_end; sb += WARPS_PER_ROW) {
            const uint8_t* block = row_weights + size_t(sb) * BLOCK_BYTES;
            const float a = half_at(block) * float(field5(block, group * 5));
            const float m = half_at(block + 2) * float(field5(block, 80 + group * 5));
            const uint32_t packed = *reinterpret_cast<const uint32_t*>(block + HEADER_BYTES + lane * 4);
            const half2* x = reinterpret_cast<const half2*>(s_input + (sb * 256 - chunk_base) + lane * 8);
            float sq = 0.0f, sx = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 v = __half22float2(x[i]);
                const unsigned byte = (packed >> (i * 8)) & 0xffu;
                sq = fmaf(float(byte & 15u), v.x, sq);
                sq = fmaf(float(byte >> 4), v.y, sq);
                sx += v.x + v.y;
            }
            acc = fmaf(a, sq, acc);
            acc = fmaf(-m, sx, acc);
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

// v2: U superblocks per warp iteration, loads issued before any math. The 24-byte
// headers are read cooperatively (lane 8u+w holds word w of superblock u) and
// fields are fetched with shuffles instead of eight byte loads per lane.
template<int U>
__device__ __forceinline__ float hq44_superblocks(const uint8_t* row_weights, const half* s_block0,
                                                  int sb, int stride, int lane) {
    uint32_t header = 0;
    {
        const int u = lane >> 3, w = lane & 7;
        if (u < U && w < 6)
            header = *reinterpret_cast<const uint32_t*>(row_weights + size_t(sb + u * stride) * BLOCK_BYTES + w * 4);
    }
    uint32_t codes[U];
    #pragma unroll
    for (int u = 0; u < U; ++u)
        codes[u] = *reinterpret_cast<const uint32_t*>(
            row_weights + size_t(sb + u * stride) * BLOCK_BYTES + HEADER_BYTES + lane * 4);
    const int group = lane >> 1;
    const int sbit = 32 + group * 5, mbit = 112 + group * 5;   // bit offsets from the block start
    float acc = 0.0f;
    #pragma unroll
    for (int u = 0; u < U; ++u) {
        const int base = u * 8;
        const uint32_t w0 = __shfl_sync(0xffffffffu, header, base);
        const uint32_t s_lo = __shfl_sync(0xffffffffu, header, base + (sbit >> 5));
        const uint32_t s_hi = __shfl_sync(0xffffffffu, header, base + (sbit >> 5) + 1);
        const uint32_t m_lo = __shfl_sync(0xffffffffu, header, base + (mbit >> 5));
        const uint32_t m_hi = __shfl_sync(0xffffffffu, header, base + (mbit >> 5) + 1);
        const float d = __half2float(__ushort_as_half(uint16_t(w0 & 0xffffu)));
        const float dmin = __half2float(__ushort_as_half(uint16_t(w0 >> 16)));
        const float a = d * float(__funnelshift_r(s_lo, s_hi, sbit & 31) & 31u);
        const float m = dmin * float(__funnelshift_r(m_lo, m_hi, mbit & 31) & 31u);
        const float4 raw = *reinterpret_cast<const float4*>(s_block0 + size_t(u * stride) * 256 + lane * 8);
        const half2* x = reinterpret_cast<const half2*>(&raw);
        float sq = 0.0f, sx = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 v = __half22float2(x[i]);
            const unsigned byte = (codes[u] >> (i * 8)) & 0xffu;
            sq = fmaf(float(byte & 15u), v.x, sq);
            sq = fmaf(float(byte >> 4), v.y, sq);
            sx += v.x + v.y;
        }
        acc = fmaf(a, sq, acc);
        acc = fmaf(-m, sx, acc);
    }
    return acc;
}

template<int WARPS_PER_ROW, int ROWS_PER_BLOCK, int U>
__global__ void gemv_hq44_v2_kernel(const half* __restrict__ input, const uint8_t* __restrict__ weights,
                                    half* __restrict__ output, int K, int N) {
    static_assert(U == 1 || U == 2 || U == 4, "header lanes hold at most four superblocks");
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
                acc += hq44_superblocks<U>(row_weights, s_input + (sb * 256 - chunk_base), sb, WARPS_PER_ROW, lane);
            for (; sb < sb_end; sb += WARPS_PER_ROW)
                acc += hq44_superblocks<1>(row_weights, s_input + (sb * 256 - chunk_base), sb, WARPS_PER_ROW, lane);
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
static void launch_gemv2(const half* in, const uint8_t* weights, half* out, int K, int N, cudaStream_t stream) {
    const int threads = WPR * RPB * 32;
    const int blocks = (N + RPB - 1) / RPB;
    const size_t shared = std::min(K, INPUT_CHUNK) * sizeof(half) + RPB * WPR * sizeof(float);
    gemv_hq44_v2_kernel<WPR, RPB, U><<<blocks, threads, shared, stream>>>(in, weights, out, K, N);
}

template<int WPR, int RPB>
static void launch_gemv(const half* in, const uint8_t* weights, half* out, int K, int N, cudaStream_t stream) {
    const int threads = WPR * RPB * 32;
    const int blocks = (N + RPB - 1) / RPB;
    const size_t shared = std::min(K, INPUT_CHUNK) * sizeof(half) + RPB * WPR * sizeof(float);
    gemv_hq44_kernel<WPR, RPB><<<blocks, threads, shared, stream>>>(in, weights, out, K, N);
}

} } } // helios::kernels::hqs_v4
