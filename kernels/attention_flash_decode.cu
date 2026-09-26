// Decode attention split along the sequence (flash-decoding).
//
// The reference decode walks each partition position by position with a warp
// reduction per key, so a long context is latency-bound on a handful of warps.
// Here every (split, head) is its own CTA: scores for a tile are computed with
// several independent keys per warp, then V is streamed with one thread per
// dimension pair and no per-key reduction. A second kernel merges the splits.
//
// The split count is fixed at launch (CUDA Graph friendly) and the per-split
// chunk is derived on device from the current length, so short turns simply
// leave trailing CTAs idle. Summation order differs from the reference.
#include "kernels.hpp"
#include <cuda_fp16.h>

namespace helios {
namespace kernels {

namespace {

constexpr int FD_THREADS = 128;
constexpr int FD_WARPS = FD_THREADS / 32;
constexpr int FD_ILP = 4;            // keys per warp in flight during QK
constexpr int FD_TILE = 256;         // keys per shared-memory tile
constexpr int FD_MAX_HD2 = 256;      // head_dim 512 in half2 units
constexpr int FD_MAX_DPT = (FD_MAX_HD2 + FD_THREADS - 1) / FD_THREADS;

__device__ __forceinline__ void fd_split_range(int seq_len, int window, int splits, int min_chunk,
                                               int& first, int& chunk, int& active_splits) {
    first = window > 0 ? max(0, seq_len - window) : 0;
    const int active = seq_len - first;
    chunk = max(min_chunk, (active + splits - 1) / splits);
    active_splits = (active + chunk - 1) / chunk;
}

__device__ __forceinline__ float fd_block_reduce(float v, float* scratch, bool is_max) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float other = __shfl_xor_sync(0xffffffff, v, o);
        v = is_max ? fmaxf(v, other) : v + other;
    }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float r = scratch[0];
    #pragma unroll
    for (int w = 1; w < FD_WARPS; ++w) r = is_max ? fmaxf(r, scratch[w]) : r + scratch[w];
    return r;
}

__global__ __launch_bounds__(FD_THREADS) void flash_decode_partial_kernel(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
    float* __restrict__ partials, const int32_t* __restrict__ d_seq_len,
    int num_heads, int num_kv_heads, int head_dim, float scale, int window,
    int cache_slots, int splits, int min_chunk) {
    const int split = blockIdx.x;
    const int h = blockIdx.y;
    int first, chunk, active_splits;
    fd_split_range(*d_seq_len, window, splits, min_chunk, first, chunk, active_splits);
    if (split >= active_splits) return;
    const int seq_len = *d_seq_len;
    const int begin = first + split * chunk;
    const int end = min(begin + chunk, seq_len);

    const int hd2 = head_dim >> 1;
    const int kv_h = h / (num_heads / num_kv_heads);
    const size_t slot_stride2 = size_t(num_kv_heads) * hd2;
    const half2* K2 = reinterpret_cast<const half2*>(K) + size_t(kv_h) * hd2;
    const half2* V2 = reinterpret_cast<const half2*>(V) + size_t(kv_h) * hd2;
    const half2* Q2 = reinterpret_cast<const half2*>(Q) + size_t(h) * hd2;

    __shared__ float2 q_s[FD_MAX_HD2];
    __shared__ float s_p[FD_TILE];
    __shared__ float s_red[FD_WARPS];
    __shared__ float2 s_acc[FD_THREADS];

    for (int d = threadIdx.x; d < hd2; d += FD_THREADS) {
        const float2 qv = __half22float2(Q2[d]);
        q_s[d] = make_float2(qv.x * scale, qv.y * scale);
    }
    __syncthreads();

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    // V ownership: hd2 >= threads -> each thread owns dpt pairs over all keys;
    // otherwise threads split into groups that share keys and reduce at the end.
    const int groups = hd2 >= FD_THREADS ? 1 : FD_THREADS / hd2;
    const int group = hd2 >= FD_THREADS ? 0 : threadIdx.x / hd2;
    const int d_first = hd2 >= FD_THREADS ? threadIdx.x : threadIdx.x % hd2;
    const bool v_active = group < groups && d_first < hd2;

    float m = -INFINITY, l = 0.0f;
    float2 acc[FD_MAX_DPT];
    #pragma unroll
    for (int i = 0; i < FD_MAX_DPT; ++i) acc[i] = make_float2(0.0f, 0.0f);

    for (int tile = begin; tile < end; tile += FD_TILE) {
        const int n = min(FD_TILE, end - tile);
        for (int base = warp * FD_ILP; base < n; base += FD_WARPS * FD_ILP) {
            float dot[FD_ILP];
            const half2* rows[FD_ILP];
            #pragma unroll
            for (int j = 0; j < FD_ILP; ++j) {
                dot[j] = 0.0f;
                const int p = min(base + j, n - 1);
                rows[j] = K2 + size_t((tile + p) % cache_slots) * slot_stride2;
            }
            for (int d = lane; d < hd2; d += 32) {
                const float2 qv = q_s[d];
                #pragma unroll
                for (int j = 0; j < FD_ILP; ++j) {
                    const float2 kv = __half22float2(rows[j][d]);
                    dot[j] = fmaf(qv.x, kv.x, dot[j]);
                    dot[j] = fmaf(qv.y, kv.y, dot[j]);
                }
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                #pragma unroll
                for (int j = 0; j < FD_ILP; ++j) dot[j] += __shfl_xor_sync(0xffffffff, dot[j], o);
            }
            if (lane < FD_ILP && base + lane < n) {
                float v = dot[0];
                #pragma unroll
                for (int j = 1; j < FD_ILP; ++j) if (lane == j) v = dot[j];
                s_p[base + lane] = v;
            }
        }
        __syncthreads();

        float local = -INFINITY;
        for (int i = threadIdx.x; i < n; i += FD_THREADS) local = fmaxf(local, s_p[i]);
        const float new_m = fmaxf(m, fd_block_reduce(local, s_red, true));
        float part = 0.0f;
        for (int i = threadIdx.x; i < n; i += FD_THREADS) {
            const float e = expf(s_p[i] - new_m);
            s_p[i] = e;
            part += e;
        }
        const float tile_sum = fd_block_reduce(part, s_red, false);
        const float corr = expf(m - new_m);
        l = l * corr + tile_sum;
        m = new_m;

        if (v_active) {
            #pragma unroll
            for (int i = 0; i < FD_MAX_DPT; ++i) { acc[i].x *= corr; acc[i].y *= corr; }
            for (int p = group; p < n; p += groups) {
                const float w = s_p[p];
                const half2* row = V2 + size_t((tile + p) % cache_slots) * slot_stride2;
                #pragma unroll
                for (int i = 0; i < FD_MAX_DPT; ++i) {
                    const int d = d_first + i * FD_THREADS;
                    if (d < hd2) {
                        const float2 vv = __half22float2(row[d]);
                        acc[i].x = fmaf(w, vv.x, acc[i].x);
                        acc[i].y = fmaf(w, vv.y, acc[i].y);
                    }
                }
            }
        }
        __syncthreads();
    }

    float* out = partials + (size_t(h) * splits + split) * (head_dim + 2);
    if (threadIdx.x == 0) { out[0] = m; out[1] = l; }
    float2* out_acc = reinterpret_cast<float2*>(out + 2);
    if (groups == 1) {
        #pragma unroll
        for (int i = 0; i < FD_MAX_DPT; ++i) {
            const int d = d_first + i * FD_THREADS;
            if (d < hd2) out_acc[d] = acc[i];
        }
        return;
    }
    s_acc[threadIdx.x] = v_active ? acc[0] : make_float2(0.0f, 0.0f);
    __syncthreads();
    if (group == 0 && d_first < hd2) {
        float2 sum = s_acc[threadIdx.x];
        for (int g = 1; g < groups; ++g) {
            const float2 o = s_acc[g * hd2 + d_first];
            sum.x += o.x; sum.y += o.y;
        }
        out_acc[d_first] = sum;
    }
}

__global__ __launch_bounds__(FD_THREADS) void flash_decode_merge_kernel(
    const float* __restrict__ partials, half* __restrict__ output,
    const int32_t* __restrict__ d_seq_len, int head_dim, int window, int splits, int min_chunk) {
    const int h = blockIdx.x;
    int first, chunk, active_splits;
    fd_split_range(*d_seq_len, window, splits, min_chunk, first, chunk, active_splits);
    const float* base = partials + size_t(h) * splits * (head_dim + 2);
    float gm = -INFINITY;
    for (int s = 0; s < active_splits; ++s) gm = fmaxf(gm, base[s * (head_dim + 2)]);
    float total = 0.0f;
    for (int s = 0; s < active_splits; ++s) {
        const float* p = base + s * (head_dim + 2);
        total += p[1] * expf(p[0] - gm);
    }
    const float inv = total > 0.0f ? 1.0f / total : 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += FD_THREADS) {
        float sum = 0.0f;
        for (int s = 0; s < active_splits; ++s) {
            const float* p = base + s * (head_dim + 2);
            sum = fmaf(p[2 + d], expf(p[0] - gm), sum);
        }
        output[size_t(h) * head_dim + d] = __float2half(sum * inv);
    }
}

}  // namespace

size_t attention_flash_decode_workspace_bytes(int num_heads, int head_dim, int splits) {
    return size_t(num_heads) * splits * (head_dim + 2) * sizeof(float);
}

void launch_attention_flash_decode_dp(
    const half* q, const half* k_cache, const half* v_cache, half* output,
    float* partials, const int32_t* d_seq_len, int num_heads, int num_kv_heads,
    int head_dim, float scale, int window_size, int cache_slots, int splits,
    int min_chunk, cudaStream_t stream) {
    flash_decode_partial_kernel<<<dim3(splits, num_heads), FD_THREADS, 0, stream>>>(
        q, k_cache, v_cache, partials, d_seq_len, num_heads, num_kv_heads, head_dim,
        scale, window_size, cache_slots, splits, min_chunk);
    flash_decode_merge_kernel<<<num_heads, FD_THREADS, 0, stream>>>(
        partials, output, d_seq_len, head_dim, window_size, splits, min_chunk);
}

}  // namespace kernels
}  // namespace helios
