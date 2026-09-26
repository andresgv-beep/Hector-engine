// Cached prefill attention on tensor cores through cuBLAS.
//
// The reference prefill gives every (query, head) its own CTA and walks the
// whole KV range with warp reductions, so a 512-token chunk over 15k keys
// re-reads the cache hundreds of times without tensor cores. Here, per block
// of keys: S = Q K^T (FP32, batched over the heads sharing a KV head), a mask
// + partial softmax kernel, O_b = P V, and an online merge across key blocks.
// Keys are taken in slot order; each slot's absolute position is recovered
// from the ring, so sliding windows and wrapped caches need no reordering.
#include "kernels.hpp"
#include "cublas_context.hpp"
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>

namespace helios {
namespace kernels {

namespace {

constexpr int PG_THREADS = 256;
constexpr size_t PG_SCORE_BUDGET = size_t(128) << 20;  // bytes of FP32 scores per key block

struct Workspace {
    float* scores = nullptr; half* probs = nullptr; float* block_out = nullptr;
    float* acc = nullptr; float* stats = nullptr;
    size_t score_elems = 0, out_elems = 0, stat_elems = 0;
};
Workspace g_ws;

bool ensure(float*& p, size_t& have, size_t need) {
    if (need <= have) return true;
    if (p) cudaFree(p);
    p = nullptr; have = 0;
    if (cudaMalloc(&p, need * sizeof(float)) != cudaSuccess) { p = nullptr; return false; }
    have = need;
    return true;
}

bool ensure_workspace(size_t score_elems, size_t out_elems, size_t stat_elems) {
    if (score_elems > g_ws.score_elems) {
        if (g_ws.probs) cudaFree(g_ws.probs);
        g_ws.probs = nullptr;
        size_t have = g_ws.score_elems;
        if (!ensure(g_ws.scores, have, score_elems)) return false;
        if (cudaMalloc(&g_ws.probs, score_elems * sizeof(half)) != cudaSuccess) return false;
        g_ws.score_elems = score_elems;
    }
    size_t have_out = g_ws.out_elems;
    if (out_elems > have_out) {
        if (g_ws.acc) cudaFree(g_ws.acc);
        g_ws.acc = nullptr;
        if (!ensure(g_ws.block_out, have_out, out_elems)) return false;
        if (cudaMalloc(&g_ws.acc, out_elems * sizeof(float)) != cudaSuccess) return false;
        g_ws.out_elems = out_elems;
    }
    // stats: row max / sum for the block and for the running accumulator.
    size_t have_stats = g_ws.stat_elems;
    if (!ensure(g_ws.stats, have_stats, 4 * stat_elems)) return false;
    g_ws.stat_elems = have_stats;
    return true;
}

__device__ __forceinline__ float block_reduce(float v, float* scratch, bool is_max) {
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
    for (int w = 1; w < PG_THREADS / 32; ++w) r = is_max ? fmaxf(r, scratch[w]) : r + scratch[w];
    return r;
}

// One CTA per (query, head): mask the scores of this key block, write
// unnormalized probabilities and the block's row max / sum.
__global__ __launch_bounds__(PG_THREADS) void prefill_block_softmax_kernel(
    const float* __restrict__ scores, half* __restrict__ probs, float* __restrict__ block_max,
    float* __restrict__ block_sum, int seq_new, int past_len, int key_base, int nkeys, int ld,
    int kv_end, int cache_slots, int window, int bidir_begin, int bidir_end) {
    const int s = blockIdx.x, h = blockIdx.y;
    const size_t row = (size_t(h) * seq_new + s);
    const float* in = scores + row * ld;
    half* out = probs + row * ld;
    const int qpos = past_len + s;
    __shared__ float scratch[PG_THREADS / 32];

    auto key_pos = [&](int c) {
        const int slot = key_base + c;
        return slot + ((kv_end - 1 - slot) / cache_slots) * cache_slots;
    };
    // Gemma 4 12B sliding layers: the tokens of one image see each other in
    // both directions (Gemma 3's blockwise overlay). Global layers pass an
    // empty block and stay causal.
    const bool q_in_block = qpos >= bidir_begin && qpos < bidir_end;
    auto valid = [&](int p) {
        const bool visible = p <= qpos || (q_in_block && p >= bidir_begin && p < bidir_end);
        return visible && (window <= 0 || p > qpos - window);
    };

    float local = -INFINITY;
    for (int c = threadIdx.x; c < nkeys; c += PG_THREADS)
        if (valid(key_pos(c))) local = fmaxf(local, in[c]);
    const float m = block_reduce(local, scratch, true);
    float sum = 0.0f;
    for (int c = threadIdx.x; c < nkeys; c += PG_THREADS) {
        float e = 0.0f;
        if (m != -INFINITY && valid(key_pos(c))) e = expf(in[c] - m);
        out[c] = __float2half(e);
        sum += e;
    }
    const float l = block_reduce(sum, scratch, false);
    if (threadIdx.x == 0) { block_max[row] = m; block_sum[row] = l; }
}

// Online merge of a key block into the running accumulator (first block copies).
__global__ __launch_bounds__(PG_THREADS) void prefill_merge_kernel(
    const float* __restrict__ block_out, const float* __restrict__ block_max,
    const float* __restrict__ block_sum, float* __restrict__ acc, float* __restrict__ run_max,
    float* __restrict__ run_sum, int seq_new, int head_dim, bool first) {
    const int s = blockIdx.x, h = blockIdx.y;
    const size_t row = size_t(h) * seq_new + s;
    const float mb = block_max[row], lb = block_sum[row];
    float a = 0.0f, b = 1.0f, m = mb;
    if (!first) {
        const float ma = run_max[row];
        m = fmaxf(ma, mb);
        a = ma == -INFINITY ? 0.0f : expf(ma - m);
        b = mb == -INFINITY ? 0.0f : expf(mb - m);
    }
    const float* src = block_out + row * head_dim;
    float* dst = acc + row * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += PG_THREADS)
        dst[d] = first ? src[d] : fmaf(dst[d], a, src[d] * b);
    __syncthreads();
    if (threadIdx.x == 0) {
        run_max[row] = m;
        run_sum[row] = first ? lb : run_sum[row] * a + lb * b;
    }
}

__global__ __launch_bounds__(PG_THREADS) void prefill_finalize_kernel(
    const float* __restrict__ acc, const float* __restrict__ run_sum, half* __restrict__ output,
    int seq_new, int num_heads, int head_dim) {
    const int s = blockIdx.x, h = blockIdx.y;
    const size_t row = size_t(h) * seq_new + s;
    const float l = run_sum[row];
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    half* out = output + (size_t(s) * num_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += PG_THREADS)
        out[d] = __float2half(acc[row * head_dim + d] * inv);
}

}  // namespace

bool launch_attention_prefill_gemm_fp16(
    const half* q, const half* k_cache, const half* v_cache, half* output,
    int seq_new, int past_len, int num_heads, int num_kv_heads, int head_dim,
    int max_seq_len, float scale, int window_size, cudaStream_t stream, int cache_slots,
    int bidir_begin, int bidir_end) {
    if (cache_slots <= 0) cache_slots = max_seq_len;
    // The block may only reach keys already written by this chunk.
    if (bidir_end > bidir_begin &&
        (bidir_begin < past_len || bidir_end > past_len + seq_new)) return false;
    const int kv_end = past_len + seq_new;
    const int nkeys_total = std::min(cache_slots, kv_end);
    const int group = num_heads / num_kv_heads;
    // Key block: as many keys as fit the score budget, in multiples of 256.
    const size_t per_key = size_t(num_heads) * seq_new * sizeof(float);
    int block = int(std::max<size_t>(256, PG_SCORE_BUDGET / per_key / 256 * 256));
    block = std::min(block, (nkeys_total + 7) / 8 * 8);
    const size_t rows = size_t(num_heads) * seq_new;
    if (!ensure_workspace(rows * block, rows * head_dim, rows)) return false;
    cublasHandle_t handle = cublas_handle_for_stream(stream);
    if (!handle) return false;

    float* block_max = g_ws.stats;
    float* block_sum = g_ws.stats + rows;
    float* run_max = g_ws.stats + 2 * rows;
    float* run_sum = g_ws.stats + 3 * rows;
    const int kv_ld = num_kv_heads * head_dim, q_ld = num_heads * head_dim;
    const float one = 1.0f, zero = 0.0f;
    const dim3 grid(seq_new, num_heads);

    for (int key_base = 0; key_base < nkeys_total; key_base += block) {
        const int nk = std::min(block, nkeys_total - key_base);
        for (int j = 0; j < group; ++j) {
            // S[h] (keys x queries, col-major) = K_g^T Q_h for h = g*group + j.
            if (cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, nk, seq_new, head_dim, &scale,
                    k_cache + size_t(key_base) * kv_ld, CUDA_R_16F, kv_ld, head_dim,
                    q + size_t(j) * head_dim, CUDA_R_16F, q_ld, int64_t(group) * head_dim, &zero,
                    g_ws.scores + size_t(j) * seq_new * block, CUDA_R_32F, block, int64_t(group) * seq_new * block,
                    num_kv_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                return false;
        }
        prefill_block_softmax_kernel<<<grid, PG_THREADS, 0, stream>>>(
            g_ws.scores, g_ws.probs, block_max, block_sum, seq_new, past_len, key_base, nk, block,
            kv_end, cache_slots, window_size, bidir_begin, bidir_end);
        for (int j = 0; j < group; ++j) {
            // O_b[h] (head_dim x queries, col-major) = V_g P_h.
            if (cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, seq_new, nk, &one,
                    v_cache + size_t(key_base) * kv_ld, CUDA_R_16F, kv_ld, head_dim,
                    g_ws.probs + size_t(j) * seq_new * block, CUDA_R_16F, block, int64_t(group) * seq_new * block,
                    &zero, g_ws.block_out + size_t(j) * seq_new * head_dim, CUDA_R_32F, head_dim,
                    int64_t(group) * seq_new * head_dim, num_kv_heads, CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                return false;
        }
        prefill_merge_kernel<<<grid, PG_THREADS, 0, stream>>>(
            g_ws.block_out, block_max, block_sum, g_ws.acc, run_max, run_sum, seq_new, head_dim,
            key_base == 0);
    }
    prefill_finalize_kernel<<<grid, PG_THREADS, 0, stream>>>(g_ws.acc, run_sum, output, seq_new,
                                                             num_heads, head_dim);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace kernels
}  // namespace helios
