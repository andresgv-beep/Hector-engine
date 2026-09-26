// Fused causal prefill attention for head_dim 128 (FlashAttention-2 style).
//
// The GEMM prefill writes every score block to memory and reads it back for
// the softmax and for P V; with long contexts that traffic dominates. Here a
// CTA of four warps owns 64 queries of one head, streams 64-key K/V tiles
// through shared memory and keeps scores, probabilities and the output
// accumulator in registers (mma.sync m16n8k16, FP16 in, FP32 accumulate,
// online softmax). Only full-attention layers whose keys sit at their
// positions (no sliding ring) take this path.
#include "kernels.hpp"
#include <cuda_fp16.h>
#include <cstdint>

namespace helios {
namespace kernels {

namespace {

constexpr int FA_HD = 128;
constexpr int FA_BR = 64;            // queries per CTA (16 per warp)
constexpr int FA_BC = 64;            // keys per tile
constexpr int FA_WARPS = 4;
constexpr int FA_THREADS = FA_WARPS * 32;
constexpr int FA_STRIDE = FA_HD + 8; // halves per smem row: +16 B avoids ldmatrix bank conflicts

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const half* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const half* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    const half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Copy `rows` rows of 128 halves (row stride `ld` halves in global) into smem,
// zero-filling rows past `valid`.
__device__ __forceinline__ void load_tile(half* dst, const half* src, size_t ld, int rows, int valid) {
    constexpr int CHUNKS = FA_HD / 8;  // 16-byte chunks per row
    for (int i = threadIdx.x; i < rows * CHUNKS; i += FA_THREADS) {
        const int r = i / CHUNKS, c = i % CHUNKS;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r < valid) v = *reinterpret_cast<const uint4*>(src + size_t(r) * ld + c * 8);
        *reinterpret_cast<uint4*>(dst + r * FA_STRIDE + c * 8) = v;
    }
}

__global__ __launch_bounds__(FA_THREADS) void attention_prefill_flash_hd128_kernel(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
    half* __restrict__ output, int seq_new, int past_len, int num_heads, int num_kv_heads,
    float scale_log2) {
    // Q is staged in the K buffer: once in registers it is not read again.
    static_assert(FA_BR == FA_BC, "Q staging reuses the K tile buffer");
    __shared__ __align__(16) half sK[FA_BC * FA_STRIDE];
    __shared__ __align__(16) half sV[FA_BC * FA_STRIDE];

    const int q0 = blockIdx.x * FA_BR;
    const int h = blockIdx.y;
    const int kvh = h / (num_heads / num_kv_heads);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const size_t q_ld = size_t(num_heads) * FA_HD, kv_ld = size_t(num_kv_heads) * FA_HD;

    load_tile(sK, Q + size_t(q0) * q_ld + size_t(h) * FA_HD, q_ld, FA_BR, seq_new - q0);
    __syncthreads();

    // Q fragments for this warp's 16 rows, all 8 k-steps of head_dim.
    uint32_t qf[FA_HD / 16][4];
    #pragma unroll
    for (int ks = 0; ks < FA_HD / 16; ++ks)
        ldmatrix_x4(qf[ks], sK + (warp * 16 + (lane % 16)) * FA_STRIDE + ks * 16 + (lane / 16) * 8);

    float o[FA_HD / 8][4];
    #pragma unroll
    for (int n = 0; n < FA_HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m_row[2] = {-INFINITY, -INFINITY}, l_row[2] = {0.0f, 0.0f};

    // Absolute positions of the two rows this thread accumulates.
    const int row_a = q0 + warp * 16 + lane / 4;
    const int pos_row[2] = {past_len + row_a, past_len + row_a + 8};
    const int q_last = min(q0 + FA_BR, seq_new) - 1;
    const int kv_end = past_len + q_last + 1;
    const int block_first_pos = past_len + q0;

    for (int k0 = 0; k0 < kv_end; k0 += FA_BC) {
        __syncthreads();
        load_tile(sK, K + size_t(k0) * kv_ld + size_t(kvh) * FA_HD, kv_ld, FA_BC, kv_end - k0);
        load_tile(sV, V + size_t(k0) * kv_ld + size_t(kvh) * FA_HD, kv_ld, FA_BC, kv_end - k0);
        __syncthreads();

        // S = Q K^T for 16 rows x 64 keys: 8 n-tiles of 8 keys.
        float s[FA_BC / 8][4];
        #pragma unroll
        for (int n = 0; n < FA_BC / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
        #pragma unroll
        for (int ks = 0; ks < FA_HD / 16; ++ks) {
            #pragma unroll
            for (int n = 0; n < FA_BC / 8; n += 2) {
                uint32_t b[4];
                ldmatrix_x4(b, sK + (n * 8 + (lane % 8) + (lane / 16) * 8) * FA_STRIDE + ks * 16 + ((lane / 8) % 2) * 8);
                mma16816(s[n], qf[ks], b[0], b[1]);
                mma16816(s[n + 1], qf[ks], b[2], b[3]);
            }
        }

        // Causal mask only on tiles that reach past the first query of the CTA.
        const bool needs_mask = k0 + FA_BC - 1 > block_first_pos;
        float tile_max[2] = {-INFINITY, -INFINITY};
        #pragma unroll
        for (int n = 0; n < FA_BC / 8; ++n) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int key = k0 + n * 8 + (lane % 4) * 2 + (e % 2);
                const int r = e / 2;
                float v = s[n][e] * scale_log2;
                if (key >= kv_end || (needs_mask && key > pos_row[r])) v = -INFINITY;
                s[n][e] = v;
                tile_max[r] = fmaxf(tile_max[r], v);
            }
        }
        float alpha[2];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffff, tile_max[r], 1));
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffff, tile_max[r], 2));
            const float m_new = fmaxf(m_row[r], tile_max[r]);
            alpha[r] = m_new == -INFINITY ? 1.0f : exp2f(m_row[r] - m_new);
            m_row[r] = m_new;
            l_row[r] *= alpha[r];
        }
        #pragma unroll
        for (int n = 0; n < FA_HD / 8; ++n) {
            o[n][0] *= alpha[0]; o[n][1] *= alpha[0];
            o[n][2] *= alpha[1]; o[n][3] *= alpha[1];
        }
        #pragma unroll
        for (int n = 0; n < FA_BC / 8; ++n) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r = e / 2;
                const float p = m_row[r] == -INFINITY ? 0.0f : exp2f(s[n][e] - m_row[r]);
                s[n][e] = p;
                l_row[r] += p;
            }
        }

        // O += P V: P from the S accumulators (C layout == A layout pairwise).
        #pragma unroll
        for (int ks = 0; ks < FA_BC / 16; ++ks) {
            uint32_t a[4];
            a[0] = pack_half2(s[2 * ks][0], s[2 * ks][1]);
            a[1] = pack_half2(s[2 * ks][2], s[2 * ks][3]);
            a[2] = pack_half2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
            a[3] = pack_half2(s[2 * ks + 1][2], s[2 * ks + 1][3]);
            #pragma unroll
            for (int n = 0; n < FA_HD / 8; n += 2) {
                uint32_t b[4];
                ldmatrix_x4_trans(b, sV + (ks * 16 + (lane % 8) + ((lane / 8) % 2) * 8) * FA_STRIDE + n * 8 + (lane / 16) * 8);
                mma16816(o[n], a, b[0], b[1]);
                mma16816(o[n + 1], a, b[2], b[3]);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        l_row[r] += __shfl_xor_sync(0xffffffff, l_row[r], 1);
        l_row[r] += __shfl_xor_sync(0xffffffff, l_row[r], 2);
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = row_a + r * 8;
        if (row >= seq_new) continue;
        const float inv = l_row[r] > 0.0f ? 1.0f / l_row[r] : 0.0f;
        half* out = output + (size_t(row) * num_heads + h) * FA_HD + (lane % 4) * 2;
        #pragma unroll
        for (int n = 0; n < FA_HD / 8; ++n)
            *reinterpret_cast<half2*>(out + n * 8) = __floats2half2_rn(o[n][2 * r] * inv, o[n][2 * r + 1] * inv);
    }
}

}  // namespace

bool launch_attention_prefill_flash_fp16(
    const half* q, const half* k_cache, const half* v_cache, half* output,
    int seq_new, int past_len, int num_heads, int num_kv_heads, int head_dim,
    int max_seq_len, float scale, int window_size, cudaStream_t stream, int cache_slots) {
    if (cache_slots <= 0) cache_slots = max_seq_len;
    // Keys must sit at their own positions: full attention, no wrapped ring.
    if (head_dim != FA_HD || window_size > 0 || past_len + seq_new > cache_slots ||
        num_heads % num_kv_heads != 0)
        return false;
    const dim3 grid((seq_new + FA_BR - 1) / FA_BR, num_heads);
    attention_prefill_flash_hd128_kernel<<<grid, FA_THREADS, 0, stream>>>(
        q, k_cache, v_cache, output, seq_new, past_len, num_heads, num_kv_heads,
        scale * 1.4426950408889634f);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace kernels
}  // namespace helios
