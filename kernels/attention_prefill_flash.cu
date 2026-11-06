// Fused causal prefill attention (FlashAttention-2 style).
//
// The GEMM prefill writes every score block to memory and reads it back for
// the softmax and for P V; with long contexts that traffic dominates. Here
// scores and probabilities never leave the SM (mma.sync m16n8k16, FP16 in,
// FP32 accumulate, online softmax).
//
// head_dim 128: a CTA of four warps owns 64 queries of one head and keeps the
// output accumulator in registers. Only full-attention layers whose keys sit
// at their positions (no sliding ring) take this path.
//
// head_dim 256 / 512 (Gemma): 16 rows x 512 dims of FP32 do not fit one warp,
// so a CTA of eight warps splits head_dim. Rows go in groups of 16; each warp
// of a group keeps its slice of Q in registers, computes the partial scores
// of that slice over the whole key tile, and the group sums them through
// shared memory in the same order everywhere, so every warp holds the same S.
// The softmax then runs redundantly per warp and P feeds P V from registers,
// each warp accumulating its slice of O. Rows pack the query heads that share
// a KV head, so one K/V tile serves all of them. Keys are walked by absolute
// position (slot = pos % cache_slots), which covers sliding windows, wrapped
// rings and the image block of the 12B.
#include "kernels.hpp"
#include <cuda_fp16.h>
#include <algorithm>
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
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
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


// ---- head_dim 256 / 512 ----------------------------------------------------

constexpr int FW_WARPS = 8;
constexpr int FW_THREADS = FW_WARPS * 32;

// Q/K/V rows are head_dim halves; chunk c of row r lives at c ^ (r % 8) so
// the eight rows of an ldmatrix hit eight different banks.
template <int HD>
__device__ __forceinline__ int fw_swz(int row, int chunk) { return row * HD + ((chunk ^ (row & 7)) << 3); }

__device__ __forceinline__ void cp_async16(half* smem, const half* gmem, bool valid) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(valid ? 16 : 0));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

template <int HD, int BM, int BN>
struct FwShape {
    static constexpr int RG = BM / 16;         // row groups of 16
    static constexpr int WPR = FW_WARPS / RG;  // warps per row group
    static constexpr int ON = HD / WPR;        // head_dim slice per warp (S and O)
    static constexpr int CH = HD / 8;
    static constexpr int NT = BN / 8;          // key n-tiles
    static constexpr int KS = ON / 16;         // k-steps of the warp's slice
    static constexpr size_t SMEM = 2 * size_t(BN) * HD * 2 + size_t(RG) * WPR * NT * 32 * 16;
    static_assert(BM % 16 == 0 && FW_WARPS % RG == 0 && BN % 16 == 0, "tiles");
    static_assert(BM <= 2 * BN, "Q is staged in the K and V buffers");
    static_assert(KS % 2 == 0 && ON % 16 == 0 && CH >= 8, "head_dim slice in steps of 32");
};

template <int HD, int BM, int BN, int MINB>
__global__ __launch_bounds__(FW_THREADS, MINB) void attention_prefill_flash_wide_kernel(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
    half* __restrict__ output, int seq_new, int past_len, int num_heads, int num_kv_heads,
    float scale_log2, int window, int cache_slots, int bidir_begin, int bidir_end) {
    using Sh = FwShape<HD, BM, BN>;
    constexpr int NT = Sh::NT, KS = Sh::KS, ON = Sh::ON, WPR = Sh::WPR, CH = Sh::CH;
    extern __shared__ __align__(16) unsigned char fw_smem[];
    half* sK = reinterpret_cast<half*>(fw_smem);
    half* sV = sK + BN * HD;
    float4* sS = reinterpret_cast<float4*>(sV + BN * HD);  // [row group][warp][n-tile][lane]

    const int tile = gridDim.x - 1 - blockIdx.x;
    const int g = blockIdx.y;
    const int group = num_heads / num_kv_heads;
    const int row0 = tile * BM;
    const int total_rows = seq_new * group;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wr = warp / WPR, wn = warp % WPR;
    const size_t q_ld = size_t(num_heads) * HD, kv_ld = size_t(num_kv_heads) * HD;
    const int kv_end = past_len + seq_new;

    const int q_first = past_len + row0 / group;
    const int q_last = past_len + min(total_rows - 1, row0 + BM - 1) / group;
    const bool has_block = bidir_end > bidir_begin;
    int lo = max(0, kv_end - cache_slots);
    if (window > 0) lo = max(lo, q_first - window + 1);
    int hi = q_last;
    if (has_block && q_last >= bidir_begin && q_first < bidir_end) hi = max(hi, bidir_end - 1);
    hi = min(hi, kv_end - 1);

    for (int i = threadIdx.x; i < BM * CH; i += FW_THREADS) {
        const int r = i / CH, c = i % CH, gr = row0 + r;
        const bool ok = gr < total_rows;
        const int qi = ok ? gr / group : 0, h = g * group + (ok ? gr % group : 0);
        cp_async16(sK + fw_swz<HD>(r, c), Q + size_t(qi) * q_ld + size_t(h) * HD + c * 8, ok);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    uint32_t qf[KS][4];
    #pragma unroll
    for (int ks = 0; ks < KS; ++ks)
        ldmatrix_x4(qf[ks], sK + fw_swz<HD>(wr * 16 + (lane % 16), (wn * ON) / 8 + 2 * ks + lane / 16));
    __syncthreads();

    auto load_kv = [&](half* dst, const half* src, int t0) {
        for (int i = threadIdx.x; i < BN * CH; i += FW_THREADS) {
            const int r = i / CH, c = i % CH, p = t0 + r;
            const bool ok = p <= hi;
            const int slot = ok ? p % cache_slots : 0;
            cp_async16(dst + fw_swz<HD>(r, c), src + size_t(slot) * kv_ld + size_t(g) * HD + c * 8, ok);
        }
    };
    load_kv(sK, K, lo);
    cp_async_commit();

    int qpos[2];
    bool row_ok[2], in_block[2];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int gr = row0 + wr * 16 + lane / 4 + 8 * r;
        row_ok[r] = gr < total_rows;
        qpos[r] = past_len + gr / group;
        in_block[r] = has_block && qpos[r] >= bidir_begin && qpos[r] < bidir_end;
    }
    auto visible = [&](int p, int r) {
        const bool seen = p <= qpos[r] || (in_block[r] && p >= bidir_begin && p < bidir_end);
        return row_ok[r] && p <= hi && seen && (window <= 0 || p > qpos[r] - window);
    };

    float o[ON / 8][4];
    #pragma unroll
    for (int n = 0; n < ON / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m_row[2] = {-INFINITY, -INFINITY}, l_row[2] = {0.0f, 0.0f};
    float4* mine = sS + (warp * NT) * 32 + lane;
    const float4* group_s = sS + (wr * WPR * NT) * 32 + lane;

    for (int t0 = lo; t0 <= hi; t0 += BN) {
        cp_async_wait<0>();
        __syncthreads();  // K(t0) landed; V and the partial scores of the last tile are free
        load_kv(sV, V, t0);
        cp_async_commit();

        float sa[NT][4];
        #pragma unroll
        for (int n = 0; n < NT; ++n) sa[n][0] = sa[n][1] = sa[n][2] = sa[n][3] = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < KS / 2; ++kk) {
            #pragma unroll
            for (int n = 0; n < NT; ++n) {
                uint32_t b[4];
                ldmatrix_x4(b, sK + fw_swz<HD>(n * 8 + (lane % 8), (wn * ON) / 8 + 4 * kk + lane / 8));
                mma16816(sa[n], qf[2 * kk], b[0], b[1]);
                mma16816(sa[n], qf[2 * kk + 1], b[2], b[3]);
            }
        }
        #pragma unroll
        for (int n = 0; n < NT; ++n) mine[n * 32] = make_float4(sa[n][0], sa[n][1], sa[n][2], sa[n][3]);
        __syncthreads();  // partial scores visible, K buffer free
        const bool more = t0 + BN <= hi;
        if (more) {
            load_kv(sK, K, t0 + BN);
            cp_async_commit();
        }

        float s[NT][4];
        float tmax[2] = {-INFINITY, -INFINITY};
        #pragma unroll
        for (int n = 0; n < NT; ++n) {
            float4 v = group_s[n * 32];
            #pragma unroll
            for (int w = 1; w < WPR; ++w) {
                const float4 u = group_s[(w * NT + n) * 32];
                v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w;
            }
            const float vv[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int key = t0 + n * 8 + (lane % 4) * 2 + (e & 1);
                s[n][e] = visible(key, e >> 1) ? vv[e] * scale_log2 : -INFINITY;
                tmax[e >> 1] = fmaxf(tmax[e >> 1], s[n][e]);
            }
        }
        float alpha[2];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffff, tmax[r], 1));
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffff, tmax[r], 2));
            const float m_new = fmaxf(m_row[r], tmax[r]);
            alpha[r] = m_new == -INFINITY ? 1.0f : exp2f(m_row[r] - m_new);
            m_row[r] = m_new;
            l_row[r] *= alpha[r];
        }
        #pragma unroll
        for (int n = 0; n < ON / 8; ++n) {
            o[n][0] *= alpha[0]; o[n][1] *= alpha[0];
            o[n][2] *= alpha[1]; o[n][3] *= alpha[1];
        }
        #pragma unroll
        for (int n = 0; n < NT; ++n) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r = e >> 1;
                const float p = m_row[r] == -INFINITY ? 0.0f : exp2f(s[n][e] - m_row[r]);
                s[n][e] = p;
                l_row[r] += p;
            }
        }
        if (more) cp_async_wait<1>(); else cp_async_wait<0>();
        __syncthreads();  // V(t0) landed

        // O += P V: P from the S accumulators (C layout == A layout pairwise).
        #pragma unroll
        for (int ks = 0; ks < BN / 16; ++ks) {
            uint32_t a[4];
            a[0] = pack_half2(s[2 * ks][0], s[2 * ks][1]);
            a[1] = pack_half2(s[2 * ks][2], s[2 * ks][3]);
            a[2] = pack_half2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
            a[3] = pack_half2(s[2 * ks + 1][2], s[2 * ks + 1][3]);
            #pragma unroll
            for (int n = 0; n < ON / 8; n += 2) {
                uint32_t b[4];
                ldmatrix_x4_trans(b, sV + fw_swz<HD>(ks * 16 + (lane % 8) + ((lane / 8) % 2) * 8,
                                                     (wn * ON + n * 8) / 8 + lane / 16));
                mma16816(o[n], a, b[0], b[1]);
                mma16816(o[n + 1], a, b[2], b[3]);
            }
        }
    }
    cp_async_wait<0>();

    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        l_row[r] += __shfl_xor_sync(0xffffffff, l_row[r], 1);
        l_row[r] += __shfl_xor_sync(0xffffffff, l_row[r], 2);
        if (!row_ok[r]) continue;
        const float inv = l_row[r] > 0.0f ? 1.0f / l_row[r] : 0.0f;
        const int gr = row0 + wr * 16 + lane / 4 + 8 * r;
        const int h = g * group + gr % group;
        half* out = output + (size_t(gr / group) * num_heads + h) * HD + wn * ON + (lane % 4) * 2;
        #pragma unroll
        for (int n = 0; n < ON / 8; ++n)
            *reinterpret_cast<half2*>(out + n * 8) = __floats2half2_rn(o[n][2 * r] * inv, o[n][2 * r + 1] * inv);
    }
}

template <int HD, int BM, int BN, int MINB>
bool launch_flash_wide(const half* q, const half* k_cache, const half* v_cache, half* output,
                         int seq_new, int past_len, int num_heads, int num_kv_heads, float scale,
                         int window_size, int cache_slots, int bidir_begin, int bidir_end,
                         cudaStream_t stream) {
    using Sh = FwShape<HD, BM, BN>;
    const auto kernel = attention_prefill_flash_wide_kernel<HD, BM, BN, MINB>;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(Sh::SMEM)) !=
        cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    const int rows = seq_new * (num_heads / num_kv_heads);
    const dim3 grid((rows + BM - 1) / BM, num_kv_heads);
    kernel<<<grid, FW_THREADS, Sh::SMEM, stream>>>(
        q, k_cache, v_cache, output, seq_new, past_len, num_heads, num_kv_heads,
        scale * 1.4426950408889634f, window_size, cache_slots, bidir_begin, bidir_end);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace

bool launch_attention_prefill_flash_fp16(
    const half* q, const half* k_cache, const half* v_cache, half* output,
    int seq_new, int past_len, int num_heads, int num_kv_heads, int head_dim,
    int max_seq_len, float scale, int window_size, cudaStream_t stream, int cache_slots,
    int bidir_begin, int bidir_end) {
    if (!device_has_m16n8k16()) return false;
    if (cache_slots <= 0) cache_slots = max_seq_len;
    if (seq_new <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0) return false;
    const bool has_block = bidir_end > bidir_begin;
    // The block may only reach keys already written by this chunk.
    if (has_block && (bidir_begin < past_len || bidir_end > past_len + seq_new)) return false;
    // Two CTAs per SM (128 registers, ~40 KB of shared memory each) hide one
    // another's loads and barriers. With many keys the 512-wide layers still
    // lose to the cuBLAS GEMMs (0.8x at 16k), so they only take the short ones.
    if (head_dim == 256)
        return launch_flash_wide<256, 64, 32, 2>(q, k_cache, v_cache, output, seq_new, past_len, num_heads,
                                                 num_kv_heads, scale, window_size, cache_slots, bidir_begin,
                                                 bidir_end, stream);
    if (head_dim == 512)
        return std::min(past_len + seq_new, cache_slots) <= 2048 &&
               launch_flash_wide<512, 32, 16, 2>(q, k_cache, v_cache, output, seq_new, past_len, num_heads,
                                                 num_kv_heads, scale, window_size, cache_slots, bidir_begin,
                                                 bidir_end, stream);
    // Keys must sit at their own positions: full attention, no wrapped ring.
    if (head_dim != FA_HD || window_size > 0 || past_len + seq_new > cache_slots || has_block)
        return false;
    const dim3 grid((seq_new + FA_BR - 1) / FA_BR, num_heads);
    attention_prefill_flash_hd128_kernel<<<grid, FA_THREADS, 0, stream>>>(
        q, k_cache, v_cache, output, seq_new, past_len, num_heads, num_kv_heads,
        scale * 1.4426950408889634f);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace kernels
}  // namespace helios
