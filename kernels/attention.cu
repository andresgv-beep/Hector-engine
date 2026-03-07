// kernels/attention.cu
// ============================================================================
// ATTENTION KERNELS — v2: Multi-warp cached attention + naive prefill
// ============================================================================
//
// CACHED ATTENTION v2 (decode, M=1):
//   4 warps per head (was 1) — each warp processes seq_len/4 positions
//   - Q in registers (same as v1)
//   - Each warp does online softmax on its chunk
//   - Shared memory merge of 4 partial results
//   - Supports GQA, head_dim up to 256
//
//   Why 4 warps: seq_len grows during generation. With 128 tokens:
//     v1: 1 warp loops 128 times — serial bottleneck
//     v2: 4 warps loop 32 times each — 4x parallelism
//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace helios {
namespace kernels {

// ============================================================================
// PREFILL ATTENTION — Naive (unchanged, runs once)
// ============================================================================

__global__ void attention_naive_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ output,
    int batch_size,
    int seq_q,
    int seq_kv,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    bool causal
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_q * num_heads * head_dim;
    
    if (idx >= total) return;
    
    int d = idx % head_dim;
    int h = (idx / head_dim) % num_heads;
    int s = (idx / (head_dim * num_heads)) % seq_q;
    int b = idx / (head_dim * num_heads * seq_q);
    
    int kv_h = h / (num_heads / num_kv_heads);
    
    float sum_exp = 0.0f;
    float max_score = -1e10f;
    
    for (int k = 0; k < seq_kv; k++) {
        if (causal && k > s) continue;
        float score = 0.0f;
        for (int dd = 0; dd < head_dim; dd++) {
            float q_val = __half2float(Q[b * seq_q * num_heads * head_dim + 
                                         s * num_heads * head_dim + h * head_dim + dd]);
            float k_val = __half2float(K[b * seq_kv * num_kv_heads * head_dim + 
                                         k * num_kv_heads * head_dim + kv_h * head_dim + dd]);
            score += q_val * k_val;
        }
        score *= scale;
        if (score > max_score) max_score = score;
    }
    
    for (int k = 0; k < seq_kv; k++) {
        if (causal && k > s) continue;
        float score = 0.0f;
        for (int dd = 0; dd < head_dim; dd++) {
            float q_val = __half2float(Q[b * seq_q * num_heads * head_dim + 
                                         s * num_heads * head_dim + h * head_dim + dd]);
            float k_val = __half2float(K[b * seq_kv * num_kv_heads * head_dim + 
                                         k * num_kv_heads * head_dim + kv_h * head_dim + dd]);
            score += q_val * k_val;
        }
        score *= scale;
        sum_exp += expf(score - max_score);
    }
    
    float out_val = 0.0f;
    for (int k = 0; k < seq_kv; k++) {
        if (causal && k > s) continue;
        float score = 0.0f;
        for (int dd = 0; dd < head_dim; dd++) {
            float q_val = __half2float(Q[b * seq_q * num_heads * head_dim + 
                                         s * num_heads * head_dim + h * head_dim + dd]);
            float k_val = __half2float(K[b * seq_kv * num_kv_heads * head_dim + 
                                         k * num_kv_heads * head_dim + kv_h * head_dim + dd]);
            score += q_val * k_val;
        }
        score *= scale;
        float attn_weight = expf(score - max_score) / sum_exp;
        float v_val = __half2float(V[b * seq_kv * num_kv_heads * head_dim + 
                                     k * num_kv_heads * head_dim + kv_h * head_dim + d]);
        out_val += attn_weight * v_val;
    }
    
    output[idx] = __float2half(out_val);
}

void launch_attention_fp16(
    const half* q, const half* k, const half* v, half* output,
    int batch_size, int seq_q, int seq_kv,
    int num_heads, int num_kv_heads, int head_dim,
    float scale, bool causal, cudaStream_t stream
) {
    int total = batch_size * seq_q * num_heads * head_dim;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    attention_naive_kernel<<<num_blocks, block_size, 0, stream>>>(
        q, k, v, output, batch_size, seq_q, seq_kv,
        num_heads, num_kv_heads, head_dim, scale, causal
    );
}

// ============================================================================
// CACHED ATTENTION v2 — Multi-warp with online softmax merge
// ============================================================================
//
// 4 warps per head, 1 block = 4 warps = 128 threads per head
//
// Each warp:
//   - Loads Q to registers
//   - Processes positions [warp_id * chunk, (warp_id+1) * chunk)
//   - Computes partial online softmax (running_max, running_sum, acc[])
//
// Merge phase (shared memory):
//   - Each warp writes its (max, sum, acc[]) to shared mem
//   - Warp 0 reads all 4 partial results and merges online softmax
//   - Writes final output
//

constexpr int WARP_SIZE = 32;
constexpr int ATTN_WARPS = 4;   // Warps per head
constexpr int ATTN_BLOCK = ATTN_WARPS * WARP_SIZE;  // 128 threads per block
constexpr int MAX_HD_PER_THREAD = 8;  // head_dim up to 256

// Shared memory layout for merge:
//   float partial_max[ATTN_WARPS]
//   float partial_sum[ATTN_WARPS]
//   float partial_acc[ATTN_WARPS][head_dim] — stored striped by lane

__global__ void attention_cached_v2_kernel(
    const half* __restrict__ Q,          // [batch, 1, heads, head_dim]
    const half* __restrict__ K_cache,    // [batch, max_seq, kv_heads, head_dim]
    const half* __restrict__ V_cache,    // [batch, max_seq, kv_heads, head_dim]
    half* __restrict__ output,           // [batch, 1, heads, head_dim]
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    float scale
) {
    // Block = 4 warps for 1 head
    const int head_id = blockIdx.x;  // Which (batch, head) pair
    const int warp_id = threadIdx.x / WARP_SIZE;  // 0..3
    const int lane = threadIdx.x % WARP_SIZE;     // 0..31
    
    const int b = head_id / num_heads;
    const int h = head_id % num_heads;
    
    if (b >= batch_size) return;
    
    // GQA mapping
    const int kv_h = (num_kv_heads == num_heads) ? h : (h / (num_heads / num_kv_heads));
    
    // Load Q to registers
    const int q_base = b * num_heads * head_dim + h * head_dim;
    float q_reg[MAX_HD_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < MAX_HD_PER_THREAD; i++) {
        int d = lane + i * WARP_SIZE;
        q_reg[i] = (d < head_dim) ? __half2float(Q[q_base + d]) : 0.0f;
    }
    
    // KV cache addressing
    const int kv_batch_base = b * max_seq_len * num_kv_heads * head_dim;
    const int kv_head_offset = kv_h * head_dim;
    
    // Each warp processes a chunk of seq_len
    const int chunk = (seq_len + ATTN_WARPS - 1) / ATTN_WARPS;
    const int pos_start = warp_id * chunk;
    const int pos_end = min(pos_start + chunk, seq_len);
    
    // Online softmax accumulators (per warp)
    float running_max = -1e10f;
    float running_sum = 0.0f;
    float acc[MAX_HD_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < MAX_HD_PER_THREAD; i++) acc[i] = 0.0f;
    
    // Process assigned chunk
    for (int pos = pos_start; pos < pos_end; pos++) {
        const int kv_base = kv_batch_base + pos * num_kv_heads * head_dim + kv_head_offset;
        
        // Q·K dot product
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < MAX_HD_PER_THREAD; i++) {
            int d = lane + i * WARP_SIZE;
            if (d < head_dim) {
                dot += q_reg[i] * __half2float(K_cache[kv_base + d]);
            }
        }
        
        // Warp reduce
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
        }
        float score = __shfl_sync(0xFFFFFFFF, dot, 0) * scale;
        
        // Online softmax update
        float new_max = fmaxf(running_max, score);
        float exp_score = expf(score - new_max);
        float correction = expf(running_max - new_max);
        
        running_sum = running_sum * correction + exp_score;
        #pragma unroll
        for (int i = 0; i < MAX_HD_PER_THREAD; i++) {
            acc[i] = acc[i] * correction;
        }
        running_max = new_max;
        
        // Accumulate V
        #pragma unroll
        for (int i = 0; i < MAX_HD_PER_THREAD; i++) {
            int d = lane + i * WARP_SIZE;
            if (d < head_dim) {
                acc[i] += exp_score * __half2float(V_cache[kv_base + d]);
            }
        }
    }
    
    // ====================================================================
    // MERGE PHASE: Combine 4 warps' partial online softmax results
    // ====================================================================
    // Shared memory layout:
    //   [0..3]: partial_max (4 floats)
    //   [4..7]: partial_sum (4 floats)
    //   [8..8+4*MAX_HD_PER_THREAD*32-1]: partial_acc[warp][lane*MAX_HD + i]
    
    extern __shared__ float s_merge[];
    float* s_max = s_merge;                          // [4]
    float* s_sum = s_merge + ATTN_WARPS;             // [4]
    float* s_acc = s_merge + 2 * ATTN_WARPS;         // [4][MAX_HD_PER_THREAD * 32]
    
    // Each warp writes its partial results
    if (lane == 0) {
        s_max[warp_id] = running_max;
        s_sum[warp_id] = running_sum;
    }
    
    // Write acc values — each thread writes its elements
    const int acc_stride = MAX_HD_PER_THREAD * WARP_SIZE;
    #pragma unroll
    for (int i = 0; i < MAX_HD_PER_THREAD; i++) {
        s_acc[warp_id * acc_stride + lane * MAX_HD_PER_THREAD + i] = acc[i];
    }
    __syncthreads();
    
    // Warp 0 does the merge and writes output
    if (warp_id == 0) {
        // Step 1: Find global max across all warps
        float global_max = s_max[0];
        for (int w = 1; w < ATTN_WARPS; w++) {
            global_max = fmaxf(global_max, s_max[w]);
        }
        
        // Step 2: Rescale and merge all partial results
        float merged_acc[MAX_HD_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < MAX_HD_PER_THREAD; i++) merged_acc[i] = 0.0f;
        float merged_sum = 0.0f;
        
        for (int w = 0; w < ATTN_WARPS; w++) {
            float correction = expf(s_max[w] - global_max);
            float w_sum = s_sum[w] * correction;
            merged_sum += w_sum;
            
            #pragma unroll
            for (int i = 0; i < MAX_HD_PER_THREAD; i++) {
                merged_acc[i] += s_acc[w * acc_stride + lane * MAX_HD_PER_THREAD + i] * correction;
            }
        }
        
        // Step 3: Normalize and write
        float inv_sum = (merged_sum > 0.0f) ? (1.0f / merged_sum) : 0.0f;
        const int out_base = b * num_heads * head_dim + h * head_dim;
        
        #pragma unroll
        for (int i = 0; i < MAX_HD_PER_THREAD; i++) {
            int d = lane + i * WARP_SIZE;
            if (d < head_dim) {
                output[out_base + d] = __float2half(merged_acc[i] * inv_sum);
            }
        }
    }
}

void launch_attention_cached_fp16(
    const half* q, const half* k_cache, const half* v_cache, half* output,
    int batch_size, int seq_len, int num_heads, int num_kv_heads,
    int head_dim, int max_seq_len, float scale, cudaStream_t stream
) {
    int num_blocks = batch_size * num_heads;
    
    // Shared memory for merge: max[4] + sum[4] + acc[4][MAX_HD*32]
    size_t smem = (2 * ATTN_WARPS + ATTN_WARPS * MAX_HD_PER_THREAD * WARP_SIZE) * sizeof(float);
    
    attention_cached_v2_kernel<<<num_blocks, ATTN_BLOCK, smem, stream>>>(
        q, k_cache, v_cache, output,
        batch_size, seq_len, num_heads, num_kv_heads, head_dim,
        max_seq_len, scale
    );
}

// Device-pointer version: reads seq_len (total_seq) from device memory
__global__ void attention_cached_v2_kernel_dp(
    const half* __restrict__ Q,
    const half* __restrict__ K_cache,
    const half* __restrict__ V_cache,
    half* __restrict__ output,
    int batch_size,
    const int32_t* __restrict__ d_seq_len,  // total_seq from device
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    float scale
) {
    int seq_len = *d_seq_len;
    
    const int head_id = blockIdx.x;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    
    const int b = head_id / num_heads;
    const int h = head_id % num_heads;
    
    if (b >= batch_size) return;
    
    const int kv_h = (num_kv_heads == num_heads) ? h : (h / (num_heads / num_kv_heads));
    
    const int q_base = b * num_heads * head_dim + h * head_dim;
    float q_reg[MAX_HD_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < MAX_HD_PER_THREAD; i++) {
        int d = lane + i * WARP_SIZE;
        q_reg[i] = (d < head_dim) ? __half2float(Q[q_base + d]) : 0.0f;
    }
    
    const int kv_batch_base = b * max_seq_len * num_kv_heads * head_dim;
    const int kv_head_offset = kv_h * head_dim;
    
    const int chunk = (seq_len + ATTN_WARPS - 1) / ATTN_WARPS;
    const int pos_start = warp_id * chunk;
    const int pos_end = min(pos_start + chunk, seq_len);
    
    float running_max = -1e10f;
    float running_sum = 0.0f;
    float acc[MAX_HD_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < MAX_HD_PER_THREAD; i++) acc[i] = 0.0f;
    
    for (int pos = pos_start; pos < pos_end; pos++) {
        const int kv_base = kv_batch_base + pos * num_kv_heads * head_dim + kv_head_offset;
        
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < MAX_HD_PER_THREAD; i++) {
            int d = lane + i * WARP_SIZE;
            if (d < head_dim) {
                dot += q_reg[i] * __half2float(K_cache[kv_base + d]);
            }
        }
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
        }
        float score = __shfl_sync(0xFFFFFFFF, dot, 0) * scale;
        
        float new_max = fmaxf(running_max, score);
        float exp_score = expf(score - new_max);
        float correction = expf(running_max - new_max);
        
        running_sum = running_sum * correction + exp_score;
        #pragma unroll
        for (int i = 0; i < MAX_HD_PER_THREAD; i++) {
            acc[i] = acc[i] * correction;
        }
        
        #pragma unroll
        for (int i = 0; i < MAX_HD_PER_THREAD; i++) {
            int d = lane + i * WARP_SIZE;
            if (d < head_dim) {
                acc[i] += exp_score * __half2float(V_cache[kv_base + d]);
            }
        }
        running_max = new_max;
    }
    
    extern __shared__ float s_mem[];
    float* s_max = s_mem;
    float* s_sum = s_max + ATTN_WARPS;
    float* s_acc = s_sum + ATTN_WARPS;
    
    if (lane == 0) {
        s_max[warp_id] = running_max;
        s_sum[warp_id] = running_sum;
    }
    
    const int acc_stride = MAX_HD_PER_THREAD * WARP_SIZE;
    #pragma unroll
    for (int i = 0; i < MAX_HD_PER_THREAD; i++) {
        s_acc[warp_id * acc_stride + lane * MAX_HD_PER_THREAD + i] = acc[i];
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float global_max = s_max[0];
        for (int w = 1; w < ATTN_WARPS; w++) {
            global_max = fmaxf(global_max, s_max[w]);
        }
        
        float merged_acc[MAX_HD_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < MAX_HD_PER_THREAD; i++) merged_acc[i] = 0.0f;
        float merged_sum = 0.0f;
        
        for (int w = 0; w < ATTN_WARPS; w++) {
            float correction = expf(s_max[w] - global_max);
            float w_sum = s_sum[w] * correction;
            merged_sum += w_sum;
            
            #pragma unroll
            for (int i = 0; i < MAX_HD_PER_THREAD; i++) {
                merged_acc[i] += s_acc[w * acc_stride + lane * MAX_HD_PER_THREAD + i] * correction;
            }
        }
        
        float inv_sum = (merged_sum > 0.0f) ? (1.0f / merged_sum) : 0.0f;
        const int out_base = b * num_heads * head_dim + h * head_dim;
        
        #pragma unroll
        for (int i = 0; i < MAX_HD_PER_THREAD; i++) {
            int d = lane + i * WARP_SIZE;
            if (d < head_dim) {
                output[out_base + d] = __float2half(merged_acc[i] * inv_sum);
            }
        }
    }
}

void launch_attention_cached_fp16_dp(
    const half* q, const half* k_cache, const half* v_cache, half* output,
    int batch_size, const int32_t* d_seq_len, int num_heads, int num_kv_heads,
    int head_dim, int max_seq_len, float scale, cudaStream_t stream
) {
    int num_blocks = batch_size * num_heads;
    size_t smem = (2 * ATTN_WARPS + ATTN_WARPS * MAX_HD_PER_THREAD * WARP_SIZE) * sizeof(float);
    
    attention_cached_v2_kernel_dp<<<num_blocks, ATTN_BLOCK, smem, stream>>>(
        q, k_cache, v_cache, output,
        batch_size, d_seq_len, num_heads, num_kv_heads, head_dim,
        max_seq_len, scale
    );
}

// ============================================================================
// ROPE — Rotary Position Embedding (unchanged)
// ============================================================================

__global__ void rope_kernel(
    half* __restrict__ qk,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int rotary_dim,
    int position_offset,
    float theta_base,
    float scaling_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_rotary = rotary_dim / 2;
    int total = batch_size * seq_len * num_heads * half_rotary;
    
    if (idx >= total) return;
    
    int pair = idx % half_rotary;
    int h = (idx / half_rotary) % num_heads;
    int s = (idx / (half_rotary * num_heads)) % seq_len;
    int b = idx / (half_rotary * num_heads * seq_len);
    
    float pos = (float)(s + position_offset) / scaling_factor;
    float freq = 1.0f / powf(theta_base, float(2 * pair) / float(rotary_dim));
    float angle = pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);
    
    int base_idx = b * seq_len * num_heads * head_dim + 
                   s * num_heads * head_dim + 
                   h * head_dim;
    int i0 = base_idx + pair;
    int i1 = base_idx + pair + half_rotary;
    
    float x0 = __half2float(qk[i0]);
    float x1 = __half2float(qk[i1]);
    
    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x0 * sin_val + x1 * cos_val;
    
    qk[i0] = __float2half(y0);
    qk[i1] = __float2half(y1);
}

// Device-pointer version: reads position_offset from device memory
// Used by CUDA Graphs (capture-once, replay-many)
__global__ void rope_kernel_dp(
    half* __restrict__ qk,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int rotary_dim,
    const int32_t* __restrict__ d_position_offset,
    float theta_base,
    float scaling_factor
) {
    int position_offset = *d_position_offset;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_rotary = rotary_dim / 2;
    int total = batch_size * seq_len * num_heads * half_rotary;
    
    if (idx >= total) return;
    
    int pair = idx % half_rotary;
    int h = (idx / half_rotary) % num_heads;
    int s = (idx / (half_rotary * num_heads)) % seq_len;
    int b = idx / (half_rotary * num_heads * seq_len);
    
    float pos = (float)(s + position_offset) / scaling_factor;
    float freq = 1.0f / powf(theta_base, float(2 * pair) / float(rotary_dim));
    float angle = pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);
    
    int base_idx = b * seq_len * num_heads * head_dim + 
                   s * num_heads * head_dim + 
                   h * head_dim;
    int i0 = base_idx + pair;
    int i1 = base_idx + pair + half_rotary;
    
    float x0 = __half2float(qk[i0]);
    float x1 = __half2float(qk[i1]);
    
    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x0 * sin_val + x1 * cos_val;
    
    qk[i0] = __float2half(y0);
    qk[i1] = __float2half(y1);
}

void launch_rope_fp16(
    half* q, half* k,
    int batch_size, int seq_len, int num_heads, int num_kv_heads,
    int head_dim, int rotary_dim, int position_offset,
    float theta, float scaling_factor, cudaStream_t stream
) {
    if (rotary_dim <= 0) rotary_dim = head_dim;
    if (scaling_factor <= 0.0f) scaling_factor = 1.0f;
    int half_rotary = rotary_dim / 2;
    int block_size = 256;
    
    int total_q = batch_size * seq_len * num_heads * half_rotary;
    int num_blocks_q = (total_q + block_size - 1) / block_size;
    rope_kernel<<<num_blocks_q, block_size, 0, stream>>>(
        q, batch_size, seq_len, num_heads, head_dim, rotary_dim,
        position_offset, theta, scaling_factor
    );
    
    int total_k = batch_size * seq_len * num_kv_heads * half_rotary;
    int num_blocks_k = (total_k + block_size - 1) / block_size;
    rope_kernel<<<num_blocks_k, block_size, 0, stream>>>(
        k, batch_size, seq_len, num_kv_heads, head_dim, rotary_dim,
        position_offset, theta, scaling_factor
    );
}

void launch_rope_inplace_fp16(
    half* qk,
    int batch_size, int seq_len, int num_heads,
    int head_dim, int rotary_dim, int position_offset,
    float theta, float scaling_factor, cudaStream_t stream
) {
    if (rotary_dim <= 0) rotary_dim = head_dim;
    if (scaling_factor <= 0.0f) scaling_factor = 1.0f;
    int half_rotary = rotary_dim / 2;
    int block_size = 256;
    
    int total = batch_size * seq_len * num_heads * half_rotary;
    int num_blocks = (total + block_size - 1) / block_size;
    
    rope_kernel<<<num_blocks, block_size, 0, stream>>>(
        qk, batch_size, seq_len, num_heads, head_dim, rotary_dim,
        position_offset, theta, scaling_factor
    );
}

// Device-pointer version for CUDA Graphs
void launch_rope_inplace_fp16_dp(
    half* qk,
    int batch_size, int seq_len, int num_heads,
    int head_dim, int rotary_dim, const int32_t* d_position_offset,
    float theta, float scaling_factor, cudaStream_t stream
) {
    if (rotary_dim <= 0) rotary_dim = head_dim;
    if (scaling_factor <= 0.0f) scaling_factor = 1.0f;
    int half_rotary = rotary_dim / 2;
    int block_size = 256;
    
    int total = batch_size * seq_len * num_heads * half_rotary;
    int num_blocks = (total + block_size - 1) / block_size;
    
    rope_kernel_dp<<<num_blocks, block_size, 0, stream>>>(
        qk, batch_size, seq_len, num_heads, head_dim, rotary_dim,
        d_position_offset, theta, scaling_factor
    );
}

} // namespace kernels
} // namespace helios

// ============================================================================
// KV CACHE OPERATIONS (unchanged)
// ============================================================================

namespace helios {
namespace kernels {

__global__ void kv_cache_update_kernel(
    const half* __restrict__ new_kv,
    half* __restrict__ cache,
    int batch_size,
    int seq_len,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int position
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * kv_heads * head_dim;
    
    if (idx >= total) return;
    
    half val = new_kv[idx];
    
    int d = idx % head_dim;
    int h = (idx / head_dim) % kv_heads;
    int s = (idx / (head_dim * kv_heads)) % seq_len;
    int b = idx / (head_dim * kv_heads * seq_len);
    
    int cache_pos = position + s;
    if (cache_pos >= max_seq_len) return;
    
    int dst_idx = b * max_seq_len * kv_heads * head_dim +
                  cache_pos * kv_heads * head_dim +
                  h * head_dim + d;
    
    cache[dst_idx] = val;
}

void launch_kv_cache_update(
    const half* new_k, const half* new_v,
    half* k_cache, half* v_cache,
    int batch_size, int seq_len, int kv_heads, int head_dim,
    int max_seq_len, int position, cudaStream_t stream
) {
    int total = batch_size * seq_len * kv_heads * head_dim;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    kv_cache_update_kernel<<<num_blocks, block_size, 0, stream>>>(
        new_k, k_cache, batch_size, seq_len, kv_heads, head_dim, max_seq_len, position
    );
    
    kv_cache_update_kernel<<<num_blocks, block_size, 0, stream>>>(
        new_v, v_cache, batch_size, seq_len, kv_heads, head_dim, max_seq_len, position
    );
}

// Device-pointer version: reads position from device memory
__global__ void kv_cache_update_kernel_dp(
    const half* __restrict__ new_kv,
    half* __restrict__ cache,
    int batch_size,
    int seq_len,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    const int32_t* __restrict__ d_position
) {
    int position = *d_position;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * kv_heads * head_dim;
    
    if (idx >= total) return;
    
    half val = new_kv[idx];
    
    int d = idx % head_dim;
    int h = (idx / head_dim) % kv_heads;
    int s = (idx / (head_dim * kv_heads)) % seq_len;
    int b = idx / (head_dim * kv_heads * seq_len);
    
    int cache_pos = position + s;
    if (cache_pos >= max_seq_len) return;
    
    int dst_idx = b * max_seq_len * kv_heads * head_dim +
                  cache_pos * kv_heads * head_dim +
                  h * head_dim + d;
    
    cache[dst_idx] = val;
}

void launch_kv_cache_update_dp(
    const half* new_k, const half* new_v,
    half* k_cache, half* v_cache,
    int batch_size, int seq_len, int kv_heads, int head_dim,
    int max_seq_len, const int32_t* d_position, cudaStream_t stream
) {
    int total = batch_size * seq_len * kv_heads * head_dim;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    kv_cache_update_kernel_dp<<<num_blocks, block_size, 0, stream>>>(
        new_k, k_cache, batch_size, seq_len, kv_heads, head_dim, max_seq_len, d_position
    );
    
    kv_cache_update_kernel_dp<<<num_blocks, block_size, 0, stream>>>(
        new_v, v_cache, batch_size, seq_len, kv_heads, head_dim, max_seq_len, d_position
    );
}

} // namespace kernels
} // namespace helios
