// kernels/kernels.hpp
// ============================================================================
// HELIOS KERNELS - Public API
// ============================================================================
// Registra todos los kernels CUDA en el Engine.
// Diseño polimórfico: kernels se registran por OpTypeID + DTypeID
//

#pragma once

#include "engine.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace helios {
namespace kernels {

// ============================================================================
// KERNEL REGISTRATION
// ============================================================================

// Register all available kernels
void register_all_kernels(Engine& engine);

// Register by category (for selective loading)
void register_elementwise_kernels(Engine& engine);
void register_activation_kernels(Engine& engine);
void register_norm_kernels(Engine& engine);
void register_linear_kernels(Engine& engine);    // matmul
void register_attention_kernels(Engine& engine);
void register_memory_kernels(Engine& engine);    // copy, embedding

// ============================================================================
// MATMUL HQS LAUNCH FUNCTIONS
// ============================================================================



void launch_matmul_hq31k(
    const half* input,
    const uint8_t* weights,
    half* output,
    int M, int K, int N,
    cudaStream_t stream = nullptr
);

void launch_matmul_hq41k(
    const half* input,
    const uint8_t* weights,
    half* output,
    int M, int K, int N,
    cudaStream_t stream = nullptr
);

void launch_matmul_hq51k(
    const half* input,
    const uint8_t* weights,
    half* output,
    int M, int K, int N,
    cudaStream_t stream = nullptr
);

// FP16 x FP16 matmul (for activations, non-quantized weights)
void launch_matmul_fp16(
    const half* A,          // [M, K]
    const half* B,          // [K, N]
    half* C,                // [M, N]
    int M, int K, int N,
    cudaStream_t stream = nullptr
);

// Gemma 4 visual patch frontier. Pixel values are transformed from [0,1] to
// [-1,1] before the FP16 cast; the second kernel adds learned X/Y positions
// and leaves padding coordinates (-1,-1) unchanged.
void launch_gemma4_patch_input_fp16(
    const float* input,
    half* output,
    size_t elements,
    cudaStream_t stream = nullptr
);

void launch_gemma4_add_xy_position_fp16(
    half* hidden,
    const half* table,
    const int32_t* positions,
    uint32_t patches,
    uint32_t hidden_size,
    uint32_t position_embedding_size,
    cudaStream_t stream = nullptr
);

// Learned bounds used by every clippable visual linear. Bounds are FP16
// scalar tensors from the HNF and may be applied in-place.
void launch_clamp_tensor_bounds_fp16(
    const half* input,
    const half* minimum,
    const half* maximum,
    half* output,
    size_t elements,
    cudaStream_t stream = nullptr
);

// Per-head RMSNorm followed by optional Gemma 4 2-D RoPE and transpose from
// token-major [patches, heads, head_dim] to head-major
// [heads, patches, head_dim]. `weight == nullptr` selects the weightless V
// norm; `apply_rope == false` skips RoPE.
void launch_gemma4_vision_norm_rope_transpose_fp16(
    const half* input,
    const half* weight,
    const int32_t* positions,
    half* output_head_major,
    uint32_t patches,
    uint32_t heads,
    uint32_t head_dim,
    float eps,
    float theta,
    bool apply_rope,
    cudaStream_t stream = nullptr
);

// Full bidirectional visual attention. Q/K/V and output are head-major.
// Scores and probabilities are FP16, while softmax reductions are FP32.
bool launch_gemma4_vision_attention_fp16(
    const half* q_head_major,
    const half* k_head_major,
    const half* v_head_major,
    const int32_t* positions,
    half* scores,
    half* output_head_major,
    uint32_t patches,
    uint32_t heads,
    uint32_t head_dim,
    float scale,
    cudaStream_t stream = nullptr
);

void launch_gemma4_vision_head_to_token_fp16(
    const half* input_head_major,
    half* output_token_major,
    uint32_t patches,
    uint32_t heads,
    uint32_t head_dim,
    cudaStream_t stream = nullptr
);

// Spatial 3x3 average pooling. The FP32 result intentionally preserves the
// post-FP16-pool sqrt(hidden) scale used by upstream before standardization.
void launch_gemma4_vision_pool3x3_fp32(
    const half* hidden,
    float* pooled,
    uint32_t patch_columns,
    uint32_t patch_rows,
    uint32_t hidden_size,
    cudaStream_t stream = nullptr
);

void launch_fp32_to_fp16(
    const float* input,
    half* output,
    size_t elements,
    cudaStream_t stream = nullptr
);

// ============================================================================
// ELEMENTWISE KERNELS
// ============================================================================

void launch_add_fp16(
    const half* a, const half* b, half* c,
    size_t numel,
    cudaStream_t stream = nullptr
);

void launch_mul_fp16(
    const half* a, const half* b, half* c,
    size_t numel,
    cudaStream_t stream = nullptr
);

void launch_mul_scalar_tensor_fp16(
    const half* input, const half* scalar, half* output,
    size_t numel,
    cudaStream_t stream = nullptr
);

void launch_scale_fp16(
    const half* input, half* output,
    float scalar,
    size_t numel,
    cudaStream_t stream = nullptr
);

void launch_add_bias_fp16(
    const half* input,
    const half* bias,
    half* output,
    size_t total_elements,
    size_t dim,
    cudaStream_t stream = nullptr
);

// ============================================================================
// SPLIT KERNELS (single-launch, replaces cudaMemcpy loops)
// ============================================================================

// Split [batch_seq, A+B+C] → three tensors (for fused QKV)
void launch_split_3way(
    const half* src,
    half* dst_a, half* dst_b, half* dst_c,
    int batch_seq, int a_size, int b_size, int c_size,
    cudaStream_t stream = nullptr
);

// Split [batch_seq, A+B] → two tensors (for fused gate_up)
void launch_split_2way(
    const half* src,
    half* dst_a, half* dst_b,
    int batch_seq, int a_size, int b_size,
    cudaStream_t stream = nullptr
);

// Gather one layer from token-major packed PLE [rows, layers, dim].
void launch_ple_slice_fp16(
    const half* packed, half* output,
    int rows, int layers, int dim, int layer,
    cudaStream_t stream = nullptr
);

// ============================================================================
// ACTIVATION KERNELS
// ============================================================================

void launch_silu_fp16(
    const half* input, half* output,
    size_t numel,
    cudaStream_t stream = nullptr
);

void launch_gelu_fp16(
    const half* input, half* output,
    size_t numel,
    cudaStream_t stream = nullptr
);

void launch_softmax_fp16(
    const half* input, half* output,
    int batch_size, int seq_len,
    cudaStream_t stream = nullptr
);

// Fused activation × mul (replaces act(gate) + mul(gate, up))
void launch_silu_mul_fp16(
    const half* gate, const half* up, half* output,
    size_t numel,
    cudaStream_t stream = nullptr
);

void launch_gelu_mul_fp16(
    const half* gate, const half* up, half* output,
    size_t numel,
    cudaStream_t stream = nullptr
);

void launch_softcap_fp16(
    const half* input, half* output,
    float cap, size_t numel,
    cudaStream_t stream = nullptr
);

// ============================================================================
// NORMALIZATION KERNELS
// ============================================================================

void launch_rmsnorm_fp16(
    const half* input,      // [batch, dim]
    const half* weight,     // [dim]
    half* output,           // [batch, dim]
    int batch_size,
    int dim,
    float eps,
    cudaStream_t stream = nullptr
);

void launch_rmsnorm_no_weight_fp16(
    const half* input,
    half* output,
    int batch_size,
    int dim,
    float eps,
    cudaStream_t stream = nullptr
);

void launch_fused_add_rmsnorm_fp16(
    const half* a, const half* b, const half* weight,
    half* residual_out, half* normed_out,
    int batch_size, int dim, float eps,
    cudaStream_t stream = nullptr
);

void launch_layernorm_fp16(
    const half* input,
    const half* weight,
    const half* bias,       // Can be nullptr
    half* output,
    int batch_size,
    int dim,
    float eps,
    cudaStream_t stream = nullptr
);

// ============================================================================
// ROPE KERNEL
// ============================================================================


// Single tensor RoPE
void launch_rope_inplace_fp16(
    half* qk,               // Single Q or K tensor
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int rotary_dim,         // Dims to rotate (<= head_dim). 0 = full head_dim.
    int position_offset,
    float theta,
    float scaling_factor = 1.0f,  // Linear position scaling. 1.0 = none.
    cudaStream_t stream = nullptr
);

// QK-norm + RoPE fusionado (Qwen3): rmsnorm por-head de Q y K + RoPE, 1 kernel
void launch_qk_norm_rope_fp16(
    half* q, half* k, const half* q_norm_w, const half* k_norm_w,
    int batch_size, int seq_len, int num_heads, int num_kv_heads,
    int head_dim, int rotary_dim, int position_offset, float eps,
    float theta, float scaling_factor, cudaStream_t stream);

void launch_qk_norm_rope_kv_fp16(
    half* q, half* k, const half* v, const half* q_norm_w, const half* k_norm_w,
    half* k_cache, half* v_cache,
    int batch_size, int seq_len, int num_heads, int num_kv_heads,
    int head_dim, int rotary_dim, int position_offset,
    const int32_t* d_position_offset, int max_seq_len, float eps,
    float theta, float scaling_factor, cudaStream_t stream);

void launch_qk_norm_rope_fp16_dp(
    half* q, half* k, const half* q_norm_w, const half* k_norm_w,
    int batch_size, int seq_len, int num_heads, int num_kv_heads,
    int head_dim, int rotary_dim, const int32_t* d_position_offset, float eps,
    float theta, float scaling_factor, cudaStream_t stream);

// Device-pointer variant: position_offset leído de device (CUDA Graph replay)
void launch_rope_inplace_fp16_dp(
    half* qk,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int rotary_dim,
    const int32_t* d_position_offset,
    float theta,
    float scaling_factor,
    cudaStream_t stream
);

// Gemma 4 proportional RoPE rotates cross-half pairs selected by a proportion
// of head_dim. It is not equivalent to ordinary contiguous partial RoPE.
void launch_rope_proportional_inplace_fp16(
    half* qk,
    int batch_size, int seq_len,
    int num_heads, int head_dim,
    float rotary_proportion,
    int position_offset,
    float theta,
    float scaling_factor = 1.0f,
    cudaStream_t stream = nullptr
);

void launch_rope_proportional_inplace_fp16_dp(
    half* qk,
    int batch_size, int seq_len,
    int num_heads, int head_dim,
    float rotary_proportion,
    const int32_t* d_position_offset,
    float theta,
    float scaling_factor,
    cudaStream_t stream
);

// ============================================================================
// ATTENTION KERNEL
// ============================================================================

void launch_attention_fp16(
    const half* q,          // [batch, seq_q, heads, head_dim]
    const half* k,          // [batch, seq_kv, kv_heads, head_dim]
    const half* v,          // [batch, seq_kv, kv_heads, head_dim]
    half* output,           // [batch, seq_q, heads, head_dim]
    int batch_size,
    int seq_q,
    int seq_kv,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    bool causal,
    int window_size,
    cudaStream_t stream = nullptr
);

// ============================================================================
// KV CACHE OPERATIONS
// ============================================================================

// Update KV cache with new K/V at position
void launch_kv_cache_update(
    const half* new_k,      // [batch, seq_len, kv_heads, head_dim]
    const half* new_v,      // [batch, seq_len, kv_heads, head_dim]
    half* k_cache,          // [batch, max_seq, kv_heads, head_dim]
    half* v_cache,          // [batch, max_seq, kv_heads, head_dim]
    int batch_size,
    int seq_len,            // Number of new tokens to write
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int position,           // Starting position to write
    cudaStream_t stream = nullptr
);

// Device-pointer variant: position leído de device (CUDA Graph replay)
void launch_kv_cache_update_dp(
    const half* new_k,
    const half* new_v,
    half* k_cache,
    half* v_cache,
    int batch_size,
    int seq_len,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    const int32_t* d_position,
    cudaStream_t stream
);

// Attention with KV cache (autoregressive)
void launch_attention_cached_fp16(
    const half* q,          // [batch, 1, heads, head_dim] - single query
    const half* k_cache,    // [batch, max_seq, kv_heads, head_dim]
    const half* v_cache,    // [batch, max_seq, kv_heads, head_dim]
    half* output,           // [batch, 1, heads, head_dim]
    int batch_size,
    int seq_len,            // Valid length in cache
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    float scale,
    int window_size,
    cudaStream_t stream = nullptr
);

// Prefill sobre cache: S_new queries atienden causalmente a [0..past+q_idx]
void launch_attention_prefill_cached_fp16(
    const half* q, const half* k_cache, const half* v_cache, half* output,
    int seq_new, int past_len, int num_heads, int num_kv_heads,
    int head_dim, int max_seq_len, float scale, int window_size,
    cudaStream_t stream);

// Device-pointer variant: seq_len (total_seq) leído de device (CUDA Graph replay)
void launch_attention_cached_fp16_dp(
    const half* q,
    const half* k_cache,
    const half* v_cache,
    half* output,
    int batch_size,
    const int32_t* d_seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    float scale,
    int window_size,
    cudaStream_t stream
);

// ============================================================================
// MEMORY KERNELS
// ============================================================================

void launch_copy_fp16(
    const half* src, half* dst,
    size_t numel,
    cudaStream_t stream = nullptr
);

void launch_scatter_rows_fp16(
    const half* rows,
    const int32_t* indices,
    half* output,
    int row_count,
    int row_width,
    int output_rows,
    cudaStream_t stream = nullptr
);

void launch_embedding_fp16(
    const int32_t* indices, // [batch, seq]
    const half* table,      // [vocab, dim]
    half* output,           // [batch, seq, dim]
    int batch_size,
    int seq_len,
    int vocab_size,
    int dim,
    cudaStream_t stream = nullptr
);


void launch_embedding_hq51k(
    const int32_t* indices,
    const uint8_t* table,
    half* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    int dim,
    cudaStream_t stream = nullptr
);

// ============================================================================
// MATMUL cuBLAS ACCELERATED (dequant + tensor cores)
// ============================================================================



void launch_matmul_fp16_cublas(
    const half* A, const half* B, half* C,
    int M, int K, int N, cudaStream_t stream = nullptr
);

// Call on shutdown to free cuBLAS handle and dequant buffer
void cleanup_cublas();
half* get_dequant_buffer();  // DEBUG

// ============================================================================
// SAMPLING KERNELS
// ============================================================================

void launch_argmax_fp16(
    const half* input,      // [vocab_size]
    int32_t* output,        // [1] - token ID
    int vocab_size,
    cudaStream_t stream = nullptr
);

void launch_temperature_scale(
    half* logits,           // [vocab_size] - modified in place
    int vocab_size,
    float temperature,
    cudaStream_t stream = nullptr
);

void launch_softmax_sampling_fp16(
    const half* input,
    half* output,
    int vocab_size,
    cudaStream_t stream = nullptr
);

void launch_top_k(
    const half* input,              // [vocab_size]
    float* top_values,              // [k]
    int32_t* top_indices,           // [k]
    int vocab_size,
    int k,
    cudaStream_t stream = nullptr
);

void launch_top_p_cutoff(
    const float* sorted_probs,      // [k] sorted descending
    int32_t* cutoff_idx,            // [1]
    int k,
    float p,
    cudaStream_t stream = nullptr
);

void launch_categorical_sample(
    const float* probs,             // [n]
    const int32_t* indices,         // [n]
    int32_t* output,                // [1]
    int n,
    float random_val,
    cudaStream_t stream = nullptr
);

void launch_repetition_penalty(
    half* logits,                   // [vocab_size] — modified in place
    const int32_t* context,         // [context_len] — token IDs
    const int32_t* counts,          // [vocab_size] — occurrence counts
    int vocab_size,
    float rep_penalty,
    float freq_penalty,
    float pres_penalty,
    cudaStream_t stream = nullptr
);

} // namespace kernels
} // namespace helios
