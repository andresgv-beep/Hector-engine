// src/sampler.cpp
// ============================================================================
// HELIOS SAMPLER - Implementation with Repetition Penalty
// ============================================================================
// Pipeline polimórfico:
//   1. Copy logits to scratch (preserve original)
//   2. Apply penalties (rep/freq/pres) based on context window
//   3. Temperature scaling
//   4. Top-K / Top-P filtering
//   5. Sample from distribution
//
// Todas las penalties son opcionales (disabled by default value).
// El contexto se acumula externamente vía add_context().

#include "sampler.hpp"
#include "../kernels/kernels.hpp"
#include <stdexcept>
#include <cstring>
#include <unordered_map>
#include <algorithm>

namespace helios {

namespace kernels {
    void launch_argmax_fp16(const half*, int32_t*, int, cudaStream_t);
    void launch_temperature_scale(half*, int, float, cudaStream_t);
    void launch_top_k(const half*, float*, int32_t*, int, int, cudaStream_t);
    void launch_top_p_cutoff(const float*, int32_t*, int, float, cudaStream_t);
    void launch_categorical_sample(const float*, const int32_t*, int32_t*, int, float, cudaStream_t);
    void launch_repetition_penalty(half*, const int32_t*, const int32_t*, int, float, float, float, cudaStream_t);
}

// ============================================================================
// PENALTY APPLICATION
// ============================================================================
// Builds count map from context window, uploads to GPU, runs kernel.
// Cost: O(window) CPU + O(vocab) GPU — negligible vs matmul.

void Sampler::apply_penalties(
    half* logits,
    int vocab_size,
    const SamplingConfig& config,
    cudaStream_t stream
) {
    bool has_rep = config.repetition_penalty != 1.0f;
    bool has_freq = config.frequency_penalty != 0.0f;
    bool has_pres = config.presence_penalty != 0.0f;
    
    if (!has_rep && !has_freq && !has_pres) return;
    if (context_.empty()) return;
    
    // Determine window: last N tokens from context
    int window_start = std::max(0, (int)context_.size() - penalty_window_);
    int window_len = (int)context_.size() - window_start;
    
    // Build count map on CPU (fast: window is small, typically < 256)
    std::vector<int32_t> counts(vocab_size, 0);
    for (int i = window_start; i < (int)context_.size(); i++) {
        int32_t tid = context_[i];
        if (tid >= 0 && tid < vocab_size) {
            counts[tid]++;
        }
    }
    
    // Upload counts to GPU
    ensure_penalty_buffer(vocab_size);
    cudaMemcpyAsync(penalty_gpu_, counts.data(), vocab_size * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    
    // Launch penalty kernel
    kernels::launch_repetition_penalty(
        logits,
        nullptr,
        static_cast<int32_t*>(penalty_gpu_),
        vocab_size,
        config.repetition_penalty,
        config.frequency_penalty,
        config.presence_penalty,
        stream
    );
}

// ============================================================================
// GREEDY SAMPLING (with penalties)
// ============================================================================

int32_t Sampler::sample_greedy(
    const half* logits,
    int vocab_size,
    cudaStream_t stream
) {
    SamplingConfig cfg;
    cfg.temperature = 0.0f;
    cfg.repetition_penalty = 1.2f;
    return sample(logits, vocab_size, cfg, stream);
}

// ============================================================================
// MAIN SAMPLE ENTRY POINT
// ============================================================================

int32_t Sampler::sample(
    const half* logits,
    int vocab_size,
    const SamplingConfig& config,
    cudaStream_t stream
) {
    bool needs_penalties = (config.repetition_penalty != 1.0f ||
                           config.frequency_penalty != 0.0f ||
                           config.presence_penalty != 0.0f) && !context_.empty();
    
    const half* working_logits = logits;
    half* mutable_logits = nullptr;
    
    // If we need to modify logits, work on a copy
    if (needs_penalties) {
        int top_k_eff = config.top_k > 0 ? config.top_k : 50;
        ensure_scratch(vocab_size, top_k_eff);
        
        // Logits copy is at the end of scratch buffer
        size_t top_k_bytes = top_k_eff * (sizeof(float) + sizeof(int32_t) + sizeof(float)) + sizeof(int32_t);
        mutable_logits = reinterpret_cast<half*>(static_cast<char*>(scratch_gpu_) + top_k_bytes);
        
        cudaMemcpyAsync(mutable_logits, logits, vocab_size * sizeof(half),
                        cudaMemcpyDeviceToDevice, stream);
        
        apply_penalties(mutable_logits, vocab_size, config, stream);
        working_logits = mutable_logits;
    }
    
    // Greedy path
    if (config.temperature < 0.01f) {
        kernels::launch_argmax_fp16(working_logits, result_gpu_, vocab_size, stream);
        cudaMemcpyAsync(result_cpu_, result_gpu_, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return *result_cpu_;
    }
    
    // Temperature + top-k/top-p path
    return sample_with_temperature(
        working_logits, vocab_size,
        config.temperature,
        config.top_k,
        config.top_p,
        stream
    );
}

// ============================================================================
// TEMPERATURE + TOP-K/TOP-P SAMPLING
// ============================================================================

int32_t Sampler::sample_with_temperature(
    const half* logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    cudaStream_t stream
) {
    if (top_k <= 0) {
        top_k = 50;
    }
    
    ensure_scratch(vocab_size, top_k);
    
    float* top_values = (float*)scratch_gpu_;
    int32_t* top_indices = (int32_t*)(top_values + top_k);
    float* probs = (float*)(top_indices + top_k);
    int32_t* cutoff_idx = (int32_t*)(probs + top_k);
    
    // 1. Get top-k values and indices
    kernels::launch_top_k(logits, top_values, top_indices, vocab_size, top_k, stream);
    
    // 2. Apply temperature and compute softmax on CPU
    std::vector<float> h_values(top_k);
    std::vector<int32_t> h_indices(top_k);
    
    cudaMemcpyAsync(h_values.data(), top_values, top_k * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_indices.data(), top_indices, top_k * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    float inv_temp = 1.0f / temperature;
    float max_val = h_values[0];
    
    std::vector<float> h_probs(top_k);
    float sum_exp = 0.0f;
    for (int i = 0; i < top_k; i++) {
        h_probs[i] = expf((h_values[i] - max_val) * inv_temp);
        sum_exp += h_probs[i];
    }
    for (int i = 0; i < top_k; i++) {
        h_probs[i] /= sum_exp;
    }
    
    // 3. Top-p filtering
    int num_tokens = top_k;
    if (top_p < 1.0f) {
        float cumsum = 0.0f;
        for (int i = 0; i < top_k; i++) {
            cumsum += h_probs[i];
            if (cumsum >= top_p) {
                num_tokens = i + 1;
                break;
            }
        }
        
        float new_sum = 0.0f;
        for (int i = 0; i < num_tokens; i++) {
            new_sum += h_probs[i];
        }
        for (int i = 0; i < num_tokens; i++) {
            h_probs[i] /= new_sum;
        }
    }
    
    // 4. Sample
    float r = uniform_(rng_);
    float cumsum = 0.0f;
    for (int i = 0; i < num_tokens; i++) {
        cumsum += h_probs[i];
        if (r < cumsum) {
            return h_indices[i];
        }
    }
    
    return h_indices[num_tokens - 1];
}

// ============================================================================
// BATCH SAMPLING
// ============================================================================

void Sampler::sample_batch(
    const half* logits,
    int32_t* output_tokens,
    int batch_size,
    int vocab_size,
    const SamplingConfig& config,
    cudaStream_t stream
) {
    for (int b = 0; b < batch_size; b++) {
        const half* batch_logits = logits + b * vocab_size;
        int32_t token = sample(batch_logits, vocab_size, config, stream);
        output_tokens[b] = token;
    }
}

} // namespace helios
