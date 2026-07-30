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
    void launch_sample_topk_gpu(const float*, const int32_t*, int32_t*, int, float, float, float, cudaStream_t);
    void launch_repetition_penalty(half*, const int32_t*, const int32_t*, int, float, float, float, cudaStream_t);
    void launch_window_penalty(half*, const int32_t*, const int32_t*, int, float, float, float, cudaStream_t);
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

    // v2: deduplicar la ventana en CPU (≤128 elementos, coste trivial) y subir
    // SOLO los pares (id, count) — ~1 KB. La v1 subía counts[vocab] = 608 KB
    // por token, que hundía el decode del chat a un tercio de su velocidad.
    std::vector<int32_t> ids;
    std::vector<int32_t> counts;
    ids.reserve(penalty_window_);
    counts.reserve(penalty_window_);
    for (int i = window_start; i < (int)context_.size(); i++) {
        int32_t tid = context_[i];
        if (tid < 0 || tid >= vocab_size) continue;
        bool found = false;
        for (size_t j = 0; j < ids.size(); j++) {
            if (ids[j] == tid) { counts[j]++; found = true; break; }
        }
        if (!found) { ids.push_back(tid); counts.push_back(1); }
    }
    if (ids.empty()) return;

    int n = (int)ids.size();
    ensure_penalty_buffer(2 * penalty_window_);  // ids + counts, en int32
    int32_t* d_ids = static_cast<int32_t*>(penalty_gpu_);
    int32_t* d_counts = d_ids + penalty_window_;
    cudaMemcpyAsync(d_ids, ids.data(), n * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_counts, counts.data(), n * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);

    kernels::launch_window_penalty(
        logits, d_ids, d_counts, n,
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

    // Pipeline 100% GPU: top-k → softmax+top-p+muestreo en un kernel →
    // copiar de vuelta UN int32. Mismo coste de sincronización que el greedy.
    // (Antes: 2 viajes D2H de valores/índices + softmax en CPU ≈ ms/token.)
    kernels::launch_top_k(logits, top_values, top_indices, vocab_size, top_k, stream);

    float r = uniform_(rng_);  // el RNG vive en CPU: reproducible y barato
    kernels::launch_sample_topk_gpu(
        top_values, top_indices, result_gpu_,
        top_k, 1.0f / temperature, top_p, r, stream);

    cudaMemcpyAsync(result_cpu_, result_gpu_, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return *result_cpu_;
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
