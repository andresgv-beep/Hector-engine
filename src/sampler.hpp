// src/sampler.hpp
// ============================================================================
// HELIOS SAMPLER - Polimórfico y Extensible
// ============================================================================
// Estrategias de sampling registrables en runtime.
// Sin hardcoding de métodos específicos.
//

#pragma once

#include "tensor.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <functional>
#include <unordered_map>
#include <string>
#include <random>
#include <memory>

namespace helios {

// ============================================================================
// SAMPLING CONFIG - Parámetros flexibles
// ============================================================================

struct SamplingConfig {
    float temperature = 1.0f;   // 1.0 = no change, <1 = sharper, >1 = flatter
    int top_k = 0;              // 0 = disabled
    float top_p = 1.0f;         // 1.0 = disabled (nucleus sampling)
    float repetition_penalty = 1.0f;  // 1.0 = disabled, >1 = penalize repeats
    float frequency_penalty = 0.0f;   // 0.0 = disabled, penalize by count
    float presence_penalty = 0.0f;    // 0.0 = disabled, penalize if present at all
    uint64_t seed = 0;          // 0 = random seed
    
    // Factory methods — polimórfico, sin hardcoding
    static SamplingConfig greedy() {
        return SamplingConfig{.temperature = 0.0f, .repetition_penalty = 1.2f};
    }
    
    static SamplingConfig creative(float temp = 0.8f, int k = 50, float p = 0.9f) {
        return SamplingConfig{.temperature = temp, .top_k = k, .top_p = p,
                              .repetition_penalty = 1.15f, .frequency_penalty = 0.1f};
    }
    
    static SamplingConfig balanced(float temp = 0.7f, int k = 40) {
        return SamplingConfig{.temperature = temp, .top_k = k, 
                              .repetition_penalty = 1.2f};
    }
    
    static SamplingConfig deterministic() {
        return SamplingConfig{.temperature = 0.0f};
    }
};

// ============================================================================
// SAMPLER CLASS
// ============================================================================

class Sampler {
public:
    Sampler();
    ~Sampler();
    
    // No copy (owns GPU memory)
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    
    // ========================================
    // MAIN API
    // ========================================
    
    // Sample next token from logits (context-aware)
    int32_t sample(
        const half* logits,     // [vocab_size] on GPU
        int vocab_size,
        const SamplingConfig& config,
        cudaStream_t stream = nullptr
    );
    
    // Greedy (argmax) with repetition penalty
    int32_t sample_greedy(
        const half* logits,
        int vocab_size,
        cudaStream_t stream = nullptr
    );
    
    // ========================================
    // CONTEXT MANAGEMENT
    // ========================================
    
    // Feed tokens into context (prompt + generated)
    void add_context(int32_t token_id);
    void add_context(const std::vector<int32_t>& tokens);
    void clear_context();
    
    // Set max context window for penalty lookback
    void set_penalty_window(int window) { penalty_window_ = window; }
    
    // ========================================
    // BATCH API (for parallel generation)
    // ========================================
    
    void sample_batch(
        const half* logits,     // [batch, vocab_size]
        int32_t* output_tokens, // [batch]
        int batch_size,
        int vocab_size,
        const SamplingConfig& config,
        cudaStream_t stream = nullptr
    );
    
    // ========================================
    // STATE
    // ========================================
    
    void set_seed(uint64_t seed);
    void reset_rng();
    
private:
    // GPU scratch memory for sampling operations
    void* scratch_gpu_ = nullptr;
    size_t scratch_size_ = 0;
    
    // CPU output buffer
    int32_t* result_cpu_ = nullptr;
    int32_t* result_gpu_ = nullptr;
    
    // Random number generator
    std::mt19937_64 rng_;
    std::uniform_real_distribution<float> uniform_{0.0f, 1.0f};
    
    // Context for repetition penalty
    std::vector<int32_t> context_;
    int penalty_window_ = 128;  // Look back N tokens
    
    // GPU buffer for penalty counts
    void* penalty_gpu_ = nullptr;
    size_t penalty_gpu_size_ = 0;
    
    // Ensure scratch is allocated
    void ensure_scratch(int vocab_size, int top_k);
    void ensure_penalty_buffer(int vocab_size);
    
    // Apply repetition/frequency/presence penalties to logits (GPU)
    void apply_penalties(
        half* logits,           // [vocab_size] — modified in place
        int vocab_size,
        const SamplingConfig& config,
        cudaStream_t stream
    );
    
    // Internal sampling implementations
    int32_t sample_with_temperature(
        const half* logits,
        int vocab_size,
        float temperature,
        int top_k,
        float top_p,
        cudaStream_t stream
    );
};

// ============================================================================
// IMPLEMENTATION (header-only for simplicity)
// ============================================================================

inline Sampler::Sampler() {
    cudaMalloc(&result_gpu_, sizeof(int32_t));
    result_cpu_ = new int32_t;
    
    std::random_device rd;
    rng_.seed(rd());
}

inline Sampler::~Sampler() {
    if (scratch_gpu_) cudaFree(scratch_gpu_);
    if (penalty_gpu_) cudaFree(penalty_gpu_);
    if (result_gpu_) cudaFree(result_gpu_);
    delete result_cpu_;
}

inline void Sampler::set_seed(uint64_t seed) {
    rng_.seed(seed);
}

inline void Sampler::reset_rng() {
    std::random_device rd;
    rng_.seed(rd());
}

inline void Sampler::add_context(int32_t token_id) {
    context_.push_back(token_id);
}

inline void Sampler::add_context(const std::vector<int32_t>& tokens) {
    context_.insert(context_.end(), tokens.begin(), tokens.end());
}

inline void Sampler::clear_context() {
    context_.clear();
}

inline void Sampler::ensure_scratch(int vocab_size, int top_k) {
    size_t needed = 0;
    if (top_k > 0) {
        needed = top_k * sizeof(float) +      // top_values
                 top_k * sizeof(int32_t) +    // top_indices
                 top_k * sizeof(float) +      // probs (after softmax)
                 sizeof(int32_t);             // cutoff index
    }
    // Add space for logits copy (needed for penalty application)
    needed += vocab_size * sizeof(half);
    
    if (needed > scratch_size_) {
        if (scratch_gpu_) cudaFree(scratch_gpu_);
        cudaMalloc(&scratch_gpu_, needed);
        scratch_size_ = needed;
    }
}

inline void Sampler::ensure_penalty_buffer(int vocab_size) {
    size_t needed = vocab_size * sizeof(int32_t);
    if (needed > penalty_gpu_size_) {
        if (penalty_gpu_) cudaFree(penalty_gpu_);
        cudaMalloc(&penalty_gpu_, needed);
        penalty_gpu_size_ = needed;
    }
}

} // namespace helios
