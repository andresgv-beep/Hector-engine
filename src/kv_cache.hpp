// src/kv_cache.hpp
// ============================================================================
// HELIOS KV CACHE - Polimórfico y Flexible
// ============================================================================
// Cache para Key/Value en atención autoregresiva.
// 
// Diseño:
//   - Dimensiones desde config, NO hardcodeadas
//   - Soporta cualquier número de layers/heads
//   - GQA automático (num_kv_heads puede ser != num_heads)
//   - Memoria pre-allocada para max_seq_len
//

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdint>

namespace helios {

// ============================================================================
// KV CACHE CONFIG - Todo desde runtime, nada hardcodeado
// ============================================================================

struct KVCacheConfig {
    uint32_t num_layers = 0;       // Desde model config
    uint32_t num_kv_heads = 0;     // Desde model config (GQA)
    uint32_t head_dim = 0;         // Desde model config
    uint32_t max_batch_size = 1;   // Configurable
    uint32_t max_seq_len = 2048;   // Configurable
    
    // Computed
    size_t bytes_per_layer() const {
        // K + V para una layer: [batch, seq, kv_heads, head_dim] * 2
        return 2 * max_batch_size * max_seq_len * num_kv_heads * head_dim * sizeof(half);
    }
    
    size_t total_bytes() const {
        return num_layers * bytes_per_layer();
    }
};

// ============================================================================
// KV CACHE
// ============================================================================

class KVCache {
public:
    KVCache() = default;
    ~KVCache();
    
    // No copy (owns GPU memory)
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;
    
    // Move OK
    KVCache(KVCache&& other) noexcept;
    KVCache& operator=(KVCache&& other) noexcept;
    
    // ========================================
    // LIFECYCLE
    // ========================================
    
    // Allocate cache for given config
    bool allocate(const KVCacheConfig& config);
    
    // Free all memory
    void free();
    
    // Reset position (start new sequence)
    void reset();
    
    // ========================================
    // ACCESS
    // ========================================
    
    // Get K cache pointer for layer [batch, seq, kv_heads, head_dim]
    half* k_cache(uint32_t layer_idx);
    const half* k_cache(uint32_t layer_idx) const;
    
    // Get V cache pointer for layer
    half* v_cache(uint32_t layer_idx);
    const half* v_cache(uint32_t layer_idx) const;
    
    // Current sequence position (0-indexed, next write position)
    uint32_t position() const { return position_; }
    
    // Advance position after writing new KV
    void advance(uint32_t num_tokens = 1);
    
    // ========================================
    // INFO
    // ========================================
    
    bool is_allocated() const { return k_data_ != nullptr; }
    const KVCacheConfig& config() const { return config_; }
    
    // For attention kernel: get dimensions
    uint32_t num_layers() const { return config_.num_layers; }
    uint32_t num_kv_heads() const { return config_.num_kv_heads; }
    uint32_t head_dim() const { return config_.head_dim; }
    uint32_t max_seq_len() const { return config_.max_seq_len; }
    uint32_t current_seq_len() const { return position_; }
    
private:
    KVCacheConfig config_;
    
    // GPU memory: [num_layers, batch, max_seq, kv_heads, head_dim]
    half* k_data_ = nullptr;
    half* v_data_ = nullptr;
    
    // Current position in sequence
    uint32_t position_ = 0;
    
    // Size of one layer's K or V cache
    size_t layer_stride_ = 0;
};

// ============================================================================
// IMPLEMENTATION
// ============================================================================

inline KVCache::~KVCache() {
    free();
}

inline KVCache::KVCache(KVCache&& other) noexcept
    : config_(other.config_)
    , k_data_(other.k_data_)
    , v_data_(other.v_data_)
    , position_(other.position_)
    , layer_stride_(other.layer_stride_)
{
    other.k_data_ = nullptr;
    other.v_data_ = nullptr;
    other.position_ = 0;
}

inline KVCache& KVCache::operator=(KVCache&& other) noexcept {
    if (this != &other) {
        free();
        config_ = other.config_;
        k_data_ = other.k_data_;
        v_data_ = other.v_data_;
        position_ = other.position_;
        layer_stride_ = other.layer_stride_;
        other.k_data_ = nullptr;
        other.v_data_ = nullptr;
        other.position_ = 0;
    }
    return *this;
}

inline bool KVCache::allocate(const KVCacheConfig& config) {
    // Validate
    if (config.num_layers == 0 || config.num_kv_heads == 0 || 
        config.head_dim == 0 || config.max_seq_len == 0) {
        return false;
    }
    
    // Free existing
    free();
    
    config_ = config;
    
    // Calculate sizes
    // Per layer: [batch, max_seq, kv_heads, head_dim]
    layer_stride_ = config_.max_batch_size * config_.max_seq_len * 
                    config_.num_kv_heads * config_.head_dim;
    
    size_t total_elements = config_.num_layers * layer_stride_;
    size_t total_bytes = total_elements * sizeof(half);
    
    // Allocate K cache
    cudaError_t err = cudaMalloc(&k_data_, total_bytes);
    if (err != cudaSuccess) {
        k_data_ = nullptr;
        return false;
    }
    
    // Allocate V cache
    err = cudaMalloc(&v_data_, total_bytes);
    if (err != cudaSuccess) {
        cudaFree(k_data_);
        k_data_ = nullptr;
        v_data_ = nullptr;
        return false;
    }
    
    // Zero initialize
    cudaMemset(k_data_, 0, total_bytes);
    cudaMemset(v_data_, 0, total_bytes);
    
    position_ = 0;
    return true;
}

inline void KVCache::free() {
    if (k_data_) {
        cudaFree(k_data_);
        k_data_ = nullptr;
    }
    if (v_data_) {
        cudaFree(v_data_);
        v_data_ = nullptr;
    }
    position_ = 0;
    layer_stride_ = 0;
}

inline void KVCache::reset() {
    position_ = 0;
    // Optionally zero memory, but not strictly necessary
}

inline half* KVCache::k_cache(uint32_t layer_idx) {
    if (!k_data_ || layer_idx >= config_.num_layers) return nullptr;
    return k_data_ + layer_idx * layer_stride_;
}

inline const half* KVCache::k_cache(uint32_t layer_idx) const {
    if (!k_data_ || layer_idx >= config_.num_layers) return nullptr;
    return k_data_ + layer_idx * layer_stride_;
}

inline half* KVCache::v_cache(uint32_t layer_idx) {
    if (!v_data_ || layer_idx >= config_.num_layers) return nullptr;
    return v_data_ + layer_idx * layer_stride_;
}

inline const half* KVCache::v_cache(uint32_t layer_idx) const {
    if (!v_data_ || layer_idx >= config_.num_layers) return nullptr;
    return v_data_ + layer_idx * layer_stride_;
}

inline void KVCache::advance(uint32_t num_tokens) {
    position_ += num_tokens;
    if (position_ > config_.max_seq_len) {
        position_ = config_.max_seq_len;  // Clamp, don't overflow
    }
}

} // namespace helios
