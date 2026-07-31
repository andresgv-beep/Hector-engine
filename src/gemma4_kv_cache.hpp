// Gemma 4 heterogeneous/shared KV cache.
//
// The first N - num_kv_shared_layers layers own physical cache slots. Later
// local layers alias the last local source slot and later global layers alias
// the last global source slot. Slot strides follow each source head_dim, so a
// 256-wide local cache never pays for the 512-wide global geometry.

#pragma once

#include "hnf_loader.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace helios {

class Gemma4KVCache {
public:
    Gemma4KVCache() = default;
    ~Gemma4KVCache() { free(); }

    Gemma4KVCache(const Gemma4KVCache&) = delete;
    Gemma4KVCache& operator=(const Gemma4KVCache&) = delete;

    Gemma4KVCache(Gemma4KVCache&& other) noexcept { move_from(other); }
    Gemma4KVCache& operator=(Gemma4KVCache&& other) noexcept {
        if (this != &other) {
            free();
            move_from(other);
        }
        return *this;
    }

    bool allocate(const Gemma4Config& gemma, uint32_t num_kv_heads,
                  uint32_t max_batch_size, uint32_t max_seq_len) {
        free();
        if (gemma.layers.empty() || num_kv_heads == 0 ||
            max_batch_size == 0 || max_seq_len == 0 ||
            gemma.num_kv_shared_layers > gemma.layers.size()) {
            return false;
        }

        const uint32_t layers = static_cast<uint32_t>(gemma.layers.size());
        const uint32_t first_shared = layers - gemma.num_kv_shared_layers;
        if (first_shared == 0) return false;

        num_kv_heads_ = num_kv_heads;
        max_batch_size_ = max_batch_size;
        max_seq_len_ = max_seq_len;
        first_shared_layer_ = first_shared;
        source_layers_.resize(layers);
        offsets_.resize(layers);
        head_dims_.resize(layers);

        int32_t last_local = -1;
        int32_t last_global = -1;
        size_t total_elements = 0;

        for (uint32_t i = 0; i < first_shared; ++i) {
            const auto& layer = gemma.layers[i];
            if (layer.head_dim == 0) return fail_layout();
            source_layers_[i] = i;
            offsets_[i] = total_elements;
            head_dims_[i] = layer.head_dim;
            if (layer.is_global_attention()) last_global = static_cast<int32_t>(i);
            else last_local = static_cast<int32_t>(i);

            size_t stride = 0;
            if (!checked_stride(layer.head_dim, stride) ||
                total_elements > std::numeric_limits<size_t>::max() - stride) {
                return fail_layout();
            }
            total_elements += stride;
        }

        for (uint32_t i = first_shared; i < layers; ++i) {
            const auto& layer = gemma.layers[i];
            const int32_t source = layer.is_global_attention() ? last_global : last_local;
            if (source < 0 || layer.head_dim != gemma.layers[source].head_dim) {
                return fail_layout();
            }
            source_layers_[i] = static_cast<uint32_t>(source);
            offsets_[i] = offsets_[source];
            head_dims_[i] = layer.head_dim;
        }

        if (total_elements == 0 ||
            total_elements > std::numeric_limits<size_t>::max() / sizeof(half)) {
            return fail_layout();
        }
        elements_per_plane_ = total_elements;
        const size_t bytes = total_elements * sizeof(half);
        if (cudaMalloc(&k_data_, bytes) != cudaSuccess) {
            k_data_ = nullptr;
            return fail_layout();
        }
        if (cudaMalloc(&v_data_, bytes) != cudaSuccess) {
            cudaFree(k_data_);
            k_data_ = nullptr;
            v_data_ = nullptr;
            return fail_layout();
        }
        if (cudaMemset(k_data_, 0, bytes) != cudaSuccess ||
            cudaMemset(v_data_, 0, bytes) != cudaSuccess) {
            return fail_layout();
        }
        position_ = 0;
        return true;
    }

    void free() {
        if (k_data_) cudaFree(k_data_);
        if (v_data_) cudaFree(v_data_);
        k_data_ = nullptr;
        v_data_ = nullptr;
        source_layers_.clear();
        offsets_.clear();
        head_dims_.clear();
        num_kv_heads_ = 0;
        max_batch_size_ = 0;
        max_seq_len_ = 0;
        first_shared_layer_ = 0;
        elements_per_plane_ = 0;
        position_ = 0;
    }

    void register_tensors(Engine& engine, const std::string& prefix) {
        if (!is_allocated()) {
            throw std::runtime_error("Gemma4KVCache: cache is not allocated");
        }
        for (uint32_t layer = 0; layer < num_layers(); ++layer) {
            const std::vector<uint32_t> shape{
                max_batch_size_, max_seq_len_, num_kv_heads_, head_dims_[layer]};
            const std::string base = prefix + ".layer" + std::to_string(layer);
            engine.tensors().register_external(base + ".k", k_cache(layer), shape,
                                               dtype::FP16());
            engine.tensors().register_external(base + ".v", v_cache(layer), shape,
                                               dtype::FP16());
        }
    }

    half* k_cache(uint32_t layer) {
        return valid_layer(layer) ? k_data_ + offsets_[layer] : nullptr;
    }
    const half* k_cache(uint32_t layer) const {
        return valid_layer(layer) ? k_data_ + offsets_[layer] : nullptr;
    }
    half* v_cache(uint32_t layer) {
        return valid_layer(layer) ? v_data_ + offsets_[layer] : nullptr;
    }
    const half* v_cache(uint32_t layer) const {
        return valid_layer(layer) ? v_data_ + offsets_[layer] : nullptr;
    }

    bool is_allocated() const { return k_data_ && v_data_; }
    uint32_t num_layers() const { return static_cast<uint32_t>(source_layers_.size()); }
    uint32_t physical_slots() const { return first_shared_layer_; }
    uint32_t first_shared_layer() const { return first_shared_layer_; }
    uint32_t source_layer(uint32_t layer) const {
        return layer < source_layers_.size() ? source_layers_[layer]
                                             : std::numeric_limits<uint32_t>::max();
    }
    bool is_shared(uint32_t layer) const {
        return layer < source_layers_.size() && source_layers_[layer] != layer;
    }
    uint32_t head_dim(uint32_t layer) const {
        return layer < head_dims_.size() ? head_dims_[layer] : 0;
    }
    uint32_t num_kv_heads() const { return num_kv_heads_; }
    uint32_t max_batch_size() const { return max_batch_size_; }
    uint32_t max_seq_len() const { return max_seq_len_; }
    size_t total_bytes() const { return 2 * elements_per_plane_ * sizeof(half); }

    uint32_t position() const { return position_; }
    void reset() { position_ = 0; }
    void advance(uint32_t tokens = 1) {
        position_ = tokens > max_seq_len_ - position_ ? max_seq_len_
                                                       : position_ + tokens;
    }
    void rewind_to(uint32_t position) {
        if (position <= position_) position_ = position;
    }

private:
    bool checked_stride(uint32_t head_dim, size_t& stride) const {
        size_t value = max_batch_size_;
        const size_t factors[] = {max_seq_len_, num_kv_heads_, head_dim};
        for (size_t factor : factors) {
            if (factor == 0 || value > std::numeric_limits<size_t>::max() / factor) {
                return false;
            }
            value *= factor;
        }
        stride = value;
        return true;
    }

    bool fail_layout() {
        free();
        return false;
    }

    bool valid_layer(uint32_t layer) const {
        return is_allocated() && layer < offsets_.size();
    }

    void move_from(Gemma4KVCache& other) {
        k_data_ = other.k_data_;
        v_data_ = other.v_data_;
        source_layers_ = std::move(other.source_layers_);
        offsets_ = std::move(other.offsets_);
        head_dims_ = std::move(other.head_dims_);
        num_kv_heads_ = other.num_kv_heads_;
        max_batch_size_ = other.max_batch_size_;
        max_seq_len_ = other.max_seq_len_;
        first_shared_layer_ = other.first_shared_layer_;
        elements_per_plane_ = other.elements_per_plane_;
        position_ = other.position_;
        other.k_data_ = nullptr;
        other.v_data_ = nullptr;
        other.free();
    }

    half* k_data_ = nullptr;
    half* v_data_ = nullptr;
    std::vector<uint32_t> source_layers_;
    std::vector<size_t> offsets_;
    std::vector<uint32_t> head_dims_;
    uint32_t num_kv_heads_ = 0;
    uint32_t max_batch_size_ = 0;
    uint32_t max_seq_len_ = 0;
    uint32_t first_shared_layer_ = 0;
    size_t elements_per_plane_ = 0;
    uint32_t position_ = 0;
};

} // namespace helios
