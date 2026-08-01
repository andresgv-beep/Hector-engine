// dtype.hpp
// HELIOS ENGINE - Extensible Data Types
// ======================================
// Sistema de tipos de datos extensible en runtime.
// Nuevos dtypes se registran, no se hardcodean.
//

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <functional>
#include <memory>

namespace helios {

// ============================================================================
// DTYPE ID (lightweight, copyable)
// ============================================================================

using DTypeID = uint16_t;

constexpr DTypeID DTYPE_INVALID = 0;

// ============================================================================
// DTYPE INFO (full description)
// ============================================================================

struct DTypeInfo {
    DTypeID id;
    std::string name;
    
    // Size info
    size_t element_bits;      // Bits per element (0 if variable/block-based)
    size_t block_elements;    // Elements per block (for quantized types)
    size_t block_bytes;       // Bytes per block (for quantized types)
    
    // Flags
    bool is_quantized;
    bool is_floating;
    bool is_signed;
    
    // Size calculator (for complex types)
    // Returns bytes needed for `num_elements` elements
    std::function<size_t(size_t num_elements)> calc_size;
    
    // Convenience methods
    size_t size_bytes(size_t num_elements) const {
        if (calc_size) {
            return calc_size(num_elements);
        }
        // Simple calculation for non-quantized
        return (num_elements * element_bits + 7) / 8;
    }
    
    bool is_valid() const { return id != DTYPE_INVALID; }
};

// ============================================================================
// DTYPE REGISTRY (singleton)
// ============================================================================

class DTypeRegistry {
public:
    // Get singleton instance
    static DTypeRegistry& instance();
    
    // Register new dtype, returns assigned ID
    DTypeID register_dtype(const DTypeInfo& info);
    
    // Register with builder pattern
    class Builder {
    public:
        Builder(const std::string& name) { info_.name = name; }
        
        Builder& bits(size_t b) { info_.element_bits = b; return *this; }
        Builder& block(size_t elements, size_t bytes) {
            info_.block_elements = elements;
            info_.block_bytes = bytes;
            return *this;
        }
        Builder& quantized(bool q = true) { info_.is_quantized = q; return *this; }
        Builder& floating(bool f = true) { info_.is_floating = f; return *this; }
        Builder& signed_type(bool s = true) { info_.is_signed = s; return *this; }
        Builder& size_fn(std::function<size_t(size_t)> fn) { 
            info_.calc_size = fn; 
            return *this; 
        }
        
        // Build and register in default registry
        DTypeID build() { return DTypeRegistry::instance().register_dtype(info_); }
        
        // Build info without registering (for internal use)
        const DTypeInfo& info() const { return info_; }
        
    private:
        DTypeInfo info_{};
    };
    
    // Lookup
    const DTypeInfo* get(DTypeID id) const;
    const DTypeInfo* get(const std::string& name) const;
    DTypeID get_id(const std::string& name) const;
    
    // Iteration
    const std::unordered_map<DTypeID, DTypeInfo>& all() const { return by_id_; }
    
private:
    DTypeRegistry();
    void register_builtins();
    
    std::unordered_map<DTypeID, DTypeInfo> by_id_;
    std::unordered_map<std::string, DTypeID> by_name_;
    DTypeID next_id_ = 1;
};

// ============================================================================
// CONVENIENCE ACCESSORS
// ============================================================================

inline const DTypeInfo* dtype_info(DTypeID id) {
    return DTypeRegistry::instance().get(id);
}

inline const char* dtype_name(DTypeID id) {
    auto* info = dtype_info(id);
    return info ? info->name.c_str() : "invalid";
}

inline size_t dtype_size(DTypeID id, size_t num_elements) {
    auto* info = dtype_info(id);
    return info ? info->size_bytes(num_elements) : 0;
}

inline bool dtype_is_quantized(DTypeID id) {
    auto* info = dtype_info(id);
    return info ? info->is_quantized : false;
}

// ============================================================================
// BUILTIN DTYPE IDS (functions to avoid static init order issues)
// ============================================================================

namespace dtype {
    inline DTypeID FP32() { return DTypeRegistry::instance().get_id("fp32"); }
    inline DTypeID FP16() { return DTypeRegistry::instance().get_id("fp16"); }
    inline DTypeID BF16() { return DTypeRegistry::instance().get_id("bf16"); }
    inline DTypeID FP8() { return DTypeRegistry::instance().get_id("fp8"); }
    inline DTypeID INT8() { return DTypeRegistry::instance().get_id("int8"); }
    inline DTypeID INT32() { return DTypeRegistry::instance().get_id("int32"); }
    inline DTypeID HQ4K() { return DTypeRegistry::instance().get_id("hq4k"); }
    inline DTypeID HQ5K() { return DTypeRegistry::instance().get_id("hq5k"); }
    inline DTypeID HQ31K() { return DTypeRegistry::instance().get_id("hq31k"); }
    inline DTypeID HQ41K() { return DTypeRegistry::instance().get_id("hq41k"); }
    inline DTypeID HQ51K() { return DTypeRegistry::instance().get_id("hq51k"); }
}

} // namespace helios
