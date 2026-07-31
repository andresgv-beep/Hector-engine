// optype.hpp
// HELIOS ENGINE - Extensible Operation Types
// ==========================================
// Sistema de operaciones extensible en runtime.
// Nuevas ops se registran, no se hardcodean.
//

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <functional>

namespace helios {

// ============================================================================
// OPTYPE ID (lightweight, copyable)
// ============================================================================

using OpTypeID = uint32_t;

constexpr OpTypeID OP_INVALID = 0;

// ============================================================================
// OPTYPE INFO
// ============================================================================

struct OpTypeInfo {
    OpTypeID id;
    std::string name;
    std::string category;     // "elementwise", "linear", "norm", "attention", etc.
    
    // Constraints
    uint8_t min_inputs;       // Minimum number of inputs
    uint8_t max_inputs;       // Maximum number of inputs (0 = unlimited)
    bool requires_output;     // Must have output tensor
    
    // Flags
    bool is_inplace;          // Can operate in-place
    bool is_reduction;        // Reduces dimensions
    bool is_quantized;        // Operates on quantized data
    
    bool is_valid() const { return id != OP_INVALID; }
};

// ============================================================================
// OPTYPE REGISTRY (singleton)
// ============================================================================

class OpTypeRegistry {
public:
    // Get singleton instance
    static OpTypeRegistry& instance();
    
    // Register new op type, returns assigned ID
    OpTypeID register_op(const OpTypeInfo& info);
    
    // Register with builder pattern
    class Builder {
    public:
        Builder(const std::string& name) { info_.name = name; }
        
        Builder& category(const std::string& c) { info_.category = c; return *this; }
        Builder& inputs(uint8_t min, uint8_t max = 0) { 
            info_.min_inputs = min; 
            info_.max_inputs = max;
            return *this; 
        }
        Builder& requires_output(bool r = true) { info_.requires_output = r; return *this; }
        Builder& inplace(bool i = true) { info_.is_inplace = i; return *this; }
        Builder& reduction(bool r = true) { info_.is_reduction = r; return *this; }
        Builder& quantized(bool q = true) { info_.is_quantized = q; return *this; }
        
        // Build and register in default registry
        OpTypeID build() { return OpTypeRegistry::instance().register_op(info_); }
        
        // Get info without registering (for internal use)
        const OpTypeInfo& info() const { return info_; }
        
    private:
        OpTypeInfo info_{};
    };
    
    // Lookup
    const OpTypeInfo* get(OpTypeID id) const;
    const OpTypeInfo* get(const std::string& name) const;
    OpTypeID get_id(const std::string& name) const;
    
    // Check existence
    bool exists(const std::string& name) const;
    bool exists(OpTypeID id) const;
    
    // Iteration
    const std::unordered_map<OpTypeID, OpTypeInfo>& all() const { return by_id_; }
    
private:
    OpTypeRegistry();
    void register_builtins();
    
    std::unordered_map<OpTypeID, OpTypeInfo> by_id_;
    std::unordered_map<std::string, OpTypeID> by_name_;
    OpTypeID next_id_ = 1;
};

// ============================================================================
// CONVENIENCE ACCESSORS
// ============================================================================

inline const OpTypeInfo* op_info(OpTypeID id) {
    return OpTypeRegistry::instance().get(id);
}

inline const char* op_name(OpTypeID id) {
    auto* info = op_info(id);
    return info ? info->name.c_str() : "invalid";
}

inline OpTypeID op_id(const std::string& name) {
    return OpTypeRegistry::instance().get_id(name);
}

// ============================================================================
// BUILTIN OP IDS (functions to avoid static init order issues)
// ============================================================================

namespace op {
    inline OpTypeID NOP() { return OpTypeRegistry::instance().get_id("nop"); }
    inline OpTypeID COPY() { return OpTypeRegistry::instance().get_id("copy"); }
    
    inline OpTypeID ADD() { return OpTypeRegistry::instance().get_id("add"); }
    inline OpTypeID ADD_BIAS() { return OpTypeRegistry::instance().get_id("add_bias"); }
    inline OpTypeID MUL() { return OpTypeRegistry::instance().get_id("mul"); }
    inline OpTypeID MUL_SCALAR_TENSOR() { return OpTypeRegistry::instance().get_id("mul_scalar_tensor"); }
    inline OpTypeID SCALE() { return OpTypeRegistry::instance().get_id("scale"); }
    
    inline OpTypeID SILU() { return OpTypeRegistry::instance().get_id("silu"); }
    inline OpTypeID GELU() { return OpTypeRegistry::instance().get_id("gelu"); }
    inline OpTypeID RELU() { return OpTypeRegistry::instance().get_id("relu"); }
    inline OpTypeID SOFTMAX() { return OpTypeRegistry::instance().get_id("softmax"); }
    inline OpTypeID SOFTCAP() { return OpTypeRegistry::instance().get_id("softcap"); }
    
    inline OpTypeID RMSNORM() { return OpTypeRegistry::instance().get_id("rmsnorm"); }
    inline OpTypeID LAYERNORM() { return OpTypeRegistry::instance().get_id("layernorm"); }
    
    inline OpTypeID MATMUL() { return OpTypeRegistry::instance().get_id("matmul"); }
    inline OpTypeID MATMUL_T() { return OpTypeRegistry::instance().get_id("matmul_t"); }
    
    inline OpTypeID ROPE() { return OpTypeRegistry::instance().get_id("rope"); }
    
    inline OpTypeID ATTENTION() { return OpTypeRegistry::instance().get_id("attention"); }
    inline OpTypeID ATTENTION_CACHED() { return OpTypeRegistry::instance().get_id("attention_cached"); }
    inline OpTypeID KV_CACHE_UPDATE() { return OpTypeRegistry::instance().get_id("kv_cache_update"); }
    
    inline OpTypeID DEQUANT() { return OpTypeRegistry::instance().get_id("dequant"); }
    
    inline OpTypeID RESHAPE() { return OpTypeRegistry::instance().get_id("reshape"); }
    inline OpTypeID TRANSPOSE() { return OpTypeRegistry::instance().get_id("transpose"); }
    inline OpTypeID CONCAT() { return OpTypeRegistry::instance().get_id("concat"); }
    inline OpTypeID SPLIT() { return OpTypeRegistry::instance().get_id("split"); }
    inline OpTypeID SPLIT_QKV() { return OpTypeRegistry::instance().get_id("split_qkv"); }
    inline OpTypeID SPLIT_HALF() { return OpTypeRegistry::instance().get_id("split_half"); }
    inline OpTypeID EMBEDDING() { return OpTypeRegistry::instance().get_id("embedding"); }
}

} // namespace helios
