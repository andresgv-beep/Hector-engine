// optype.cpp
// HELIOS ENGINE - OpType Registry Implementation
// ==============================================

#include "optype.hpp"
#include <stdexcept>

namespace helios {

// ============================================================================
// OPTYPE REGISTRY
// ============================================================================

OpTypeRegistry& OpTypeRegistry::instance() {
    static OpTypeRegistry instance;
    return instance;
}

OpTypeRegistry::OpTypeRegistry() {
    register_builtins();
}

void OpTypeRegistry::register_builtins() {
    // Usar register_op directamente para evitar recursión via instance()
    OpTypeInfo info;
    
    // Core
    info = OpTypeInfo{}; info.name = "nop"; info.category = "core"; 
    info.min_inputs = 0; info.max_inputs = 0;
    register_op(info);
    
    info = OpTypeInfo{}; info.name = "copy"; info.category = "core";
    info.min_inputs = 1; info.max_inputs = 1;
    register_op(info);
    
    // Elementwise
    info = OpTypeInfo{}; info.name = "add"; info.category = "elementwise";
    info.min_inputs = 2; info.max_inputs = 2;
    register_op(info);
    
    info = OpTypeInfo{}; info.name = "add_bias"; info.category = "elementwise";
    info.min_inputs = 2; info.max_inputs = 2;
    register_op(info);
    
    info = OpTypeInfo{}; info.name = "mul"; info.category = "elementwise";
    info.min_inputs = 2; info.max_inputs = 2;
    register_op(info);

    info = OpTypeInfo{}; info.name = "mul_scalar_tensor"; info.category = "elementwise";
    info.min_inputs = 2; info.max_inputs = 2;
    register_op(info);
    
    info = OpTypeInfo{}; info.name = "scale"; info.category = "elementwise";
    info.min_inputs = 1; info.max_inputs = 1;
    register_op(info);
    
    // Activations
    info = OpTypeInfo{}; info.name = "silu"; info.category = "activation";
    info.min_inputs = 1; info.max_inputs = 1;
    register_op(info);
    
    info = OpTypeInfo{}; info.name = "gelu"; info.category = "activation";
    info.min_inputs = 1; info.max_inputs = 1;
    register_op(info);
    
    info = OpTypeInfo{}; info.name = "relu"; info.category = "activation";
    info.min_inputs = 1; info.max_inputs = 1;
    register_op(info);
    
    info = OpTypeInfo{}; info.name = "softmax"; info.category = "activation";
    info.min_inputs = 1; info.max_inputs = 1; info.is_reduction = true;
    register_op(info);

    info = OpTypeInfo{}; info.name = "softcap"; info.category = "activation";
    info.min_inputs = 1; info.max_inputs = 1;
    register_op(info);

    info = OpTypeInfo{}; info.name = "silu_mul"; info.category = "activation";
    info.min_inputs = 2; info.max_inputs = 2;
    register_op(info);

    info = OpTypeInfo{}; info.name = "gelu_mul"; info.category = "activation";
    info.min_inputs = 2; info.max_inputs = 2;
    register_op(info);
    
    // Normalization
    info = OpTypeInfo{}; info.name = "rmsnorm"; info.category = "norm";
    info.min_inputs = 1; info.max_inputs = 2;
    register_op(info);
    
    info = OpTypeInfo{}; info.name = "layernorm"; info.category = "norm";
    info.min_inputs = 2; info.max_inputs = 3;
    register_op(info);

    info = OpTypeInfo{}; info.name = "add_rmsnorm"; info.category = "norm";
    info.min_inputs = 3; info.max_inputs = 3;
    register_op(info);
    
    // Linear algebra
    info = OpTypeInfo{}; info.name = "matmul"; info.category = "linear";
    info.min_inputs = 2; info.max_inputs = 2;
    register_op(info);
    
    info = OpTypeInfo{}; info.name = "matmul_t"; info.category = "linear";
    info.min_inputs = 2; info.max_inputs = 2;
    register_op(info);
    
    // Position encoding
    info = OpTypeInfo{}; info.name = "rope"; info.category = "position";
    info.min_inputs = 1; info.max_inputs = 2;
    register_op(info);
    
    // Attention
    info = OpTypeInfo{}; info.name = "attention"; info.category = "attention";
    info.min_inputs = 3; info.max_inputs = 4;
    register_op(info);
    
    info = OpTypeInfo{}; info.name = "attention_cached"; info.category = "attention";
    info.min_inputs = 5; info.max_inputs = 5;  // Q, K_cache, V_cache, K_new, V_new
    register_op(info);
    
    info = OpTypeInfo{}; info.name = "kv_cache_update"; info.category = "attention";
    info.min_inputs = 2; info.max_inputs = 2;  // K/V new, K/V cache
    register_op(info);

    info = OpTypeInfo{}; info.name = "qk_norm_rope"; info.category = "attention";
    info.min_inputs = 4; info.max_inputs = 4;
    register_op(info);

    info = OpTypeInfo{}; info.name = "attention_prefill_cached"; info.category = "attention";
    info.min_inputs = 3; info.max_inputs = 3;
    register_op(info);
    
    // Quantization
    info = OpTypeInfo{}; info.name = "dequant"; info.category = "quantization";
    info.min_inputs = 1; info.max_inputs = 1; info.is_quantized = true;
    register_op(info);
    
    // Memory
    info = OpTypeInfo{}; info.name = "reshape"; info.category = "memory";
    info.min_inputs = 1; info.max_inputs = 1; info.is_inplace = true;
    register_op(info);
    
    info = OpTypeInfo{}; info.name = "transpose"; info.category = "memory";
    info.min_inputs = 1; info.max_inputs = 1;
    register_op(info);
    
    info = OpTypeInfo{}; info.name = "concat"; info.category = "memory";
    info.min_inputs = 2; info.max_inputs = 0;  // unlimited
    register_op(info);
    
    info = OpTypeInfo{}; info.name = "split"; info.category = "memory";
    info.min_inputs = 1; info.max_inputs = 1;
    register_op(info);
    
    info = OpTypeInfo{}; info.name = "split_qkv"; info.category = "memory";
    info.min_inputs = 1; info.max_inputs = 1;  // Input: fused QKV
    register_op(info);
    
    info = OpTypeInfo{}; info.name = "split_half"; info.category = "memory";
    info.min_inputs = 1; info.max_inputs = 1;  // Input: fused tensor
    register_op(info);
    
    info = OpTypeInfo{}; info.name = "embedding"; info.category = "memory";
    info.min_inputs = 2; info.max_inputs = 2;
    register_op(info);

    // Replaces complete rows in an existing activation tensor. The source
    // rows and their INT32 destination indices are separate inputs so the op
    // remains capturable by CUDA Graphs.
    info = OpTypeInfo{}; info.name = "scatter_rows"; info.category = "memory";
    info.min_inputs = 2; info.max_inputs = 2; info.is_inplace = true;
    register_op(info);

    info = OpTypeInfo{}; info.name = "ple_slice"; info.category = "memory";
    info.min_inputs = 1; info.max_inputs = 1;
    register_op(info);
}

OpTypeID OpTypeRegistry::register_op(const OpTypeInfo& info) {
    if (info.name.empty()) {
        throw std::runtime_error("Op name cannot be empty");
    }
    
    // Check for duplicate name
    if (by_name_.count(info.name)) {
        throw std::runtime_error("Op already registered: " + info.name);
    }
    
    OpTypeID id = next_id_++;
    
    OpTypeInfo stored = info;
    stored.id = id;
    
    by_id_[id] = stored;
    by_name_[info.name] = id;
    
    return id;
}

const OpTypeInfo* OpTypeRegistry::get(OpTypeID id) const {
    auto it = by_id_.find(id);
    return (it != by_id_.end()) ? &it->second : nullptr;
}

const OpTypeInfo* OpTypeRegistry::get(const std::string& name) const {
    auto it = by_name_.find(name);
    if (it == by_name_.end()) return nullptr;
    return get(it->second);
}

OpTypeID OpTypeRegistry::get_id(const std::string& name) const {
    auto it = by_name_.find(name);
    return (it != by_name_.end()) ? it->second : OP_INVALID;
}

bool OpTypeRegistry::exists(const std::string& name) const {
    return by_name_.count(name) > 0;
}

bool OpTypeRegistry::exists(OpTypeID id) const {
    return by_id_.count(id) > 0;
}

} // namespace helios
