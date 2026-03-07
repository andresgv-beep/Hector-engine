// dtype.cpp
// HELIOS ENGINE - DType Registry Implementation
// ==============================================

#include "dtype.hpp"
#include <stdexcept>

namespace helios {

// ============================================================================
// DTYPE REGISTRY
// ============================================================================

DTypeRegistry& DTypeRegistry::instance() {
    static DTypeRegistry instance;
    return instance;
}

DTypeRegistry::DTypeRegistry() {
    register_builtins();
}

void DTypeRegistry::register_builtins() {
    // Usar register_dtype directamente para evitar recursión via instance()
    DTypeInfo info;
    
    // FP32
    info = DTypeInfo{};
    info.name = "fp32";
    info.element_bits = 32;
    info.is_floating = true;
    info.is_signed = true;
    register_dtype(info);
    
    // FP16
    info = DTypeInfo{};
    info.name = "fp16";
    info.element_bits = 16;
    info.is_floating = true;
    info.is_signed = true;
    register_dtype(info);
    
    // BF16
    info = DTypeInfo{};
    info.name = "bf16";
    info.element_bits = 16;
    info.is_floating = true;
    info.is_signed = true;
    register_dtype(info);
    
    // FP8
    info = DTypeInfo{};
    info.name = "fp8";
    info.element_bits = 8;
    info.is_floating = true;
    info.is_signed = true;
    register_dtype(info);
    
    // INT8
    info = DTypeInfo{};
    info.name = "int8";
    info.element_bits = 8;
    info.is_signed = true;
    register_dtype(info);
    
    // INT32
    info = DTypeInfo{};
    info.name = "int32";
    info.element_bits = 32;
    info.is_signed = true;
    register_dtype(info);
    
    // HQ4K - HELIOS Quant 4-bit
    info = DTypeInfo{};
    info.name = "hq4k";
    info.block_elements = 256;
    info.block_bytes = 256;
    info.is_quantized = true;
    info.calc_size = [](size_t n) -> size_t {
        size_t num_blocks = (n + 255) / 256;
        return num_blocks * 256;
    };
    register_dtype(info);
    
    // HQ5K - HELIOS Quant 5-bit
    info = DTypeInfo{};
    info.name = "hq5k";
    info.block_elements = 256;
    info.block_bytes = 288;
    info.is_quantized = true;
    info.calc_size = [](size_t n) -> size_t {
        size_t num_blocks = (n + 255) / 256;
        return num_blocks * 288;
    };
    register_dtype(info);
    
    // HQ4.1K - HELIOS Quant 4-bit compact header
    info = DTypeInfo{};
    info.name = "hq41k";
    info.block_elements = 256;
    info.block_bytes = 168;
    info.is_quantized = true;
    info.calc_size = [](size_t n) -> size_t {
        size_t num_blocks = (n + 255) / 256;
        return num_blocks * 168;
    };
    register_dtype(info);
    
    // HQ5.1K - HELIOS Quant 5-bit compact header
    info = DTypeInfo{};
    info.name = "hq51k";
    info.block_elements = 256;
    info.block_bytes = 200;
    info.is_quantized = true;
    info.calc_size = [](size_t n) -> size_t {
        size_t num_blocks = (n + 255) / 256;
        return num_blocks * 200;
    };
    register_dtype(info);
}

DTypeID DTypeRegistry::register_dtype(const DTypeInfo& info) {
    if (info.name.empty()) {
        throw std::runtime_error("DType name cannot be empty");
    }
    
    // Check for duplicate name
    if (by_name_.count(info.name)) {
        throw std::runtime_error("DType already registered: " + info.name);
    }
    
    DTypeID id = next_id_++;
    
    DTypeInfo stored = info;
    stored.id = id;
    
    by_id_[id] = stored;
    by_name_[info.name] = id;
    
    return id;
}

const DTypeInfo* DTypeRegistry::get(DTypeID id) const {
    auto it = by_id_.find(id);
    return (it != by_id_.end()) ? &it->second : nullptr;
}

const DTypeInfo* DTypeRegistry::get(const std::string& name) const {
    auto it = by_name_.find(name);
    if (it == by_name_.end()) return nullptr;
    return get(it->second);
}

DTypeID DTypeRegistry::get_id(const std::string& name) const {
    auto it = by_name_.find(name);
    return (it != by_name_.end()) ? it->second : DTYPE_INVALID;
}

} // namespace helios
