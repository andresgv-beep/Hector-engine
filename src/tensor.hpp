// tensor.hpp
// HELIOS ENGINE - TensorRegistry
// ================================
// Base del sistema de tensores. Todo pasa por aquí.
//
// Diseño:
//   - TensorInfo: metadata de un tensor
//   - TensorRegistry: diccionario name -> TensorInfo
//   - No posee memoria, solo registra punteros
//   - DType extensible via DTypeRegistry
//

#pragma once

#include "dtype.hpp"
#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <vector>

namespace helios {

// ============================================================================
// TENSOR INFO
// ============================================================================

struct TensorInfo {
    void* ptr = nullptr;        // Device-visible pointer
    std::vector<uint32_t> shape; // Dimensions (variable size)
    DTypeID dtype = DTYPE_INVALID; // Data type (extensible)
    size_t size_bytes = 0;      // Total size in bytes
    bool owns_memory = false;   // If true, registry releases allocation_ptr/ptr
    bool host_mapped = false;   // Owned allocation came from cudaHostAlloc
    bool file_mapped = false;   // Owned allocation came from mmap(MAP_PRIVATE)
    void* allocation_ptr = nullptr; // Original host/device allocation handle
    size_t allocation_size = 0; // Full owned allocation/mapping when it differs
    
    // Computed properties
    size_t numel() const {
        size_t n = 1;
        for (auto d : shape) {
            n *= d;
        }
        return n;
    }
    
    uint8_t ndim() const {
        return static_cast<uint8_t>(shape.size());
    }
    
    bool is_quantized() const {
        return dtype_is_quantized(dtype);
    }
    
    const char* dtype_str() const {
        return dtype_name(dtype);
    }
};

// ============================================================================
// TENSOR REGISTRY
// ============================================================================

class TensorRegistry {
public:
    TensorRegistry();
    ~TensorRegistry();
    
    // Prevent copying (owns GPU memory references)
    TensorRegistry(const TensorRegistry&) = delete;
    TensorRegistry& operator=(const TensorRegistry&) = delete;
    
    // Move OK
    TensorRegistry(TensorRegistry&&) = default;
    TensorRegistry& operator=(TensorRegistry&&) = default;
    
    // ========================================
    // REGISTRATION
    // ========================================
    
    // Register tensor with existing device pointer
    // If owns_memory=true, registry will cudaFree on destruction
    void register_tensor(const std::string& name, TensorInfo info);
    
    // Register external GPU memory (not owned by registry)
    void register_external(
        const std::string& name,
        void* ptr,
        const std::vector<uint32_t>& shape,
        DTypeID dtype
    );
    
    // Allocate and register (convenience)
    // Returns device pointer
    void* allocate_and_register(
        const std::string& name,
        const std::vector<uint32_t>& shape,
        DTypeID dtype,
        bool zero_init = false
    );
    
    // ========================================
    // ACCESS
    // ========================================
    
    // Get tensor info (nullptr if not found)
    TensorInfo* get(const std::string& name);
    const TensorInfo* get(const std::string& name) const;
    
    // Check existence
    bool exists(const std::string& name) const;
    
    // Get or throw
    TensorInfo& at(const std::string& name);
    const TensorInfo& at(const std::string& name) const;
    
    // ========================================
    // MANAGEMENT
    // ========================================
    
    // Remove tensor (frees memory if owns_memory)
    void remove(const std::string& name);
    
    // Clear all tensors
    void clear();
    
    // List all tensor names
    std::vector<std::string> names() const;
    
    // ========================================
    // STATS
    // ========================================
    
    size_t count() const { return tensors_.size(); }
    size_t total_bytes() const { return total_bytes_; }
    size_t owned_bytes() const { return owned_bytes_; }
    
    // Print summary
    void print_summary() const;
    
private:
    std::unordered_map<std::string, TensorInfo> tensors_;
    size_t total_bytes_;    // All registered tensors
    size_t owned_bytes_;    // Only tensors with owns_memory=true
};

} // namespace helios
