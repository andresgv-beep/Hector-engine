// tensor.cpp
// HELIOS ENGINE - TensorRegistry Implementation
// ==============================================

#include "tensor.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <sys/mman.h>

namespace helios {

namespace {

void release_tensor_memory(TensorInfo& info) {
    if (!info.owns_memory) return;
    void* allocation = info.allocation_ptr ? info.allocation_ptr : info.ptr;
    if (!allocation) return;
    if (info.file_mapped) munmap(allocation, info.allocation_size);
    else if (info.host_mapped) cudaFreeHost(allocation);
    else cudaFree(allocation);
    info.ptr = nullptr;
    info.allocation_ptr = nullptr;
    info.allocation_size = 0;
}

} // namespace

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

TensorRegistry::TensorRegistry()
    : total_bytes_(0)
    , owned_bytes_(0)
{
}

TensorRegistry::~TensorRegistry() {
    clear();
}

// ============================================================================
// REGISTRATION
// ============================================================================

void TensorRegistry::register_tensor(const std::string& name, TensorInfo info) {
    // Check if already exists
    if (exists(name)) {
        throw std::runtime_error("Tensor already registered: " + name);
    }
    
    // Update stats
    total_bytes_ += info.size_bytes;
    if (info.owns_memory) {
        owned_bytes_ += info.allocation_size
            ? info.allocation_size : info.size_bytes;
    }
    
    // Store
    tensors_[name] = std::move(info);
}

void* TensorRegistry::allocate_and_register(
    const std::string& name,
    const std::vector<uint32_t>& shape,
    DTypeID dtype,
    bool zero_init
) {
    if (shape.empty()) {
        throw std::runtime_error("Invalid shape: empty");
    }
    
    // Calculate size
    size_t numel = 1;
    for (auto d : shape) {
        numel *= d;
    }
    
    size_t size_bytes = dtype_size(dtype, numel);
    if (size_bytes == 0) {
        throw std::runtime_error("Cannot calculate size for dtype: " + 
                                  std::string(dtype_name(dtype)));
    }
    
    // Allocate GPU memory
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "cudaMalloc failed: " + std::string(cudaGetErrorString(err))
        );
    }
    
    // Zero initialize if requested
    if (zero_init) {
        cudaMemset(ptr, 0, size_bytes);
    }
    
    // Build TensorInfo
    TensorInfo info;
    info.ptr = ptr;
    info.shape = shape;
    info.dtype = dtype;
    info.size_bytes = size_bytes;
    info.owns_memory = true;
    info.allocation_ptr = ptr;
    
    // Register
    register_tensor(name, std::move(info));
    
    return ptr;
}

void TensorRegistry::register_external(
    const std::string& name,
    void* ptr,
    const std::vector<uint32_t>& shape,
    DTypeID dtype
) {
    if (!ptr) {
        throw std::runtime_error("Cannot register null pointer: " + name);
    }
    
    // Calculate size
    size_t numel = 1;
    for (auto d : shape) {
        numel *= d;
    }
    size_t size_bytes = dtype_size(dtype, numel);
    
    // Build TensorInfo (does NOT own memory)
    TensorInfo info;
    info.ptr = ptr;
    info.shape = shape;
    info.dtype = dtype;
    info.size_bytes = size_bytes;
    info.owns_memory = false;  // External memory
    
    // Register
    register_tensor(name, std::move(info));
}

// ============================================================================
// ACCESS
// ============================================================================

TensorInfo* TensorRegistry::get(const std::string& name) {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? &it->second : nullptr;
}

const TensorInfo* TensorRegistry::get(const std::string& name) const {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? &it->second : nullptr;
}

bool TensorRegistry::exists(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

TensorInfo& TensorRegistry::at(const std::string& name) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

const TensorInfo& TensorRegistry::at(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

// ============================================================================
// MANAGEMENT
// ============================================================================

void TensorRegistry::remove(const std::string& name) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return;  // Not found, nothing to do
    }
    
    TensorInfo& info = it->second;
    
    // Update stats
    total_bytes_ -= info.size_bytes;
    if (info.owns_memory) {
        owned_bytes_ -= info.allocation_size
            ? info.allocation_size : info.size_bytes;
        // Free GPU memory
        release_tensor_memory(info);
    }
    
    tensors_.erase(it);
}

void TensorRegistry::clear() {
    // Free all owned memory
    for (auto& [name, info] : tensors_) {
        release_tensor_memory(info);
    }
    
    tensors_.clear();
    total_bytes_ = 0;
    owned_bytes_ = 0;
}

std::vector<std::string> TensorRegistry::names() const {
    std::vector<std::string> result;
    result.reserve(tensors_.size());
    for (const auto& [name, _] : tensors_) {
        result.push_back(name);
    }
    std::sort(result.begin(), result.end());
    return result;
}

// ============================================================================
// STATS
// ============================================================================

void TensorRegistry::print_summary() const {
    std::cout << "=== TensorRegistry ===" << std::endl;
    std::cout << "Count: " << count() << " tensors" << std::endl;
    std::cout << "Total: " << (total_bytes_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Owned: " << (owned_bytes_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << std::endl;
    
    // List tensors
    auto sorted_names = names();
    for (const auto& name : sorted_names) {
        const TensorInfo& info = tensors_.at(name);
        
        std::cout << "  " << name << ": [";
        for (size_t i = 0; i < info.shape.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << info.shape[i];
        }
        std::cout << "] " << info.dtype_str();
        std::cout << " (" << (info.size_bytes / 1024.0) << " KB)";
        if (info.owns_memory) std::cout << " [owned]";
        std::cout << std::endl;
    }
}

} // namespace helios
