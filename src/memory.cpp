// memory.cpp
// HELIOS ENGINE - Memory Pool Implementation
// ==========================================

#include "memory.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace helios {

// ============================================================================
// VRAM QUERY
// ============================================================================

VRAMInfo query_vram() {
    VRAMInfo info{};
    
    cudaError_t err = cudaMemGetInfo(&info.free_bytes, &info.total_bytes);
    if (err != cudaSuccess) {
        // Return zeros on error
        return info;
    }
    
    info.used_bytes = info.total_bytes - info.free_bytes;
    return info;
}

// ============================================================================
// MEMORY POOL
// ============================================================================

MemoryPool::MemoryPool(const MemoryPoolConfig& config)
    : config_(config)
    , base_ptr_(nullptr)
    , capacity_(0)
    , offset_(0)
{
    size_t size_bytes = config.pool_size_bytes;
    
    // Auto-detect size if not specified
    if (size_bytes == 0) {
        VRAMInfo vram = query_vram();
        size_bytes = static_cast<size_t>(vram.free_bytes * config.auto_fraction);
    }
    
    // Apply min/max constraints
    size_bytes = std::max(size_bytes, config.min_size_bytes);
    if (config.max_size_bytes > 0) {
        size_bytes = std::min(size_bytes, config.max_size_bytes);
    }
    
    init(size_bytes);
}

MemoryPool::MemoryPool(size_t size_bytes)
    : config_()
    , base_ptr_(nullptr)
    , capacity_(0)
    , offset_(0)
{
    config_.pool_size_bytes = size_bytes;
    init(size_bytes);
}

MemoryPool::~MemoryPool() {
    if (base_ptr_) {
        cudaFree(base_ptr_);
        base_ptr_ = nullptr;
    }
}

MemoryPool::MemoryPool(MemoryPool&& other) noexcept
    : config_(std::move(other.config_))
    , base_ptr_(other.base_ptr_)
    , capacity_(other.capacity_)
    , offset_(other.offset_)
{
    other.base_ptr_ = nullptr;
    other.capacity_ = 0;
    other.offset_ = 0;
}

MemoryPool& MemoryPool::operator=(MemoryPool&& other) noexcept {
    if (this != &other) {
        // Free our memory
        if (base_ptr_) {
            cudaFree(base_ptr_);
        }
        
        // Take other's
        config_ = std::move(other.config_);
        base_ptr_ = other.base_ptr_;
        capacity_ = other.capacity_;
        offset_ = other.offset_;
        
        // Clear other
        other.base_ptr_ = nullptr;
        other.capacity_ = 0;
        other.offset_ = 0;
    }
    return *this;
}

void MemoryPool::init(size_t size_bytes) {
    if (size_bytes == 0) {
        return;
    }
    
    cudaError_t err = cudaMalloc(&base_ptr_, size_bytes);
    if (err != cudaSuccess) {
        base_ptr_ = nullptr;
        capacity_ = 0;
        throw std::runtime_error(
            "MemoryPool cudaMalloc failed: " + std::string(cudaGetErrorString(err))
        );
    }
    
    capacity_ = size_bytes;
    offset_ = 0;
    
    if (config_.zero_init) {
        cudaMemset(base_ptr_, 0, capacity_);
    }
}

// ============================================================================
// ALLOCATION
// ============================================================================

void* MemoryPool::allocate(size_t bytes) {
    return allocate_aligned(bytes, config_.alignment);
}

void* MemoryPool::allocate_aligned(size_t bytes, size_t alignment) {
    if (!base_ptr_ || bytes == 0) {
        return nullptr;
    }
    
    // Align current offset
    size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);
    
    // Check if we have space
    if (aligned_offset + bytes > capacity_) {
        return nullptr;  // Out of memory
    }
    
    // Calculate pointer
    void* ptr = static_cast<char*>(base_ptr_) + aligned_offset;
    
    // Update offset
    offset_ = aligned_offset + bytes;
    
    return ptr;
}

void MemoryPool::reset() {
    offset_ = 0;
    // Memory is NOT zeroed - just reset the offset
    // This is intentional for speed
}

// ============================================================================
// DEBUG
// ============================================================================

void MemoryPool::print_stats() const {
    std::cout << "=== MemoryPool [" << config_.name << "] ===" << std::endl;
    std::cout << "  Capacity:  " << (capacity_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Allocated: " << (offset_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Available: " << (available() / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Utilization: " << (utilization() * 100.0f) << "%" << std::endl;
}

// ============================================================================
// SCRATCH SCOPE
// ============================================================================

ScratchScope::ScratchScope(MemoryPool& pool)
    : pool_(pool)
    , saved_offset_(pool.allocated())
{
}

ScratchScope::~ScratchScope() {
    // Restore offset to saved position
    // This effectively "frees" all allocations made in this scope
    pool_.offset_ = saved_offset_;
}

void* ScratchScope::allocate(size_t bytes) {
    return pool_.allocate(bytes);
}

} // namespace helios
