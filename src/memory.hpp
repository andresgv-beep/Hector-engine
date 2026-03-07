// memory.hpp
// HELIOS ENGINE - Memory Pool
// ============================
// Pool de memoria GPU reutilizable para activaciones.
//
// Diseño:
//   - Tamaño configurable en runtime
//   - Arena allocator (fast alloc, bulk free)
//   - Sin fragmentación para uso típico de inference
//   - Query de VRAM disponible
//

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <string>

namespace helios {

// ============================================================================
// VRAM QUERY (runtime, no hardcode)
// ============================================================================

struct VRAMInfo {
    size_t total_bytes;
    size_t free_bytes;
    size_t used_bytes;
    
    double total_gb() const { return total_bytes / (1024.0 * 1024.0 * 1024.0); }
    double free_gb() const { return free_bytes / (1024.0 * 1024.0 * 1024.0); }
    double used_gb() const { return used_bytes / (1024.0 * 1024.0 * 1024.0); }
};

// Query current VRAM state (calls cudaMemGetInfo)
VRAMInfo query_vram();

// ============================================================================
// MEMORY POOL CONFIG
// ============================================================================

struct MemoryPoolConfig {
    // Size of pool (0 = auto-detect based on available VRAM)
    size_t pool_size_bytes = 0;
    
    // If auto-detect, what fraction of free VRAM to use (0.0-1.0)
    float auto_fraction = 0.8f;
    
    // Minimum pool size (even with auto-detect)
    size_t min_size_bytes = 256 * 1024 * 1024;  // 256 MB
    
    // Maximum pool size (even with auto-detect)
    size_t max_size_bytes = 0;  // 0 = no limit
    
    // Alignment for allocations
    size_t alignment = 256;  // Good for most CUDA ops
    
    // Zero-initialize on creation
    bool zero_init = false;
    
    // Name for debugging
    std::string name = "default";
};

// ============================================================================
// MEMORY POOL
// ============================================================================

class MemoryPool {
public:
    // Create pool with config
    explicit MemoryPool(const MemoryPoolConfig& config = MemoryPoolConfig{});
    
    // Create pool with simple size (convenience)
    explicit MemoryPool(size_t size_bytes);
    
    ~MemoryPool();
    
    // No copy
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
    // Move OK
    MemoryPool(MemoryPool&& other) noexcept;
    MemoryPool& operator=(MemoryPool&& other) noexcept;
    
    // ========================================
    // ALLOCATION
    // ========================================
    
    // Allocate from pool (returns nullptr if no space)
    void* allocate(size_t bytes);
    
    // Allocate with specific alignment
    void* allocate_aligned(size_t bytes, size_t alignment);
    
    // Reset pool (all allocations invalidated, but memory kept)
    // This is the fast path for between-inference cleanup
    void reset();
    
    // ========================================
    // STATE
    // ========================================
    
    // Pool capacity
    size_t capacity() const { return capacity_; }
    
    // Currently allocated
    size_t allocated() const { return offset_; }
    
    // Available
    size_t available() const { return capacity_ - offset_; }
    
    // Utilization (0.0 - 1.0)
    float utilization() const { 
        return capacity_ > 0 ? static_cast<float>(offset_) / capacity_ : 0.0f; 
    }
    
    // Is valid (successfully initialized)
    bool valid() const { return base_ptr_ != nullptr; }
    
    // Base pointer (for advanced use)
    void* base_ptr() { return base_ptr_; }
    const void* base_ptr() const { return base_ptr_; }
    
    // Config used
    const MemoryPoolConfig& config() const { return config_; }
    
    // ========================================
    // DEBUG
    // ========================================
    
    void print_stats() const;
    
private:
    friend class ScratchScope;  // Needs access to offset_
    
    MemoryPoolConfig config_;
    void* base_ptr_;
    size_t capacity_;
    size_t offset_;  // Current allocation offset (arena style)
    
    // Initialize pool
    void init(size_t size_bytes);
};

// ============================================================================
// SCRATCH ALLOCATOR (temporary per-operation memory)
// ============================================================================

// RAII wrapper for temporary allocations that auto-reset
class ScratchScope {
public:
    explicit ScratchScope(MemoryPool& pool);
    ~ScratchScope();
    
    // Allocate within this scope
    void* allocate(size_t bytes);
    
    // No copy/move (stack-based RAII)
    ScratchScope(const ScratchScope&) = delete;
    ScratchScope& operator=(const ScratchScope&) = delete;
    
private:
    MemoryPool& pool_;
    size_t saved_offset_;
};

} // namespace helios
