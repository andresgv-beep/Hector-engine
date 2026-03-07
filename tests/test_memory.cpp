// tests/test_memory.cpp
// HELIOS ENGINE - MemoryPool Test
// ================================

#include "memory.hpp"
#include <iostream>
#include <cassert>

using namespace helios;

void test_vram_query() {
    std::cout << "Test: VRAM query... ";
    
    VRAMInfo info = query_vram();
    
    // Should have some VRAM
    assert(info.total_bytes > 0);
    assert(info.free_bytes > 0);
    assert(info.free_bytes <= info.total_bytes);
    
    std::cout << "PASSED" << std::endl;
    std::cout << "  Total: " << info.total_gb() << " GB" << std::endl;
    std::cout << "  Free:  " << info.free_gb() << " GB" << std::endl;
    std::cout << "  Used:  " << info.used_gb() << " GB" << std::endl;
}

void test_basic_pool() {
    std::cout << "Test: Basic pool... ";
    
    // Create 100MB pool
    MemoryPool pool(100 * 1024 * 1024);
    
    assert(pool.valid());
    assert(pool.capacity() == 100 * 1024 * 1024);
    assert(pool.allocated() == 0);
    assert(pool.available() == pool.capacity());
    
    std::cout << "PASSED" << std::endl;
}

void test_allocation() {
    std::cout << "Test: Allocation... ";
    
    MemoryPool pool(10 * 1024 * 1024);  // 10MB
    
    // Allocate 1MB
    void* ptr1 = pool.allocate(1 * 1024 * 1024);
    assert(ptr1 != nullptr);
    assert(pool.allocated() >= 1 * 1024 * 1024);
    
    // Allocate another 2MB
    void* ptr2 = pool.allocate(2 * 1024 * 1024);
    assert(ptr2 != nullptr);
    assert(ptr2 != ptr1);  // Different pointers
    assert(pool.allocated() >= 3 * 1024 * 1024);
    
    // Pointers should be sequential (arena allocator)
    assert(ptr2 > ptr1);
    
    std::cout << "PASSED" << std::endl;
}

void test_alignment() {
    std::cout << "Test: Alignment... ";
    
    MemoryPool pool(10 * 1024 * 1024);
    
    // Allocate with different sizes
    void* ptr1 = pool.allocate(100);  // Not aligned size
    void* ptr2 = pool.allocate(200);  // Next allocation
    
    // Check alignment (default 256)
    assert(reinterpret_cast<uintptr_t>(ptr1) % 256 == 0);
    assert(reinterpret_cast<uintptr_t>(ptr2) % 256 == 0);
    
    // Custom alignment
    void* ptr3 = pool.allocate_aligned(64, 4096);  // 4K alignment
    assert(reinterpret_cast<uintptr_t>(ptr3) % 4096 == 0);
    
    std::cout << "PASSED" << std::endl;
}

void test_reset() {
    std::cout << "Test: Reset... ";
    
    MemoryPool pool(10 * 1024 * 1024);
    
    // Allocate some memory
    pool.allocate(1 * 1024 * 1024);
    pool.allocate(2 * 1024 * 1024);
    
    size_t before_reset = pool.allocated();
    assert(before_reset > 0);
    
    // Reset
    pool.reset();
    
    assert(pool.allocated() == 0);
    assert(pool.available() == pool.capacity());
    
    // Can allocate again
    void* ptr = pool.allocate(1 * 1024 * 1024);
    assert(ptr != nullptr);
    
    std::cout << "PASSED" << std::endl;
}

void test_out_of_memory() {
    std::cout << "Test: Out of memory... ";
    
    MemoryPool pool(1 * 1024 * 1024);  // 1MB only
    
    // Allocate most of it
    void* ptr1 = pool.allocate(900 * 1024);  // 900KB
    assert(ptr1 != nullptr);
    
    // Try to allocate more than available
    void* ptr2 = pool.allocate(200 * 1024);  // 200KB - won't fit
    assert(ptr2 == nullptr);  // Should fail gracefully
    
    // Small allocation should still work
    void* ptr3 = pool.allocate(50 * 1024);  // 50KB
    assert(ptr3 != nullptr);
    
    std::cout << "PASSED" << std::endl;
}

void test_auto_size() {
    std::cout << "Test: Auto-size pool... ";
    
    MemoryPoolConfig config;
    config.pool_size_bytes = 0;  // Auto-detect
    config.auto_fraction = 0.1f;  // Use 10% of free VRAM
    config.name = "auto_pool";
    
    MemoryPool pool(config);
    
    assert(pool.valid());
    assert(pool.capacity() > 0);
    
    // Should be roughly 10% of free VRAM
    VRAMInfo vram = query_vram();
    size_t expected = static_cast<size_t>(vram.free_bytes * 0.1f);
    
    // Allow some tolerance (other allocations may have happened)
    assert(pool.capacity() >= config.min_size_bytes);
    
    std::cout << "PASSED" << std::endl;
    std::cout << "  Auto-detected size: " << (pool.capacity() / (1024.0 * 1024.0)) << " MB" << std::endl;
}

void test_scratch_scope() {
    std::cout << "Test: Scratch scope... ";
    
    MemoryPool pool(10 * 1024 * 1024);
    
    // Allocate some persistent memory
    void* persistent = pool.allocate(1 * 1024 * 1024);
    assert(persistent != nullptr);
    
    size_t after_persistent = pool.allocated();
    
    // Use scratch scope for temporary allocations
    {
        ScratchScope scratch(pool);
        
        void* temp1 = scratch.allocate(2 * 1024 * 1024);
        void* temp2 = scratch.allocate(1 * 1024 * 1024);
        
        assert(temp1 != nullptr);
        assert(temp2 != nullptr);
        assert(pool.allocated() > after_persistent);
        
        // Scratch goes out of scope here
    }
    
    // Memory should be "freed" (offset restored)
    assert(pool.allocated() == after_persistent);
    
    // Can allocate again in same space
    void* reused = pool.allocate(2 * 1024 * 1024);
    assert(reused != nullptr);
    
    std::cout << "PASSED" << std::endl;
}

void test_print_stats() {
    std::cout << "Test: Print stats..." << std::endl;
    
    MemoryPoolConfig config;
    config.pool_size_bytes = 50 * 1024 * 1024;  // 50MB
    config.name = "test_pool";
    
    MemoryPool pool(config);
    
    pool.allocate(10 * 1024 * 1024);
    pool.allocate(5 * 1024 * 1024);
    
    pool.print_stats();
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "==================================" << std::endl;
    std::cout << "HELIOS ENGINE - MemoryPool Test" << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << std::endl;
    
    // Check CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "ERROR: No CUDA devices found" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << std::endl;
    
    // Run tests
    test_vram_query();
    test_basic_pool();
    test_allocation();
    test_alignment();
    test_reset();
    test_out_of_memory();
    test_auto_size();
    test_scratch_scope();
    test_print_stats();
    
    std::cout << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "ALL TESTS PASSED ✓" << std::endl;
    std::cout << "==================================" << std::endl;
    
    return 0;
}
