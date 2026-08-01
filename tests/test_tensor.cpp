// tests/test_tensor.cpp
// HELIOS ENGINE - TensorRegistry Test (v2 - extensible dtypes)
// ============================================================

#include "tensor.hpp"
#include "dtype.hpp"
#include <iostream>
#include <cassert>

using namespace helios;

void test_dtype_registry() {
    std::cout << "Test: DType registry... ";
    
    // Builtins should be registered
    assert(dtype::FP32() != DTYPE_INVALID);
    assert(dtype::FP16() != DTYPE_INVALID);
    
    // Lookup by name
    auto& reg = DTypeRegistry::instance();
    assert(reg.get_id("fp32") == dtype::FP32());
    
    // Get info
    auto* fp32_info = reg.get(dtype::FP32());
    assert(fp32_info != nullptr);
    assert(fp32_info->element_bits == 32);
    assert(fp32_info->is_floating);
    
    
    std::cout << "PASSED" << std::endl;
}

void test_dtype_extensibility() {
    std::cout << "Test: DType extensibility... ";
    
    // Register a custom dtype
    DTypeID my_dtype = DTypeRegistry::Builder("my_custom_q3")
        .block(256, 192)  // 3-bit = 192 bytes for 256 elements
        .quantized()
        .build();
    
    assert(my_dtype != DTYPE_INVALID);
    
    // Should be findable
    auto& reg = DTypeRegistry::instance();
    assert(reg.get_id("my_custom_q3") == my_dtype);
    
    auto* info = reg.get(my_dtype);
    assert(info != nullptr);
    assert(info->is_quantized);
    assert(info->block_bytes == 192);
    
    std::cout << "PASSED" << std::endl;
}

void test_hq_sizes() {
    std::cout << "Test: HQ size calculations... ";
    
    
    
    // FP32: simple
    assert(dtype_size(dtype::FP32(), 100) == 400);
    
    // FP16: simple
    assert(dtype_size(dtype::FP16(), 100) == 200);
    
    std::cout << "PASSED" << std::endl;
}

void test_basic_registration() {
    std::cout << "Test: Basic registration... ";
    
    TensorRegistry registry;
    
    // Allocate on GPU
    float* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    
    // Create TensorInfo
    TensorInfo info;
    info.ptr = d_data;
    info.shape = {32, 32};
    info.dtype = dtype::FP32();
    info.size_bytes = 1024 * sizeof(float);
    info.owns_memory = true;
    
    // Register
    registry.register_tensor("test_tensor", info);
    
    // Verify
    assert(registry.exists("test_tensor"));
    assert(registry.count() == 1);
    
    TensorInfo* retrieved = registry.get("test_tensor");
    assert(retrieved != nullptr);
    assert(retrieved->ndim() == 2);
    assert(retrieved->shape[0] == 32);
    assert(retrieved->shape[1] == 32);
    assert(retrieved->numel() == 1024);
    
    std::cout << "PASSED" << std::endl;
}

void test_allocate_and_register() {
    std::cout << "Test: Allocate and register... ";
    
    TensorRegistry registry;
    
    // Allocate via registry
    void* ptr = registry.allocate_and_register(
        "embedding",
        {1024, 512},  // 1024 tokens, 512 dim
        dtype::FP16(),
        true  // zero init
    );
    
    assert(ptr != nullptr);
    assert(registry.exists("embedding"));
    
    TensorInfo* info = registry.get("embedding");
    assert(info->ndim() == 2);
    assert(info->shape[0] == 1024);
    assert(info->shape[1] == 512);
    assert(info->numel() == 1024 * 512);
    assert(info->size_bytes == 1024 * 512 * 2);  // FP16 = 2 bytes
    assert(info->owns_memory == true);
    
    std::cout << "PASSED" << std::endl;
}

void test_quantized_tensor() {
    std::cout << "Test: Quantized tensor... ";
    
    TensorRegistry registry;
    
    void* ptr = registry.allocate_and_register(
        "weights_q",
        {4096, 4096},
        dtype::HQ41K()
    );
    
    assert(ptr != nullptr);
    
    TensorInfo* info = registry.get("weights_q");
    assert(info->is_quantized());
    
    // 4096*4096 = 16M elementos -> 65536 superbloques de 168 B (HQ4.1K)
    size_t expected = dtype_size(dtype::HQ41K(), 4096 * 4096);
    assert(info->size_bytes == expected);
    
    std::cout << "PASSED" << std::endl;
}

void test_remove_and_clear() {
    std::cout << "Test: Remove and clear... ";
    
    TensorRegistry registry;
    
    registry.allocate_and_register("a", {100}, dtype::FP32());
    registry.allocate_and_register("b", {200}, dtype::FP32());
    registry.allocate_and_register("c", {300}, dtype::FP32());
    
    assert(registry.count() == 3);
    
    // Remove one
    registry.remove("b");
    assert(registry.count() == 2);
    assert(!registry.exists("b"));
    assert(registry.exists("a"));
    assert(registry.exists("c"));
    
    // Clear all
    registry.clear();
    assert(registry.count() == 0);
    assert(registry.total_bytes() == 0);
    
    std::cout << "PASSED" << std::endl;
}

void test_print_summary() {
    std::cout << "Test: Print summary... " << std::endl;
    
    TensorRegistry registry;
    
    registry.allocate_and_register("text.embedding", {32000, 2048}, dtype::FP16());
    registry.allocate_and_register("text.layer0.attn.q", {2048, 2048}, dtype::HQ41K());
    registry.allocate_and_register("text.layer0.attn.k", {2048, 256}, dtype::HQ41K());
    registry.allocate_and_register("scratch_buffer", {1024, 1024}, dtype::FP32(), true);
    
    registry.print_summary();
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "==================================" << std::endl;
    std::cout << "HELIOS ENGINE - TensorRegistry Test (v2)" << std::endl;
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
    std::cout << "VRAM: " << (prop.totalGlobalMem / (1024*1024*1024.0)) << " GB" << std::endl;
    std::cout << std::endl;
    
    // Run tests
    test_dtype_registry();
    test_dtype_extensibility();
    test_hq_sizes();
    test_basic_registration();
    test_allocate_and_register();
    test_quantized_tensor();
    test_remove_and_clear();
    test_print_summary();
    
    std::cout << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "ALL TESTS PASSED ✓" << std::endl;
    std::cout << "==================================" << std::endl;
    
    return 0;
}
