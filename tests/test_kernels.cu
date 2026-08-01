// tests/test_kernels.cpp
// ============================================================================
// HELIOS ENGINE - Kernel Test
// ============================================================================

#include "engine.hpp"
#include "kernels.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <random>
#include <sys/mman.h>
#include <unistd.h>

using namespace helios;

// Helper: allocate and fill tensor with random data
void fill_random_fp16(half* ptr, size_t numel, float min_val, float max_val) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    std::vector<float> host(numel);
    for (size_t i = 0; i < numel; i++) {
        host[i] = dist(rng);
    }
    
    // Convert to half and copy
    std::vector<half> host_half(numel);
    for (size_t i = 0; i < numel; i++) {
        host_half[i] = __float2half(host[i]);
    }
    
    cudaMemcpy(ptr, host_half.data(), numel * sizeof(half), cudaMemcpyHostToDevice);
}

// Helper: copy GPU tensor to host
std::vector<float> to_host(half* ptr, size_t numel) {
    std::vector<half> host_half(numel);
    cudaMemcpy(host_half.data(), ptr, numel * sizeof(half), cudaMemcpyDeviceToHost);
    
    std::vector<float> host(numel);
    for (size_t i = 0; i < numel; i++) {
        host[i] = __half2float(host_half[i]);
    }
    return host;
}

void test_add_kernel() {
    std::cout << "Test: ADD kernel... ";
    
    Engine engine;
    kernels::register_all_kernels(engine);
    
    size_t numel = 1024;
    
    engine.tensors().allocate_and_register("a", {(uint32_t)numel}, dtype::FP16());
    engine.tensors().allocate_and_register("b", {(uint32_t)numel}, dtype::FP16());
    engine.tensors().allocate_and_register("c", {(uint32_t)numel}, dtype::FP16());
    
    auto* a = engine.tensors().get("a");
    auto* b = engine.tensors().get("b");
    auto* c = engine.tensors().get("c");
    
    // Fill with known values
    std::vector<half> a_host(numel), b_host(numel);
    for (size_t i = 0; i < numel; i++) {
        a_host[i] = __float2half(1.0f);
        b_host[i] = __float2half(2.0f);
    }
    cudaMemcpy(a->ptr, a_host.data(), numel * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(b->ptr, b_host.data(), numel * sizeof(half), cudaMemcpyHostToDevice);
    
    // Execute
    CommandBuffer cb;
    cb.add_add("c", "a", "b");
    engine.execute(cb);
    engine.sync();
    
    // Verify
    auto c_result = to_host(static_cast<half*>(c->ptr), numel);
    for (size_t i = 0; i < numel; i++) {
        assert(std::abs(c_result[i] - 3.0f) < 0.01f);
    }
    
    std::cout << "PASSED" << std::endl;
}

void test_silu_kernel() {
    std::cout << "Test: SILU kernel... ";
    
    Engine engine;
    kernels::register_all_kernels(engine);
    
    size_t numel = 256;
    
    engine.tensors().allocate_and_register("x", {(uint32_t)numel}, dtype::FP16());
    engine.tensors().allocate_and_register("y", {(uint32_t)numel}, dtype::FP16());
    
    auto* x = engine.tensors().get("x");
    auto* y = engine.tensors().get("y");
    
    // Fill with test values
    std::vector<half> x_host(numel);
    for (size_t i = 0; i < numel; i++) {
        x_host[i] = __float2half((float)i / numel * 4.0f - 2.0f);  // Range [-2, 2]
    }
    cudaMemcpy(x->ptr, x_host.data(), numel * sizeof(half), cudaMemcpyHostToDevice);
    
    // Execute
    CommandBuffer cb;
    cb.add_silu("y", "x");
    engine.execute(cb);
    engine.sync();
    
    // Verify (SiLU = x * sigmoid(x))
    auto y_result = to_host(static_cast<half*>(y->ptr), numel);
    for (size_t i = 0; i < numel; i++) {
        float x_val = __half2float(x_host[i]);
        float expected = x_val / (1.0f + std::exp(-x_val));
        assert(std::abs(y_result[i] - expected) < 0.05f);
    }
    
    std::cout << "PASSED" << std::endl;
}

void test_rmsnorm_kernel() {
    std::cout << "Test: RMSNORM kernel... ";
    
    Engine engine;
    kernels::register_all_kernels(engine);
    
    int batch = 4;
    int dim = 64;
    
    engine.tensors().allocate_and_register("x", {(uint32_t)batch, (uint32_t)dim}, dtype::FP16());
    engine.tensors().allocate_and_register("w", {(uint32_t)dim}, dtype::FP16());
    engine.tensors().allocate_and_register("y", {(uint32_t)batch, (uint32_t)dim}, dtype::FP16());
    
    auto* x = engine.tensors().get("x");
    auto* w = engine.tensors().get("w");
    auto* y = engine.tensors().get("y");
    
    // Fill
    fill_random_fp16(static_cast<half*>(x->ptr), batch * dim, -1.0f, 1.0f);
    
    // Weight = 1.0 for simplicity
    std::vector<half> w_host(dim);
    for (int i = 0; i < dim; i++) {
        w_host[i] = __float2half(1.0f);
    }
    cudaMemcpy(w->ptr, w_host.data(), dim * sizeof(half), cudaMemcpyHostToDevice);
    
    // Execute
    CommandBuffer cb;
    cb.add_rmsnorm("y", "x", "w", 1e-5f);
    engine.execute(cb);
    engine.sync();
    
    // Verify output has reasonable values
    auto y_result = to_host(static_cast<half*>(y->ptr), batch * dim);
    
    // Check each row has RMS ≈ 1
    for (int b = 0; b < batch; b++) {
        float sum_sq = 0.0f;
        for (int d = 0; d < dim; d++) {
            float val = y_result[b * dim + d];
            sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / dim);
        // RMS should be close to 1 after normalization (with weight=1)
        assert(std::abs(rms - 1.0f) < 0.2f);  // Allow some tolerance
    }
    
    std::cout << "PASSED" << std::endl;
}

void test_embedding_pageable_mmap() {
    std::cout << "Test: EMBEDDING from pageable mmap... ";

    int device = 0;
    int pageable = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, device);
    if (!pageable) {
        std::cout << "SKIPPED (device has no pageableMemoryAccess)" << std::endl;
        return;
    }

    constexpr int vocab = 4;
    constexpr int dim = 8;
    const long page_size = sysconf(_SC_PAGESIZE);
    assert(page_size > 0);
    void* mapping = mmap(nullptr, size_t(page_size), PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(mapping != MAP_FAILED);
    half* table = static_cast<half*>(mapping);
    for (int row = 0; row < vocab; ++row) {
        for (int col = 0; col < dim; ++col) {
            table[row * dim + col] = __float2half(float(row * 10 + col));
        }
    }

    int32_t token = 2;
    int32_t* d_token = nullptr;
    half* d_output = nullptr;
    cudaMalloc(&d_token, sizeof(token));
    cudaMalloc(&d_output, dim * sizeof(half));
    cudaMemcpy(d_token, &token, sizeof(token), cudaMemcpyHostToDevice);

    kernels::launch_embedding_fp16(
        d_token, table, d_output, 1, 1, vocab, dim, nullptr);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    const auto output = to_host(d_output, dim);
    for (int col = 0; col < dim; ++col) {
        assert(output[col] == float(20 + col));
    }

    cudaFree(d_output);
    cudaFree(d_token);
    munmap(mapping, size_t(page_size));
    std::cout << "PASSED" << std::endl;
}

void test_matmul_fp16_kernel() {
    std::cout << "Test: MATMUL FP16 kernel... ";
    
    Engine engine;
    kernels::register_all_kernels(engine);
    
    int M = 4;
    int K = 64;
    int N = 32;
    
    engine.tensors().allocate_and_register("A", {(uint32_t)M, (uint32_t)K}, dtype::FP16());
    engine.tensors().allocate_and_register("B", {(uint32_t)K, (uint32_t)N}, dtype::FP16());
    engine.tensors().allocate_and_register("C", {(uint32_t)M, (uint32_t)N}, dtype::FP16());
    
    auto* A = engine.tensors().get("A");
    auto* B = engine.tensors().get("B");
    auto* C = engine.tensors().get("C");
    
    // Fill with simple values for verification
    // A = all 1s, B = all 1s -> C should be all K
    std::vector<half> a_host(M * K), b_host(K * N);
    for (int i = 0; i < M * K; i++) a_host[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) b_host[i] = __float2half(1.0f);
    
    cudaMemcpy(A->ptr, a_host.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(B->ptr, b_host.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // Execute
    CommandBuffer cb;
    cb.add_matmul("C", "A", "B");
    engine.execute(cb);
    engine.sync();
    
    // Verify
    auto c_result = to_host(static_cast<half*>(C->ptr), M * N);
    for (int i = 0; i < M * N; i++) {
        assert(std::abs(c_result[i] - (float)K) < 1.0f);  // Should be K
    }
    
    std::cout << "PASSED" << std::endl;
}

void test_softmax_kernel() {
    std::cout << "Test: SOFTMAX kernel... ";
    
    Engine engine;
    kernels::register_all_kernels(engine);
    
    int batch = 4;
    int seq = 16;
    
    engine.tensors().allocate_and_register("x", {(uint32_t)batch, (uint32_t)seq}, dtype::FP16());
    engine.tensors().allocate_and_register("y", {(uint32_t)batch, (uint32_t)seq}, dtype::FP16());
    
    auto* x = engine.tensors().get("x");
    auto* y = engine.tensors().get("y");
    
    fill_random_fp16(static_cast<half*>(x->ptr), batch * seq, -2.0f, 2.0f);
    
    // Execute
    CommandBuffer cb;
    cb.add_softmax("y", "x", -1);
    engine.execute(cb);
    engine.sync();
    
    // Verify each row sums to 1
    auto y_result = to_host(static_cast<half*>(y->ptr), batch * seq);
    
    for (int b = 0; b < batch; b++) {
        float sum = 0.0f;
        for (int s = 0; s < seq; s++) {
            float val = y_result[b * seq + s];
            assert(val >= 0.0f && val <= 1.0f);  // Probability
            sum += val;
        }
        assert(std::abs(sum - 1.0f) < 0.05f);  // Should sum to 1
    }
    
    std::cout << "PASSED" << std::endl;
}

void test_engine_forward_simulation() {
    std::cout << "Test: Forward pass simulation... " << std::endl;
    
    Engine engine;
    kernels::register_all_kernels(engine);
    
    // Simulate a mini transformer layer
    int batch = 1;
    int seq = 4;
    int dim = 64;
    int ff_dim = 256;
    
    // Allocate tensors
    engine.tensors().allocate_and_register("hidden", {(uint32_t)batch, (uint32_t)seq, (uint32_t)dim}, dtype::FP16());
    engine.tensors().allocate_and_register("ln_w", {(uint32_t)dim}, dtype::FP16());
    engine.tensors().allocate_and_register("h_norm", {(uint32_t)batch, (uint32_t)seq, (uint32_t)dim}, dtype::FP16());
    engine.tensors().allocate_and_register("ff_w1", {(uint32_t)dim, (uint32_t)ff_dim}, dtype::FP16());
    engine.tensors().allocate_and_register("ff_h", {(uint32_t)batch * (uint32_t)seq, (uint32_t)ff_dim}, dtype::FP16());
    engine.tensors().allocate_and_register("ff_act", {(uint32_t)batch * (uint32_t)seq, (uint32_t)ff_dim}, dtype::FP16());
    engine.tensors().allocate_and_register("ff_w2", {(uint32_t)ff_dim, (uint32_t)dim}, dtype::FP16());
    engine.tensors().allocate_and_register("ff_out", {(uint32_t)batch * (uint32_t)seq, (uint32_t)dim}, dtype::FP16());
    engine.tensors().allocate_and_register("output", {(uint32_t)batch, (uint32_t)seq, (uint32_t)dim}, dtype::FP16());
    
    // Initialize
    fill_random_fp16(static_cast<half*>(engine.tensors().get("hidden")->ptr), batch * seq * dim, -1.0f, 1.0f);
    fill_random_fp16(static_cast<half*>(engine.tensors().get("ln_w")->ptr), dim, 0.5f, 1.5f);
    fill_random_fp16(static_cast<half*>(engine.tensors().get("ff_w1")->ptr), dim * ff_dim, -0.1f, 0.1f);
    fill_random_fp16(static_cast<half*>(engine.tensors().get("ff_w2")->ptr), ff_dim * dim, -0.1f, 0.1f);
    
    // Build forward pass
    CommandBuffer cb;
    
    // 1. RMSNorm
    cb.add_rmsnorm("h_norm", "hidden", "ln_w", 1e-5f);
    
    // 2. FF up projection (need to reshape h_norm for matmul)
    // For simplicity, treat [batch, seq, dim] as [batch*seq, dim]
    cb.add_matmul("ff_h", "h_norm", "ff_w1");
    
    // 3. SiLU activation
    cb.add_silu("ff_act", "ff_h");
    
    // 4. FF down projection
    cb.add_matmul("ff_out", "ff_act", "ff_w2");
    
    // 5. Residual add
    cb.add_add("output", "hidden", "ff_out");
    
    std::cout << "  Command buffer: " << cb.size() << " commands" << std::endl;
    
    // Execute
    engine.execute(cb);
    engine.sync();
    
    // Verify output has reasonable values (not NaN/Inf)
    auto output = to_host(static_cast<half*>(engine.tensors().get("output")->ptr), batch * seq * dim);
    
    int valid_count = 0;
    for (size_t i = 0; i < output.size(); i++) {
        if (std::isfinite(output[i]) && std::abs(output[i]) < 100.0f) {
            valid_count++;
        }
    }
    
    float valid_ratio = (float)valid_count / output.size();
    std::cout << "  Valid outputs: " << (valid_ratio * 100.0f) << "%" << std::endl;
    
    assert(valid_ratio > 0.95f);  // At least 95% should be valid
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "==================================" << std::endl;
    std::cout << "HELIOS ENGINE - Kernel Test" << std::endl;
    std::cout << "==================================" << std::endl;
    
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
    test_add_kernel();
    test_silu_kernel();
    test_rmsnorm_kernel();
    test_embedding_pageable_mmap();
    test_matmul_fp16_kernel();
    test_softmax_kernel();
    test_engine_forward_simulation();
    
    std::cout << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "ALL TESTS PASSED ✓" << std::endl;
    std::cout << "==================================" << std::endl;
    
    return 0;
}
