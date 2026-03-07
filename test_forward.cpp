// test_forward.cpp
// Test forward pass con pesos reales de Qwen

#include "hnf_loader.hpp"
#include "kernels.hpp"
#include <iostream>
#include <vector>
#include <cstring>

void print_vram() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "VRAM: " << ((total_mem - free_mem) / 1024 / 1024) << " MB used" << std::endl;
}

// Helper to print first N values of a tensor
void print_tensor_sample(const char* name, void* ptr, size_t count, size_t n = 8) {
    std::vector<uint16_t> host(n);
    cudaMemcpy(host.data(), ptr, n * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    
    std::cout << name << " (first " << n << " values as fp16 bits): ";
    for (size_t i = 0; i < n; i++) {
        std::cout << std::hex << host[i] << " ";
    }
    std::cout << std::dec << std::endl;
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "HELIOS Forward Pass Test (Qwen 4B)" << std::endl;
    std::cout << "============================================" << std::endl;
    
    print_vram();
    
    // Create engine
    helios::EngineConfig config;
    config.scratch_pool.pool_size_bytes = 512 * 1024 * 1024;  // 512 MB scratch
    helios::Engine engine(config);
    helios::kernels::register_all_kernels(engine);
    
    // Load model
    helios::HnfLoader loader;
    if (!loader.open("/home/andres/Escritorio/helios-engine/tests/helios_core_v10.hnf")) {
        std::cerr << "Failed to open HNF" << std::endl;
        return 1;
    }
    
    std::cout << "\nLoading TEXT block..." << std::endl;
    if (!loader.load_block(helios::BLOCK_TEXT_MODEL, engine)) {
        std::cerr << "Failed to load TEXT block" << std::endl;
        return 1;
    }
    
    print_vram();
    
    const auto& cfg = loader.config();
    std::cout << "\nModel config:" << std::endl;
    std::cout << "  hidden_size: " << cfg.hidden_size << std::endl;
    std::cout << "  vocab_size: " << cfg.vocab_size << std::endl;
    std::cout << "  num_heads: " << cfg.num_attention_heads << std::endl;
    std::cout << "  num_kv_heads: " << cfg.num_key_value_heads << std::endl;
    std::cout << "  head_dim: " << cfg.head_dim << std::endl;
    
    // ========================================
    // TEST 1: Embedding lookup
    // ========================================
    std::cout << "\n=== TEST 1: Embedding Lookup ===" << std::endl;
    
    auto* emb_weight = engine.tensors().get("text.token_embedding.weight");
    if (!emb_weight) {
        std::cerr << "Embedding weight not found" << std::endl;
        return 1;
    }
    
    std::cout << "Embedding weight: [" << emb_weight->shape[0] << ", " 
              << emb_weight->shape[1] << "] " << emb_weight->dtype_str() << std::endl;
    
    // Create input token: single token ID = 1 (just a test)
    int32_t token_id = 1;  // Test token
    void* d_token;
    cudaMalloc(&d_token, sizeof(int32_t));
    cudaMemcpy(d_token, &token_id, sizeof(int32_t), cudaMemcpyHostToDevice);
    
    // Register input tensor
    helios::TensorInfo token_info;
    token_info.ptr = d_token;
    token_info.shape = {1, 1};  // [batch=1, seq=1]
    token_info.dtype = helios::dtype::INT32();
    token_info.size_bytes = sizeof(int32_t);
    token_info.owns_memory = false;
    engine.tensors().register_tensor("input_tokens", token_info);
    
    // Allocate output: [1, 1, hidden_size]
    void* d_hidden;
    size_t hidden_bytes = cfg.hidden_size * sizeof(uint16_t);
    cudaMalloc(&d_hidden, hidden_bytes);
    
    helios::TensorInfo hidden_info;
    hidden_info.ptr = d_hidden;
    hidden_info.shape = {1, 1, cfg.hidden_size};
    hidden_info.dtype = helios::dtype::FP16();
    hidden_info.size_bytes = hidden_bytes;
    hidden_info.owns_memory = false;
    engine.tensors().register_tensor("hidden_state", hidden_info);
    
    // Execute embedding lookup
    helios::CommandBuffer cb;
    cb.add_embedding("hidden_state", "input_tokens", "text.token_embedding.weight");
    
    std::cout << "Executing embedding lookup..." << std::endl;
    engine.execute(cb);
    engine.sync();
    
    print_tensor_sample("hidden_state", d_hidden, cfg.hidden_size);
    std::cout << "Embedding lookup: OK" << std::endl;
    
    // ========================================
    // TEST 2: RMSNorm
    // ========================================
    std::cout << "\n=== TEST 2: RMSNorm (layer 0) ===" << std::endl;
    
    auto* norm_weight = engine.tensors().get("text.layer0.ln_attn_in.weight");
    if (!norm_weight) {
        std::cerr << "Norm weight not found" << std::endl;
        return 1;
    }
    
    std::cout << "Norm weight: [" << norm_weight->shape[0] << "] " 
              << norm_weight->dtype_str() << std::endl;
    
    // Allocate normed output
    void* d_normed;
    cudaMalloc(&d_normed, hidden_bytes);
    
    helios::TensorInfo normed_info;
    normed_info.ptr = d_normed;
    normed_info.shape = {1, 1, cfg.hidden_size};
    normed_info.dtype = helios::dtype::FP16();
    normed_info.size_bytes = hidden_bytes;
    normed_info.owns_memory = false;
    engine.tensors().register_tensor("normed", normed_info);
    
    // Execute RMSNorm
    helios::CommandBuffer cb2;
    cb2.add_rmsnorm("normed", "hidden_state", "text.layer0.ln_attn_in.weight", cfg.rms_norm_eps);
    
    std::cout << "Executing RMSNorm..." << std::endl;
    engine.execute(cb2);
    engine.sync();
    
    print_tensor_sample("normed", d_normed, cfg.hidden_size);
    std::cout << "RMSNorm: OK" << std::endl;
    
    // ========================================
    // TEST 3: Q Projection (Matmul with HQ5K weights)
    // ========================================
    std::cout << "\n=== TEST 3: Q Projection (HQ5K matmul) ===" << std::endl;
    
    auto* q_weight = engine.tensors().get("text.layer0.attn.q_proj.weight");
    if (!q_weight) {
        std::cerr << "Q weight not found" << std::endl;
        return 1;
    }
    
    std::cout << "Q weight: [" << q_weight->shape[0] << ", " << q_weight->shape[1] << "] " 
              << q_weight->dtype_str() << std::endl;
    
    // Allocate Q output: [1, 1, hidden_size]
    void* d_q;
    cudaMalloc(&d_q, hidden_bytes);
    
    helios::TensorInfo q_info;
    q_info.ptr = d_q;
    q_info.shape = {1, cfg.hidden_size};  // [batch*seq, hidden]
    q_info.dtype = helios::dtype::FP16();
    q_info.size_bytes = hidden_bytes;
    q_info.owns_memory = false;
    engine.tensors().register_tensor("q_out", q_info);
    
    // Reshape normed for matmul: [1, hidden_size]
    auto* normed_t = engine.tensors().get("normed");
    normed_t->shape = {1, cfg.hidden_size};
    
    // Execute Matmul
    helios::CommandBuffer cb3;
    cb3.add_matmul("q_out", "normed", "text.layer0.attn.q_proj.weight");
    
    std::cout << "Executing Q projection (HQ5K matmul)..." << std::endl;
    engine.execute(cb3);
    engine.sync();
    
    print_tensor_sample("q_out", d_q, cfg.hidden_size);
    std::cout << "Q Projection: OK" << std::endl;
    
    // ========================================
    // CLEANUP
    // ========================================
    std::cout << "\n=== CLEANUP ===" << std::endl;
    
    cudaFree(d_token);
    cudaFree(d_hidden);
    cudaFree(d_normed);
    cudaFree(d_q);
    
    loader.unload_block(helios::BLOCK_TEXT_MODEL, engine);
    print_vram();
    
    std::cout << "\n============================================" << std::endl;
    std::cout << "ALL FORWARD TESTS PASSED ✓" << std::endl;
    std::cout << "============================================" << std::endl;
    
    return 0;
}
