// test_multimodal.cpp
// Test forward pass for CORTEX (Phi-4) and CODE (DeepSeek)

#include "src/engine.hpp"
#include "src/hnf_loader.hpp"
#include "src/graph_builder.hpp"
#include "src/sampler.hpp"
#include "src/kv_cache.hpp"
#include "kernels/kernels.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>

using namespace helios;

void print_vram() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "VRAM: " << ((total_mem - free_mem) / 1024 / 1024) << " MB used / "
              << (total_mem / 1024 / 1024) << " MB total (free: " 
              << (free_mem / 1024 / 1024) << " MB)" << std::endl;
}

bool test_block(const std::string& hnf_path, BlockID block_id, const std::string& block_name) {
    std::cout << "\n============================================" << std::endl;
    std::cout << "Testing: " << block_name << std::endl;
    std::cout << "============================================" << std::endl;
    
    print_vram();
    
    // Fresh engine for each test - NO scratch pool initially
    EngineConfig config;
    config.scratch_pool.pool_size_bytes = 64 * 1024 * 1024;  // 64MB minimal
    config.scratch_pool.auto_fraction = 0.0f;  // Disable auto-sizing!
    Engine engine(config);
    kernels::register_all_kernels(engine);
    
    // Fresh loader
    HnfLoader loader;
    if (!loader.open(hnf_path)) {
        std::cerr << "Failed to open HNF" << std::endl;
        return false;
    }
    
    // Get config for this block
    const ModelConfig& model_config = loader.config_for_block(block_id);
    
    std::cout << "Config:" << std::endl;
    std::cout << "  Arch: " << model_config.arch() << std::endl;
    std::cout << "  Layers: " << model_config.num_hidden_layers() << std::endl;
    std::cout << "  Hidden: " << model_config.hidden_size() << std::endl;
    std::cout << "  Heads: " << model_config.num_attention_heads() 
              << " (KV: " << model_config.num_key_value_heads() << ")" << std::endl;
    std::cout << "  Vocab: " << model_config.vocab_size() << std::endl;
    std::cout << "  MLP type: " << model_config.get<std::string>("mlp_type", "unknown") << std::endl;
    std::cout << "  RoPE type: " << model_config.get<std::string>("rope_type", "default") << std::endl;
    
    // Load block
    std::cout << "\n>>> Loading " << block_name << " tensors..." << std::endl;
    print_vram();
    
    // Check how many tensors belong to this block
    int block_tensor_count = 0;
    size_t block_total_size = 0;
    for (const auto& t : loader.tensors()) {
        std::string target = "";
        if (block_id == BLOCK_CORTEX) target = "cortex";
        else if (block_id == BLOCK_CODE_EXEC) target = "code_exec";
        else if (block_id == BLOCK_TEXT_MODEL) target = "text_model";
        
        if (t.block == target) {
            block_tensor_count++;
            block_total_size += t.size;
        }
    }
    std::cout << "  Block has " << block_tensor_count << " tensors, " 
              << (block_total_size / 1024 / 1024) << " MB total" << std::endl;
    
    if (!loader.load_block(block_id, engine)) {
        std::cerr << "Failed to load block!" << std::endl;
        return false;
    }
    std::cout << "  Loaded " << engine.tensors().count() << " tensors" << std::endl;
    print_vram();
    
    // List first few tensors to see naming
    std::cout << "\n>>> First 10 tensors:" << std::endl;
    auto names = engine.tensors().names();
    std::sort(names.begin(), names.end());
    for (size_t i = 0; i < std::min(size_t(10), names.size()); i++) {
        auto* t = engine.tensors().get(names[i]);
        std::cout << "  " << names[i] << " [";
        for (size_t j = 0; j < t->shape.size(); j++) {
            if (j > 0) std::cout << ",";
            std::cout << t->shape[j];
        }
        std::cout << "] " << dtype_name(t->dtype) << std::endl;
    }
    
    // Detect architecture
    GraphBuilder gb;
    
    // Determine prefix from tensor names
    std::string prefix = "";
    for (const auto& name : names) {
        if (name.find("cortex.") == 0) { prefix = "cortex"; break; }
        if (name.find("code.") == 0) { prefix = "code"; break; }
        if (name.find("text.") == 0) { prefix = "text"; break; }
        if (name.find("model.") == 0) { prefix = "model"; break; }
    }
    
    std::cout << "\n>>> Detected prefix: \"" << prefix << "\"" << std::endl;
    
    try {
        auto arch = gb.detect_architecture(engine, prefix);
        std::cout << ">>> Architecture detected:" << std::endl;
        std::cout << "  Prefix: " << arch.prefix << std::endl;
        std::cout << "  Layers: " << arch.num_layers << std::endl;
        std::cout << "  Fused QKV: " << (arch.has_fused_qkv ? "yes" : "no") << std::endl;
        std::cout << "  Fused gate_up: " << (arch.has_fused_gate_up ? "yes" : "no") << std::endl;
        std::cout << "  Has gate: " << (arch.has_gate ? "yes" : "no") << std::endl;
        
        // Validate weights
        auto err = gb.validate_weights(engine, model_config, arch);
        if (!err.empty()) {
            std::cerr << "Validation failed: " << err << std::endl;
            return false;
        }
        std::cout << "  Validation: OK" << std::endl;
        
        // Try to allocate scratch and do one forward pass
        std::cout << "\n>>> Attempting forward pass..." << std::endl;
        print_vram();
        
        gb.allocate_scratch(engine, model_config, arch, 1, 1);
        
        // Create input token
        engine.tensors().allocate_and_register(
            "input_tokens", {1, 1}, dtype::INT32()
        );
        int32_t token = 1;
        auto* input_info = engine.tensors().get("input_tokens");
        cudaMemcpy(input_info->ptr, &token, sizeof(int32_t), cudaMemcpyHostToDevice);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        auto cb = gb.build_forward(engine, model_config, arch, "input_tokens", 1, 1);
        engine.execute(cb);
        engine.sync();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << "  Forward pass: " << ms << " ms - SUCCESS!" << std::endl;
        
        // Get logits
        auto* logits = gb.get_logits(engine);
        if (logits) {
            std::cout << "  Logits shape: [";
            for (size_t i = 0; i < logits->shape.size(); i++) {
                if (i > 0) std::cout << ",";
                std::cout << logits->shape[i];
            }
            std::cout << "]" << std::endl;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.hnf> [block]" << std::endl;
        std::cerr << "  block: cortex, code, text (default: test all)" << std::endl;
        return 1;
    }
    
    std::string hnf_path = argv[1];
    std::string test_block_name = argc > 2 ? argv[2] : "all";
    
    std::cout << "============================================" << std::endl;
    std::cout << "HELIOS Multimodal Test" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Model: " << hnf_path << std::endl;
    print_vram();
    
    bool success = true;
    
    if (test_block_name == "all" || test_block_name == "code") {
        // Test CODE first (smaller: 1.4GB)
        if (!test_block(hnf_path, BLOCK_CODE_EXEC, "CODE (DeepSeek 1.3B)")) {
            success = false;
        }
    }
    
    if (test_block_name == "all" || test_block_name == "cortex") {
        // Test CORTEX (larger: 4.3GB)
        if (!test_block(hnf_path, BLOCK_CORTEX, "CORTEX (Phi-4)")) {
            success = false;
        }
    }
    
    if (test_block_name == "all" || test_block_name == "text") {
        // Test TEXT
        if (!test_block(hnf_path, BLOCK_TEXT_MODEL, "TEXT (Qwen2 3B)")) {
            success = false;
        }
    }
    
    std::cout << "\n============================================" << std::endl;
    std::cout << "TEST " << (success ? "PASSED" : "FAILED") << std::endl;
    std::cout << "============================================" << std::endl;
    print_vram();
    
    return success ? 0 : 1;
}
