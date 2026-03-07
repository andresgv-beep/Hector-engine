// test_generate.cpp
// ============================================================================
// HELIOS TEXT GENERATION TEST
// ============================================================================
// Primera generación de texto con el motor HELIOS.
// El monstruo habla.
//

#include "src/engine.hpp"
#include "src/hnf_loader.hpp"
#include "src/graph_builder.hpp"
#include "src/sampler.hpp"
#include "kernels/kernels.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

using namespace helios;

// Simple token to string mapping for Qwen (subset for demo)
// In production, load from tokenizer
std::string token_to_string(int32_t token_id) {
    // This is a placeholder - in real use, load from tokenizer.json
    // For now, return token ID as string
    return "[" + std::to_string(token_id) + "]";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.hnf> [max_tokens] [temperature]" << std::endl;
        return 1;
    }
    
    std::string hnf_path = argv[1];
    int max_tokens = argc > 2 ? std::atoi(argv[2]) : 20;
    float temperature = argc > 3 ? std::atof(argv[3]) : 0.0f;  // 0 = greedy
    
    std::cout << "============================================" << std::endl;
    std::cout << "HELIOS Text Generation" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Model: " << hnf_path << std::endl;
    std::cout << "Max tokens: " << max_tokens << std::endl;
    std::cout << "Temperature: " << temperature << std::endl;
    std::cout << std::endl;
    
    try {
        // 1. Setup engine - NO auto-allocate scratch pool
        EngineConfig config;
        config.scratch_pool.pool_size_bytes = 0;  // Don't pre-allocate
        config.scratch_pool.auto_fraction = 0.0f;  // Disable auto-sizing
        Engine engine(config);
        kernels::register_all_kernels(engine);
        
        // 2. Load model
        std::cout << ">>> Loading model..." << std::endl;
        HnfLoader loader;
        if (!loader.open(hnf_path)) {
            throw std::runtime_error("Failed to open HNF: " + hnf_path);
        }
        
        auto model_config = loader.config();
        std::cout << "  Vocab: " << model_config.vocab_size() << std::endl;
        std::cout << "  Hidden: " << model_config.hidden_size() << std::endl;
        std::cout << "  Layers: " << model_config.num_hidden_layers() << std::endl;
        
        // Load text model block
        if (!loader.load_block(BLOCK_TEXT_MODEL, engine)) {
            throw std::runtime_error("Failed to load text model block");
        }
        
        // 3. Setup graph builder
        GraphBuilder gb;
        auto arch = gb.detect_architecture(engine, "text");
        
        std::cout << ">>> Architecture detected:" << std::endl;
        std::cout << "  Prefix: " << arch.prefix << std::endl;
        std::cout << "  Layers: " << arch.num_layers << std::endl;
        
        // Validate
        auto err = gb.validate_weights(engine, model_config, arch);
        if (!err.empty()) {
            throw std::runtime_error("Validation failed: " + err);
        }
        
        // 4. Setup sampler
        Sampler sampler;
        SamplingConfig sample_config;
        if (temperature < 0.01f) {
            sample_config = SamplingConfig::greedy();
        } else {
            sample_config = SamplingConfig::creative(temperature, 50, 0.9f);
        }
        
        // 5. Allocate scratch for seq_len = 1 (autoregressive)
        gb.allocate_scratch(engine, model_config, arch, 1, 1);
        
        // 6. Create input token tensor
        // Start with token ID 1 (often <s> or similar)
        std::vector<int32_t> input_tokens = {1};  // Initial token
        
        engine.tensors().allocate_and_register(
            "input_tokens",
            {1, 1},  // [batch=1, seq=1]
            dtype::INT32()
        );
        
        // 7. Generate tokens
        std::cout << "\n>>> Generating..." << std::endl;
        std::cout << "Tokens: ";
        
        std::vector<int32_t> generated;
        generated.push_back(input_tokens[0]);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int step = 0; step < max_tokens; step++) {
            // Copy current token to GPU
            auto* input_info = engine.tensors().get("input_tokens");
            cudaMemcpy(input_info->ptr, &input_tokens.back(), sizeof(int32_t),
                       cudaMemcpyHostToDevice);
            
            // Build and execute forward pass
            auto cb = gb.build_forward(engine, model_config, arch, "input_tokens", 1, 1);
            engine.execute(cb);
            engine.sync();
            
            // Get logits
            auto* logits_info = gb.get_logits(engine);
            if (!logits_info) {
                throw std::runtime_error("Failed to get logits");
            }
            
            // Sample next token
            int32_t next_token = sampler.sample(
                (const half*)logits_info->ptr,
                model_config.vocab_size(),
                sample_config,
                nullptr  // default stream
            );
            
            // Output
            std::cout << next_token << " " << std::flush;
            
            // Store
            generated.push_back(next_token);
            input_tokens[0] = next_token;
            
            // Check for EOS (token 2 in many models)
            // Note: In production, get actual EOS token from config
            if (next_token == 2 || next_token == 151643) {  // Common EOS tokens
                std::cout << "[EOS]";
                break;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << std::endl << std::endl;
        
        // 8. Stats
        int tokens_generated = generated.size() - 1;  // Exclude initial
        float tokens_per_sec = tokens_generated * 1000.0f / duration.count();
        
        std::cout << "============================================" << std::endl;
        std::cout << "GENERATION COMPLETE" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "Tokens generated: " << tokens_generated << std::endl;
        std::cout << "Time: " << duration.count() << " ms" << std::endl;
        std::cout << "Speed: " << tokens_per_sec << " tok/s" << std::endl;
        std::cout << std::endl;
        
        std::cout << "Generated sequence:" << std::endl;
        std::cout << "  ";
        for (int32_t t : generated) {
            std::cout << t << " ";
        }
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
