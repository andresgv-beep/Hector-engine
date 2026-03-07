// test_generate_full.cpp
// ============================================================================
// HELIOS FULL TEXT GENERATION - Tokenizer Real + Forward + Sampling + Decode
// ============================================================================
// El monstruo habla con palabras reales.
//

#include "src/engine.hpp"
#include "src/hnf_loader.hpp"
#include "src/graph_builder.hpp"
#include "src/sampler.hpp"
#include "src/kv_cache.hpp"
#include "kernels/kernels.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

using namespace helios;

void print_tokens(const std::vector<int32_t>& ids, const HTFTokenizer* tok) {
    for (int32_t id : ids) {
        std::string t = tok->id_to_token(id);
        // Escape for display
        std::string disp;
        for (char c : t) {
            if (c == '\n') disp += "\\n";
            else if (c == '\t') disp += "\\t";
            else if (c < 32 || c > 126) disp += "?";
            else disp += c;
        }
        std::cout << "[" << disp << "] ";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.hnf> [prompt] [max_tokens] [temperature]" << std::endl;
        std::cerr << "Example: " << argv[0] << " model.hnf \"Hello, I am\" 50 0.7" << std::endl;
        return 1;
    }
    
    std::string hnf_path = argv[1];
    std::string prompt = argc > 2 ? argv[2] : "Hello";
    int max_tokens = argc > 3 ? std::atoi(argv[3]) : 32;
    float temperature = argc > 4 ? std::atof(argv[4]) : 0.7f;
    
    std::cout << "============================================" << std::endl;
    std::cout << "HELIOS Full Text Generation" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Model: " << hnf_path << std::endl;
    std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Max tokens: " << max_tokens << std::endl;
    std::cout << "Temperature: " << temperature << std::endl;
    std::cout << std::endl;
    
    try {
        // 1. Setup engine
        EngineConfig config;
        config.scratch_pool.pool_size_bytes = 0;
        config.scratch_pool.auto_fraction = 0.0f;
        Engine engine(config);
        kernels::register_all_kernels(engine);
        
        // 2. Load model
        std::cout << ">>> Loading model..." << std::endl;
        HnfLoader loader;
        if (!loader.open(hnf_path)) {
            throw std::runtime_error("Failed to open HNF: " + hnf_path);
        }
        
        auto model_config = loader.config();
        std::cout << "  Arch: " << model_config.arch() << std::endl;
        std::cout << "  Vocab: " << model_config.vocab_size() << std::endl;
        std::cout << "  Layers: " << model_config.num_hidden_layers() << std::endl;
        std::cout << "  Hidden: " << model_config.hidden_size() << std::endl;
        
        // Load text model block
        if (!loader.load_block(BLOCK_TEXT_MODEL, engine)) {
            throw std::runtime_error("Failed to load text model block");
        }
        std::cout << "  Loaded " << engine.tensors().count() << " tensors" << std::endl;
        
        // 3. Get tokenizer
        std::cout << "\n>>> Loading tokenizer..." << std::endl;
        const HTFTokenizer* tokenizer = loader.tokenizer("text");
        if (!tokenizer) {
            throw std::runtime_error("Failed to load tokenizer");
        }
        std::cout << "  Vocab: " << tokenizer->vocab_size() << std::endl;
        std::cout << "  BOS: " << (tokenizer->bos_token_id() ? std::to_string(*tokenizer->bos_token_id()) : "none") << std::endl;
        std::cout << "  EOS: " << (tokenizer->eos_token_id() ? std::to_string(*tokenizer->eos_token_id()) : "none") << std::endl;
        
        // 4. Encode prompt
        std::cout << "\n>>> Encoding prompt..." << std::endl;
        std::vector<int32_t> prompt_ids = tokenizer->encode(prompt, false, false);
        std::cout << "  Tokens (" << prompt_ids.size() << "): ";
        print_tokens(prompt_ids, tokenizer);
        std::cout << std::endl;
        
        // 5. Setup graph builder
        GraphBuilder gb;
        auto arch = gb.detect_architecture(engine, "text");
        std::cout << "\n>>> Architecture: " << arch.prefix << " (" << arch.num_layers << " layers)" << std::endl;
        
        // Validate
        auto err = gb.validate_weights(engine, model_config, arch);
        if (!err.empty()) {
            throw std::runtime_error("Validation failed: " + err);
        }
        
        // 6. Setup KV cache
        KVCacheConfig kv_config;
        kv_config.num_layers = arch.num_layers;
        kv_config.num_kv_heads = model_config.num_key_value_heads();
        kv_config.head_dim = model_config.head_dim();
        kv_config.max_seq_len = 2048;
        
        KVCache kv_cache;
        if (!kv_cache.allocate(kv_config)) {
            throw std::runtime_error("Failed to allocate KV cache");
        }
        std::cout << "  KV cache: " << (kv_config.total_bytes() / 1024 / 1024) << " MB" << std::endl;
        
        // Register KV cache tensors in engine for GraphBuilder
        std::string kv_prefix = "kv_cache";
        for (uint32_t layer = 0; layer < arch.num_layers; layer++) {
            std::string k_name = kv_prefix + ".layer" + std::to_string(layer) + ".k";
            std::string v_name = kv_prefix + ".layer" + std::to_string(layer) + ".v";
            
            // Register external pointers (KVCache owns the memory)
            engine.tensors().register_external(
                k_name,
                kv_cache.k_cache(layer),
                {1, kv_config.max_seq_len, kv_config.num_kv_heads, kv_config.head_dim},
                dtype::FP16()
            );
            engine.tensors().register_external(
                v_name,
                kv_cache.v_cache(layer),
                {1, kv_config.max_seq_len, kv_config.num_kv_heads, kv_config.head_dim},
                dtype::FP16()
            );
        }
        
        // 7. Setup sampler
        Sampler sampler;
        SamplingConfig sample_config;
        if (temperature < 0.01f) {
            sample_config = SamplingConfig::greedy();
        } else {
            sample_config = SamplingConfig::creative(temperature, 40, 0.9f);
        }
        
        // 8. Allocate scratch
        gb.allocate_scratch(engine, model_config, arch, 1, 1);
        
        // Create input tensor
        engine.tensors().allocate_and_register(
            "input_tokens",
            {1, 1},
            dtype::INT32()
        );
        
        // 9. Prefill: Process prompt tokens
        std::cout << "\n>>> Prefill (" << prompt_ids.size() << " tokens)..." << std::endl;
        auto prefill_start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < prompt_ids.size(); i++) {
            int32_t token = prompt_ids[i];
            
            // Copy token to GPU
            auto* input_info = engine.tensors().get("input_tokens");
            cudaMemcpy(input_info->ptr, &token, sizeof(int32_t), cudaMemcpyHostToDevice);
            
            // Forward pass with KV cache
            auto cb = gb.build_forward_cached(
                engine, model_config, arch,
                "input_tokens", 1, 1,  // batch=1, seq=1
                kv_prefix,
                static_cast<uint32_t>(i),  // cache position
                kv_config.max_seq_len
            );
            engine.execute(cb);
            engine.sync();
        }
        
        auto prefill_end = std::chrono::high_resolution_clock::now();
        auto prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end - prefill_start).count();
        std::cout << "  Prefill time: " << prefill_ms << " ms" << std::endl;
        
        // 10. Generate tokens
        std::cout << "\n>>> Generating..." << std::endl;
        std::cout << "Output: " << prompt;
        
        std::vector<int32_t> generated_ids = prompt_ids;  // Include prompt
        uint32_t position = prompt_ids.size();
        
        int32_t eos_id = tokenizer->eos_token_id().value_or(151645);
        
        auto gen_start = std::chrono::high_resolution_clock::now();
        
        for (int step = 0; step < max_tokens; step++) {
            // Get logits from last forward
            auto* logits_info = gb.get_logits(engine);
            if (!logits_info) {
                throw std::runtime_error("Failed to get logits");
            }
            
            // Sample next token
            int32_t next_token = sampler.sample(
                (const half*)logits_info->ptr,
                model_config.vocab_size(),
                sample_config,
                nullptr
            );
            
            // Decode and print
            std::string token_str = tokenizer->decode({next_token});
            std::cout << token_str << std::flush;
            
            generated_ids.push_back(next_token);
            
            // Check EOS
            if (next_token == eos_id) {
                break;
            }
            
            // Forward pass for next token
            auto* input_info = engine.tensors().get("input_tokens");
            cudaMemcpy(input_info->ptr, &next_token, sizeof(int32_t), cudaMemcpyHostToDevice);
            
            auto cb = gb.build_forward_cached(
                engine, model_config, arch,
                "input_tokens", 1, 1,
                kv_prefix,
                position,
                kv_config.max_seq_len
            );
            engine.execute(cb);
            engine.sync();
            
            position++;
        }
        
        auto gen_end = std::chrono::high_resolution_clock::now();
        auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count();
        
        std::cout << std::endl << std::endl;
        
        // 11. Stats
        int tokens_generated = generated_ids.size() - prompt_ids.size();
        float tokens_per_sec = tokens_generated * 1000.0f / std::max(1L, gen_ms);
        
        std::cout << "============================================" << std::endl;
        std::cout << "GENERATION COMPLETE" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "Prompt tokens: " << prompt_ids.size() << std::endl;
        std::cout << "Generated tokens: " << tokens_generated << std::endl;
        std::cout << "Prefill: " << prefill_ms << " ms" << std::endl;
        std::cout << "Generation: " << gen_ms << " ms" << std::endl;
        std::cout << "Speed: " << tokens_per_sec << " tok/s" << std::endl;
        std::cout << std::endl;
        
        // Full decode
        std::cout << "Full output:" << std::endl;
        std::cout << "\"" << tokenizer->decode(generated_ids) << "\"" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
