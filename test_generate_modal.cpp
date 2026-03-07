// test_generate_modal.cpp
// ============================================================================
// HELIOS Modal Generation Test - TEXT, CODE, or CORTEX
// ============================================================================

#include "src/engine.hpp"
#include "src/hnf_loader.hpp"
#include "src/graph_builder.hpp"
#include "src/sampler.hpp"
#include "src/kv_cache.hpp"
#include "kernels/kernels.hpp"
#include <iostream>
#include <chrono>

using namespace helios;

void print_vram() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "VRAM: " << ((total_mem - free_mem) / 1024 / 1024) << " MB used / "
              << (total_mem / 1024 / 1024) << " MB total" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.hnf> <modal> [prompt] [max_tokens] [temperature]" << std::endl;
        std::cerr << "  modal: text, code, cortex" << std::endl;
        std::cerr << "Example: " << argv[0] << " model.hnf code \"def fibonacci(n):\" 50 0.7" << std::endl;
        return 1;
    }
    
    std::string hnf_path = argv[1];
    std::string modal = argv[2];
    std::string prompt = argc > 3 ? argv[3] : "Hello";
    int max_tokens = argc > 4 ? std::atoi(argv[4]) : 32;
    float temperature = argc > 5 ? std::atof(argv[5]) : 0.7f;
    
    // Determine block ID
    BlockID block_id;
    std::string prefix;
    if (modal == "text") {
        block_id = BLOCK_TEXT_MODEL;
        prefix = "text";
    } else if (modal == "code") {
        block_id = BLOCK_CODE_EXEC;
        prefix = "code";
    } else if (modal == "cortex") {
        block_id = BLOCK_CORTEX;
        prefix = "cortex";
    } else {
        std::cerr << "Unknown modal: " << modal << std::endl;
        return 1;
    }
    
    std::cout << "============================================" << std::endl;
    std::cout << "HELIOS " << modal << " Generation" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Model: " << hnf_path << std::endl;
    std::cout << "Modal: " << modal << std::endl;
    std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Max tokens: " << max_tokens << std::endl;
    std::cout << "Temperature: " << temperature << std::endl;
    print_vram();
    std::cout << std::endl;
    
    try {
        // 1. Setup engine with profiling
        EngineConfig config;
        config.scratch_pool.pool_size_bytes = 64 * 1024 * 1024;
        config.scratch_pool.auto_fraction = 0.0f;
        config.enable_profiling = true;  // Debug
        Engine engine(config);
        kernels::register_all_kernels(engine);
        
        // 2. Load HNF
        std::cout << ">>> Loading model..." << std::endl;
        HnfLoader loader;
        if (!loader.open(hnf_path)) {
            throw std::runtime_error("Failed to open HNF");
        }
        
        // Get config for this modal
        const ModelConfig& model_config = loader.config_for_block(block_id);
        std::cout << "  Arch: " << model_config.arch() << std::endl;
        std::cout << "  Layers: " << model_config.num_hidden_layers() << std::endl;
        std::cout << "  Hidden: " << model_config.hidden_size() << std::endl;
        std::cout << "  Vocab: " << model_config.vocab_size() << std::endl;
        
        // Load block
        if (!loader.load_block(block_id, engine)) {
            throw std::runtime_error("Failed to load block");
        }
        std::cout << "  Loaded " << engine.tensors().count() << " tensors" << std::endl;
        print_vram();
        
        // 3. Get tokenizer
        // List available domains first
        auto domains = loader.tokenizer_domains();
        std::cout << "  Available tokenizer domains: ";
        if (domains.empty()) {
            std::cout << "(none detected yet)";
        } else {
            for (const auto& d : domains) std::cout << d << " ";
        }
        std::cout << std::endl;
        
        // Try to get tokenizer for this modal
        std::string tok_domain = modal;  // "text", "code", "cortex"
        const HTFTokenizer* tokenizer = loader.tokenizer(tok_domain);
        bool use_raw_tokens = false;
        
        if (!tokenizer) {
            // Fallback to text tokenizer
            std::cout << "  No '" << tok_domain << "' tokenizer, trying 'text'..." << std::endl;
            tokenizer = loader.tokenizer("text");
        }
        
        if (!tokenizer) {
            std::cout << "  No tokenizer available, using raw token mode" << std::endl;
            use_raw_tokens = true;
        } else {
            std::cout << "  Tokenizer vocab: " << tokenizer->vocab_size() << std::endl;
            
            // Check if tokenizer vocab matches model vocab
            // Allow small difference (some models have extra special tokens)
            int64_t vocab_diff = std::abs(static_cast<int64_t>(tokenizer->vocab_size()) - 
                                          static_cast<int64_t>(model_config.vocab_size()));
            if (vocab_diff > 1000) {
                std::cout << "  WARNING: Tokenizer vocab (" << tokenizer->vocab_size() 
                          << ") differs significantly from Model vocab (" << model_config.vocab_size() << ")" << std::endl;
                std::cout << "  Using raw token mode for this model" << std::endl;
                use_raw_tokens = true;
            } else if (vocab_diff > 0) {
                std::cout << "  Note: Tokenizer vocab (" << tokenizer->vocab_size() 
                          << ") != Model vocab (" << model_config.vocab_size() 
                          << "), diff=" << vocab_diff << " (OK)" << std::endl;
            }
        }
        
        // 4. Encode prompt
        std::cout << "\n>>> Encoding prompt..." << std::endl;
        std::vector<int32_t> prompt_ids;
        
        if (use_raw_tokens) {
            // For models without matching tokenizer, use simple token IDs
            // Token 1 is usually a good start token
            prompt_ids = {1};
            std::cout << "  Using raw start token: [1]" << std::endl;
        } else {
            prompt_ids = tokenizer->encode(prompt, false, false);
            
            // Add BOS token at the beginning (required by DeepSeek and most models)
            int32_t bos_id = tokenizer->bos_token_id().value_or(1);
            prompt_ids.insert(prompt_ids.begin(), bos_id);
            std::cout << "  Added BOS token: " << bos_id << std::endl;
            
            std::cout << "  Tokens (" << prompt_ids.size() << "): ";
            for (size_t i = 0; i < std::min(prompt_ids.size(), size_t(10)); i++) {
                std::cout << prompt_ids[i] << " ";
            }
            if (prompt_ids.size() > 10) std::cout << "...";
            std::cout << std::endl;
        }
        
        // 5. Detect architecture
        GraphBuilder gb;
        auto arch = gb.detect_architecture(engine, prefix);
        std::cout << "\n>>> Architecture: " << arch.prefix << std::endl;
        std::cout << "  Layers: " << arch.num_layers << std::endl;
        std::cout << "  Fused QKV: " << (arch.has_fused_qkv ? "yes" : "no") << std::endl;
        std::cout << "  Fused gate_up: " << (arch.has_fused_gate_up ? "yes" : "no") << std::endl;
        
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
        
        // Register KV cache tensors
        std::string kv_prefix = "kv_cache";
        for (uint32_t layer = 0; layer < arch.num_layers; layer++) {
            std::string k_name = kv_prefix + ".layer" + std::to_string(layer) + ".k";
            std::string v_name = kv_prefix + ".layer" + std::to_string(layer) + ".v";
            
            engine.tensors().register_external(
                k_name, kv_cache.k_cache(layer),
                {1, kv_config.max_seq_len, kv_config.num_kv_heads, kv_config.head_dim},
                dtype::FP16()
            );
            engine.tensors().register_external(
                v_name, kv_cache.v_cache(layer),
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
        
        engine.tensors().allocate_and_register(
            "input_tokens", {1, 1}, dtype::INT32()
        );
        
        // 9. Prefill
        std::cout << "\n>>> Prefill (" << prompt_ids.size() << " tokens)..." << std::endl;
        auto prefill_start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < prompt_ids.size(); i++) {
            int32_t token = prompt_ids[i];
            auto* input_info = engine.tensors().get("input_tokens");
            
            // Update shape for single token
            input_info->shape = {1, 1};
            
            cudaMemcpy(input_info->ptr, &token, sizeof(int32_t), cudaMemcpyHostToDevice);
            
            auto cb = gb.build_forward_cached(
                engine, model_config, arch,
                "input_tokens", 1, 1,
                kv_prefix, static_cast<uint32_t>(i), kv_config.max_seq_len
            );
            engine.execute(cb);
            engine.sync();
            
            // Debug: check KV cache for layer 0 after first few tokens
            if (i < 3) {
                auto* k_cache = engine.tensors().get(kv_prefix + ".layer0.k");
                if (k_cache) {
                    std::vector<half> k_host(8);  // Just first 8 values
                    // Read from position i in cache
                    size_t offset = i * model_config.num_key_value_heads() * model_config.head_dim();
                    cudaMemcpy(k_host.data(), (half*)k_cache->ptr + offset, 
                              8 * sizeof(half), cudaMemcpyDeviceToHost);
                    std::cerr << "  [Prefill " << i << "] K_cache[" << i << "][0:8]: ";
                    for (int j = 0; j < 8; j++) {
                        std::cerr << __half2float(k_host[j]) << " ";
                    }
                    std::cerr << std::endl;
                }
            }
        }
        
        auto prefill_end = std::chrono::high_resolution_clock::now();
        auto prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end - prefill_start).count();
        std::cout << "  Prefill: " << prefill_ms << " ms (" 
                  << (prompt_ids.size() * 1000.0 / prefill_ms) << " tok/s)" << std::endl;
        
        // 10. Generate
        std::cout << "\n>>> Generating..." << std::endl;
        if (use_raw_tokens) {
            std::cout << "Output (raw tokens): " << std::flush;
        } else {
            std::cout << "Output: " << prompt << std::flush;
        }
        
        // Debug: show generated token IDs
        bool debug_tokens = true;
        
        std::vector<int32_t> generated_ids = prompt_ids;
        uint32_t position = prompt_ids.size();
        
        // Use model's vocab size for sampling
        uint32_t vocab_size = model_config.vocab_size();
        int32_t eos_id = use_raw_tokens ? 2 : tokenizer->eos_token_id().value_or(2);
        
        auto gen_start = std::chrono::high_resolution_clock::now();
        
        for (int step = 0; step < max_tokens; step++) {
            auto* logits_info = gb.get_logits(engine);
            if (!logits_info) {
                throw std::runtime_error("Failed to get logits");
            }
            
            // Debug: print top 5 logits
            uint32_t sample_vocab_local = std::min(vocab_size, static_cast<uint32_t>(logits_info->shape.back()));
            
            if (step < 3) {
                std::vector<half> logits_host(sample_vocab_local);
                cudaMemcpy(logits_host.data(), logits_info->ptr, 
                          sample_vocab_local * sizeof(half), cudaMemcpyDeviceToHost);
                
                // Find top 5
                std::vector<std::pair<float, int>> scored;
                for (int i = 0; i < (int)sample_vocab_local; i++) {
                    scored.push_back({__half2float(logits_host[i]), i});
                }
                std::partial_sort(scored.begin(), scored.begin() + 5, scored.end(),
                                 [](auto& a, auto& b) { return a.first > b.first; });
                
                std::cerr << "\n[Step " << step << "] Top 5 logits: ";
                for (int i = 0; i < 5; i++) {
                    std::cerr << scored[i].second << "(" << scored[i].first << ") ";
                }
                std::cerr << std::endl;
            }
            
            // Sample
            
            int32_t next_token = sampler.sample(
                (const half*)logits_info->ptr,
                sample_vocab_local,
                sample_config,
                nullptr
            );
            
            // Decode and print
            if (use_raw_tokens) {
                std::cout << "[" << next_token << "]" << std::flush;
            } else {
                std::string token_str = tokenizer->decode({next_token});
                std::cout << token_str << std::flush;
                if (debug_tokens && step < 10) {
                    std::cerr << "[" << next_token << ":" << token_str << "]";
                }
            }
            
            generated_ids.push_back(next_token);
            
            // Check EOS
            if (next_token == eos_id || next_token == 0) {
                break;
            }
            
            // Forward for next token with KV cache
            auto* input_info = engine.tensors().get("input_tokens");
            
            // Update tensor shape for single token generation
            input_info->shape = {1, 1};  // [batch=1, seq=1]
            
            cudaMemcpy(input_info->ptr, &next_token, sizeof(int32_t), cudaMemcpyHostToDevice);
            
            auto cb = gb.build_forward_cached(
                engine, model_config, arch,
                "input_tokens", 1, 1,
                kv_prefix, position, kv_config.max_seq_len
            );
            engine.execute(cb);
            engine.sync();
            
            position++;
        }
        
        auto gen_end = std::chrono::high_resolution_clock::now();
        auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count();
        
        std::cout << std::endl << std::endl;
        
        // Stats
        int tokens_generated = generated_ids.size() - prompt_ids.size();
        float tokens_per_sec = tokens_generated * 1000.0f / std::max(1L, gen_ms);
        
        std::cout << "============================================" << std::endl;
        std::cout << "GENERATION COMPLETE" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "Modal: " << modal << " (" << model_config.arch() << ")" << std::endl;
        std::cout << "Prompt tokens: " << prompt_ids.size() << std::endl;
        std::cout << "Generated tokens: " << tokens_generated << std::endl;
        std::cout << "Prefill: " << prefill_ms << " ms" << std::endl;
        std::cout << "Generation: " << gen_ms << " ms" << std::endl;
        std::cout << "Speed: " << tokens_per_sec << " tok/s" << std::endl;
        print_vram();
        
        // Print profiling info
        std::cout << "\n>>> Kernel Profile:" << std::endl;
        engine.print_profile_summary();
        
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
