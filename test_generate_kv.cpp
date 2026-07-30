// test_generate_kv.cpp
// ============================================================================
// HELIOS TEXT GENERATION WITH KV CACHE
// ============================================================================

#include "src/engine.hpp"
#include "src/hnf_loader.hpp"
#include "src/graph_builder.hpp"
#include "src/sampler.hpp"
#include "src/hexos_bridge.hpp"
#include "src/kv_cache.hpp"
#include "kernels/kernels.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

using namespace helios;

// Registrar KV cache como tensores en el engine
void register_kv_cache(Engine& engine, KVCache& cache, const std::string& prefix) {
    auto& cfg = cache.config();
    
    for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
        std::string k_name = prefix + ".layer" + std::to_string(layer) + ".k";
        std::string v_name = prefix + ".layer" + std::to_string(layer) + ".v";
        
        TensorInfo k_info;
        k_info.ptr = cache.k_cache(layer);
        k_info.shape = {cfg.max_batch_size, cfg.max_seq_len, cfg.num_kv_heads, cfg.head_dim};
        k_info.dtype = dtype::FP16();
        k_info.size_bytes = cfg.max_batch_size * cfg.max_seq_len * cfg.num_kv_heads * cfg.head_dim * sizeof(half);
        k_info.owns_memory = false;  // KVCache owns it
        
        TensorInfo v_info = k_info;
        v_info.ptr = cache.v_cache(layer);
        
        engine.tensors().register_tensor(k_name, k_info);
        engine.tensors().register_tensor(v_name, v_info);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.hnf> [prompt] [max_tokens] [temperature]" << std::endl;
        std::cerr << "   or: " << argv[0] << " <model.hnf> [max_tokens] [temperature]  (sin prompt)" << std::endl;
        return 1;
    }

    std::string hnf_path = argv[1];

    // argv[2] puede ser prompt (texto) o max_tokens (número) — se detecta
    std::string prompt;
    int arg_idx = 2;
    if (argc > 2) {
        char* endp = nullptr;
        long v = strtol(argv[2], &endp, 10);
        (void)v;
        if (endp && *endp != '\0') {
            prompt = argv[2];   // no es número puro → es prompt
            arg_idx = 3;
        }
    }
    int max_tokens = argc > arg_idx ? std::atoi(argv[arg_idx]) : 100;
    float temperature = argc > arg_idx + 1 ? std::atof(argv[arg_idx + 1]) : 0.0f;
    
    std::cout << "============================================" << std::endl;
    std::cout << "HELIOS Text Generation (with KV Cache)" << std::endl;
    std::cout << "============================================" << std::endl;
    
    try {
        // 1. Setup engine
        // Stream dedicado: el stream legacy (nullptr) no admite CUDA Graph capture
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        EngineConfig config;
        config.scratch_pool.pool_size_bytes = 0;
        config.scratch_pool.auto_fraction = 0.0f;
        config.stream = stream;
        Engine engine(config);
        kernels::register_all_kernels(engine);
        
        // 2. Load model
        std::cout << ">>> Loading model..." << std::endl;
        HnfLoader loader;
        if (!loader.open(hnf_path)) {
            throw std::runtime_error("Failed to open HNF");
        }
        
        auto model_config = loader.config();
        std::cout << "  Layers: " << model_config.num_hidden_layers() << std::endl;
        std::cout << "  KV Heads: " << model_config.num_key_value_heads() << std::endl;
        std::cout << "  Head dim: " << model_config.head_dim() << std::endl;
        
        if (!loader.load_block(BLOCK_TEXT_MODEL, engine)) {
            throw std::runtime_error("Failed to load text model block");
        }
        
        // 3. Setup KV cache
        std::cout << ">>> Allocating KV cache..." << std::endl;
        KVCacheConfig kv_config;
        kv_config.num_layers = model_config.num_hidden_layers();
        kv_config.num_kv_heads = model_config.num_key_value_heads();
        kv_config.head_dim = model_config.head_dim();
        kv_config.max_batch_size = 1;
        kv_config.max_seq_len = 4096;  // 576 MB de KV para Qwen3-4B (36L × 8KVH × 128HD)
        
        KVCache kv_cache;
        if (!kv_cache.allocate(kv_config)) {
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            std::cerr << "KV cache allocation failed" << std::endl;
            std::cerr << "  Requested: " << kv_config.total_bytes() / 1024 / 1024 << " MB" << std::endl;
            std::cerr << "  Free VRAM: " << free_mem / 1024 / 1024 << " MB" << std::endl;
            throw std::runtime_error("Failed to allocate KV cache");
        }
        
        std::cout << "  KV cache: " << kv_config.total_bytes() / 1024 / 1024 << " MB" << std::endl;
        
        // Register KV cache tensors in engine
        std::string kv_prefix = "_kv";
        register_kv_cache(engine, kv_cache, kv_prefix);
        
        // 4. Setup
        GraphBuilder gb;
        auto arch = gb.detect_architecture(engine, "text");
        gb.allocate_scratch(engine, model_config, arch, 1, 1);
        
        Sampler sampler;
        SamplingConfig sample_config = temperature < 0.01f 
            ? SamplingConfig::greedy()
            : SamplingConfig::creative(temperature, 50, 0.9f);
        
        engine.tensors().allocate_and_register("input_tokens", {1, 1}, dtype::INT32());
        
        // 5. Tokenizer embebido en el HNF: prompt y salida en texto
        const HTFTokenizer* tokenizer = loader.tokenizer("text");

        std::vector<int32_t> prompt_ids;
        if (!prompt.empty() && tokenizer) {
            // Modo raw explícito: prefijo "raw:" → continuación pura sin plantilla
            bool raw_mode = prompt.rfind("raw:", 0) == 0;
            std::string user_text = raw_mode ? prompt.substr(4) : prompt;

            auto im_start = tokenizer->token_to_id("<|im_start|>");
            auto im_end   = tokenizer->token_to_id("<|im_end|>");

            if (!raw_mode && im_start && im_end) {
                // Plantilla ChatML (Qwen instruct): los marcadores van como ids
                // especiales directos — nunca por BPE, que los despedaza.
                auto push_text = [&](const std::string& s) {
                    auto seg = tokenizer->encode(s, false, false);
                    prompt_ids.insert(prompt_ids.end(), seg.begin(), seg.end());
                };
                prompt_ids.push_back(*im_start);
                push_text("user\n" + user_text);
                prompt_ids.push_back(*im_end);
                push_text("\n");
                prompt_ids.push_back(*im_start);
                push_text("assistant\n");
            } else {
                prompt_ids = tokenizer->encode(user_text, false, false);
            }
            std::cout << "  Prompt: " << prompt_ids.size() << " tokens"
                      << (raw_mode ? " (raw)" : " (chat)") << std::endl;
        }
        if (prompt_ids.empty()) prompt_ids.push_back(1);  // arranque clásico sin prompt

        int32_t eos_id = tokenizer ? tokenizer->eos_token_id().value_or(151645) : 151645;

        std::cout << "\n>>> Generating with KV cache..." << std::endl;
        if (!prompt.empty()) {
            std::cout << "\033[36m" << prompt << "\033[0m" << std::flush;  // prompt en cyan
        }

        std::vector<int32_t> generated;
        size_t prompt_pos = 0;
        int32_t current_token = prompt_ids[0];
        int gen_count = 0;

        auto start_time = std::chrono::high_resolution_clock::now();
        auto gen_start = start_time;
        int total_steps = (int)prompt_ids.size() - 1 + max_tokens;
        
        // HEXOS bridge: si el monitor está corriendo, publicar telemetría
        // de inferencia al blackboard (/dev/shm/hexos_state). No-op si no.
        HexosBridge hexos;
        if (hexos.connect()) {
            std::cout << ">>> HEXOS detectado — publicando telemetria" << std::endl;
            hexos.update_vram_budgets(
                4096 /* aprox modelo MB; TODO: tamaño real del bloque */,
                (uint32_t)(kv_config.total_bytes() / 1024 / 1024),
                0);
        }
        uint64_t hexos_total_tokens = 0;
        auto hexos_last = std::chrono::high_resolution_clock::now();

        // FASE 1: command buffer construido UNA vez; por token solo se actualiza
        // la posición en device (4 bytes async) y se relanza el CUDA Graph.
        CommandBuffer cb;
        bool cb_built = false;

        for (int step = 0; step < total_steps; step++) {
            // Copy token
            auto* input_info = engine.tensors().get("input_tokens");
            cudaMemcpy(input_info->ptr, &current_token, sizeof(int32_t),
                       cudaMemcpyHostToDevice);

            // Posición del cache → device (leída por los kernels _dp)
            uint32_t position = kv_cache.position();
            engine.update_device_cache_pos(position, 1);

            if (!cb_built) {
                // Primer token: build + ejecución normal.
                // (calienta el auto-tune del GEMV, que sincroniza y no es capturable)
                cb = gb.build_forward_cached(
                    engine, model_config, arch,
                    "input_tokens", 1, 1,
                    kv_prefix, position, kv_config.max_seq_len
                );
                cb_built = true;
                engine.execute(cb);
                engine.sync();
            } else {
                // Resto: captura en el token 2, replay puro desde el 3
                engine.execute_graph_replay(cb);
            }

            // Advance position
            kv_cache.advance(1);

            // Sample (sincroniza el stream internamente para leer el token)
            auto* logits_info = gb.get_logits(engine);
            int32_t next_token = sampler.sample(
                (const half*)logits_info->ptr,
                model_config.vocab_size(),
                sample_config,
                stream
            );
            
            // Telemetría HEXOS: throughput instantáneo por forward
            if (hexos.connected()) {
                auto now_t = std::chrono::high_resolution_clock::now();
                float dt = std::chrono::duration<float>(now_t - hexos_last).count();
                hexos_last = now_t;
                hexos_total_tokens++;
                if (dt > 0.0f) {
                    hexos.update_inference(1.0f / dt, hexos_total_tokens, true);
                }
            }

            // ¿Seguimos consumiendo el prompt? (prefill token a token)
            if (prompt_pos + 1 < prompt_ids.size()) {
                prompt_pos++;
                current_token = prompt_ids[prompt_pos];
                if (prompt_pos + 1 == prompt_ids.size()) {
                    gen_start = std::chrono::high_resolution_clock::now();
                }
                continue;
            }

            // Token generado: texto si hay tokenizer, id crudo si no
            if (tokenizer) {
                std::cout << tokenizer->decode({next_token}) << std::flush;
            } else {
                std::cout << next_token << " " << std::flush;
            }
            generated.push_back(next_token);
            gen_count++;
            current_token = next_token;

            if (next_token == eos_id || next_token == 151643 || next_token == 2) {
                std::cout << " [EOS]";
                break;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Motor parado: dejar el blackboard limpio
        hexos.update_inference(0.0f, hexos_total_tokens, false);
        
        std::cout << std::endl << std::endl;
        
        auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - gen_start).count();
        auto prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            gen_start - start_time).count();
        float tokens_per_sec = (gen_count > 0 && gen_ms > 0)
            ? gen_count * 1000.0f / gen_ms : 0;

        std::cout << "============================================" << std::endl;
        std::cout << "Prompt: " << prompt_ids.size() << " tokens ("
                  << prefill_ms << " ms)" << std::endl;
        std::cout << "Tokens: " << gen_count << std::endl;
        std::cout << "Time: " << duration.count() << " ms" << std::endl;
        std::cout << "Speed: " << tokens_per_sec << " tok/s" << std::endl;
        std::cout << "============================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
