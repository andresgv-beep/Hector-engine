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
        std::cerr << "Usage: " << argv[0] << " <model.hnf> [max_tokens] [temperature]" << std::endl;
        return 1;
    }
    
    std::string hnf_path = argv[1];
    int max_tokens = argc > 2 ? std::atoi(argv[2]) : 20;
    float temperature = argc > 3 ? std::atof(argv[3]) : 0.0f;
    
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
        kv_config.max_seq_len = 128;
        
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
        
        // 5. Generate with KV cache
        std::cout << "\n>>> Generating with KV cache..." << std::endl;
        std::cout << "Tokens: ";
        
        std::vector<int32_t> generated;
        int32_t current_token = 1;  // Start token
        generated.push_back(current_token);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
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

        for (int step = 0; step < max_tokens; step++) {
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
            
            std::cout << next_token << " " << std::flush;
            generated.push_back(next_token);
            current_token = next_token;

            // Telemetría HEXOS: throughput instantáneo por token
            if (hexos.connected()) {
                auto now_t = std::chrono::high_resolution_clock::now();
                float dt = std::chrono::duration<float>(now_t - hexos_last).count();
                hexos_last = now_t;
                hexos_total_tokens++;
                if (dt > 0.0f) {
                    hexos.update_inference(1.0f / dt, hexos_total_tokens, true);
                }
            }
            
            if (next_token == 2 || next_token == 151643) {
                std::cout << "[EOS]";
                break;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Motor parado: dejar el blackboard limpio
        hexos.update_inference(0.0f, hexos_total_tokens, false);
        
        std::cout << std::endl << std::endl;
        
        int tokens_generated = generated.size() - 1;
        float tokens_per_sec = tokens_generated > 0 ? tokens_generated * 1000.0f / duration.count() : 0;
        
        std::cout << "============================================" << std::endl;
        std::cout << "Tokens: " << tokens_generated << std::endl;
        std::cout << "Time: " << duration.count() << " ms" << std::endl;
        std::cout << "Speed: " << tokens_per_sec << " tok/s" << std::endl;
        std::cout << "============================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
