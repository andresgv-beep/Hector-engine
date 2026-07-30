// helios_chat.cpp
// ============================================================================
// HELIOS CHAT — Chat multi-turno con continuidad de KV cache
// ============================================================================
//
// El embrión del runtime conversacional de HELIOS. Junta todo lo probado:
//   - CUDA Graph replay (command buffer una vez, replay por token)
//   - Plantilla ChatML con tokens especiales por id (no BPE)
//   - Streaming de texto con EOS real del tokenizer
//   - Telemetría a HEXOS vía blackboard
//
// Y las dos piezas dinámicas nuevas:
//   - THINKING ADAPTATIVO (nivel 1): saludos/mensajes triviales → /no_think.
//     El usuario manda: "/think" o "/no_think" en su mensaje lo fuerzan.
//   - GOVERNOR DE RITMO: "piensa rápido, habla tranquilo" — el bloque <think>
//     corre a toda máquina; el texto visible sale a ritmo de lectura
//     (~18 tok/s), bajando consumo y temperatura en la fase dominante.
//     Comando /fast alterna a velocidad plena.
//
// CONTINUIDAD: la conversación vive en el KV cache — cada turno solo
// prefillea SU texto nuevo; la historia ya está en VRAM. El contexto no se
// reprocesa jamás (la compactación al llenarse es trabajo futuro del CK).
//

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
#include <thread>
#include <algorithm>

using namespace helios;

// ============================================================================
// KV CACHE REGISTRATION (igual que test_generate_kv)
// ============================================================================

static void register_kv_cache(Engine& engine, KVCache& cache, const std::string& prefix) {
    const auto& cfg = cache.config();
    for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
        std::string k_name = prefix + ".layer" + std::to_string(layer) + ".k";
        std::string v_name = prefix + ".layer" + std::to_string(layer) + ".v";

        TensorInfo k_info;
        k_info.ptr = cache.k_cache(layer);
        k_info.shape = {cfg.max_batch_size, cfg.max_seq_len, cfg.num_kv_heads, cfg.head_dim};
        k_info.dtype = dtype::FP16();
        k_info.size_bytes = cfg.max_batch_size * cfg.max_seq_len * cfg.num_kv_heads * cfg.head_dim * sizeof(half);
        k_info.owns_memory = false;

        TensorInfo v_info = k_info;
        v_info.ptr = cache.v_cache(layer);

        engine.tensors().register_tensor(k_name, k_info);
        engine.tensors().register_tensor(v_name, v_info);
    }
}

// ============================================================================
// THINKING ADAPTATIVO — heurística nivel 1
// ============================================================================
// Trivial (saludos, acks, mensajes muy cortos sin pregunta) → /no_think.
// Conservador a propósito: ante la duda, se deja pensar.

static bool is_trivial_message(const std::string& msg) {
    if (msg.size() > 60) return false;

    std::string lower = msg;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    // Preguntas y peticiones técnicas nunca son triviales
    if (lower.find('?') != std::string::npos) return false;
    for (const char* kw : {"por que", "por qué", "como ", "cómo ", "explica",
                           "escribe", "code", "codigo", "código", "error",
                           "ayuda", "traduce", "calcula", "cuanto", "cuánto"}) {
        if (lower.find(kw) != std::string::npos) return false;
    }

    for (const char* g : {"hola", "buenas", "buenos dias", "buenos días",
                          "hey", "que tal", "qué tal", "gracias", "ok",
                          "vale", "genial", "perfecto", "adios", "adiós",
                          "hasta luego", "jaja"}) {
        if (lower.find(g) != std::string::npos) return true;
    }

    // Muy corto y sin señales de tarea → trivial
    return msg.size() < 12;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.hnf> [temperature]" << std::endl;
        return 1;
    }
    std::string hnf_path = argv[1];
    float temperature = argc > 2 ? std::atof(argv[2]) : 0.7f;

    std::cout << "╔══════════════════════════════════════════╗" << std::endl;
    std::cout << "║  HELIOS CHAT — continuidad por KV cache  ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════╝" << std::endl;

    try {
        // --- Engine con stream dedicado (capturable) ---
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        EngineConfig config;
        config.scratch_pool.pool_size_bytes = 0;
        config.scratch_pool.auto_fraction = 0.0f;
        config.stream = stream;
        Engine engine(config);
        kernels::register_all_kernels(engine);

        // --- Modelo ---
        std::cout << ">>> Cargando modelo..." << std::endl;
        HnfLoader loader;
        if (!loader.open(hnf_path)) throw std::runtime_error("Failed to open HNF");
        auto model_config = loader.config();
        if (!loader.load_block(BLOCK_TEXT_MODEL, engine))
            throw std::runtime_error("Failed to load text model block");

        const HTFTokenizer* tokenizer = loader.tokenizer("text");
        if (!tokenizer) throw std::runtime_error("El chat necesita tokenizer embebido");

        auto im_start = tokenizer->token_to_id("<|im_start|>");
        auto im_end   = tokenizer->token_to_id("<|im_end|>");
        if (!im_start || !im_end)
            throw std::runtime_error("Modelo sin tokens ChatML — usa un instruct");
        int32_t eos_id = tokenizer->eos_token_id().value_or(151645);

        // --- KV cache ---
        KVCacheConfig kv_config;
        kv_config.num_layers = model_config.num_hidden_layers();
        kv_config.num_kv_heads = model_config.num_key_value_heads();
        kv_config.head_dim = model_config.head_dim();
        kv_config.max_batch_size = 1;
        kv_config.max_seq_len = 4096;
        KVCache kv_cache;
        if (!kv_cache.allocate(kv_config)) throw std::runtime_error("KV alloc failed");
        std::string kv_prefix = "_kv";
        register_kv_cache(engine, kv_cache, kv_prefix);

        GraphBuilder gb;
        auto arch = gb.detect_architecture(engine, "text");
        gb.allocate_scratch(engine, model_config, arch, 1, 1);

        Sampler sampler;
        SamplingConfig sample_config = temperature < 0.01f
            ? SamplingConfig::greedy()
            : SamplingConfig::creative(temperature, 50, 0.9f);

        engine.tensors().allocate_and_register("input_tokens", {1, 1}, dtype::INT32());

        HexosBridge hexos;
        if (hexos.connect()) {
            std::cout << ">>> HEXOS conectado — telemetría activa" << std::endl;
            hexos.update_vram_budgets(3400,
                (uint32_t)(kv_config.total_bytes() / 1024 / 1024), 0);
        }

        // --- Estado del chat ---
        CommandBuffer cb;
        bool cb_built = false;
        uint64_t total_tokens = 0;
        bool fast_mode = false;
        const float SPEAK_PACE_TOKS = 18.0f;   // ritmo de lectura humana
        auto hexos_last = std::chrono::high_resolution_clock::now();

        // forward de UN token: devuelve el token muestreado
        auto forward_one = [&](int32_t token) -> int32_t {
            auto* input_info = engine.tensors().get("input_tokens");
            cudaMemcpy(input_info->ptr, &token, sizeof(int32_t), cudaMemcpyHostToDevice);
            uint32_t position = kv_cache.position();
            engine.update_device_cache_pos(position, 1);

            if (!cb_built) {
                cb = gb.build_forward_cached(engine, model_config, arch,
                                             "input_tokens", 1, 1,
                                             kv_prefix, position, kv_config.max_seq_len);
                cb_built = true;
                engine.execute(cb);
                engine.sync();
            } else {
                engine.execute_graph_replay(cb);
            }
            kv_cache.advance(1);

            auto* logits_info = gb.get_logits(engine);
            int32_t next = sampler.sample((const half*)logits_info->ptr,
                                          model_config.vocab_size(),
                                          sample_config, stream);
            total_tokens++;
            if (hexos.connected()) {
                auto now_t = std::chrono::high_resolution_clock::now();
                float dt = std::chrono::duration<float>(now_t - hexos_last).count();
                hexos_last = now_t;
                if (dt > 0.0f) hexos.update_inference(1.0f / dt, total_tokens, true);
            }
            return next;
        };

        std::cout << "\nComandos: /fast (velocidad plena on/off), /salir" << std::endl;
        std::cout << "Contexto: " << kv_config.max_seq_len << " posiciones\n" << std::endl;

        std::string line;
        while (true) {
            // --- Turno del usuario ---
            std::cout << "\033[1;36mtú>\033[0m " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (line.empty()) continue;
            if (line == "/salir" || line == "/exit") break;
            if (line == "/fast") {
                fast_mode = !fast_mode;
                std::cout << "(velocidad " << (fast_mode ? "PLENA" : "tranquila") << ")\n";
                continue;
            }

            // Guardia de contexto (compactación = trabajo futuro del CK)
            if (kv_cache.position() + 600 > kv_config.max_seq_len) {
                std::cout << "\033[33m[contexto casi lleno — " << kv_cache.position()
                          << "/" << kv_config.max_seq_len
                          << ". La compactación llegará con el CK]\033[0m\n";
                if (kv_cache.position() + 100 > kv_config.max_seq_len) break;
            }

            // --- Thinking adaptativo (el usuario puede forzar con /think, /no_think) ---
            bool user_forced = line.find("/think") != std::string::npos ||
                               line.find("/no_think") != std::string::npos;
            std::string user_msg = line;
            bool trivial = false;
            if (!user_forced && is_trivial_message(line)) {
                user_msg += " /no_think";
                trivial = true;
            }

            // --- Prefill del turno (solo el texto NUEVO: la historia ya está en KV) ---
            std::vector<int32_t> turn_ids;
            auto push_text = [&](const std::string& s) {
                auto seg = tokenizer->encode(s, false, false);
                turn_ids.insert(turn_ids.end(), seg.begin(), seg.end());
            };
            turn_ids.push_back(*im_start);
            push_text("user\n" + user_msg);
            turn_ids.push_back(*im_end);
            push_text("\n");
            turn_ids.push_back(*im_start);
            push_text("assistant\n");

            auto t0 = std::chrono::high_resolution_clock::now();
            int32_t next = 0;
            for (size_t i = 0; i < turn_ids.size(); i++) {
                next = forward_one(turn_ids[i]);
            }
            auto t_prefill = std::chrono::high_resolution_clock::now();

            // --- Generación con governor de ritmo ---
            std::cout << "\033[1;35mhelios>\033[0m " << std::flush;
            int gen_count = 0, think_count = 0;
            bool in_think = false;
            const int MAX_GEN = 1500;

            while (gen_count < MAX_GEN) {
                if (next == eos_id || next == *im_end) break;

                std::string piece = tokenizer->decode({next});
                if (piece.find("<think>") != std::string::npos) in_think = true;

                if (in_think) {
                    think_count++;
                    // pensamiento invisible: solo un latido para que se vea vida
                    if (think_count == 1) std::cout << "\033[90m(pensando" << std::flush;
                    else if (think_count % 40 == 0) std::cout << "." << std::flush;
                } else {
                    std::cout << piece << std::flush;
                    // GOVERNOR: hablar a ritmo de lectura (el think va a tope)
                    if (!fast_mode) {
                        std::this_thread::sleep_for(std::chrono::microseconds(
                            (int)(1e6f / SPEAK_PACE_TOKS)));
                    }
                }

                if (piece.find("</think>") != std::string::npos) {
                    in_think = false;
                    std::cout << ")\033[0m " << std::flush;
                }

                gen_count++;
                next = forward_one(next);
                if (kv_cache.position() >= kv_config.max_seq_len - 2) break;
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            float gen_s = std::chrono::duration<float>(t1 - t_prefill).count();
            float prefill_ms = std::chrono::duration<float>(t_prefill - t0).count() * 1000;

            std::cout << "\n\033[90m[" << gen_count << " tok"
                      << (think_count ? (" (" + std::to_string(think_count) + " pensando)") : "")
                      << (trivial ? " · trivial→sin think" : "")
                      << " · prefill " << (int)prefill_ms << "ms"
                      << " · " << (gen_s > 0 ? (int)(gen_count / gen_s) : 0) << " tok/s"
                      << " · ctx " << kv_cache.position() << "/" << kv_config.max_seq_len
                      << "]\033[0m\n" << std::endl;
        }

        hexos.update_inference(0.0f, total_tokens, false);
        std::cout << "\nHasta luego. (" << total_tokens << " tokens esta sesión)" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
