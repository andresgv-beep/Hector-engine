// Prueba aislada de relevo secuencial TEXT -> CODE -> TEXT dentro de un HNF.
// Un solo proceso y un solo Engine. Los pesos cambian de residencia, pero el
// KV de TEXT permanece vivo mientras CODE trabaja.

#include "engine.hpp"
#include "graph_builder.hpp"
#include "hnf_loader.hpp"
#include "kv_cache.hpp"
#include "sampler.hpp"
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using namespace helios;

struct KvState {
    KVCache cache;
    KVCacheConfig config;
    std::string prefix;
    uint32_t position = 0;
    bool registered = false;
};

struct StageResult {
    std::string text;
    int generated_tokens = 0;
    double load_ms = 0.0;
    double prefill_ms = 0.0;
    double generation_ms = 0.0;
    uint32_t kv_before = 0;
    uint32_t kv_after = 0;
};

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

size_t used_vram_mib() {
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return (total_bytes - free_bytes) / 1024 / 1024;
}

std::string chatml_first(const std::string& system, const std::string& user) {
    return "<|im_start|>system\n" + system + "<|im_end|>\n" +
           "<|im_start|>user\n" + user + "<|im_end|>\n" +
           "<|im_start|>assistant\n";
}

std::string chatml_continue(const std::string& user) {
    // La sesión anterior ya terminó de forma natural o fue cerrada de forma
    // explícita al alcanzar su presupuesto. Solo empieza el turno siguiente.
    return "<|im_start|>user\n" + user + "<|im_end|>\n" +
           "<|im_start|>assistant\n";
}

std::string visible_only(const std::string& raw) {
    // Qwen3 puede entregar deliberación entre <think> y </think>. La prueba
    // muestra el estado "pensando", no el contenido interno.
    std::string out;
    size_t pos = 0;
    while (pos < raw.size()) {
        size_t begin = raw.find("<think>", pos);
        if (begin == std::string::npos) {
            out.append(raw, pos, std::string::npos);
            break;
        }
        out.append(raw, pos, begin - pos);
        size_t end = raw.find("</think>", begin + 7);
        if (end == std::string::npos) break;
        pos = end + 8;
    }
    while (!out.empty() && (out.front() == '\n' || out.front() == ' ')) out.erase(out.begin());
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
    return out;
}

void register_kv(Engine& engine, KvState& state, const ModelConfig& config,
                 const ArchDescriptor& arch, uint32_t max_seq_len) {
    if (state.registered) return;
    state.config.num_layers = arch.num_layers;
    state.config.num_kv_heads = config.num_key_value_heads();
    state.config.head_dim = config.head_dim();
    state.config.max_batch_size = 1;
    state.config.max_seq_len = max_seq_len;
    if (!state.cache.allocate(state.config)) {
        throw std::runtime_error("no pude reservar el KV de " + state.prefix);
    }
    for (uint32_t layer = 0; layer < arch.num_layers; ++layer) {
        const std::string base = state.prefix + ".layer" + std::to_string(layer);
        const std::vector<uint32_t> shape = {
            1, max_seq_len, state.config.num_kv_heads, state.config.head_dim};
        engine.tensors().register_external(base + ".k", state.cache.k_cache(layer),
                                           shape, dtype::FP16());
        engine.tensors().register_external(base + ".v", state.cache.v_cache(layer),
                                           shape, dtype::FP16());
    }
    state.registered = true;
}

void unregister_kv(Engine& engine, KvState& state) {
    if (!state.registered) return;
    for (uint32_t layer = 0; layer < state.config.num_layers; ++layer) {
        const std::string base = state.prefix + ".layer" + std::to_string(layer);
        engine.tensors().remove(base + ".k");
        engine.tensors().remove(base + ".v");
    }
    state.registered = false;
    state.position = 0;
    state.cache.free();
}

StageResult run_stage(Engine& engine, HnfLoader& loader, BlockID block,
                      const std::string& domain, const std::string& prefix,
                      const std::string& formatted_prompt, int max_tokens,
                      KvState& state, bool preserve_kv) {
    StageResult result;
    result.kv_before = state.position;
    const auto load_start = Clock::now();
    if (!loader.load_block(block, engine)) {
        throw std::runtime_error("no pude cargar el bloque " + domain +
                                 ": " + loader.last_error());
    }
    result.load_ms = elapsed_ms(load_start);

    const ModelConfig& config = loader.config_for_block(block);
    const HTFTokenizer* tokenizer = loader.tokenizer(domain);
    if (!tokenizer) throw std::runtime_error("falta tokenizer " + domain);

    GraphBuilder graph;
    ArchDescriptor arch = graph.detect_architecture(engine, prefix, config);
    const std::string validation = graph.validate_weights(engine, config, arch);
    if (!validation.empty()) throw std::runtime_error(validation);

    register_kv(engine, state, config, arch, 2048);
    graph.allocate_scratch(engine, config, arch, 1, 1);
    engine.tensors().allocate_and_register("relay.input", {1, 1}, dtype::INT32());

    std::vector<int32_t> ids = tokenizer->encode(formatted_prompt, false, false);
    if (state.position == 0) {
        const int32_t bos = tokenizer->bos_token_id().value_or(1);
        ids.insert(ids.begin(), bos);
    }
    if (state.position + ids.size() + size_t(max_tokens) >= state.config.max_seq_len) {
        throw std::runtime_error("el turno supera el contexto de la prueba");
    }

    Sampler sampler;
    const SamplingConfig sampling = SamplingConfig::greedy();
    auto forward = [&](int32_t token) {
        TensorInfo* input = engine.tensors().get("relay.input");
        input->shape = {1, 1};
        cudaMemcpy(input->ptr, &token, sizeof(token), cudaMemcpyHostToDevice);
        CommandBuffer cb = graph.build_forward_cached(
            engine, config, arch, "relay.input", 1, 1, state.prefix,
            state.position, state.config.max_seq_len);
        engine.execute(cb);
        engine.sync();
        ++state.position;
    };

    const auto prefill_start = Clock::now();
    for (int32_t id : ids) forward(id);
    result.prefill_ms = elapsed_ms(prefill_start);

    const int32_t eos = tokenizer->eos_token_id().value_or(2);
    const int32_t turn_end = tokenizer->token_to_id("<|im_end|>").value_or(eos);
    bool natural_stop = false;
    const auto generation_start = Clock::now();
    for (int step = 0; step < max_tokens; ++step) {
        TensorInfo* logits = graph.get_logits(engine);
        if (!logits) throw std::runtime_error("el grafo no produjo logits");
        const uint32_t vocab = std::min(config.vocab_size(), logits->shape.back());
        const int32_t next = sampler.sample(
            static_cast<const half*>(logits->ptr), vocab, sampling, nullptr);
        if (next == eos || next == turn_end || next == 0) {
            natural_stop = true;
            break;
        }
        result.text += tokenizer->decode({next});
        ++result.generated_tokens;
        forward(next);
    }
    // Si el presupuesto cortó la respuesta, deja el KV en una frontera de
    // turno válida antes de descargar los pesos. La reanudación de TEXT no
    // debe depender de un token terminal que nunca llegó a muestrearse.
    if (!natural_stop) {
        forward(turn_end);
        for (int32_t newline : tokenizer->encode("\n", false, false)) forward(newline);
    }
    result.generation_ms = elapsed_ms(generation_start);
    result.kv_after = state.position;

    engine.invalidate_graph();
    graph.free_scratch(engine);
    engine.tensors().remove("relay.input");
    if (!loader.unload_block(block, engine)) {
        throw std::runtime_error("no pude descargar el bloque " + domain);
    }
    if (!preserve_kv) unregister_kv(engine, state);
    return result;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <text-code.hnf> [petición]\n";
        return 1;
    }
    const std::string hnf = argv[1];
    std::string request;
    if (argc >= 3) {
        request = argv[2];
    } else {
        std::cout << "TÚ › " << std::flush;
        std::getline(std::cin, request);
    }
    if (request.empty()) {
        std::cerr << "La petición está vacía.\n";
        return 1;
    }

    try {
        EngineConfig cfg;
        cfg.scratch_pool.pool_size_bytes = 64 * 1024 * 1024;
        cfg.scratch_pool.auto_fraction = 0.0f;
        Engine engine(cfg);
        kernels::register_all_kernels(engine);

        HnfLoader loader;
        if (!loader.open(hnf)) throw std::runtime_error("no pude abrir el HNF");
        if (!loader.has_block(BLOCK_TEXT_MODEL) || !loader.has_block(BLOCK_CODE_EXEC)) {
            throw std::runtime_error("el HNF necesita TEXT (0x0) y CODE_EXEC (0x8)");
        }

        KvState text_kv;
        text_kv.prefix = "relay.text_kv";
        KvState code_kv;
        code_kv.prefix = "relay.code_kv";

        std::cout << "\n⟳ pensando con TEXT…" << std::flush;
        const std::string planning_prompt = chatml_first(
            "Interpreta peticiones de programación. Devuelve únicamente una "
            "especificación breve para otro modelo programador: lenguaje, entradas, "
            "salida y restricciones. No escribas el código.",
            request + "\n/no_think");
        StageResult plan = run_stage(engine, loader, BLOCK_TEXT_MODEL, "text", "text",
                                     planning_prompt, 96, text_kv, true);
        std::string brief = visible_only(plan.text);
        if (brief.empty()) brief = request;
        std::cout << " listo (" << plan.generated_tokens << " tok, carga "
                  << int(plan.load_ms) << " ms)\n";
        std::cout << "⇄ TEXT descargado; KV conservado en posición "
                  << plan.kv_after << " · " << used_vram_mib() << " MiB usados\n";

        std::cout << "⚙ ejecutando bloque CODE…" << std::flush;
        const std::string coder_prompt = chatml_first(
            "Eres el especialista de código. Resuelve la especificación con código "
            "completo, sencillo y correcto. Devuelve el código en un bloque Markdown "
            "y como máximo una explicación breve.", brief);
        StageResult code = run_stage(engine, loader, BLOCK_CODE_EXEC, "code", "code",
                                     coder_prompt, 256, code_kv, false);
        std::string generated_code = visible_only(code.text);
        std::cout << " listo (" << code.generated_tokens << " tok, carga "
                  << int(code.load_ms) << " ms, "
                  << int(code.generated_tokens * 1000.0 /
                         std::max(1.0, code.generation_ms)) << " tok/s)\n";

        std::cout << "⇄ volviendo a TEXT…" << std::flush;
        const std::string final_prompt = chatml_continue(
            "El bloque CODE ha producido la solución que aparece más abajo. Redacta "
            "una introducción útil y muy breve para la persona. No repitas ni "
            "reescribas el código; se adjuntará literalmente después.\n\nSOLUCIÓN CODE:\n" +
            generated_code + "\n/no_think");
        StageResult final_text = run_stage(
            engine, loader, BLOCK_TEXT_MODEL, "text", "text", final_prompt, 96,
            text_kv, true);
        std::cout << " listo (carga " << int(final_text.load_ms)
                  << " ms, KV TEXT " << final_text.kv_before << "→"
                  << final_text.kv_after << ")\n\n";

        std::cout << "HÉCTOR › " << visible_only(final_text.text) << "\n\n";
        std::cout << generated_code << "\n";

        unregister_kv(engine, text_kv);
        std::cout << "\n[relevo TEXT→CODE→TEXT completado en un solo proceso]\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n";
        return 1;
    }
}
