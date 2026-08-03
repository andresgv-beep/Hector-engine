#include "inference_session.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "engine.hpp"
#include "graph_builder.hpp"
#include "gemma4_kv_cache.hpp"
#include "hnf_loader.hpp"
#include "htf_tokenizer.hpp"
#include "kv_cache.hpp"
#include "model_capabilities.hpp"
#include "sampler.hpp"
#include "../kernels/kernels.hpp"

namespace helios {
namespace {

constexpr uint32_t kPrefillChunk = 512;

// Un fragmento solo se emite cuando es UTF-8 completo. El protocolo lo mete
// en una cadena JSON, y un multibyte partido la invalida — ya nos mordió una
// vez en la UI con "cr??ticas".
size_t utf8_complete_prefix(const std::string& s) {
    size_t n = s.size();
    if (n == 0) return 0;
    size_t i = n;
    // Retrocede hasta el inicio del último carácter.
    while (i > 0 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80) i--;
    if (i == 0) return 0;
    unsigned char lead = static_cast<unsigned char>(s[i - 1]);
    size_t len = (lead < 0x80) ? 1
               : ((lead & 0xE0) == 0xC0) ? 2
               : ((lead & 0xF0) == 0xE0) ? 3
               : ((lead & 0xF8) == 0xF0) ? 4 : 1;
    return (i - 1 + len <= n) ? n : i - 1;
}

float env_f(const char* k, float def) {
    const char* v = getenv(k);
    return v ? static_cast<float>(atof(v)) : def;
}
int env_i(const char* k, int def) {
    const char* v = getenv(k);
    return v ? atoi(v) : def;
}

}  // namespace

const char* InferenceSession::finish_reason_name(FinishReason r) {
    switch (r) {
        case FinishReason::Eos:       return "eos";
        case FinishReason::MaxTokens: return "max_tokens";
        case FinishReason::Stop:      return "stop";
        case FinishReason::Cancelled: return "cancelled";
    }
    return "stop";
}

struct InferenceSession::Impl {
    EngineConfig engine_config;
    std::unique_ptr<Engine> engine;
    HnfLoader loader;
    const HTFTokenizer* tokenizer = nullptr;
    ModelConfig model_config;
    GraphBuilder gb;
    ArchDescriptor arch{};
    KVCacheConfig kv_config;
    KVCache kv_cache;
    Gemma4KVCache gemma_kv_cache;
    Sampler sampler;
    SamplingConfig sample_config;
    ModelInfo info;

    bool is_gemma4 = false;
    std::string kv_prefix = "_kv";
    int32_t turn_start = 0, turn_end = 0, eos_id = 0;
    std::optional<int32_t> think_open, think_close;

    CommandBuffer decode_cb;
    bool decode_cb_built = false;

    uint32_t position() const {
        return is_gemma4 ? gemma_kv_cache.position() : kv_cache.position();
    }
    void advance(uint32_t n) {
        if (is_gemma4) gemma_kv_cache.advance(n); else kv_cache.advance(n);
    }
    void rewind(uint32_t p) {
        if (is_gemma4) gemma_kv_cache.rewind_to(p); else kv_cache.rewind_to(p);
    }
    void clear() {
        if (is_gemma4) gemma_kv_cache.reset(); else kv_cache.reset();
    }

    // Codifica los mensajes NUEVOS como fragmento incremental: la historia ya
    // está en KV, así que la plantilla completa (con su BOS) solo vale para el
    // primer prefill de la sesión.
    std::vector<int32_t> encode(const std::vector<ChatMessage>& messages,
                                std::string* error_code, std::string* error);

    int32_t forward_batch(const std::vector<int32_t>& ids, cudaStream_t stream);
    int32_t forward_one(int32_t token, cudaStream_t stream);
};

std::vector<int32_t> InferenceSession::Impl::encode(
        const std::vector<ChatMessage>& messages,
        std::string* error_code, std::string* error) {
    if (is_gemma4) {
        // Gemma 4 no admite un `system` nuevo entre turnos: su plantilla lo
        // fusiona con el primer user. En vez de inventarnos un encaje, se
        // rechaza y que E2 decida cómo entregar lo efímero.
        if (position() > 0) {
            for (const auto& m : messages) {
                if (m.role == "system") {
                    *error_code = "unsupported_role_sequence";
                    *error = "Gemma 4 no admite un mensaje 'system' entre "
                             "turnos; solo al inicio de la sesión";
                    return {};
                }
            }
        }
        Gemma4ChatOptions options;
        options.add_generation_prompt = true;
        auto ids = tokenizer->encode(format_gemma4_chat(messages, options),
                                     false, false);
        const int32_t bos = tokenizer->bos_token_id().value_or(2);
        if (position() > 0 && !ids.empty() && ids.front() == bos) {
            ids.erase(ids.begin());   // el BOS ya está en el KV
        }
        return ids;
    }

    std::vector<int32_t> ids;
    auto push = [&](const std::string& s) {
        auto seg = tokenizer->encode(s, false, false);
        ids.insert(ids.end(), seg.begin(), seg.end());
    };
    for (const auto& m : messages) {
        ids.push_back(turn_start);
        push(m.role + "\n" + m.content);
        ids.push_back(turn_end);
        push("\n");
    }
    ids.push_back(turn_start);
    push("assistant\n");
    return ids;
}

int32_t InferenceSession::Impl::forward_one(int32_t token, cudaStream_t stream) {
    auto* input_info = engine->tensors().get("input_tokens");
    input_info->shape = {1, 1};
    cudaMemcpy(input_info->ptr, &token, sizeof(int32_t), cudaMemcpyHostToDevice);
    uint32_t pos = position();
    engine->update_device_cache_pos(pos, 1);

    if (!decode_cb_built) {
        if (is_gemma4) {
            const KVCacheParams params{kv_prefix, pos, kv_config.max_seq_len};
            decode_cb = gb.build_gemma4_forward_cached(
                *engine, model_config, loader.gemma4_config(), arch,
                "input_tokens", 1, 1, params);
        } else {
            decode_cb = gb.build_forward_cached(
                *engine, model_config, arch, "input_tokens", 1, 1,
                kv_prefix, pos, kv_config.max_seq_len);
        }
        decode_cb_built = true;
        engine->execute(decode_cb);
        engine->sync();
    } else {
        engine->execute_graph_replay(decode_cb);
    }
    advance(1);
    auto* logits = gb.get_logits(*engine);
    return sampler.sample((const half*)logits->ptr, model_config.vocab_size(),
                          sample_config, stream);
}

int32_t InferenceSession::Impl::forward_batch(const std::vector<int32_t>& ids,
                                              cudaStream_t stream) {
    int32_t next = 0;
    size_t done = 0;
    while (done < ids.size()) {
        size_t n = std::min(static_cast<size_t>(kPrefillChunk), ids.size() - done);
        if (n == 1) { next = forward_one(ids[done], stream); done += 1; continue; }

        auto* input_info = engine->tensors().get("input_tokens");
        input_info->shape = {1, static_cast<uint32_t>(n)};
        cudaMemcpy(input_info->ptr, ids.data() + done, n * sizeof(int32_t),
                   cudaMemcpyHostToDevice);
        uint32_t pos = position();
        CommandBuffer pcb;
        if (is_gemma4) {
            const KVCacheParams params{kv_prefix, pos, kv_config.max_seq_len};
            pcb = gb.build_gemma4_forward_cached(
                *engine, model_config, loader.gemma4_config(), arch,
                "input_tokens", 1, static_cast<uint32_t>(n), params);
        } else {
            pcb = gb.build_forward_cached(
                *engine, model_config, arch, "input_tokens", 1,
                static_cast<uint32_t>(n), kv_prefix, pos, kv_config.max_seq_len);
        }
        engine->execute(pcb);
        engine->sync();
        advance(static_cast<uint32_t>(n));
        auto* logits = gb.get_logits(*engine);
        next = sampler.sample((const half*)logits->ptr, model_config.vocab_size(),
                              sample_config, stream);
        done += n;
    }
    return next;
}

InferenceSession::InferenceSession() : impl_(std::make_unique<Impl>()) {}
InferenceSession::~InferenceSession() = default;

const InferenceSession::ModelInfo& InferenceSession::info() const {
    return impl_->info;
}
uint32_t InferenceSession::cache_position() const { return impl_->position(); }

void InferenceSession::reset() {
    impl_->clear();
    impl_->sampler.clear_context();
}

bool InferenceSession::load(const Config& config, std::string* error) {
    auto& s = *impl_;
    try {
        s.engine = std::make_unique<Engine>(s.engine_config);
        kernels::register_all_kernels(*s.engine);

        if (!s.loader.open(config.hnf_path)) {
            *error = "no pude abrir el HNF";
            return false;
        }
        s.model_config = s.loader.config();
        if (!s.loader.load_block(BLOCK_TEXT_MODEL, *s.engine)) {
            *error = "no pude cargar el bloque de texto";
            return false;
        }
        s.tokenizer = s.loader.tokenizer("text");
        if (!s.tokenizer) {
            *error = "el HNF no trae tokenizer embebido";
            return false;
        }
        s.is_gemma4 = s.loader.has_gemma4_config();

        auto ts = s.tokenizer->token_to_id(s.is_gemma4 ? "<|turn>" : "<|im_start|>");
        auto te = s.tokenizer->token_to_id(s.is_gemma4 ? "<turn|>" : "<|im_end|>");
        if (!ts || !te) {
            *error = "el modelo no trae tokens de turno canónicos";
            return false;
        }
        s.turn_start = *ts;
        s.turn_end = *te;
        s.eos_id = s.tokenizer->eos_token_id().value_or(s.is_gemma4 ? 1 : 151645);
        if (!s.is_gemma4) {
            s.think_open = s.tokenizer->token_to_id("<think>");
            s.think_close = s.tokenizer->token_to_id("</think>");
        }

        s.kv_config.num_layers = s.model_config.num_hidden_layers();
        s.kv_config.num_kv_heads = s.model_config.num_key_value_heads();
        s.kv_config.head_dim = s.model_config.head_dim();
        s.kv_config.max_batch_size = 1;
        s.kv_config.max_seq_len = config.max_seq_len;
        if (s.is_gemma4) {
            if (!s.gemma_kv_cache.allocate(s.loader.gemma4_config(),
                                           s.model_config.num_key_value_heads(),
                                           1, s.kv_config.max_seq_len)) {
                *error = "no pude reservar el KV heterogéneo de Gemma 4";
                return false;
            }
            s.gemma_kv_cache.register_tensors(*s.engine, s.kv_prefix);
        } else {
            if (!s.kv_cache.allocate(s.kv_config)) {
                *error = "no pude reservar el KV";
                return false;
            }
            // Nombres y forma EXACTOS del oráculo: el grafo los busca por
            // "<prefijo>.layerN.k" y con esta disposición concreta.
            const auto& c = s.kv_cache.config();
            for (uint32_t l = 0; l < c.num_layers; l++) {
                TensorInfo k_info;
                k_info.ptr = s.kv_cache.k_cache(l);
                k_info.shape = {c.max_batch_size, c.max_seq_len,
                                c.num_kv_heads, c.head_dim};
                k_info.dtype = dtype::FP16();
                k_info.size_bytes = (size_t)c.max_batch_size * c.max_seq_len *
                                    c.num_kv_heads * c.head_dim * sizeof(half);
                k_info.owns_memory = false;
                TensorInfo v_info = k_info;
                v_info.ptr = s.kv_cache.v_cache(l);
                const std::string base = s.kv_prefix + ".layer" + std::to_string(l);
                s.engine->tensors().register_tensor(base + ".k", k_info);
                s.engine->tensors().register_tensor(base + ".v", v_info);
            }
        }

        s.arch = s.gb.detect_architecture(*s.engine, "text");
        s.engine->tensors().allocate_and_register(
            "input_tokens", {1, kPrefillChunk}, dtype::INT32());
        if (s.is_gemma4) {
            s.gb.allocate_gemma4_scratch(*s.engine, s.model_config,
                                         s.loader.gemma4_config(), s.arch,
                                         1, kPrefillChunk);
        } else {
            s.gb.allocate_scratch(*s.engine, s.model_config, s.arch,
                                  1, kPrefillChunk);
        }

        // Mismos valores por defecto que el oráculo: E1 no cambia conducta.
        const float rep  = env_f("HELIOS_REP", s.is_gemma4 ? 1.0f : 1.15f);
        const float freq = env_f("HELIOS_FREQ", s.is_gemma4 ? 0.0f : 0.1f);
        const int   win  = env_i("HELIOS_WINDOW", 384);
        const int   topk = env_i("HELIOS_TOPK", s.is_gemma4 ? 64 : 50);
        const float topp = env_f("HELIOS_TOPP", s.is_gemma4 ? 0.95f : 0.9f);
        const int   seed = env_i("HELIOS_SEED", 0);
        s.sampler.set_penalty_window(win);
        if (seed) s.sampler.set_seed(static_cast<uint64_t>(seed));
        s.sample_config = config.temperature < 0.01f
            ? SamplingConfig::greedy()
            : SamplingConfig::creative(config.temperature, topk, topp);
        s.sample_config.repetition_penalty = rep;
        s.sample_config.frequency_penalty = freq;

        const ModelCapabilities caps = inspect_model_capabilities(s.loader);
        const auto* vision = caps.find(ModelModality::Vision);
        s.info.architecture = caps.text_architecture;
        s.info.max_seq_len = s.kv_config.max_seq_len;
        s.info.multimodal = caps.multimodal;
        if (vision && vision->status == AdapterStatus::RuntimeReady) {
            s.info.vision_adapter = vision->adapter_id;
        }
        return true;
    } catch (const std::exception& e) {
        *error = e.what();
        return false;
    }
}

bool InferenceSession::run_turn(const std::vector<ChatMessage>& messages,
                                const GenConfig& gen,
                                const TextCallback& on_text,
                                const ThinkingCallback& on_thinking,
                                const std::atomic<bool>& cancel_flag,
                                TurnStats* stats,
                                FinishReason* reason,
                                std::string* error_code,
                                std::string* error) {
    auto& s = *impl_;
    using clock = std::chrono::high_resolution_clock;

    stats->cache_position_before = s.position();
    stats->cache_position = stats->cache_position_before;

    auto ids = s.encode(messages, error_code, error);
    if (ids.empty()) {
        if (error_code->empty()) {
            *error_code = "empty_turn";
            *error = "los mensajes no produjeron tokens";
        }
        return false;
    }
    if (s.position() + ids.size() + 8 >= s.kv_config.max_seq_len) {
        // Antes de emitir nada: el KV se queda como estaba (§4).
        *error_code = "context_full";
        *error = "el turno no cabe en el contexto restante";
        return false;
    }

    s.sample_config.temperature = gen.temperature;
    if (gen.temperature < 0.01f) s.sample_config = SamplingConfig::greedy();

    cudaStream_t stream = s.engine_config.stream;
    auto t0 = clock::now();
    int32_t next = s.forward_batch(ids, stream);
    auto t1 = clock::now();
    stats->prefill_tokens = static_cast<uint32_t>(ids.size());
    stats->prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::string visible, pending;
    bool in_think = false;
    uint32_t visible_tokens = 0, thinking = 0, generated = 0;
    *reason = FinishReason::Eos;

    const int hard_cap = gen.max_visible_tokens + gen.max_thinking_tokens + 64;
    while (generated < static_cast<uint32_t>(hard_cap)) {
        if (cancel_flag.load(std::memory_order_relaxed)) {
            *reason = FinishReason::Cancelled;
            break;
        }
        if (next == s.eos_id || next == s.turn_end) {
            *reason = FinishReason::Eos;
            break;
        }
        if (s.position() + 2 >= s.kv_config.max_seq_len) {
            *reason = FinishReason::Stop;
            break;
        }

        if (s.think_open && next == *s.think_open) in_think = true;

        if (in_think) {
            thinking++;
            if ((thinking % 16) == 0 && on_thinking) on_thinking(thinking);
            if (s.think_close && next == *s.think_close) in_think = false;
            if (static_cast<int>(thinking) >= gen.max_thinking_tokens &&
                s.think_close) {
                // Rienda mecánica: cerrar el pensamiento y seguir. Cuánto vale
                // el techo lo decide el llamante, no esta clase.
                next = s.forward_one(*s.think_close, stream);
                in_think = false;
                generated++;
                continue;
            }
        } else {
            std::string piece = s.tokenizer->decode({next});
            pending += piece;
            size_t cut = utf8_complete_prefix(pending);
            if (cut > 0) {
                std::string chunk = pending.substr(0, cut);
                pending.erase(0, cut);
                visible += chunk;
                if (on_text) on_text(chunk);
            }
            visible_tokens++;
            if (static_cast<int>(visible_tokens) >= gen.max_visible_tokens) {
                *reason = FinishReason::MaxTokens;
                generated++;
                break;
            }
        }

        next = s.forward_one(next, stream);
        generated++;
    }
    if (generated >= static_cast<uint32_t>(hard_cap)) *reason = FinishReason::MaxTokens;

    if (!pending.empty()) {          // cola incompleta: se emite tal cual
        visible += pending;
        if (on_text) on_text(pending);
    }
    auto t2 = clock::now();

    stats->generated_tokens = generated;
    stats->thinking_tokens = thinking;
    stats->decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    stats->cache_position = s.position();
    return true;
}

}  // namespace helios
