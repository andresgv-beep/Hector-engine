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
#include "multimodal_adapter.hpp"
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
    std::unique_ptr<MultimodalAdapter> multimodal;

    // Parámetros mecánicos del sampler, fijados al cargar. Se guardan porque
    // la config hay que RECONSTRUIRLA en cada turno: un turno greedy sustituye
    // el objeto entero y el siguiente turno creativo, si solo cambiara la
    // temperatura, seguiría corriendo en greedy.
    float cfg_rep = 1.0f, cfg_freq = 0.0f, cfg_topp = 0.95f;
    int cfg_topk = 64;

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
                                bool con_adjunto,
                                std::string* error_code, std::string* error);

    int32_t forward_batch(const std::vector<int32_t>& ids, cudaStream_t stream);
    int32_t forward_one(int32_t token, cudaStream_t stream);

    SamplingConfig build_sampling(float temperature) const {
        SamplingConfig c = temperature < 0.01f
            ? SamplingConfig::greedy()
            : SamplingConfig::creative(temperature, cfg_topk, cfg_topp);
        c.repetition_penalty = cfg_rep;
        c.frequency_penalty = cfg_freq;
        return c;
    }

    // El penalty jamás debe caer sobre tokens especiales: penalizar </think>
    // es lo que rompía el cierre del pensamiento en el oráculo.
    bool penalizable(int32_t id) const { return is_gemma4 || id < 151643; }
};

std::vector<int32_t> InferenceSession::Impl::encode(
        const std::vector<ChatMessage>& messages,
        bool con_adjunto,
        std::string* error_code, std::string* error) {
    // El adaptador busca su marcador dentro del turno ya formateado para
    // expandirlo a los soft tokens de la imagen. Sin él, el prefill visual
    // falla aunque los píxeles estén perfectos.
    std::vector<ChatMessage> msgs = messages;
    if (con_adjunto) {
        for (auto it = msgs.rbegin(); it != msgs.rend(); ++it) {
            if (it->role == "user") { it->content = "<|image|>\n" + it->content; break; }
        }
    }
    const std::vector<ChatMessage>& messages_ref = msgs;
    if (is_gemma4) {
        // Gemma 4 no admite un `system` nuevo entre turnos: su plantilla lo
        // fusiona con el primer user. En vez de inventarnos un encaje, se
        // rechaza y que E2 decida cómo entregar lo efímero.
        if (position() > 0) {
            for (const auto& m : messages_ref) {
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
        auto ids = tokenizer->encode(format_gemma4_chat(messages_ref, options),
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
    for (const auto& m : messages_ref) {
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
        s.cfg_rep = rep; s.cfg_freq = freq; s.cfg_topk = topk; s.cfg_topp = topp;
        s.sample_config = s.build_sampling(config.temperature);

        const ModelCapabilities caps = inspect_model_capabilities(s.loader);
        const auto* vision = caps.find(ModelModality::Vision);
        if (vision && vision->status == AdapterStatus::RuntimeReady) {
            std::string adapter_error;
            s.multimodal = create_multimodal_adapter(
                vision->adapter_id, *s.engine, s.loader, s.gb, s.arch,
                kPrefillChunk, &adapter_error);
            if (!s.multimodal) {
                *error = "no pude crear el adaptador visual: " + adapter_error;
                return false;
            }
        }
        s.info.architecture = caps.text_architecture;
        s.info.max_seq_len = s.kv_config.max_seq_len;
        s.info.multimodal = caps.multimodal;
        // Solo se anuncia lo que el ejecutable puede cumplir de verdad.
        if (s.multimodal) s.info.vision_adapter = s.multimodal->id();
        s.info.multimodal = (s.multimodal != nullptr);
        return true;
    } catch (const std::exception& e) {
        *error = e.what();
        return false;
    }
}

bool InferenceSession::run_turn(const std::vector<ChatMessage>& messages,
                                const std::vector<ImageAttachment>& attachments,
                                const GenConfig& gen,
                                const TextCallback& on_text,
                                const ThinkingCallback& on_thinking,
                                const PrefillCallback& on_prefill,
                                const std::atomic<bool>& cancel_flag,
                                TurnStats* stats,
                                FinishReason* reason,
                                std::string* error_code,
                                std::string* error) {
    auto& s = *impl_;
    using clock = std::chrono::high_resolution_clock;

    const uint32_t antes = s.position();
    stats->cache_position_before = antes;
    stats->cache_position = antes;

    auto ids = s.encode(messages, !attachments.empty(), error_code, error);
    if (ids.empty()) {
        if (error_code->empty()) {
            *error_code = "empty_turn";
            *error = "los mensajes no produjeron tokens";
        }
        return false;   // nada tocado: el KV sigue donde estaba
    }
    if (!attachments.empty() && !s.multimodal) {
        *error_code = "unsupported_attachment";
        *error = "este modelo no declara adaptador visual";
        return false;
    }
    if (antes + ids.size() + 8 >= s.kv_config.max_seq_len) {
        *error_code = "context_full";
        *error = "el turno no cabe en el contexto restante";
        return false;
    }

    // Config MECÁNICA COMPLETA por turno: reconstruirla evita que un turno
    // greedy deje al siguiente creativo corriendo en greedy.
    s.sample_config = s.build_sampling(gen.temperature);

    cudaStream_t stream = s.engine_config.stream;
    std::string visible, pending;
    uint32_t visible_tokens = 0, thinking = 0, generated = 0;
    bool in_think = false, think_cut = false;
    *reason = FinishReason::Eos;

    // Volcar la cola pendiente sin dejar UTF-8 a medias: lo incompleto se
    // sustituye por U+FFFD porque emitirlo crudo invalidaría la cadena JSON.
    auto flush_pending = [&]() {
        if (pending.empty()) return;
        size_t cut = utf8_complete_prefix(pending);
        std::string chunk = pending.substr(0, cut);
        if (cut < pending.size()) chunk += "\xEF\xBF\xBD";
        pending.clear();
        if (!chunk.empty()) {
            visible += chunk;
            if (on_text) on_text(chunk);
        }
    };

    try {
        auto t0 = clock::now();
        int32_t next = 0;
        if (!attachments.empty()) {
            // El adaptador expande el placeholder y prefillea; la sesión solo
            // avanza su posición lógica si la llamada tuvo éxito.
            MultimodalTurnInput turn;
            turn.formatted_token_ids = ids;
            for (const auto& a : attachments) {
                turn.attachments.push_back({AttachmentKind::ImageRgb8,
                                            a.data, a.byte_size, "image/rgb8",
                                            a.width, a.height,
                                            a.row_stride_bytes, 0, 0});
            }
            std::string verr;
            if (!validate_multimodal_turn(turn, s.multimodal->limits(), &verr)) {
                *error_code = "invalid_attachment";
                *error = verr;
                return false;
            }
            MultimodalPrefillResult pr;
            if (!s.multimodal->prefill(turn, {s.kv_prefix, antes,
                                              s.kv_config.max_seq_len},
                                       pr, &verr)) {
                *error_code = "attachment_prefill_failed";
                *error = verr;
                s.rewind(antes);          // §4: nada emitido, sin rastro
                stats->cache_position = s.position();
                return false;
            }
            s.advance(pr.sequence_tokens);
            auto* logits = s.gb.get_logits(*s.engine);
            next = s.sampler.sample(static_cast<const half*>(logits->ptr),
                                    s.model_config.vocab_size(),
                                    s.sample_config, stream);
            stats->prefill_tokens = pr.sequence_tokens;
        } else {
            next = s.forward_batch(ids, stream);
            stats->prefill_tokens = static_cast<uint32_t>(ids.size());
        }
        auto t1 = clock::now();
        stats->prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (on_prefill) on_prefill(stats->prefill_tokens, stats->prefill_ms);

        const int hard_cap = gen.max_visible_tokens + gen.max_thinking_tokens + 64;
        bool natural_stop = false;
        while (generated < static_cast<uint32_t>(hard_cap)) {
            if (cancel_flag.load(std::memory_order_relaxed)) {
                *reason = FinishReason::Cancelled;
                break;
            }
            if (next == s.eos_id || next == s.turn_end) {
                natural_stop = true;
                *reason = FinishReason::Eos;
                break;
            }
            if (s.position() + 4 >= s.kv_config.max_seq_len) {
                *reason = FinishReason::Stop;
                break;
            }

            if (s.think_open && next == *s.think_open) in_think = true;

            if (in_think) {
                thinking++;
                if ((thinking % 16) == 0 && on_thinking) on_thinking(thinking);
                if (s.think_close && next == *s.think_close) in_think = false;
                if (static_cast<int>(thinking) >= gen.max_thinking_tokens &&
                    s.think_close && in_think) {
                    next = s.forward_one(*s.think_close, stream);
                    in_think = false;
                    think_cut = true;
                    generated++;
                    continue;
                }
            } else {
                pending += s.tokenizer->decode({next});
                size_t cut = utf8_complete_prefix(pending);
                if (cut > 0) {
                    std::string chunk = pending.substr(0, cut);
                    pending.erase(0, cut);
                    visible += chunk;
                    if (on_text) on_text(chunk);
                }
                visible_tokens++;
            }

            // El penalty ve el token ACTUAL antes de muestrear el siguiente:
            // registrarlo después desplaza la ventana una posición y el
            // sampler no ve lo que acaba de emitir. Sin esto, repetition y
            // frequency penalty están desconectados y no hay paridad posible.
            if (!in_think && s.penalizable(next)) s.sampler.add_context(next);
            generated++;

            const bool tope_visible =
                static_cast<int>(visible_tokens) >= gen.max_visible_tokens;
            // Aunque se corte por presupuesto, el token DEBE entrar al KV:
            // si no, el usuario lo ve y la siguiente generación no.
            next = s.forward_one(next, stream);
            if (tope_visible) { *reason = FinishReason::MaxTokens; break; }
        }
        if (generated >= static_cast<uint32_t>(hard_cap)) {
            *reason = FinishReason::MaxTokens;
        }
        flush_pending();

        // CERRAR EL TURNO, igual que el oráculo: el terminal muestreado aún no
        // está en KV. Gemma 4 necesita <turn|> siempre; ChatML solo cuando el
        // runtime corta. Sin esto la continuidad se rompe en el turno 2 aunque
        // el primero parezca perfecto.
        const bool forzado = (*reason != FinishReason::Eos) || think_cut;
        if ((s.is_gemma4 && natural_stop) || forzado) {
            if (s.position() + 2 < s.kv_config.max_seq_len) {
                (void)s.forward_one(s.turn_end, stream);
                for (int32_t t : s.tokenizer->encode("\n", false, false)) {
                    if (s.position() + 2 >= s.kv_config.max_seq_len) break;
                    (void)s.forward_one(t, stream);
                }
            }
        }
        auto t2 = clock::now();
        stats->decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    } catch (const std::exception& e) {
        flush_pending();
        *error_code = "engine_failure";
        *error = e.what();
        stats->generated_tokens = generated;
        stats->thinking_tokens = thinking;
        // §4: si no llegó a emitir texto, el turno no deja rastro; si ya
        // emitió, se conserva lo emitido y se reporta la posición real.
        if (visible.empty()) s.rewind(antes);
        stats->cache_position = s.position();
        return false;
    }

    stats->generated_tokens = generated;
    stats->thinking_tokens = thinking;
    stats->cache_position = s.position();
    return true;
}

}  // namespace helios
