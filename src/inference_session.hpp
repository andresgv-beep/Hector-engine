#pragma once

// Sesión de inferencia reutilizable — E1 del plan de evacuación cognitiva.
//
// Encapsula lo que SÍ pertenece a Héctor según la tabla de propiedad del plan:
// HNF, tokenizer, plantillas del modelo, KV, prefill, decode y sampling. Nada
// más. Aquí no hay identidad, memoria, perfiles, actos, registro social ni
// presupuestos cognitivos: los topes de tokens son parámetros mecánicos y
// quien decide sus valores es el llamante.
//
// `helios_chat` NO usa esta clase: sigue intacto como oráculo hasta que la
// ruta nueva demuestre paridad. Duplicar la composición durante la transición
// es deliberado — es más barato que arriesgar el oráculo contra el que se
// mide todo lo demás.

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "chat_template.hpp"

namespace helios {

class InferenceSession {
public:
    struct Config {
        std::string hnf_path;
        uint32_t max_seq_len = 4096;
        float temperature = 0.7f;   // base; cada turno puede cambiarla
    };

    struct ModelInfo {
        std::string architecture;
        uint32_t max_seq_len = 0;
        bool multimodal = false;
        std::string vision_adapter;   // vacío si no hay
    };

    struct GenConfig {
        float temperature = 0.7f;
        int max_visible_tokens = 512;
        int max_thinking_tokens = 400;
    };

    enum class FinishReason { Eos, MaxTokens, Stop, Cancelled };
    static const char* finish_reason_name(FinishReason r);

    struct TurnStats {
        uint32_t prefill_tokens = 0;
        uint32_t generated_tokens = 0;
        uint32_t thinking_tokens = 0;
        double prefill_ms = 0.0;
        double decode_ms = 0.0;
        uint32_t cache_position_before = 0;
        uint32_t cache_position = 0;
    };

    // Fragmento de texto VISIBLE, siempre UTF-8 completo: el protocolo lo
    // mete en una cadena JSON y un multibyte partido la invalidaría.
    using TextCallback = std::function<void(const std::string&)>;
    // Latido del pensamiento: solo el contador, nunca el contenido.
    using ThinkingCallback = std::function<void(uint32_t)>;

    InferenceSession();
    ~InferenceSession();
    InferenceSession(const InferenceSession&) = delete;
    InferenceSession& operator=(const InferenceSession&) = delete;

    bool load(const Config& config, std::string* error);
    const ModelInfo& info() const;

    uint32_t cache_position() const;
    void reset();

    // Añade `messages` al KV y genera. Devuelve false y llena `error_code` /
    // `error` si el turno no pudo ejecutarse.
    //
    // Semántica del KV (§4 del protocolo): si falla ANTES de emitir texto
    // visible, el KV vuelve a `cache_position_before`; si ya emitió, se
    // conserva lo emitido. La cancelación conserva exactamente lo emitido.
    bool run_turn(const std::vector<ChatMessage>& messages,
                  const GenConfig& gen,
                  const TextCallback& on_text,
                  const ThinkingCallback& on_thinking,
                  const std::atomic<bool>& cancel_flag,
                  TurnStats* stats,
                  FinishReason* reason,
                  std::string* error_code,
                  std::string* error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace helios
