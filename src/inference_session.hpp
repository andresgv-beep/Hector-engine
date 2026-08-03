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

// ============================================================================
// MODELO COMPARTIDO
// ============================================================================
// Lo que se carga una vez y no pertenece a ninguna conversación: los pesos del
// HNF, el tokenizer, las plantillas, el grafo y sus buffers de trabajo, el
// adaptador visual. Cargarlo cuesta segundos y gigabytes de VRAM; el estado de
// una conversación cuesta un KV.
//
// La separación no es estética. Sin ella, "otra sesión" significa "otro
// proceso con el modelo entero otra vez", y eso pone un techo absurdo a
// cualquier cosa que necesite dos hilos de conversación sobre la misma GPU.
//
// Héctor NO sabe para qué se usa cada sesión. Aquí no hay reflexión, ni
// consolidación, ni chat: hay sesiones genéricas sobre unos pesos. Para qué
// sirve cada una lo decide quien llama.
class Model {
public:
    struct Config {
        std::string hnf_path;
        uint32_t max_seq_len = 4096;
        float temperature = 0.7f;   // base; cada turno puede cambiarla
    };

    struct Info {
        std::string architecture;
        uint32_t max_seq_len = 0;
        bool multimodal = false;
        std::string vision_adapter;   // vacío si no hay
    };

    // Carga los pesos. Devuelve nullptr y llena `error` si no puede.
    static std::shared_ptr<Model> load(const Config& config, std::string* error);

    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    const Info& info() const;

    // Cuántas sesiones se han creado sobre estos pesos. Permite comprobar
    // desde fuera que de verdad se están compartiendo.
    size_t sessions_created() const;

private:
    Model();
    friend class InferenceSession;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// SESIÓN
// ============================================================================
class InferenceSession {
public:
    // Los nombres de siempre siguen valiendo: quien ya usaba
    // InferenceSession::Config no tiene que cambiar nada.
    using Config = Model::Config;
    using ModelInfo = Model::Info;

    struct GenConfig {
        float temperature = 0.7f;
        int max_visible_tokens = 512;
        int max_thinking_tokens = 400;
    };

    // Adjunto RGB8 prestado: el motor recibe píxeles ya decodificados. PNG y
    // JPEG no cruzan esta frontera — los decodifica quien tenga librería de
    // imágenes, que no es Héctor.
    struct ImageAttachment {
        const void* data = nullptr;
        size_t byte_size = 0;
        uint32_t width = 0, height = 0;
        size_t row_stride_bytes = 0;
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
    // El prefill real, en cuanto termina y ANTES del primer fragmento: el
    // protocolo exige que su evento preceda a cualquier text_delta y con
    // cifras verdaderas, no ceros de relleno.
    using PrefillCallback = std::function<void(uint32_t tokens, double ms)>;

    InferenceSession();
    ~InferenceSession();
    InferenceSession(const InferenceSession&) = delete;
    InferenceSession& operator=(const InferenceSession&) = delete;

    // Carga un modelo SOLO para esta sesión. Sigue existiendo porque hay
    // llamantes de una sola conversación para los que montar un `Model`
    // aparte no aporta nada.
    bool load(const Config& config, std::string* error);

    // Se engancha a unos pesos ya cargados. Varias sesiones sobre el mismo
    // `Model` comparten VRAM y no comparten NADA de la conversación: cada una
    // tiene su KV, su muestreador y su grafo de decode.
    //
    // EN SERIE. Comparten los buffers de trabajo del grafo, así que dos turnos
    // a la vez se pisarían las activaciones. Un cerrojo interno los serializa:
    // si dos hilos entran a la vez, uno espera. Correr de verdad en paralelo
    // pide scratch por sesión, y eso todavía no está.
    bool attach(std::shared_ptr<Model> model, std::string* error);

    const ModelInfo& info() const;

    // Prefijo con el que esta sesión registra su KV en el motor. Dos sesiones
    // tienen prefijos distintos: es lo que impide que una escriba en el caché
    // de la otra.
    const std::string& kv_namespace() const;

    uint32_t cache_position() const;
    void reset();

    // Añade `messages` al KV y genera. Devuelve false y llena `error_code` /
    // `error` si el turno no pudo ejecutarse.
    //
    // Semántica del KV (§4 del protocolo): si falla ANTES de emitir texto
    // visible, el KV vuelve a `cache_position_before`; si ya emitió, se
    // conserva lo emitido. La cancelación conserva exactamente lo emitido.
    // `attachments` vacío = turno de solo texto. Si no hay adaptador visual
    // en el HNF, un turno con adjunto falla con `unsupported_attachment`.
    bool run_turn(const std::vector<ChatMessage>& messages,
                  const std::vector<ImageAttachment>& attachments,
                  const GenConfig& gen,
                  const TextCallback& on_text,
                  const ThinkingCallback& on_thinking,
                  const PrefillCallback& on_prefill,
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
