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
        // Optional modality-only HNF (Gemma 4 12B unified image/audio embedders).
        std::string multimodal_hnf_path;
        uint32_t max_seq_len = 4096;
        float temperature = 0.7f;   // base; cada turno puede cambiarla
        bool use_cuda_graphs = true; // replay frente a ejecución normal para A/B
        bool use_split_attention = true; // solo geometrías/longitudes validadas; false para A/B
        bool use_coalesced_prefill = true; // geometrías Gemma validadas; false para A/B
        bool use_flash_decode = true;      // decode partido en la secuencia; false para A/B
        uint32_t flash_decode_min_seq = 1024; // por debajo el kernel de referencia es más rápido
        bool use_gemm_prefill = true;      // atención de prefill con GEMMs de cuBLAS; false para A/B
        bool use_flash_prefill = true;     // atención de prefill fusionada (HD128 sin ventana); false para A/B
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
        // Opt-in para adaptadores externos: prompt ya formateado y paradas token.
        // No interpreta herramientas ni decide qué acción debe ejecutarse.
        bool preformatted = false;
        bool close_turn = true;
        // Reaprovechar el prefijo que ya está en el KV en vez de reprocesarlo.
        // Con prompt preformateado el adaptador reenvía la conversación entera
        // cada vez: sin esto, la segunda generación de un turno vuelve a digerir
        // los cinco mil tokens que acaba de leer para añadir mil.
        bool reuse_prefix = false;
        std::vector<std::string> stop_tokens;
    };

    // Adjunto prestado: el motor recibe píxeles RGB8 ya decodificados o audio
    // PCM float32. PNG, JPEG o WebM no cruzan esta frontera — los decodifica
    // quien tenga librería para ello, que no es Héctor.
    struct ImageAttachment {
        const void* data = nullptr;
        size_t byte_size = 0;
        uint32_t width = 0, height = 0;
        size_t row_stride_bytes = 0;
        // Audio: mono PCM float32 a sample_rate. Los campos de imagen quedan a 0.
        bool audio = false;
        uint32_t sample_rate = 0, channels = 0;
    };

    enum class FinishReason { Eos, MaxTokens, Stop, Cancelled, ContextFull };
    static const char* finish_reason_name(FinishReason r);

    struct TurnStats {
        bool stopped_on_token = false;
        uint32_t prefill_tokens = 0;
        uint32_t generated_tokens = 0;
        uint32_t thinking_tokens = 0;
        double prefill_ms = 0.0;
        double decode_ms = 0.0;
        uint32_t cache_position_before = 0;
        uint32_t cache_position = 0;
        // Tokens que no hubo que volver a prefillear por estar ya en el KV.
        uint32_t prefill_reused = 0;
        uint32_t decode_graph_captures = 0;
        uint32_t decode_graph_replays = 0;
        uint32_t decode_graph_fallbacks = 0;
        double queue_ms = 0.0;
        // Desde run_turn hasta el primer fragmento visible; -1 si no lo hubo.
        double first_token_ms = -1.0;
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
    // Progreso por tanda de texto completada; total excluye el prefijo reutilizado.
    // No sustituye al evento final on_prefill. Puede señalar cancel_flag.
    using PrefillProgressCallback =
        std::function<void(uint32_t processed, uint32_t total, double ms)>;

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
    // tiene su KV, su muestreador y sus comandos de decode. El grafo CUDA
    // compartido se invalida y vuelve a capturar al comenzar cada turno.
    //
    // EN SERIE. Comparten los buffers de trabajo del grafo, así que dos turnos
    // a la vez se pisarían las activaciones. Un cerrojo interno los serializa:
    // si dos hilos entran a la vez, uno espera. Correr de verdad en paralelo
    // pide scratch por sesión, y eso todavía no está.
    // `max_seq_len` a 0 usa el del modelo. Una sesión auxiliar —preámbulos,
    // extracciones— no necesita la ventana entera y su KV cuesta VRAM.
    bool attach(std::shared_ptr<Model> model, std::string* error,
                uint32_t max_seq_len = 0);

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
    // visible, se intenta volver a `cache_position_before`; si el anillo ya
    // perdió esa ventana, se vacía el KV y el llamante debe reenviar el historial.
    // Un error de ejecución visual también vacía el KV. La posición resultante
    // se devuelve en stats. Cancelar antes de ejecutar no cambia el KV. Durante
    // prefill se vuelve al inicio efectivo, o se vacía si el anillo lo perdió.
    // Durante decode se conserva lo generado y se respeta close_turn.
    // La cancelación visual se comprueba antes/después del adaptador; no
    // interrumpe su operación interna. Los callbacks se ejecutan bajo el cerrojo
    // del modelo: pueden señalar cancelación, pero no ejecutar/resetear sesiones.
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
                  std::string* error,
                  const PrefillProgressCallback& on_prefill_progress = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace helios
