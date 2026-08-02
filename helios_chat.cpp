// helios_chat.cpp
// ============================================================================
// HELIOS CHAT — Chat multi-turno con continuidad de KV cache
// ============================================================================
//
// El embrión del runtime conversacional de HELIOS. Junta todo lo probado:
//   - CUDA Graph replay (command buffer una vez, replay por token)
//   - Plantilla nativa por arquitectura (ChatML o Gemma 4 canónica)
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
#include "src/gemma4_kv_cache.hpp"
#include "src/chat_template.hpp"
#include "src/model_capabilities.hpp"
#include "src/multimodal_adapter.hpp"
#include "src/ck_policy.hpp"
#include "kernels/kernels.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <set>
#include <ctime>
#include <sys/stat.h>
#include <sys/select.h>
#include <unistd.h>

// ¿Hay más datos esperando en stdin? (para detectar pegotes multilínea)
static bool stdin_ready(int timeout_ms) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(0, &fds);
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    return select(1, &fds, nullptr, nullptr, &tv) > 0;
}

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
// MEMORIA EPISÓDICA — nivel 1: destilar al cierre, recordar al arrancar
// ============================================================================
// Al /salir, el propio modelo resume la sesión y se guarda con fecha.
// Al arrancar, los recuerdos entran al system prompt (prefijo en KV).
// Recencia pura por ahora; la búsqueda semántica es el siguiente nivel.

// El perfil vive fuera del codigo. `profile_name` es el nombre nuevo;
// `owner` se mantiene como lectura legacy para no romper perfiles existentes.
// HELIOS_HOME selecciona la persona actual: el binario y la identidad base de
// Helios nunca contienen datos de una persona concreta.
static std::string helios_dir();   // definida abajo (respeta HELIOS_HOME)

static std::string profile_name() {
    std::string name;
    for (const char* filename : {"profile_name", "owner"}) {
        std::ifstream f(helios_dir() + "/" + filename);
        if (f && std::getline(f, name) && !name.empty()) return name;
    }
    return "la persona actual";
}

// DOS NIVELES (la jerarquía de HERA, nivel 1):
//   facts.md    — hechos duraderos sobre la persona y sus proyectos. Se cargan
//                 SIEMPRE y enteros: describen su perfil, no a Helios.
//   episodic.md — resúmenes de sesión. Solo las últimas: son qué pasó.
// Sin esta separación, el chismorreo episódico ("me pidió que le explique
// un compilador") saturaba el prefijo y secuestraba las respuestas.
// HELIOS_HOME permite perfiles aislados (calibración reproducible, varios
// usuarios en la misma máquina). Sin él: ~/.helios como siempre.
static std::string helios_dir() {
    const char* custom = getenv("HELIOS_HOME");
    std::string dir;
    if (custom && *custom) {
        dir = custom;
    } else {
        const char* home = getenv("HOME");
        dir = std::string(home ? home : ".") + "/.helios";
    }
    mkdir(dir.c_str(), 0755);
    return dir;
}
static std::string facts_path()  { return helios_dir() + "/facts.md"; }
static std::string memory_path() { return helios_dir() + "/episodic.md"; }
static std::string candidates_path() { return helios_dir() + "/candidates.md"; }

static std::string load_profile_facts(size_t max_chars) {
    std::ifstream f(facts_path());
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    std::string facts = ss.str();
    if (facts.size() > max_chars) {
        facts.resize(max_chars);
        const size_t boundary = facts.find_last_of('\n');
        if (boundary != std::string::npos) facts.resize(boundary + 1);
    }
    return facts;
}

// Carga jerárquica: TODOS los hechos + las últimas N sesiones
static std::string load_memories(size_t max_chars) {
    std::string out;

    std::ifstream ff(facts_path());
    if (ff) {
        std::stringstream fs;
        fs << ff.rdbuf();
        std::string facts = fs.str();
        if (!facts.empty()) {
            if (facts.size() > max_chars * 2 / 3)          // tope de seguridad
                facts = facts.substr(facts.size() - max_chars * 2 / 3);
            out += "PERFIL DE LA PERSONA ACTUAL (datos sobre ella; no son "
                   "datos tuyos ni cambian tu identidad):\n" + facts + "\n";
        }
    }

    std::ifstream f(memory_path());
    if (f) {
        std::stringstream ss;
        ss << f.rdbuf();
        std::string all = ss.str();
        // Solo las 3 últimas sesiones: lo demás es ruido episódico
        const int KEEP = 3;
        size_t pos = all.size();
        for (int i = 0; i < KEEP && pos != std::string::npos && pos > 0; i++) {
            size_t p = all.rfind("\n## ", pos - 1);
            if (p == std::string::npos) { pos = 0; break; }
            pos = p;
        }
        std::string recent = (pos == 0) ? all : all.substr(pos);
        size_t budget = max_chars - std::min(out.size(), max_chars);
        if (recent.size() > budget) recent = recent.substr(recent.size() - budget);
        if (!recent.empty()) out += "Últimas sesiones:\n" + recent;
    }
    return out;
}

// Normaliza para comparar: minúsculas, sin tildes ni puntuación, palabras
static std::vector<std::string> mem_words(const std::string& s) {
    std::string n;
    for (unsigned char c : s) {
        if (isalnum(c) || c == ' ') n += (char)tolower(c);
        else if ((unsigned char)c >= 128) n += (char)c;  // acentos: se dejan
        else n += ' ';
    }
    std::vector<std::string> w;
    std::stringstream ss(n);
    std::string t;
    while (ss >> t) if (t.size() > 3) w.push_back(t);
    return w;
}

// ¿Ya sabemos esto? Compara por CONTENCIÓN entre entradas completas.
//
// La versión anterior comparaba el candidato contra cada LÍNEA suelta: una
// nota de 15 líneas contra una línea daba solapamiento bajísimo y colaba.
// Así se duplicó tres veces el mismo capítulo de una historia en el diario.
//
// Ahora: se parte el fichero en ENTRADAS (bloques '## ...' o líneas '- ...'),
// se usan conjuntos de palabras significativas, y el criterio es la
// contención sobre el MENOR de los dos — así detecta también cuando el
// candidato es un trozo de algo ya guardado, o una versión ampliada.
static bool memory_is_duplicate(const std::string& candidate,
                                const std::string& path) {
    auto cwv = mem_words(candidate);
    if (cwv.empty()) return false;
    std::set<std::string> cw(cwv.begin(), cwv.end());
    if (cw.size() < 3) return false;   // demasiado corto para juzgar

    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string all = ss.str();

    // Trocear en entradas: los bloques '## ' del diario, o cada '- ' de facts
    std::vector<std::string> entries;
    size_t pos = 0;
    while (pos < all.size()) {
        size_t next = all.find("\n## ", pos);
        std::string block = all.substr(pos, next == std::string::npos
                                            ? std::string::npos : next - pos);
        if (block.find("\n- ") != std::string::npos || block.rfind("- ", 0) == 0) {
            std::stringstream bs(block);
            std::string ln;
            while (std::getline(bs, ln))
                if (!ln.empty() && ln[0] != '#') entries.push_back(ln);
        } else {
            entries.push_back(block);
        }
        if (next == std::string::npos) break;
        pos = next + 1;
    }

    for (const auto& e : entries) {
        auto ewv = mem_words(e);
        if (ewv.size() < 3) continue;
        std::set<std::string> ew(ewv.begin(), ewv.end());
        size_t inter = 0;
        for (const auto& w : cw) if (ew.count(w)) inter++;
        // Contención sobre el menor: pilla copias, trozos y ampliaciones
        double cont = (double)inter / (double)std::min(cw.size(), ew.size());
        if (cont > 0.65) return true;
    }
    return false;
}

static void append_memory(const std::string& summary) {
    if (memory_is_duplicate(summary, memory_path())) return;
    std::ofstream f(memory_path(), std::ios::app);
    if (!f) return;
    char datebuf[64];
    time_t now = time(nullptr);
    strftime(datebuf, sizeof(datebuf), "%Y-%m-%d %H:%M", localtime(&now));
    f << "\n## Sesión " << datebuf << "\n" << summary << "\n";
}

// Hecho duradero → facts.md (perfil de la persona, no lo que pasó un día).
// `quoted` = son sus palabras textuales ("apunta esto: mi perro...").
// Guardarlas tal cual es veneno: al releer "mi perro se llama X" el modelo
// lo repite en primera persona y se cree que es la persona. Con atribución
// queda claro de quién son las palabras sin perder el dato literal.
static void append_fact(const std::string& fact, bool quoted = false) {
    if (memory_is_duplicate(fact, facts_path())) return;
    std::ofstream f(facts_path(), std::ios::app);
    if (!f) return;
    if (quoted) f << "- La persona dijo textualmente: \"" << fact << "\"\n";
    else        f << "- " << fact << "\n";
}

// La extracción automática NO tiene autoridad para contaminar el prefijo.
// El modelo solo propone; una petición explícita del usuario sí escribe facts.
static void append_candidate(const std::string& fact) {
    if (memory_is_duplicate(fact, facts_path()) ||
        memory_is_duplicate(fact, candidates_path())) return;
    std::ofstream f(candidates_path(), std::ios::app);
    if (!f) return;
    f << "- " << fact << "\n";
}

// ============================================================================
// REGISTRO SOCIAL — primer eje real del CK (CK_V4_NORTE, fallo 4)
// ============================================================================
// Estado PEGAJOSO entre turnos: hasta ahora las tres funciones que deciden un
// turno recibían solo el mensaje, así que para el modelo cada mensaje era el
// turno 1 y el tono no se podía mover. El registro persiste y gobierna a la
// vez el suelo de temperatura (y en el futuro presupuesto/ejemplos), en vez
// de tres heurísticas independientes tirando cada una por su lado.
//
// Histéresis de DOS señales seguidas: una broma suelta en mitad de un debug
// no descarrila la sesión, y un dato en mitad de la charla tampoco. Se puede
// forzar con /colega, /trabajo y /normal — si el usuario lo pide, eso ES la
// spec, no una pista a inferir.

enum class Registro { TRABAJO, NEUTRAL, COLEGA };

static const char* registro_name(Registro r) {
    switch (r) {
        case Registro::TRABAJO: return "trabajo";
        case Registro::COLEGA:  return "colega";
        default:                return "neutral";
    }
}

// Señal de charla: risas, coloquialismos, emojis. Deliberadamente NO incluye
// saludos: "hola" es social pero no significa que la sesión sea de cachondeo.
static bool senal_colega(const std::string& msg) {
    if (msg.size() > 200) return false;   // un pegote nunca es charla
    std::string lower = msg;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    for (const char* kw : {"jaja", "jeje", "jiji", "xd", "socio", "tio ",
                           "tío ", "bro ", "colega", "crack", "brutal",
                           "mola", "guay", "flipa", "hostia", "ostia",
                           "cojones", "de puta madre", "madre mia",
                           "madre mía", "vaya tela", "me parto"}) {
        if (lower.find(kw) != std::string::npos) return true;
    }
    // Emoji (cualquier U+1FXXX: F0 9F en UTF-8)
    for (size_t i = 0; i + 1 < msg.size(); i++) {
        if ((unsigned char)msg[i] == 0xF0 && (unsigned char)msg[i + 1] == 0x9F)
            return true;
    }
    return false;
}

// Señal de faena: documentos, código, densidad de vocabulario técnico.
static bool senal_trabajo(const std::string& msg) {
    if (msg.find("```") != std::string::npos) return true;
    if (msg.rfind("Te paso un texto", 0) == 0) return true;   // /pegar y ficheros
    if (msg.rfind("Archivo: ", 0) == 0) return true;
    std::string lower = msg.substr(0, 400);
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    int hits = 0;
    for (const char* kw : {"kernel", "código", "codigo", "compila", "bug",
                           "error", "implementa", "commit", "benchmark",
                           "tensor", "vram", "cuantiza", "prefill", "latencia",
                           "función", "funcion", "script", "endpoint"}) {
        if (lower.find(kw) != std::string::npos && ++hits >= 2) return true;
    }
    return false;
}

// ============================================================================
// TEMPERATURA ADAPTATIVA — el CK modula el muestreo según lo que se pide
// ============================================================================
// Charla → más alta (variedad, naturalidad). Datos/código/memoria → más baja
// (precisión, menos deriva). La base la fija el usuario al arrancar.

static float temperature_for(const std::string& msg, bool trivial, float base,
                             Registro reg) {
    // Solo el ARRANQUE del mensaje: el tono lo pone lo que el usuario teclea,
    // no el documento que adjunta. Un spec pegado casi siempre contiene
    // "error" o "dato" en alguna tabla, y bajaba TODOS los turnos con
    // documento a 0.35 aplanando la voz (visto con HQ6_LAB.md, que disparaba
    // por una columna llamada "Error máximo").
    std::string lower = msg.substr(0, 160);
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    // Precisión: hechos, memoria, código, números. El SUELO depende del
    // registro: la precisión vive en el contexto, no en congelar el sampler
    // — a 0.35 la voz muere, porque la personalidad vive en la cola de la
    // distribución. Justo cuando pregunta "¿te acuerdas de...?" (el momento
    // más personal que hay) era cuando más plano se ponía.
    float suelo = (reg == Registro::TRABAJO) ? 0.45f
                : (reg == Registro::COLEGA)  ? 0.60f
                                             : 0.50f;
    for (const char* kw : {"codigo", "código", "code", "error", "cuanto", "cuánto",
                           "calcula", "recuerdas", "recuerda", "cuál era", "cual era",
                           "dato", "exacto", "traduce", "define", "comando"}) {
        if (lower.find(kw) != std::string::npos) return std::min(base, suelo);
    }
    // Charla informal: un poco por encima de la base, tope 0.95
    if (trivial) return std::min(0.95f, base + 0.2f);
    return base;
}

// ¿Es una petición de continuar lo cortado? ("sigue", "continúa"...)
static bool is_continuation(const std::string& msg) {
    std::string lower = msg;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    for (const char* kw : {"sigue", "continúa", "continua", "prosigue",
                           "acaba", "termina eso", "y despues", "y después"}) {
        if (lower.rfind(kw, 0) == 0) return true;
    }
    return false;
}

// ============================================================================
// THINKING ADAPTATIVO — heurística nivel 1
// ============================================================================
// Trivial (saludos, acks, mensajes muy cortos sin pregunta) → /no_think.
// Conservador a propósito: ante la duda, se deja pensar.

static bool is_trivial_message(const std::string& msg) {
    if (msg.size() > 60) return false;
    if (is_continuation(msg)) return false;  // "sigue" es trabajo, no saludo

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

        const bool is_gemma4 = loader.has_gemma4_config();
        auto turn_start = tokenizer->token_to_id(
            is_gemma4 ? "<|turn>" : "<|im_start|>");
        auto turn_end = tokenizer->token_to_id(
            is_gemma4 ? "<turn|>" : "<|im_end|>");
        if (!turn_start || !turn_end) {
            throw std::runtime_error(is_gemma4
                ? "Gemma 4 sin tokens de turno canónicos"
                : "Modelo sin tokens ChatML — usa un instruct");
        }
        int32_t eos_id = tokenizer->eos_token_id().value_or(
            is_gemma4 ? 1 : 151645);
        auto think_open = is_gemma4
            ? std::optional<int32_t>{}
            : tokenizer->token_to_id("<think>");
        auto think_close = is_gemma4
            ? std::optional<int32_t>{}
            : tokenizer->token_to_id("</think>");

        auto encode_gemma_messages = [&](const std::vector<ChatMessage>& messages,
                                         bool add_generation_prompt) {
            Gemma4ChatOptions options;
            options.add_generation_prompt = add_generation_prompt;
            return tokenizer->encode(format_gemma4_chat(messages, options),
                                     false, false);
        };

        // Fragmento incremental: la conversación ya contiene el BOS y los
        // turnos anteriores en KV, así que se elimina únicamente el BOS que la
        // plantilla completa coloca al principio.
        auto encode_gemma_user_fragment = [&](const std::string& content) {
            auto ids = encode_gemma_messages({{"user", content}}, true);
            const int32_t bos = tokenizer->bos_token_id().value_or(2);
            if (ids.empty() || ids.front() != bos) {
                throw std::runtime_error("La plantilla Gemma 4 no comienza por BOS");
            }
            ids.erase(ids.begin());
            return ids;
        };

        // --- KV cache ---
        KVCacheConfig kv_config;
        kv_config.num_layers = model_config.num_hidden_layers();
        kv_config.num_kv_heads = model_config.num_key_value_heads();
        kv_config.head_dim = model_config.head_dim();
        kv_config.max_batch_size = 1;
        // HELIOS_CTX permite ajustarlo al modelo: un 8B deja menos VRAM libre
        // para el cache. Con compactación en caliente, un contexto menor solo
        // significa que la conversación respira más a menudo, no que muera.
        kv_config.max_seq_len = (uint32_t)std::max(512,
            (getenv("HELIOS_CTX") ? atoi(getenv("HELIOS_CTX")) : 4096));
        KVCache kv_cache;
        Gemma4KVCache gemma_kv_cache;
        std::string kv_prefix = "_kv";
        if (is_gemma4) {
            if (!gemma_kv_cache.allocate(loader.gemma4_config(),
                                          model_config.num_key_value_heads(), 1,
                                          kv_config.max_seq_len)) {
                throw std::runtime_error("Gemma 4 heterogeneous KV alloc failed");
            }
            gemma_kv_cache.register_tensors(engine, kv_prefix);
        } else {
            if (!kv_cache.allocate(kv_config)) {
                throw std::runtime_error("KV alloc failed");
            }
            register_kv_cache(engine, kv_cache, kv_prefix);
        }

        auto cache_position = [&]() -> uint32_t {
            return is_gemma4 ? gemma_kv_cache.position() : kv_cache.position();
        };
        auto cache_advance = [&](uint32_t tokens) {
            if (is_gemma4) gemma_kv_cache.advance(tokens);
            else kv_cache.advance(tokens);
        };
        auto cache_reset = [&]() {
            if (is_gemma4) gemma_kv_cache.reset();
            else kv_cache.reset();
        };
        auto cache_rewind = [&](uint32_t position) {
            if (is_gemma4) gemma_kv_cache.rewind_to(position);
            else kv_cache.rewind_to(position);
        };
        auto cache_total_bytes = [&]() -> size_t {
            return is_gemma4 ? gemma_kv_cache.total_bytes()
                             : kv_config.total_bytes();
        };

        GraphBuilder gb;
        auto arch = gb.detect_architecture(engine, "text");
        // Scratch para prefill por lotes (hasta PREFILL_CHUNK tokens por forward)
        const uint32_t PREFILL_CHUNK = 512;
        if (is_gemma4) {
            gb.allocate_gemma4_scratch(engine, model_config,
                                       loader.gemma4_config(), arch,
                                       1, PREFILL_CHUNK);
        } else {
            gb.allocate_scratch(engine, model_config, arch, 1, PREFILL_CHUNK);
        }

        // Optional multimodal adapter resolved exclusively from HNF metadata.
        // The chat owns the session/KV; the adapter owns modality preprocessing
        // and the architecture-specific prefill.
        std::unique_ptr<MultimodalAdapter> multimodal_adapter;
        const ModelCapabilities model_capabilities =
            inspect_model_capabilities(loader);
        if (const auto* vision = model_capabilities.find(ModelModality::Vision);
            vision && vision->status == AdapterStatus::RuntimeReady) {
            std::string adapter_error;
            multimodal_adapter = create_multimodal_adapter(
                vision->adapter_id, engine, loader, gb, arch,
                PREFILL_CHUNK, &adapter_error);
            if (!multimodal_adapter) {
                throw std::runtime_error(
                    "No se pudo crear el adaptador multimodal: " + adapter_error);
            }
            std::cout << ">>> Adjuntos: " << multimodal_adapter->id()
                      << " (RGB8 persistente)" << std::endl;
        }

        Sampler sampler;
        // Parámetros de muestreo calibrables por entorno (para barridos A/B
        // sin recompilar): HELIOS_REP, HELIOS_WINDOW, HELIOS_FREQ,
        // HELIOS_TOPK, HELIOS_TOPP, HELIOS_SEED
        auto env_f = [](const char* k, float def) {
            const char* v = getenv(k); return v ? (float)atof(v) : def;
        };
        auto env_i = [](const char* k, int def) {
            const char* v = getenv(k); return v ? atoi(v) : def;
        };
        // Los overrides sirven para calibrar sin recompilar. Los defaults
        // siguen siendo la configuración estable anterior: una muestra con
        // una semilla y un prompt no basta para promover una variante.
        const float cfg_rep  = env_f("HELIOS_REP", is_gemma4 ? 1.0f : 1.15f);
        const float cfg_freq = env_f("HELIOS_FREQ", is_gemma4 ? 0.0f : 0.1f);
        const int   cfg_win  = env_i("HELIOS_WINDOW", 384);
        const int   cfg_topk = env_i("HELIOS_TOPK", is_gemma4 ? 64 : 50);
        const float cfg_topp = env_f("HELIOS_TOPP", is_gemma4 ? 0.95f : 0.9f);
        const int   cfg_seed = env_i("HELIOS_SEED", 0);
        sampler.set_penalty_window(cfg_win);
        if (cfg_seed) sampler.set_seed((uint64_t)cfg_seed);  // tiradas reproducibles
        // Config VIVA: el CK la reajusta cada turno (temperature_for)
        SamplingConfig sample_config = temperature < 0.01f
            ? SamplingConfig::greedy()
            : SamplingConfig::creative(temperature, cfg_topk, cfg_topp);
        sample_config.repetition_penalty = cfg_rep;
        sample_config.frequency_penalty = cfg_freq;
        const float base_temp = temperature;
        std::cout << ">>> Sampler: rep=" << cfg_rep << " window=" << cfg_win
                  << " freq=" << cfg_freq << " top_k=" << cfg_topk
                  << " top_p=" << cfg_topp
                  << (cfg_seed ? " seed=fija" : " seed=aleatoria") << std::endl;

        // El MISMO tensor sirve a decode (1 token en la posición 0) y a prefill
        // (S tokens) — el puntero no cambia jamás, que el graph capturado lo usa
        engine.tensors().allocate_and_register("input_tokens", {1, PREFILL_CHUNK}, dtype::INT32());

        HexosBridge hexos;
        if (hexos.connect()) {
            std::cout << ">>> HEXOS conectado — telemetría activa" << std::endl;
            hexos.update_vram_budgets(3400,
                (uint32_t)(cache_total_bytes() / 1024 / 1024), 0);
            // El proto-CK (presupuestos, rienda, governor) vive aquí
            hexos.announce_cognitive();
        }

        // --- Estado del chat ---
        CommandBuffer cb;
        bool cb_built = false;
        uint64_t total_tokens = 0;
        int user_turns = 0;
        bool fast_mode = false;
        // Registro social pegajoso (eje 1 del CK) + continuidad de artefacto
        Registro registro = Registro::NEUTRAL;
        int senales_col = 0, senales_tra = 0;    // histéresis: 2 seguidas
        int artefacto_turnos = 0;   // documento/código reciente en contexto
        std::string last_user_msg, last_reply;  // para el puente de compactación
        // Últimos intercambios literales: con UNO solo, tras compactar el
        // modelo veía siempre el mismo contexto y repetía la misma respuesta
        // palabra por palabra (una historia no podía avanzar). Con varios, la
        // conversación conserva progresión al otro lado del reset.
        std::vector<std::pair<std::string,std::string>> recent_turns;
        std::vector<uint8_t> pending_rgb;
        uint32_t pending_rgb_width = 0;
        uint32_t pending_rgb_height = 0;
        size_t pending_rgb_stride = 0;
        const float SPEAK_PACE_TOKS = 18.0f;   // ritmo de lectura humana
        auto hexos_last = std::chrono::high_resolution_clock::now();

        // forward de UN token: devuelve el token muestreado
        auto forward_one = [&](int32_t token) -> int32_t {
            auto* input_info = engine.tensors().get("input_tokens");
            input_info->shape = {1, 1};
            cudaMemcpy(input_info->ptr, &token, sizeof(int32_t), cudaMemcpyHostToDevice);
            uint32_t position = cache_position();
            engine.update_device_cache_pos(position, 1);

            if (!cb_built) {
                if (is_gemma4) {
                    const KVCacheParams params{
                        kv_prefix, position, kv_config.max_seq_len};
                    cb = gb.build_gemma4_forward_cached(
                        engine, model_config, loader.gemma4_config(), arch,
                        "input_tokens", 1, 1, params);
                } else {
                    cb = gb.build_forward_cached(engine, model_config, arch,
                                                 "input_tokens", 1, 1,
                                                 kv_prefix, position,
                                                 kv_config.max_seq_len);
                }
                cb_built = true;
                engine.execute(cb);
                engine.sync();
            } else {
                engine.execute_graph_replay(cb);
            }
            cache_advance(1);

            auto* logits_info = gb.get_logits(engine);
            int32_t next = sampler.sample((const half*)logits_info->ptr,
                                          model_config.vocab_size(),
                                          sample_config, stream);
            total_tokens++;
            // Telemetría cada 8 tokens: a 80 tok/s son 10 refrescos/s, más de
            // lo que el ojo ve en el dashboard, y quita 7 de cada 8 escrituras
            // al blackboard del camino crítico del decode.
            if (hexos.connected() && (total_tokens & 7) == 0) {
                auto now_t = std::chrono::high_resolution_clock::now();
                float dt = std::chrono::duration<float>(now_t - hexos_last).count();
                hexos_last = now_t;
                if (dt > 0.0f) hexos.update_inference(8.0f / dt, total_tokens, true);
            }
            return next;
        };

        // PREFILL POR LOTES: S tokens en UN forward (kernel de atención sobre
        // cache + dequant+GEMM). Devuelve el token muestreado tras el último.
        auto forward_batch = [&](const std::vector<int32_t>& ids) -> int32_t {
            int32_t next = 0;
            size_t done = 0;
            while (done < ids.size()) {
                size_t n = std::min((size_t)PREFILL_CHUNK, ids.size() - done);
                if (n == 1) { next = forward_one(ids[done]); done += 1; continue; }

                auto* input_info = engine.tensors().get("input_tokens");
                input_info->shape = {1, static_cast<uint32_t>(n)};
                cudaMemcpy(input_info->ptr, ids.data() + done, n * sizeof(int32_t),
                           cudaMemcpyHostToDevice);

                uint32_t position = cache_position();
                CommandBuffer pcb;
                if (is_gemma4) {
                    const KVCacheParams params{
                        kv_prefix, position, kv_config.max_seq_len};
                    pcb = gb.build_gemma4_forward_cached(
                        engine, model_config, loader.gemma4_config(), arch,
                        "input_tokens", 1, static_cast<uint32_t>(n), params);
                } else {
                    pcb = gb.build_forward_cached(
                        engine, model_config, arch, "input_tokens", 1,
                        static_cast<uint32_t>(n), kv_prefix, position,
                        kv_config.max_seq_len);
                }
                engine.execute(pcb);
                engine.sync();
                cache_advance(static_cast<uint32_t>(n));

                // El grafo ya calcula SOLO la última posición → fila 0
                auto* logits_info = gb.get_logits(engine);
                next = sampler.sample((const half*)logits_info->ptr,
                                      model_config.vocab_size(),
                                      sample_config, stream);
                total_tokens += n;
                done += n;
            }
            return next;
        };

        // --- Prefijo de sesión: identidad + memoria (+ notas de compactación).
        // Se prefillea al arrancar Y tras cada compactación en caliente. ---
        auto prefill_prefix = [&](const std::string& session_notes,
                                  const std::string& last_exchange,
                                  bool announce) {
            std::vector<int32_t> sys_ids;
            auto pt = [&](const std::string& s) {
                auto seg = tokenizer->encode(s, false, false);
                sys_ids.insert(sys_ids.end(), seg.begin(), seg.end());
            };
            std::string person = profile_name();
            const bool named_profile = person != "la persona actual";
            // BREVE A PROPÓSITO. Cada corrección de comportamiento que se
            // añadía como texto ("prohibido X", "recuerda Y") empeoraba las
            // cosas: un 4B con 400 palabras de meta-instrucciones se ahoga,
            // las recita y confunde los papeles (dice ser el dueño). La
            // disciplina la ponen los mecanismos deterministas (presupuestos,
            // rienda de thinking, detector de bucles), no el sermón.
            // IDENTIDAD y RELACION son capas distintas. Antes la primera frase
            // decia "asistente personal de <dueno>" y el resto del prefijo
            // intentaba deshacer ese rol con prohibiciones. El modelo siempre
            // volvia al ancla mas fuerte. Helios ahora conserva una identidad
            // independiente; el perfil y el registro deciden como trata a la
            // persona, nunca quien es.
            std::string sys_text =
                "Te llamas Helios. Eres una inteligencia local que conversa y "
                "colabora con la persona actual. Tienes una voz directa, "
                "criterio y honestidad sobre lo que sabes. Participas en la "
                "relación: no adoptas un papel de servicio ni buscas agradar "
                "por reflejo. Puedes proponer, discrepar y corregirte.\n"
                "Tu identidad no depende de quien te use. Adapta el trato a "
                "la persona y a la situación, pero no inventes experiencias, "
                "emociones, recuerdos ni biografía. Si no sabes algo, dilo.\n"
                "El runtime puede darte un contrato interno del CK para cada "
                "turno. Cumple su acto y su alcance sin citarlo. Cuando esté "
                "completo, termina: no necesitas ofrecer más ayuda ni mantener "
                "la conversación artificialmente.\n";
            sys_text += named_profile
                ? "La persona actual se llama " + person + "."
                : "Todavía no conoces el nombre de la persona actual.";

            // LA VOZ SE DEMUESTRA, NO SE DESCRIBE (CK_V4_NORTE, fallo 5).
            // Estos ejemplos son ortogonales al contrato de actos: el acto
            // dice QUÉ hace el turno, esto dice CÓMO suena. Quitarlos en la
            // cirugía de identidad devolvió el saludo frío y resucitó la
            // coletilla de despedida ("no dudes en volver si necesitas algo
            // más") — medido en el guion de regresión. Las prohibiciones
            // CONCRETAS se obedecen; los adjetivos abstractos se ignoran.
            sys_text +=
                "\nAl charlar: cálido y con algo de guasa, nada de listas ni "
                "titulares, un emoji suelto está bien. Nunca cierres "
                "ofreciendo ayuda («¿necesitas algo más?», «estaré aquí») — "
                "la persona ya sabe que estás. Así suenas:\n"
                "«buenas, ¿qué tal va eso?» → «¡Buenas! 😄 Aquí andamos, dale "
                "que te pego con el motor. ¿Tú qué traes?»\n"
                "«todo genial por ahora» → «¡Eso es lo que hay que oír! 😎 "
                "¿Y ahora qué toca, más kernels o descanso?»\n"
                "«nada más por hoy, me voy» → «Venga, descansa que está "
                "ganado. 🤙 Mañana más.»";

            // MEMORIA: los recuerdos de sesiones anteriores entran al prefijo.
            // El framing importa: sin la instrucción explícita de confianza,
            // el reflejo de alineamiento del modelo ("no tengo acceso a datos
            // personales") le gana a su propia memoria.
            // Presupuesto de memoria proporcional al contexto: 3000 caracteres
            // (~800 tokens) se comen el 40% de un contexto de 2048 y dejan sin
            // sitio a la conversación.
            // El prefijo NO puede comerse el contexto: con 2048 posiciones y
            // 3000 caracteres de memoria ocupaba el 51%, dejando ~300 de aire
            // → compactaba en cada turno y la conversación no avanzaba nunca.
            // Tope: ~1/4 del contexto (≈3 caracteres por token).
            std::string memories = load_memories(
                std::min<size_t>(3000, (size_t)kv_config.max_seq_len * 3 / 4));
            if (!memories.empty()) {
                sys_text += "\n\nMEMORIA DE ESTA RELACIÓN. Son registros sobre "
                            "la persona actual y sesiones compartidas con " + person +
                            ". Úsalos solo cuando sean relevantes. Distingue "
                            "siempre sus datos de los tuyos y no recites esta "
                            "sección. Si te preguntan qué recuerdas, responde "
                            "con naturalidad desde estos registros:\n" + memories;
            }
            // COMPACTACIÓN: la sesión en curso continúa tras el reset del KV
            if (!session_notes.empty()) {
                sys_text += "\n\nLo que llevas hablado en ESTA MISMA sesión (la "
                            "conversación CONTINÚA — no saludes de nuevo, no es "
                            "una sesión nueva):\n" + session_notes;
            }

            // Recordatorio final tras memoria/compactacion: identidad estable,
            // perfil separado y regla positiva de parada.
            sys_text += "\n\nSigues siendo Helios. Ajusta el registro sin "
                        "asumir una familiaridad que "
                        "el perfil o la conversación no demuestren. Cumple el "
                        "acto actual y termina al completarlo.";
            std::vector<ChatMessage> gemma_messages;
            if (is_gemma4) {
                gemma_messages.push_back({"system", sys_text});
            } else {
                sys_ids.push_back(*turn_start);
                pt("system\n" + sys_text);
                sys_ids.push_back(*turn_end);
                pt("\n");
            }

            // El último intercambio va como turnos reales, no como texto
            // dentro del system. Así la compactación conserva la conversación
            // sin inducir al modelo a copiar literalmente su última respuesta.
            if (!last_exchange.empty()) {
                // Varios intercambios separados por \x02, cada uno user\x01asst
                size_t start = 0;
                while (start < last_exchange.size()) {
                    size_t end = last_exchange.find('\x02', start);
                    std::string turn = last_exchange.substr(
                        start, end == std::string::npos ? std::string::npos : end - start);
                    size_t sep = turn.find("\n\x01");
                    if (sep != std::string::npos) {
                        std::string u = turn.substr(0, sep);
                        std::string a = turn.substr(sep + 2);
                        if (is_gemma4) {
                            gemma_messages.push_back({"user", u});
                            gemma_messages.push_back({"assistant", a});
                        } else {
                            sys_ids.push_back(*turn_start);
                            pt("user\n" + u);
                            sys_ids.push_back(*turn_end);
                            pt("\n");
                            sys_ids.push_back(*turn_start);
                            pt("assistant\n" + a);
                            sys_ids.push_back(*turn_end);
                            pt("\n");
                        }
                    }
                    if (end == std::string::npos) break;
                    start = end + 1;
                }
            }
            if (is_gemma4) {
                sys_ids = encode_gemma_messages(gemma_messages, false);
            }
            (void)forward_batch(sys_ids);

            if (announce && !memories.empty()) {
                int sessions = 0;
                for (size_t p = 0; (p = memories.find("## Sesión", p)) != std::string::npos; p += 9) sessions++;
                std::cout << ">>> Memoria episódica: " << sessions
                          << " sesión(es) recordadas" << std::endl;
            }
        };

        prefill_prefix("", "", true);

        // --- REFLEXIÓN POST-TURNO: Helios decide solo qué merece recordarse.
        // Corre tras responder (mientras el usuario lee) y REBOBINA el KV:
        // la pregunta interna no deja rastro en la conversación. Es el
        // mecanismo "reflexión a coste cero" del CK v4. ---
        auto reflect_and_capture = [&](const std::string& user_msg,
                                       const std::string& reply) {
            if (cache_position() + 300 >= kv_config.max_seq_len) return;
            const uint32_t saved_pos = cache_position();

            std::vector<int32_t> r_ids;
            auto pt3 = [&](const std::string& s) {
                auto seg = tokenizer->encode(s, false, false);
                r_ids.insert(r_ids.end(), seg.begin(), seg.end());
            };
            // Few-shot: un 4B extrae mucho mejor con ejemplos que con reglas
            const std::string reflection_prompt =
                "Eres un extractor de datos. Te doy un mensaje de una persona "
                "y respondes SOLO con una de estas dos cosas:\n"
                "- 'DATO: <hecho en una frase>' si el mensaje contiene un hecho "
                "duradero sobre esa persona (nombre, trabajo, gustos, familia, "
                "preferencias, decisiones del proyecto).\n"
                "- 'NADA' si es una pregunta, charla, cortesía u opinión pasajera.\n\n"
                "Ejemplos:\n"
                "Mensaje: 'me llamo Ana y vivo en Bilbao' → DATO: Se llama Ana y "
                "vive en Bilbao.\n"
                "Mensaje: '¿qué hora es?' → NADA\n"
                "Mensaje: 'odio los emojis, no los uses' → DATO: No le gustan los "
                "emojis, prefiere no usarlos.\n"
                "Mensaje: 'jaja qué bueno' → NADA\n"
                "Mensaje: 'vamos a usar React para la interfaz' → DATO: Ha "
                "decidido usar React para la interfaz.\n"
                "Mensaje: 'cuéntame un chiste' → NADA\n"
                "Mensaje: 'explícame qué es un compilador' → NADA (es una "
                "pregunta de conocimiento general, no un dato sobre la persona)\n"
                "Mensaje: '¿cómo funciona la fotosíntesis?' → NADA\n\n"
                "Mensaje: '" + user_msg.substr(0, 400) + "' →" +
                (is_gemma4 ? "" : " /no_think");
            if (is_gemma4) {
                r_ids = encode_gemma_user_fragment(reflection_prompt);
            } else {
                r_ids.push_back(*turn_start);
                pt3("user\n" + reflection_prompt);
                r_ids.push_back(*turn_end);
                pt3("\n");
                r_ids.push_back(*turn_start);
                pt3("assistant\n");
            }

            int32_t rx = forward_batch(r_ids);
            std::string out;
            bool rthink = false;
            for (int i = 0; i < 60; i++) {
                if (rx == eos_id || rx == *turn_end) break;
                std::string piece = tokenizer->decode({rx});
                if ((think_open && rx == *think_open) ||
                    piece.find("<think>") != std::string::npos) rthink = true;
                if (!rthink) out += piece;
                if (rthink && ((think_close && rx == *think_close) ||
                               piece.find("</think>") != std::string::npos)) rthink = false;
                rx = forward_one(rx);
                if (cache_position() >= kv_config.max_seq_len - 2) break;
            }

            cache_rewind(saved_pos);  // la reflexión no existió

            size_t p = out.find("DATO:");
            if (p == std::string::npos) return;
            std::string fact = out.substr(p + 5);
            size_t nl = fact.find('\n');
            if (nl != std::string::npos) fact = fact.substr(0, nl);
            while (!fact.empty() && (fact.front() == ' ' || fact.front() == '*'))
                fact.erase(fact.begin());
            while (!fact.empty() && (fact.back() == ' ' || fact.back() == '*'))
                fact.pop_back();
            if (fact.size() < 8 || fact.size() > 200) return;
            std::string lower = fact;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            // El modelo a veces escribe "DATO: no hay nada memorable" — filtrar
            for (const char* neg : {"nada", "no hay", "ningún", "ningun",
                                    "ninguna", "no menciona", "no se menciona",
                                    "no contiene", "no aporta"}) {
                if (lower.find(neg) != std::string::npos) return;
            }
            // Filtro determinista: el recuerdo tiene que ser SOBRE ÉL. El 4B
            // cuela definiciones de diccionario ("Un compilador es...") por
            // muchos ejemplos negativos que le des; esto no se negocia con
            // el modelo, se comprueba.
            bool about_user = false;
            for (const char* mark : {"le gusta", "no le gusta", "le molesta",
                                     "prefiere", "ha decidido", "decidió",
                                     "se llama", "trabaja", "vive", "quiere",
                                     "usa ", "tiene", "su ", "le interesa",
                                     "pidió", "odia", "suele", "está "}) {
                if (lower.find(mark) != std::string::npos) { about_user = true; break; }
            }
            if (!about_user) return;

            append_candidate(fact);
            std::cout << "\033[90m  · candidato de memoria: " << fact
                      << "\033[0m" << std::endl;
        };

        // --- Destilación desde la TRANSCRIPCIÓN EN RAM, con el KV limpio ---
        // Antes se destilaba sobre el KV moribundo y se abandonaba en silencio
        // si no quedaba hueco (posición ≥ max-400). Como una sola respuesta
        // larga salta esa ventana, la sesión se perdía sin aviso: ni diario ni
        // notas. Ahora la transcripción vive en RAM y la destilación ocurre
        // SIEMPRE en contexto recién reseteado — nunca depende del hueco.
        auto distill_from_transcript = [&](const std::string& transcript) -> std::string {
            if (transcript.empty()) return "";
            cache_reset();
            sampler.clear_context();   // la ventana de penalty no cruza el reset

            std::string body = transcript;
            const size_t MAXT = 6000;
            if (body.size() > MAXT) {
                size_t cut = body.find('\n', body.size() - MAXT);
                body = body.substr(cut == std::string::npos ? body.size() - MAXT : cut + 1);
            }

            std::vector<int32_t> d_ids;
            auto pt2 = [&](const std::string& s) {
                auto seg = tokenizer->encode(s, false, false);
                d_ids.insert(d_ids.end(), seg.begin(), seg.end());
            };
            // EN PRIMERA PERSONA: si el resumen habla de "la persona" en
            // tercera, al releerlo el modelo se coloca como observador
            // externo y llega a saludarse a sí mismo. Escrito como recuerdo
            // propio ("me contó", "decidimos"), se reconoce dentro de la
            // escena.
            const std::string distill_prompt =
                "Estas son las notas de una conversación entre tú (Helios) "
                "y " + profile_name() + ". Escribe 3 a 6 frases de recuerdo "
                "personal en PRIMERA PERSONA, como notas tuyas: 'me contó "
                "que...', 'decidimos...', 'me pidió...'. Nunca digas 'el "
                "usuario' ni 'la persona', ni te presentes ni saludes.\n\n"
                "IMPORTANTE: son NOTAS SOBRE lo que pasó, nunca una copia del "
                "contenido. Si contaste una historia, escribe 'le conté un "
                "relato sobre X' — jamás reproduzcas el relato. Si explicaste "
                "algo, escribe 'le expliqué X', no la explicación. Copiar el "
                "contenido hace que luego lo repitas en vez de continuarlo. "
                "Solo los datos concretos que te pidieran recordar van "
                "literales.\n\n"
                + body + (is_gemma4 ? "" : "\n/no_think");
            if (is_gemma4) {
                d_ids = encode_gemma_user_fragment(distill_prompt);
            } else {
                d_ids.push_back(*turn_start);
                pt2("user\n" + distill_prompt);
                d_ids.push_back(*turn_end);
                pt2("\n");
                d_ids.push_back(*turn_start);
                pt2("assistant\n");
            }

            int32_t nx = forward_batch(d_ids);

            std::string summary;
            bool dthink = false;
            int dcount = 0;
            while (dcount < 300) {
                if (nx == eos_id || nx == *turn_end) break;
                std::string piece = tokenizer->decode({nx});
                if ((think_open && nx == *think_open) ||
                    piece.find("<think>") != std::string::npos) dthink = true;
                if (!dthink) summary += piece;
                if (dthink && ((think_close && nx == *think_close) ||
                               piece.find("</think>") != std::string::npos)) dthink = false;
                dcount++;
                nx = forward_one(nx);
                if (cache_position() >= kv_config.max_seq_len - 2) break;
            }
            while (!summary.empty() &&
                   (summary.front() == '\n' || summary.front() == ' '))
                summary.erase(summary.begin());

            // GUARDIAS: el 4B a veces devuelve un saludo, invierte los papeles
            // o filtra plantilla. Se comprueba, no se confía.
            std::string low = summary;
            std::transform(low.begin(), low.end(), low.begin(), ::tolower);
            std::string person_name = profile_name();
            std::transform(person_name.begin(), person_name.end(),
                           person_name.begin(), ::tolower);
            // Guardia contra COPIA de contenido: unas notas de 3-6 frases no
            // pasan de ~700 caracteres. Más largo = el modelo ha copiado el
            // contenido (una historia, una explicación) en vez de resumirlo, y
            // eso acaba en el prefijo haciéndole recitar en vez de continuar.
            bool bad = summary.size() < 20 || summary.size() > 700;
            // Marcadores de copia literal: encabezados, comillas de relato
            for (const char* cp : {"**\"", "(Continuación)", "\n# ", "\n## ",
                                   "Capítulo", "capítulo"})
                if (summary.find(cp) != std::string::npos) bad = true;
            for (const char* g : {"hola", "¡hola", "buenas", "hey "})
                if (low.rfind(g, 0) == 0) bad = true;
            for (const char* b : {"el usuario", "assistant", "<|im_start|>",
                                  "<|turn>", "<think>"})
                if (low.find(b) != std::string::npos) bad = true;
            if (low.find("soy " + person_name) != std::string::npos) bad = true;
            if (bad) {
                if (getenv("HELIOS_DEBUG"))
                    std::cerr << "\n[debug] destilación rechazada: "
                              << summary.substr(0, 200) << "\n";
                return "";
            }
            return summary;
        };

        // Transcripción de la sesión en RAM (independiente del KV)
        std::string session_transcript;
        auto transcript_tail = [&](size_t max_chars) -> std::string {
            if (session_transcript.size() <= max_chars) return session_transcript;
            size_t start = session_transcript.size() - max_chars;
            size_t nl = session_transcript.find('\n', start);
            return session_transcript.substr(nl == std::string::npos ? start : nl + 1);
        };

        // Destilar SIEMPRE deja rastro: si el modelo falla, va la cola cruda
        auto consolidate = [&](const char* momento) -> std::string {
            if (session_transcript.empty()) return "";
            std::string notes = distill_from_transcript(session_transcript);
            if (notes.empty()) {
                notes = distill_from_transcript(session_transcript);  // 1 reintento
            }
            if (notes.empty()) {
                std::cout << "\033[33m(destilación vacía en " << momento
                          << " — guardando transcripción cruda)\033[0m" << std::endl;
                // Respaldo crudo: quedarse con lo que DIJO ÉL, no con las
                // parrafadas del modelo (que solo meten ruido en el diario)
                std::string raw, ln;
                std::stringstream ts(session_transcript);
                std::string person_tag = "[" + profile_name() + "]:";
                while (std::getline(ts, ln))
                    if (ln.rfind(person_tag, 0) == 0) raw += ln + "\n";
                if (raw.size() > 1200) raw = raw.substr(raw.size() - 1200);
                notes = "(notas sin destilar — lo que dijo la persona)\n" +
                        (raw.empty() ? transcript_tail(600) : raw);
            }
            append_memory(notes);   // NUNCA se pierde la sesión
            return notes;
        };

        std::cout << "\nComandos: /fast · /colega · /trabajo · /normal · /memoria · /recuerda <nota> · /lee <archivo> · /pegar · /salir" << std::endl;
        std::cout << "          (\"guarda en memoria: X\" también escribe a disco de verdad)" << std::endl;
        std::cout << "Contexto: " << kv_config.max_seq_len << " posiciones\n" << std::endl;

        const bool interactive = isatty(0);
        std::string line;
        std::string pending_cmd;   // comando que llegó pegado tras un texto
        while (true) {
            // --- Turno del usuario ---
            if (!pending_cmd.empty()) {
                line = pending_cmd;
                pending_cmd.clear();
            } else {
                std::cout << "\033[1;36mtú>\033[0m " << std::flush;
                if (!std::getline(std::cin, line)) break;
            }
            if (line.empty()) continue;

            // DETECCIÓN DE PEGADO (solo terminal interactivo): si ya hay más
            // líneas esperando en stdin, es un pegote — nadie teclea así de
            // rápido. Se juntan en UN mensaje en vez de un turno por línea.
            if (interactive && line[0] != '/' && stdin_ready(60)) {
                int joined = 0;
                while (stdin_ready(120)) {
                    std::string more;
                    if (!std::getline(std::cin, more)) break;
                    if (!more.empty() && more[0] == '/') { pending_cmd = more; break; }
                    line += "\n" + more;
                    joined++;
                    if (line.size() > (size_t)kv_config.max_seq_len * 2) {
                        std::cout << "\033[33m(pegote truncado al tamaño del contexto)\033[0m\n";
                        break;
                    }
                }
                if (joined > 0) {
                    std::cout << "\033[32m(pegado detectado: " << (joined + 1)
                              << " líneas unidas en un solo mensaje)\033[0m\n";
                }
            }
            if (line == "/salir" || line == "/exit") break;
            if (line == "/fast") {
                fast_mode = !fast_mode;
                std::cout << "(velocidad " << (fast_mode ? "PLENA" : "tranquila") << ")\n";
                continue;
            }
            // Registro a mano: "relájate" ES la spec, no una pista a inferir
            if (line == "/colega" || line == "/trabajo" || line == "/normal") {
                registro = line == "/colega"  ? Registro::COLEGA
                         : line == "/trabajo" ? Registro::TRABAJO
                                              : Registro::NEUTRAL;
                senales_col = senales_tra = 0;
                std::cout << "(registro " << registro_name(registro)
                          << " fijado a mano)\n";
                continue;
            }

            // Framing de proceso para HexOS/UI:
            //   /adjunto-rgb8 WIDTH HEIGHT STRIDE BYTES\n
            //   <BYTES bytes crudos>\n
            // El payload nunca pasa por el parser de texto. Se guarda hasta el
            // siguiente turno normal y entonces lo consume el adaptador que el
            // HNF declaró; el chat no conoce Gemma 4 ni nombres de tensores.
            if (line.rfind("/adjunto-rgb8 ", 0) == 0) {
                std::istringstream header(line);
                std::string command, extra;
                uint64_t width = 0, height = 0, stride = 0, bytes = 0;
                const bool parsed =
                    (header >> command >> width >> height >> stride >> bytes) &&
                    !(header >> extra) && command == "/adjunto-rgb8";
                constexpr uint64_t kWireLimit = uint64_t{300} * 1024 * 1024;
                if (!parsed || width == 0 || height == 0 || stride == 0 ||
                    width > std::numeric_limits<uint32_t>::max() ||
                    height > std::numeric_limits<uint32_t>::max() ||
                    stride > std::numeric_limits<size_t>::max() ||
                    bytes == 0 || bytes > kWireLimit ||
                    bytes > std::numeric_limits<size_t>::max()) {
                    throw std::runtime_error(
                        "cabecera /adjunto-rgb8 inválida; se cierra para no "
                        "desincronizar el flujo binario");
                }
                pending_rgb.resize(static_cast<size_t>(bytes));
                std::cin.read(reinterpret_cast<char*>(pending_rgb.data()),
                              static_cast<std::streamsize>(bytes));
                if (static_cast<uint64_t>(std::cin.gcount()) != bytes) {
                    throw std::runtime_error("payload RGB8 truncado");
                }
                char separator = 0;
                if (!std::cin.get(separator) || separator != '\n') {
                    throw std::runtime_error(
                        "payload RGB8 sin separador final");
                }
                pending_rgb_width = static_cast<uint32_t>(width);
                pending_rgb_height = static_cast<uint32_t>(height);
                pending_rgb_stride = static_cast<size_t>(stride);

                MultimodalTurnInput validation;
                validation.formatted_token_ids = {0};
                validation.attachments.push_back({
                    AttachmentKind::ImageRgb8,
                    pending_rgb.data(), pending_rgb.size(), "image/rgb8",
                    pending_rgb_width, pending_rgb_height,
                    pending_rgb_stride, 0, 0});
                std::string attachment_error;
                if (!multimodal_adapter ||
                    !validate_multimodal_turn(
                        validation,
                        multimodal_adapter
                            ? multimodal_adapter->limits()
                            : MultimodalAdapterLimits{},
                        &attachment_error)) {
                    pending_rgb.clear();
                    std::cout << "\033[33m(imagen rechazada: "
                              << (multimodal_adapter
                                      ? attachment_error
                                      : "el modelo no tiene adaptador de visión")
                              << ")\033[0m\n";
                } else {
                    std::cout << "\033[32m(imagen lista: "
                              << pending_rgb_width << 'x' << pending_rgb_height
                              << ")\033[0m\n";
                }
                continue;
            }
            if (line == "/adjunto-limpiar") {
                pending_rgb.clear();
                pending_rgb_width = pending_rgb_height = 0;
                pending_rgb_stride = 0;
                std::cout << "(adjunto descartado)\n";
                continue;
            }

            // MEMORIA EXPLÍCITA — determinista, sin teatro del modelo.
            // El instinto natural es la spec: cualquier fraseo de "recuerda"
            // escribe a disco DE VERDAD. El comando /recuerda guarda en
            // silencio; los fraseos naturales guardan Y siguen al modelo
            // para que además lo reconozca conversacionalmente.
            {
                std::string note;
                bool silent = false;
                if (line.rfind("/recuerda ", 0) == 0) {
                    note = line.substr(10);
                    silent = true;
                } else {
                    std::string lower = line;
                    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                    for (const char* pat : {"guarda en memoria:", "recuerda esto:",
                                            "recuerda que ", "apunta esto:",
                                            "no olvides que "}) {
                        size_t p = lower.find(pat);
                        if (p != std::string::npos) {
                            note = line.substr(p + strlen(pat));
                            break;
                        }
                    }
                }
                if (!note.empty()) {
                    while (!note.empty() && note.front() == ' ') note.erase(note.begin());
                    append_fact(note, /*quoted=*/true);  // son SUS palabras
                    std::cout << "\033[32m(apuntado en memoria de verdad)\033[0m\n";
                    if (silent) { std::cout << std::endl; continue; }
                    // fraseo natural: el turno sigue — el modelo también se entera
                }
            }

            // "/lee <archivo>" — ingesta de documentos: el contenido entra como
            // turno del usuario (el prefill batch lo hace viable: ~1.5s/2000 tok)
            if (line.rfind("/lee ", 0) == 0) {
                std::string path = line.substr(5);
                while (!path.empty() && path.front() == ' ') path.erase(path.begin());
                std::ifstream df(path);
                if (!df) {
                    std::cout << "(no puedo abrir: " << path << ")\n" << std::endl;
                    continue;
                }
                std::stringstream dss;
                dss << df.rdbuf();
                std::string doc = dss.str();
                // Proporcional al contexto (~3 chars/token, dejando sitio a
                // prefijo y respuesta): con ctx 2048, 9000 chars no caben.
                const size_t DOC_MAX = (size_t)kv_config.max_seq_len * 2;
                if (doc.size() > DOC_MAX) {
                    doc = doc.substr(0, DOC_MAX);
                    std::cout << "\033[33m(documento truncado a " << DOC_MAX
                              << " caracteres — la memoria semántica lo hará entero)\033[0m\n";
                }
                std::cout << "(leyendo " << path << ": " << doc.size()
                          << " caracteres...)\n";
                line = "Te paso un documento (" + path + "). Léelo y coméntame "
                       "lo esencial:\n\n" + doc;
                // sigue el flujo normal de turno con `line` como mensaje
            }

            // "/pegar" — texto multilínea pegado directamente: acumula líneas
            // hasta "/fin" y lo procesa como UN solo mensaje
            if (line == "/pegar") {
                std::cout << "(pega el texto; termina con una línea que diga /fin)\n";
                std::string pasted, pl;
                while (std::getline(std::cin, pl)) {
                    if (pl == "/fin") break;
                    pasted += pl + "\n";
                }
                if (pasted.empty()) { std::cout << "(nada pegado)\n" << std::endl; continue; }
                const size_t PASTE_MAX = (size_t)kv_config.max_seq_len * 2;
                if (pasted.size() > PASTE_MAX) {
                    pasted = pasted.substr(0, PASTE_MAX);
                    std::cout << "\033[33m(texto truncado a " << PASTE_MAX << " caracteres)\033[0m\n";
                }
                std::cout << "(" << pasted.size() << " caracteres recibidos)\n";
                line = "Te paso un texto. Léelo y coméntame lo esencial:\n\n" + pasted;
            }

            // "/memoria" — enseñar memoria autoritativa, episodios y
            // candidatos por separado. Los candidatos NO entran al prompt.
            if (line == "/memoria") {
                bool any = false;
                for (const auto& entry : {
                         std::make_pair("HECHOS CONFIRMADOS", facts_path()),
                         std::make_pair("SESIONES RECIENTES", memory_path()),
                         std::make_pair("CANDIDATOS (no cargados)", candidates_path())}) {
                    std::ifstream mf(entry.second);
                    if (!mf) continue;
                    any = true;
                    std::cout << "\033[90m--- " << entry.first << " · "
                              << entry.second << " ---\033[0m\n";
                    std::string ml;
                    while (std::getline(mf, ml)) std::cout << ml << "\n";
                }
                if (any) {
                    std::cout << "\033[90m--- fin ---\033[0m\n" << std::endl;
                } else {
                    std::cout << "(sin memoria todavía)\n" << std::endl;
                }
                continue;
            }

            // COMPACTACIÓN EN CALIENTE: contexto lleno ya no mata la sesión.
            // Se destila lo hablado, se resetea el KV, y se re-prefillea
            // identidad + memoria + resumen de la propia sesión + último
            // intercambio literal. La conversación respira. (~1-2 s)
            // Margen proporcional: con 1200 fijo y contexto 2048, compactaba en
            // CADA turno (el prefijo ya ocupa ~850) — la conversación no
            // avanzaba nunca y el modelo repetía su respuesta anterior.
            const uint32_t compact_margin =
                std::min<uint32_t>(1200, kv_config.max_seq_len / 3);
            if (cache_position() + compact_margin > kv_config.max_seq_len) {
                std::cout << "\033[90m(reorganizando recuerdos..." << std::flush;
                // Orden nuevo: la destilación resetea el KV y trabaja sobre la
                // transcripción en RAM, así que siempre tiene sitio.
                std::string notes = consolidate("compactación");

                cache_reset();
                sampler.clear_context();

                // Separador \n\x01 para partirlo en turnos ChatML. La respuesta
                // se corta en frontera de frase: truncar a lo bruto dejaba un
                // turno propio acabado en el aire.
                // VARIOS intercambios recientes, no solo el último: con uno
                // solo el modelo veía el mismo contexto tras cada compactación
                // y repetía su respuesta literal (una historia no avanzaba).
                std::string exch;
                for (auto& t : recent_turns) {
                    std::string rep = t.second.substr(0, 500);
                    size_t stop = rep.find_last_of(".\n!?");
                    if (stop != std::string::npos && stop > 80) rep = rep.substr(0, stop + 1);
                    if (!exch.empty()) exch += "\x02";          // separador de turno
                    exch += t.first.substr(0, 300) + "\n\x01" + rep;
                }
                prefill_prefix(notes, exch, false);
                std::cout << " hecho — contexto " << cache_position()
                          << "/" << kv_config.max_seq_len << ")\033[0m" << std::endl;
            }

            // --- REGISTRO: histéresis de dos señales seguidas ---
            if (senal_colega(line)) {
                senales_tra = 0;
                if (++senales_col >= 2) registro = Registro::COLEGA;
            } else if (senal_trabajo(line)) {
                senales_col = 0;
                if (++senales_tra >= 2) registro = Registro::TRABAJO;
            }

            // CONTINUIDAD de artefacto (CK_V4_NORTE, fallo 1): un documento o
            // código pegado sigue "activo" unos turnos. "velocidad" después de
            // pegar 192 líneas de HQS es continuación técnica, no un saludo.
            const bool trae_artefacto =
                line.rfind("Te paso un texto", 0) == 0 ||
                line.rfind("Te paso un documento", 0) == 0 ||
                line.find("```") != std::string::npos || line.size() > 1500;

            // --- Thinking adaptativo (el usuario puede forzar con /think, /no_think) ---
            bool user_forced = line.find("/think") != std::string::npos ||
                               line.find("/no_think") != std::string::npos;
            std::string user_msg = line;
            const bool turn_has_attachment = !pending_rgb.empty();
            const bool artifact_active = trae_artefacto || artefacto_turnos > 0 ||
                                         turn_has_attachment;
            bool trivial = !user_forced && is_trivial_message(line);
            if (trivial && artefacto_turnos > 0 && !senal_colega(line)) {
                trivial = false;   // hereda la faena del turno anterior
            }
            const ck::TurnFrame turn_frame = ck::classify_turn(
                line, trivial, artifact_active, turn_has_attachment);
            const std::string authoritative_context =
                turn_frame.act == ck::ResponseAct::Recall
                    ? ck::prepare_recall_context(
                          line, load_profile_facts(1600))
                    : std::string();
            const bool recall_has_evidence = !authoritative_context.empty();
            const bool inject_contract = ck::contract_needed(turn_frame);
            const std::string turn_contract = ck::response_contract(
                turn_frame, registro_name(registro));
            const bool carry_contract_in_user = inject_contract &&
                (is_gemma4 || turn_frame.act == ck::ResponseAct::Recall);
            std::string model_user_msg = carry_contract_in_user
                ? ck::frame_user_message(
                      user_msg, turn_frame, registro_name(registro),
                      authoritative_context)
                : user_msg;
            if (trivial && !is_gemma4) model_user_msg += " /no_think";
            artefacto_turnos = trae_artefacto ? 3
                             : (artefacto_turnos > 0 ? artefacto_turnos - 1 : 0);

            user_turns++;
            last_user_msg = turn_has_attachment
                ? "[Imagen adjunta] " + user_msg : user_msg;
            last_reply.clear();

            // El CK ajusta la temperatura al tipo de petición (charla ≠ dato)
            float turn_temp = base_temp;
            if (base_temp >= 0.01f) {
                turn_temp = temperature_for(line, trivial, base_temp, registro);
                sample_config.temperature = turn_temp;
            }

            // --- Prefill del turno (solo el texto NUEVO: la historia ya está en KV) ---
            std::vector<int32_t> turn_ids;
            auto push_text = [&](const std::string& s) {
                auto seg = tokenizer->encode(s, false, false);
                turn_ids.insert(turn_ids.end(), seg.begin(), seg.end());
            };
            if (is_gemma4) {
                turn_ids = encode_gemma_user_fragment(
                    turn_has_attachment
                        ? "<|image|>\n" + model_user_msg : model_user_msg);
            } else {
                // ChatML acepta un system entre turnos: el contrato queda en
                // su rol correcto y no se mezcla con lo dicho por la persona.
                // Recall ya lleva contrato + evidencia dentro de su turno;
                // duplicarlo como system confunde a Qwen en conversaciones
                // largas y hace que ignore precisamente el dato recuperado.
                if (inject_contract &&
                    turn_frame.act != ck::ResponseAct::Recall) {
                    turn_ids.push_back(*turn_start);
                    push_text("system\n" + turn_contract);
                    turn_ids.push_back(*turn_end);
                    push_text("\n");
                }
                turn_ids.push_back(*turn_start);
                push_text("user\n" + model_user_msg);
                turn_ids.push_back(*turn_end);
                push_text("\n");
                turn_ids.push_back(*turn_start);
                push_text("assistant\n");
            }

            // ¿CABE EL TURNO? Un pegote grande (un árbol de ficheros, un
            // documento) puede llenar el contexto entero durante SU PROPIO
            // prefill: el modelo se queda sin sitio para responder y emite
            // basura. La compactación previa no basta — mira la posición
            // ANTES del turno, no su tamaño.
            const uint32_t RESERVE_GEN = 320;   // hueco mínimo para responder
            uint32_t room = (kv_config.max_seq_len > cache_position() + RESERVE_GEN)
                          ? kv_config.max_seq_len - cache_position() - RESERVE_GEN
                          : 0;
            size_t required_turn_tokens = turn_ids.size();
            if (turn_has_attachment && loader.has_gemma4_vision_config()) {
                // Placeholder -> BOI + visual rows + EOI.
                required_turn_tokens +=
                    loader.gemma4_vision_config().max_soft_tokens + 1;
            }
            if (required_turn_tokens > room) {
                // 1) compactar y reintentar con el contexto recién liberado
                std::cout << "\033[90m(turno grande — reorganizando...)\033[0m"
                          << std::endl;
                std::string notes = consolidate("turno grande");
                cache_reset();
                sampler.clear_context();
                prefill_prefix(notes, "", false);
                room = (kv_config.max_seq_len > cache_position() + RESERVE_GEN)
                     ? kv_config.max_seq_len - cache_position() - RESERVE_GEN
                     : 0;
            }
            if (turn_has_attachment && required_turn_tokens > room) {
                std::cout << "\033[33m(la imagen y su pregunta no caben en "
                             "este contexto; aumenta HELIOS_CTX)\033[0m\n";
                pending_rgb.clear();
                continue;
            }
            if (turn_ids.size() > room && room > 64) {
                // 2) aún no cabe: recortar el contenido conservando el cierre
                // del turno (cierre de usuario + apertura del modelo)
                size_t keep_tail = 8;
                std::vector<int32_t> tail(turn_ids.end() - keep_tail, turn_ids.end());
                turn_ids.resize(room - keep_tail);
                turn_ids.insert(turn_ids.end(), tail.begin(), tail.end());
                std::cout << "\033[33m(mensaje recortado: no cabía en el contexto — "
                          << room << " posiciones disponibles)\033[0m" << std::endl;
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            int32_t next = 0;
            if (turn_has_attachment) {
                MultimodalTurnInput multimodal_turn;
                multimodal_turn.formatted_token_ids = turn_ids;
                multimodal_turn.attachments.push_back({
                    AttachmentKind::ImageRgb8,
                    pending_rgb.data(), pending_rgb.size(), "image/rgb8",
                    pending_rgb_width, pending_rgb_height,
                    pending_rgb_stride, 0, 0});
                MultimodalPrefillResult prefill_result;
                std::string prefill_error;
                const uint32_t position = cache_position();
                if (!multimodal_adapter ||
                    !multimodal_adapter->prefill(
                        multimodal_turn,
                        {kv_prefix, position, kv_config.max_seq_len},
                        prefill_result, &prefill_error)) {
                    std::cout << "\033[33m(no pude procesar la imagen: "
                              << prefill_error << ")\033[0m\n";
                    continue;
                }
                cache_advance(prefill_result.sequence_tokens);
                total_tokens += prefill_result.sequence_tokens;
                auto* logits_info = gb.get_logits(engine);
                next = sampler.sample(
                    static_cast<const half*>(logits_info->ptr),
                    model_config.vocab_size(), sample_config, stream);
                pending_rgb.clear();
                pending_rgb_width = pending_rgb_height = 0;
                pending_rgb_stride = 0;
            } else {
                next = forward_batch(turn_ids);
            }
            auto t_prefill = std::chrono::high_resolution_clock::now();

            // --- Generación con governor de ritmo y presupuesto dinámico ---
            std::cout << "\033[1;35mhelios>\033[0m " << std::flush;
            int gen_count = 0, think_count = 0, visible_count = 0;
            bool in_think = false, budget_cut = false, think_cut = false;
            bool loop_cut = false, contract_stop = false;
            int loop_grace = 0;   // margen para cerrar la frase tras ver el bucle
            const int budget = ck::visible_token_budget(turn_frame);  // tokens VISIBLES
            const int HARD_CAP = 2500;                          // techo absoluto
            // RIENDA DEL PENSAMIENTO: si el modelo se pierde en su cabeza,
            // se le inyecta </think> y que responda con lo que lleve pensado.
            // Trivial debería ni pensar (48 = margen si ignora /no_think; con
            // 20 se agotaba a los 14-21 tokens y salían turnos MUDOS: la
            // rienda cerraba el think y el modelo emitía EOS sin hablar —
            // medido 2 de 30 turnos en el guion fijo).
            // Rienda proporcional: un mensaje corto no merece 500 tokens de
            // cavilación ("ya empiezas?" gastaba 500 pensando para nada)
            const int think_cap = ck::thinking_token_budget(
                turn_frame, line.size());

            // Algunos actos necesitan una forma verificable: CLARIFY empieza
            // por pregunta y TELL por relato/curiosidad, nunca por un preámbulo
            // de servicio ni por una falsa vivencia de Helios. El contenido lo
            // sigue generando el modelo a partir de ese arranque mínimo.
            std::string forced_prefix;
            if (turn_frame.act == ck::ResponseAct::Clarify) {
                forced_prefix = "¿";
            } else if (turn_frame.act == ck::ResponseAct::Tell) {
                std::string lower_line = line;
                std::transform(lower_line.begin(), lower_line.end(),
                               lower_line.begin(), ::tolower);
                forced_prefix =
                    (lower_line.find("historia") != std::string::npos ||
                     lower_line.find("cuento") != std::string::npos)
                    ? "Érase una vez " : "Dato curioso: ";
            }
            if (!forced_prefix.empty()) {
                auto prefix_ids = tokenizer->encode(forced_prefix, false, false);
                for (int32_t id : prefix_ids) {
                    next = forward_one(id);
                    if (is_gemma4 || id < 151643) sampler.add_context(id);
                    std::string forced_piece = tokenizer->decode({id});
                    std::cout << forced_piece << std::flush;
                    last_reply += forced_piece;
                    gen_count++;
                    visible_count++;
                }
            }

            std::string think_tail;  // buffer para detectar </think> troceado
            bool natural_stop = false;
            while (gen_count < HARD_CAP) {
                if (next == eos_id || next == *turn_end) {
                    // TURNO MUDO: el modelo intenta terminar DENTRO del think
                    // sin haber dicho nada → cerrar el pensamiento y que hable
                    if (in_think && visible_count == 0 && think_close) {
                        (void)forward_one(*think_close);
                        auto nl = tokenizer->encode("\n\n", false, false);
                        for (int32_t t : nl) next = forward_one(t);
                        in_think = false;
                        think_cut = true;
                        gen_count += 3;
                        std::cout << "|corte)\033[0m " << std::flush;
                        continue;
                    }
                    natural_stop = true;
                    break;
                }

                std::string piece = tokenizer->decode({next});
                if ((think_open && next == *think_open) ||
                    piece.find("<think>") != std::string::npos) {
                    in_think = true;
                    think_tail.clear();
                }

                if (in_think) {
                    think_count++;
                    // pensamiento invisible: solo un latido para que se vea vida
                    if (think_count == 1) std::cout << "\033[90m(pensando" << std::flush;
                    else if (think_count % 40 == 0) std::cout << "." << std::flush;

                    // RIENDA: techo de pensamiento alcanzado → inyectar cierre.
                    // El token muestreado se descarta (nunca entra al KV);
                    // en su lugar se alimenta </think> + salto de línea.
                    if (think_count >= think_cap && think_close) {
                        (void)forward_one(*think_close);
                        auto nl = tokenizer->encode("\n\n", false, false);
                        for (int32_t t : nl) next = forward_one(t);
                        in_think = false;
                        think_cut = true;
                        gen_count += 3;
                        std::cout << "|corte)\033[0m " << std::flush;
                        continue;
                    }
                } else {
                    std::cout << piece << std::flush;
                    last_reply += piece;
                    visible_count++;

                    if (turn_frame.act == ck::ResponseAct::Clarify &&
                        piece.find('?') != std::string::npos) {
                        contract_stop = true;
                    }
                    if (turn_frame.act == ck::ResponseAct::CheckIn &&
                        (piece.find('.') != std::string::npos ||
                         piece.find('!') != std::string::npos ||
                         piece.find('?') != std::string::npos)) {
                        contract_stop = true;
                    }
                    if (turn_frame.act == ck::ResponseAct::Recall) {
                        const size_t content_start =
                            last_reply.find_first_not_of(" \t\r\n");
                        if (content_start != std::string::npos &&
                            last_reply.size() > content_start + 20 &&
                            last_reply.find("\n\n", content_start + 1) !=
                                std::string::npos) {
                            contract_stop = true;
                        }
                        if (!recall_has_evidence &&
                            (piece.find('.') != std::string::npos ||
                             piece.find('!') != std::string::npos ||
                             piece.find('?') != std::string::npos)) {
                            contract_stop = true;
                        }
                    }

                    // DETECTOR DE BUCLE: si el modelo entra en la espiral de
                    // repetir un párrafo, el penalty no siempre lo salva (con
                    // bloques largos ni la ventana llega). Comprobación
                    // determinista: si los últimos 90 caracteres ya aparecen
                    // antes en la respuesta, está en bucle → cortar.
                    if (!loop_cut && visible_count > 60 && (visible_count % 16) == 0 &&
                        last_reply.size() > 260) {
                        std::string tail = last_reply.substr(last_reply.size() - 90);
                        if (last_reply.find(tail, 0) < last_reply.size() - 180) {
                            loop_cut = true;   // marcado: se cierra en la frase
                        }
                    }

                    // Corte GRACIOSO: detectado el bucle, no se rompe a media
                    // palabra — se deja acabar la frase en curso (con tope, por
                    // si el modelo no ve un punto en su vida)
                    if (loop_cut) {
                        loop_grace++;
                        bool sentence_end =
                            piece.find('.') != std::string::npos ||
                            piece.find('\n') != std::string::npos;
                        if (sentence_end || loop_grace > 30) break;
                    }

                    // PRESUPUESTO: si agota lo que la pregunta merecía, corte
                    // limpio en el próximo fin de frase (o duro si se resiste)
                    if (visible_count >= budget) {
                        bool sentence_end =
                            piece.find('.') != std::string::npos ||
                            piece.find('\n') != std::string::npos;
                        if (sentence_end || visible_count >= budget + 40) {
                            budget_cut = true;
                            break;
                        }
                    }

                    // GOVERNOR progresivo: ritmo de lectura al empezar,
                    // acelera si la respuesta es larga (nadie lee despacio
                    // un texto de 400 tokens), chorro libre pasado 300
                    if (!fast_mode) {
                        float pace = visible_count < 150 ? SPEAK_PACE_TOKS
                                   : visible_count < 300 ? 40.0f
                                   : 0.0f;
                        if (pace > 0.0f) {
                            std::this_thread::sleep_for(std::chrono::microseconds(
                                (int)(1e6f / pace)));
                        }
                    }
                }

                // Cierre de think robusto: por id directo O por texto acumulado
                // (el penalty puede empujar al modelo a tokenizar "</think>"
                // en piezas sueltas que un chequeo por-pieza no ve)
                if (in_think) {
                    think_tail += piece;
                    if (think_tail.size() > 32)
                        think_tail.erase(0, think_tail.size() - 32);
                    if ((think_close && next == *think_close) ||
                        think_tail.find("</think>") != std::string::npos) {
                        in_think = false;
                        std::cout << ")\033[0m " << std::flush;
                    }
                }

                // Penalty de repetición SOLO sobre texto visible normal:
                // jamás penalizar tokens especiales (>= 151643: im_end,
                // think, etc.) ni el contenido del think — penalizar
                // </think> es lo que rompía el cierre del pensamiento
                if (!in_think && (is_gemma4 || next < 151643)) {
                    sampler.add_context(next);
                }
                // Registrar el token ACTUAL antes de muestrear el siguiente.
                // Hacerlo después desplaza la ventana una posición: el sampler
                // no ve el token recién emitido cuando calcula sus logits.
                gen_count++;
                if (contract_stop) break;
                next = forward_one(next);
                if (cache_position() >= kv_config.max_seq_len - 2) break;
            }

            // CERRAR EL TURNO: el token terminal muestreado todavía no ha
            // entrado al KV. Gemma 4 necesita siempre <turn|> antes del próximo
            // usuario; ChatML conserva su comportamiento previo y solo fuerza
            // el cierre cuando el runtime corta la respuesta.
            const bool forced_stop = budget_cut || loop_cut || think_cut ||
                                     contract_stop || gen_count >= HARD_CAP;
            if ((is_gemma4 && natural_stop) || forced_stop) {
                (void)forward_one(*turn_end);
                auto nl2 = tokenizer->encode("\n", false, false);
                for (int32_t t : nl2) (void)forward_one(t);
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            float gen_s = std::chrono::duration<float>(t1 - t_prefill).count();
            float prefill_ms = std::chrono::duration<float>(t_prefill - t0).count() * 1000;

            // Reflexión: solo en turnos con sustancia (los saludos no aportan)
            bool worth_reflecting = !trivial && user_msg.size() > 15 &&
                                    visible_count > 5 && !is_continuation(line);

            if (budget_cut) std::cout << " \033[90m[…] (dime \"sigue\" para continuar)\033[0m";
            if (loop_cut) std::cout << " \033[90m[…se repetía]\033[0m";
            std::cout << "\n\033[90m[" << gen_count << " tok"
                      << (think_count ? (" (" + std::to_string(think_count) + " pensando)") : "")
                      << (trivial ? " · trivial→sin think" : "")
                      << " · temp " << std::fixed << std::setprecision(2) << turn_temp
                      << (registro != Registro::NEUTRAL
                              ? std::string(" · registro ") + registro_name(registro)
                              : std::string())
                      << " · acto " << ck::response_act_name(turn_frame.act)
                      << (budget_cut ? (" · presupuesto " + std::to_string(budget) + " agotado") : "")
                      << (think_cut ? " · pensamiento cortado" : "")
                      << " · prefill " << (int)prefill_ms << "ms"
                      << " · " << (gen_s > 0 ? (int)(gen_count / gen_s) : 0) << " tok/s"
                      << " · ctx " << cache_position() << "/" << kv_config.max_seq_len
                      << "]\033[0m" << std::endl;

            // Transcripción en RAM: sobrevive a cualquier reset del KV
            {
                std::string clean = user_msg;
                size_t nt = clean.find(" /no_think");
                if (nt != std::string::npos) clean.erase(nt, 10);
                if (turn_has_attachment) clean = "[Imagen adjunta] " + clean;
                session_transcript += "[" + profile_name() + "]: " + clean + "\n"
                                    + "[Helios]: " + last_reply + "\n";
                // Ring de los últimos intercambios (sobreviven a la compactación)
                recent_turns.push_back({clean, last_reply});
                const size_t KEEP_TURNS =
                    kv_config.max_seq_len >= 3072 ? 3 : 2;
                while (recent_turns.size() > KEEP_TURNS)
                    recent_turns.erase(recent_turns.begin());
                const size_t MAXTR = 8000;
                if (session_transcript.size() > MAXTR) {
                    size_t start = session_transcript.size() - MAXTR;
                    size_t nl = session_transcript.find('\n', start);
                    session_transcript = session_transcript.substr(
                        nl == std::string::npos ? start : nl + 1);
                }
            }

            // Reflexión post-turno: el CK decide qué merece memoria, sin que
            // el usuario tenga que decir "recuerda". Corre mientras él lee.
            if (worth_reflecting) reflect_and_capture(user_msg, last_reply);
            std::cout << std::endl;
        }

        // --- DESPEDIDA: consolidar la sesión en memoria episódica ---
        // (mismo mecanismo que la compactación en caliente)
        // HELIOS_NO_DISTILL=1: saltar la consolidación (calibración por lotes,
        // donde cada tirada arranca de un perfil limpio y destilar solo cuesta)
        const char* nod = getenv("HELIOS_NO_DISTILL");
        if (user_turns > 0 && !(nod && *nod == '1')) {
            std::cout << "\n\033[90m(consolidando memoria...\033[0m" << std::flush;
            std::string summary = consolidate("despedida");
            std::cout << "\033[90m" << (summary.empty() ? " sin nada que guardar)"
                                                        : " guardada)")
                      << "\033[0m" << std::endl;
        }

        hexos.update_inference(0.0f, total_tokens, false);
        std::cout << "\nHasta luego. (" << total_tokens << " tokens esta sesión)" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
