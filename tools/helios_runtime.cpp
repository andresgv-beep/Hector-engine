// helios_runtime — E1 del plan de evacuación cognitiva.
//
// Mismo motor que helios_chat, eventos tipados en vez de marcadores de
// terminal. NO tiene mente: ni identidad, ni memoria, ni actos, ni registro.
// Recibe mensajes con rol, aplica la plantilla del modelo y emite eventos.
//
// Protocolo: hexos-core/docs/E1_PROTOCOLO_RUNTIME.md
//   stdin  = una petición JSON por línea
//   stdout = un evento JSON por línea, y NADA más
//   stderr = log humano
//
// Un solo `turn` activo. `cancel` y `status` se atienden MIENTRAS genera, así
// que stdin se lee en un hilo aparte: si el bucle de decode leyera stdin, la
// cancelación no podría llegar nunca.

#include <atomic>
#include <cmath>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../src/inference_session.hpp"
#include "mini_json.hpp"

using helios::ChatMessage;
using helios::InferenceSession;
using helios::Model;
namespace mj = helios::mini_json;

namespace {

// --------------------------------------------------------------------------
// Escape de salida. El parseo lo hace mini_json, que respeta cadenas,
// escapes y pares surrogate: el anterior contaba llaves dentro de strings y
// bastaba pegarle código a Héctor para romperlo.
// --------------------------------------------------------------------------

std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\t': o += "\\t";  break;
            case '\r': break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
                else o += static_cast<char>(c);
        }
    }
    return o;
}

// --------------------------------------------------------------------------
// Salida: una línea por evento, y solo por stdout.
// --------------------------------------------------------------------------

std::mutex g_out;

void emit(const std::string& line) {
    std::lock_guard<std::mutex> lock(g_out);
    std::fputs(line.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);   // el cliente lee en streaming: sin flush no hay UI
}

std::string state_json(uint32_t before, uint32_t now, uint32_t max_seq) {
    return "\"model_state\":{\"cache_position_before\":" + std::to_string(before) +
           ",\"cache_position\":" + std::to_string(now) +
           ",\"max_seq_len\":" + std::to_string(max_seq) + "}";
}

void emit_result(const std::string& rid, bool ok, uint32_t before,
                 uint32_t now, uint32_t max_seq) {
    emit("{\"type\":\"result\",\"request_id\":\"" + json_escape(rid) +
         "\",\"ok\":" + (ok ? "true" : "false") + "," +
         state_json(before, now, max_seq) + "}");
}

void emit_error(const std::string& rid, const std::string& code,
                const std::string& msg, uint32_t before, uint32_t now,
                uint32_t max_seq) {
    emit("{\"type\":\"error\",\"request_id\":\"" + json_escape(rid) +
         "\",\"code\":\"" + json_escape(code) +
         "\",\"message\":\"" + json_escape(msg) + "\"," +
         state_json(before, now, max_seq) + "}");
}

// --------------------------------------------------------------------------
// Cola de peticiones: el hilo lector nunca bloquea al de generación.
// --------------------------------------------------------------------------

struct Peticion { std::string tipo, id; mj::Value doc; };

std::mutex g_cola_mtx;
std::deque<Peticion> g_cola;
std::atomic<bool> g_cerrar{false};
std::atomic<bool> g_cancelar{false};
std::string g_turno_activo;          // protegido por g_cola_mtx
std::string g_sesion_generando;      // sesión del turno en vuelo; misma cerradura
std::string g_pausa_pendiente;       // sesión a la que hay que anunciar pausa

// ---------------------------------------------------------------------------
// SESIONES. Héctor ofrece sesiones GENÉRICAS sobre unos mismos pesos: no sabe
// que unas son chats y otras cualquier otra cosa. Quien llama decide.
//
// Solo una genera a la vez, y es a propósito: comparten los buffers de trabajo
// del grafo. Cuando llega un turno para otra sesión, la que estuviera
// generando se cancela en un límite seguro. Durante decode conserva lo
// generado; durante prefill vuelve al inicio efectivo o vacía el anillo si
// ya perdió esa ventana. El estado final indica si hace falta reenviar contexto.
//
// Una sesión pausada no consume GPU porque no ejecuta nada; lo que sí ocupa es
// la VRAM de su KV, y eso no lo arregla pausar. Es el techo real al número de
// chats abiertos, y conviene decirlo en vez de descubrirlo con un OOM.
// ---------------------------------------------------------------------------
constexpr const char* kSesionPorDefecto = "0";

// Solo el hilo dueño la llama, y solo cuando el turno ha terminado de verdad.
void anunciar_pausa(const std::string& sesion) {
    if (sesion.empty()) return;
    emit("{\"type\":\"session_paused\",\"session\":\"" +
         json_escape(sesion) + "\",\"reason\":\"switch\"}");
}
struct Adjunto {
    std::vector<unsigned char> pixels;
    uint32_t width = 0, height = 0;
    size_t stride = 0;
};
std::map<std::string, Adjunto> g_adjuntos;   // protegido por g_cola_mtx

struct CancelPend { std::string id, objetivo; };
std::deque<CancelPend> g_cancel_pend;   // protegido por g_cola_mtx
std::atomic<bool> g_stdin_cerrado{false};

void hilo_lector() {
    std::string linea;
    while (std::getline(std::cin, linea)) {
        if (linea.empty()) continue;
        Peticion p;
        std::string perr;
        if (!mj::Parser::parse(linea, &p.doc, &perr)) {
            // Un cliente que manda basura recibe un error explícito, no un
            // objeto a medias interpretado a ojo.
            emit("{\"type\":\"error\",\"request_id\":\"\",\"code\":"
                 "\"malformed_json\",\"message\":\"" + json_escape(perr) +
                 "\",\"model_state\":{\"cache_position_before\":0,"
                 "\"cache_position\":0,\"max_seq_len\":0}}");
            continue;
        }
        if (const auto* t = p.doc.get("type")) p.tipo = t->str();
        if (const auto* r = p.doc.get("request_id")) p.id = r->str();

        // `cancel` se atiende AQUÍ, no en la cola: si esperase su turno
        // llegaría después de que termine lo que pretende cancelar.
        // `cancel` no espera en la cola —llegaría después de lo que pretende
        // cancelar— pero TAMPOCO lo confirma este hilo: no conoce el estado
        // del KV y podría aceptar una cancelación cuando run_turn() ya
        // terminó y g_turno_activo aún no se ha limpiado. Aquí solo se señala;
        // el hilo dueño de la sesión responde con el estado real.
        // El payload binario va PEGADO a su cabecera: se lee del mismo flujo
        // con read(), no con getline, porque son bytes crudos que pueden
        // contener saltos de línea.
        if (p.tipo == "attachment") {
            const auto num = [&](const char* k) -> long long {
                const auto* v = p.doc.get(k);
                return v ? (long long)v->num(0) : 0;
            };
            const std::string kind = p.doc.get("kind")
                                   ? p.doc.get("kind")->str() : "";
            const long long w = num("width"), h = num("height"),
                            st = num("stride"), by = num("bytes");
            constexpr long long kTope = 300LL * 1024 * 1024;
            const bool geo_ok = kind == "rgb8" && w > 0 && h > 0 &&
                                st >= w * 3 && by == st * h && by <= kTope;
            std::vector<unsigned char> buf;
            if (by > 0 && by <= kTope) {
                buf.resize((size_t)by);
                std::cin.read(reinterpret_cast<char*>(buf.data()), by);
                if (std::cin.peek() == '\n') std::cin.get();
            }
            if (!geo_ok || (long long)buf.size() != by) {
                emit_error(p.id, "invalid_attachment",
                           "geometría RGB8 incoherente o payload incompleto",
                           0, 0, 0);
                continue;
            }
            {
                std::lock_guard<std::mutex> l(g_cola_mtx);
                g_adjuntos[p.id] = Adjunto{std::move(buf), (uint32_t)w,
                                           (uint32_t)h, (size_t)st};
            }
            emit_result(p.id, true, 0, 0, 0);
            continue;
        }

        if (p.tipo == "cancel") {
            std::string objetivo;
            if (const auto* t = p.doc.get("target")) objetivo = t->str();
            bool active = false;
            {
                std::lock_guard<std::mutex> l(g_cola_mtx);
                active = !g_turno_activo.empty() && objetivo == g_turno_activo;
                if (active) {
                    g_cancel_pend.push_back({p.id, objetivo});
                    g_cancelar.store(true);
                }
            }
            if (!active) emit_error(p.id, "not_active", "el turno objetivo no está activo", 0, 0, 0);
            continue;
        }
        if (p.tipo == "attachment_drop") {
            const auto* id = p.doc.get("attachment_id");
            { std::lock_guard<std::mutex> l(g_cola_mtx);
              if (id) g_adjuntos.erase(id->str()); }
            emit_result(p.id, true, 0, 0, 0);
            continue;
        }
        // CONMUTACIÓN. Si llega trabajo para otra sesión mientras una está
        // generando, se cancela la de ahora. La cancelación es la que ya
        // existe: para entre tandas de prefill o en un límite de token de decode.
        // El dueño de la sesión devuelve la posición real del KV tras cancelar.
        //
        // Se señala aquí, desde el lector, para que la espera del usuario sea
        // corta; quien manda de verdad es el hilo dueño, que confirma el
        // estado real cuando su turno termina.
        if (p.tipo == "turn" || p.tipo == "reset" || p.tipo == "session_close") {
            std::string destino = kSesionPorDefecto;
            if (const auto* v = p.doc.get("session")) destino = v->str();
            std::lock_guard<std::mutex> l(g_cola_mtx);
            if (!g_sesion_generando.empty() && g_sesion_generando != destino) {
                // Se APUNTA la pausa; la anuncia el hilo dueño cuando run_turn
                // haya vuelto de verdad. Anunciarla aquí diría "pausada"
                // mientras la sesión todavía está terminando su token, y quien
                // lea la traza deduciría que después de `session_paused` no
                // puede venir texto de A — que es justo lo que pasaría.
                // HELIOS_PAUSA_TEMPRANA es un GRIFO DE SABOTAJE: reproduce el
                // fallo causal —anunciar la pausa al PEDIRLA— para que la
                // batería pueda verse fallar. Sin la variable no cambia nada.
                if (getenv("HELIOS_PAUSA_TEMPRANA")) {
                    emit("{\"type\":\"session_paused\",\"session\":\"" +
                         json_escape(g_sesion_generando) +
                         "\",\"reason\":\"switch\"}");
                } else {
                    g_pausa_pendiente = g_sesion_generando;
                }
                g_cancelar.store(true);
            }
        }

        { std::lock_guard<std::mutex> l(g_cola_mtx); g_cola.push_back(p); }
        if (p.tipo == "shutdown") break;
    }
    g_stdin_cerrado.store(true);
}

}  // namespace

int main(int argc, char** argv) {
    std::string hnf, multimodal;
    uint32_t ctx = 4096;
    float temp = 0.7f;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) hnf = argv[++i];
        else if (a == "--multimodal" && i + 1 < argc) multimodal = argv[++i];
        else if (a == "--ctx" && i + 1 < argc) ctx = (uint32_t)atoi(argv[++i]);
        else if (a == "--temp" && i + 1 < argc) temp = (float)atof(argv[++i]);
        else {
            std::fprintf(stderr, "uso: %s --model <hnf> [--multimodal <hnf>] [--ctx N] [--temp T]\n",
                         argv[0]);
            return 2;
        }
    }
    if (hnf.empty()) {
        std::fprintf(stderr, "falta --model\n");
        return 2;
    }

    // El protocolo manda: por stdout SOLO JSONL. El loader y los kernels
    // imprimen diagnósticos con std::cout ("[MMAP] …", "CUDA Graph …"), así
    // que se desvía el buffer entero a stderr. Los eventos se escriben con
    // fputs directamente al descriptor, así que no les afecta.
    std::cout.rdbuf(std::cerr.rdbuf());

    Model::Config cfg;
    cfg.hnf_path = hnf;
    cfg.multimodal_hnf_path = multimodal;
    cfg.max_seq_len = ctx;
    cfg.temperature = temp;
    std::string err;
    auto pesos = Model::load(cfg, &err);
    if (!pesos) {
        std::fprintf(stderr, "[runtime] no pude cargar el modelo: %s\n", err.c_str());
        return 1;
    }

    // Las sesiones se crean bajo demanda. `unique_ptr` porque InferenceSession
    // no es copiable y el mapa tiene que poder crecer sin invalidar nada.
    std::map<std::string, std::unique_ptr<InferenceSession>> sesiones;
    auto sesion_de = [&](const std::string& id, uint32_t capacity = 0) -> InferenceSession* {
        auto it = sesiones.find(id);
        if (it != sesiones.end()) return it->second.get();
        auto nueva = std::make_unique<InferenceSession>();
        std::string e;
        if (!nueva->attach(pesos, &e, capacity)) {
            std::fprintf(stderr, "[runtime] no pude abrir la sesión %s: %s\n",
                         id.c_str(), e.c_str());
            return nullptr;
        }
        emit("{\"type\":\"session_opened\",\"session\":\"" +
             json_escape(id) + "\"}");
        auto* raw = nueva.get();
        sesiones.emplace(id, std::move(nueva));
        return raw;
    };
    const auto& info = pesos->info();

    emit("{\"type\":\"ready\",\"protocol\":2,\"model\":{\"architecture\":\"" +
         json_escape(info.architecture) + "\",\"max_seq_len\":" +
         std::to_string(info.max_seq_len) +
         // Capacidad EFECTIVA: la sesión solo declara visión si logró crear
         // el adaptador, así que anunciarla ya no es una promesa vacía.
         ",\"multimodal\":" + (info.multimodal ? "true" : "false") +
         ",\"vision_adapter\":" +
         (info.vision_adapter.empty()
              ? std::string("null")
              : "\"" + json_escape(info.vision_adapter) + "\"") + "}}");

    // La sesión por defecto se abre DESPUÉS de `ready`: el protocolo exige que
    // `ready` sea el primer evento, y `session_opened` es un evento.
    if (!sesion_de(kSesionPorDefecto)) return 1;

    std::thread lector(hilo_lector);

    while (!g_cerrar.load()) {
        Peticion p;
        bool hay = false;
        {
            std::lock_guard<std::mutex> l(g_cola_mtx);
            if (!g_cola.empty()) { p = g_cola.front(); g_cola.pop_front(); hay = true; }
        }
        if (!hay) {
            if (g_stdin_cerrado.load()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        // A qué sesión va esto. Sin campo, la de siempre: el protocolo viejo
        // sigue valiendo palabra por palabra.
        std::string sid = kSesionPorDefecto;
        if (const auto* v = p.doc.get("session")) sid = v->str();
        uint32_t capacity = 0;
        if (p.tipo == "session_open") {
            if (const auto* v = p.doc.get("max_seq_len")) {
                const double n = v->num(0);
                if (!std::isfinite(n) || n < 1 || n > info.max_seq_len || n != static_cast<uint32_t>(n)) {
                    emit_error(p.id, "invalid_context", "max_seq_len fuera de rango", 0, 0, info.max_seq_len);
                    continue;
                }
                capacity = static_cast<uint32_t>(n);
            }
            auto existing = sesiones.find(sid);
            if (capacity && existing != sesiones.end() && existing->second->info().max_seq_len != capacity) {
                emit_error(p.id, "session_exists", "la sesión ya tiene otra capacidad", 0, 0, info.max_seq_len);
                continue;
            }
        }
        InferenceSession* ses = sesion_de(sid, capacity);
        if (!ses) {
            emit_error(p.id, "session_unavailable",
                       "no pude abrir la sesión " + sid, 0, 0, info.max_seq_len);
            continue;
        }
        InferenceSession& session = *ses;
        const uint32_t session_capacity = session.info().max_seq_len;

        if (p.tipo == "session_close") {
            if (sid == kSesionPorDefecto) {
                emit_error(p.id, "session_undeletable",
                           "la sesión por defecto no se cierra", 0, 0,
                           session_capacity);
                continue;
            }
            const uint32_t pos = session.cache_position();
            sesiones.erase(sid);
            emit("{\"type\":\"session_closed\",\"session\":\"" +
                 json_escape(sid) + "\"}");
            emit_result(p.id, true, pos, 0, session_capacity);
            continue;
        }
        if (p.tipo == "session_open") {
            emit_result(p.id, true, session.cache_position(),
                        session.cache_position(), session_capacity);
            continue;
        }

        const uint32_t antes = session.cache_position();

        if (p.tipo == "shutdown") {
            emit_result(p.id, true, antes, antes, session_capacity);
            g_cerrar.store(true);
            break;
        }
        if (p.tipo == "status") {
            emit_result(p.id, true, antes, antes, session_capacity);
            continue;
        }
        if (p.tipo == "reset") {
            session.reset();
            emit_result(p.id, true, antes, session.cache_position(), session_capacity);
            continue;
        }
        if (p.tipo != "turn") {
            emit_error(p.id, "unknown_request", "tipo no soportado: " + p.tipo,
                       antes, antes, session_capacity);
            continue;
        }

        std::vector<ChatMessage> messages;
        if (const auto* arr = p.doc.get("messages"); arr && arr->is_array()) {
            for (const auto& m : *arr->array) {
                ChatMessage cm;
                if (const auto* r = m.get("role")) cm.role = r->str();
                if (const auto* c = m.get("content")) cm.content = c->str();
                if (!cm.role.empty()) messages.push_back(cm);
            }
        }
        if (messages.empty()) {
            emit_error(p.id, "empty_turn", "el turno no trae mensajes",
                       antes, antes, session_capacity);
            continue;
        }

        // Por ÁMBITO: una "temperature" escrita dentro del texto del usuario
        // no puede cambiar el muestreo.
        std::vector<InferenceSession::ImageAttachment> adjuntos;
        std::vector<std::string> ids_usados;
        bool adjunto_malo = false;
        if (const auto* arr = p.doc.get("attachment_ids");
            arr && arr->is_array()) {
            std::lock_guard<std::mutex> l(g_cola_mtx);
            for (const auto& v : *arr->array) {
                auto it = g_adjuntos.find(v.str());
                if (it == g_adjuntos.end()) { adjunto_malo = true; break; }
                adjuntos.push_back({it->second.pixels.data(),
                                    it->second.pixels.size(),
                                    it->second.width, it->second.height,
                                    it->second.stride});
                ids_usados.push_back(v.str());
            }
        }
        if (adjunto_malo) {
            emit_error(p.id, "unknown_attachment",
                       "adjunto inexistente o ya consumido",
                       antes, antes, session_capacity);
            continue;
        }

        InferenceSession::GenConfig gen;
        gen.temperature = temp;
        bool progress_requested = false;
        if (const auto* g = p.doc.get("generation")) {
            if (const auto* v = g->get("temperature"))
                gen.temperature = (float)v->num(gen.temperature);
            if (const auto* v = g->get("max_visible_tokens"))
                gen.max_visible_tokens = (int)v->num(gen.max_visible_tokens);
            if (const auto* v = g->get("max_thinking_tokens"))
                gen.max_thinking_tokens = (int)v->num(gen.max_thinking_tokens);
            if (const auto* v = g->get("prefill_progress"))
                progress_requested = v->boo(false);
            if (const auto* v = g->get("preformatted"))
                gen.preformatted = v->boo(false);
            if (const auto* v = g->get("reuse_prefix"))
                gen.reuse_prefix = v->boo(false);
            if (const auto* v = g->get("close_turn"))
                gen.close_turn = v->boo(true);
            if (const auto* arr = g->get("stop_tokens"); arr && arr->is_array())
                for (const auto& v : *arr->array) gen.stop_tokens.push_back(v.str());
        }

        {
            std::lock_guard<std::mutex> l(g_cola_mtx);
            g_cancelar.store(false);
            g_turno_activo = p.id;
            g_sesion_generando = sid;
        }
        if (progress_requested) emit("{\"type\":\"turn_started\",\"request_id\":\"" +
                                     json_escape(p.id) + "\"}");

        std::string acumulado;
        InferenceSession::TurnStats st;
        InferenceSession::FinishReason reason;
        std::string code;

        auto on_text = [&](const std::string& chunk) {
            acumulado += chunk;
            emit("{\"type\":\"text_delta\",\"request_id\":\"" + json_escape(p.id) +
                 "\",\"text\":\"" + json_escape(chunk) + "\"}");
        };
        auto on_think = [&](uint32_t n) {
            emit("{\"type\":\"thinking\",\"request_id\":\"" + json_escape(p.id) +
                 "\",\"tokens\":" + std::to_string(n) + "}");
        };

        auto on_prefill = [&](uint32_t tokens, double ms) {
            emit("{\"type\":\"prefill\",\"request_id\":\"" + json_escape(p.id) +
                 "\",\"tokens\":" + std::to_string(tokens) +
                 ",\"ms\":" + std::to_string(ms) + "}");
        };

        std::string msg;
        InferenceSession::PrefillProgressCallback on_progress;
        if (progress_requested) on_progress = [&](uint32_t done, uint32_t total, double ms) {
            emit("{\"type\":\"prefill_progress\",\"request_id\":\"" + json_escape(p.id) +
                 "\",\"processed_tokens\":" + std::to_string(done) +
                 ",\"total_tokens\":" + std::to_string(total) +
                 ",\"ms\":" + std::to_string(ms) + "}");
        };
        const bool ok = session.run_turn(messages, adjuntos, gen, on_text,
                                         on_think, on_prefill, g_cancelar,
                                         &st, &reason, &code, &msg, on_progress);
        if (ok && !(reason == InferenceSession::FinishReason::Cancelled &&
                    st.generated_tokens == 0)) {
            // Fallo/cancelación sin generación permiten reintentar el adjunto
            // sin volver a subir los píxeles.
            std::lock_guard<std::mutex> l(g_cola_mtx);
            for (const auto& id : ids_usados) g_adjuntos.erase(id);
        }
        // Las cancelaciones se confirman AQUÍ, con el estado real y ya sin
        // carrera: el turno ha terminado y sabemos si de verdad se canceló.
        std::deque<CancelPend> pendientes;
        std::string pausar;
        {
            std::lock_guard<std::mutex> l(g_cola_mtx);
            pendientes.swap(g_cancel_pend);
            g_turno_activo.clear();
            g_sesion_generando.clear();
            pausar.swap(g_pausa_pendiente);
        }
        for (const auto& c : pendientes) {
            const bool aplico = (c.objetivo == p.id) &&
                                reason == InferenceSession::FinishReason::Cancelled;
            emit_result(c.id, aplico, st.cache_position_before,
                        st.cache_position, session_capacity);
        }

        if (!ok) {
            // Si ya emitió, el error debe traer texto, uso y tiempos: el
            // cliente ya pintó ese texto y su propia prueba lo exige.
            const std::string extra = acumulado.empty() ? "" :
                ",\"visible_text\":\"" + json_escape(acumulado) +
                "\",\"usage\":{\"prefill_tokens\":" + std::to_string(st.prefill_tokens) +
                ",\"generated_tokens\":" + std::to_string(st.generated_tokens) +
                ",\"thinking_tokens\":" + std::to_string(st.thinking_tokens) +
                "},\"timings\":{\"prefill_ms\":" + std::to_string(st.prefill_ms) +
                ",\"decode_ms\":" + std::to_string(st.decode_ms) +
                ",\"tokens_per_second\":0.0}";
            emit("{\"type\":\"error\",\"request_id\":\"" + json_escape(p.id) +
                 "\",\"code\":\"" + json_escape(code) +
                 "\",\"message\":\"" + json_escape(msg) + "\"," +
                 state_json(st.cache_position_before, st.cache_position,
                            session_capacity) + extra + "}");
            anunciar_pausa(pausar);
            continue;
        }

        emit("{\"type\":\"completed\",\"request_id\":\"" + json_escape(p.id) +
             "\",\"visible_text\":\"" + json_escape(acumulado) +
             "\",\"finish_reason\":\"" +
             InferenceSession::finish_reason_name(reason) +
             "\",\"usage\":{\"prefill_tokens\":" + std::to_string(st.prefill_tokens) +
             ",\"prefill_reused\":" + std::to_string(st.prefill_reused) +
             ",\"generated_tokens\":" + std::to_string(st.generated_tokens) +
             ",\"thinking_tokens\":" + std::to_string(st.thinking_tokens) +
             "},\"timings\":{\"prefill_ms\":" + std::to_string(st.prefill_ms) +
             ",\"queue_ms\":" + std::to_string(st.queue_ms) +
             ",\"first_token_ms\":" + std::to_string(st.first_token_ms) +
             ",\"decode_ms\":" + std::to_string(st.decode_ms) +
             ",\"tokens_per_second\":" +
             std::to_string(st.decode_ms > 0
                 ? st.generated_tokens * 1000.0 / st.decode_ms : 0.0) +
             "}," + state_json(st.cache_position_before, st.cache_position,
                               session_capacity) + "}");

        // AHORA sí está pausada: run_turn ha vuelto y su terminal ya salió.
        // El orden que ve quien lee la traza es el orden de lo que pasó:
        //   ... text_delta A · completed A: cancelled · session_paused A ...
        anunciar_pausa(pausar);
    }

    g_cerrar.store(true);
    if (lector.joinable()) {
        // join, no detach: en las salidas normales el lector ya terminó tras
        // `shutdown` o EOF, y dejarlo suelto es una carrera gratis durante la
        // destrucción global.
        lector.join();
    }
    std::fprintf(stderr, "[runtime] fin\n");
    return 0;
}
