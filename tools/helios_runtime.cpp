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
#include <cstdio>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../src/inference_session.hpp"

using helios::ChatMessage;
using helios::InferenceSession;

namespace {

// --------------------------------------------------------------------------
// JSON mínimo. Un parser completo sería una dependencia nueva para un contrato
// de nueve tipos; esto cubre exactamente el protocolo y falla si no.
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

std::string json_unescape(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] != '\\') { o += s[i]; continue; }
        if (++i >= s.size()) break;
        switch (s[i]) {
            case 'n': o += '\n'; break;
            case 't': o += '\t'; break;
            case 'r': break;
            case 'u': {
                if (i + 4 < s.size()) {
                    int cp = std::stoi(s.substr(i + 1, 4), nullptr, 16);
                    if (cp < 0x80) o += static_cast<char>(cp);
                    else if (cp < 0x800) {
                        o += static_cast<char>(0xC0 | (cp >> 6));
                        o += static_cast<char>(0x80 | (cp & 0x3F));
                    } else {
                        o += static_cast<char>(0xE0 | (cp >> 12));
                        o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                        o += static_cast<char>(0x80 | (cp & 0x3F));
                    }
                    i += 4;
                }
                break;
            }
            default: o += s[i];
        }
    }
    return o;
}

// Valor de una clave de primer nivel, sin recursión: basta para el protocolo.
bool json_string(const std::string& src, const std::string& key, std::string* out) {
    std::string pat = "\"" + key + "\"";
    size_t k = src.find(pat);
    if (k == std::string::npos) return false;
    size_t c = src.find(':', k + pat.size());
    if (c == std::string::npos) return false;
    size_t q = src.find('"', c);
    if (q == std::string::npos) return false;
    std::string v;
    for (size_t i = q + 1; i < src.size(); i++) {
        if (src[i] == '\\') { v += src[i]; if (++i < src.size()) v += src[i]; continue; }
        if (src[i] == '"') break;
        v += src[i];
    }
    *out = json_unescape(v);
    return true;
}

bool json_number(const std::string& src, const std::string& key, double* out) {
    std::string pat = "\"" + key + "\"";
    size_t k = src.find(pat);
    if (k == std::string::npos) return false;
    size_t c = src.find(':', k + pat.size());
    if (c == std::string::npos) return false;
    try { *out = std::stod(src.substr(c + 1)); } catch (...) { return false; }
    return true;
}

// messages: [{"role":"…","content":"…"}, …]
std::vector<ChatMessage> parse_messages(const std::string& src) {
    std::vector<ChatMessage> out;
    size_t k = src.find("\"messages\"");
    if (k == std::string::npos) return out;
    size_t i = src.find('[', k);
    if (i == std::string::npos) return out;
    int depth = 0;
    size_t start = 0;
    for (size_t j = i; j < src.size(); j++) {
        if (src[j] == '\\') { j++; continue; }
        if (src[j] == '{') { if (depth++ == 0) start = j; }
        else if (src[j] == '}') {
            if (--depth == 0) {
                std::string obj = src.substr(start, j - start + 1);
                ChatMessage m;
                json_string(obj, "role", &m.role);
                json_string(obj, "content", &m.content);
                if (!m.role.empty()) out.push_back(m);
            }
        } else if (src[j] == ']' && depth == 0) break;
    }
    return out;
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

struct Peticion { std::string tipo, id, cruda; };

std::mutex g_cola_mtx;
std::deque<Peticion> g_cola;
std::atomic<bool> g_cerrar{false};
std::atomic<bool> g_cancelar{false};
std::string g_turno_activo;          // protegido por g_cola_mtx
std::atomic<bool> g_stdin_cerrado{false};

void hilo_lector() {
    std::string linea;
    while (std::getline(std::cin, linea)) {
        if (linea.empty()) continue;
        Peticion p;
        p.cruda = linea;
        json_string(linea, "type", &p.tipo);
        json_string(linea, "request_id", &p.id);

        // `cancel` se atiende AQUÍ, no en la cola: si esperase su turno
        // llegaría después de que termine lo que pretende cancelar.
        if (p.tipo == "cancel") {
            std::string objetivo;
            json_string(linea, "target", &objetivo);
            std::string activo;
            { std::lock_guard<std::mutex> l(g_cola_mtx); activo = g_turno_activo; }
            const bool aplica = !activo.empty() && objetivo == activo;
            if (aplica) g_cancelar.store(true);
            emit_result(p.id, aplica, 0, 0, 0);
            continue;
        }
        { std::lock_guard<std::mutex> l(g_cola_mtx); g_cola.push_back(p); }
        if (p.tipo == "shutdown") break;
    }
    g_stdin_cerrado.store(true);
}

}  // namespace

int main(int argc, char** argv) {
    std::string hnf;
    uint32_t ctx = 4096;
    float temp = 0.7f;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) hnf = argv[++i];
        else if (a == "--ctx" && i + 1 < argc) ctx = (uint32_t)atoi(argv[++i]);
        else if (a == "--temp" && i + 1 < argc) temp = (float)atof(argv[++i]);
        else {
            std::fprintf(stderr, "uso: %s --model <hnf> [--ctx N] [--temp T]\n",
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

    InferenceSession session;
    InferenceSession::Config cfg;
    cfg.hnf_path = hnf;
    cfg.max_seq_len = ctx;
    cfg.temperature = temp;
    std::string err;
    if (!session.load(cfg, &err)) {
        std::fprintf(stderr, "[runtime] no pude cargar el modelo: %s\n", err.c_str());
        return 1;
    }
    const auto& info = session.info();

    emit("{\"type\":\"ready\",\"protocol\":2,\"model\":{\"architecture\":\"" +
         json_escape(info.architecture) + "\",\"max_seq_len\":" +
         std::to_string(info.max_seq_len) + ",\"multimodal\":" +
         (info.multimodal ? "true" : "false") + ",\"vision_adapter\":" +
         (info.vision_adapter.empty()
              ? std::string("null")
              : "\"" + json_escape(info.vision_adapter) + "\"") + "}}");

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

        const uint32_t antes = session.cache_position();

        if (p.tipo == "shutdown") {
            emit_result(p.id, true, antes, antes, info.max_seq_len);
            g_cerrar.store(true);
            break;
        }
        if (p.tipo == "status") {
            emit_result(p.id, true, antes, antes, info.max_seq_len);
            continue;
        }
        if (p.tipo == "reset") {
            session.reset();
            emit_result(p.id, true, antes, session.cache_position(), info.max_seq_len);
            continue;
        }
        if (p.tipo != "turn") {
            emit_error(p.id, "unknown_request", "tipo no soportado: " + p.tipo,
                       antes, antes, info.max_seq_len);
            continue;
        }

        auto messages = parse_messages(p.cruda);
        if (messages.empty()) {
            emit_error(p.id, "empty_turn", "el turno no trae mensajes",
                       antes, antes, info.max_seq_len);
            continue;
        }

        InferenceSession::GenConfig gen;
        gen.temperature = temp;
        double d = 0;
        if (json_number(p.cruda, "temperature", &d)) gen.temperature = (float)d;
        if (json_number(p.cruda, "max_visible_tokens", &d)) gen.max_visible_tokens = (int)d;
        if (json_number(p.cruda, "max_thinking_tokens", &d)) gen.max_thinking_tokens = (int)d;

        { std::lock_guard<std::mutex> l(g_cola_mtx); g_turno_activo = p.id; }
        g_cancelar.store(false);

        std::string acumulado;
        auto t_pref = std::chrono::high_resolution_clock::now();
        bool prefill_emitido = false;
        InferenceSession::TurnStats st;
        InferenceSession::FinishReason reason;
        std::string code;

        auto on_text = [&](const std::string& chunk) {
            if (!prefill_emitido) {
                // El prefill se conoce al terminar, pero el protocolo exige
                // que su evento preceda a cualquier text_delta.
                prefill_emitido = true;
            }
            acumulado += chunk;
            emit("{\"type\":\"text_delta\",\"request_id\":\"" + json_escape(p.id) +
                 "\",\"text\":\"" + json_escape(chunk) + "\"}");
        };
        auto on_think = [&](uint32_t n) {
            emit("{\"type\":\"thinking\",\"request_id\":\"" + json_escape(p.id) +
                 "\",\"tokens\":" + std::to_string(n) + "}");
        };

        // El prefill se emite ANTES de generar: su duración real se rellena
        // luego en `timings`, pero el orden del protocolo se respeta.
        emit("{\"type\":\"prefill\",\"request_id\":\"" + json_escape(p.id) +
             "\",\"tokens\":0,\"ms\":0.0}");
        (void)t_pref;

        std::string msg;
        const bool ok = session.run_turn(messages, gen, on_text, on_think,
                                         g_cancelar, &st, &reason, &code, &msg);
        { std::lock_guard<std::mutex> l(g_cola_mtx); g_turno_activo.clear(); }

        if (!ok) {
            const std::string extra = acumulado.empty() ? "" :
                ",\"visible_text\":\"" + json_escape(acumulado) + "\"";
            emit("{\"type\":\"error\",\"request_id\":\"" + json_escape(p.id) +
                 "\",\"code\":\"" + json_escape(code) +
                 "\",\"message\":\"" + json_escape(msg) + "\"," +
                 state_json(st.cache_position_before, st.cache_position,
                            info.max_seq_len) + extra + "}");
            continue;
        }

        emit("{\"type\":\"completed\",\"request_id\":\"" + json_escape(p.id) +
             "\",\"visible_text\":\"" + json_escape(acumulado) +
             "\",\"finish_reason\":\"" +
             InferenceSession::finish_reason_name(reason) +
             "\",\"usage\":{\"prefill_tokens\":" + std::to_string(st.prefill_tokens) +
             ",\"generated_tokens\":" + std::to_string(st.generated_tokens) +
             ",\"thinking_tokens\":" + std::to_string(st.thinking_tokens) +
             "},\"timings\":{\"prefill_ms\":" + std::to_string(st.prefill_ms) +
             ",\"decode_ms\":" + std::to_string(st.decode_ms) +
             ",\"tokens_per_second\":" +
             std::to_string(st.decode_ms > 0
                 ? st.generated_tokens * 1000.0 / st.decode_ms : 0.0) +
             "}," + state_json(st.cache_position_before, st.cache_position,
                               info.max_seq_len) + "}");
    }

    g_cerrar.store(true);
    if (lector.joinable()) {
        // stdin puede seguir abierto: el proceso termina igualmente.
        lector.detach();
    }
    std::fprintf(stderr, "[runtime] fin\n");
    return 0;
}
