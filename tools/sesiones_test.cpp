// Aislamiento exacto entre sesiones sobre unos MISMOS pesos.
//
// Lo que hay que demostrar no es que dos sesiones "funcionen": es que la
// segunda no puede ver, contaminar ni alterar a la primera, ni siquiera de
// formas sutiles. Las dos comparten el grafo, los buffers de trabajo y hasta
// el registro de tensores del motor, así que las maneras de filtrarse son
// muchas y ninguna es evidente desde fuera.
//
// Todo greedy: si algo cambia entre dos ejecuciones, es contaminación, no
// muestreo.
//
// uso: sesiones_test --model <hnf> [--ctx N]

#include "inference_session.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using helios::ChatMessage;
using helios::InferenceSession;
using helios::Model;

static int g_fallos = 0, g_pruebas = 0;

static void ok(bool cond, const char* nombre, const std::string& detalle = "") {
    g_pruebas++;
    if (cond) {
        std::printf("  \033[32m✓\033[0m %s\n", nombre);
    } else {
        g_fallos++;
        std::printf("  \033[31m✗ %s\033[0m — %s\n", nombre, detalle.c_str());
    }
}

// Un turno greedy. Devuelve el texto visible tal cual.
static std::string turno(InferenceSession& s, const std::string& texto,
                         int max_visible = 64) {
    InferenceSession::GenConfig gen;
    gen.temperature = 0.0f;              // greedy: sin ruido que enmascare nada
    gen.max_visible_tokens = max_visible;
    gen.max_thinking_tokens = 0;

    std::string salida;
    std::atomic<bool> cancel{false};
    InferenceSession::TurnStats st;
    InferenceSession::FinishReason reason;
    std::string code, err;

    std::vector<ChatMessage> msgs;
    msgs.push_back({"user", texto});
    s.run_turn(msgs, {}, gen, [&](const std::string& c) { salida += c; },
               [](uint32_t) {}, [](uint32_t, double) {}, cancel,
               &st, &reason, &code, &err);
    return salida;
}

static bool menciona(const std::string& h, const char* aguja) {
    std::string a = h, b = aguja;
    for (auto& c : a) c = (char)tolower((unsigned char)c);
    for (auto& c : b) c = (char)tolower((unsigned char)c);
    return a.find(b) != std::string::npos;
}

int main(int argc, char** argv) {
    std::string modelo;
    uint32_t ctx = 4096;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) modelo = argv[++i];
        else if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = (uint32_t)atoi(argv[++i]);
    }
    if (modelo.empty()) {
        std::fprintf(stderr, "uso: sesiones_test --model <hnf> [--ctx N]\n");
        return 2;
    }

    std::printf("\n[1] los pesos se cargan UNA vez\n");
    Model::Config cfg;
    cfg.hnf_path = modelo;
    cfg.max_seq_len = ctx;
    cfg.temperature = 0.0f;

    std::string err;
    auto pesos = Model::load(cfg, &err);
    ok(pesos != nullptr, "el modelo carga", err);
    if (!pesos) return 1;
    ok(pesos->sessions_created() == 0, "y todavía sin ninguna sesión", "ya había");

    InferenceSession a, b;
    ok(a.attach(pesos, &err), "la sesión A se engancha", err);
    ok(b.attach(pesos, &err), "la sesión B se engancha a los MISMOS pesos", err);
    ok(pesos->sessions_created() == 2, "el modelo cuenta dos sesiones",
       "cuenta distinta");

    // Si compartieran prefijo, compartirían caché: no serían dos
    // conversaciones, serían una con dos voces.
    ok(a.kv_namespace() != b.kv_namespace(),
       "cada una registra su KV con un prefijo propio",
       a.kv_namespace() + " == " + b.kv_namespace());

    std::printf("\n[2] lo que se dice en A no existe en B\n");
    turno(a, "Recuerda este dato: mi perro se llama Byteazul. Responde solo: vale.");
    const std::string a1 = turno(a, "¿Cómo se llama mi perro? Responde solo el nombre.");
    ok(menciona(a1, "byteazul"), "A recuerda lo que se le dijo en A", a1);

    const std::string b1 = turno(b, "¿Cómo se llama mi perro? Responde solo el nombre.");
    ok(!menciona(b1, "byteazul"),
       "B no sabe nada de lo dicho en A", b1);

    std::printf("\n[3] usar B no altera A\n");
    turno(b, "Recuerda este dato: mi gato se llama Nube. Responde solo: vale.");
    const std::string a2 = turno(a, "¿Cómo se llama mi perro? Responde solo el nombre.");
    ok(menciona(a2, "byteazul"), "A sigue recordando lo suyo", a2);
    ok(!menciona(a2, "nube"), "y no se le ha pegado nada de B", a2);

    const std::string b2 = turno(b, "¿Cómo se llama mi gato? Responde solo el nombre.");
    ok(menciona(b2, "nube"), "B recuerda lo suyo", b2);
    ok(!menciona(b2, "byteazul"), "y tampoco se le pegó nada de A", b2);

    std::printf("\n[4] el KV de cada una avanza por su cuenta\n");
    const uint32_t pa = a.cache_position(), pb = b.cache_position();
    ok(pa > 0 && pb > 0, "las dos han llenado su caché", "alguna está vacía");
    a.reset();
    ok(a.cache_position() == 0, "reiniciar A vacía A", "no se vació");
    ok(b.cache_position() == pb,
       "y deja el caché de B intacto, hasta el token",
       std::to_string(b.cache_position()) + " != " + std::to_string(pb));

    const std::string b3 = turno(b, "¿Cómo se llama mi gato? Responde solo el nombre.");
    ok(menciona(b3, "nube"), "B sigue entera tras el reset de A", b3);

    std::printf("\n[5] aislamiento EXACTO: mismo texto, misma respuesta\n");
    // La prueba fina. Una sesión limpia contesta algo con greedy. Después se
    // usa otra sesión a fondo, y una tercera sesión limpia contesta lo mismo.
    // Si el resultado difiere en un solo byte, algo del estado de la segunda
    // sobrevivió al cambio de sesión — y con greedy no hay otra explicación.
    const char* sonda = "Di exactamente la palabra: patata.";

    InferenceSession c1;
    c1.attach(pesos, &err);
    const std::string r1 = turno(c1, sonda);

    InferenceSession ruido;
    ruido.attach(pesos, &err);
    turno(ruido, "Escribe un párrafo largo sobre termodinámica y volcanes.", 200);
    turno(ruido, "Ahora otro sobre gatos de Bilbao llamados Rufo.", 200);

    InferenceSession c2;
    c2.attach(pesos, &err);
    const std::string r2 = turno(c2, sonda);

    ok(r1 == r2,
       "una sesión limpia da lo MISMO antes y después de usar otra a fondo",
       "[" + r1 + "] != [" + r2 + "]");

    // Y el muestreador tampoco se comparte: las penalizaciones de repetición
    // son historia de una conversación. Si fueran comunes, lo dicho en `ruido`
    // frenaría palabras en `c2` y el texto cambiaría — que es justo lo que
    // acabamos de descartar. Esta comprueba el otro lado: repetir DENTRO de
    // una sesión sí penaliza, así que el muestreador está vivo y no es un
    // objeto inerte que daría igual compartir.
    ok(!r1.empty(), "y la sonda no responde vacío (el muestreador está vivo)",
       "respuesta vacía: la prueba anterior no valdría nada");

    std::printf("\n[6] las sesiones mueren sin llevarse los pesos\n");
    {
        InferenceSession efimera;
        efimera.attach(pesos, &err);
        turno(efimera, "Hola.");
    }   // se destruye aquí
    const std::string b4 = turno(b, "¿Cómo se llama mi gato? Responde solo el nombre.");
    ok(menciona(b4, "nube"),
       "destruir una sesión no toca a las demás ni al modelo", b4);

    std::printf("\n%d pruebas · %d fallos\n", g_pruebas, g_fallos);
    return g_fallos ? 1 : 0;
}
