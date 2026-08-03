// Fixtures HOSTILES para el parser del protocolo.
//
// El parser anterior contaba llaves ignorando las que iban dentro de cadenas y
// buscaba claves por posición en la línea. No hacía falta un cliente malicioso
// para romperlo: bastaba con pegarle código a Héctor. Estas pruebas son
// exactamente los casos que una sesión feliz nunca produce.

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

#include "../tools/mini_json.hpp"

using namespace helios::mini_json;

namespace {

int fallos = 0;

void check(bool ok, const char* nombre, const std::string& detalle = {}) {
    std::printf("  %-52s %s%s\n", nombre, ok ? "OK" : "MAL",
                (ok || detalle.empty()) ? "" : ("  <- " + detalle).c_str());
    if (!ok) fallos++;
}

Value parse_ok(const std::string& src, const char* nombre) {
    Value v;
    std::string err;
    const bool ok = Parser::parse(src, &v, &err);
    check(ok, nombre, err);
    return v;
}

void parse_debe_fallar(const std::string& src, const char* nombre) {
    Value v;
    std::string err;
    check(!Parser::parse(src, &v, &err), nombre, "se aceptó JSON inválido");
}

std::string contenido(const Value& turno, size_t idx) {
    const Value* msgs = turno.get("messages");
    if (!msgs || !msgs->is_array() || msgs->array->size() <= idx) return "<falta>";
    const Value* c = (*msgs->array)[idx].get("content");
    return c ? c->str() : "<falta>";
}

}  // namespace

int main() {
    std::printf("mini_json — fixtures hostiles\n");

    // 1) Código con llaves y comillas escapadas dentro del contenido: el caso
    //    que rompía al parser anterior.
    {
        const std::string src =
            R"({"type":"turn","request_id":"r1","messages":[{"role":"user",)"
            R"("content":"if (x) { return {\"ok\": true}; }"}]})";
        Value v = parse_ok(src, "código con llaves y comillas escapadas");
        check(contenido(v, 0) == "if (x) { return {\"ok\": true}; }",
              "  el contenido sobrevive intacto", contenido(v, 0));
    }

    // 2) Una clave del protocolo escrita DENTRO del texto del usuario no debe
    //    confundirse con la de verdad.
    {
        const std::string src =
            R"({"type":"turn","request_id":"r1",)"
            R"("messages":[{"role":"user","content":"pon \"temperature\": 9.9 en el json"}],)"
            R"("generation":{"temperature":0.35}})";
        Value v = parse_ok(src, "clave del protocolo dentro del contenido");
        const Value* g = v.get("generation");
        const Value* t = g ? g->get("temperature") : nullptr;
        check(t && t->num(-1) == 0.35, "  gana la temperature real, no la del texto",
              t ? std::to_string(t->num(-1)) : "<falta>");
    }

    // 3) Par surrogate: un emoji escapado debe salir como UTF-8 válido.
    {
        Value v = parse_ok(R"({"messages":[{"role":"user","content":"hola 😀 fin"}]})",
                           "par surrogate (emoji escapado)");
        const std::string c = contenido(v, 0);
        check(c == "hola \xF0\x9F\x98\x80 fin", "  emoji decodificado a UTF-8", c);
    }

    // 4) Acentos escapados y literales conviviendo.
    {
        Value v = parse_ok(R"({"messages":[{"role":"user","content":"ingeniería crítica ñ"}]})",
                           "acentos escapados y literales");
        check(contenido(v, 0) == "ingeniería crítica ñ", "  se decodifican bien",
              contenido(v, 0));
    }

    // 5) Varios mensajes con objetos anidados en el contenido.
    {
        const std::string src =
            R"({"messages":[{"role":"system","content":"{\"a\":1}"},)"
            R"({"role":"user","content":"y {esto} también"}]})";
        Value v = parse_ok(src, "varios mensajes con objetos en el texto");
        check(contenido(v, 0) == "{\"a\":1}" && contenido(v, 1) == "y {esto} también",
              "  ambos contenidos intactos");
        const Value* msgs = v.get("messages");
        check(msgs && msgs->array->size() == 2, "  se detectan exactamente 2 mensajes");
    }

    // 6) Escapes de barra y control.
    {
        Value v = parse_ok(R"({"messages":[{"role":"user","content":"ruta\/con\\barra\ty tab"}]})",
                           "escapes de barra, contrabarra y tab");
        check(contenido(v, 0) == "ruta/con\\barra\ty tab", "  escapes correctos",
              contenido(v, 0));
    }

    // 7) JSON malformado: debe fallar EXPLÍCITAMENTE, no devolver medio objeto.
    parse_debe_fallar(R"({"type":"turn",)", "objeto sin cerrar");
    parse_debe_fallar(R"({"content":"sin comilla final})", "cadena sin cerrar");
    parse_debe_fallar(R"({"a":1} sobra)", "contenido sobrante tras el objeto");
    parse_debe_fallar("{\"c\":\"\\u00\"}", "escape unicode truncado");
    parse_debe_fallar(R"({"c":"\uD83D solo"})", "surrogate alto sin pareja");
    parse_debe_fallar(R"({"c":"\uDE00 solo"})", "surrogate bajo huérfano");
    parse_debe_fallar(R"({"a":})", "valor ausente");
    parse_debe_fallar("", "entrada vacía");

    // 8) Ámbito: `temperature` fuera de `generation` no debe colarse.
    {
        Value v = parse_ok(R"({"temperature":0.1,"generation":{"temperature":0.9}})",
                           "misma clave en dos ámbitos");
        const Value* g = v.get("generation");
        check(g && g->get("temperature")->num(-1) == 0.9,
              "  se lee la del ámbito correcto");
    }

    std::printf("\nmini_json: %s\n", fallos == 0 ? "TODO OK" : "HAY FALLOS");
    return fallos == 0 ? 0 : 1;
}
