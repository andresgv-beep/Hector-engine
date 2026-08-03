// Volcador de referencia del CK — E2 del plan de evacuación.
//
// Lee el corpus de comparación y emite, por cada caso, TODO lo que `ck_policy`
// decide: acto, presupuestos, si lleva contrato y los textos exactos. El CK
// portado a HexOS debe producir este mismo fichero byte por byte.
//
// Se compara la FUNCIÓN, no sus consecuencias: mensaje + estado → decisión.
// Conversar compararía resultados del modelo, que dependen del muestreo.

#include <cstdio>
#include <iostream>
#include <string>

#include "../src/ck_policy.hpp"
#include "mini_json.hpp"

using namespace helios;
namespace mj = helios::mini_json;

namespace {

std::string esc(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\t': o += "\\t";  break;
            case '\r': o += "\\r";  break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
                else o += static_cast<char>(c);
        }
    }
    return o;
}

const char* b(bool v) { return v ? "true" : "false"; }

}  // namespace

int main() {
    std::string linea;
    while (std::getline(std::cin, linea)) {
        if (linea.empty()) continue;
        mj::Value in;
        std::string err;
        if (!mj::Parser::parse(linea, &in, &err)) {
            std::fprintf(stderr, "corpus ilegible: %s\n", err.c_str());
            return 1;
        }
        const auto s = [&](const char* k) -> std::string {
            const auto* v = in.get(k); return v ? v->str() : std::string();
        };
        const auto flag = [&](const char* k) -> bool {
            const auto* v = in.get(k); return v && v->boo(false);
        };

        const std::string msg = s("message");
        const std::string registro = s("registro");
        const ck::TurnFrame frame = ck::classify_turn(
            msg, flag("trivial"), flag("artifact_active"), flag("has_attachment"));

        std::string recall;
        if (!s("recall_query").empty()) {
            recall = ck::prepare_recall_context(s("recall_query"),
                                                s("confirmed_facts"));
        }
        const std::string contrato =
            ck::response_contract(frame, registro, recall);
        const std::string framed =
            ck::frame_user_message(msg, frame, registro, recall);
        const std::string steer = ck::social_steer(frame);

        std::printf(
            "{\"id\":\"%s\",\"act\":\"%s\",\"artifact_active\":%s,"
            "\"has_attachment\":%s,\"trivial\":%s,\"visible_budget\":%d,"
            "\"thinking_budget\":%d,\"contract_needed\":%s,\"contract\":\"%s\","
            "\"social_steer\":\"%s\",\"framed_user\":\"%s\","
            "\"recall_context\":\"%s\"}\n",
            esc(s("id")).c_str(),
            ck::response_act_name(frame.act),
            b(frame.artifact_active), b(frame.has_attachment), b(frame.trivial),
            ck::visible_token_budget(frame),
            ck::thinking_token_budget(frame, msg.size()),
            b(ck::contract_needed(frame)),
            esc(contrato).c_str(), esc(steer).c_str(),
            esc(framed).c_str(), esc(recall).c_str());
    }
    return 0;
}
