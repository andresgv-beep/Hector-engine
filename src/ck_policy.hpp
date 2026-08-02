#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace helios::ck {

// El registro describe COMO se habla. El acto describe QUE debe hacer la
// respuesta. Mantenerlos separados evita que "colega" se convierta en una
// licencia para ampliar la tarea o que "tecnico" vuelva seco cualquier turno.
enum class ResponseAct {
    Acknowledge,
    Greet,
    CheckIn,
    Converse,
    Answer,
    Recall,
    Explain,
    Analyze,
    Summarize,
    Extract,
    Rewrite,
    Critique,
    Implement,
    Create,
    Tell,
    CorrectSelf,
    ContinueArtifact,
    Clarify,
    Collaborate,
};

struct TurnFrame {
    ResponseAct act = ResponseAct::Converse;
    bool artifact_active = false;
    bool has_attachment = false;
    bool trivial = false;
};

const char* response_act_name(ResponseAct act);

// Clasificador deliberadamente conservador y determinista. No pretende
// entender toda la lengua: fija el acto cuando hay una señal clara y deja el
// resto como conversación/colaboración acotada.
TurnFrame classify_turn(std::string_view message,
                        bool trivial,
                        bool artifact_active,
                        bool has_attachment);

// Contrato aislado para arquitecturas con mensajes system/developer entre
// turnos (ChatML). Nunca se guarda en memoria o transcripción.
std::string response_contract(const TurnFrame& frame,
                              std::string_view social_register,
                              std::string_view authoritative_context = {});

// Fallback para plantillas que solo permiten `system` al inicio (Gemma 4): el
// contrato va ANTES y el mensaje real queda al final como objeto a responder.
std::string frame_user_message(std::string_view message,
                               const TurnFrame& frame,
                               std::string_view social_register,
                               std::string_view authoritative_context = {});

// Presupuesto visible por acto, no por palabras sueltas. Es un techo de
// seguridad; el contrato siempre ordena parar antes cuando el acto termine.
int visible_token_budget(const TurnFrame& frame);

// El esfuerzo de razonamiento también nace del acto. Pedir un único dato no
// merece 400 tokens de pensamiento aunque el mensaje tenga muchas letras.
int thinking_token_budget(const TurnFrame& frame, size_t input_chars);

// Normaliza relaciones obvias que un modelo pequeño puede no enlazar aunque
// tenga el hecho delante (p.ej. "loro llamado Byteazul" -> mascota=Byteazul).
std::string prepare_recall_context(std::string_view query,
                                   std::string_view confirmed_facts);

}  // namespace helios::ck
