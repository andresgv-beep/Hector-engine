#pragma once

// ============================================================================
// CONGELADO — LEGACY (E2 del plan de evacuación cognitiva)
// ============================================================================
// Esta política ya vive portada en hexos-core (`hexos/src/ck/ck_policy.c`) y
// la equivalencia está demostrada byte a byte sobre 71 casos
// (`tools/ck_diff.py`). Este fichero NO se elimina todavía: `helios_chat` es
// el oráculo contra el que se mide la evacuación y la ruta legacy sigue
// siendo la predeterminada.
//
// NO SE MODIFICA. Cualquier cambio aquí invalida la referencia congelada
// (`informes/e2_ref_ck_policy.meta.json` guarda su SHA) y habría que
// regenerarla y volver a demostrar la igualdad.
//
// Se retira en E6, cuando desaparezca la ruta legacy.
// ============================================================================

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

// Los actos sociales NO llevan contrato inyectado: los gobiernan el registro
// y los ejemplos de voz del prefijo. Meterles meta-texto invita a recitarlo
// — observado en vivo: "Voy a seguir la conversación de forma natural y sin
// inventar experiencias propias" + "(Nota: esta pregunta está diseñada
// para...)" son el contrato parafraseado en la boca del modelo. Cuando el
// acto es charlar y no hay tarea concreta, el contrato es lo único que el
// modelo tiene delante... y lo comenta. Presupuestos y pensamiento siguen
// aplicando: son deterministas y no pasan por la boca del modelo.
bool contract_needed(const TurnFrame& frame);

// Los actos sociales tampoco pueden ir sin freno: quitarles el contrato mató
// la recitación pero los dejó sin condición de parada, y el modelo llenaba
// 300 tokens de ánimos, listas de opciones y valoraciones sobre la persona
// hasta que el presupuesto cortaba a media frase. Esto es una orden corta e
// IMPERATIVA, no una descripción de la respuesta: lo que se recitaba era el
// contrato meta ("Acto: conversar. Registro: ..."), no una instrucción seca.
// Vacío para los actos con contrato propio.
std::string social_steer(const TurnFrame& frame);

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
