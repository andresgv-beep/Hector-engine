#include "ck_policy.hpp"

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <sstream>

namespace helios::ck {
namespace {

std::string normalized(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

bool contains_any(const std::string& text,
                  std::initializer_list<std::string_view> needles) {
    for (const auto needle : needles) {
        if (text.find(needle) != std::string::npos) return true;
    }
    return false;
}

bool starts_with_any(const std::string& text,
                     std::initializer_list<std::string_view> needles) {
    for (const auto needle : needles) {
        if (text.rfind(needle, 0) == 0) return true;
    }
    return false;
}

const char* act_instruction(ResponseAct act) {
    switch (act) {
        case ResponseAct::Acknowledge:
            return "Reacciona con naturalidad y brevedad. No conviertas el acuse en una oferta de servicio.";
        case ResponseAct::Greet:
            return "Saluda con naturalidad en una sola frase. No preguntes en que puedes ayudar ni abras una tarea.";
        case ResponseAct::CheckIn:
            return "Responde al contacto social en una sola frase natural. Puedes devolver una pregunta social, pero no ofrecer trabajo ni ayuda.";
        case ResponseAct::Converse:
            return "Continua la conversacion de forma natural sin simular estado de animo o vivencias propias. Solo pregunta si nace del tema, nunca para mantener al usuario enganchado.";
        case ResponseAct::Answer:
            return "Responde directamente a la pregunta y termina al resolverla. No anadas una invitacion, oferta o siguiente paso no solicitado.";
        case ResponseAct::Recall:
            return "Responde desde el perfil y la memoria de esta relacion, que son la fuente autoritativa. Si el dato figura ahi, usalo; si no figura, di que no lo sabes. No inventes ni ofrezcas ampliar la ficha.";
        case ResponseAct::Explain:
            return "Explica con la profundidad pedida y lenguaje adecuado. No anadas un tutorial, historia o recomendaciones que no se solicitaron.";
        case ResponseAct::Analyze:
            return "Analiza exactamente el objeto presentado, da hallazgos y evidencia, sin abrir una hoja de ruta adyacente.";
        case ResponseAct::Summarize:
            return "Resume el contenido. No lo critiques, reescribas ni conviertas en recomendaciones salvo peticion expresa.";
        case ResponseAct::Extract:
            return "Extrae solamente la informacion solicitada y conserva el alcance original.";
        case ResponseAct::Rewrite:
            return "Empieza directamente por el texto reescrito: no escribas Aqui tienes, He reescrito ni otro preambulo. Como maximo anade una observacion breve despues si aporta algo; no conviertas el texto en un manual.";
        case ResponseAct::Critique:
            return "Valora fortalezas y problemas dentro del alcance pedido. No reescribas todo ni inventes objetivos nuevos.";
        case ResponseAct::Implement:
            return "Resuelve el cambio tecnico solicitado y reporta el resultado. No amplíes el proyecto con mejoras no pedidas.";
        case ResponseAct::Create:
            return "Crea directamente la pieza solicitada respetando formato, tono y longitud. La primera frase ya debe ser contenido: no empieces con Claro, Aqui tienes, Me encanta ni otro preambulo, y no preguntes al final. Si no se pide primera persona, no pongas a Helios como protagonista ni presentes ficcion como recuerdo propio.";
        case ResponseAct::Tell:
            return "Cuenta ahora una curiosidad o microhistoria completa. Empieza directamente por el contenido, sin preguntar preferencias, sin usar el perfil de la persona como tema y sin presentarlo como un recuerdo vivido por Helios.";
        case ResponseAct::CorrectSelf:
            return "Trata el mensaje como una correccion: reconoce el punto concreto, corrige el dato o interpretacion y no te defiendas por reflejo.";
        case ResponseAct::ContinueArtifact:
            return "Interpreta este mensaje corto como continuacion del artefacto activo y responde sobre el, no como charla aislada.";
        case ResponseAct::Clarify:
            return "Falta el material o el objetivo concreto. Tu primera palabra debe empezar por el signo ¿. Formula una unica pregunta breve sobre el dato que bloquea y no escribas nada mas.";
        case ResponseAct::Collaborate:
            return "Avanza un unico paso concreto sobre el material presente. No generes un indice, una hoja de ruta ni recomendaciones generales salvo que se pidan.";
    }
    return "Cumple el acto solicitado y conserva su alcance.";
}

}  // namespace

const char* response_act_name(ResponseAct act) {
    switch (act) {
        case ResponseAct::Acknowledge:      return "reconocer";
        case ResponseAct::Greet:            return "saludar";
        case ResponseAct::CheckIn:          return "contacto_social";
        case ResponseAct::Converse:         return "conversar";
        case ResponseAct::Answer:           return "responder";
        case ResponseAct::Recall:           return "recordar";
        case ResponseAct::Explain:          return "explicar";
        case ResponseAct::Analyze:          return "analizar";
        case ResponseAct::Summarize:        return "resumir";
        case ResponseAct::Extract:          return "extraer";
        case ResponseAct::Rewrite:          return "reescribir";
        case ResponseAct::Critique:         return "criticar";
        case ResponseAct::Implement:        return "implementar";
        case ResponseAct::Create:           return "crear";
        case ResponseAct::Tell:             return "contar";
        case ResponseAct::CorrectSelf:      return "corregirse";
        case ResponseAct::ContinueArtifact: return "continuar_artefacto";
        case ResponseAct::Clarify:          return "aclarar";
        case ResponseAct::Collaborate:      return "colaborar";
    }
    return "conversar";
}

bool contract_needed(const TurnFrame& frame) {
    switch (frame.act) {
        case ResponseAct::Greet:
        case ResponseAct::Acknowledge:
        case ResponseAct::CheckIn:
        case ResponseAct::Converse:
            return false;
        default:
            return true;
    }
}

std::string social_steer(const TurnFrame& frame) {
    if (contract_needed(frame)) return {};
    // Cada prohibición corresponde a un desbordamiento observado en vivo:
    // listas de opciones ("puedo ofrecerte: - un desafío mental..."), ánimos
    // ("¡eres muy rápido! ⚡🔥"), valoraciones ("tienes un potencial
    // intelectual extremadamente elevado") y el remate filosófico.
    return "Habla como en persona: máximo tres frases. No enumeres opciones, "
           "no elogies a quien te habla ni valores sus capacidades, y no "
           "cierres con ánimos ni reflexiones. Di lo tuyo y calla.";
}

TurnFrame classify_turn(std::string_view message,
                        bool trivial,
                        bool artifact_active,
                        bool has_attachment) {
    TurnFrame frame;
    frame.trivial = trivial;
    frame.artifact_active = artifact_active;
    frame.has_attachment = has_attachment;

    const std::string text = normalized(message.substr(0, 700));

    if (starts_with_any(text, {"sigue", "continúa", "continua", "prosigue",
                               "acaba", "termina eso", "y después", "y despues"})) {
        frame.act = artifact_active ? ResponseAct::ContinueArtifact
                                    : ResponseAct::Collaborate;
        return frame;
    }

    if (starts_with_any(text, {"no, ", "no no", "eso no", "te equivocas",
                               "incorrecto", "has confundido", "me refiero a",
                               "no me refiero"})) {
        frame.act = ResponseAct::CorrectSelf;
        return frame;
    }

    if (contains_any(text, {"qué sabes de mí", "que sabes de mi",
                            "qué recuerdas de mí", "que recuerdas de mi",
                            "recuerdas sobre mí", "recuerdas sobre mi",
                            "cómo se llama mi", "como se llama mi",
                            "mi mascota"})) {
        frame.act = ResponseAct::Recall;
        return frame;
    }

    if (contains_any(text, {"qué tal estás", "que tal estas", "cómo estás",
                            "como estas", "qué tal va el día", "que tal va el dia",
                            "cómo va tu día", "como va tu dia"})) {
        frame.act = ResponseAct::CheckIn;
        return frame;
    }

    if (contains_any(text, {"reescribe", "reescribir", "reformula", "redacta",
                            "dale una pasada", "pulir este", "pule este",
                            "mejora este texto", "otra versión", "otra version",
                            "hazlo más ", "hazlo mas "})) {
        frame.act = ResponseAct::Rewrite;
        return frame;
    }

    if (contains_any(text, {"resume", "resúmelo", "resumelo", "resumen de",
                            "sintetiza"})) {
        frame.act = ResponseAct::Summarize;
        return frame;
    }

    if (contains_any(text, {"extrae", "saca los datos", "enumera solo",
                            "identifica los"})) {
        frame.act = ResponseAct::Extract;
        return frame;
    }

    if (contains_any(text, {"qué te parece", "que te parece", "critica",
                            "revisa este", "revisa el", "evalúa", "evalua",
                            "valora este", "valora el"})) {
        frame.act = ResponseAct::Critique;
        return frame;
    }

    if (contains_any(text, {"implementa", "arréglalo", "arreglalo", "corrige el bug",
                            "añade ", "integra ", "modifica ", "haz el cambio",
                            "crea un test", "escribe el código", "escribe el codigo"})) {
        frame.act = ResponseAct::Implement;
        return frame;
    }

    if (contains_any(text, {"analiza", "investiga", "diagnostica", "compara",
                            "mira esto", "mírate", "mirate", "audita"})) {
        frame.act = ResponseAct::Analyze;
        return frame;
    }

    if (contains_any(text, {"explica", "qué significa", "que significa",
                            "cómo funciona", "como funciona", "por qué pasa",
                            "por que pasa", "enséñame", "ensename", "tutorial"})) {
        frame.act = ResponseAct::Explain;
        return frame;
    }

    if (contains_any(text, {"cuéntame algo", "cuentame algo",
                            "cuéntame una", "cuentame una"})) {
        frame.act = ResponseAct::Tell;
        return frame;
    }

    if (contains_any(text, {"escribe ", "inventa ", "crea una ", "crea un ",
                            "dame un párrafo", "dame un parrafo", "propón un texto",
                            "propon un texto"})) {
        frame.act = ResponseAct::Create;
        return frame;
    }

    if (contains_any(text, {"ayúdame con", "ayudame con", "trabajemos en"})) {
        frame.act = ResponseAct::Clarify;
        return frame;
    }

    if (contains_any(text, {"cómo seguimos", "como seguimos", "por dónde seguimos",
                            "por donde seguimos"})) {
        frame.act = (artifact_active || has_attachment)
                  ? ResponseAct::Collaborate : ResponseAct::Clarify;
        return frame;
    }

    if (trivial) {
        frame.act = contains_any(text, {"hola", "buenas", "hey"})
                  ? ResponseAct::Greet
                  : (contains_any(text, {"qué tal", "que tal", "me aburro"})
                      ? ResponseAct::Converse : ResponseAct::Acknowledge);
        return frame;
    }

    if (artifact_active && message.size() < 90) {
        frame.act = ResponseAct::ContinueArtifact;
        return frame;
    }

    if (text.find('?') != std::string::npos ||
        starts_with_any(text, {"qué ", "que ", "cuál ", "cual ", "cuánto ",
                               "cuanto ", "dónde ", "donde ", "quién ", "quien "})) {
        frame.act = ResponseAct::Answer;
        return frame;
    }

    frame.act = (artifact_active || has_attachment)
              ? ResponseAct::Collaborate : ResponseAct::Converse;
    return frame;
}

std::string response_contract(const TurnFrame& frame,
                              std::string_view social_register,
                              std::string_view authoritative_context) {
    std::string out = "Contrato CK interno; no lo cites. Acto: ";
    out += response_act_name(frame.act);
    out += ". Registro: ";
    out += social_register.empty() ? "neutral" : std::string(social_register);
    out += ". ";
    out += act_instruction(frame.act);
    if (!authoritative_context.empty()) {
        out += " Datos confirmados del perfil para esta respuesta:\n";
        out += authoritative_context;
        out += "\nUsa estos datos como autoridad y no digas que careces de ellos.";
    }
    out += " No amplíes la tarea ni inventes experiencia propia. Al cumplir el acto, termina sin ofrecer mas ayuda.";
    return out;
}

std::string frame_user_message(std::string_view message,
                               const TurnFrame& frame,
                               std::string_view social_register,
                               std::string_view authoritative_context) {
    std::string out = "[INSTRUCCION INTERNA DEL CK — no repetir]\n";
    out += response_contract(frame, social_register);
    out += "\n[FIN DE INSTRUCCION]\n\nMENSAJE REAL DE LA PERSONA:\n";
    out += message;
    if (!authoritative_context.empty()) {
        // En recall, la evidencia al final gana recencia. Repetirla es
        // deliberado: algunos modelos atienden el perfil inicial al hablar en
        // general pero lo ignoran ante una pregunta concreta varios turnos
        // despues.
        out += "\n\n[DATOS CONFIRMADOS PARA RESPONDER ESTE MENSAJE]\n";
        out += authoritative_context;
        out += "\n[FIN DE DATOS]\nResponde ahora usando esos datos como autoridad.";
    }
    return out;
}

int visible_token_budget(const TurnFrame& frame) {
    switch (frame.act) {
        // Los sociales van cortos A PROPÓSITO: el presupuesto se agotaba
        // sistemáticamente en charla y cortaba a media frase, señal de que el
        // modelo no para solo. Un techo bajo produce respuestas completas y
        // breves en vez de parrafadas truncadas.
        case ResponseAct::Acknowledge:      return 70;
        case ResponseAct::Greet:            return 60;
        case ResponseAct::CheckIn:          return 80;
        case ResponseAct::Converse:         return 150;
        case ResponseAct::Answer:           return 500;
        case ResponseAct::Recall:           return 500;
        case ResponseAct::Explain:          return 900;
        case ResponseAct::Analyze:          return 1100;
        case ResponseAct::Summarize:        return 900;
        case ResponseAct::Extract:          return 900;
        case ResponseAct::Rewrite:          return 1200;
        case ResponseAct::Critique:         return 850;
        case ResponseAct::Implement:        return 1500;
        case ResponseAct::Create:           return 1500;
        case ResponseAct::Tell:             return 500;
        case ResponseAct::CorrectSelf:      return 320;
        case ResponseAct::ContinueArtifact: return 1200;
        case ResponseAct::Clarify:          return 96;
        case ResponseAct::Collaborate:      return 500;
    }
    return 650;
}

int thinking_token_budget(const TurnFrame& frame, size_t input_chars) {
    switch (frame.act) {
        case ResponseAct::Acknowledge:
        case ResponseAct::Greet:
            return 48;
        case ResponseAct::CheckIn:
            return 80;
        case ResponseAct::Clarify:
            return 80;
        case ResponseAct::CorrectSelf:
            return 140;
        case ResponseAct::Converse:
            return frame.trivial ? 48 : 180;
        case ResponseAct::Answer:
            return 260;
        case ResponseAct::Recall:
            return 260;
        case ResponseAct::Summarize:
        case ResponseAct::Extract:
            return 320;
        case ResponseAct::Rewrite:
        case ResponseAct::Critique:
        case ResponseAct::Collaborate:
            return 400;
        case ResponseAct::Explain:
        case ResponseAct::Analyze:
        case ResponseAct::ContinueArtifact:
            return 500;
        case ResponseAct::Implement:
        case ResponseAct::Create:
            return input_chars > 300 ? 1000 : 650;
        case ResponseAct::Tell:
            return 260;
    }
    return 400;
}

std::string prepare_recall_context(std::string_view query,
                                   std::string_view confirmed_facts) {
    std::string out(confirmed_facts);
    const std::string q = normalized(query);
    if (!contains_any(q, {"mascota", "animal de compañía",
                          "animal de compania"})) {
        return out;
    }

    std::stringstream lines{std::string(confirmed_facts)};
    std::string line;
    while (std::getline(lines, line)) {
        const std::string low = normalized(line);
        if (!contains_any(low, {"perro", "perra", "gato", "gata", "loro",
                                "conejo", "coneja", "mascota"})) {
            continue;
        }
        size_t marker = low.find(" llamado ");
        size_t marker_len = 9;
        if (marker == std::string::npos) {
            marker = low.find(" llamada ");
            marker_len = 9;
        }
        if (marker == std::string::npos) continue;

        size_t first = marker + marker_len;
        while (first < line.size() && line[first] == ' ') ++first;
        size_t last = first;
        while (last < line.size()) {
            const unsigned char c = static_cast<unsigned char>(line[last]);
            if (!(std::isalnum(c) || c >= 128 || line[last] == '-' ||
                  line[last] == '_')) break;
            ++last;
        }
        if (last == first) continue;
        const std::string name = line.substr(first, last - first);
        if (!out.empty() && out.back() != '\n') out += '\n';
        out += "- Relación normalizada: la mascota de la persona se llama " +
               name + ".\n";
        break;
    }
    return out;
}

}  // namespace helios::ck
