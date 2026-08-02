#include "ck_policy.hpp"

#include <cassert>
#include <iostream>
#include <string>

using helios::ck::ResponseAct;

static void expect_act(const std::string& message,
                       ResponseAct expected,
                       bool trivial = false,
                       bool artifact = false,
                       bool attachment = false) {
    const auto frame = helios::ck::classify_turn(
        message, trivial, artifact, attachment);
    if (frame.act != expected) {
        std::cerr << "acto inesperado para: " << message << "\n"
                  << "esperado=" << helios::ck::response_act_name(expected)
                  << " real=" << helios::ck::response_act_name(frame.act)
                  << std::endl;
        std::abort();
    }
}

int main() {
    // El caso que destapo el fallo: documento + reescritura no es consultoria.
    expect_act("Dale una pasada para que sea tecnico pero se entienda",
               ResponseAct::Rewrite, false, true);
    expect_act("Ayudame con este capitulo", ResponseAct::Clarify,
               false, true);
    expect_act("Ayudame con este capitulo", ResponseAct::Clarify);
    expect_act("Como seguimos?", ResponseAct::Collaborate, false, true);
    expect_act("Resume este informe", ResponseAct::Summarize, false, true);
    expect_act("Que te parece este parrafo?", ResponseAct::Critique,
               false, true);

    // Continuidad: una palabra tras un artefacto no se convierte en charla.
    expect_act("velocidad", ResponseAct::ContinueArtifact, false, true);
    expect_act("sigue", ResponseAct::ContinueArtifact, false, true);

    // La correccion tiene postura propia y presupuesto acotado.
    expect_act("No, me refiero a la tabla anterior", ResponseAct::CorrectSelf);
    auto correction = helios::ck::classify_turn(
        "te equivocas, soy soldador", false, false, false);
    assert(correction.act == ResponseAct::CorrectSelf);
    assert(helios::ck::visible_token_budget(correction) == 320);
    assert(helios::ck::thinking_token_budget(correction, 100) == 140);

    expect_act("hola", ResponseAct::Greet, true);
    expect_act("que tal va el dia?", ResponseAct::CheckIn);
    expect_act("perfecto", ResponseAct::Acknowledge, true);
    expect_act("Explica como funciona una GPU", ResponseAct::Explain);
    expect_act("Cual es la capital de Portugal?", ResponseAct::Answer);
    expect_act("Que sabes de mi?", ResponseAct::Recall);
    expect_act("Como se llama mi mascota?", ResponseAct::Recall);
    expect_act("me aburro, cuentame algo", ResponseAct::Tell);
    expect_act("Implementa el cambio y añade un test", ResponseAct::Implement);

    const auto rewrite = helios::ck::classify_turn(
        "reescribe el parrafo", false, true, false);
    const std::string framed = helios::ck::frame_user_message(
        "reescribe el parrafo", rewrite, "colega");
    assert(framed.find("Acto: reescribir") != std::string::npos);
    assert(framed.find("Registro: colega") != std::string::npos);
    assert(framed.find("no ampl") != std::string::npos);
    assert(framed.find("MENSAJE REAL DE LA PERSONA:\nreescribe el parrafo") !=
           std::string::npos);
    assert(framed.find("Acto: reescribir") <
           framed.find("MENSAJE REAL DE LA PERSONA"));

    const std::string isolated = helios::ck::response_contract(
        rewrite, "colega");
    assert(isolated.find("reescribe el parrafo") == std::string::npos);

    const auto recall = helios::ck::classify_turn(
        "como se llama mi mascota?", false, false, false);
    const std::string recall_contract = helios::ck::response_contract(
        recall, "neutral", "- Tiene un loro llamado Byteazul.");
    assert(recall_contract.find("Byteazul") != std::string::npos);
    assert(recall_contract.find("fuente autoritativa") != std::string::npos);
    const std::string normalized_pet = helios::ck::prepare_recall_context(
        "como se llama mi mascota?", "- Tiene un loro llamado Byteazul.\n");
    assert(normalized_pet.find("mascota de la persona se llama Byteazul") !=
           std::string::npos);
    const std::string untouched = helios::ck::prepare_recall_context(
        "en que trabaja?", "- Tiene un loro llamado Byteazul.\n");
    assert(untouched.find("Relación normalizada") == std::string::npos);

    const auto clarify = helios::ck::classify_turn(
        "Ayudame con el capitulo", false, false, false);
    assert(helios::ck::visible_token_budget(clarify) == 96);
    assert(helios::ck::thinking_token_budget(clarify, 1000) == 80);

    // El contrato no contiene nombres ni datos de una persona concreta.
    assert(framed.find("Andr") == std::string::npos);
    assert(framed.find("dueño") == std::string::npos);

    std::cout << "CK policy: OK" << std::endl;
    return 0;
}
