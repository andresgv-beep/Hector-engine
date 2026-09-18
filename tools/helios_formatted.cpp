// Puente opt-in de prompts preformateados. Protocolo interno de longitud explícita.
// Reconstruye KV en cada petición. No contiene nombres ni ejecutores de herramientas.
#include "inference_session.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc!=4) return 2;
    std::cout.rdbuf(std::cerr.rdbuf());
    std::string error, code;
    helios::Model::Config cfg;
    cfg.hnf_path=argv[1]; cfg.max_seq_len=std::stoul(argv[2]);
    auto modelo = helios::Model::load(cfg,&error);
    if (!modelo) { std::cerr<<error; return 1; }
    // Dos sesiones sobre los MISMOS pesos. La auxiliar existe para que un
    // preámbulo o una extracción —prompt corto y propio, que no es la
    // conversación— no pise el KV de la conversación y la deje sin prefijo que
    // reaprovechar. Su ventana es pequeña porque esos prompts lo son.
    helios::InferenceSession session, auxiliar;
    if (!session.attach(modelo,&error)) { std::cerr<<error; return 1; }
    // Las generaciones auxiliares son avisos y extracciones breves. Darles 2K
    // duplicaba innecesariamente el anillo KV de las capas locales; 1K cubre
    // esos prompts y deja más margen para el modelo principal en tarjetas de 12 GB.
    if (!auxiliar.attach(modelo,&error,1024)) { std::cerr<<error; return 1; }
    std::printf("READY\n"); std::fflush(stdout);
    std::string header;
    while (std::getline(std::cin,header)) {
        // Cabecera: "<bytes_prompt>" o "<bytes_prompt> <bytes_pixeles> <ancho> <alto> <stride>".
        // La forma corta sigue siendo valida: un turno sin imagen no cambia.
        // Un '*' delante del tamano marca una generacion AUXILIAR: prompt propio y
        // corto, que no pertenece a la conversacion. Se ejecuta y se deshace, porque
        // si no pisa el KV y deja al turno siguiente sin prefijo que reaprovechar.
        size_t n=0, px=0; unsigned w=0,h=0,stride=0; bool es_auxiliar=false;
        {
            std::string cabeza=header;
            if(!cabeza.empty() && cabeza[0]=='*'){ es_auxiliar=true; cabeza.erase(0,1); }
            std::istringstream campos(cabeza);
            if(!(campos>>n)) return 5;
            if(campos>>px){ if(!(campos>>w>>h>>stride)) return 5; }
        }
        if(n>1024*1024) return 3;
        if(px>64u*1024u*1024u) return 6;
        std::string prompt(n,'\0'); std::cin.read(prompt.data(),n);
        if(static_cast<size_t>(std::cin.gcount())!=n) return 4;
        std::vector<unsigned char> pixeles(px);
        if(px){
            std::cin.read(reinterpret_cast<char*>(pixeles.data()),
                          static_cast<std::streamsize>(px));
            if(static_cast<size_t>(std::cin.gcount())!=px) return 4;
        }
        std::vector<helios::InferenceSession::ImageAttachment> adjuntos;
        if(px) adjuntos.push_back({pixeles.data(),pixeles.size(),w,h,stride});
        // Antes se reseteaba siempre, y con el prompt entero reenviado en cada
        // generación eso obligaba a reprocesar el mismo preámbulo una y otra vez:
        // medido, el 86% del turno era prefill y el 83% de la segunda generación
        // era texto idéntico al de la primera. Ahora la sesión compara tokens y
        // solo procesa lo nuevo. Con imagen sí se reinicia: el marcador visual se
        // expande a soft tokens y el prefijo deja de ser comparable.
        std::atomic<bool> stop{false};
        std::string output;
        helios::InferenceSession::GenConfig gen;
        // Una respuesta con código puede superar 1536 tokens. Ese techo hacía
        // que el agente descartara una salida correcta como `max_tokens` aunque
        // quedaran miles de tokens de contexto y casi 2 GiB de VRAM. El límite
        // operativo del agente corta el KV a 12K, así que 3072 aún cabe en 16K.
        gen.temperature=0; gen.max_visible_tokens=3072; gen.max_thinking_tokens=0;
        gen.preformatted=true; gen.close_turn=false; gen.stop_tokens={argv[3]};
        gen.reuse_prefix=true;
        auto& s_activa = es_auxiliar ? auxiliar : session;
        if(px) s_activa.reset();
        helios::InferenceSession::TurnStats stats;
        helios::InferenceSession::FinishReason reason;
        auto t=std::chrono::steady_clock::now();
        const bool ok=s_activa.run_turn({{"user",prompt}}, adjuntos,gen,
            [&](const std::string& s){output+=s;
                },
            {},{},stop,&stats,&reason,&code,&error);
        auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t).count();
        if(!ok) output="ERROR: "+code+" "+error;
        std::printf("%zu %ld %u %d %s %u %u\n",output.size(),ms,stats.generated_tokens,
                    stats.stopped_on_token?1:0,ok?helios::InferenceSession::finish_reason_name(reason):"error",
                    stats.prefill_tokens,stats.prefill_reused);
        std::fwrite(output.data(),1,output.size(),stdout);std::fflush(stdout);
    }
}
