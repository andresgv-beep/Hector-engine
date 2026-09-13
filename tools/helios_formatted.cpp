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
    helios::InferenceSession session;
    std::string error, code;
    helios::Model::Config cfg;
    cfg.hnf_path=argv[1]; cfg.max_seq_len=std::stoul(argv[2]);
    if (!session.load(cfg,&error)) { std::cerr<<error; return 1; }
    std::printf("READY\n"); std::fflush(stdout);
    std::string header;
    while (std::getline(std::cin,header)) {
        // Cabecera: "<bytes_prompt>" o "<bytes_prompt> <bytes_pixeles> <ancho> <alto> <stride>".
        // La forma corta sigue siendo valida: un turno sin imagen no cambia.
        size_t n=0, px=0; unsigned w=0,h=0,stride=0;
        {
            std::istringstream campos(header);
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
        gen.temperature=0; gen.max_visible_tokens=1536; gen.max_thinking_tokens=0;
        gen.preformatted=true; gen.close_turn=false; gen.stop_tokens={argv[3]};
        gen.reuse_prefix=true;
        if(px) session.reset();
        helios::InferenceSession::TurnStats stats;
        helios::InferenceSession::FinishReason reason;
        auto t=std::chrono::steady_clock::now();
        const bool ok=session.run_turn({{"user",prompt}}, adjuntos,gen,
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
