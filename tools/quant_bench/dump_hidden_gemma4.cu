// Vuelca el hidden de Hector tras CADA capa, token a token por el camino de
// decode con cache — el mismo camino que mide argmax_all_positions_gemma4.
// Salida: <outdir>/capaNN.bin con [S, hidden] en fp32, comparable contra los
// valores de oro de ref_gemma4.py (REF_DUMP_LAYERS).
#include "src/hnf_loader.hpp"
#include "src/graph_builder.hpp"
#include "src/gemma4_kv_cache.hpp"
#include "kernels/kernels.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
int main(int argc, char** argv) {
    if (argc < 4) { printf("uso: dump_hidden_gemma4 modelo.hnf toks outdir\n"); return 1; }
    std::vector<int32_t> toks; { std::string cur;
        for (const char* q = argv[2]; ; ++q) { if (*q==','||!*q){ toks.push_back(atoi(cur.c_str())); cur.clear(); if(!*q) break; } else cur+=*q; } }
    const uint32_t S = (uint32_t)toks.size();
    helios::EngineConfig cfg; cfg.scratch_pool.auto_fraction=0.0f; cfg.scratch_pool.min_size_bytes=0;
    helios::Engine engine(cfg);
    helios::kernels::register_all_kernels(engine);
    helios::HnfLoader L;
    if (!L.open(argv[1]) || !L.load_block(helios::BLOCK_TEXT_MODEL, engine)) { printf("no carga\n"); return 1; }
    const auto config = L.config(); const auto& gemma = L.gemma4_config();
    const uint32_t D = config.hidden_size(), NL = config.num_hidden_layers();
    helios::GraphBuilder b;
    const auto arch = b.detect_architecture(engine, "text", config);
    b.allocate_gemma4_scratch(engine, config, gemma, arch, 1, 1);
    helios::Gemma4KVCache kv;
    kv.allocate(gemma, config.num_key_value_heads(), 1, 4096);
    kv.register_tensors(engine, "_k");
    void* tk = engine.tensors().allocate_and_register("t", {1,1}, helios::dtype::INT32());
    // [capa][token][dim]
    std::vector<std::vector<half>> dump(NL, std::vector<half>(size_t(S)*D));
    std::vector<half> fila(D);
    for (uint32_t p = 0; p < S; p++) {
        cudaMemcpy(tk, &toks[p], 4, cudaMemcpyHostToDevice);
        engine.execute(b.build_gemma4_input(engine, config, gemma, arch, "t", 1, 1));
        helios::KVCacheParams cache{"_k", p, 4096};
        for (uint32_t l = 0; l < NL; l++) {
            engine.execute(b.build_gemma4_layer_cached(engine, config, gemma, arch, l, 1, 1, cache));
            engine.sync();
            cudaMemcpy(dump[l].data() + size_t(p)*D,
                       engine.tensors().at("_s.hidden").ptr, D*2, cudaMemcpyDeviceToHost);
        }
        if (p%8==0) fprintf(stderr,"\r  %u/%u", p, S);
    }
    fprintf(stderr,"\r  %u/%u\n", S, S);
    for (uint32_t l = 0; l < NL; l++) {
        char ruta[512]; snprintf(ruta, sizeof ruta, "%s/capa%02u.bin", argv[3], l);
        FILE* f=fopen(ruta,"wb");
        std::vector<float> f32(size_t(S)*D);
        for (size_t i=0;i<f32.size();i++) f32[i]=__half2float(dump[l][i]);
        fwrite(f32.data(),4,f32.size(),f); fclose(f);
    }
    return 0;
}
