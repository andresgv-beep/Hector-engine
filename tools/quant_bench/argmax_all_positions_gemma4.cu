// argmax en CADA posicion, generando de a un token (asi hay logits por paso).
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
    std::vector<int32_t> toks; { std::string cur;
        for (const char* q = argv[2]; ; ++q) { if (*q==','||!*q){ toks.push_back(atoi(cur.c_str())); cur.clear(); if(!*q) break; } else cur+=*q; } }
    const uint32_t S = (uint32_t)toks.size();
    helios::EngineConfig cfg; cfg.scratch_pool.auto_fraction=0.0f; cfg.scratch_pool.min_size_bytes=0;
    helios::Engine engine(cfg);
    helios::kernels::register_all_kernels(engine);
    helios::HnfLoader L;
    if (!L.open(argv[1]) || !L.load_block(helios::BLOCK_TEXT_MODEL, engine)) { printf("no carga\n"); return 1; }
    const auto config = L.config(); const auto& gemma = L.gemma4_config();
    helios::GraphBuilder b;
    const auto arch = b.detect_architecture(engine, "text", config);
    b.allocate_gemma4_scratch(engine, config, gemma, arch, 1, 1);
    helios::Gemma4KVCache kv;
    kv.allocate(gemma, config.num_key_value_heads(), 1, 4096);
    kv.register_tensors(engine, "_k");
    const uint32_t V = config.vocab_size();
    void* tk = engine.tensors().allocate_and_register("t", {1,1}, helios::dtype::INT32());
    std::vector<int32_t> am(S);
    std::vector<half> h(V);
    for (uint32_t p = 0; p < S; p++) {
        cudaMemcpy(tk, &toks[p], 4, cudaMemcpyHostToDevice);
        engine.execute(b.build_gemma4_forward_cached(engine, config, gemma, arch, "t", 1, 1, {"_k",p,4096}));
        engine.sync();
        cudaMemcpy(h.data(), engine.tensors().at("_s.logits").ptr, V*2, cudaMemcpyDeviceToHost);
        uint32_t best=0; float bv=-1e30f;
        for (uint32_t i=0;i<V;i++){ float x=__half2float(h[i]); if(x>bv){bv=x;best=i;} }
        am[p]=(int32_t)best;
        if (p%200==0) fprintf(stderr,"\r  %u/%u", p, S);
    }
    fprintf(stderr,"\r  %u/%u\n", S, S);
    FILE* f=fopen(argv[3],"wb"); fwrite(am.data(),4,S,f); fclose(f);
    return 0;
}
