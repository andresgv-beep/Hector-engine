// Prefill POR LOTES de S tokens en una sola pasada, y vuelca los logits de la
// ultima posicion. Complementa a argmax_all_positions_gemma4, que va token a
// token (M=1) y por tanto NUNCA ejerce el camino dequant+cuBLAS del prefill.
// Sirve para comparar builds que solo difieran en ese camino.
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
    if (argc < 4) { printf("uso: prefill_logits_gemma4 modelo.hnf toks salida.bin\n"); return 1; }
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
    b.allocate_gemma4_scratch(engine, config, gemma, arch, 1, S);
    helios::Gemma4KVCache kv;
    kv.allocate(gemma, config.num_key_value_heads(), 1, 4096);
    kv.register_tensors(engine, "_k");
    const uint32_t V = config.vocab_size();
    void* tk = engine.tensors().allocate_and_register("t", {1, S}, helios::dtype::INT32());
    cudaMemcpy(tk, toks.data(), (size_t)S*4, cudaMemcpyHostToDevice);

    // una pasada de calentamiento y luego la medida, que el primer run paga
    // el autotune y no vale para cronometrar
    auto cb = b.build_gemma4_forward_cached(engine, config, gemma, arch, "t", 1, S, {"_k",0,4096});
    engine.execute(cb); engine.sync();

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float mejor = 1e30f;
    // se reusa el mismo cache: al prefillear siempre desde la posicion 0 se
    // sobrescribe entero, asi que las repeticiones son equivalentes
    for (int rep = 0; rep < 5; rep++) {
        cudaEventRecord(e0);
        engine.execute(b.build_gemma4_forward_cached(engine, config, gemma, arch, "t", 1, S, {"_k",0,4096}));
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0.0f; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < mejor) mejor = ms;
    }
    fprintf(stderr, "  prefill de %u tokens: %.1f ms (mejor de 5)\n", S, mejor);

    std::vector<half> h(V);
    cudaMemcpy(h.data(), engine.tensors().at("_s.logits").ptr, (size_t)V*2, cudaMemcpyDeviceToHost);
    std::vector<float> f32(V);
    uint32_t best=0; float bv=-1e30f;
    for (uint32_t i=0;i<V;i++){ f32[i]=__half2float(h[i]); if(f32[i]>bv){bv=f32[i];best=i;} }
    fprintf(stderr, "  argmax=%u  max=%.4f\n", best, bv);
    FILE* f=fopen(argv[3],"wb"); fwrite(f32.data(),4,V,f); fclose(f);

    // GEN=<n>: seguir generando greedy DESDE el KV que construyo el prefill.
    // Es lo que decide de verdad: comparar solo los logits del prefill no ve
    // si el error se arrastra por el cache a lo largo de la generacion.
    const char* gen_env = getenv("GEN");
    if (gen_env) {
        int n_gen = atoi(gen_env);
        void* tk1 = engine.tensors().allocate_and_register("t1", {1,1}, helios::dtype::INT32());
        b.allocate_gemma4_scratch(engine, config, gemma, arch, 1, 1);
        std::vector<int32_t> salida;
        int32_t tok = (int32_t)best;
        for (int i = 0; i < n_gen; i++) {
            salida.push_back(tok);
            cudaMemcpy(tk1, &tok, 4, cudaMemcpyHostToDevice);
            engine.execute(b.build_gemma4_forward_cached(
                engine, config, gemma, arch, "t1", 1, 1, {"_k", S + (uint32_t)i, 4096}));
            engine.sync();
            cudaMemcpy(h.data(), engine.tensors().at("_s.logits").ptr, (size_t)V*2, cudaMemcpyDeviceToHost);
            uint32_t bi=0; float bvv=-1e30f;
            for (uint32_t j=0;j<V;j++){ float x=__half2float(h[j]); if(x>bvv){bvv=x;bi=j;} }
            tok = (int32_t)bi;
        }
        std::string ruta = std::string(argv[3]) + ".gen";
        FILE* g=fopen(ruta.c_str(),"wb"); fwrite(salida.data(),4,salida.size(),g); fclose(g);
        fprintf(stderr, "  %d tokens generados -> %s\n", n_gen, ruta.c_str());
    }
    return 0;
}
