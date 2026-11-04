// Verosimilitud de continuaciones, para bancos de opcion multiple tipo PIQA.
//
// El modelo se carga UNA vez y se leen secuencias de la entrada estandar, una por linea:
//
//     <tokens_de_prefijo>,<tok0>,<tok1>,...
//
// El primer numero dice cuantos tokens son prompt; se puntua desde ahi. Por cada linea
// imprime "<suma de logprobs> <tokens puntuados>". La suma es log P(continuacion | prefijo)
// con log-softmax en fp32 sobre los logits de cada posicion.
//
// Mismo camino que argmax_all_positions_gemma4.cu: un token por paso, que es como se
// consiguen logits en cada posicion.
#include "src/hnf_loader.hpp"
#include "src/graph_builder.hpp"
#include "src/gemma4_kv_cache.hpp"
#include "kernels/kernels.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "uso: logprob_secuencias <modelo.hnf> < secuencias.txt\n"); return 1; }
    helios::EngineConfig cfg; cfg.scratch_pool.auto_fraction = 0.0f; cfg.scratch_pool.min_size_bytes = 0;
    helios::Engine engine(cfg);
    helios::kernels::register_all_kernels(engine);
    helios::HnfLoader L;
    if (!L.open(argv[1]) || !L.load_block(helios::BLOCK_TEXT_MODEL, engine)) { fprintf(stderr, "no carga\n"); return 1; }
    const auto config = L.config(); const auto& gemma = L.gemma4_config();
    helios::GraphBuilder b;
    const auto arch = b.detect_architecture(engine, "text", config);
    b.allocate_gemma4_scratch(engine, config, gemma, arch, 1, 1);
    helios::Gemma4KVCache kv;
    kv.allocate(gemma, config.num_key_value_heads(), 1, 4096);
    kv.register_tensors(engine, "_k");
    const uint32_t V = config.vocab_size();
    void* tk = engine.tensors().allocate_and_register("t", {1, 1}, helios::dtype::INT32());
    std::vector<half> h(V);

    std::string linea;
    char buffer[1 << 20];
    long hechas = 0;
    while (fgets(buffer, sizeof(buffer), stdin)) {
        std::vector<int32_t> nums; { std::string cur;
            for (const char* q = buffer; ; ++q) {
                if (*q == ',' || *q == '\n' || *q == '\0') { if (!cur.empty()) { nums.push_back(atoi(cur.c_str())); cur.clear(); }
                    if (*q != ',') break; } else cur += *q; } }
        if (nums.size() < 3) continue;
        const uint32_t prefijo = (uint32_t)nums[0];
        std::vector<int32_t> toks(nums.begin() + 1, nums.end());
        const uint32_t S = (uint32_t)toks.size();
        double suma = 0.0; uint32_t contados = 0;
        for (uint32_t p = 0; p + 1 < S; p++) {
            cudaMemcpy(tk, &toks[p], 4, cudaMemcpyHostToDevice);
            engine.execute(b.build_gemma4_forward_cached(engine, config, gemma, arch, "t", 1, 1, {"_k", p, 4096}));
            engine.sync();
            if (p + 1 < prefijo) continue;          // todavia dentro del prompt
            cudaMemcpy(h.data(), engine.tensors().at("_s.logits").ptr, (size_t)V * 2, cudaMemcpyDeviceToHost);
            float mx = -1e30f;
            for (uint32_t i = 0; i < V; i++) { float x = __half2float(h[i]); if (x > mx) mx = x; }
            double acc = 0.0;
            for (uint32_t i = 0; i < V; i++) acc += exp((double)(__half2float(h[i]) - mx));
            const float objetivo = __half2float(h[toks[p + 1]]);
            suma += (double)(objetivo - mx) - log(acc);
            contados++;
        }
        printf("%.6f %u\n", suma, contados);
        fflush(stdout);
        if (++hechas % 100 == 0) fprintf(stderr, "\r  %ld secuencias", hechas);
    }
    fprintf(stderr, "\r  %ld secuencias\n", hechas);
    return 0;
}
