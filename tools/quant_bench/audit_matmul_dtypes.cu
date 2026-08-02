// Audita que dtype recibe cada MATMUL del grafo real. Sirve para saber si un
// camino del despachador esta vivo o es codigo muerto, sin deducirlo del
// manifiesto ni de leer el grafo a ojo.
#include "src/hnf_loader.hpp"
#include "src/graph_builder.hpp"
#include "src/gemma4_kv_cache.hpp"
#include "kernels/kernels.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { printf("uso: audit_matmul_dtypes modelo.hnf\n"); return 1; }
    helios::EngineConfig cfg; cfg.scratch_pool.auto_fraction=0.0f; cfg.scratch_pool.min_size_bytes=0;
    helios::Engine engine(cfg);
    helios::kernels::register_all_kernels(engine);
    helios::HnfLoader L;
    if (!L.open(argv[1]) || !L.load_block(helios::BLOCK_TEXT_MODEL, engine)) { printf("no carga\n"); return 1; }
    const auto config = L.config();
    helios::GraphBuilder b;
    const auto arch = b.detect_architecture(engine, "text", config);

    helios::CommandBuffer cb;
    if (config.arch() == "gemma4") {
        const auto& gemma = L.gemma4_config();
        b.allocate_gemma4_scratch(engine, config, gemma, arch, 1, 1);
        helios::Gemma4KVCache kv;
        kv.allocate(gemma, config.num_key_value_heads(), 1, 512);
        kv.register_tensors(engine, "_k");
        engine.tensors().allocate_and_register("t", {1,1}, helios::dtype::INT32());
        cb = b.build_gemma4_forward_cached(engine, config, gemma, arch, "t", 1, 1, {"_k",0,512});
    } else {
        b.allocate_scratch(engine, config, arch, 1, 1);
        engine.tensors().allocate_and_register("t", {1,1}, helios::dtype::INT32());
        cb = b.build_forward_cached(engine, config, arch, "t", 1, 1, "_kv", 0, 512);
    }

    std::map<std::string,int> cuenta;
    auto& reg = helios::DTypeRegistry::instance();
    for (const auto& c : cb.commands()) {
        if (c.op != helios::op::MATMUL()) continue;
        // el segundo input de un MATMUL es la matriz de pesos
        if (c.inputs.size() < 2) continue;
        const helios::TensorInfo* w = engine.tensors().get(c.inputs[1]);
        const auto* info = w ? reg.get(w->dtype) : nullptr;
        cuenta[info ? info->name : "???"]++;
    }
    printf("  %s\n", argv[1]);
    int total = 0;
    for (const auto& [d,n] : cuenta) { printf("    MATMUL con pesos %-8s x%d\n", d.c_str(), n); total += n; }
    printf("    total %d matmuls en el grafo de una capa completa\n", total);
    return 0;
}
