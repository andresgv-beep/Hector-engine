// tools/qwen35_dump.cu
// ============================================================================
// Qwen3.5 capa a capa, token a token, volcando cada salida a .npy
// ============================================================================
//
// Ejecuta un HNF de la familia Qwen3.5 con el camino general del motor, pero capa por capa,
// y deja en disco lo mismo que tools/reference/qwen35_ref.py: el embedding, lo que aporta la
// atención de cada capa, la salida de cada capa, la norma final y los logits. Comparando los
// dos volcados se ve en qué capa empieza a apartarse el motor, si se aparta.
//
//     qwen35_dump <modelo.hnf> <carpeta de salida> <id> [<id> ...]
//
// No pasa por InferenceSession: reserva él mismo la caché de las capas de atención y el
// estado de las lineales.

#include "engine.hpp"
#include "graph_builder.hpp"
#include "hnf_loader.hpp"
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace helios;

namespace {

void need(cudaError_t error, const char* where) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(error));
}

// Un .npy de float32 con forma [filas, columnas].
void write_npy(const std::filesystem::path& path, const std::vector<float>& data, size_t rows, size_t cols) {
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(rows) + ", " +
                         std::to_string(cols) + "), }";
    while ((10 + header.size() + 1) % 64 != 0) header += ' ';
    header += '\n';
    std::ofstream f(path, std::ios::binary);
    f.write("\x93NUMPY\x01\x00", 8);
    const uint16_t length = static_cast<uint16_t>(header.size());
    f.write(reinterpret_cast<const char*>(&length), 2);
    f.write(header.data(), header.size());
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    if (!f) throw std::runtime_error("no pude escribir " + path.string());
}

// Las primeras `count` posiciones de un tensor FP16 del motor, como float.
void grab(Engine& engine, const std::string& name, size_t count, std::vector<float>& out) {
    TensorInfo* t = engine.tensors().get(name);
    if (!t) throw std::runtime_error("no existe el tensor " + name);
    std::vector<half> raw(count);
    need(cudaMemcpy(raw.data(), t->ptr, count * sizeof(half), cudaMemcpyDeviceToHost), "bajar tensor");
    for (half v : raw) out.push_back(__half2float(v));
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "uso: " << argv[0] << " <modelo.hnf> <carpeta de salida> <id> [<id> ...]" << std::endl;
        return 1;
    }
    try {
        const std::filesystem::path out_dir = argv[2];
        std::vector<int32_t> ids;
        for (int i = 3; i < argc; ++i) ids.push_back(std::atoi(argv[i]));
        std::filesystem::create_directories(out_dir);

        cudaStream_t stream;
        need(cudaStreamCreate(&stream), "stream");
        EngineConfig engine_config;
        engine_config.scratch_pool.pool_size_bytes = 0;
        engine_config.scratch_pool.auto_fraction = 0.0f;
        engine_config.stream = stream;
        Engine engine(engine_config);
        kernels::register_all_kernels(engine);

        HnfLoader loader;
        if (!loader.open(argv[1])) throw std::runtime_error("no abro el HNF: " + loader.last_error());
        if (!loader.load_block(BLOCK_TEXT_MODEL, engine)) throw std::runtime_error("no cargo el texto: " + loader.last_error());
        const ModelConfig config = loader.config();

        GraphBuilder gb;
        const ArchDescriptor arch = gb.detect_architecture(engine, "text", config);
        if (!arch.has_linear()) throw std::runtime_error("este HNF no trae capas de atención lineal: no es de la familia Qwen3.5");
        gb.allocate_scratch(engine, config, arch, 1, 1);

        // Caché solo para las capas de atención; estado para las lineales. Todo a cero.
        const std::string prefix = "_dump";
        // QWEN35_BENCH=N: en vez de volcar, N pasos de decode cronometrados (repite los ids).
        const char* bench_env = std::getenv("QWEN35_BENCH");
        const uint32_t bench = bench_env ? static_cast<uint32_t>(std::atoi(bench_env)) : 0;
        const uint32_t max_seq = (bench ? 2 * bench : static_cast<uint32_t>(ids.size())) + 8;
        const uint32_t KVH = config.num_key_value_heads(), HD = config.head_dim();
        const auto zeroed = [&](const std::string& name, std::vector<uint32_t> shape, DTypeID type) {
            engine.tensors().allocate_and_register(name, shape, type);
            TensorInfo* t = engine.tensors().get(name);
            need(cudaMemset(t->ptr, 0, t->size_bytes), "poner a cero");
        };
        uint32_t attention_layers = 0;
        for (uint32_t l = 0; l < arch.num_layers; ++l) {
            if (arch.is_linear(l)) {
                zeroed(GraphBuilder::linear_state_name(prefix, l),
                       {arch.lin_value_heads, arch.lin_key_dim, arch.lin_value_dim}, dtype::FP32());
                zeroed(GraphBuilder::linear_conv_name(prefix, l), {arch.lin_channels(), arch.lin_kernel - 1}, dtype::FP32());
            } else {
                const std::string base = prefix + ".layer" + std::to_string(l);
                zeroed(base + ".k", {1, max_seq, KVH, HD}, dtype::FP16());
                zeroed(base + ".v", {1, max_seq, KVH, HD}, dtype::FP16());
                ++attention_layers;
            }
        }
        engine.tensors().allocate_and_register("input_tokens", {1, 1}, dtype::INT32());
        std::cerr << arch.num_layers << " capas (" << attention_layers << " de atención, "
                  << arch.num_layers - attention_layers << " lineales), tipos " << arch.layer_types << std::endl;

        // Con torre de visión, la posición de cada token va en tres ejes y la pone la sesión
        // en _s.rope3. Aquí no hay sesión: sin esto la atención giraría todo a la posición 0.
        const auto place = [&](uint32_t position) {
            if (!arch.rope_axes3) return;
            const int32_t row[3] = {int32_t(position), int32_t(position), int32_t(position)};
            need(cudaMemcpy(engine.tensors().get("_s.rope3")->ptr, row, sizeof(row), cudaMemcpyHostToDevice), "posición");
        };

        if (bench) {
            using clock = std::chrono::steady_clock;
            const auto ms = [](clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
            uint32_t position = 0;
            const auto feed = [&](uint32_t step) {
                need(cudaMemcpy(engine.tensors().get("input_tokens")->ptr, &ids[step % ids.size()], sizeof(int32_t),
                                cudaMemcpyHostToDevice), "token");
                engine.update_device_cache_pos(position, 1);
                place(position);
            };
            // Pasada 1: el token entero, una sola sincronización al final. Se separa lo que
            // cuesta montar los comandos de lo que cuesta ejecutarlos.
            double build_ms = 0.0, run_ms = 0.0;
            for (uint32_t step = 0; step < bench; ++step, ++position) {
                feed(step);
                const KVCacheParams cache{prefix, position, max_seq};
                const auto t0 = clock::now();
                std::vector<CommandBuffer> buffers;
                buffers.push_back(gb.build_embedding_only(arch, "input_tokens"));
                for (uint32_t l = 0; l < arch.num_layers; ++l) buffers.push_back(gb.build_layer_only(config, arch, l, 1, 1, cache));
                buffers.push_back(gb.build_head_only(config, arch, 1));
                const auto t1 = clock::now();
                for (CommandBuffer& buffer : buffers) engine.execute(buffer);
                engine.sync();
                const auto t2 = clock::now();
                if (step >= 8) { build_ms += ms(t1 - t0); run_ms += ms(t2 - t1); }   // 8 de calentamiento
            }
            const double counted = double(bench - 8);
            std::cout << "decode: " << run_ms / counted << " ms/token ejecutando (" << 1000.0 * counted / run_ms
                      << " tok/s), " << build_ms / counted << " ms/token montando comandos" << std::endl;
            // Pasada 2: sincronizando tras cada capa, para repartir el tiempo por tipo.
            double linear_ms = 0.0, attention_ms = 0.0, ends_ms = 0.0;
            for (uint32_t step = 0; step < bench; ++step, ++position) {
                feed(step);
                const KVCacheParams cache{prefix, position, max_seq};
                CommandBuffer embedding = gb.build_embedding_only(arch, "input_tokens");
                auto t0 = clock::now();
                engine.execute(embedding);
                engine.sync();
                ends_ms += ms(clock::now() - t0);
                for (uint32_t l = 0; l < arch.num_layers; ++l) {
                    CommandBuffer layer = gb.build_layer_only(config, arch, l, 1, 1, cache);
                    t0 = clock::now();
                    engine.execute(layer);
                    engine.sync();
                    (arch.is_linear(l) ? linear_ms : attention_ms) += ms(clock::now() - t0);
                }
                CommandBuffer head = gb.build_head_only(config, arch, 1);
                t0 = clock::now();
                engine.execute(head);
                engine.sync();
                ends_ms += ms(clock::now() - t0);
            }
            const uint32_t linear_layers = arch.num_layers - attention_layers;
            std::cout << "por capa, con sincronización: lineal " << linear_ms / bench / linear_layers
                      << " ms, atención " << attention_ms / bench / attention_layers
                      << " ms, embedding+cabeza " << ends_ms / bench << " ms" << std::endl;
            return 0;
        }

        const size_t D = config.hidden_size(), V = config.vocab_size();
        std::map<std::string, std::vector<float>> dumps;
        char name[32];
        for (size_t t = 0; t < ids.size(); ++t) {
            const uint32_t position = static_cast<uint32_t>(t);
            need(cudaMemcpy(engine.tensors().get("input_tokens")->ptr, &ids[t], sizeof(int32_t), cudaMemcpyHostToDevice), "token");
            engine.update_device_cache_pos(position, 1);
            place(position);
            const KVCacheParams cache{prefix, position, max_seq};

            CommandBuffer embedding = gb.build_embedding_only(arch, "input_tokens");
            engine.execute(embedding);
            engine.sync();
            grab(engine, "_s.hidden", D, dumps["emb"]);
            for (uint32_t l = 0; l < arch.num_layers; ++l) {
                CommandBuffer layer = gb.build_layer_only(config, arch, l, 1, 1, cache);
                engine.execute(layer);
                engine.sync();
                std::snprintf(name, sizeof(name), "attn%02u", l);
                grab(engine, "_s.attn_proj", D, dumps[name]);
                if (arch.has_moe()) {
                    // Los expertos elegidos, ordenados por id: la referencia vuelca lo mismo.
                    std::vector<int32_t> chosen(arch.moe_top_k);
                    need(cudaMemcpy(chosen.data(), engine.tensors().get("_s.moe_indices")->ptr,
                                    chosen.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "bajar ruta");
                    std::sort(chosen.begin(), chosen.end());
                    std::snprintf(name, sizeof(name), "ruta%02u", l);
                    for (int32_t e : chosen) dumps[name].push_back(float(e));
                }
                std::snprintf(name, sizeof(name), "capa%02u", l);
                grab(engine, "_s.hidden", D, dumps[name]);
            }
            CommandBuffer head = gb.build_head_only(config, arch, 1);
            engine.execute(head);
            engine.sync();
            grab(engine, "_s.normed", D, dumps["final"]);
            grab(engine, "_s.logits", V, dumps["logits"]);
        }
        for (const auto& [key, values] : dumps) {
            write_npy(out_dir / (key + ".npy"), values, ids.size(), values.size() / ids.size());
        }
        std::cerr << "volcado en " << out_dir << " (" << dumps.size() << " tensores, " << ids.size() << " tokens)" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
