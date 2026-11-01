// Pasa los residuales de la muestra de control por la cola MLP/MoE real de la GPU,
// con los pesos down del HNF y con los candidatos calibrados del conversor. La
// captura termina en la entrada a down: esto cubre down, suma, normas y salida.
//
// uso: gemma4_moe_control_replay modelo.hnf captura/ pilot-down/ salida/

#include "graph_builder.hpp"
#include "hnf_loader.hpp"
#include "kernels.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace helios;
namespace fs = std::filesystem;

namespace {

constexpr uint32_t TOKENS = 256, K = 8, I = 704, H = 2816, P = 768, E = 128;
const std::vector<uint32_t> LAYERS{0, 5, 15, 29};
const std::vector<std::string> VARIANTS{"unit", "second_moment", "routing_weighted"};
const char* const HIDDEN = "_s.hidden";

void check(cudaError_t error, const std::string& what) {
    if (error != cudaSuccess) throw std::runtime_error(what + ": " + cudaGetErrorString(error));
}

template <typename T>
std::vector<T> read_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("no abro " + path.string());
    std::vector<T> values(size_t(file.tellg()) / sizeof(T));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()), std::streamsize(values.size() * sizeof(T)));
    return values;
}

template <typename T>
void write_file(const fs::path& path, const std::vector<T>& values) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(T)));
    if (!file) throw std::runtime_error("no escribo " + path.string());
}

std::vector<float> read_fp16(Engine& engine, const char* name, size_t count) {
    engine.sync();
    std::vector<half> raw(count);
    check(cudaMemcpy(raw.data(), engine.tensors().at(name).ptr, count * sizeof(half), cudaMemcpyDeviceToHost), name);
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) values[i] = __half2float(raw[i]);
    return values;
}

int layer_of(const std::string& name) {
    const size_t mark = name.find(".layer");
    return mark == std::string::npos ? -1 : std::atoi(name.c_str() + mark + 6);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "uso: gemma4_moe_control_replay modelo.hnf captura/ pilot-down/ salida/\n";
        return 2;
    }
    const fs::path capture = argv[2], payloads = argv[3], output = argv[4];
    try {
        EngineConfig engine_config;
        check(cudaStreamCreate(&engine_config.stream), "stream");
        engine_config.use_flash_decode = true;
        Engine engine(engine_config);
        kernels::register_all_kernels(engine);
        HnfLoader loader;
        if (!loader.open(argv[1])) throw std::runtime_error(loader.last_error());
        loader.set_tensor_filter([](const TensorEntry& tensor) {
            const int layer = layer_of(tensor.name);
            return layer >= 0 && std::find(LAYERS.begin(), LAYERS.end(), uint32_t(layer)) != LAYERS.end();
        });
        if (!loader.load_block(BLOCK_TEXT_MODEL, engine)) throw std::runtime_error(loader.last_error());

        const ModelConfig& config = loader.config();
        const Gemma4Config& gemma = loader.gemma4_config();
        ArchDescriptor arch;
        arch.prefix = "text";
        arch.num_layers = config.num_hidden_layers();
        GraphBuilder builder;
        builder.allocate_gemma4_scratch(engine, config, gemma, arch, 1, 1);

        cudaDeviceProp properties{};
        int device = 0;
        check(cudaGetDevice(&device), "GPU");
        check(cudaGetDeviceProperties(&properties, device), "GPU");
        std::cout << properties.name << std::endl;

        const std::map<std::pair<std::string, int>, std::string> steps{
            {{"_s.mlp_out", 2}, "shared_branch_post"},
            {{"_s.g4.moe_out", 1}, "expert_sum"},
            {{"_s.g4.moe_out", 2}, "expert_branch_post"},
            {{"_s.mlp_out", 3}, "combined"},
            {{HIDDEN, 2}, "output"},
        };

        for (uint32_t layer : LAYERS) {
            const fs::path control = capture / ("control-layer-" + std::to_string(layer));
            const auto residual = read_file<float>(control / "residual.f32");
            const auto want_ids = read_file<int32_t>(control / "expert_ids.i32");
            const auto want_weights = read_file<float>(control / "expert_weights.f32");
            const auto want_input = read_file<float>(control / "down_input.f32");
            std::vector<half> residual_half(residual.size());
            for (size_t i = 0; i < residual.size(); ++i) residual_half[i] = __float2half(residual[i]);

            char name[32];
            std::snprintf(name, sizeof(name), "layer-%02u", layer);
            const fs::path layer_dir = output / name;
            fs::create_directories(layer_dir);

            const std::string down_name = "text.layer" + std::to_string(layer) + ".moe.experts.down.weight";
            auto* down = static_cast<uint8_t*>(engine.tensors().at(down_name).ptr);
            constexpr size_t STRIDE = size_t(H) * (P / 256) * 152;
            std::vector<uint8_t> original(STRIDE * E);
            check(cudaMemcpy(original.data(), down, original.size(), cudaMemcpyDeviceToHost), "leer down");

            const auto tail = builder.build_gemma4_mlp_ple_tail(engine, config, gemma, arch, layer, 1, 1);
            for (const std::string& variant : VARIANTS) {
                check(cudaMemcpy(down, original.data(), original.size(), cudaMemcpyHostToDevice), "restaurar down");
                int replaced = 0;
                if (variant != "unit") {
                    for (uint32_t expert = 0; expert < E; ++expert) {
                        char payload[80];
                        std::snprintf(payload, sizeof(payload), "%s-expert-%03u-%s.hq44", name, expert, variant.c_str());
                        if (!fs::exists(payloads / payload)) continue;
                        const auto matrix = read_file<uint8_t>(payloads / payload);
                        if (matrix.size() != STRIDE) throw std::runtime_error("payload de tamaño incorrecto");
                        check(cudaMemcpy(down + expert * STRIDE, matrix.data(), STRIDE, cudaMemcpyHostToDevice), "subir candidato");
                        ++replaced;
                    }
                }

                std::map<std::string, std::vector<float>> saved;
                std::vector<float> rows;
                size_t wrong_ids = 0, wrong_weights = 0, wrong_input = 0;
                for (uint32_t token = 0; token < TOKENS; ++token) {
                    check(cudaMemcpy(engine.tensors().at(HIDDEN).ptr, residual_half.data() + size_t(token) * H,
                                     H * sizeof(half), cudaMemcpyHostToDevice), "residual");
                    std::map<std::string, int> seen;
                    for (const Command& command : tail.commands()) {
                        engine.execute_command(command);
                        const auto step = steps.find({command.output, ++seen[command.output]});
                        if (step != steps.end()) {
                            const auto values = read_fp16(engine, command.output.c_str(), H);
                            auto& all = saved[step->second];
                            all.insert(all.end(), values.begin(), values.end());
                        }
                        if (command.op != op::MOE_EXPERTS_DECODE()) continue;
                        std::vector<int32_t> ids(K);
                        std::vector<float> weights(K);
                        engine.sync();
                        check(cudaMemcpy(ids.data(), engine.tensors().at("_s.g4.moe_indices").ptr, K * 4, cudaMemcpyDeviceToHost), "ids");
                        check(cudaMemcpy(weights.data(), engine.tensors().at("_s.g4.moe_weights").ptr, K * 4, cudaMemcpyDeviceToHost), "pesos");
                        for (uint32_t k = 0; k < K; ++k) {
                            wrong_ids += ids[k] != want_ids[size_t(token) * K + k];
                            wrong_weights += weights[k] != want_weights[size_t(token) * K + k];
                        }
                        const auto hidden = read_fp16(engine, "_s.g4.moe_hidden", size_t(K) * P);
                        for (uint32_t k = 0; k < K; ++k) {
                            for (uint32_t j = 0; j < I; ++j) {
                                wrong_input += hidden[size_t(k) * P + j] !=
                                               want_input[(size_t(token) * K + k) * I + j];
                            }
                        }
                        const auto row = read_fp16(engine, "_s.g4.moe_rows", size_t(K) * H);
                        rows.insert(rows.end(), row.begin(), row.end());
                    }
                }
                write_file(layer_dir / (variant + "_expert_rows.f32"), rows);
                for (const auto& [step, values] : saved) write_file(layer_dir / (variant + "_" + step + ".f32"), values);

                engine.invalidate_graph();
                std::vector<double> runs;
                for (uint32_t token = 0; token < TOKENS; ++token) {
                    check(cudaMemcpy(engine.tensors().at(HIDDEN).ptr, residual_half.data() + size_t(token) * H,
                                     H * sizeof(half), cudaMemcpyHostToDevice), "residual");
                    const auto start = std::chrono::steady_clock::now();
                    engine.execute_graph_replay(tail);
                    engine.sync();
                    if (token >= 8) {
                        runs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
                    }
                }
                const bool graph = engine.graph_ready();
                engine.invalidate_graph();
                std::sort(runs.begin(), runs.end());
                std::cout << "capa " << layer << ' ' << variant << ": candidatos " << replaced << ", ids distintos "
                          << wrong_ids << ", pesos distintos " << wrong_weights << ", entrada a down distinta "
                          << wrong_input << ", cola " << runs[runs.size() / 2] << " ms, grafo " << graph << std::endl;
            }
            check(cudaMemcpy(down, original.data(), original.size(), cudaMemcpyHostToDevice), "restaurar down");
        }
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
