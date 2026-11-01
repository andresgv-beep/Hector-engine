// Cola MLP + MoE de una capa real contra el oraculo FP32 de
// converter/tools/gemma4_moe_layer_oracle.py. Pesos originales BF16 pasados a FP16.
//
// uso: test_gemma4_moe_layer references/ [salidas/]
//
// Con un segundo argumento vuelca los 23 tensores que pide
// converter/tools/compare_moe_oracle.py en un directorio nuevo.
#include "graph_builder.hpp"
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

using helios::dtype::FP16;

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void cuda_require(cudaError_t error, const std::string& where) {
    require(error == cudaSuccess, where + ": " + cudaGetErrorString(error));
}

template <typename T>
std::vector<T> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(bool(file), "no abro " + path);
    std::vector<T> values(size_t(file.tellg()) / sizeof(T));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(T));
    require(bool(file), "lectura incompleta de " + path);
    return values;
}

// El lector minimo que hace falta: la cabecera la escribe gemma4-moe-reference.
class LayerFile {
public:
    explicit LayerFile(const std::string& path) : file_(path, std::ios::binary) {
        require(bool(file_), "no abro " + path);
        uint64_t size = 0;
        file_.read(reinterpret_cast<char*>(&size), sizeof(size));
        header_.resize(size);
        file_.read(header_.data(), size);
        base_ = sizeof(size) + size;
    }

    // Promueve BF16 a FP32 sin perdida y baja a FP16, como cargaria un HNF sin cuantizar.
    std::vector<half> read_fp16(const std::string& name) {
        const size_t key = header_.find("\"" + name + "\"");
        require(key != std::string::npos, "falta " + name);
        const size_t dtype = header_.find("\"dtype\":\"", key);
        require(header_.compare(dtype + 9, 4, "BF16") == 0, name + " no es BF16");
        const size_t offsets = header_.find("\"data_offsets\":[", key);
        const uint64_t begin = std::stoull(header_.substr(offsets + 16));
        const uint64_t end = std::stoull(header_.substr(header_.find(',', offsets) + 1));
        std::vector<uint16_t> raw((end - begin) / 2);
        file_.seekg(base_ + begin);
        file_.read(reinterpret_cast<char*>(raw.data()), end - begin);
        require(bool(file_), "lectura incompleta de " + name);
        std::vector<half> result(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) {
            const uint32_t bits = uint32_t(raw[i]) << 16;
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            result[i] = __float2half(value);
        }
        return result;
    }

private:
    std::ifstream file_;
    std::string header_;
    uint64_t base_ = 0;
};

std::vector<half> to_half(const std::vector<float>& values) {
    std::vector<half> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) result[i] = __float2half(values[i]);
    return result;
}

std::vector<float> scratch_floats(helios::Engine& engine, const std::string& name, size_t count) {
    engine.sync();
    std::vector<half> host(count);
    cuda_require(cudaMemcpy(host.data(), engine.tensors().at(name).ptr, count * sizeof(half),
                            cudaMemcpyDeviceToHost), "leer " + name);
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) result[i] = __half2float(host[i]);
    return result;
}

double relative_error(const std::vector<float>& got, const std::vector<float>& want) {
    require(got.size() == want.size(), "tamaños distintos");
    double difference = 0, reference = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        difference += (double(got[i]) - want[i]) * (double(got[i]) - want[i]);
        reference += double(want[i]) * want[i];
    }
    return std::sqrt(difference / reference);
}

void check_error(const std::string& name, const std::vector<float>& got,
                 const std::vector<float>& want, double limit) {
    const double error = relative_error(got, want);
    std::cout << "  " << name << ": error relativo " << error << std::endl;
    require(error < limit, name);
}

constexpr uint32_t E = 128, K = 8, H = 2816, I = 704, TOKENS = 4;
const std::string HIDDEN = "_s.hidden";

void set_residual(helios::Engine& engine, const std::vector<half>& rows) {
    cuda_require(cudaMemcpy(engine.tensors().at(HIDDEN).ptr, rows.data(),
                            rows.size() * sizeof(half), cudaMemcpyHostToDevice), "residual");
}

template <typename T>
void write_file(const std::string& path, const std::vector<T>& values) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    require(bool(file), "no escribo " + path);
}

template <typename T>
std::vector<T> download(const void* device, size_t count) {
    std::vector<T> host(count);
    cuda_require(cudaDeviceSynchronize(), "sincronizar");
    cuda_require(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost), "bajar");
    return host;
}

// Ejecuta la cola comando a comando y guarda cada salida intermedia, así lo
// volcado es lo que produce el grafo real y no una segunda implementación.
void dump_outputs(helios::Engine& engine, const helios::CommandBuffer& tail,
                  const std::vector<half>& residual, const std::string& oracle, const std::string& out) {
    using namespace helios::kernels;
    require(std::filesystem::create_directory(out), out + " ya existe");
    const size_t tokens = TOKENS;
    const std::map<std::pair<std::string, int>, std::string> steps{
        {{"_s.normed", 1}, "shared_input"},
        {{"_s.mlp_out", 2}, "shared_branch_post"},
        {{"_s.g4.moe_router", 2}, "router_input"},
        {{"_s.g4.moe_logits", 1}, "router_logits"},
        {{"_s.normed", 2}, "expert_input"},
        {{"_s.g4.moe_out", 1}, "expert_sum"},
        {{"_s.g4.moe_out", 2}, "expert_branch_post"},
        {{"_s.mlp_out", 3}, "combined_before_common_norm"},
        {{"_s.mlp_out", 4}, "common_norm"},
        {{HIDDEN, 2}, "output"},
    };
    set_residual(engine, residual);
    std::map<std::string, int> seen;
    int saved = 0;
    for (const auto& command : tail.commands()) {
        engine.execute_command(command);
        const auto step = steps.find({command.output, ++seen[command.output]});
        if (step == steps.end()) continue;
        const uint32_t width = command.output == "_s.g4.moe_logits" ? E : H;
        write_file(out + "/" + step->second + ".f32",
                   scratch_floats(engine, command.output, tokens * width));
        ++saved;
    }
    require(saved == int(steps.size()), "la cola no tiene los pasos esperados");

    const void* logits = engine.tensors().at("_s.g4.moe_logits").ptr;
    const void* indices = engine.tensors().at("_s.g4.moe_indices").ptr;
    const auto ids = download<int32_t>(indices, tokens * K);
    const auto weights = download<float>(engine.tensors().at("_s.g4.moe_weights").ptr, tokens * K);
    write_file(out + "/expert_ids.i32", ids);
    write_file(out + "/expert_weights.f32", weights);

    float* probabilities = nullptr;
    float* normalized = nullptr;
    int32_t* route_ids = nullptr;
    half* unit_scale = nullptr;
    cuda_require(cudaMalloc(&probabilities, tokens * E * sizeof(float)), "probabilidades");
    cuda_require(cudaMalloc(&normalized, tokens * K * sizeof(float)), "normalizados");
    cuda_require(cudaMalloc(&route_ids, tokens * K * sizeof(int32_t)), "ids");
    cuda_require(cudaMalloc(&unit_scale, E * sizeof(half)), "escala");
    const std::vector<half> ones(E, __float2half(1.0f));
    cuda_require(cudaMemcpy(unit_scale, ones.data(), E * sizeof(half), cudaMemcpyHostToDevice), "escala");
    launch_moe_probabilities(static_cast<const half*>(logits), probabilities, TOKENS, E, 0);
    write_file(out + "/router_probabilities.f32", download<float>(probabilities, tokens * E));
    // Con escala 1 moe_route deja justo los pesos renormalizados.
    launch_moe_route(static_cast<const half*>(logits), unit_scale, route_ids, normalized, TOKENS, E, K, 0);
    write_file(out + "/normalized_weights.f32", download<float>(normalized, tokens * K));

    const half* input = static_cast<const half*>(engine.tensors().at("_s.normed").ptr);
    const half* gate_up = static_cast<const half*>(engine.tensors().at("text.layer0.moe.experts.gate_up.weight").ptr);
    const half* down = static_cast<const half*>(engine.tensors().at("text.layer0.moe.experts.down.weight").ptr);
    half* projected = nullptr;
    half* activated = nullptr;
    half* row = nullptr;
    cuda_require(cudaMalloc(&projected, 2 * I * sizeof(half)), "gate_up");
    cuda_require(cudaMalloc(&activated, I * sizeof(half)), "activado");
    cuda_require(cudaMalloc(&row, H * sizeof(half)), "fila");
    std::vector<float> gate, up, gelu_mul, contributions;
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t slot = 0; slot < K; ++slot) {
            const size_t expert = size_t(ids[token * K + slot]);
            launch_matmul_fp16_cublas(input + token * H, gate_up + expert * 2 * I * H, projected, 1, H, 2 * I, 0);
            launch_moe_gate_up_mul(projected, activated, 1, I, 0);
            launch_matmul_fp16_cublas(activated, down + expert * H * I, row, 1, I, H, 0);
            const auto gate_up_row = download<half>(projected, 2 * I);
            for (size_t i = 0; i < I; ++i) gate.push_back(__half2float(gate_up_row[i]));
            for (size_t i = 0; i < I; ++i) up.push_back(__half2float(gate_up_row[I + i]));
            for (half value : download<half>(activated, I)) gelu_mul.push_back(__half2float(value));
            const float weight = weights[token * K + slot];
            for (half value : download<half>(row, H)) contributions.push_back(__half2float(value) * weight);
        }
    }
    write_file(out + "/gate.f32", gate);
    write_file(out + "/up.f32", up);
    write_file(out + "/gelu_mul.f32", gelu_mul);
    write_file(out + "/expert_contributions.f32", contributions);

    // El residual y las entradas de empate son lo que recibió el motor, sin tocar.
    for (const char* name : {"residual.f32", "tie_logits.f32", "tie_scales.f32"}) {
        write_file(out + "/" + name, read_file<char>(oracle + "/" + name));
    }
    for (void* p : {static_cast<void*>(probabilities), static_cast<void*>(normalized), static_cast<void*>(route_ids),
                    static_cast<void*>(unit_scale), static_cast<void*>(projected), static_cast<void*>(activated),
                    static_cast<void*>(row)}) {
        cudaFree(p);
    }
}

void test_ties(helios::Engine& engine, const std::string& oracle, const std::string& out) {
    const auto logits = to_half(read_file<float>(oracle + "/tie_logits.f32"));
    const auto scale = to_half(read_file<float>(oracle + "/tie_scales.f32"));
    const auto want_ids = read_file<int32_t>(oracle + "/tie_ids.i32");
    const auto want_weights = read_file<float>(oracle + "/tie_weights.f32");
    auto& tensors = engine.tensors();
    cudaMemcpy(tensors.allocate_and_register("test.tie.logits", {1, E}, FP16()), logits.data(),
               logits.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(tensors.allocate_and_register("test.tie.scale", {E}, FP16()), scale.data(),
               scale.size() * sizeof(half), cudaMemcpyHostToDevice);
    void* ids = tensors.allocate_and_register("test.tie.ids", {1, K}, helios::dtype::INT32());
    void* weights = tensors.allocate_and_register("test.tie.weights", {1, K}, helios::dtype::FP32());

    helios::CommandBuffer commands;
    commands.add_moe_route("test.tie.ids", "test.tie.weights", "test.tie.logits", "test.tie.scale", K);
    engine.execute(commands);
    engine.sync();
    std::vector<int32_t> got_ids(K);
    std::vector<float> got_weights(K);
    cuda_require(cudaMemcpy(got_ids.data(), ids, K * sizeof(int32_t), cudaMemcpyDeviceToHost), "ids");
    cuda_require(cudaMemcpy(got_weights.data(), weights, K * sizeof(float), cudaMemcpyDeviceToHost), "pesos");
    require(got_ids == want_ids, "empate que cruza el puesto k");
    check_error("empates", got_weights, want_weights, 1e-6);
    if (!out.empty()) {
        std::filesystem::create_directories(out + ".ties");
        write_file(out + ".ties/tie_ids.i32", got_ids);
        write_file(out + ".ties/tie_weights.f32", got_weights);
    }
}

void test_layer(const std::string& references, const std::string& out) {
    const std::string oracle = references + "/layer-00-oracle-fp32";
    helios::Engine engine;
    helios::kernels::register_all_kernels(engine);
    test_ties(engine, oracle, out);

    helios::ModelConfig config;
    config.set("arch", std::string("gemma4"));
    config.set("num_hidden_layers", int64_t(1));
    config.set("hidden_size", int64_t(H));
    config.set("intermediate_size", int64_t(2112));
    config.set("vocab_size", int64_t(16));
    config.set("num_attention_heads", int64_t(16));
    config.set("num_key_value_heads", int64_t(8));
    config.set("rms_norm_eps", 1.0e-6);
    config.set("hidden_act", std::string("gelu_pytorch_tanh"));

    helios::Gemma4Config gemma;
    gemma.flags = helios::GEMMA4_EXT_FLAG_MOE | helios::GEMMA4_EXT_FLAG_LAYER_SCALAR;
    gemma.moe_experts = E;
    gemma.moe_top_k = K;
    gemma.moe_intermediate_size = I;
    helios::Gemma4LayerConfig layer{};
    layer.head_dim = 256;
    layer.intermediate_size = 2112;
    gemma.layers.push_back(layer);

    helios::ArchDescriptor arch;
    arch.prefix = "text";
    arch.num_layers = 1;

    struct Weight { const char* name; const char* source; std::vector<uint32_t> shape; };
    const std::vector<Weight> weights{
        {"ln_mlp_in.weight", "pre_feedforward_layernorm.weight", {H}},
        {"ln_mlp_branch_post.weight", "post_feedforward_layernorm_1.weight", {H}},
        {"ln_mlp_post.weight", "post_feedforward_layernorm.weight", {H}},
        {"ln_moe_in.weight", "pre_feedforward_layernorm_2.weight", {H}},
        {"ln_moe_branch_post.weight", "post_feedforward_layernorm_2.weight", {H}},
        {"layer_scalar", "layer_scalar", {1}},
        {"mlp.gate.weight", "mlp.gate_proj.weight", {2112, H}},
        {"mlp.up.weight", "mlp.up_proj.weight", {2112, H}},
        {"mlp.down.weight", "mlp.down_proj.weight", {H, 2112}},
        {"moe.router.input_scale", "router.scale", {H}},
        {"moe.router.weight", "router.proj.weight", {E, H}},
        {"moe.router.expert_scale", "router.per_expert_scale", {E}},
        {"moe.experts.gate_up.weight", "experts.gate_up_proj", {E, 2 * I, H}},
        {"moe.experts.down.weight", "experts.down_proj", {E, H, I}},
    };
    LayerFile source(references + "/layer-00-original/layer.safetensors");
    for (const Weight& weight : weights) {
        const auto values = source.read_fp16(std::string("model.language_model.layers.0.") + weight.source);
        void* device = engine.tensors().allocate_and_register(
            std::string("text.layer0.") + weight.name, weight.shape, FP16());
        cuda_require(cudaMemcpy(device, values.data(), values.size() * sizeof(half),
                                cudaMemcpyHostToDevice), weight.name);
    }

    helios::GraphBuilder builder;
    builder.allocate_gemma4_scratch(engine, config, gemma, arch, 1, TOKENS);
    const auto residual = to_half(read_file<float>(oracle + "/residual.f32"));
    const auto want_output = read_file<float>(oracle + "/output.f32");

    set_residual(engine, residual);
    engine.execute(builder.build_gemma4_mlp_ple_tail(engine, config, gemma, arch, 0, 1, TOKENS));

    check_error("logits", scratch_floats(engine, "_s.g4.moe_logits", size_t(TOKENS) * E),
                read_file<float>(oracle + "/router_logits.f32"), 2e-3);
    const auto want_ids = read_file<int32_t>(oracle + "/expert_ids.i32");
    std::vector<int32_t> got_ids(want_ids.size());
    std::vector<float> got_weights(want_ids.size());
    cuda_require(cudaMemcpy(got_ids.data(), engine.tensors().at("_s.g4.moe_indices").ptr,
                            got_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "indices");
    cuda_require(cudaMemcpy(got_weights.data(), engine.tensors().at("_s.g4.moe_weights").ptr,
                            got_weights.size() * sizeof(float), cudaMemcpyDeviceToHost), "pesos");
    require(got_ids == want_ids, "expertos elegidos");
    std::cout << "  expertos elegidos: identicos" << std::endl;
    check_error("pesos", got_weights, read_file<float>(oracle + "/expert_weights.f32"), 2e-3);
    check_error("rama de expertos", scratch_floats(engine, "_s.g4.moe_out", size_t(TOKENS) * H),
                read_file<float>(oracle + "/expert_branch_post.f32"), 2e-3);
    check_error("norma comun", scratch_floats(engine, "_s.mlp_out", size_t(TOKENS) * H),
                read_file<float>(oracle + "/common_norm.f32"), 2e-3);
    const auto batched = scratch_floats(engine, HIDDEN, size_t(TOKENS) * H);
    check_error("salida de la capa", batched, want_output, 2e-3);

    // L > 1 y token a token deben dar lo mismo: es lo que comprueba el reparto por experto.
    std::vector<float> one_by_one;
    for (uint32_t token = 0; token < TOKENS; ++token) {
        set_residual(engine, std::vector<half>(residual.begin() + size_t(token) * H,
                                               residual.begin() + size_t(token + 1) * H));
        engine.execute(builder.build_gemma4_mlp_ple_tail(engine, config, gemma, arch, 0, 1, 1));
        const auto row = scratch_floats(engine, HIDDEN, H);
        one_by_one.insert(one_by_one.end(), row.begin(), row.end());
    }
    check_error("token a token contra lote", one_by_one, batched, 2e-3);

    // La referencia sincroniza: pedir grafo debe caer a ejecucion normal, no capturarla.
    set_residual(engine, residual);
    const auto tail = builder.build_gemma4_mlp_ple_tail(engine, config, gemma, arch, 0, 1, TOKENS);
    engine.execute_graphed(tail);
    check_error("execute_graphed", scratch_floats(engine, HIDDEN, size_t(TOKENS) * H), batched, 1e-6);
    set_residual(engine, residual);
    engine.execute_graph_replay(tail);
    check_error("execute_graph_replay", scratch_floats(engine, HIDDEN, size_t(TOKENS) * H), batched, 1e-6);

    if (!out.empty()) {
        dump_outputs(engine, tail, residual, oracle, out);
        for (const char* name : {"tie_ids.i32", "tie_weights.f32"}) {
            std::filesystem::rename(out + ".ties/" + name, out + "/" + name);
        }
        std::filesystem::remove(out + ".ties");
        std::cout << "  salidas en " << out << std::endl;
    }

    // Con la bandera y sin dimensiones hay que fallar, no correr con top-k 0.
    helios::Gemma4Config broken = gemma;
    broken.moe_top_k = 0;
    bool rejected = false;
    try {
        builder.allocate_gemma4_scratch(engine, config, broken, arch, 1, 1);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "bandera MoE sin dimensiones");
}

}  // namespace

int main(int argc, char** argv) {
    require(argc == 2 || argc == 3, "uso: test_gemma4_moe_layer references/ [salidas/]");
    test_layer(argv[1], argc == 3 ? argv[2] : "");
    std::cout << "Gemma 4 MoE capa real OK" << std::endl;
    return 0;
}
