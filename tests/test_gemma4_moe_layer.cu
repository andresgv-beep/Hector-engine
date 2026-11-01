// Cola MLP + MoE de una capa real contra el oraculo FP32 de
// converter/tools/gemma4_moe_layer_oracle.py. Pesos originales BF16 pasados a FP16.
//
// uso: test_gemma4_moe_layer references/ [salidas/ [payloads/]]
//
// Con un segundo argumento vuelca los 23 tensores que pide
// converter/tools/compare_moe_oracle.py en un directorio nuevo. Con un tercero
// comprueba los payloads de capa 0 del conversor y vuelca su decode en
// salidas-fp16/ y salidas-hq44/.
#include "graph_builder.hpp"
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
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
            launch_moe_gate_up_mul(projected, activated, 1, I, I, 0);
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

// ---------------------------------------------------------------------------
// Payloads reales del conversor: descriptor, lectura, salida de la cola y tiempos.
// ---------------------------------------------------------------------------
constexpr uint32_t PHYSICAL_I = 768, SHARED_I = 2112, PHYSICAL_SHARED = 2304;

void check_descriptor(const std::string& path, uint32_t role, uint32_t storage, uint32_t rows,
                      uint32_t cols, uint32_t physical, uint32_t experts, size_t payload_bytes) {
    const auto bytes = read_file<char>(path);
    require(bytes.size() == sizeof(helios::Gemma4MoeMatrixBin), path + ": tamaño");
    helios::Gemma4MoeMatrixBin bin{};
    std::memcpy(&bin, bytes.data(), sizeof(bin));
    const bool hq44 = storage == helios::GEMMA4_MOE_STORAGE_HQ44;
    const uint64_t row_bytes = hq44 ? uint64_t{physical} / 256 * 152 : uint64_t{physical} * 2;
    require(bin.role == role && bin.storage == storage && bin.rows == rows && bin.logical_cols == cols &&
            bin.physical_rows == rows && bin.physical_cols == physical && bin.experts == experts &&
            bin.layout == (hq44 ? 1u : 0u) && bin.row_bytes == row_bytes &&
            bin.expert_stride == row_bytes * rows, path + ": campos");
    require(bin.expert_stride * experts == payload_bytes, path + ": el payload no cuadra con el descriptor");
}

std::vector<char> read_payload(const std::string& payloads, const std::string& base, const char* storage,
                               uint32_t role, uint32_t rows, uint32_t cols, uint32_t physical, uint32_t experts) {
    const std::string stem = payloads + "/" + base + "." + storage;
    auto bytes = read_file<char>(stem + ".bin");
    const uint32_t code = std::string(storage) == "hq44" ? helios::GEMMA4_MOE_STORAGE_HQ44
                                                         : helios::GEMMA4_MOE_STORAGE_FP16;
    check_descriptor(stem + ".descriptor.bin", role, code, rows, cols, physical, experts, bytes.size());
    return bytes;
}

void replace_weight(helios::Engine& engine, const std::string& name, const std::vector<uint32_t>& shape,
                    helios::DTypeID type, const std::vector<char>& bytes) {
    engine.tensors().remove(name);
    void* device = engine.tensors().allocate_and_register(name, shape, type);
    require(engine.tensors().at(name).size_bytes == bytes.size(), name + ": tamaño en el registro");
    cuda_require(cudaMemcpy(device, bytes.data(), bytes.size(), cudaMemcpyHostToDevice), name);
}

// El payload FP16 debe ser el peso original en las columnas lógicas y cero en el relleno.
void check_fp16_payload(const std::string& label, const std::vector<char>& payload,
                        const std::vector<half>& source, size_t cols, size_t physical) {
    const half* values = reinterpret_cast<const half*>(payload.data());
    const size_t rows = source.size() / cols;
    require(payload.size() == rows * physical * sizeof(half), label + ": filas");
    size_t different = 0, nonzero = 0;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t c = 0; c < physical; ++c) {
            const uint16_t got = __half_as_ushort(values[row * physical + c]);
            if (c >= cols) nonzero += got != 0;
            else different += got != __half_as_ushort(source[row * cols + c]);
        }
    }
    std::cout << "  " << label << " FP16: " << different << " valores distintos del original, "
              << nonzero << " no nulos en el relleno" << std::endl;
    require(different == 0 && nonzero == 0, label + ": payload FP16");
}

// El motor descuantiza el HQ44 y lo compara con el original: es su lectura, no la del conversor.
void check_hq44_payload(const std::string& label, const std::vector<char>& payload, const std::vector<half>& source,
                        size_t matrices, size_t rows, size_t cols, size_t physical, double reported) {
    const size_t stride = rows * (physical / 256) * 152;
    require(payload.size() == matrices * stride, label + ": tamaño HQ44");
    uint8_t* weights = nullptr;
    half* decoded = nullptr;
    cuda_require(cudaMalloc(&weights, stride), label);
    cuda_require(cudaMalloc(&decoded, rows * physical * sizeof(half)), label);
    double error = 0.0, norm = 0.0, padding = 0.0;
    for (size_t matrix = 0; matrix < matrices; ++matrix) {
        cuda_require(cudaMemcpy(weights, payload.data() + matrix * stride, stride, cudaMemcpyHostToDevice), label);
        helios::kernels::launch_dequant_hq44k(weights, decoded, int(physical), int(rows), 0);
        const auto got = download<half>(decoded, rows * physical);
        const half* want = source.data() + matrix * rows * cols;
        for (size_t row = 0; row < rows; ++row) {
            for (size_t c = 0; c < physical; ++c) {
                const double value = __half2float(got[row * physical + c]);
                if (c >= cols) {
                    padding = std::max(padding, std::fabs(value));
                    continue;
                }
                const double expected = __half2float(want[row * cols + c]);
                error += (value - expected) * (value - expected);
                norm += expected * expected;
            }
        }
    }
    cudaFree(weights);
    cudaFree(decoded);
    const double relative = std::sqrt(error / norm);
    std::cout << "  " << label << " HQ44 leido por el motor: error " << relative << " (conversor " << reported
              << "), maximo en el relleno " << padding << std::endl;
    require(std::fabs(relative - reported) < 1e-4, label + ": lectura HQ44 distinta del conversor");
    require(padding == 0.0, label + ": relleno HQ44 no nulo");
}

struct DecodeDump {
    std::vector<float> expert_input, weights;
    std::vector<int32_t> ids;
};

// Los 23 tensores del oráculo por la ruta real de decode, un token cada vez.
DecodeDump dump_decode(helios::Engine& engine, helios::GraphBuilder& builder, const helios::ModelConfig& config,
                       const helios::Gemma4Config& gemma, const helios::ArchDescriptor& arch,
                       const std::vector<half>& residual, const std::string& oracle, const std::string& base_out,
                       const std::string& out) {
    using namespace helios::kernels;
    require(std::filesystem::create_directory(out), out + " ya existe");
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
    std::map<std::string, std::vector<float>> saved;
    const auto keep = [&](const std::string& name, const std::vector<float>& values) {
        auto& all = saved[name];
        all.insert(all.end(), values.begin(), values.end());
    };
    DecodeDump dump;
    float* probabilities = nullptr;
    float* normalized = nullptr;
    int32_t* route_ids = nullptr;
    half* unit_scale = nullptr;
    cuda_require(cudaMalloc(&probabilities, E * sizeof(float)), "probabilidades");
    cuda_require(cudaMalloc(&normalized, K * sizeof(float)), "normalizados");
    cuda_require(cudaMalloc(&route_ids, K * sizeof(int32_t)), "ids");
    cuda_require(cudaMalloc(&unit_scale, E * sizeof(half)), "escala");
    const std::vector<half> ones(E, __float2half(1.0f));
    cuda_require(cudaMemcpy(unit_scale, ones.data(), E * sizeof(half), cudaMemcpyHostToDevice), "escala");

    for (uint32_t token = 0; token < TOKENS; ++token) {
        set_residual(engine, std::vector<half>(residual.begin() + size_t(token) * H,
                                               residual.begin() + size_t(token + 1) * H));
        const auto tail = builder.build_gemma4_mlp_ple_tail(engine, config, gemma, arch, 0, 1, 1);
        std::map<std::string, int> seen;
        int found = 0;
        for (const auto& command : tail.commands()) {
            engine.execute_command(command);
            const auto step = steps.find({command.output, ++seen[command.output]});
            if (step == steps.end()) continue;
            keep(step->second, scratch_floats(engine, command.output, command.output == "_s.g4.moe_logits" ? E : H));
            ++found;
        }
        require(found == int(steps.size()), "la cola de decode no tiene los pasos esperados");

        const auto ids = download<int32_t>(engine.tensors().at("_s.g4.moe_indices").ptr, K);
        const auto weights = download<float>(engine.tensors().at("_s.g4.moe_weights").ptr, K);
        dump.ids.insert(dump.ids.end(), ids.begin(), ids.end());
        dump.weights.insert(dump.weights.end(), weights.begin(), weights.end());

        const half* logits = static_cast<const half*>(engine.tensors().at("_s.g4.moe_logits").ptr);
        launch_moe_probabilities(logits, probabilities, 1, E, 0);
        keep("router_probabilities", download<float>(probabilities, E));
        launch_moe_route(logits, unit_scale, route_ids, normalized, 1, E, K, 0);
        keep("normalized_weights", download<float>(normalized, K));

        const auto gate_up = download<half>(engine.tensors().at("_s.g4.moe_gate_up").ptr, size_t(K) * 2 * I);
        const auto hidden = download<half>(engine.tensors().at("_s.g4.moe_hidden").ptr, size_t(K) * PHYSICAL_I);
        const auto rows = download<half>(engine.tensors().at("_s.g4.moe_rows").ptr, size_t(K) * H);
        for (uint32_t slot = 0; slot < K; ++slot) {
            for (uint32_t i = 0; i < I; ++i) {
                saved["gate"].push_back(__half2float(gate_up[size_t(slot) * 2 * I + i]));
                saved["gelu_mul"].push_back(__half2float(hidden[size_t(slot) * PHYSICAL_I + i]));
            }
            for (uint32_t i = 0; i < I; ++i) saved["up"].push_back(__half2float(gate_up[size_t(slot) * 2 * I + I + i]));
            for (uint32_t i = 0; i < H; ++i) {
                saved["expert_contributions"].push_back(__half2float(rows[size_t(slot) * H + i]) * weights[slot]);
            }
        }
    }
    dump.expert_input = saved["expert_input"];
    for (const auto& [name, values] : saved) write_file(out + "/" + name + ".f32", values);
    write_file(out + "/expert_ids.i32", dump.ids);
    write_file(out + "/expert_weights.f32", dump.weights);
    for (const char* name : {"residual.f32", "tie_logits.f32", "tie_scales.f32"}) {
        write_file(out + "/" + name, read_file<char>(oracle + "/" + name));
    }
    for (const char* name : {"tie_ids.i32", "tie_weights.f32"}) {
        write_file(out + "/" + name, read_file<char>(base_out + "/" + name));
    }
    for (void* p : {static_cast<void*>(probabilities), static_cast<void*>(normalized),
                    static_cast<void*>(route_ids), static_cast<void*>(unit_scale)}) {
        cudaFree(p);
    }
    return dump;
}

double median_ms(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

// Los expertos que eligió el router real, solos, dentro de un grafo.
void time_decode(helios::Engine& engine, const std::string& label, const DecodeDump& dump) {
    helios::CommandBuffer experts;
    experts.add_moe_experts_decode("_s.g4.moe_out", "_s.normed", "_s.g4.moe_indices", "_s.g4.moe_weights",
                                   "text.layer0.moe.experts.gate_up.weight",
                                   "text.layer0.moe.experts.down.weight",
                                   "_s.g4.moe_gate_up", "_s.g4.moe_hidden", "_s.g4.moe_rows");
    const auto input = to_half(dump.expert_input);
    engine.invalidate_graph();
    std::vector<double> runs;
    for (int run = 0; run < 400; ++run) {
        const size_t token = size_t(run) % TOKENS;
        cuda_require(cudaMemcpy(engine.tensors().at("_s.normed").ptr, input.data() + token * H, H * sizeof(half),
                                cudaMemcpyHostToDevice), "entrada");
        cuda_require(cudaMemcpy(engine.tensors().at("_s.g4.moe_indices").ptr, dump.ids.data() + token * K,
                                K * sizeof(int32_t), cudaMemcpyHostToDevice), "indices");
        cuda_require(cudaMemcpy(engine.tensors().at("_s.g4.moe_weights").ptr, dump.weights.data() + token * K,
                                K * sizeof(float), cudaMemcpyHostToDevice), "pesos");
        const auto start = std::chrono::steady_clock::now();
        engine.execute_graph_replay(experts);
        engine.sync();
        if (run >= 20) {
            runs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
    }
    require(engine.graph_ready(), "los expertos deben grabarse en el grafo");
    engine.invalidate_graph();
    const double ms = median_ms(runs);
    std::cout << "  TIEMPO " << label << ": " << ms << " ms por capa, " << ms * 30 << " ms por 30 capas" << std::endl;
}

void test_payloads(helios::Engine& engine, helios::GraphBuilder& builder, const helios::ModelConfig& config,
                   const helios::Gemma4Config& padded, const helios::ArchDescriptor& arch, LayerFile& source,
                   const std::vector<half>& residual, const std::string& oracle, const std::string& out,
                   const std::string& payloads) {
    const std::string prefix = "model.language_model.layers.0.";
    const auto source_gate_up = source.read_fp16(prefix + "experts.gate_up_proj");
    const auto source_down = source.read_fp16(prefix + "experts.down_proj");
    const auto source_shared = source.read_fp16(prefix + "mlp.down_proj.weight");
    const std::string gate_up = "text.layer0.moe.experts.gate_up.weight";
    const std::string down = "text.layer0.moe.experts.down.weight";
    const std::string shared = "text.layer0.mlp.down.weight";
    const auto read = [&](const char* storage) {
        return std::array<std::vector<char>, 3>{
            read_payload(payloads, "moe.experts.gate_up.weight", storage, helios::GEMMA4_MOE_ROLE_GATE_UP,
                         2 * I, H, H, E),
            read_payload(payloads, "moe.experts.down.weight", storage, helios::GEMMA4_MOE_ROLE_DOWN,
                         H, I, PHYSICAL_I, E),
            read_payload(payloads, "mlp.down.weight", storage, helios::GEMMA4_MOE_ROLE_SHARED_DOWN,
                         H, SHARED_I, PHYSICAL_SHARED, 1)};
    };
    const auto fp16 = read("fp16");
    const auto hq44 = read("hq44");
    std::cout << "  descriptores y tamaños: correctos en los 6 payloads" << std::endl;

    check_fp16_payload("gate_up", fp16[0], source_gate_up, H, H);
    check_fp16_payload("down", fp16[1], source_down, I, PHYSICAL_I);
    check_fp16_payload("down compartido", fp16[2], source_shared, SHARED_I, PHYSICAL_SHARED);
    check_hq44_payload("gate_up", hq44[0], source_gate_up, E, 2 * I, H, H, 0.05983316);
    check_hq44_payload("down", hq44[1], source_down, E, H, I, PHYSICAL_I, 0.05965538);
    check_hq44_payload("down compartido", hq44[2], source_shared, 1, H, SHARED_I, PHYSICAL_SHARED, 0.06373880);

    const auto load = [&](const std::array<std::vector<char>, 3>& weights, helios::DTypeID type) {
        replace_weight(engine, gate_up, {E, 2 * I, H}, type, weights[0]);
        replace_weight(engine, down, {E, H, PHYSICAL_I}, type, weights[1]);
        replace_weight(engine, shared, {H, PHYSICAL_SHARED}, type, weights[2]);
    };
    load(fp16, FP16());
    const auto fp16_dump = dump_decode(engine, builder, config, padded, arch, residual, oracle, out, out + "-fp16");
    time_decode(engine, "expertos FP16", fp16_dump);
    load(hq44, helios::dtype::HQ44K());
    const auto hq44_dump = dump_decode(engine, builder, config, padded, arch, residual, oracle, out, out + "-hq44");
    time_decode(engine, "expertos HQ44", hq44_dump);
    require(fp16_dump.ids == hq44_dump.ids, "el router no depende de los expertos");

    set_residual(engine, std::vector<half>(residual.begin(), residual.begin() + H));
    const auto tail = builder.build_gemma4_mlp_ple_tail(engine, config, padded, arch, 0, 1, 1);
    engine.invalidate_graph();
    std::vector<double> runs;
    for (int run = 0; run < 240; ++run) {
        const size_t token = size_t(run) % TOKENS;
        set_residual(engine, std::vector<half>(residual.begin() + token * H, residual.begin() + (token + 1) * H));
        const auto start = std::chrono::steady_clock::now();
        engine.execute_graph_replay(tail);
        engine.sync();
        if (run >= 20) {
            runs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
    }
    require(engine.graph_ready(), "la cola debe grabarse en el grafo");
    engine.invalidate_graph();
    std::cout << "  TIEMPO cola completa HQ44 en grafo: " << median_ms(runs) << " ms por capa" << std::endl;
}

void test_layer(const std::string& references, const std::string& out, const std::string& payloads) {
    const std::string oracle = references + "/layer-00-oracle-fp32";
    // El stream por defecto no se puede capturar en un CUDA Graph.
    helios::EngineConfig engine_config;
    cuda_require(cudaStreamCreate(&engine_config.stream), "stream");
    helios::Engine engine(engine_config);
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

    // Un token va por el decode. Aqui los matmul FP16 van por cuBLAS y no se graban,
    // asi que el replay cae a execute; la captura se prueba con la ruta MoE sola.
    const auto count_ops = [](const helios::CommandBuffer& commands, helios::OpTypeID op) {
        return std::count_if(commands.commands().begin(), commands.commands().end(),
                             [&](const helios::Command& cmd) { return cmd.op == op; });
    };
    set_residual(engine, std::vector<half>(residual.begin(), residual.begin() + H));
    const auto decode = builder.build_gemma4_mlp_ple_tail(engine, config, gemma, arch, 0, 1, 1);
    require(count_ops(decode, helios::op::MOE_EXPERTS_DECODE()) == 1 &&
            count_ops(decode, helios::op::MOE_EXPERTS()) == 0, "un token usa el decode");
    engine.execute(decode);
    const auto direct = scratch_floats(engine, HIDDEN, H);
    check_error("decode contra lote", direct,
                std::vector<float>(one_by_one.begin(), one_by_one.begin() + H), 1e-6);
    for (uint32_t token = 0; token < TOKENS; ++token) {
        set_residual(engine, std::vector<half>(residual.begin() + size_t(token) * H,
                                               residual.begin() + size_t(token + 1) * H));
        engine.execute_graph_replay(decode);
        const auto row = scratch_floats(engine, HIDDEN, H);
        check_error("replay del decode, token " + std::to_string(token), row,
                    std::vector<float>(one_by_one.begin() + size_t(token) * H,
                                       one_by_one.begin() + size_t(token + 1) * H), 1e-6);
    }

    helios::CommandBuffer moe;
    moe.add_moe_route("_s.g4.moe_indices", "_s.g4.moe_weights", "_s.g4.moe_logits",
                      "text.layer0.moe.router.expert_scale", K);
    moe.add_moe_experts_decode("_s.g4.moe_out", HIDDEN, "_s.g4.moe_indices", "_s.g4.moe_weights",
                               "text.layer0.moe.experts.gate_up.weight",
                               "text.layer0.moe.experts.down.weight",
                               "_s.g4.moe_gate_up", "_s.g4.moe_hidden", "_s.g4.moe_rows");
    const auto set_logits = [&](uint32_t token) {
        std::vector<half> logits(E);
        for (uint32_t e = 0; e < E; ++e) {
            logits[e] = __float2half(std::sin(float(e * (token + 3)) * 0.7f) * 4.0f);
        }
        cuda_require(cudaMemcpy(engine.tensors().at("_s.g4.moe_logits").ptr, logits.data(),
                                E * sizeof(half), cudaMemcpyHostToDevice), "logits");
    };
    std::vector<std::vector<float>> direct_out;
    for (uint32_t token = 0; token < TOKENS; ++token) {
        set_residual(engine, std::vector<half>(residual.begin() + size_t(token) * H,
                                               residual.begin() + size_t(token + 1) * H));
        set_logits(token);
        engine.execute(moe);
        direct_out.push_back(scratch_floats(engine, "_s.g4.moe_out", H));
    }
    engine.invalidate_graph();
    for (uint32_t token = 0; token < TOKENS; ++token) {
        set_residual(engine, std::vector<half>(residual.begin() + size_t(token) * H,
                                               residual.begin() + size_t(token + 1) * H));
        set_logits(token);
        engine.execute_graph_replay(moe);
        require(engine.graph_ready(), "la ruta MoE de un token debe grabarse en el grafo");
        check_error("grafo MoE, token " + std::to_string(token),
                    scratch_floats(engine, "_s.g4.moe_out", H), direct_out[token], 1e-6);
    }
    engine.invalidate_graph();

    // Con padding el resultado es el mismo. Las columnas extra llevan pesos de 1:
    // si la entrada no llega a cero se nota en la salida.
    const auto pad_weight = [&](const std::string& name, uint32_t rows, uint32_t cols, uint32_t physical) {
        const helios::TensorInfo& tensor = engine.tensors().at(name);
        const size_t matrices = tensor.numel() / (size_t(rows) * cols);
        const std::vector<uint32_t> shape = tensor.shape.size() == 3
            ? std::vector<uint32_t>{tensor.shape[0], rows, physical}
            : std::vector<uint32_t>{rows, physical};
        const auto values = download<half>(static_cast<const half*>(tensor.ptr), tensor.numel());
        std::vector<half> padded(matrices * rows * physical, __float2half(1.0f));
        for (size_t row = 0; row < matrices * rows; ++row) {
            std::copy_n(values.begin() + row * cols, cols, padded.begin() + row * physical);
        }
        engine.tensors().remove(name);
        void* device = engine.tensors().allocate_and_register(name, shape, FP16());
        cuda_require(cudaMemcpy(device, padded.data(), padded.size() * sizeof(half),
                                cudaMemcpyHostToDevice), "peso con relleno");
    };
    constexpr uint32_t PI = 768, PS = 2304;
    pad_weight("text.layer0.moe.experts.down.weight", H, I, PI);
    pad_weight("text.layer0.mlp.down.weight", H, 2112, PS);
    helios::Gemma4Config padded = gemma;
    padded.layers[0].moe_down.physical_cols = PI;
    padded.layers[0].moe_shared_down.physical_cols = PS;
    builder.allocate_gemma4_scratch(engine, config, padded, arch, 1, TOKENS);
    set_residual(engine, residual);
    const auto padded_batch = builder.build_gemma4_mlp_ple_tail(engine, config, padded, arch, 0, 1, TOKENS);
    require(count_ops(padded_batch, helios::op::PAD_COLUMNS()) == 1, "el down compartido se rellena");
    engine.execute(padded_batch);
    check_error("lote con padding", scratch_floats(engine, HIDDEN, size_t(TOKENS) * H), batched, 2e-3);
    for (uint32_t token = 0; token < TOKENS; ++token) {
        set_residual(engine, std::vector<half>(residual.begin() + size_t(token) * H,
                                               residual.begin() + size_t(token + 1) * H));
        engine.execute(builder.build_gemma4_mlp_ple_tail(engine, config, padded, arch, 0, 1, 1));
        check_error("decode con padding, token " + std::to_string(token),
                    scratch_floats(engine, HIDDEN, H),
                    std::vector<float>(one_by_one.begin() + size_t(token) * H,
                                       one_by_one.begin() + size_t(token + 1) * H), 2e-3);
    }

    if (!payloads.empty()) {
        test_payloads(engine, builder, config, padded, arch, source, residual, oracle, out, payloads);
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
    require(argc >= 2 && argc <= 4, "uso: test_gemma4_moe_layer references/ [salidas/ [payloads/]]");
    test_layer(argv[1], argc >= 3 ? argv[2] : "", argc == 4 ? argv[3] : "");
    std::cout << "Gemma 4 MoE capa real OK" << std::endl;
    return 0;
}
