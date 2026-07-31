#include "gemma4_ple.hpp"
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void cuda_require(cudaError_t error, const std::string& where) {
    require(error == cudaSuccess, where + ": " + cudaGetErrorString(error));
}

float fp16(float value) {
    return __half2float(__float2half(value));
}

float read_fp16(const std::vector<uint8_t>& bytes, size_t index) {
    require((index + 1) * sizeof(half) <= bytes.size(), "FP16 read outside tensor");
    half value{};
    std::memcpy(&value, bytes.data() + index * sizeof(half), sizeof(value));
    return __half2float(value);
}

float decode_hq51k_element(const uint8_t* block, int in_block) {
    half d_scale_h{}, d_min_h{}, min_base_h{};
    std::memcpy(&d_scale_h, block, 2);
    std::memcpy(&d_min_h, block + 2, 2);
    std::memcpy(&min_base_h, block + 4, 2);

    const int group = in_block / 8;
    const int in_group = in_block % 8;
    const uint8_t scale_byte = block[8 + group / 2];
    const uint8_t q_scale = group % 2 == 0 ? scale_byte >> 4 : scale_byte & 0x0f;
    const uint8_t min_byte = block[24 + group / 2];
    const uint8_t q_min = group % 2 == 0 ? min_byte >> 4 : min_byte & 0x0f;
    const float scale = __half2float(d_scale_h) * float(q_scale) / 15.0f;
    const float minimum = __half2float(min_base_h) +
                          __half2float(d_min_h) * float(q_min) / 15.0f;

    const uint8_t* payload = block + 40 + group * 5;
    const uint64_t packed = uint64_t(payload[0]) |
                            (uint64_t(payload[1]) << 8) |
                            (uint64_t(payload[2]) << 16) |
                            (uint64_t(payload[3]) << 24) |
                            (uint64_t(payload[4]) << 32);
    const uint32_t q = uint32_t((packed >> (in_group * 5)) & 31);
    return minimum + float(q) * scale / 31.0f;
}

const helios::TensorEntry& tensor_named(const helios::HnfLoader& loader,
                                         const std::string& name) {
    for (const auto& tensor : loader.tensors()) {
        if (tensor.name == name) return tensor;
    }
    throw std::runtime_error("missing tensor: " + name);
}

std::vector<uint8_t> read_bytes(const std::string& path, uint64_t offset,
                                size_t size) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "cannot open HNF: " + path);
    input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    require(input.good(), "cannot seek inside HNF");
    std::vector<uint8_t> result(size);
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(size));
    require(input.gcount() == static_cast<std::streamsize>(size),
            "short HNF tensor read");
    return result;
}

void test_command_contract() {
    helios::ModelConfig model;
    model.set("hidden_size", int64_t(1536));
    model.set("num_hidden_layers", int64_t(35));
    model.set("rms_norm_eps", 1.0e-6);

    helios::Gemma4Config gemma;
    gemma.flags = helios::GEMMA4_EXT_FLAG_PLE |
                  helios::GEMMA4_EXT_FLAG_LAYER_SCALAR;
    gemma.ple_hidden_size = 256;
    gemma.layers.resize(35);

    helios::CommandBuffer commands;
    helios::append_gemma4_ple_preparation(commands, model, gemma);
    require(commands.size() == 7, "PLE preparation must contain seven commands");
    require(commands[0].op == helios::op::EMBEDDING(), "PLE command 0 must be lookup");
    require(commands[1].op == helios::op::SCALE(), "PLE command 1 must scale token PLE");
    require(commands[2].op == helios::op::MATMUL(), "PLE command 2 must project context");
    require(commands[3].op == helios::op::SCALE(), "PLE command 3 must scale context");
    require(commands[4].op == helios::op::RMSNORM(), "PLE command 4 must normalize context");
    require(commands[4].get<uint32_t>("dim", 0) == 256,
            "PLE RMSNorm must operate on 256-value segments");
    require(commands[5].op == helios::op::ADD(), "PLE command 5 must combine components");
    require(commands[6].op == helios::op::SCALE(), "PLE command 6 must apply 1/sqrt(2)");
    require(std::fabs(commands[1].get<float>("scalar", 0.0f) - 16.0f) < 1.0e-6f,
            "PLE token scale must be sqrt(256)");
    require(std::fabs(commands[3].get<float>("scalar", 0.0f) -
                      1.0f / std::sqrt(1536.0f)) < 1.0e-7f,
            "PLE context scale must be 1/sqrt(hidden)");

    helios::CommandBuffer layer_commands;
    helios::append_gemma4_ple_layer_injection(
        layer_commands, model, gemma, 4);
    require(layer_commands.size() == 7,
            "PLE layer injection must contain seven commands");
    require(layer_commands[0].op == helios::op::MATMUL() &&
            layer_commands[0].inputs[1] == "text.layer4.ple.input_gate.weight",
            "PLE layer injection must use the selected gate");
    require(layer_commands[1].op == helios::op::GELU(),
            "PLE layer injection must use GELU tanh");
    require(layer_commands[2].op == helios::op::MUL(),
            "PLE layer injection must multiply its segment");
    require(layer_commands[3].op == helios::op::MATMUL(),
            "PLE layer injection must project to hidden");
    require(layer_commands[4].op == helios::op::RMSNORM(),
            "PLE layer injection must apply its post norm");
    require(layer_commands[5].op == helios::op::ADD(),
            "PLE layer injection must add its residual");
    require(layer_commands[6].op == helios::op::MUL_SCALAR_TENSOR(),
            "PLE layer injection must finish with layer_scalar");
    std::cout << "PASS: PLE preparation command contract" << std::endl;
}

void test_real_rows(const std::string& path, const helios::HnfLoader& loader) {
    const auto& ple = tensor_named(loader, "text.ple.token_embedding.weight");
    require(ple.dtype == "hq51k" && ple.shape.size() == 2,
            "real PLE table must be HQ51K rank 2");
    const int vocab = static_cast<int>(ple.shape[0]);
    const int width = static_cast<int>(ple.shape[1]);
    require(width % 256 == 0, "PLE width must contain whole HQ51K blocks");
    const size_t row_bytes = size_t(width / 256) * 200;
    const std::vector<int> token_ids{0, 42, vocab - 1};
    std::vector<uint8_t> rows(row_bytes * token_ids.size());
    for (size_t row = 0; row < token_ids.size(); ++row) {
        const auto data = read_bytes(path,
            ple.offset + uint64_t(token_ids[row]) * row_bytes, row_bytes);
        std::copy(data.begin(), data.end(), rows.begin() + row * row_bytes);
    }

    uint8_t* device_rows = nullptr;
    int32_t* device_ids = nullptr;
    half* device_output = nullptr;
    cuda_require(cudaMalloc(&device_rows, rows.size()), "allocate real PLE rows");
    cuda_require(cudaMalloc(&device_ids, token_ids.size() * sizeof(int32_t)),
                 "allocate compact PLE ids");
    cuda_require(cudaMalloc(&device_output,
                            token_ids.size() * size_t(width) * sizeof(half)),
                 "allocate real PLE row output");
    const std::vector<int32_t> compact_ids{0, 1, 2};
    cuda_require(cudaMemcpy(device_rows, rows.data(), rows.size(), cudaMemcpyHostToDevice),
                 "copy real PLE rows");
    cuda_require(cudaMemcpy(device_ids, compact_ids.data(),
                            compact_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
                 "copy compact PLE ids");
    helios::kernels::launch_embedding_hq51k(
        device_ids, device_rows, device_output, 1, compact_ids.size(),
        compact_ids.size(), width);
    cuda_require(cudaDeviceSynchronize(), "real PLE row lookup");
    std::vector<half> output(token_ids.size() * size_t(width));
    cuda_require(cudaMemcpy(output.data(), device_output,
                            output.size() * sizeof(half), cudaMemcpyDeviceToHost),
                 "copy real PLE lookup output");
    cudaFree(device_rows);
    cudaFree(device_ids);
    cudaFree(device_output);

    for (size_t row = 0; row < token_ids.size(); ++row) {
        for (int d = 0; d < width; ++d) {
            const uint8_t* block = rows.data() + row * row_bytes + size_t(d / 256) * 200;
            const float expected = fp16(decode_hq51k_element(block, d % 256));
            require(std::fabs(__half2float(output[row * width + d]) - expected) < 0.003f,
                    "real PLE lookup differs from CPU decoder");
        }
    }
    std::cout << "PASS: real PLE rows 0, 42 and " << vocab - 1
              << " (" << rows.size() << " bytes read)" << std::endl;
}

void upload(helios::TensorRegistry& tensors, const std::string& name,
            const std::vector<uint32_t>& shape, helios::DTypeID dtype,
            const void* data, size_t size) {
    void* destination = tensors.allocate_and_register(name, shape, dtype);
    require(tensors.at(name).size_bytes == size,
            "allocated tensor size differs for " + name);
    cuda_require(cudaMemcpy(destination, data, size, cudaMemcpyHostToDevice),
                 "upload " + name);
}

std::vector<float> cpu_ple_reference(
    const std::vector<uint8_t>& main_row,
    const std::vector<uint8_t>& ple_row,
    const std::vector<uint8_t>& projection,
    const std::vector<uint8_t>& norm_bytes,
    uint32_t hidden, uint32_t layers, uint32_t ple_hidden, float eps) {
    const uint32_t width = layers * ple_hidden;
    std::vector<float> main(hidden);
    for (uint32_t k = 0; k < hidden; ++k) {
        main[k] = fp16(read_fp16(main_row, k) * std::sqrt(float(hidden)));
    }

    std::vector<float> identity(width);
    for (uint32_t d = 0; d < width; ++d) {
        const uint8_t* block = ple_row.data() + size_t(d / 256) * 200;
        identity[d] = fp16(fp16(decode_hq51k_element(block, d % 256)) *
                           std::sqrt(float(ple_hidden)));
    }

    const size_t blocks_per_projection_row = hidden / 256;
    std::vector<float> context(width);
    for (uint32_t n = 0; n < width; ++n) {
        float sum = 0.0f;
        const uint8_t* row = projection.data() +
                             size_t(n) * blocks_per_projection_row * 200;
        for (uint32_t k = 0; k < hidden; ++k) {
            const float weight = decode_hq51k_element(
                row + size_t(k / 256) * 200, k % 256);
            sum += main[k] * weight;
        }
        context[n] = fp16(fp16(sum) / std::sqrt(float(hidden)));
    }

    for (uint32_t layer = 0; layer < layers; ++layer) {
        float sum_squares = 0.0f;
        const uint32_t base = layer * ple_hidden;
        for (uint32_t d = 0; d < ple_hidden; ++d) {
            sum_squares += context[base + d] * context[base + d];
        }
        const float inverse_rms = 1.0f /
            std::sqrt(sum_squares / float(ple_hidden) + eps);
        for (uint32_t d = 0; d < ple_hidden; ++d) {
            context[base + d] = fp16(context[base + d] * inverse_rms *
                                     read_fp16(norm_bytes, d));
        }
    }

    std::vector<float> result(width);
    for (uint32_t d = 0; d < width; ++d) {
        const float combined = fp16(context[d] + identity[d]);
        result[d] = fp16(combined * (1.0f / std::sqrt(2.0f)));
    }
    return result;
}

std::vector<float> cpu_hq51k_matmul(const std::vector<float>& input,
                                     const std::vector<uint8_t>& weights,
                                     uint32_t output_width) {
    require(input.size() % 256 == 0, "CPU HQ51K matmul input alignment");
    const uint32_t input_width = static_cast<uint32_t>(input.size());
    const size_t row_bytes = size_t(input_width / 256) * 200;
    require(weights.size() == size_t(output_width) * row_bytes,
            "CPU HQ51K matmul storage size");
    std::vector<float> output(output_width);
    for (uint32_t n = 0; n < output_width; ++n) {
        float sum = 0.0f;
        const uint8_t* row = weights.data() + size_t(n) * row_bytes;
        for (uint32_t k = 0; k < input_width; ++k) {
            const float weight = decode_hq51k_element(
                row + size_t(k / 256) * 200, k % 256);
            sum += input[k] * weight;
        }
        output[n] = fp16(sum);
    }
    return output;
}

std::vector<float> cpu_ple_layer_reference(
    const std::vector<float>& hidden,
    const std::vector<float>& ple_segment,
    const std::vector<uint8_t>& gate_weight,
    const std::vector<uint8_t>& projection_weight,
    const std::vector<uint8_t>& norm_bytes,
    const std::vector<uint8_t>& scalar_bytes,
    float eps) {
    std::vector<float> gate = cpu_hq51k_matmul(
        hidden, gate_weight, static_cast<uint32_t>(ple_segment.size()));
    for (size_t i = 0; i < gate.size(); ++i) {
        const float x = gate[i];
        const float activated = 0.5f * x *
            (1.0f + std::tanh(0.7978845608f *
             (x + 0.044715f * x * x * x)));
        gate[i] = fp16(fp16(activated) * ple_segment[i]);
    }

    std::vector<float> projected = cpu_hq51k_matmul(
        gate, projection_weight, static_cast<uint32_t>(hidden.size()));
    float sum_squares = 0.0f;
    for (float value : projected) sum_squares += value * value;
    const float inverse_rms = 1.0f /
        std::sqrt(sum_squares / float(projected.size()) + eps);
    std::vector<float> result(hidden.size());
    for (size_t i = 0; i < result.size(); ++i) {
        const float normalized = fp16(projected[i] * inverse_rms *
                                      read_fp16(norm_bytes, i));
        const float residual = fp16(hidden[i] + normalized);
        result[i] = fp16(residual * read_fp16(scalar_bytes, 0));
    }
    return result;
}

void test_real_complete_path(const std::string& path,
                             const helios::HnfLoader& loader) {
    constexpr uint32_t token_id = 42;
    const auto& main_entry = tensor_named(loader, "text.token_embedding.weight");
    const auto& ple_entry = tensor_named(loader, "text.ple.token_embedding.weight");
    const auto& projection_entry = tensor_named(loader, "text.ple.model_projection.weight");
    const auto& norm_entry = tensor_named(loader, "text.ple.projection_norm.weight");
    const auto& model = loader.config();
    const auto& gemma = loader.gemma4_config();
    const uint32_t hidden = model.hidden_size();
    const uint32_t layers = model.num_hidden_layers();
    const uint32_t ple_hidden = gemma.ple_hidden_size;
    const uint32_t width = layers * ple_hidden;
    require(hidden % 256 == 0 && width % 256 == 0,
            "real PLE dimensions must align to HQ51K blocks");

    const size_t main_row_bytes = size_t(hidden) * sizeof(half);
    const size_t ple_row_bytes = size_t(width / 256) * 200;
    const auto main_row = read_bytes(path,
        main_entry.offset + uint64_t(token_id) * main_row_bytes, main_row_bytes);
    const auto ple_row = read_bytes(path,
        ple_entry.offset + uint64_t(token_id) * ple_row_bytes, ple_row_bytes);
    const auto projection = read_bytes(path, projection_entry.offset,
                                       projection_entry.size);
    const auto norm = read_bytes(path, norm_entry.offset, norm_entry.size);

    helios::Engine engine;
    helios::kernels::register_all_kernels(engine);
    const int32_t compact_token = 0;
    upload(engine.tensors(), "input_tokens", {1, 1}, helios::dtype::INT32(),
           &compact_token, sizeof(compact_token));
    upload(engine.tensors(), "_scratch.hidden", {1, 1, hidden}, helios::dtype::FP16(),
           main_row.data(), main_row.size());
    engine.tensors().allocate_and_register(
        "_scratch.gemma4_ple", {1, 1, width}, helios::dtype::FP16());
    engine.tensors().allocate_and_register(
        "_scratch.gemma4_ple_context", {1, 1, width}, helios::dtype::FP16());
    upload(engine.tensors(), "text.ple.token_embedding.weight", {1, width},
           helios::dtype::HQ51K(), ple_row.data(), ple_row.size());
    upload(engine.tensors(), "text.ple.model_projection.weight", {width, hidden},
           helios::dtype::HQ51K(), projection.data(), projection.size());
    upload(engine.tensors(), "text.ple.projection_norm.weight", {ple_hidden},
           helios::dtype::FP16(), norm.data(), norm.size());

    helios::CommandBuffer commands;
    commands.add_scale("_scratch.hidden", "_scratch.hidden", std::sqrt(float(hidden)));
    helios::append_gemma4_ple_preparation(commands, model, gemma);
    engine.execute(commands);
    engine.sync();

    std::vector<half> gpu_half(width);
    cuda_require(cudaMemcpy(gpu_half.data(), engine.tensors().at("_scratch.gemma4_ple").ptr,
                            gpu_half.size() * sizeof(half), cudaMemcpyDeviceToHost),
                 "copy complete PLE result");
    const auto cpu = cpu_ple_reference(main_row, ple_row, projection, norm,
                                       hidden, layers, ple_hidden,
                                       model.rms_norm_eps());

    float maximum_absolute_error = 0.0f;
    float maximum_relative_error = 0.0f;
    std::vector<float> per_layer_max(layers, 0.0f);
    for (uint32_t d = 0; d < width; ++d) {
        const float actual = __half2float(gpu_half[d]);
        const float expected = cpu[d];
        require(std::isfinite(actual), "complete PLE produced non-finite value");
        const float absolute = std::fabs(actual - expected);
        const float relative = absolute / std::max(0.05f, std::fabs(expected));
        maximum_absolute_error = std::max(maximum_absolute_error, absolute);
        maximum_relative_error = std::max(maximum_relative_error, relative);
        per_layer_max[d / ple_hidden] = std::max(per_layer_max[d / ple_hidden], absolute);
        require(absolute <= 0.08f + 0.03f * std::fabs(expected),
                "complete PLE differs from CPU reference at element " +
                std::to_string(d));
    }
    for (uint32_t layer = 0; layer < layers; ++layer) {
        require(per_layer_max[layer] <= 0.20f,
                "PLE layer segment exceeds fixed tolerance");
    }
    std::cout << "PASS: complete real PLE path for token 42 across " << layers
              << " layers (max_abs=" << maximum_absolute_error
              << ", max_rel=" << maximum_relative_error << ")" << std::endl;

    engine.tensors().allocate_and_register(
        "_scratch.gemma4_ple_gate", {1, 1, ple_hidden}, helios::dtype::FP16());
    engine.tensors().allocate_and_register(
        "_scratch.gemma4_ple_projected", {1, 1, hidden}, helios::dtype::FP16());
    std::vector<half> initial_hidden(hidden);
    std::vector<float> cpu_hidden(hidden);
    for (uint32_t d = 0; d < hidden; ++d) {
        cpu_hidden[d] = fp16(read_fp16(main_row, d) * std::sqrt(float(hidden)));
        initial_hidden[d] = __float2half(cpu_hidden[d]);
    }

    const std::vector<uint32_t> selected_layers{0, 4, 15, 34};
    half* packed_ple = static_cast<half*>(
        engine.tensors().at("_scratch.gemma4_ple").ptr);
    for (uint32_t layer : selected_layers) {
        const std::string prefix = "text.layer" + std::to_string(layer) + '.';
        const auto& gate_entry = tensor_named(loader, prefix + "ple.input_gate.weight");
        const auto& projection_layer_entry = tensor_named(
            loader, prefix + "ple.projection.weight");
        const auto& norm_layer_entry = tensor_named(loader, prefix + "ln_ple_post.weight");
        const auto& scalar_entry = tensor_named(loader, prefix + "layer_scalar");
        const auto gate_weight = read_bytes(path, gate_entry.offset, gate_entry.size);
        const auto projection_weight = read_bytes(
            path, projection_layer_entry.offset, projection_layer_entry.size);
        const auto norm_weight = read_bytes(path, norm_layer_entry.offset,
                                            norm_layer_entry.size);
        const auto scalar_weight = read_bytes(path, scalar_entry.offset,
                                              scalar_entry.size);
        upload(engine.tensors(), gate_entry.name, gate_entry.shape,
               helios::dtype::HQ51K(), gate_weight.data(), gate_weight.size());
        upload(engine.tensors(), projection_layer_entry.name,
               projection_layer_entry.shape, helios::dtype::HQ51K(),
               projection_weight.data(), projection_weight.size());
        upload(engine.tensors(), norm_layer_entry.name, norm_layer_entry.shape,
               helios::dtype::FP16(), norm_weight.data(), norm_weight.size());
        upload(engine.tensors(), scalar_entry.name, scalar_entry.shape,
               helios::dtype::FP16(), scalar_weight.data(), scalar_weight.size());

        const std::string segment_name = "_scratch.gemma4_ple_layer." +
                                         std::to_string(layer);
        engine.tensors().register_external(
            segment_name, packed_ple + size_t(layer) * ple_hidden,
            {1, 1, ple_hidden}, helios::dtype::FP16());
        cuda_require(cudaMemcpy(engine.tensors().at("_scratch.hidden").ptr,
                                initial_hidden.data(), hidden * sizeof(half),
                                cudaMemcpyHostToDevice),
                     "reset hidden before PLE layer injection");

        helios::Gemma4PleLayerNames layer_names;
        layer_names.ple_segment = segment_name;
        helios::CommandBuffer layer_commands;
        helios::append_gemma4_ple_layer_injection(
            layer_commands, model, gemma, layer, layer_names);
        engine.execute(layer_commands);
        engine.sync();

        std::vector<half> gpu_layer(hidden);
        cuda_require(cudaMemcpy(gpu_layer.data(),
                                engine.tensors().at("_scratch.hidden").ptr,
                                hidden * sizeof(half), cudaMemcpyDeviceToHost),
                     "copy PLE layer output");
        const std::vector<float> ple_segment(
            cpu.begin() + size_t(layer) * ple_hidden,
            cpu.begin() + size_t(layer + 1) * ple_hidden);
        const auto cpu_layer = cpu_ple_layer_reference(
            cpu_hidden, ple_segment, gate_weight, projection_weight,
            norm_weight, scalar_weight, model.rms_norm_eps());

        float layer_max_error = 0.0f;
        for (uint32_t d = 0; d < hidden; ++d) {
            const float actual = __half2float(gpu_layer[d]);
            const float absolute = std::fabs(actual - cpu_layer[d]);
            layer_max_error = std::max(layer_max_error, absolute);
            require(std::isfinite(actual), "PLE layer injection produced non-finite value");
            require(absolute <= 0.12f + 0.04f * std::fabs(cpu_layer[d]),
                    "PLE layer injection differs from CPU reference in layer " +
                    std::to_string(layer));
        }
        std::cout << "PASS: real PLE injection layer " << layer
                  << " (max_abs=" << layer_max_error << ")" << std::endl;
    }
}

} // namespace

int main(int argc, char** argv) {
    test_command_contract();
    require(argc <= 2, "Usage: test_gemma4_ple [compact-gemma4.hnf]");
    if (argc == 1) return 0;

    int devices = 0;
    cuda_require(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    require(devices > 0, "CUDA device required for real PLE validation");

    helios::HnfLoader loader;
    require(loader.load_metadata(argv[1]), "cannot read real Gemma 4 metadata");
    require(loader.config().arch() == "gemma4" && loader.has_gemma4_config(),
            "real HNF is not Gemma 4 with GM4X");
    test_real_rows(argv[1], loader);
    test_real_complete_path(argv[1], loader);
    return 0;
}
