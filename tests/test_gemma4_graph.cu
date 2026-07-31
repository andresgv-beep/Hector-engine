#include "graph_builder.hpp"
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
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

std::vector<half> to_half(const std::vector<float>& values) {
    std::vector<half> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) result[i] = __float2half(values[i]);
    return result;
}

void test_no_weight_rmsnorm_command(helios::Engine& engine) {
    const std::vector<float> values{1, 2, 3, 4, -2, 0.5f, 1.5f, -3};
    const auto input = to_half(values);
    void* source = engine.tensors().allocate_and_register(
        "test.rms.input", {2, 4}, helios::dtype::FP16());
    engine.tensors().allocate_and_register(
        "test.rms.output", {2, 4}, helios::dtype::FP16());
    cuda_require(cudaMemcpy(source, input.data(), input.size() * sizeof(half),
                            cudaMemcpyHostToDevice), "copy RMS input");

    helios::CommandBuffer commands;
    commands.add_rmsnorm_no_weight(
        "test.rms.output", "test.rms.input", 1.0e-6f, 4);
    require(commands[0].inputs.size() == 1 &&
            commands[0].get<bool>("no_weight", false),
            "weightless RMSNorm command contract");
    engine.execute(commands);
    engine.sync();

    std::vector<half> output(values.size());
    cuda_require(cudaMemcpy(output.data(), engine.tensors().at("test.rms.output").ptr,
                            output.size() * sizeof(half), cudaMemcpyDeviceToHost),
                 "copy RMS output");
    for (size_t row = 0; row < 2; ++row) {
        float sum = 0.0f;
        for (size_t d = 0; d < 4; ++d) {
            const float value = values[row * 4 + d];
            sum += value * value;
        }
        const float inverse = 1.0f / std::sqrt(sum / 4.0f + 1.0e-6f);
        for (size_t d = 0; d < 4; ++d) {
            const float expected = values[row * 4 + d] * inverse;
            require(std::fabs(__half2float(output[row * 4 + d]) - expected) < 0.003f,
                    "weightless RMSNorm command numerical mismatch");
        }
    }
    std::cout << "PASS: executable weightless V RMSNorm command" << std::endl;
}

void test_multitoken_ple_slice(helios::Engine& engine) {
    constexpr uint32_t rows = 6;
    constexpr uint32_t layers = 3;
    constexpr uint32_t dim = 4;
    std::vector<float> values(rows * layers * dim);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t layer = 0; layer < layers; ++layer) {
            for (uint32_t d = 0; d < dim; ++d) {
                values[(row * layers + layer) * dim + d] =
                    float(row * 100 + layer * 10 + d);
            }
        }
    }
    const auto packed = to_half(values);
    void* source = engine.tensors().allocate_and_register(
        "test.ple.packed", {2, 3, layers * dim}, helios::dtype::FP16());
    engine.tensors().allocate_and_register(
        "test.ple.segment", {2, 3, dim}, helios::dtype::FP16());
    cuda_require(cudaMemcpy(source, packed.data(), packed.size() * sizeof(half),
                            cudaMemcpyHostToDevice), "copy packed PLE");

    helios::CommandBuffer commands;
    commands.add_ple_slice(
        "test.ple.segment", "test.ple.packed", 1, layers, dim);
    engine.execute(commands);
    engine.sync();
    std::vector<half> output(rows * dim);
    cuda_require(cudaMemcpy(output.data(), engine.tensors().at("test.ple.segment").ptr,
                            output.size() * sizeof(half), cudaMemcpyDeviceToHost),
                 "copy PLE segment");
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t d = 0; d < dim; ++d) {
            require(__half2float(output[row * dim + d]) ==
                    float(row * 100 + 10 + d),
                    "multi-token PLE slice selected the wrong layer");
        }
    }
    std::cout << "PASS: token-major PLE slice for batch=2, sequence=3" << std::endl;
}

helios::ModelConfig synthetic_model() {
    helios::ModelConfig config;
    config.set("arch", std::string("gemma4"));
    config.set("num_hidden_layers", int64_t(35));
    config.set("hidden_size", int64_t(1536));
    config.set("intermediate_size", int64_t(6144));
    config.set("vocab_size", int64_t(262144));
    config.set("num_attention_heads", int64_t(8));
    config.set("num_key_value_heads", int64_t(1));
    config.set("rms_norm_eps", 1.0e-6);
    config.set("hidden_act", std::string("gelu_pytorch_tanh"));
    return config;
}

helios::Gemma4Config synthetic_gemma() {
    helios::Gemma4Config gemma;
    gemma.version = 1;
    gemma.flags = helios::GEMMA4_EXT_FLAG_PLE |
                  helios::GEMMA4_EXT_FLAG_LAYER_SCALAR;
    gemma.ple_hidden_size = 256;
    gemma.num_kv_shared_layers = 20;
    gemma.layers.resize(35);
    for (uint32_t i = 0; i < gemma.layers.size(); ++i) {
        auto& layer = gemma.layers[i];
        layer.attention_kind = (i % 5 == 4) ? 1 : 0;
        layer.sliding_window = layer.attention_kind == 0 ? 512 : 0;
        layer.head_dim = layer.attention_kind == 1 ? 512 : 256;
        layer.intermediate_size = i >= 15 ? 12288 : 6144;
        layer.rope_type = helios::ROPE_PROPORTIONAL;
        layer.rope_theta = 1000000.0f;
        layer.partial_rotary_factor = layer.attention_kind == 1 ? 0.25f : 0.5f;
    }
    return gemma;
}

std::vector<std::string> register_detection_contract(
    helios::Engine& engine, void* dummy) {
    std::vector<std::string> names;
    auto reg = [&](const std::string& name) {
        engine.tensors().register_external(name, dummy, {1}, helios::dtype::FP16());
        names.push_back(name);
    };
    for (uint32_t i = 0; i < 35; ++i) {
        reg("text.layer" + std::to_string(i) + ".ln_attn_in.weight");
    }
    reg("text.layer0.ln_attn_post.weight");
    reg("text.layer0.attn.q_norm.weight");
    reg("text.layer0.mlp.gate.weight");
    reg("text.token_embedding.weight");
    reg("text.final_norm.weight");
    reg("text.lm_head.weight");
    return names;
}

const helios::Command& command_with_op(const helios::CommandBuffer& commands,
                                       helios::OpTypeID op, size_t occurrence = 0) {
    for (const auto& command : commands.commands()) {
        if (command.op == op) {
            if (occurrence == 0) return command;
            --occurrence;
        }
    }
    std::cerr << "FAIL: required command not found: " << helios::op_name(op) << std::endl;
    std::exit(1);
}

void test_heterogeneous_graph_contract(helios::Engine& engine) {
    void* dummy = nullptr;
    cuda_require(cudaMalloc(&dummy, 1), "allocate detection dummy");
    const auto detection_names = register_detection_contract(engine, dummy);
    const auto config = synthetic_model();
    const auto gemma = synthetic_gemma();

    helios::GraphBuilder builder;
    const auto arch = builder.detect_architecture(engine, "text", config);
    require(arch.num_layers == 35, "Gemma 4 layer detection");
    require(arch.post_attn_norm == "ln_attn_post",
            "Gemma 4 canonical post-attention norm detection");
    require(arch.activation == helios::ActivationType::GELU_NEW,
            "gelu_pytorch_tanh detection");
    builder.allocate_gemma4_scratch(engine, config, gemma, arch, 2, 3);

    require(engine.tensors().at("_s.g4.layer0.q").shape ==
            std::vector<uint32_t>({2, 3, 2048}), "local Q scratch shape");
    require(engine.tensors().at("_s.g4.layer4.q").shape ==
            std::vector<uint32_t>({2, 3, 4096}), "global Q scratch shape");
    require(engine.tensors().at("_s.g4.layer0.gate").shape ==
            std::vector<uint32_t>({2, 3, 6144}), "base MLP scratch shape");
    require(engine.tensors().at("_s.g4.layer15.gate").shape ==
            std::vector<uint32_t>({2, 3, 12288}), "wide MLP scratch shape");

    std::vector<int32_t> tokens(6, 0);
    void* token_ptr = engine.tensors().allocate_and_register(
        "graph.input_tokens", {2, 3}, helios::dtype::INT32());
    cuda_require(cudaMemcpy(token_ptr, tokens.data(), tokens.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice), "copy graph tokens");
    const auto input_commands = builder.build_gemma4_input(
        engine, config, gemma, arch, "graph.input_tokens", 2, 3);
    require(input_commands.size() == 9, "Gemma 4 input graph command count");
    require(input_commands[0].op == helios::op::EMBEDDING() &&
            input_commands[1].op == helios::op::SCALE(),
            "Gemma 4 main embedding order");

    const auto layer = builder.build_gemma4_single_layer(
        engine, config, gemma, arch, 0, 2, 3, 7);
    require(layer.size() == 29, "Gemma 4 explicit layer command count");
    const auto& v_norm = command_with_op(layer, helios::op::RMSNORM(), 3);
    require(v_norm.inputs.size() == 1 && v_norm.get<bool>("no_weight", false) &&
            v_norm.get<uint32_t>("dim", 0) == 256,
            "Gemma 4 V must use weightless per-head RMSNorm");
    const auto& attention = command_with_op(layer, helios::op::ATTENTION());
    require(attention.get<float>("scale", 0.0f) == 1.0f &&
            attention.get<uint32_t>("head_dim", 0) == 256,
            "Gemma 4 local attention scale and head_dim");
    const auto& first_rope = command_with_op(layer, helios::op::ROPE());
    require(first_rope.get<uint32_t>("proportional", 0) == 1 &&
            first_rope.get<uint32_t>("offset", 0) == 7,
            "Gemma 4 proportional RoPE command");
    const auto& ple_slice = command_with_op(layer, helios::op::PLE_SLICE());
    require(ple_slice.get<uint32_t>("layer", 99) == 0 &&
            ple_slice.get<uint32_t>("layers", 0) == 35 &&
            ple_slice.get<uint32_t>("dim", 0) == 256,
            "Gemma 4 packed PLE selection command");
    require(layer[layer.size() - 1].op == helios::op::MUL_SCALAR_TENSOR(),
            "Gemma 4 layer must finish with layer_scalar");

    bool rejected_shared = false;
    try {
        (void)builder.build_gemma4_single_layer(
            engine, config, gemma, arch, 15, 1, 1, 0);
    } catch (const std::invalid_argument&) {
        rejected_shared = true;
    }
    require(rejected_shared, "Phase 5 must reject shared-KV layers explicitly");
    builder.free_scratch(engine);
    for (const std::string& name : detection_names) engine.tensors().remove(name);
    cudaFree(dummy);
    std::cout << "PASS: heterogeneous Gemma 4 scratch and first local layer graph"
              << std::endl;
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

float fp16(float value) {
    return __half2float(__float2half(value));
}

float read_fp16(const std::vector<uint8_t>& bytes, size_t index) {
    require((index + 1) * sizeof(half) <= bytes.size(), "FP16 read outside tensor");
    half value{};
    std::memcpy(&value, bytes.data() + index * sizeof(half), sizeof(value));
    return __half2float(value);
}

float decode_hq51k(const uint8_t* block, uint32_t index) {
    half d_scale_h{}, d_min_h{}, min_base_h{};
    std::memcpy(&d_scale_h, block, 2);
    std::memcpy(&d_min_h, block + 2, 2);
    std::memcpy(&min_base_h, block + 4, 2);
    const uint32_t group = index / 8;
    const uint32_t in_group = index % 8;
    const uint8_t scale_packed = block[8 + group / 2];
    const uint8_t minimum_packed = block[24 + group / 2];
    const uint8_t q_scale = group % 2 == 0 ? scale_packed >> 4 : scale_packed & 15;
    const uint8_t q_min = group % 2 == 0 ? minimum_packed >> 4 : minimum_packed & 15;
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

float decode_hq41k(const uint8_t* block, uint32_t index) {
    half d_scale_h{}, d_min_h{}, min_base_h{};
    std::memcpy(&d_scale_h, block, 2);
    std::memcpy(&d_min_h, block + 2, 2);
    std::memcpy(&min_base_h, block + 4, 2);
    const uint32_t group = index / 8;
    const uint32_t in_group = index % 8;
    const uint8_t scale_packed = block[8 + group / 2];
    const uint8_t minimum_packed = block[24 + group / 2];
    const uint8_t q_scale = group % 2 == 0 ? scale_packed >> 4 : scale_packed & 15;
    const uint8_t q_min = group % 2 == 0 ? minimum_packed >> 4 : minimum_packed & 15;
    const float coefficient = __half2float(d_scale_h) * float(q_scale) /
                              (15.0f * 15.0f);
    const float minimum = __half2float(min_base_h) +
                          __half2float(d_min_h) * float(q_min) / 15.0f;
    const uint8_t packed = block[40 + group * 4 + in_group / 2];
    const uint8_t q = in_group % 2 == 0 ? packed >> 4 : packed & 15;
    return minimum + float(q) * coefficient;
}

std::vector<float> cpu_quantized_matmul(
    const std::vector<float>& input,
    const std::vector<uint8_t>& weights,
    uint32_t output_width,
    const std::string& dtype) {
    const uint32_t input_width = static_cast<uint32_t>(input.size());
    const size_t block_bytes = dtype == "hq51k" ? 200 : 168;
    const size_t blocks_per_row = (input_width + 255) / 256;
    require(weights.size() == size_t(output_width) * blocks_per_row * block_bytes,
            "CPU quantized matmul storage mismatch");
    std::vector<float> output(output_width);
    for (uint32_t n = 0; n < output_width; ++n) {
        float sum = 0.0f;
        const uint8_t* row = weights.data() + size_t(n) * blocks_per_row * block_bytes;
        for (uint32_t k = 0; k < input_width; ++k) {
            const uint8_t* block = row + size_t(k / 256) * block_bytes;
            const float weight = dtype == "hq51k"
                ? decode_hq51k(block, k % 256)
                : decode_hq41k(block, k % 256);
            sum = std::fma(input[k], weight, sum);
        }
        output[n] = fp16(sum);
    }
    return output;
}

std::vector<float> cpu_rmsnorm(const std::vector<float>& input,
                               const std::vector<uint8_t>* weight,
                               float eps) {
    float sum = 0.0f;
    for (float value : input) sum += value * value;
    const float inverse = 1.0f / std::sqrt(sum / float(input.size()) + eps);
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const float scale = weight ? read_fp16(*weight, i) : 1.0f;
        output[i] = fp16(input[i] * inverse * scale);
    }
    return output;
}

std::vector<uint8_t> real_tensor_bytes(const helios::HnfLoader& loader,
                                       const std::string& path,
                                       const std::string& name) {
    const auto& entry = tensor_named(loader, name);
    return read_bytes(path, entry.offset, entry.size);
}

std::vector<float> cpu_real_mlp_ple_tail(
    const helios::HnfLoader& loader,
    const std::string& path,
    uint32_t layer_index,
    std::vector<float> hidden,
    const std::vector<float>& ple_segment) {
    const float eps = loader.config().rms_norm_eps();
    const uint32_t hidden_size = loader.config().hidden_size();
    const uint32_t intermediate =
        loader.gemma4_config().layers.at(layer_index).intermediate_size;
    const std::string prefix = "text.layer" + std::to_string(layer_index) + '.';
    auto bytes = [&](const std::string& suffix) {
        return real_tensor_bytes(loader, path, prefix + suffix);
    };

    const auto ln_mlp_in = bytes("ln_mlp_in.weight");
    const auto normed_mlp = cpu_rmsnorm(hidden, &ln_mlp_in, eps);
    const auto gate_weight = bytes("mlp.gate.weight");
    const auto up_weight = bytes("mlp.up.weight");
    std::vector<float> gate = cpu_quantized_matmul(
        normed_mlp, gate_weight, intermediate, "hq41k");
    const std::vector<float> up = cpu_quantized_matmul(
        normed_mlp, up_weight, intermediate, "hq41k");
    for (size_t i = 0; i < gate.size(); ++i) {
        const float x = gate[i];
        gate[i] = fp16(fp16(0.5f * x * (1.0f + std::tanh(
            0.7978845608f * (x + 0.044715f * x * x * x)))) * up[i]);
    }
    const auto down_weight = bytes("mlp.down.weight");
    std::vector<float> mlp = cpu_quantized_matmul(
        gate, down_weight, hidden_size, "hq41k");
    const auto ln_mlp_post = bytes("ln_mlp_post.weight");
    mlp = cpu_rmsnorm(mlp, &ln_mlp_post, eps);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = fp16(hidden[i] + mlp[i]);

    const auto ple_gate_weight = bytes("ple.input_gate.weight");
    std::vector<float> ple_gate = cpu_quantized_matmul(
        hidden, ple_gate_weight, 256, "hq51k");
    for (size_t i = 0; i < ple_gate.size(); ++i) {
        const float x = ple_gate[i];
        ple_gate[i] = fp16(fp16(0.5f * x * (1.0f + std::tanh(
            0.7978845608f * (x + 0.044715f * x * x * x)))) * ple_segment[i]);
    }
    const auto ple_projection_weight = bytes("ple.projection.weight");
    std::vector<float> ple_projection = cpu_quantized_matmul(
        ple_gate, ple_projection_weight, hidden_size, "hq51k");
    const auto ln_ple_post = bytes("ln_ple_post.weight");
    ple_projection = cpu_rmsnorm(ple_projection, &ln_ple_post, eps);
    const auto scalar = bytes("layer_scalar");
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = fp16(fp16(hidden[i] + ple_projection[i]) * read_fp16(scalar, 0));
    }
    return hidden;
}

std::vector<float> cpu_real_layer_reference(
    const helios::HnfLoader& loader,
    const std::string& path,
    uint32_t layer_index,
    std::vector<float> hidden,
    const std::vector<float>& ple_segment) {
    const float eps = loader.config().rms_norm_eps();
    const auto& layer_config = loader.gemma4_config().layers.at(layer_index);
    const uint32_t hidden_size = loader.config().hidden_size();
    const uint32_t heads = loader.config().num_attention_heads();
    const uint32_t head_dim = layer_config.head_dim;
    const std::string prefix = "text.layer" + std::to_string(layer_index) + '.';
    auto bytes = [&](const std::string& suffix) {
        return real_tensor_bytes(loader, path, prefix + suffix);
    };

    const auto ln_attn_in = bytes("ln_attn_in.weight");
    const auto normed_attention = cpu_rmsnorm(hidden, &ln_attn_in, eps);
    const auto v_weight = bytes("attn.v_proj.weight");
    std::vector<float> value = cpu_quantized_matmul(
        normed_attention, v_weight, head_dim, "hq51k");
    value = cpu_rmsnorm(value, nullptr, eps);

    std::vector<float> attention_output(heads * head_dim);
    for (uint32_t head = 0; head < heads; ++head) {
        std::copy(value.begin(), value.end(),
                  attention_output.begin() + size_t(head) * head_dim);
    }
    const auto o_weight = bytes("attn.o_proj.weight");
    std::vector<float> attention_projection = cpu_quantized_matmul(
        attention_output, o_weight, hidden_size, "hq51k");
    const auto ln_attn_post = bytes("ln_attn_post.weight");
    attention_projection = cpu_rmsnorm(attention_projection, &ln_attn_post, eps);
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = fp16(hidden[i] + attention_projection[i]);
    }

    return cpu_real_mlp_ple_tail(
        loader, path, layer_index, std::move(hidden), ple_segment);
}

helios::DTypeID dtype_id(const std::string& dtype) {
    if (dtype == "fp16") return helios::dtype::FP16();
    if (dtype == "hq41k") return helios::dtype::HQ41K();
    if (dtype == "hq51k") return helios::dtype::HQ51K();
    throw std::runtime_error("unsupported test dtype: " + dtype);
}

void upload_bytes(helios::Engine& engine, const std::string& name,
                  const std::vector<uint32_t>& shape, helios::DTypeID dtype,
                  const std::vector<uint8_t>& bytes) {
    void* destination = engine.tensors().allocate_and_register(name, shape, dtype);
    require(engine.tensors().at(name).size_bytes == bytes.size(),
            "real tensor storage mismatch: " + name);
    cuda_require(cudaMemcpy(destination, bytes.data(), bytes.size(),
                            cudaMemcpyHostToDevice), "upload " + name);
}

void upload_tensor(helios::Engine& engine, const helios::HnfLoader& loader,
                   const std::string& path, const std::string& name) {
    const auto& entry = tensor_named(loader, name);
    upload_bytes(engine, entry.name, entry.shape, dtype_id(entry.dtype),
                 read_bytes(path, entry.offset, entry.size));
}

void execute_checked_fp16(helios::Engine& engine,
                          const helios::CommandBuffer& commands,
                          const std::string& label) {
    for (size_t index = 0; index < commands.size(); ++index) {
        const auto& command = commands[index];
        engine.execute_command(command);
        engine.sync();
        const helios::TensorInfo* output = engine.tensors().get(command.output);
        if (!output || output->dtype != helios::dtype::FP16()) continue;
        std::vector<half> values(output->numel());
        cuda_require(cudaMemcpy(values.data(), output->ptr,
                                values.size() * sizeof(half), cudaMemcpyDeviceToHost),
                     "inspect " + label + " command output");
        for (half encoded : values) {
            require(std::isfinite(__half2float(encoded)),
                    label + " first produced a non-finite value at command " +
                    std::to_string(index) + " (" + helios::op_name(command.op) +
                    " -> " + command.output + ")");
        }
    }
}

void test_real_first_local_layer(helios::Engine& engine,
                                 const std::string& path) {
    constexpr uint32_t token_id = 42;
    helios::HnfLoader loader;
    require(loader.load_metadata(path), "cannot load real Gemma 4 metadata");
    require(loader.config().arch() == "gemma4" && loader.has_gemma4_config(),
            "real graph test requires Gemma 4 GM4X");
    const auto& config = loader.config();
    const auto& gemma = loader.gemma4_config();
    const uint32_t hidden = config.hidden_size();
    const uint32_t ple_width = config.num_hidden_layers() * gemma.ple_hidden_size;

    const auto& main = tensor_named(loader, "text.token_embedding.weight");
    const auto& ple = tensor_named(loader, "text.ple.token_embedding.weight");
    const size_t main_row_bytes = size_t(hidden) * sizeof(half);
    const size_t ple_row_bytes = size_t(ple_width / 256) * 200;
    upload_bytes(engine, main.name, {1, hidden}, helios::dtype::FP16(),
                 read_bytes(path, main.offset + uint64_t(token_id) * main_row_bytes,
                            main_row_bytes));
    upload_bytes(engine, ple.name, {1, ple_width}, helios::dtype::HQ51K(),
                 read_bytes(path, ple.offset + uint64_t(token_id) * ple_row_bytes,
                            ple_row_bytes));
    upload_tensor(engine, loader, path, "text.ple.model_projection.weight");
    upload_tensor(engine, loader, path, "text.ple.projection_norm.weight");

    const std::vector<std::string> layer_weights{
        "attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight",
        "attn.o_proj.weight", "attn.q_norm.weight", "attn.k_norm.weight",
        "mlp.gate.weight", "mlp.up.weight", "mlp.down.weight",
        "ln_attn_in.weight", "ln_attn_post.weight", "ln_mlp_in.weight",
        "ln_mlp_post.weight", "ln_ple_post.weight", "ple.input_gate.weight",
        "ple.projection.weight", "layer_scalar"
    };
    for (uint32_t layer = 0; layer <= 4; ++layer) {
        for (const std::string& suffix : layer_weights) {
            upload_tensor(engine, loader, path,
                          "text.layer" + std::to_string(layer) + '.' + suffix);
        }
    }
    const std::vector<std::string> tail_weights{
        "mlp.gate.weight", "mlp.up.weight", "mlp.down.weight",
        "ln_mlp_in.weight", "ln_mlp_post.weight", "ln_ple_post.weight",
        "ple.input_gate.weight", "ple.projection.weight", "layer_scalar"
    };
    for (const std::string& suffix : tail_weights) {
        upload_tensor(engine, loader, path, "text.layer15." + suffix);
    }

    helios::ArchDescriptor arch;
    arch.prefix = "text";
    arch.num_layers = config.num_hidden_layers();
    arch.use_qk_norm = true;
    arch.has_gate = true;
    arch.mlp_gate_name = "gate";
    arch.mlp_up_name = "up";
    arch.mlp_down_name = "down";
    arch.pre_attn_norm = "ln_attn_in";
    arch.post_attn_norm = "ln_attn_post";
    arch.embedding_name = "token_embedding.weight";
    arch.final_norm_name = "final_norm.weight";
    arch.lm_head_name = "lm_head.weight";
    arch.activation = helios::ActivationType::GELU_NEW;
    arch.compute_dtype = helios::dtype::FP16();

    const int32_t compact_token = 0;
    void* token_ptr = engine.tensors().allocate_and_register(
        "real.input_tokens", {1, 1}, helios::dtype::INT32());
    cuda_require(cudaMemcpy(token_ptr, &compact_token, sizeof(compact_token),
                            cudaMemcpyHostToDevice), "copy real compact token");

    helios::GraphBuilder builder;
    builder.allocate_gemma4_scratch(engine, config, gemma, arch, 1, 1);
    const auto input_commands = builder.build_gemma4_input(
        engine, config, gemma, arch, "real.input_tokens", 1, 1);
    engine.execute(input_commands);
    engine.sync();

    std::vector<half> initial_hidden_half(hidden);
    std::vector<half> packed_ple(ple_width);
    cuda_require(cudaMemcpy(initial_hidden_half.data(),
                            engine.tensors().at("_s.hidden").ptr,
                            initial_hidden_half.size() * sizeof(half),
                            cudaMemcpyDeviceToHost),
                 "copy real scaled embedding");
    cuda_require(cudaMemcpy(packed_ple.data(), engine.tensors().at("_s.g4.ple").ptr,
                            packed_ple.size() * sizeof(half), cudaMemcpyDeviceToHost),
                 "copy packed PLE");
    std::vector<float> cpu_initial_hidden(hidden);
    for (size_t i = 0; i < hidden; ++i) {
        cpu_initial_hidden[i] = __half2float(initial_hidden_half[i]);
    }
    auto cpu_segment = [&](uint32_t layer) {
        std::vector<float> result(gemma.ple_hidden_size);
        const size_t offset = size_t(layer) * gemma.ple_hidden_size;
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = __half2float(packed_ple[offset + i]);
        }
        return result;
    };

    const auto layer_commands = builder.build_gemma4_single_layer(
        engine, config, gemma, arch, 0, 1, 1, 0);
    for (size_t i = 0; i < layer_commands.size(); ++i) {
        try {
            engine.execute_command(layer_commands[i]);
            engine.sync();
        } catch (const std::exception& error) {
            throw std::runtime_error(
                "real layer command " + std::to_string(i) + " (" +
                helios::op_name(layer_commands[i].op) + ") failed: " + error.what());
        }
    }

    std::vector<half> output(hidden);
    cuda_require(cudaMemcpy(output.data(), engine.tensors().at("_s.hidden").ptr,
                            output.size() * sizeof(half), cudaMemcpyDeviceToHost),
                 "copy real first-layer output");
    float maximum = 0.0f;
    float mean = 0.0f;
    for (half encoded : output) {
        const float value = __half2float(encoded);
        require(std::isfinite(value), "real first layer produced non-finite output");
        maximum = std::max(maximum, std::fabs(value));
        mean += value;
    }
    mean /= float(output.size());
    require(maximum > 0.0f && maximum < 10000.0f,
            "real first-layer output magnitude is implausible");

    const auto cpu_output = cpu_real_layer_reference(
        loader, path, 0, cpu_initial_hidden, cpu_segment(0));
    float maximum_error = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        const float actual = __half2float(output[i]);
        const float absolute = std::fabs(actual - cpu_output[i]);
        maximum_error = std::max(maximum_error, absolute);
        require(absolute <= 0.20f + 0.04f * std::fabs(cpu_output[i]),
                "real Gemma 4 layer 0 differs from CPU reference at element " +
                std::to_string(i));
    }

    std::vector<half> selected_segment(gemma.ple_hidden_size);
    cuda_require(cudaMemcpy(selected_segment.data(),
                            engine.tensors().at("_s.g4.ple_segment").ptr,
                            selected_segment.size() * sizeof(half), cudaMemcpyDeviceToHost),
                 "copy selected layer-0 PLE");
    require(std::memcmp(packed_ple.data(), selected_segment.data(),
                        selected_segment.size() * sizeof(half)) == 0,
            "real layer graph consumed the wrong PLE segment");
    std::cout << "PASS: real compact token 42 through Gemma 4 layer 0"
              << " (mean=" << mean << ", max_abs=" << maximum
              << ", max_cpu_error=" << maximum_error << ")" << std::endl;

    const auto layer1_commands = builder.build_gemma4_single_layer(
        engine, config, gemma, arch, 1, 1, 1, 0);
    engine.execute(layer1_commands);
    engine.sync();
    std::vector<half> output1(hidden);
    cuda_require(cudaMemcpy(output1.data(), engine.tensors().at("_s.hidden").ptr,
                            output1.size() * sizeof(half), cudaMemcpyDeviceToHost),
                 "copy real second-layer output");
    const auto cpu_output1 = cpu_real_layer_reference(
        loader, path, 1, cpu_output, cpu_segment(1));
    float maximum_error1 = 0.0f;
    float maximum1 = 0.0f;
    for (size_t i = 0; i < output1.size(); ++i) {
        const float actual = __half2float(output1[i]);
        require(std::isfinite(actual), "real second layer produced non-finite output");
        const float absolute = std::fabs(actual - cpu_output1[i]);
        maximum_error1 = std::max(maximum_error1, absolute);
        maximum1 = std::max(maximum1, std::fabs(actual));
        require(absolute <= 0.30f + 0.05f * std::fabs(cpu_output1[i]),
                "real Gemma 4 layer 1 differs from CPU reference at element " +
                std::to_string(i));
    }
    cuda_require(cudaMemcpy(selected_segment.data(),
                            engine.tensors().at("_s.g4.ple_segment").ptr,
                            selected_segment.size() * sizeof(half), cudaMemcpyDeviceToHost),
                 "copy selected layer-1 PLE");
    require(std::memcmp(packed_ple.data() + gemma.ple_hidden_size,
                        selected_segment.data(),
                        selected_segment.size() * sizeof(half)) == 0,
            "second layer graph consumed the wrong PLE segment");
    std::cout << "PASS: real compact token 42 through Gemma 4 layer 1"
              << " (max_abs=" << maximum1
              << ", max_cpu_error=" << maximum_error1 << ")" << std::endl;

    std::vector<float> cpu_chain = cpu_output1;
    for (uint32_t layer_index = 2; layer_index <= 4; ++layer_index) {
        const auto commands = builder.build_gemma4_single_layer(
            engine, config, gemma, arch, layer_index, 1, 1, 0);
        const auto& attention = command_with_op(commands, helios::op::ATTENTION());
        const uint32_t expected_head_dim = layer_index == 4 ? 512 : 256;
        require(attention.get<uint32_t>("head_dim", 0) == expected_head_dim,
                "real layer uses the wrong GM4X head_dim");
        execute_checked_fp16(
            engine, commands, "real layer " + std::to_string(layer_index));

        std::vector<half> gpu(hidden);
        cuda_require(cudaMemcpy(gpu.data(), engine.tensors().at("_s.hidden").ptr,
                                gpu.size() * sizeof(half), cudaMemcpyDeviceToHost),
                     "copy real chained layer output");
        cpu_chain = cpu_real_layer_reference(
            loader, path, layer_index, cpu_chain, cpu_segment(layer_index));
        float maximum_error_chain = 0.0f;
        float maximum_chain = 0.0f;
        for (size_t i = 0; i < gpu.size(); ++i) {
            const float actual = __half2float(gpu[i]);
            const float absolute = std::fabs(actual - cpu_chain[i]);
            maximum_error_chain = std::max(maximum_error_chain, absolute);
            maximum_chain = std::max(maximum_chain, std::fabs(actual));
            require(absolute <= 0.40f + 0.06f * std::fabs(cpu_chain[i]),
                    "real chained Gemma 4 layer differs from CPU reference at layer " +
                    std::to_string(layer_index) + ", element " + std::to_string(i));
        }
        std::cout << "PASS: real chained Gemma 4 layer " << layer_index
                  << " head_dim=" << expected_head_dim
                  << " (max_abs=" << maximum_chain
                  << ", max_cpu_error=" << maximum_error_chain << ")" << std::endl;
    }

    require(engine.tensors().at("_s.g4.layer15.gate").shape.back() == 12288,
            "real layer 15 must expose double-wide MLP scratch");
    const auto wide_tail = builder.build_gemma4_mlp_ple_tail(
        engine, config, gemma, arch, 15, 1, 1);
    execute_checked_fp16(engine, wide_tail, "real double-wide layer 15 tail");
    std::vector<half> wide_gpu(hidden);
    cuda_require(cudaMemcpy(wide_gpu.data(), engine.tensors().at("_s.hidden").ptr,
                            wide_gpu.size() * sizeof(half), cudaMemcpyDeviceToHost),
                 "copy double-wide MLP/PLE output");
    const auto wide_cpu = cpu_real_mlp_ple_tail(
        loader, path, 15, cpu_chain, cpu_segment(15));
    float maximum_wide_error = 0.0f;
    for (size_t i = 0; i < wide_gpu.size(); ++i) {
        const float actual = __half2float(wide_gpu[i]);
        const float absolute = std::fabs(actual - wide_cpu[i]);
        maximum_wide_error = std::max(maximum_wide_error, absolute);
        require(std::isfinite(actual), "double-wide MLP/PLE produced non-finite output");
        require(absolute <= 0.40f + 0.06f * std::fabs(wide_cpu[i]),
                "double-wide Gemma 4 MLP/PLE differs from CPU reference at element " +
                std::to_string(i));
    }
    std::cout << "PASS: real layer 15 double-wide MLP/PLE tail"
              << " (intermediate=12288, max_cpu_error="
              << maximum_wide_error << ")" << std::endl;
    builder.free_scratch(engine);
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    cuda_require(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    require(devices > 0, "CUDA device required");
    helios::Engine engine;
    helios::kernels::register_all_kernels(engine);
    test_no_weight_rmsnorm_command(engine);
    test_multitoken_ple_slice(engine);
    test_heterogeneous_graph_contract(engine);
    require(argc <= 2, "Usage: test_gemma4_graph [compact-gemma4.hnf]");
    if (argc == 2) test_real_first_local_layer(engine, argv[1]);
    return 0;
}
