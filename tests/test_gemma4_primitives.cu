#include "kernels.hpp"
#include "hnf_loader.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace helios::kernels;

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

std::vector<half> halves(const std::vector<float>& values) {
    std::vector<half> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) result[i] = __float2half(values[i]);
    return result;
}

std::vector<float> run_unary(
    const std::vector<float>& input,
    const std::function<void(const half*, half*, size_t)>& launch) {
    const std::vector<half> host = halves(input);
    half* device_input = nullptr;
    half* device_output = nullptr;
    cuda_require(cudaMalloc(&device_input, host.size() * sizeof(half)), "cudaMalloc input");
    cuda_require(cudaMalloc(&device_output, host.size() * sizeof(half)), "cudaMalloc output");
    cuda_require(cudaMemcpy(device_input, host.data(), host.size() * sizeof(half),
                            cudaMemcpyHostToDevice), "copy input");
    launch(device_input, device_output, host.size());
    cuda_require(cudaDeviceSynchronize(), "kernel synchronize");
    std::vector<half> output(host.size());
    cuda_require(cudaMemcpy(output.data(), device_output, output.size() * sizeof(half),
                            cudaMemcpyDeviceToHost), "copy output");
    cudaFree(device_input);
    cudaFree(device_output);
    std::vector<float> result(output.size());
    for (size_t i = 0; i < output.size(); ++i) result[i] = __half2float(output[i]);
    return result;
}

void write_half(std::vector<uint8_t>& bytes, size_t offset, float value) {
    half encoded = __float2half(value);
    std::memcpy(bytes.data() + offset, &encoded, sizeof(encoded));
}

void encode_test_hq51k_block(std::vector<uint8_t>& block, float min_base,
                             int q_offset) {
    require(block.size() == 200, "HQ51K test block size");
    write_half(block, 0, 31.0f); // d_scale; q_scale=15 yields scoeff=1
    write_half(block, 2, 0.0f);  // d_min
    write_half(block, 4, min_base);
    for (int i = 8; i < 24; ++i) block[i] = 0xff;
    for (int group = 0; group < 32; ++group) {
        uint64_t packed = 0;
        for (int i = 0; i < 8; ++i) {
            uint64_t q = uint64_t((q_offset + group * 8 + i) & 31);
            packed |= q << (i * 5);
        }
        for (int byte = 0; byte < 5; ++byte) {
            block[40 + group * 5 + byte] = uint8_t(packed >> (byte * 8));
        }
    }
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
    uint64_t packed = uint64_t(payload[0]) | (uint64_t(payload[1]) << 8) |
                      (uint64_t(payload[2]) << 16) | (uint64_t(payload[3]) << 24) |
                      (uint64_t(payload[4]) << 32);
    const uint32_t q = uint32_t((packed >> (in_group * 5)) & 31);
    return minimum + float(q) * scale / 31.0f;
}

void encode_test_hq5k_block(std::vector<uint8_t>& block,
                            float base_min, int seed) {
    require(block.size() == 288, "HQ5K test block size");
    for (int group = 0; group < 32; ++group) {
        const half minimum = __float2half(base_min + 0.01f * group);
        const half scale = __float2half(0.25f + 0.005f * group);
        std::memcpy(block.data() + group * 4, &minimum, sizeof(minimum));
        std::memcpy(block.data() + group * 4 + 2, &scale, sizeof(scale));

        uint64_t packed = 0;
        for (int i = 0; i < 8; ++i) {
            const uint32_t q = uint32_t((seed + group + i) & 31);
            packed |= uint64_t(q) << (i * 5);
        }
        uint8_t* payload = block.data() + 128 + group * 5;
        for (int byte = 0; byte < 5; ++byte) {
            payload[byte] = uint8_t((packed >> (byte * 8)) & 0xff);
        }
    }
}

float decode_hq5k_element(const uint8_t* block, int in_block) {
    const int group = in_block / 8;
    const int in_group = in_block % 8;
    half minimum{}, scale{};
    std::memcpy(&minimum, block + group * 4, sizeof(minimum));
    std::memcpy(&scale, block + group * 4 + 2, sizeof(scale));
    const uint8_t* payload = block + 128 + group * 5;
    const uint64_t packed = uint64_t(payload[0]) | (uint64_t(payload[1]) << 8) |
                            (uint64_t(payload[2]) << 16) |
                            (uint64_t(payload[3]) << 24) |
                            (uint64_t(payload[4]) << 32);
    const uint32_t q = uint32_t((packed >> (in_group * 5)) & 31);
    return __half2float(minimum) + float(q) * __half2float(scale) / 31.0f;
}

std::vector<float> run_hq5k_embedding(const std::vector<uint8_t>& table,
                                      const std::vector<int32_t>& ids,
                                      int vocab, int dim) {
    uint8_t* device_table = nullptr;
    int32_t* device_ids = nullptr;
    half* device_output = nullptr;
    cuda_require(cudaMalloc(&device_table, table.size()), "HQ5K table allocation");
    cuda_require(cudaMalloc(&device_ids, ids.size() * sizeof(int32_t)), "HQ5K ids allocation");
    cuda_require(cudaMalloc(&device_output, ids.size() * dim * sizeof(half)),
                 "HQ5K output allocation");
    cudaMemcpy(device_table, table.data(), table.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(device_ids, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    launch_embedding_hq5k(device_ids, device_table, device_output,
                          1, ids.size(), vocab, dim);
    cuda_require(cudaDeviceSynchronize(), "HQ5K embedding synchronize");
    std::vector<half> output(ids.size() * dim);
    cudaMemcpy(output.data(), device_output, output.size() * sizeof(half),
               cudaMemcpyDeviceToHost);
    cudaFree(device_table); cudaFree(device_ids); cudaFree(device_output);
    std::vector<float> result(output.size());
    for (size_t i = 0; i < output.size(); ++i) result[i] = __half2float(output[i]);
    return result;
}

std::vector<float> run_hq51k_embedding(const std::vector<uint8_t>& table,
                                       const std::vector<int32_t>& ids,
                                       int vocab, int dim) {
    uint8_t* device_table = nullptr;
    int32_t* device_ids = nullptr;
    half* device_output = nullptr;
    cuda_require(cudaMalloc(&device_table, table.size()), "HQ51K table allocation");
    cuda_require(cudaMalloc(&device_ids, ids.size() * sizeof(int32_t)), "HQ51K ids allocation");
    cuda_require(cudaMalloc(&device_output, ids.size() * dim * sizeof(half)),
                 "HQ51K output allocation");
    cudaMemcpy(device_table, table.data(), table.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(device_ids, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    launch_embedding_hq51k(device_ids, device_table, device_output,
                           1, ids.size(), vocab, dim);
    cuda_require(cudaDeviceSynchronize(), "HQ51K embedding synchronize");
    std::vector<half> output(ids.size() * dim);
    cudaMemcpy(output.data(), device_output, output.size() * sizeof(half),
               cudaMemcpyDeviceToHost);
    cudaFree(device_table); cudaFree(device_ids); cudaFree(device_output);
    std::vector<float> result(output.size());
    for (size_t i = 0; i < output.size(); ++i) result[i] = __half2float(output[i]);
    return result;
}

void test_gemma_rmsnorm() {
    const std::vector<float> input{1.0f, 2.0f, -3.0f, 4.0f};
    const std::vector<float> weights{1.0f, 1.5f, 0.75f, 2.0f};
    const std::vector<half> x = halves(input);
    const std::vector<half> w = halves(weights);
    half *dx = nullptr, *dw = nullptr, *dy = nullptr;
    cuda_require(cudaMalloc(&dx, x.size() * sizeof(half)), "rms x allocation");
    cuda_require(cudaMalloc(&dw, w.size() * sizeof(half)), "rms w allocation");
    cuda_require(cudaMalloc(&dy, x.size() * sizeof(half)), "rms y allocation");
    cuda_require(cudaMemcpy(dx, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice),
                 "rms x copy");
    cuda_require(cudaMemcpy(dw, w.data(), w.size() * sizeof(half), cudaMemcpyHostToDevice),
                 "rms w copy");
    launch_rmsnorm_fp16(dx, dw, dy, 1, 4, 1.0e-6f);
    cuda_require(cudaDeviceSynchronize(), "Gemma RMSNorm synchronize");
    std::vector<half> output(4);
    cuda_require(cudaMemcpy(output.data(), dy, output.size() * sizeof(half),
                            cudaMemcpyDeviceToHost), "rms output copy");
    const float inv_rms = 1.0f / std::sqrt(7.5f + 1.0e-6f);
    for (size_t i = 0; i < output.size(); ++i) {
        const float expected = input[i] * inv_rms * weights[i];
        require(std::fabs(__half2float(output[i]) - expected) < 0.003f,
                "Gemma 4 direct-weight RMSNorm mismatch");
    }

    launch_rmsnorm_no_weight_fp16(dx, dy, 1, 4, 1.0e-6f);
    cuda_require(cudaDeviceSynchronize(), "weightless RMSNorm synchronize");
    cudaMemcpy(output.data(), dy, output.size() * sizeof(half), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < output.size(); ++i) {
        require(std::fabs(__half2float(output[i]) - input[i] * inv_rms) < 0.003f,
                "Gemma 4 weightless V RMSNorm mismatch");
    }
    cudaFree(dx); cudaFree(dw); cudaFree(dy);
    std::cout << "PASS: Gemma 4 direct-weight and weightless RMSNorm" << std::endl;
}

void test_embedding_scale_and_gelu() {
    const float scale = std::sqrt(1536.0f);
    const auto scaled = run_unary({0.25f, -0.5f, 1.0f},
        [scale](const half* input, half* output, size_t count) {
            launch_scale_fp16(input, output, scale, count);
        });
    require(std::fabs(scaled[0] - 0.25f * scale) < 0.02f, "embedding scale value 0");
    require(std::fabs(scaled[1] + 0.5f * scale) < 0.02f, "embedding scale value 1");

    const std::vector<float> gate{-2.0f, -0.5f, 0.0f, 1.5f};
    const std::vector<float> up{0.5f, 2.0f, -1.0f, 0.25f};
    const std::vector<half> hgate = halves(gate);
    const std::vector<half> hup = halves(up);
    half *dgate = nullptr, *dup = nullptr, *dout = nullptr;
    cuda_require(cudaMalloc(&dgate, hgate.size() * sizeof(half)), "gelu gate allocation");
    cuda_require(cudaMalloc(&dup, hup.size() * sizeof(half)), "gelu up allocation");
    cuda_require(cudaMalloc(&dout, hgate.size() * sizeof(half)), "gelu output allocation");
    cudaMemcpy(dgate, hgate.data(), hgate.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dup, hup.data(), hup.size() * sizeof(half), cudaMemcpyHostToDevice);
    launch_gelu_mul_fp16(dgate, dup, dout, gate.size());
    cuda_require(cudaDeviceSynchronize(), "GeGLU synchronize");
    std::vector<half> gelu_output(gate.size());
    cudaMemcpy(gelu_output.data(), dout, gelu_output.size() * sizeof(half), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < gate.size(); ++i) {
        const float x = gate[i];
        const float gelu = 0.5f * x * (1.0f + std::tanh(0.7978845608f *
                           (x + 0.044715f * x * x * x)));
        require(std::fabs(__half2float(gelu_output[i]) - gelu * up[i]) < 0.003f,
                "gelu_pytorch_tanh GeGLU mismatch");
    }
    cudaFree(dgate); cudaFree(dup); cudaFree(dout);
    std::cout << "PASS: embedding sqrt scale and gelu_pytorch_tanh" << std::endl;
}

void test_layer_scalar_and_softcap() {
    const std::vector<float> input{2.0f, -4.0f, 8.0f};
    const std::vector<half> host = halves(input);
    const half scalar = __float2half(0.25f);
    half *din = nullptr, *dscalar = nullptr, *dout = nullptr;
    cudaMalloc(&din, host.size() * sizeof(half));
    cudaMalloc(&dscalar, sizeof(half));
    cudaMalloc(&dout, host.size() * sizeof(half));
    cudaMemcpy(din, host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dscalar, &scalar, sizeof(half), cudaMemcpyHostToDevice);
    launch_mul_scalar_tensor_fp16(din, dscalar, dout, input.size());
    cuda_require(cudaDeviceSynchronize(), "layer scalar synchronize");
    std::vector<half> scalar_output(input.size());
    cudaMemcpy(scalar_output.data(), dout, scalar_output.size() * sizeof(half),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < input.size(); ++i) {
        require(std::fabs(__half2float(scalar_output[i]) - input[i] * 0.25f) < 0.001f,
                "layer_scalar broadcast mismatch");
    }
    cudaFree(din); cudaFree(dscalar); cudaFree(dout);

    const std::vector<float> logits{-100.0f, -30.0f, 0.0f, 30.0f, 100.0f};
    const auto capped = run_unary(logits,
        [](const half* in, half* out, size_t count) {
            launch_softcap_fp16(in, out, 30.0f, count);
        });
    for (size_t i = 0; i < logits.size(); ++i) {
        const float expected = std::tanh(logits[i] / 30.0f) * 30.0f;
        require(std::fabs(capped[i] - expected) < 0.02f, "logit softcap mismatch");
    }
    std::cout << "PASS: layer scalar and final-logit softcap" << std::endl;
}

void test_proportional_rope() {
    const std::vector<float> input{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<half> host = halves(input);
    half* device = nullptr;
    cudaMalloc(&device, host.size() * sizeof(half));
    cudaMemcpy(device, host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice);
    launch_rope_proportional_inplace_fp16(
        device, 1, 1, 1, 8, 0.5f, 3, 100.0f, 1.0f);
    cuda_require(cudaDeviceSynchronize(), "proportional RoPE synchronize");
    std::vector<half> output(host.size());
    cudaMemcpy(output.data(), device, output.size() * sizeof(half), cudaMemcpyDeviceToHost);
    cudaFree(device);

    std::vector<float> expected = input;
    for (int pair = 0; pair < 2; ++pair) {
        const float frequency = 1.0f / std::pow(100.0f, float(2 * pair) / 8.0f);
        const float angle = 3.0f * frequency;
        const float x0 = input[pair];
        const float x1 = input[pair + 4];
        expected[pair] = x0 * std::cos(angle) - x1 * std::sin(angle);
        expected[pair + 4] = x0 * std::sin(angle) + x1 * std::cos(angle);
    }
    for (size_t i = 0; i < output.size(); ++i) {
        require(std::fabs(__half2float(output[i]) - expected[i]) < 0.006f,
                "proportional RoPE mismatch at dimension " + std::to_string(i));
    }
    std::cout << "PASS: proportional RoPE full-head pairing" << std::endl;
}

void test_hq51k_embedding_lookup() {
    std::vector<uint8_t> table(400, 0);
    std::vector<uint8_t> block0(200, 0), block1(200, 0);
    encode_test_hq51k_block(block0, -2.0f, 0);
    encode_test_hq51k_block(block1, 1.0f, 7);
    std::memcpy(table.data(), block0.data(), 200);
    std::memcpy(table.data() + 200, block1.data(), 200);
    const std::vector<int32_t> ids{1, 0, -1};
    const std::vector<float> output = run_hq51k_embedding(table, ids, 2, 256);
    for (int d = 0; d < 256; ++d) {
        require(std::fabs(output[d] - decode_hq51k_element(block1.data(), d)) < 0.003f,
                "HQ51K row 1 lookup mismatch");
        require(std::fabs(output[256 + d] - decode_hq51k_element(block0.data(), d)) < 0.003f,
                "HQ51K row 0 lookup mismatch");
        require(output[512 + d] == 0.0f, "HQ51K invalid token must produce zeros");
    }
    std::cout << "PASS: synthetic HQ51K embedding row lookup" << std::endl;
}

void test_hq5k_embedding_lookup() {
    std::vector<uint8_t> table(576, 0);
    std::vector<uint8_t> block0(288, 0), block1(288, 0);
    encode_test_hq5k_block(block0, -2.0f, 0);
    encode_test_hq5k_block(block1, 1.0f, 7);
    std::memcpy(table.data(), block0.data(), block0.size());
    std::memcpy(table.data() + block0.size(), block1.data(), block1.size());
    const std::vector<int32_t> ids{1, 0, -1};
    const std::vector<float> output = run_hq5k_embedding(table, ids, 2, 256);
    for (int d = 0; d < 256; ++d) {
        require(std::fabs(output[d] - decode_hq5k_element(block1.data(), d)) < 0.003f,
                "HQ5K row 1 lookup mismatch");
        require(std::fabs(output[256 + d] - decode_hq5k_element(block0.data(), d)) < 0.003f,
                "HQ5K row 0 lookup mismatch");
        require(output[512 + d] == 0.0f, "HQ5K invalid token must produce zeros");
    }
    std::cout << "PASS: synthetic HQ5K embedding row lookup" << std::endl;
}

void test_real_ple_row(const std::string& path) {
    helios::HnfLoader loader;
    require(loader.load_metadata(path), "cannot read real HNF metadata");
    const helios::TensorEntry* ple = nullptr;
    for (const auto& tensor : loader.tensors()) {
        if (tensor.name == "text.ple.token_embedding.weight") ple = &tensor;
    }
    require(ple != nullptr && ple->dtype == "hq51k" && ple->shape.size() == 2,
            "real compact HNF has no HQ51K PLE table");
    const int dim = ple->shape[1];
    require(dim % 256 == 0, "PLE row must align to HQ51K blocks");
    const size_t row_bytes = size_t(dim / 256) * 200;
    constexpr int token_id = 42;
    std::vector<uint8_t> row(row_bytes);
    std::ifstream input(path, std::ios::binary);
    input.seekg(ple->offset + uint64_t(token_id) * row_bytes);
    input.read(reinterpret_cast<char*>(row.data()), row.size());
    require(input.good(), "cannot read selected PLE row");

    const std::vector<float> output = run_hq51k_embedding(row, {0}, 1, dim);
    for (int d = 0; d < dim; ++d) {
        const uint8_t* block = row.data() + size_t(d / 256) * 200;
        const float expected = decode_hq51k_element(block, d % 256);
        require(std::fabs(output[d] - expected) < 0.003f,
                "real PLE row mismatch at dimension " + std::to_string(d));
    }
    std::cout << "PASS: real PLE HQ51K row 42 (" << dim << " values, "
              << row_bytes << " bytes read)" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    cuda_require(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    require(devices > 0, "CUDA device required");
    test_gemma_rmsnorm();
    test_embedding_scale_and_gelu();
    test_layer_scalar_and_softcap();
    test_proportional_rope();
    test_hq5k_embedding_lookup();
    test_hq51k_embedding_lookup();
    if (argc == 2) test_real_ple_row(argv[1]);
    require(argc <= 2, "Usage: test_gemma4_primitives [compact-gemma4.hnf]");
    return 0;
}
