#include "dtype.hpp"
#include "kernels.hpp"
#include "hqs_common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_require(cudaError_t status, const char* message) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
}

void write_half(uint8_t* dst, float value) {
    uint16_t bits = __half_as_ushort(__float2half(value));
    dst[0] = uint8_t(bits & 0xFF);
    dst[1] = uint8_t(bits >> 8);
}

std::vector<uint8_t> make_weights(int rows) {
    using namespace helios::hqs;
    std::vector<uint8_t> weights(size_t(rows) * HQ31K_BLOCK_SIZE, 0);
    for (int row = 0; row < rows; row++) {
        uint8_t* block = weights.data() + size_t(row) * HQ31K_BLOCK_SIZE;
        write_half(block + 0, 1.0f);   // d_scale
        write_half(block + 2, 0.0f);   // d_min
        write_half(block + 4, -0.5f);  // min_base
        for (int i = 0; i < 16; i++) block[8 + i] = 0xFF; // scale=1
        for (int group = 0; group < 32; group++) {
            uint32_t packed = 0;
            for (int i = 0; i < 8; i++) {
                uint32_t q = uint32_t((row + group + i) & 7);
                packed |= q << (i * 3);
            }
            int offset = COMPACT_HEADER_SIZE + group * 3;
            block[offset] = uint8_t(packed & 0xFF);
            block[offset + 1] = uint8_t((packed >> 8) & 0xFF);
            block[offset + 2] = uint8_t((packed >> 16) & 0xFF);
        }
    }
    return weights;
}

float cpu_weight(const uint8_t* block, int k) {
    using namespace helios::hqs;
    int group = k / 8;
    int index = k % 8;
    int offset = COMPACT_HEADER_SIZE + group * 3;
    uint32_t packed = uint32_t(block[offset])
                    | (uint32_t(block[offset + 1]) << 8)
                    | (uint32_t(block[offset + 2]) << 16);
    float q = float((packed >> (index * 3)) & 7);
    return -0.5f + q / 7.0f;
}

void test_matmul(int M) {
    using namespace helios;
    using namespace helios::hqs;
    constexpr int K = 256;
    constexpr int N = 19;
    std::vector<half> input(size_t(M) * K);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            input[size_t(m) * K + k] = __float2half(
                std::sin(float(k + m * 7) * 0.071f) * 0.5f);
    std::vector<uint8_t> weights = make_weights(N);
    std::vector<half> output(size_t(M) * N);

    half* d_input = nullptr;
    uint8_t* d_weights = nullptr;
    half* d_output = nullptr;
    cuda_require(cudaMalloc(&d_input, input.size() * sizeof(half)), "input allocation");
    cuda_require(cudaMalloc(&d_weights, weights.size()), "weight allocation");
    cuda_require(cudaMalloc(&d_output, output.size() * sizeof(half)), "output allocation");
    cuda_require(cudaMemcpy(d_input, input.data(), input.size() * sizeof(half),
                            cudaMemcpyHostToDevice), "copy input");
    cuda_require(cudaMemcpy(d_weights, weights.data(), weights.size(),
                            cudaMemcpyHostToDevice), "copy weights");

    kernels::launch_matmul_hq31k(d_input, d_weights, d_output, M, K, N);
    cuda_require(cudaDeviceSynchronize(), "HQ3.1K matmul");
    cuda_require(cudaMemcpy(output.data(), d_output, output.size() * sizeof(half),
                            cudaMemcpyDeviceToHost), "copy output");

    for (int m = 0; m < M; m++) {
        for (int row = 0; row < N; row++) {
            const uint8_t* block = weights.data() + size_t(row) * HQ31K_BLOCK_SIZE;
            float expected = 0.0f;
            for (int k = 0; k < K; k++)
                expected = std::fma(__half2float(input[size_t(m) * K + k]),
                                    cpu_weight(block, k), expected);
            float got = __half2float(output[size_t(m) * N + row]);
            require(std::fabs(got - expected) < 0.08f,
                    "HQ3.1K CUDA output differs from CPU decode");
        }
    }

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
}

} // namespace

int main() {
    using namespace helios;
    require(dtype::HQ31K() != DTYPE_INVALID, "HQ3.1K dtype not registered");
    require(dtype_size(dtype::HQ31K(), 256) == 136, "HQ3.1K one-block size");
    require(dtype_size(dtype::HQ31K(), 257) == 272, "HQ3.1K padded size");
    test_matmul(1);
    test_matmul(2);
    std::cout << "PASS: HQ3.1K dtype, 136-byte layout and CUDA matmul" << std::endl;
    return 0;
}
