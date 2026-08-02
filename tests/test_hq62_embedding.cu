#include "dtype.hpp"
#include "hqs_common.cuh"
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_require(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " +
                                 cudaGetErrorString(status));
    }
}

void write_half(uint8_t* dst, float value) {
    half encoded = __float2half(value);
    std::memcpy(dst, &encoded, sizeof(encoded));
}

uint32_t q_value(int row, int block, int group, int index) {
    return uint32_t((row * 13 + block * 17 + group * 5 + index * 7) & 63);
}

uint8_t q_min_value(int row, int block, int group) {
    return uint8_t((row * 11 + block * 19 + group * 7) & 255);
}

std::vector<uint8_t> make_table(int vocab, int dim) {
    using namespace helios::hqs;
    const int blocks_per_row = (dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    std::vector<uint8_t> table(size_t(vocab) * blocks_per_row *
                               HQ62K_BLOCK_SIZE, 0);
    for (int row = 0; row < vocab; ++row) {
        for (int sb = 0; sb < blocks_per_row; ++sb) {
            uint8_t* block = table.data() +
                (size_t(row) * blocks_per_row + sb) * HQ62K_BLOCK_SIZE;
            write_half(block + 0, 63.0f);
            write_half(block + 2, 2.0f);
            write_half(block + 4, -4.0f);
            for (int group = 0; group < NUM_GROUPS; ++group) {
                block[8 + group] = 255;
                block[40 + group] = q_min_value(row, sb, group);
                uint64_t packed = 0;
                for (int i = 0; i < GROUP_SIZE; ++i) {
                    packed |= uint64_t(q_value(row, sb, group, i)) << (i * 6);
                }
                for (int byte = 0; byte < 6; ++byte) {
                    block[HQ62K_HEADER_SIZE + group * 6 + byte] =
                        uint8_t(packed >> (byte * 8));
                }
            }
        }
    }
    return table;
}

float expected_value(int row, int dim_index) {
    using namespace helios::hqs;
    const int sb = dim_index / SUPER_BLOCK_SIZE;
    const int in_block = dim_index % SUPER_BLOCK_SIZE;
    const int group = in_block / GROUP_SIZE;
    const int index = in_block % GROUP_SIZE;
    const float minimum = -4.0f + 2.0f *
        float(q_min_value(row, sb, group)) / 255.0f;
    return std::fma(float(q_value(row, sb, group, index)), 1.0f, minimum);
}

} // namespace

int main() {
    using namespace helios;
    using namespace helios::hqs;
    using namespace helios::kernels;

    require(dtype::HQ62K() != DTYPE_INVALID, "HQ6.2K dtype not registered");
    require(dtype_size(dtype::HQ62K(), 256) == 264, "HQ6.2K one-block size");
    require(dtype_size(dtype::HQ62K(), 257) == 528, "HQ6.2K padded size");

    constexpr int vocab = 3;
    constexpr int dim = 512;
    const std::vector<uint8_t> table = make_table(vocab, dim);
    const std::vector<int32_t> ids{1, 0, -1, vocab};
    std::vector<half> output(ids.size() * dim);

    uint8_t* device_table = nullptr;
    int32_t* device_ids = nullptr;
    half* device_output = nullptr;
    cuda_require(cudaMalloc(&device_table, table.size()), "table allocation");
    cuda_require(cudaMalloc(&device_ids, ids.size() * sizeof(int32_t)),
                 "id allocation");
    cuda_require(cudaMalloc(&device_output, output.size() * sizeof(half)),
                 "output allocation");
    cuda_require(cudaMemcpy(device_table, table.data(), table.size(),
                            cudaMemcpyHostToDevice), "table copy");
    cuda_require(cudaMemcpy(device_ids, ids.data(), ids.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice), "id copy");

    launch_embedding_hq62k(device_ids, device_table, device_output,
                           1, int(ids.size()), vocab, dim);
    cuda_require(cudaDeviceSynchronize(), "HQ6.2K embedding synchronize");
    cuda_require(cudaMemcpy(output.data(), device_output,
                            output.size() * sizeof(half), cudaMemcpyDeviceToHost),
                 "output copy");

    for (size_t token = 0; token < ids.size(); ++token) {
        for (int d = 0; d < dim; ++d) {
            const float got = __half2float(output[token * dim + d]);
            if (ids[token] < 0 || ids[token] >= vocab) {
                require(got == 0.0f, "invalid token must produce zeros");
            } else {
                const float expected = expected_value(ids[token], d);
                require(std::fabs(got - expected) < 0.04f,
                        "HQ6.2K CUDA decode differs from CPU reference");
            }
        }
    }

    cudaFree(device_table);
    cudaFree(device_ids);
    cudaFree(device_output);
    std::cout << "PASS: HQ6.2K dtype, 264-byte layout and CUDA embedding lookup"
              << std::endl;
    return 0;
}
