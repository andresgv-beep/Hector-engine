#include "hqs_common.cuh"
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_require(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " +
                                 cudaGetErrorString(status));
    }
}

void pack_5bit(uint8_t* dst, int index, uint8_t value) {
    const int bit = index * 5;
    const int byte = bit / 8;
    const int shift = bit & 7;
    const uint16_t packed = uint16_t(value & 31) << shift;
    dst[byte] |= uint8_t(packed);
    if (shift > 3) dst[byte + 1] |= uint8_t(packed >> 8);
}

template<int BITS, int BLOCK_BYTES>
std::vector<uint8_t> make_weights(int rows, int k) {
    using namespace helios::hqs;
    const int superblocks = (k + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    std::vector<uint8_t> weights(size_t(rows) * superblocks * BLOCK_BYTES, 0);
    for (int row = 0; row < rows; ++row) {
        for (int sb = 0; sb < superblocks; ++sb) {
            uint8_t* block = weights.data() +
                (size_t(row) * superblocks + sb) * BLOCK_BYTES;
            const half d_step = __float2half(0.0625f + 0.015625f * ((row + sb) % 3));
            std::memcpy(block, &d_step, sizeof(d_step));
            for (int group = 0; group < NUM_GROUPS; ++group) {
                pack_5bit(block + 4, group,
                          uint8_t(1 + (row * 7 + sb * 5 + group * 3) % 31));
                if constexpr (BITS == 4) {
                    for (int pair = 0; pair < 4; ++pair) {
                        const int q0 = (row + sb + group + pair * 2) % 16;
                        const int q1 = (row * 3 + sb + group + pair * 2 + 1) % 16;
                        block[SYMMETRIC_HEADER_SIZE + group * 4 + pair] =
                            uint8_t((q0 << 4) | q1);
                    }
                } else {
                    uint64_t packed = 0;
                    for (int lane = 0; lane < GROUP_SIZE; ++lane) {
                        const int q = (row * 3 + sb * 5 + group + lane * 2) % 32;
                        packed |= uint64_t(q) << (lane * 5);
                    }
                    for (int byte = 0; byte < 5; ++byte) {
                        block[SYMMETRIC_HEADER_SIZE + group * 5 + byte] =
                            uint8_t(packed >> (byte * 8));
                    }
                }
            }
        }
    }
    return weights;
}

template<int BITS, int BLOCK_BYTES>
float decode_weight(const std::vector<uint8_t>& weights, int row, int k,
                    int width) {
    using namespace helios::hqs;
    const int superblocks = (width + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    const int sb = k / SUPER_BLOCK_SIZE;
    const int in_block = k % SUPER_BLOCK_SIZE;
    const int group = in_block / GROUP_SIZE;
    const int lane = in_block % GROUP_SIZE;
    const uint8_t* block = weights.data() +
        (size_t(row) * superblocks + sb) * BLOCK_BYTES;
    half d_step_h;
    std::memcpy(&d_step_h, block, sizeof(d_step_h));
    const int bit = group * 5;
    const int byte = bit / 8;
    const int shift = bit & 7;
    const uint16_t word = uint16_t(block[4 + byte]) |
                          (uint16_t(block[4 + byte + 1]) << 8);
    const float step = __half2float(d_step_h) * float((word >> shift) & 31) / 31.0f;
    int code;
    if constexpr (BITS == 4) {
        const uint8_t packed = block[SYMMETRIC_HEADER_SIZE + in_block / 2];
        code = ((in_block & 1) ? (packed & 15) : (packed >> 4)) - 8;
    } else {
        const uint8_t* payload = block + SYMMETRIC_HEADER_SIZE + group * 5;
        uint64_t packed = 0;
        for (int i = 0; i < 5; ++i) packed |= uint64_t(payload[i]) << (i * 8);
        code = int((packed >> (lane * 5)) & 31) - 16;
    }
    return float(code) * step;
}

template<int BITS, int BLOCK_BYTES, typename Launch>
void test_matmul(Launch launch, const char* name) {
    constexpr int K = 768;
    constexpr int N = 7;
    const auto weights = make_weights<BITS, BLOCK_BYTES>(N, K);
    for (int m_count : {1, 9}) {
        std::vector<half> input(size_t(m_count) * K);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = __float2half((int(i % 23) - 11) * 0.0078125f);
        }
        std::vector<half> output(size_t(m_count) * N);
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
        launch(d_input, d_weights, d_output, m_count, K, N, nullptr);
        cuda_require(cudaDeviceSynchronize(), name);
        cuda_require(cudaMemcpy(output.data(), d_output, output.size() * sizeof(half),
                                cudaMemcpyDeviceToHost), "copy output");

        for (int m = 0; m < m_count; ++m) {
            for (int row = 0; row < N; ++row) {
                float expected = 0.0f;
                for (int k = 0; k < K; ++k) {
                    expected = std::fma(decode_weight<BITS, BLOCK_BYTES>(weights, row, k, K),
                                        __half2float(input[size_t(m) * K + k]), expected);
                }
                const float actual = __half2float(output[size_t(m) * N + row]);
                const float tolerance = m_count == 1 ? 0.003f : 0.02f;
                require(std::fabs(actual - expected) <= tolerance,
                        std::string(name) + " mismatch at M=" + std::to_string(m_count) +
                        " row=" + std::to_string(row) + " actual=" +
                        std::to_string(actual) + " expected=" + std::to_string(expected));
            }
        }
        cudaFree(d_input);
        cudaFree(d_weights);
        cudaFree(d_output);
    }
}

// Independent CPU expansion followed by the same cuBLAS multiplication.
// Covers vector-store alignment, incomplete quantization blocks and the scalar
// fallback for K not divisible by eight. Graph replay also uses the real launcher.
template<int BITS, int BLOCK_BYTES, typename Launch>
void test_prefill_packed_stores(Launch launch) {
    cublasHandle_t handle;
    require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "create cuBLAS");
    require(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) == CUBLAS_STATUS_SUCCESS,
            "cuBLAS math mode");
    for (int K : {7, 8, 248, 255, 256, 264, 2040, 2048, 2056, 3840, 15360}) {
        const int N = 7;
        const int M = 9;
        auto weights = make_weights<BITS, BLOCK_BYTES>(N, K);
        std::vector<half> expanded(size_t(N) * K), input(size_t(M) * K);
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                expanded[size_t(n) * K + k] = __float2half(
                    decode_weight<BITS, BLOCK_BYTES>(weights, n, k, K));
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = __float2half((int(i % 29) - 14) * 0.00390625f);
        uint8_t* dw;
        half *dx, *de, *actual, *expected;
        cuda_require(cudaMalloc(&dw, weights.size()), "quant weights");
        cuda_require(cudaMalloc(&dx, input.size() * 2), "input");
        cuda_require(cudaMalloc(&de, expanded.size() * 2), "reference weights");
        cuda_require(cudaMalloc(&actual, M * N * 2), "actual");
        cuda_require(cudaMalloc(&expected, M * N * 2), "expected");
        cuda_require(cudaMemcpy(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice), "weights");
        cuda_require(cudaMemcpy(dx, input.data(), input.size() * 2, cudaMemcpyHostToDevice), "input");
        cuda_require(cudaMemcpy(de, expanded.data(), expanded.size() * 2, cudaMemcpyHostToDevice), "reference");
        half alpha = __float2half(1), beta = __float2half(0);
        require(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                           &alpha, de, K, dx, K, &beta, expected, N) == CUBLAS_STATUS_SUCCESS,
                "reference GEMM");
        cudaStream_t stream;
        cuda_require(cudaStreamCreate(&stream), "stream");
        launch(dx, dw, actual, M, K, N, stream); // Allocate scratch before capture.
        cuda_require(cudaDeviceSynchronize(), "warmup");
        std::vector<half> a(M * N), e(M * N);
        cuda_require(cudaMemcpy(e.data(), expected, e.size() * 2, cudaMemcpyDeviceToHost), "expected output");
        for (int graph = 0; graph < 2; ++graph) {
            cudaGraph_t captured = nullptr;
            cudaGraphExec_t exec = nullptr;
            if (graph) cuda_require(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture");
            launch(dx, dw, actual, M, K, N, stream);
            if (graph) {
                cuda_require(cudaStreamEndCapture(stream, &captured), "end capture");
                cuda_require(cudaGraphInstantiate(&exec, captured, nullptr, nullptr, 0), "instantiate");
                cuda_require(cudaGraphLaunch(exec, stream), "replay");
            }
            cuda_require(cudaStreamSynchronize(stream), "synchronize");
            cuda_require(cudaMemcpy(a.data(), actual, a.size() * 2, cudaMemcpyDeviceToHost), "actual output");
            require(std::memcmp(a.data(), e.data(), a.size() * 2) == 0,
                    "prefill CPU-expansion parity BITS=" + std::to_string(BITS) +
                    " K=" + std::to_string(K) + " graph=" + std::to_string(graph));
            if (graph) { cudaGraphExecDestroy(exec); cudaGraphDestroy(captured); }
        }
        cudaStreamDestroy(stream);
        cudaFree(dw); cudaFree(dx); cudaFree(de); cudaFree(actual); cudaFree(expected);
    }
    cublasDestroy(handle);
}

template<int BITS, int BLOCK_BYTES, typename Launch>
void test_embedding(Launch launch, const char* name) {
    constexpr int VOCAB = 3;
    constexpr int DIM = 768;
    const auto table = make_weights<BITS, BLOCK_BYTES>(VOCAB, DIM);
    const std::vector<int32_t> indices = {2, 0, -1, 3};
    std::vector<half> output(indices.size() * DIM);
    int32_t* d_indices = nullptr;
    uint8_t* d_table = nullptr;
    half* d_output = nullptr;
    cuda_require(cudaMalloc(&d_indices, indices.size() * sizeof(int32_t)), "indices allocation");
    cuda_require(cudaMalloc(&d_table, table.size()), "table allocation");
    cuda_require(cudaMalloc(&d_output, output.size() * sizeof(half)), "embedding allocation");
    cuda_require(cudaMemcpy(d_indices, indices.data(), indices.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice), "copy indices");
    cuda_require(cudaMemcpy(d_table, table.data(), table.size(), cudaMemcpyHostToDevice),
                 "copy table");
    launch(d_indices, d_table, d_output, 1, int(indices.size()), VOCAB, DIM, nullptr);
    cuda_require(cudaDeviceSynchronize(), name);
    cuda_require(cudaMemcpy(output.data(), d_output, output.size() * sizeof(half),
                            cudaMemcpyDeviceToHost), "copy embedding");
    for (size_t token = 0; token < indices.size(); ++token) {
        for (int d = 0; d < DIM; ++d) {
            const float expected = indices[token] >= 0 && indices[token] < VOCAB
                ? decode_weight<BITS, BLOCK_BYTES>(table, indices[token], d, DIM) : 0.0f;
            const float actual = __half2float(output[token * DIM + d]);
            require(std::fabs(actual - expected) <= 0.001f,
                    std::string(name) + " mismatch at token=" + std::to_string(token) +
                    " dim=" + std::to_string(d));
        }
    }
    cudaFree(d_indices);
    cudaFree(d_table);
    cudaFree(d_output);
}

}  // namespace

int main() {
    using namespace helios;
    using namespace helios::hqs;
    test_matmul<4, HQ42K_BLOCK_SIZE>(kernels::launch_matmul_hq42k, "HQ4.2K matmul");
    test_matmul<5, HQ52K_BLOCK_SIZE>(kernels::launch_matmul_hq52k, "HQ5.2K matmul");
    test_prefill_packed_stores<4, HQ42K_BLOCK_SIZE>(kernels::launch_matmul_hq42k);
    test_prefill_packed_stores<5, HQ52K_BLOCK_SIZE>(kernels::launch_matmul_hq52k);
    test_embedding<4, HQ42K_BLOCK_SIZE>(kernels::launch_embedding_hq42k, "HQ4.2K embedding");
    test_embedding<5, HQ52K_BLOCK_SIZE>(kernels::launch_embedding_hq52k, "HQ5.2K embedding");
    std::cout << "PASS: HQ4.2K/HQ5.2K CUDA matmul and embedding match CPU decode" << std::endl;
    return 0;
}
