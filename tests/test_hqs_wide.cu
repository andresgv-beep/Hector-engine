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

void encode_unit_blocks(std::vector<uint8_t>& weights, size_t block_size,
                        int rows, int superblocks) {
    using namespace helios::hqs;
    const half one = __float2half(1.0f);
    for (int row = 0; row < rows; ++row) {
        for (int sb = 0; sb < superblocks; ++sb) {
            uint8_t* block = weights.data() +
                (size_t(row) * superblocks + sb) * block_size;
            std::memcpy(block, &one, sizeof(one)); // d_scale = 1
            std::memset(block + 8, 0xff, 16);      // q_scale = 15
            std::memset(block + COMPACT_HEADER_SIZE, 0xff,
                        block_size - COMPACT_HEADER_SIZE); // q = q_max
        }
    }
}

template <typename Launch>
void test_wide_k(size_t block_size, Launch launch, const char* format) {
    using namespace helios::hqs;

    // Qwen2.5-Coder-7B usa K=18944 en down_proj. Este valor supera el
    // máximo que el GEMV compacto puede copiar a shared memory (16384).
    constexpr int M = 1;
    constexpr int K = 18944;
    // No es múltiplo de ROWS_PER_BLOCK: cubre también el último bloque
    // parcialmente ocupado a través de las sincronizaciones entre ventanas.
    constexpr int N = 7;
    const int superblocks = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;

    // La primera ventana no aporta nada y la segunda aporta 0.25 por peso.
    // Así el resultado demuestra que el tramo K>16384 se procesó; una prueba
    // con pesos cero solo detectaría la ausencia de recursión.
    std::vector<half> input(size_t(M) * K, __float2half(0.0f));
    for (int k = 16384; k < K; ++k) input[k] = __float2half(0.25f);
    std::vector<uint8_t> weights(size_t(N) * superblocks * block_size, 0);
    encode_unit_blocks(weights, block_size, N, superblocks);
    std::vector<half> output(size_t(M) * N, __float2half(123.0f));

    half* d_input = nullptr;
    uint8_t* d_weights = nullptr;
    half* d_output = nullptr;
    cuda_require(cudaMalloc(&d_input, input.size() * sizeof(half)),
                 "input allocation");
    cuda_require(cudaMalloc(&d_weights, weights.size()), "weight allocation");
    cuda_require(cudaMalloc(&d_output, output.size() * sizeof(half)),
                 "output allocation");
    cuda_require(cudaMemcpy(d_input, input.data(), input.size() * sizeof(half),
                            cudaMemcpyHostToDevice), "copy input");
    cuda_require(cudaMemcpy(d_weights, weights.data(), weights.size(),
                            cudaMemcpyHostToDevice), "copy weights");
    cuda_require(cudaMemcpy(d_output, output.data(), output.size() * sizeof(half),
                            cudaMemcpyHostToDevice), "copy output sentinel");

    launch(d_input, d_weights, d_output, M, K, N, nullptr);
    cuda_require(cudaDeviceSynchronize(), format);
    cuda_require(cudaMemcpy(output.data(), d_output,
                            output.size() * sizeof(half), cudaMemcpyDeviceToHost),
                 "copy output");

    constexpr float expected = float(K - 16384) * 0.25f;
    for (half value : output) {
        require(std::fabs(__half2float(value) - expected) < 1e-3f,
                "wide compact GEMV must accumulate the second K chunk");
    }

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
}

} // namespace

int main() {
    using namespace helios;
    using namespace helios::hqs;

    test_wide_k(HQ41K_BLOCK_SIZE, kernels::launch_matmul_hq41k,
                "HQ4.1K wide-K synchronize");
    test_wide_k(HQ51K_BLOCK_SIZE, kernels::launch_matmul_hq51k,
                "HQ5.1K wide-K synchronize");

    std::cout << "PASS: HQ4.1K/HQ5.1K chunked wide-K GEMV (K=18944)" << std::endl;
    return 0;
}
