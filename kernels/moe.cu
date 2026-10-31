// kernels/moe.cu
// ============================================================================
// GEMMA 4 MoE — router por token y expertos
// ============================================================================
//
// El router es definitivo. La ejecución de expertos de este archivo es la de
// REFERENCIA: agrupa en la CPU y lanza un GEMM por experto, así que sincroniza y
// no entra en un CUDA Graph. Sirve para validar el grafo contra el oráculo antes
// de escribir los kernels fundidos.

#include "kernels.hpp"
#include <cuda_fp16.h>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace helios {
namespace kernels {

namespace {

constexpr int MOE_MAX_TOP_K = 32;

// Un hilo por token: con 128 expertos y top-8 son ~1000 operaciones, y un empate
// se resuelve solo porque se recorre por id ascendente con comparación estricta.
__global__ void moe_route_kernel(
    const half* __restrict__ logits, const half* __restrict__ expert_scale,
    int32_t* __restrict__ indices, float* __restrict__ weights,
    int L, int E, int K
) {
    const int token = blockIdx.x;
    if (token >= L) return;
    const half* row = logits + token * E;

    float top = -INFINITY;
    for (int e = 0; e < E; ++e) top = fmaxf(top, __half2float(row[e]));
    // Softmax en fp32 como la referencia. Restar el máximo evita desbordar exp.
    float sum = 0.0f;
    float best[MOE_MAX_TOP_K];
    int best_id[MOE_MAX_TOP_K];
    for (int k = 0; k < K; ++k) { best[k] = -1.0f; best_id[k] = -1; }
    for (int e = 0; e < E; ++e) {
        const float value = expf(__half2float(row[e]) - top);
        sum += value;
        if (value > best[K - 1]) {
            int k = K - 1;
            while (k > 0 && value > best[k - 1]) {
                best[k] = best[k - 1];
                best_id[k] = best_id[k - 1];
                --k;
            }
            best[k] = value;
            best_id[k] = e;
        }
    }
    float kept = 0.0f;
    for (int k = 0; k < K; ++k) kept += best[k] / sum;
    for (int k = 0; k < K; ++k) {
        indices[token * K + k] = best_id[k];
        weights[token * K + k] = best[k] / sum / kept * __half2float(expert_scale[best_id[k]]);
    }
}

__global__ void moe_gather_rows_kernel(
    const half* __restrict__ input, const int32_t* __restrict__ tokens,
    half* __restrict__ output, int M, int H
) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(M) * H) return;
    const int row = int(i / H), column = int(i % H);
    output[i] = input[size_t(tokens[row]) * H + column];
}

// gate_up trae gate en la primera mitad de cada fila y up en la segunda.
__global__ void moe_gate_up_mul_kernel(
    const half* __restrict__ gate_up, half* __restrict__ output, int M, int I
) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(M) * I) return;
    const int row = int(i / I), column = int(i % I);
    const float x = __half2float(gate_up[size_t(row) * 2 * I + column]);
    const float up = __half2float(gate_up[size_t(row) * 2 * I + I + column]);
    const float x3 = x * x * x;
    const float gelu = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x3)));
    output[i] = __float2half(gelu * up);
}

__global__ void moe_scatter_add_kernel(
    const half* __restrict__ rows, const int32_t* __restrict__ tokens,
    const float* __restrict__ row_weights, float* __restrict__ sums, int M, int H
) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(M) * H) return;
    const int row = int(i / H), column = int(i % H);
    sums[size_t(tokens[row]) * H + column] += __half2float(rows[i]) * row_weights[row];
}

__global__ void moe_narrow_kernel(const float* __restrict__ input, half* __restrict__ output, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2half(input[i]);
}

constexpr int MOE_BLOCK = 256;

int blocks_for(size_t count) { return int((count + MOE_BLOCK - 1) / MOE_BLOCK); }

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (cudaMalloc(&data, count * sizeof(T)) != cudaSuccess) {
            throw std::runtime_error("MoE: sin memoria de GPU para el scratch de referencia");
        }
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

void check(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("MoE ") + where + ": " + cudaGetErrorString(error));
    }
}

}  // namespace

void launch_moe_route(const half* logits, const half* expert_scale, int32_t* indices, float* weights,
                      int L, int E, int K, cudaStream_t stream) {
    if (L <= 0 || E <= 0 || K <= 0 || K > E || K > MOE_MAX_TOP_K) {
        throw std::invalid_argument("MoE route: dimensiones inválidas");
    }
    moe_route_kernel<<<L, 1, 0, stream>>>(logits, expert_scale, indices, weights, L, E, K);
}

void run_moe_experts_reference(const half* input, const int32_t* indices, const float* weights,
                               const half* gate_up, const half* down, half* output,
                               int L, int E, int K, int H, int I, cudaStream_t stream) {
    if (L <= 0 || E <= 0 || K <= 0 || H <= 0 || I <= 0) {
        throw std::invalid_argument("MoE expertos: dimensiones inválidas");
    }
    check(cudaStreamSynchronize(stream), "esperar al router");
    std::vector<int32_t> host_indices(size_t(L) * K);
    std::vector<float> host_weights(size_t(L) * K);
    check(cudaMemcpy(host_indices.data(), indices, host_indices.size() * sizeof(int32_t),
                     cudaMemcpyDeviceToHost), "leer índices");
    check(cudaMemcpy(host_weights.data(), weights, host_weights.size() * sizeof(float),
                     cudaMemcpyDeviceToHost), "leer pesos");

    std::vector<std::vector<int32_t>> tokens(E);
    std::vector<std::vector<float>> token_weights(E);
    for (int token = 0; token < L; ++token) {
        for (int k = 0; k < K; ++k) {
            const int32_t expert = host_indices[size_t(token) * K + k];
            if (expert < 0 || expert >= E) throw std::runtime_error("MoE expertos: índice fuera de rango");
            tokens[expert].push_back(token);
            token_weights[expert].push_back(host_weights[size_t(token) * K + k]);
        }
    }

    const size_t max_rows = size_t(L);
    DeviceBuffer<float> sums(size_t(L) * H);
    DeviceBuffer<half> gathered(max_rows * H), gate_up_out(max_rows * 2 * I), hidden(max_rows * I), rows(max_rows * H);
    DeviceBuffer<int32_t> device_tokens(max_rows);
    DeviceBuffer<float> device_weights(max_rows);
    check(cudaMemset(sums.data, 0, size_t(L) * H * sizeof(float)), "limpiar suma");

    const size_t gate_up_stride = size_t(2) * I * H, down_stride = size_t(H) * I;
    for (int expert = 0; expert < E; ++expert) {
        const int M = int(tokens[expert].size());
        if (M == 0) continue;
        check(cudaMemcpy(device_tokens.data, tokens[expert].data(), M * sizeof(int32_t),
                         cudaMemcpyHostToDevice), "subir tokens");
        check(cudaMemcpy(device_weights.data, token_weights[expert].data(), M * sizeof(float),
                         cudaMemcpyHostToDevice), "subir pesos");
        const size_t count = size_t(M) * H, inner = size_t(M) * I;
        moe_gather_rows_kernel<<<blocks_for(count), MOE_BLOCK, 0, stream>>>(
            input, device_tokens.data, gathered.data, M, H);
        launch_matmul_fp16_cublas(gathered.data, gate_up + expert * gate_up_stride, gate_up_out.data,
                                  M, H, 2 * I, stream);
        moe_gate_up_mul_kernel<<<blocks_for(inner), MOE_BLOCK, 0, stream>>>(gate_up_out.data, hidden.data, M, I);
        launch_matmul_fp16_cublas(hidden.data, down + expert * down_stride, rows.data, M, I, H, stream);
        moe_scatter_add_kernel<<<blocks_for(count), MOE_BLOCK, 0, stream>>>(
            rows.data, device_tokens.data, device_weights.data, sums.data, M, H);
        check(cudaStreamSynchronize(stream), "esperar al experto");
    }
    const size_t total = size_t(L) * H;
    moe_narrow_kernel<<<blocks_for(total), MOE_BLOCK, 0, stream>>>(sums.data, output, total);
    check(cudaStreamSynchronize(stream), "esperar al resultado");
}

}  // namespace kernels
}  // namespace helios
