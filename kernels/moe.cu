// kernels/moe.cu
// ============================================================================
// GEMMA 4 MoE — router por token y expertos
// ============================================================================
//
// El router es definitivo. Con varios tokens los expertos se agrupan en la CPU y
// se lanza un GEMM por experto, así que esa ruta sincroniza y no entra en un CUDA
// Graph. Un solo token va por launch_moe_experts_decode, que sí entra.

#include "kernels.hpp"
#include "hqs_v4_gemv.cuh"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace helios {
namespace kernels {

namespace {

constexpr int MOE_MAX_TOP_K = 32;

// rms(x) * scale * factor en FP32, un bloque por token.
__global__ void moe_router_input_kernel(
    const half* __restrict__ hidden, const half* __restrict__ scale, float* __restrict__ output,
    int H, float eps, float factor
) {
    extern __shared__ float partial[];
    const half* x = hidden + size_t(blockIdx.x) * H;
    float* out = output + size_t(blockIdx.x) * H;
    float sum = 0.0f;
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        const float value = __half2float(x[c]);
        sum += value * value;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    const float inverse_rms = 1.0f / sqrtf(partial[0] / float(H) + eps);
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        out[c] = __half2float(x[c]) * inverse_rms * __half2float(scale[c]) * factor;
    }
}

// Todo el router va en FP32 con un árbol de suma fijo: el octavo experto puede quedar
// casi empatado con el noveno, y redondear a FP16 elegía distinto según el camino
// del GEMM. Así un token da los mismos ids en decode, en lote y en cualquier GPU.
__global__ void moe_logits_kernel(
    const float* __restrict__ input, const half* __restrict__ weights,
    float* __restrict__ logits, int L, int E, int H
) {
    extern __shared__ float partial[];
    const int token = blockIdx.x / E, expert = blockIdx.x % E;
    const float* x = input + size_t(token) * H;
    const half* w = weights + size_t(expert) * H;
    float sum = 0.0f;
    for (int c = threadIdx.x; c < H; c += blockDim.x) sum += x[c] * __half2float(w[c]);
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) logits[size_t(token) * E + expert] = partial[0];
}

// Un hilo por token: con 128 expertos y top-8 son ~1000 operaciones, y un empate
// se resuelve solo porque se recorre por id ascendente con comparación estricta.
__global__ void moe_route_kernel(
    const float* __restrict__ logits, const half* __restrict__ expert_scale,
    int32_t* __restrict__ indices, float* __restrict__ weights,
    int L, int E, int K
) {
    const int token = blockIdx.x;
    if (token >= L) return;
    const float* row = logits + token * E;

    float top = -INFINITY;
    for (int e = 0; e < E; ++e) top = fmaxf(top, row[e]);
    // Softmax en fp32 como la referencia. Restar el máximo evita desbordar exp.
    float sum = 0.0f;
    float best[MOE_MAX_TOP_K];
    int best_id[MOE_MAX_TOP_K];
    // Con logits NaN ninguna comparación es cierta: los ids deben seguir siendo
    // válidos, porque el decode calcula con ellos direcciones de pesos.
    for (int k = 0; k < K; ++k) { best[k] = -1.0f; best_id[k] = k; }
    for (int e = 0; e < E; ++e) {
        const float value = expf(row[e] - top);
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

// Solo inspección: la misma softmax de moe_route_kernel, sin descartar expertos.
__global__ void moe_probabilities_kernel(
    const float* __restrict__ logits, float* __restrict__ probabilities, int L, int E
) {
    const int token = blockIdx.x;
    if (token >= L) return;
    const float* row = logits + token * E;
    float* out = probabilities + token * E;
    float top = -INFINITY;
    for (int e = 0; e < E; ++e) top = fmaxf(top, row[e]);
    float sum = 0.0f;
    for (int e = 0; e < E; ++e) {
        out[e] = expf(row[e] - top);
        sum += out[e];
    }
    for (int e = 0; e < E; ++e) out[e] /= sum;
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

// gate_up trae gate en la primera mitad de cada fila y up en la segunda. La salida
// tiene P columnas: las que pasan de I son el relleno de down y van a cero.
__global__ void moe_gate_up_mul_kernel(
    const half* __restrict__ gate_up, half* __restrict__ output, int M, int I, int P
) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(M) * P) return;
    const int row = int(i / P), column = int(i % P);
    if (column >= I) {
        output[i] = __float2half(0.0f);
        return;
    }
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

// Un GEMV por experto elegido, con la dirección calculada aquí dentro: en un CUDA
// Graph los argumentos quedan fijos, pero `indices` se lee en cada ejecución.
// input_stride 0 da la misma entrada a todos; si no, cada k trae su fila.
__global__ void moe_indexed_gemv_fp16_kernel(
    const half* __restrict__ input, const int32_t* __restrict__ indices,
    const half* __restrict__ weights, half* __restrict__ output,
    size_t expert_stride, int input_stride, int K, int rows, int columns
) {
    extern __shared__ float partial[];
    const int position = blockIdx.x;
    const int k = position / rows, row = position % rows;
    const half* x = input + size_t(k) * input_stride;
    const half* w = weights + size_t(indices[k]) * expert_stride + size_t(row) * columns;
    float sum = 0.0f;
    for (int c = threadIdx.x; c < columns; c += blockDim.x) sum += __half2float(x[c]) * __half2float(w[c]);
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[size_t(k) * rows + row] = __float2half(partial[0]);
}

__global__ void moe_weighted_sum_kernel(
    const half* __restrict__ rows, const float* __restrict__ weights,
    half* __restrict__ output, int K, int H
) {
    const size_t column = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= size_t(H)) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) sum += __half2float(rows[size_t(k) * H + column]) * weights[k];
    output[column] = __float2half(sum);
}

__global__ void pad_columns_fp16_kernel(
    const half* __restrict__ input, half* __restrict__ output, int M, int logical, int physical
) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(M) * physical) return;
    const int row = int(i / physical), column = int(i % physical);
    output[i] = column < logical ? input[size_t(row) * logical + column] : __float2half(0.0f);
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

// HQ44 usa 1x8u4 fijo: el sintonizador lanza y sincroniza, y eso no cabe en un
// grafo. Es el mejor en la 4070 Ti y queda a menos del 2 % en la 2080 Ti.
void indexed_gemv(const half* input, const int32_t* indices, const void* matrix, bool hq44, half* output,
                  int input_stride, int K, int rows, int columns, cudaStream_t stream) {
    if (hq44) {
        if (columns % 256) throw std::invalid_argument("MoE decode: HQ44 exige columnas múltiplo de 256");
        hqs_v4::launch_gemv2_indexed<1, 8, 4>(
            input, indices, static_cast<const uint8_t*>(matrix), output,
            size_t(rows) * (columns / 256) * hqs_v4::BLOCK_BYTES, input_stride, K, columns, rows, stream);
        return;
    }
    const int threads = 128;
    moe_indexed_gemv_fp16_kernel<<<K * rows, threads, threads * sizeof(float), stream>>>(
        input, indices, static_cast<const half*>(matrix), output, size_t(rows) * columns, input_stride,
        K, rows, columns);
}

}  // namespace

void launch_moe_router_input(const half* hidden, const half* scale, float* output, int L, int H,
                             float eps, float factor, cudaStream_t stream) {
    if (L <= 0 || H <= 0) throw std::invalid_argument("MoE router: dimensiones inválidas");
    const int threads = 128;
    moe_router_input_kernel<<<L, threads, threads * sizeof(float), stream>>>(hidden, scale, output, H, eps, factor);
}

void launch_moe_logits(const float* input, const half* weights, float* logits, int L, int E, int H,
                       cudaStream_t stream) {
    if (L <= 0 || E <= 0 || H <= 0) throw std::invalid_argument("MoE logits: dimensiones inválidas");
    const int threads = 128;
    moe_logits_kernel<<<L * E, threads, threads * sizeof(float), stream>>>(input, weights, logits, L, E, H);
}

void launch_moe_route(const float* logits, const half* expert_scale, int32_t* indices, float* weights,
                      int L, int E, int K, cudaStream_t stream) {
    if (L <= 0 || E <= 0 || K <= 0 || K > E || K > MOE_MAX_TOP_K) {
        throw std::invalid_argument("MoE route: dimensiones inválidas");
    }
    moe_route_kernel<<<L, 1, 0, stream>>>(logits, expert_scale, indices, weights, L, E, K);
}

void launch_moe_experts_decode(const half* input, const int32_t* indices, const float* weights,
                               const void* gate_up, bool gate_up_hq44, const void* down, bool down_hq44,
                               half* gate_up_scratch, half* hidden_scratch, half* rows_scratch, half* output,
                               int K, int H, int I, int P, cudaStream_t stream) {
    if (K <= 0 || H <= 0 || I <= 0 || P < I) throw std::invalid_argument("MoE decode: dimensiones inválidas");
    indexed_gemv(input, indices, gate_up, gate_up_hq44, gate_up_scratch, 0, K, 2 * I, H, stream);
    moe_gate_up_mul_kernel<<<blocks_for(size_t(K) * P), MOE_BLOCK, 0, stream>>>(
        gate_up_scratch, hidden_scratch, K, I, P);
    indexed_gemv(hidden_scratch, indices, down, down_hq44, rows_scratch, P, K, H, P, stream);
    moe_weighted_sum_kernel<<<blocks_for(H), MOE_BLOCK, 0, stream>>>(rows_scratch, weights, output, K, H);
}

namespace {

// Con pocas filas salen GEMV. Con muchas se descuantiza a FP16 y el GEMM acumula en
// FP32: frente al decode el error baja de 0,21 % a 0,16 % en la 4070 Ti y de 0,16 %
// a 0,09 % en la 2080 Ti respecto a Hgemm.
constexpr int MOE_GEMM_MIN_ROWS = 9;

void expert_matmul(const half* input, const void* weights, bool hq44, size_t expert, size_t stride,
                   half* dequant, half* output, int M, int K, int N, cudaStream_t stream) {
    if (hq44) {
        const uint8_t* matrix = static_cast<const uint8_t*>(weights) + expert * stride;
        if (M < MOE_GEMM_MIN_ROWS) {
            launch_matmul_hq44k(input, matrix, output, M, K, N, stream);
            return;
        }
        launch_dequant_hq44k(matrix, dequant, K, N, stream);
        launch_matmul_fp16_cublas(input, dequant, output, M, K, N, stream);
    } else {
        launch_matmul_fp16_cublas(input, static_cast<const half*>(weights) + expert * stride, output,
                                  M, K, N, stream);
    }
}

}  // namespace

void launch_moe_probabilities(const float* logits, float* probabilities, int L, int E, cudaStream_t stream) {
    if (L <= 0 || E <= 0) throw std::invalid_argument("MoE probabilidades: dimensiones inválidas");
    moe_probabilities_kernel<<<L, 1, 0, stream>>>(logits, probabilities, L, E);
}

void launch_moe_gate_up_mul(const half* gate_up, half* output, int M, int I, int P, cudaStream_t stream) {
    if (M <= 0 || I <= 0 || P < I) throw std::invalid_argument("MoE gate_up: dimensiones inválidas");
    moe_gate_up_mul_kernel<<<blocks_for(size_t(M) * P), MOE_BLOCK, 0, stream>>>(gate_up, output, M, I, P);
}

void launch_pad_columns_fp16(const half* input, half* output, int M, int logical, int physical,
                             cudaStream_t stream) {
    if (M <= 0 || logical <= 0 || physical < logical) {
        throw std::invalid_argument("pad_columns: dimensiones inválidas");
    }
    pad_columns_fp16_kernel<<<blocks_for(size_t(M) * physical), MOE_BLOCK, 0, stream>>>(
        input, output, M, logical, physical);
}

void run_moe_experts_reference(const half* input, const int32_t* indices, const float* weights,
                               const void* gate_up, bool gate_up_hq44, const void* down, bool down_hq44,
                               half* output, int L, int E, int K, int H, int I, int P, cudaStream_t stream) {
    if (L <= 0 || E <= 0 || K <= 0 || H <= 0 || I <= 0 || P < I) {
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
    DeviceBuffer<half> gathered(max_rows * H), gate_up_out(max_rows * 2 * I), hidden(max_rows * P), rows(max_rows * H);
    DeviceBuffer<half> dequant(std::max(size_t(2) * I * H, size_t(H) * P));
    DeviceBuffer<int32_t> device_tokens(max_rows);
    DeviceBuffer<float> device_weights(max_rows);
    check(cudaMemset(sums.data, 0, size_t(L) * H * sizeof(float)), "limpiar suma");

    // Unidades de cada puntero: bytes en HQ44, elementos en FP16.
    const size_t gate_up_stride = gate_up_hq44 ? size_t(2) * I * (H / 256) * hqs_v4::BLOCK_BYTES
                                               : size_t(2) * I * H;
    const size_t down_stride = down_hq44 ? size_t(H) * (P / 256) * hqs_v4::BLOCK_BYTES : size_t(H) * P;
    for (int expert = 0; expert < E; ++expert) {
        const int M = int(tokens[expert].size());
        if (M == 0) continue;
        check(cudaMemcpy(device_tokens.data, tokens[expert].data(), M * sizeof(int32_t),
                         cudaMemcpyHostToDevice), "subir tokens");
        check(cudaMemcpy(device_weights.data, token_weights[expert].data(), M * sizeof(float),
                         cudaMemcpyHostToDevice), "subir pesos");
        const size_t count = size_t(M) * H, inner = size_t(M) * P;
        moe_gather_rows_kernel<<<blocks_for(count), MOE_BLOCK, 0, stream>>>(
            input, device_tokens.data, gathered.data, M, H);
        expert_matmul(gathered.data, gate_up, gate_up_hq44, expert, gate_up_stride, dequant.data,
                      gate_up_out.data, M, H, 2 * I, stream);
        moe_gate_up_mul_kernel<<<blocks_for(inner), MOE_BLOCK, 0, stream>>>(gate_up_out.data, hidden.data, M, I, P);
        expert_matmul(hidden.data, down, down_hq44, expert, down_stride, dequant.data, rows.data,
                      M, P, H, stream);
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
