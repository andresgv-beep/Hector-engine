// kernels/delta_net.cu
// ============================================================================
// GATED DELTANET — la atención lineal de la familia Qwen3.5
// ============================================================================
//
// Tres de cada cuatro capas de Qwen3.5 y 3.6 no guardan caché de claves y valores: llevan
// un estado de tamaño fijo, una matriz por cabeza, que se actualiza con cada token. Aquí van
// las dos piezas que el motor no tenía. Las proyecciones de entrada y de salida son matmul
// corrientes y se quedan fuera.
//
//   1. Convolución causal de K muestras, canal a canal, y SiLU. Recuerda las K-1 muestras
//      anteriores de cada canal.
//   2. La regla delta. Por cabeza de valor, con S de [dk, dv]:
//
//          S <- S · exp(g)
//          S <- S + k ⊗ ((v − Sᵀk) · beta)
//          o  = Sᵀ (q / √dk)
//
//      con q y k normalizadas a norma 1, g = −exp(a_log) · softplus(a + dt_bias) y
//      beta = sigmoide(b). Después, norma RMS de o por cabeza, por el peso y por silu(z).
//
// La regla es recurrente: el token t necesita el estado que dejó el t-1. Un bloque por cabeza
// recorre los tokens en orden, así que procesar L tokens de una vez da exactamente lo mismo
// que procesarlos de uno en uno, que es lo que hace falta para que prefill y decode coincidan.
// Todo el estado va en FP32, como en la referencia (modeling_qwen3_5.py de transformers).

#include "kernels.hpp"
#include <cuda_fp16.h>
#include <cmath>
#include <stdexcept>

namespace helios {
namespace kernels {

namespace {

constexpr int DELTA_MAX_DIM = 256;       // dk y dv caben en un bloque
constexpr int CONV_BLOCK = 256;
constexpr int CONV_MAX_KERNEL = 8;

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

// in(τ) es la entrada del token τ de esta llamada, o una muestra recordada si τ < 0.
__global__ void causal_conv_silu_kernel(
    const half* __restrict__ x, const half* __restrict__ weight, const float* __restrict__ state,
    half* __restrict__ out, int L, int C, int K
) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(L) * C) return;
    const int t = int(i / C), c = int(i % C);
    float sum = 0.0f;
    for (int j = 0; j < K; ++j) {
        const int tau = t - (K - 1) + j;
        const float sample = tau >= 0 ? __half2float(x[size_t(tau) * C + c]) : state[size_t(c) * (K - 1) + (K - 1) + tau];
        sum += __half2float(weight[size_t(c) * K + j]) * sample;
    }
    out[i] = __float2half(silu_f(sum));
}

// Las K-1 últimas entradas de cada canal, la más antigua primero. Un hilo por canal: lee lo
// que necesita del estado viejo antes de pisarlo.
__global__ void causal_conv_state_kernel(
    const half* __restrict__ x, float* __restrict__ state, int L, int C, int K
) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float kept[CONV_MAX_KERNEL];
    for (int j = 0; j < K - 1; ++j) {
        const int tau = L - (K - 1) + j;
        kept[j] = tau >= 0 ? __half2float(x[size_t(tau) * C + c]) : state[size_t(c) * (K - 1) + (K - 1) + tau];
    }
    for (int j = 0; j < K - 1; ++j) state[size_t(c) * (K - 1) + j] = kept[j];
}

// Un bloque por cabeza de valor, dv hilos. El hilo j lleva la columna j de S.
__global__ void gated_delta_rule_kernel(
    const half* __restrict__ mixed, const half* __restrict__ z,
    const half* __restrict__ a, const half* __restrict__ b,
    const half* __restrict__ a_log, const half* __restrict__ dt_bias, const half* __restrict__ norm_w,
    float* __restrict__ state, half* __restrict__ out,
    int L, int nk, int nv, int dk, int dv, float eps
) {
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    if (h >= nv) return;
    const int key_dim = nk * dk, value_dim = nv * dv;
    const int C = 2 * key_dim + value_dim;
    const int kh = h / (nv / nk);              // cada cabeza de clave sirve a nv/nk de valor
    float* S = state + size_t(h) * dk * dv;

    __shared__ float q_s[DELTA_MAX_DIM];
    __shared__ float k_s[DELTA_MAX_DIM];
    __shared__ float o_s[DELTA_MAX_DIM];
    __shared__ float scalars[2];

    const float A = expf(__half2float(a_log[h]));
    const float bias = __half2float(dt_bias[h]);
    const float weight = j < dv ? __half2float(norm_w[j]) : 0.0f;
    const float inv_sqrt_dk = rsqrtf(float(dk));

    for (int t = 0; t < L; ++t) {
        const half* row = mixed + size_t(t) * C;
        for (int i = j; i < dk; i += blockDim.x) {
            q_s[i] = __half2float(row[kh * dk + i]);
            k_s[i] = __half2float(row[key_dim + kh * dk + i]);
        }
        __syncthreads();
        if (j == 0) {
            float qq = 0.0f, kk = 0.0f;
            for (int i = 0; i < dk; ++i) { qq += q_s[i] * q_s[i]; kk += k_s[i] * k_s[i]; }
            scalars[0] = rsqrtf(qq + 1e-6f) * inv_sqrt_dk;
            scalars[1] = rsqrtf(kk + 1e-6f);
        }
        __syncthreads();
        const float q_scale = scalars[0], k_scale = scalars[1];
        for (int i = j; i < dk; i += blockDim.x) { q_s[i] *= q_scale; k_s[i] *= k_scale; }
        __syncthreads();

        float o = 0.0f;
        if (j < dv) {
            const float pre = __half2float(a[size_t(t) * nv + h]) + bias;
            const float softplus = pre > 20.0f ? pre : log1pf(expf(pre));
            const float decay = expf(-A * softplus);
            const float beta = 1.0f / (1.0f + expf(-__half2float(b[size_t(t) * nv + h])));
            const float v = __half2float(row[2 * key_dim + h * dv + j]);
            float remembered = 0.0f;
            for (int i = 0; i < dk; ++i) remembered += S[size_t(i) * dv + j] * decay * k_s[i];
            const float delta = (v - remembered) * beta;
            for (int i = 0; i < dk; ++i) {
                const float updated = S[size_t(i) * dv + j] * decay + k_s[i] * delta;
                S[size_t(i) * dv + j] = updated;
                o += updated * q_s[i];
            }
            o_s[j] = o;
        }
        __syncthreads();
        if (j == 0) {
            float ss = 0.0f;
            for (int i = 0; i < dv; ++i) ss += o_s[i] * o_s[i];
            scalars[0] = rsqrtf(ss / float(dv) + eps);
        }
        __syncthreads();
        if (j < dv) {
            const float gate = __half2float(z[size_t(t) * value_dim + h * dv + j]);
            out[size_t(t) * value_dim + h * dv + j] = __float2half(o * scalars[0] * weight * silu_f(gate));
        }
        __syncthreads();
    }
}

// q_proj saca, por cabeza, la consulta y después una puerta del mismo tamaño.
__global__ void split_query_gate_kernel(
    const half* __restrict__ doubled, half* __restrict__ query, half* __restrict__ gate, size_t count, int head_dim
) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const size_t head = i / head_dim, d = i % head_dim;
    query[i] = doubled[head * 2 * head_dim + d];
    gate[i] = doubled[head * 2 * head_dim + head_dim + d];
}

__global__ void sigmoid_mul_kernel(half* __restrict__ x, const half* __restrict__ gate, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    x[i] = __float2half(__half2float(x[i]) / (1.0f + expf(-__half2float(gate[i]))));
}

void check(cudaError_t error, const char* where) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(error));
}

} // namespace

void launch_causal_conv_silu(const half* x, const half* weight, float* state, half* out,
                             int L, int C, int K, cudaStream_t stream) {
    if (L <= 0 || C <= 0 || K < 2 || K > CONV_MAX_KERNEL) throw std::invalid_argument("causal_conv_silu: dimensiones inválidas");
    if (x == out) throw std::invalid_argument("causal_conv_silu: la salida no puede ser la entrada");
    const size_t total = size_t(L) * C;
    causal_conv_silu_kernel<<<unsigned((total + CONV_BLOCK - 1) / CONV_BLOCK), CONV_BLOCK, 0, stream>>>(
        x, weight, state, out, L, C, K);
    check(cudaGetLastError(), "causal_conv_silu");
    causal_conv_state_kernel<<<(C + CONV_BLOCK - 1) / CONV_BLOCK, CONV_BLOCK, 0, stream>>>(x, state, L, C, K);
    check(cudaGetLastError(), "causal_conv_state");
}

void launch_gated_delta_rule(const half* mixed, const half* z, const half* a, const half* b,
                             const half* a_log, const half* dt_bias, const half* norm_w,
                             float* state, half* out,
                             int L, int nk, int nv, int dk, int dv, float eps, cudaStream_t stream) {
    if (L <= 0 || nk <= 0 || nv <= 0 || nv % nk != 0 || dk <= 0 || dv <= 0 ||
        dk > DELTA_MAX_DIM || dv > DELTA_MAX_DIM) {
        throw std::invalid_argument("gated_delta_rule: dimensiones inválidas");
    }
    gated_delta_rule_kernel<<<nv, dv, 0, stream>>>(mixed, z, a, b, a_log, dt_bias, norm_w, state, out,
                                                  L, nk, nv, dk, dv, eps);
    check(cudaGetLastError(), "gated_delta_rule");
}

void launch_split_query_gate(const half* doubled, half* query, half* gate,
                             int rows, int num_heads, int head_dim, cudaStream_t stream) {
    if (rows <= 0 || num_heads <= 0 || head_dim <= 0) throw std::invalid_argument("split_query_gate: dimensiones inválidas");
    const size_t count = size_t(rows) * num_heads * head_dim;
    split_query_gate_kernel<<<unsigned((count + CONV_BLOCK - 1) / CONV_BLOCK), CONV_BLOCK, 0, stream>>>(
        doubled, query, gate, count, head_dim);
    check(cudaGetLastError(), "split_query_gate");
}

void launch_sigmoid_mul(half* x, const half* gate, size_t count, cudaStream_t stream) {
    if (count == 0) return;
    sigmoid_mul_kernel<<<unsigned((count + CONV_BLOCK - 1) / CONV_BLOCK), CONV_BLOCK, 0, stream>>>(x, gate, count);
    check(cudaGetLastError(), "sigmoid_mul");
}

} // namespace kernels
} // namespace helios
