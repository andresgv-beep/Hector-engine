// kernels/sampling.cu
// ============================================================================
// SAMPLING KERNELS - Argmax, Top-K, Temperature
// ============================================================================
// Diseño flexible: cada operación es independiente y combinable
//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

namespace helios {
namespace kernels {

// ============================================================================
// ARGMAX - Find index of maximum value
// ============================================================================
// input: [vocab_size] FP16
// output: single int32 (token ID)
//
// Usa reducción paralela en shared memory

__global__ void argmax_fp16_kernel(
    const half* __restrict__ input,
    int32_t* __restrict__ output,
    int vocab_size
) {
    extern __shared__ char smem[];
    float* smax = (float*)smem;
    int32_t* sidx = (int32_t*)(smem + blockDim.x * sizeof(float));
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Cada thread encuentra su máximo local
    float local_max = -FLT_MAX;
    int32_t local_idx = 0;
    
    for (int i = tid; i < vocab_size; i += stride) {
        float val = __half2float(input[i]);
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }
    
    smax[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();
    
    // Reducción en shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smax[tid + s] > smax[tid]) {
                smax[tid] = smax[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Thread 0 escribe resultado
    if (tid == 0) {
        output[0] = sidx[0];
    }
}

void launch_argmax_fp16(
    const half* input,
    int32_t* output,
    int vocab_size,
    cudaStream_t stream
) {
    int block_size = 256;
    size_t smem_size = block_size * (sizeof(float) + sizeof(int32_t));
    
    argmax_fp16_kernel<<<1, block_size, smem_size, stream>>>(
        input, output, vocab_size
    );
}

// ============================================================================
// TEMPERATURE SCALING - Divide logits by temperature
// ============================================================================
// logits = logits / temperature
// In-place operation

__global__ void temperature_scale_kernel(
    half* __restrict__ logits,
    int vocab_size,
    float inv_temperature  // 1.0 / temperature for efficiency
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < vocab_size; i += stride) {
        float val = __half2float(logits[i]);
        logits[i] = __float2half(val * inv_temperature);
    }
}

void launch_temperature_scale(
    half* logits,
    int vocab_size,
    float temperature,
    cudaStream_t stream
) {
    if (temperature == 1.0f) return;  // No-op
    
    int block_size = 256;
    int grid_size = (vocab_size + block_size - 1) / block_size;
    
    temperature_scale_kernel<<<grid_size, block_size, 0, stream>>>(
        logits, vocab_size, 1.0f / temperature
    );
}

// ============================================================================
// SOFTMAX - Convert logits to probabilities
// ============================================================================
// Numerically stable: subtract max first

__global__ void softmax_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int vocab_size
) {
    extern __shared__ float smem_softmax[];
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Pass 1: Find max
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += stride) {
        float val = __half2float(input[i]);
        local_max = fmaxf(local_max, val);
    }
    
    smem_softmax[tid] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem_softmax[tid] = fmaxf(smem_softmax[tid], smem_softmax[tid + s]);
        }
        __syncthreads();
    }
    float max_val = smem_softmax[0];
    __syncthreads();
    
    // Pass 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += stride) {
        float val = __half2float(input[i]);
        local_sum += expf(val - max_val);
    }
    
    smem_softmax[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem_softmax[tid] += smem_softmax[tid + s];
        }
        __syncthreads();
    }
    float sum_exp = smem_softmax[0];
    __syncthreads();
    
    // Pass 3: Normalize
    float inv_sum = 1.0f / sum_exp;
    for (int i = tid; i < vocab_size; i += stride) {
        float val = __half2float(input[i]);
        float prob = expf(val - max_val) * inv_sum;
        output[i] = __float2half(prob);
    }
}

void launch_softmax_fp16(
    const half* input,
    half* output,
    int vocab_size,
    cudaStream_t stream
) {
    int block_size = 256;
    size_t smem_size = block_size * sizeof(float);
    
    softmax_fp16_kernel<<<1, block_size, smem_size, stream>>>(
        input, output, vocab_size
    );
}

// ============================================================================
// TOP-K SELECTION - Find top K indices and values
// ============================================================================
// Simple O(n*k) selection - good enough for small k
// For production, use radix select or bitonic sort

// v1 (fallback para k grande): un solo thread — lento pero sin límite de k
__global__ void top_k_kernel(
    const half* __restrict__ input,
    float* __restrict__ top_values,    // [k]
    int32_t* __restrict__ top_indices, // [k]
    int vocab_size,
    int k
) {
    if (threadIdx.x != 0) return;
    for (int i = 0; i < k; i++) {
        top_values[i] = -FLT_MAX;
        top_indices[i] = -1;
    }
    for (int i = 0; i < vocab_size; i++) {
        float val = __half2float(input[i]);
        if (val > top_values[k-1]) {
            int pos = k - 1;
            while (pos > 0 && val > top_values[pos-1]) {
                top_values[pos] = top_values[pos-1];
                top_indices[pos] = top_indices[pos-1];
                pos--;
            }
            top_values[pos] = val;
            top_indices[pos] = i;
        }
    }
}

// v2: top-k paralelo por warps. Cada warp mantiene su top-k en shared con
// inserción serializada solo para candidatos (raros tras el arranque: el
// umbral my_vals[k-1] solo sube, así que leer un umbral viejo es conservador
// — nunca pierde candidatos). Fusión final de las listas por el thread 0.
// ~200× más rápido que v1 sobre vocabularios de 150k.
constexpr int TK_THREADS = 256;
constexpr int TK_WARPS = TK_THREADS / 32;
constexpr int TK_MAX_K = 64;

__global__ void top_k_kernel_v2(
    const half* __restrict__ input,
    float* __restrict__ top_values,
    int32_t* __restrict__ top_indices,
    int vocab_size,
    int k
) {
    extern __shared__ float smem_tk[];
    float* w_vals = smem_tk;                                  // [TK_WARPS][k]
    int32_t* w_idx = reinterpret_cast<int32_t*>(smem_tk + TK_WARPS * k);

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    for (int i = threadIdx.x; i < TK_WARPS * k; i += blockDim.x) {
        w_vals[i] = -FLT_MAX;
        w_idx[i] = -1;
    }
    __syncthreads();

    float* my_vals = w_vals + warp * k;
    int32_t* my_idx = w_idx + warp * k;

    for (int base = warp * 32; base < vocab_size; base += TK_WARPS * 32) {
        int i = base + lane;
        float v = (i < vocab_size) ? __half2float(input[i]) : -FLT_MAX;

        unsigned mask = __ballot_sync(0xFFFFFFFF, v > my_vals[k - 1]);
        while (mask) {
            int src = __ffs(mask) - 1;
            mask &= mask - 1;
            float cv = __shfl_sync(0xFFFFFFFF, v, src);
            int ci = base + src;
            if (lane == 0 && cv > my_vals[k - 1]) {
                int pos = k - 1;
                while (pos > 0 && cv > my_vals[pos - 1]) {
                    my_vals[pos] = my_vals[pos - 1];
                    my_idx[pos] = my_idx[pos - 1];
                    pos--;
                }
                my_vals[pos] = cv;
                my_idx[pos] = ci;
            }
            __syncwarp();
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < k; i++) {
            top_values[i] = -FLT_MAX;
            top_indices[i] = -1;
        }
        for (int w = 0; w < TK_WARPS; w++) {
            for (int j = 0; j < k; j++) {
                float v = w_vals[w * k + j];
                if (v <= top_values[k - 1]) break;  // lista ordenada: nada más que aportar
                int pos = k - 1;
                while (pos > 0 && v > top_values[pos - 1]) {
                    top_values[pos] = top_values[pos - 1];
                    top_indices[pos] = top_indices[pos - 1];
                    pos--;
                }
                top_values[pos] = v;
                top_indices[pos] = w_idx[w * k + j];
            }
        }
    }
}

void launch_top_k(
    const half* input,
    float* top_values,
    int32_t* top_indices,
    int vocab_size,
    int k,
    cudaStream_t stream
) {
    if (k <= TK_MAX_K) {
        size_t smem = (size_t)TK_WARPS * k * (sizeof(float) + sizeof(int32_t));
        top_k_kernel_v2<<<1, TK_THREADS, smem, stream>>>(
            input, top_values, top_indices, vocab_size, k
        );
    } else {
        top_k_kernel<<<1, 1, 0, stream>>>(
            input, top_values, top_indices, vocab_size, k
        );
    }
}

// ============================================================================
// TOP-P (NUCLEUS) CUMULATIVE SUM
// ============================================================================
// Given sorted probabilities, find cutoff index where cumsum >= p

__global__ void top_p_cutoff_kernel(
    const float* __restrict__ sorted_probs,  // [k] sorted descending
    int32_t* __restrict__ cutoff_idx,
    int k,
    float p
) {
    if (threadIdx.x != 0) return;
    
    float cumsum = 0.0f;
    for (int i = 0; i < k; i++) {
        cumsum += sorted_probs[i];
        if (cumsum >= p) {
            cutoff_idx[0] = i + 1;  // Number of tokens to keep
            return;
        }
    }
    cutoff_idx[0] = k;  // Keep all
}

void launch_top_p_cutoff(
    const float* sorted_probs,
    int32_t* cutoff_idx,
    int k,
    float p,
    cudaStream_t stream
) {
    top_p_cutoff_kernel<<<1, 1, 0, stream>>>(
        sorted_probs, cutoff_idx, k, p
    );
}

// ============================================================================
// CATEGORICAL SAMPLE - Sample from probability distribution
// ============================================================================
// Given probabilities and a random value [0,1), return sampled index

__global__ void categorical_sample_kernel(
    const float* __restrict__ probs,  // [n] probabilities (normalized)
    const int32_t* __restrict__ indices,  // [n] token indices
    int32_t* __restrict__ output,
    int n,
    float random_val  // Uniform [0, 1)
) {
    if (threadIdx.x != 0) return;
    
    float cumsum = 0.0f;
    for (int i = 0; i < n; i++) {
        cumsum += probs[i];
        if (random_val < cumsum) {
            output[0] = indices[i];
            return;
        }
    }
    // Fallback to last (shouldn't happen with proper normalization)
    output[0] = indices[n-1];
}

void launch_categorical_sample(
    const float* probs,
    const int32_t* indices,
    int32_t* output,
    int n,
    float random_val,
    cudaStream_t stream
) {
    categorical_sample_kernel<<<1, 1, 0, stream>>>(
        probs, indices, output, n, random_val
    );
}

// ============================================================================
// SAMPLE TOP-K GPU — softmax + top-p + muestreo categórico en UN kernel
// ============================================================================
// Trabaja sobre los k candidatos ya ordenados del top-k (k ≤ 64): el trabajo
// es trivial, lo que importa es que TODO queda en GPU. El sampler solo copia
// de vuelta un int32 — mismo coste que el greedy, sin softmax en CPU ni
// viajes de valores/índices.

__global__ void sample_topk_gpu_kernel(
    const float* __restrict__ top_values,    // [k] logits ordenados desc
    const int32_t* __restrict__ top_indices, // [k]
    int32_t* __restrict__ output,            // [1] token elegido
    int k,
    float inv_temp,
    float top_p,
    float random_val                          // uniforme [0,1)
) {
    if (threadIdx.x != 0) return;

    float probs[64];
    float maxv = top_values[0];
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        float p = expf((top_values[i] - maxv) * inv_temp);
        probs[i] = p;
        sum += p;
    }

    // Top-p (nucleus) sobre la distribución normalizada
    int n = k;
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < k; i++) {
            cum += probs[i] / sum;
            if (cum >= top_p) { n = i + 1; break; }
        }
    }

    // Renormalizar sobre los n supervivientes y muestrear
    float sum_n = 0.0f;
    for (int i = 0; i < n; i++) sum_n += probs[i];
    float r = random_val * sum_n;
    float cum = 0.0f;
    for (int i = 0; i < n; i++) {
        cum += probs[i];
        if (r < cum) { output[0] = top_indices[i]; return; }
    }
    output[0] = top_indices[n - 1];
}

void launch_sample_topk_gpu(
    const float* top_values,
    const int32_t* top_indices,
    int32_t* output,
    int k,
    float inv_temp,
    float top_p,
    float random_val,
    cudaStream_t stream
) {
    sample_topk_gpu_kernel<<<1, 1, 0, stream>>>(
        top_values, top_indices, output, k, inv_temp, top_p, random_val
    );
}

// ============================================================================
// REPETITION PENALTY - Penalize tokens that appeared in context
// ============================================================================
// For each token in context_ids, apply:
//   if logit > 0: logit /= rep_penalty
//   if logit < 0: logit *= rep_penalty
//   logit -= frequency_penalty * count(token)
//   logit -= presence_penalty * (count(token) > 0 ? 1 : 0)
//
// This matches the HuggingFace / llama.cpp convention.
// The kernel works on the full vocab — context tokens are scattered.

__global__ void apply_repetition_penalty_kernel(
    half* __restrict__ logits,             // [vocab_size] — modified in place
    const int32_t* __restrict__ context,   // [context_len] — token IDs
    const int32_t* __restrict__ counts,    // [vocab_size] — occurrence counts
    int vocab_size,
    float rep_penalty,                     // multiplicative (1.0 = disabled)
    float freq_penalty,                    // additive per-occurrence (0.0 = disabled)
    float pres_penalty                     // additive per-presence (0.0 = disabled)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size) return;
    
    int count = counts[idx];
    if (count == 0) return;  // Token not in context — no penalty
    
    float logit = __half2float(logits[idx]);
    
    // Multiplicative repetition penalty (à la Keskar et al.)
    if (rep_penalty != 1.0f) {
        if (logit > 0.0f) {
            logit /= rep_penalty;
        } else {
            logit *= rep_penalty;
        }
    }
    
    // Additive frequency penalty (linear in count)
    logit -= freq_penalty * (float)count;
    
    // Additive presence penalty (binary)
    logit -= pres_penalty;
    
    logits[idx] = __float2half(logit);
}

void launch_repetition_penalty(
    half* logits,
    const int32_t* context,
    const int32_t* counts,
    int vocab_size,
    float rep_penalty,
    float freq_penalty,
    float pres_penalty,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (vocab_size + block - 1) / block;
    apply_repetition_penalty_kernel<<<grid, block, 0, stream>>>(
        logits, context, counts, vocab_size,
        rep_penalty, freq_penalty, pres_penalty
    );
}

} // namespace kernels
} // namespace helios
