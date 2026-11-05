// kernels/normalization.cu
// ============================================================================
// NORMALIZATION KERNELS - RMSNorm, LayerNorm
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>

namespace helios {
namespace kernels {

// ============================================================================
// RMSNORM KERNEL
// ============================================================================
// output = (input / sqrt(mean(input^2) + eps)) * weight
//

__global__ void rmsnorm_fp16_kernel(
    const half* __restrict__ input,   // [batch, dim]
    const half* __restrict__ weight,  // [dim]
    half* __restrict__ output,        // [batch, dim]
    int batch_size,
    int dim,
    float eps
) {
    // Each block handles one row
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    extern __shared__ float smem[];
    
    const half* row_in = input + row * dim;
    half* row_out = output + row * dim;
    
    // Step 1: Compute sum of squares
    float thread_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(row_in[i]);
        thread_sum_sq += val * val;
    }
    
    // Reduce sum across threads
    smem[threadIdx.x] = thread_sum_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float mean_sq = smem[0] / float(dim);
    float rms_inv = rsqrtf(mean_sq + eps);
    
    // Step 2: Normalize and apply weight
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(row_in[i]);
        float w = weight ? __half2float(weight[i]) : 1.0f;
        float normalized = val * rms_inv * w;
        row_out[i] = __float2half(normalized);
    }
}

// ============================================================================
// RMSNORM VECTORIZADO
// ============================================================================
// En decode la fila es una sola (hidden = 2560 en el E4B) y el kernel de arriba
// la recorre con un único bloque: diez lecturas escalares por hilo, otra pasada
// entera para escribir y ocho __syncthreads de árbol. Medido: 9,9 µs por
// llamada y ~211 llamadas por token, unos 2 ms de los 16. Esta versión lee cada
// hilo 8 halves de una vez, los guarda en registros para no releer la fila, y
// reduce por warp con shuffles. Sin __restrict__: el grafo normaliza V en sitio.
//
// VPT = vectores de 8 halves por hilo. Exige dim % 8 == 0 y punteros alineados a
// 16 bytes; si no, se usa el kernel de arriba.

template<int VPT>
__global__ void rmsnorm_fp16_vec_kernel(
    const half* input,
    const half* weight,
    half* output,
    int dim,
    float eps
) {
    const int nvec = dim / 8;
    const uint4* row_in = reinterpret_cast<const uint4*>(input + size_t(blockIdx.x) * dim);
    uint4* row_out = reinterpret_cast<uint4*>(output + size_t(blockIdx.x) * dim);

    uint4 buf[VPT];
    float sum_sq = 0.0f;
#pragma unroll
    for (int v = 0; v < VPT; ++v) {
        const int idx = threadIdx.x + v * blockDim.x;
        if (idx < nvec) {
            buf[v] = row_in[idx];
            const half2* h = reinterpret_cast<const half2*>(&buf[v]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __half22float2(h[j]);
                sum_sq += f.x * f.x + f.y * f.y;
            }
        }
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, off);

    __shared__ float warp_sums[32];
    __shared__ float rms_inv_shared;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int nwarps = (blockDim.x + 31) / 32;
    if (nwarps > 1) {
        if (lane == 0) warp_sums[warp] = sum_sq;
        __syncthreads();
        if (warp == 0) {
            float s = lane < nwarps ? warp_sums[lane] : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                s += __shfl_xor_sync(0xffffffffu, s, off);
            if (lane == 0) rms_inv_shared = rsqrtf(s / float(dim) + eps);
        }
        __syncthreads();
    } else if (lane == 0) {
        rms_inv_shared = rsqrtf(sum_sq / float(dim) + eps);
    }
    __syncwarp();
    const float rms_inv = rms_inv_shared;

    const uint4* w4 = reinterpret_cast<const uint4*>(weight);
#pragma unroll
    for (int v = 0; v < VPT; ++v) {
        const int idx = threadIdx.x + v * blockDim.x;
        if (idx < nvec) {
            uint4 res;
            const half2* h = reinterpret_cast<const half2*>(&buf[v]);
            half2* r = reinterpret_cast<half2*>(&res);
            if (weight) {
                const uint4 wv = w4[idx];
                const half2* w = reinterpret_cast<const half2*>(&wv);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __half22float2(h[j]);
                    const float2 g = __half22float2(w[j]);
                    r[j] = __floats2half2_rn(f.x * rms_inv * g.x, f.y * rms_inv * g.y);
                }
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __half22float2(h[j]);
                    r[j] = __floats2half2_rn(f.x * rms_inv, f.y * rms_inv);
                }
            }
            row_out[idx] = res;
        }
    }
}

static bool aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

static void launch_rmsnorm_any(
    const half* input,
    const half* weight,
    half* output,
    int batch_size,
    int dim,
    float eps,
    cudaStream_t stream
) {
    if (batch_size == 0 || dim == 0) return;

    const int nvec = dim / 8;
    if (dim % 8 == 0 && nvec <= 4 * 1024 && aligned16(input) && aligned16(output) &&
        (!weight || aligned16(weight))) {
        const int vpt = (nvec + 1023) / 1024;
        const int threads = ((nvec + vpt - 1) / vpt + 31) / 32 * 32;
        switch (vpt) {
            case 1: rmsnorm_fp16_vec_kernel<1><<<batch_size, threads, 0, stream>>>(input, weight, output, dim, eps); break;
            case 2: rmsnorm_fp16_vec_kernel<2><<<batch_size, threads, 0, stream>>>(input, weight, output, dim, eps); break;
            default: rmsnorm_fp16_vec_kernel<4><<<batch_size, threads, 0, stream>>>(input, weight, output, dim, eps); break;
        }
        return;
    }

    // Potencia de dos: la reducción en árbol pierde elementos con 100 hilos.
    int block_size = 32;
    while (block_size < dim && block_size < 256) block_size *= 2;

    size_t smem_size = block_size * sizeof(float);

    rmsnorm_fp16_kernel<<<batch_size, block_size, smem_size, stream>>>(
        input, weight, output, batch_size, dim, eps
    );
}

void launch_rmsnorm_fp16(
    const half* input,
    const half* weight,
    half* output,
    int batch_size,
    int dim,
    float eps,
    cudaStream_t stream
) {
    launch_rmsnorm_any(input, weight, output, batch_size, dim, eps, stream);
}

void launch_rmsnorm_no_weight_fp16(
    const half* input,
    half* output,
    int batch_size,
    int dim,
    float eps,
    cudaStream_t stream
) {
    launch_rmsnorm_any(input, nullptr, output, batch_size, dim, eps, stream);
}

// ============================================================================
// LAYERNORM KERNEL
// ============================================================================
// output = (input - mean) / sqrt(var + eps) * weight + bias
//

__global__ void layernorm_fp16_kernel(
    const half* __restrict__ input,   // [batch, dim]
    const half* __restrict__ weight,  // [dim]
    const half* __restrict__ bias,    // [dim] or nullptr
    half* __restrict__ output,        // [batch, dim]
    int batch_size,
    int dim,
    float eps,
    bool has_bias
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    extern __shared__ float smem[];
    float* smem_sum = smem;
    float* smem_sq = smem + blockDim.x;
    
    const half* row_in = input + row * dim;
    half* row_out = output + row * dim;
    
    // Step 1: Compute sum and sum of squares
    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;
    
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(row_in[i]);
        thread_sum += val;
        thread_sum_sq += val * val;
    }
    
    smem_sum[threadIdx.x] = thread_sum;
    smem_sq[threadIdx.x] = thread_sum_sq;
    __syncthreads();
    
    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem_sum[threadIdx.x] += smem_sum[threadIdx.x + s];
            smem_sq[threadIdx.x] += smem_sq[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float mean = smem_sum[0] / float(dim);
    float var = smem_sq[0] / float(dim) - mean * mean;
    float inv_std = rsqrtf(var + eps);
    
    // Step 2: Normalize and apply weight/bias
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(row_in[i]);
        float w = __half2float(weight[i]);
        float normalized = (val - mean) * inv_std * w;
        
        if (has_bias && bias != nullptr) {
            normalized += __half2float(bias[i]);
        }
        
        row_out[i] = __float2half(normalized);
    }
}

void launch_layernorm_fp16(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    int batch_size,
    int dim,
    float eps,
    cudaStream_t stream
) {
    if (batch_size == 0 || dim == 0) return;
    
    int block_size = min(256, dim);
    block_size = max(32, block_size);
    
    // Need space for sum and sum_sq
    size_t smem_size = 2 * block_size * sizeof(float);
    
    layernorm_fp16_kernel<<<batch_size, block_size, smem_size, stream>>>(
        input, weight, bias, output, batch_size, dim, eps, bias != nullptr
    );
}

} // namespace kernels
} // namespace helios
