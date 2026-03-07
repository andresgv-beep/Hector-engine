// kernels/normalization.cu
// ============================================================================
// NORMALIZATION KERNELS - RMSNorm, LayerNorm
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

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
        float w = __half2float(weight[i]);
        float normalized = val * rms_inv * w;
        row_out[i] = __float2half(normalized);
    }
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
    if (batch_size == 0 || dim == 0) return;
    
    int block_size = min(256, dim);
    block_size = max(32, block_size);  // At least one warp
    
    size_t smem_size = block_size * sizeof(float);
    
    rmsnorm_fp16_kernel<<<batch_size, block_size, smem_size, stream>>>(
        input, weight, output, batch_size, dim, eps
    );
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
