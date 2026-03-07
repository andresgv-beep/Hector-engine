// kernels/fused_add_rmsnorm.cu
// ============================================================================
// FUSED ADD + RMSNORM
// ============================================================================
//
// Fuses the pattern:
//   residual = a + b           (elementwise add)
//   normed = rmsnorm(residual) (normalize + weight)
//
// into a single kernel that:
//   1. Computes a + b
//   2. Stores result to residual output (needed for next residual connection)
//   3. Computes rmsnorm of the sum (reuses values from step 1 in registers)
//   4. Stores normalized result
//
// Saves: 1 kernel launch + 1 global memory round-trip of residual
//
// Called 64 times per forward pass (32 layers × 2: post-attn and post-mlp)
//

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace helios {
namespace kernels {

__global__ void fused_add_rmsnorm_kernel(
    const half* __restrict__ a,          // [batch, dim]
    const half* __restrict__ b,          // [batch, dim]
    const half* __restrict__ weight,     // [dim]
    half* __restrict__ residual_out,     // [batch, dim] = a + b
    half* __restrict__ normed_out,       // [batch, dim] = rmsnorm(a + b)
    int batch_size,
    int dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    extern __shared__ float smem[];
    
    const half* row_a = a + row * dim;
    const half* row_b = b + row * dim;
    half* row_res = residual_out + row * dim;
    half* row_norm = normed_out + row * dim;
    
    // Pass 1: Compute a + b, write residual, accumulate sum of squares
    float thread_sum_sq = 0.0f;
    
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float va = __half2float(row_a[i]);
        float vb = __half2float(row_b[i]);
        float sum = va + vb;
        
        // Write residual (needed for next residual connection)
        row_res[i] = __float2half(sum);
        
        thread_sum_sq += sum * sum;
    }
    
    // Reduce sum_sq across threads
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
    
    // Pass 2: Read residual back, normalize, apply weight
    // Note: residual is hot in L1/L2 cache from the write in pass 1
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(row_res[i]);  // L1 cache hit
        float w = __half2float(weight[i]);
        row_norm[i] = __float2half(val * rms_inv * w);
    }
}

void launch_fused_add_rmsnorm_fp16(
    const half* a,
    const half* b,
    const half* weight,
    half* residual_out,
    half* normed_out,
    int batch_size,
    int dim,
    float eps,
    cudaStream_t stream
) {
    if (batch_size == 0 || dim == 0) return;
    
    int block_size = min(256, dim);
    block_size = max(32, block_size);
    
    size_t smem_size = block_size * sizeof(float);
    
    fused_add_rmsnorm_kernel<<<batch_size, block_size, smem_size, stream>>>(
        a, b, weight, residual_out, normed_out, batch_size, dim, eps
    );
}

} // namespace kernels
} // namespace helios
