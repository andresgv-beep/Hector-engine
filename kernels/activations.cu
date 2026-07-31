// kernels/activations.cu
// ============================================================================
// ACTIVATION KERNELS - SiLU, GELU, Softmax
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace helios {
namespace kernels {

constexpr int ACTIVATION_BLOCK_SIZE = 256;

// ============================================================================
// SILU KERNEL (x * sigmoid(x))
// ============================================================================

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void silu_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < numel; i += stride) {
        float x = __half2float(input[i]);
        float y = silu_f32(x);
        output[i] = __float2half(y);
    }
}

void launch_silu_fp16(
    const half* input, half* output,
    size_t numel,
    cudaStream_t stream
) {
    if (numel == 0) return;
    
    int num_blocks = (numel + ACTIVATION_BLOCK_SIZE - 1) / ACTIVATION_BLOCK_SIZE;
    num_blocks = min(num_blocks, 65535);
    
    silu_fp16_kernel<<<num_blocks, ACTIVATION_BLOCK_SIZE, 0, stream>>>(
        input, output, numel
    );
}

// ============================================================================
// GELU KERNEL (approximate: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))))
// ============================================================================

__device__ __forceinline__ float gelu_f32(float x) {
    const float c = 0.7978845608f;  // sqrt(2/pi)
    const float a = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + a * x3)));
}

__global__ void gelu_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < numel; i += stride) {
        float x = __half2float(input[i]);
        float y = gelu_f32(x);
        output[i] = __float2half(y);
    }
}

void launch_gelu_fp16(
    const half* input, half* output,
    size_t numel,
    cudaStream_t stream
) {
    if (numel == 0) return;
    
    int num_blocks = (numel + ACTIVATION_BLOCK_SIZE - 1) / ACTIVATION_BLOCK_SIZE;
    num_blocks = min(num_blocks, 65535);
    
    gelu_fp16_kernel<<<num_blocks, ACTIVATION_BLOCK_SIZE, 0, stream>>>(
        input, output, numel
    );
}

// ============================================================================
// SOFTMAX KERNEL (row-wise)
// ============================================================================

__global__ void softmax_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int batch_size,
    int seq_len
) {
    // Each block handles one row
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    extern __shared__ float smem[];
    
    const half* row_in = input + row * seq_len;
    half* row_out = output + row * seq_len;
    
    // Step 1: Find max (for numerical stability)
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        thread_max = fmaxf(thread_max, __half2float(row_in[i]));
    }
    
    // Reduce max across threads
    smem[threadIdx.x] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();
    
    // Step 2: Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float val = expf(__half2float(row_in[i]) - row_max);
        smem[blockDim.x + i] = val;  // Store for later
        thread_sum += val;
    }
    
    // Reduce sum
    smem[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    float row_sum = smem[0];
    __syncthreads();
    
    // Step 3: Normalize
    float inv_sum = 1.0f / (row_sum + 1e-6f);
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float val = smem[blockDim.x + i] * inv_sum;
        row_out[i] = __float2half(val);
    }
}

void launch_softmax_fp16(
    const half* input, half* output,
    int batch_size, int seq_len,
    cudaStream_t stream
) {
    if (batch_size == 0 || seq_len == 0) return;
    
    int block_size = min(256, seq_len);
    // Shared memory: block_size for reduction + seq_len for exp values
    size_t smem_size = (block_size + seq_len) * sizeof(float);
    
    softmax_fp16_kernel<<<batch_size, block_size, smem_size, stream>>>(
        input, output, batch_size, seq_len
    );
}

// ============================================================================
// FUSED SILU×MUL KERNEL — silu(gate) * up in one pass
// ============================================================================
// Replaces: silu(gate) → gate_act; mul(gate_act, up) → output
// Saves: 1 kernel launch + 1 intermediate buffer read/write

__global__ void silu_mul_fp16_kernel(
    const half* __restrict__ gate,    // [numel]
    const half* __restrict__ up,      // [numel]
    half* __restrict__ output,        // [numel]
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < numel; i += stride) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        float y = silu_f32(g) * u;
        output[i] = __float2half(y);
    }
}

void launch_silu_mul_fp16(
    const half* gate, const half* up, half* output,
    size_t numel,
    cudaStream_t stream
) {
    if (numel == 0) return;
    int num_blocks = (numel + ACTIVATION_BLOCK_SIZE - 1) / ACTIVATION_BLOCK_SIZE;
    num_blocks = min(num_blocks, 65535);
    silu_mul_fp16_kernel<<<num_blocks, ACTIVATION_BLOCK_SIZE, 0, stream>>>(
        gate, up, output, numel
    );
}

// ============================================================================
// FUSED GELU×MUL KERNEL — gelu(gate) * up in one pass
// ============================================================================

__global__ void gelu_mul_fp16_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < numel; i += stride) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        float y = gelu_f32(g) * u;
        output[i] = __float2half(y);
    }
}

void launch_gelu_mul_fp16(
    const half* gate, const half* up, half* output,
    size_t numel,
    cudaStream_t stream
) {
    if (numel == 0) return;
    int num_blocks = (numel + ACTIVATION_BLOCK_SIZE - 1) / ACTIVATION_BLOCK_SIZE;
    num_blocks = min(num_blocks, 65535);
    gelu_mul_fp16_kernel<<<num_blocks, ACTIVATION_BLOCK_SIZE, 0, stream>>>(
        gate, up, output, numel
    );
}

// Gemma final-logit softcap: tanh(x / cap) * cap.
__global__ void softcap_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    float cap,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        const float x = __half2float(input[i]);
        output[i] = __float2half(tanhf(x / cap) * cap);
    }
}

void launch_softcap_fp16(
    const half* input, half* output,
    float cap, size_t numel,
    cudaStream_t stream
) {
    if (numel == 0 || cap <= 0.0f) return;
    int num_blocks = (numel + ACTIVATION_BLOCK_SIZE - 1) / ACTIVATION_BLOCK_SIZE;
    num_blocks = min(num_blocks, 65535);
    softcap_fp16_kernel<<<num_blocks, ACTIVATION_BLOCK_SIZE, 0, stream>>>(
        input, output, cap, numel
    );
}

} // namespace kernels
} // namespace helios
