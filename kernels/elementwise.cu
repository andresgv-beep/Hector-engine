// kernels/elementwise.cu
// ============================================================================
// ELEMENTWISE KERNELS - Add, Mul, Scale, Copy
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace helios {
namespace kernels {

// ============================================================================
// CONFIGURATION
// ============================================================================

constexpr int ELEMENTWISE_BLOCK_SIZE = 256;
constexpr int ELEMENTS_PER_THREAD = 4;

// ============================================================================
// ADD KERNEL
// ============================================================================

__global__ void add_fp16_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Vectorized load/store when possible
    size_t vec_numel = numel / 2;
    const half2* a2 = reinterpret_cast<const half2*>(a);
    const half2* b2 = reinterpret_cast<const half2*>(b);
    half2* c2 = reinterpret_cast<half2*>(c);
    
    for (size_t i = idx; i < vec_numel; i += stride) {
        half2 va = a2[i];
        half2 vb = b2[i];
        c2[i] = __hadd2(va, vb);
    }
    
    // Handle remainder
    if (idx == 0 && (numel % 2) == 1) {
        c[numel - 1] = __hadd(a[numel - 1], b[numel - 1]);
    }
}

void launch_add_fp16(
    const half* a, const half* b, half* c,
    size_t numel,
    cudaStream_t stream
) {
    if (numel == 0) return;
    
    int num_blocks = (numel / 2 + ELEMENTWISE_BLOCK_SIZE - 1) / ELEMENTWISE_BLOCK_SIZE;
    num_blocks = min(num_blocks, 65535);
    
    add_fp16_kernel<<<num_blocks, ELEMENTWISE_BLOCK_SIZE, 0, stream>>>(
        a, b, c, numel
    );
}

// ============================================================================
// MUL KERNEL
// ============================================================================

__global__ void mul_fp16_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    size_t vec_numel = numel / 2;
    const half2* a2 = reinterpret_cast<const half2*>(a);
    const half2* b2 = reinterpret_cast<const half2*>(b);
    half2* c2 = reinterpret_cast<half2*>(c);
    
    for (size_t i = idx; i < vec_numel; i += stride) {
        half2 va = a2[i];
        half2 vb = b2[i];
        c2[i] = __hmul2(va, vb);
    }
    
    if (idx == 0 && (numel % 2) == 1) {
        c[numel - 1] = __hmul(a[numel - 1], b[numel - 1]);
    }
}

void launch_mul_fp16(
    const half* a, const half* b, half* c,
    size_t numel,
    cudaStream_t stream
) {
    if (numel == 0) return;
    
    int num_blocks = (numel / 2 + ELEMENTWISE_BLOCK_SIZE - 1) / ELEMENTWISE_BLOCK_SIZE;
    num_blocks = min(num_blocks, 65535);
    
    mul_fp16_kernel<<<num_blocks, ELEMENTWISE_BLOCK_SIZE, 0, stream>>>(
        a, b, c, numel
    );
}

// ============================================================================
// SCALE KERNEL
// ============================================================================

__global__ void scale_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    half scalar,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    half2 scalar2 = __half2half2(scalar);
    
    size_t vec_numel = numel / 2;
    const half2* in2 = reinterpret_cast<const half2*>(input);
    half2* out2 = reinterpret_cast<half2*>(output);
    
    for (size_t i = idx; i < vec_numel; i += stride) {
        out2[i] = __hmul2(in2[i], scalar2);
    }
    
    if (idx == 0 && (numel % 2) == 1) {
        output[numel - 1] = __hmul(input[numel - 1], scalar);
    }
}

void launch_scale_fp16(
    const half* input, half* output,
    float scalar,
    size_t numel,
    cudaStream_t stream
) {
    if (numel == 0) return;
    
    int num_blocks = (numel / 2 + ELEMENTWISE_BLOCK_SIZE - 1) / ELEMENTWISE_BLOCK_SIZE;
    num_blocks = min(num_blocks, 65535);
    
    half scalar_h = __float2half(scalar);
    
    scale_fp16_kernel<<<num_blocks, ELEMENTWISE_BLOCK_SIZE, 0, stream>>>(
        input, output, scalar_h, numel
    );
}

// ============================================================================
// COPY KERNEL
// ============================================================================

__global__ void copy_fp16_kernel(
    const half* __restrict__ src,
    half* __restrict__ dst,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Use 128-bit loads when possible (8 halfs)
    size_t vec_numel = numel / 8;
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    
    for (size_t i = idx; i < vec_numel; i += stride) {
        dst4[i] = src4[i];
    }
    
    // Handle remainder
    size_t remainder_start = vec_numel * 8;
    for (size_t i = remainder_start + idx; i < numel; i += stride) {
        dst[i] = src[i];
    }
}

void launch_copy_fp16(
    const half* src, half* dst,
    size_t numel,
    cudaStream_t stream
) {
    if (numel == 0) return;
    
    int num_blocks = (numel / 8 + ELEMENTWISE_BLOCK_SIZE - 1) / ELEMENTWISE_BLOCK_SIZE;
    num_blocks = max(1, min(num_blocks, 65535));
    
    copy_fp16_kernel<<<num_blocks, ELEMENTWISE_BLOCK_SIZE, 0, stream>>>(
        src, dst, numel
    );
}

} // namespace kernels
} // namespace helios

// ============================================================================
// ADD BIAS KERNEL (broadcast 1D bias to 2D/3D tensor)
// ============================================================================
// output[..., i] = input[..., i] + bias[i]
// input: [batch, seq, dim] or [batch, dim]
// bias: [dim]
// output: same as input

namespace helios {
namespace kernels {

__global__ void add_bias_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ bias,
    half* output,  // No __restrict__ - puede ser == input
    size_t total_elements,
    size_t dim
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < total_elements; i += stride) {
        size_t bias_idx = i % dim;
        half val = input[i];
        half b = bias[bias_idx];
        output[i] = __hadd(val, b);
    }
}

void launch_add_bias_fp16(
    const half* input,
    const half* bias,
    half* output,
    size_t total_elements,
    size_t dim,
    cudaStream_t stream
) {
    if (total_elements == 0) return;
    
    int num_blocks = (total_elements + ELEMENTWISE_BLOCK_SIZE - 1) / ELEMENTWISE_BLOCK_SIZE;
    num_blocks = min(num_blocks, 65535);
    
    add_bias_fp16_kernel<<<num_blocks, ELEMENTWISE_BLOCK_SIZE, 0, stream>>>(
        input, bias, output, total_elements, dim
    );
}

} // namespace kernels
} // namespace helios

// ============================================================================
// SPLIT KERNELS — single-launch replacements for cudaMemcpy loops
// ============================================================================

namespace helios {
namespace kernels {

__global__ void split_3way_kernel(
    const half* __restrict__ src,
    half* __restrict__ dst_a,
    half* __restrict__ dst_b,
    half* __restrict__ dst_c,
    int fused_dim, int a_size, int b_size, int c_size,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    int row = idx / fused_dim;
    int col = idx % fused_dim;
    half val = src[idx];
    
    if (col < a_size) {
        dst_a[row * a_size + col] = val;
    } else if (col < a_size + b_size) {
        dst_b[row * b_size + (col - a_size)] = val;
    } else {
        dst_c[row * c_size + (col - a_size - b_size)] = val;
    }
}

__global__ void split_2way_kernel(
    const half* __restrict__ src,
    half* __restrict__ dst_a,
    half* __restrict__ dst_b,
    int fused_dim, int a_size,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    int row = idx / fused_dim;
    int col = idx % fused_dim;
    half val = src[idx];
    
    if (col < a_size) {
        dst_a[row * a_size + col] = val;
    } else {
        int b_size = fused_dim - a_size;
        dst_b[row * b_size + (col - a_size)] = val;
    }
}

void launch_split_3way(
    const half* src,
    half* dst_a, half* dst_b, half* dst_c,
    int batch_seq, int a_size, int b_size, int c_size,
    cudaStream_t stream
) {
    int fused_dim = a_size + b_size + c_size;
    int total = batch_seq * fused_dim;
    int blocks = (total + 255) / 256;
    split_3way_kernel<<<blocks, 256, 0, stream>>>(
        src, dst_a, dst_b, dst_c,
        fused_dim, a_size, b_size, c_size, total
    );
}

void launch_split_2way(
    const half* src,
    half* dst_a, half* dst_b,
    int batch_seq, int a_size, int b_size,
    cudaStream_t stream
) {
    int fused_dim = a_size + b_size;
    int total = batch_seq * fused_dim;
    int blocks = (total + 255) / 256;
    split_2way_kernel<<<blocks, 256, 0, stream>>>(
        src, dst_a, dst_b,
        fused_dim, a_size, total
    );
}

} // namespace kernels
} // namespace helios
