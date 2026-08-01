#include "kernels.hpp"
#include "cublas_context.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>

namespace helios {
namespace kernels {
namespace {

__global__ void gemma4_patch_input_fp16_kernel(
    const float* __restrict__ input,
    half* __restrict__ output,
    size_t elements) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        // Preserve upstream's operation order before the FP16 cast.
        output[index] = __float2half((input[index] - 0.5f) * 2.0f);
    }
}

__global__ void gemma4_add_xy_position_fp16_kernel(
    half* __restrict__ hidden,
    const half* __restrict__ table,
    const int32_t* __restrict__ positions,
    uint32_t patches,
    uint32_t hidden_size,
    uint32_t position_embedding_size) {
    const size_t element = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(patches) * hidden_size;
    if (element >= total) return;

    const uint32_t patch = static_cast<uint32_t>(element / hidden_size);
    const uint32_t channel = static_cast<uint32_t>(element % hidden_size);
    const int32_t x = positions[size_t(patch) * 2];
    const int32_t y = positions[size_t(patch) * 2 + 1];
    if (x < 0 || y < 0) return;

    const size_t x_index = size_t(x) * hidden_size + channel;
    const size_t y_index =
        (size_t(position_embedding_size) + size_t(y)) * hidden_size + channel;
    const half position = __hadd(table[x_index], table[y_index]);
    hidden[element] = __hadd(hidden[element], position);
}

__global__ void clamp_tensor_bounds_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ minimum,
    const half* __restrict__ maximum,
    half* __restrict__ output,
    size_t elements) {
    const float lower = __half2float(minimum[0]);
    const float upper = __half2float(maximum[0]);
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    for (; index < elements; index += stride) {
        const float value = __half2float(input[index]);
        output[index] = __float2half(fminf(upper, fmaxf(lower, value)));
    }
}

__global__ void gemma4_vision_norm_rope_transpose_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const int32_t* __restrict__ positions,
    half* __restrict__ output,
    uint32_t patches,
    uint32_t heads,
    uint32_t head_dim,
    float eps,
    float theta,
    bool apply_rope) {
    const uint32_t row = blockIdx.x;
    if (row >= patches * heads) return;
    const uint32_t patch = row / heads;
    const uint32_t head = row % heads;
    const uint32_t channel = threadIdx.x;
    extern __shared__ float shared[];
    float* sums = shared;
    half* normalized = reinterpret_cast<half*>(shared + blockDim.x);

    const size_t input_base = (size_t(patch) * heads + head) * head_dim;
    float sum_squares = 0.0f;
    if (channel < head_dim) {
        const float value = __half2float(input[input_base + channel]);
        sum_squares = value * value;
    }
    sums[channel] = sum_squares;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (channel < stride) sums[channel] += sums[channel + stride];
        __syncthreads();
    }
    if (channel < head_dim) {
        const float inverse_rms = rsqrtf(sums[0] / float(head_dim) + eps);
        float value = __half2float(input[input_base + channel]) * inverse_rms;
        if (weight) value *= __half2float(weight[channel]);
        normalized[channel] = __float2half(value);
    }
    __syncthreads();

    if (channel < head_dim) {
        half value = normalized[channel];
        if (apply_rope) {
            const uint32_t spatial_width = head_dim / 2;
            const uint32_t spatial_axis = channel / spatial_width;
            const uint32_t local = channel % spatial_width;
            const uint32_t half_width = spatial_width / 2;
            const uint32_t pair_local = local < half_width
                ? local + half_width : local - half_width;
            const uint32_t pair_channel =
                spatial_axis * spatial_width + pair_local;
            half paired = normalized[pair_channel];
            if (local < half_width) paired = __hneg(paired);

            const uint32_t frequency = local % half_width;
            const float exponent = float(2 * frequency) / float(spatial_width);
            const int32_t position = positions[size_t(patch) * 2 + spatial_axis];
            const float angle = float(position) * powf(theta, -exponent);
            const half cosine = __float2half(cosf(angle));
            const half sine = __float2half(sinf(angle));
            value = __hadd(__hmul(value, cosine), __hmul(paired, sine));
        }
        const size_t output_index =
            (size_t(head) * patches + patch) * head_dim + channel;
        output[output_index] = value;
    }
}

__global__ void gemma4_vision_softmax_fp16_kernel(
    half* __restrict__ scores,
    const int32_t* __restrict__ positions,
    uint32_t patches) {
    const uint32_t row = blockIdx.x;
    half* row_scores = scores + size_t(row) * patches;
    extern __shared__ float reduction[];

    float local_max = -INFINITY;
    for (uint32_t key = threadIdx.x; key < patches; key += blockDim.x) {
        const bool valid = positions[size_t(key) * 2] >= 0 &&
                           positions[size_t(key) * 2 + 1] >= 0;
        if (valid) local_max = fmaxf(local_max, __half2float(row_scores[key]));
    }
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float maximum = reduction[0];

    float local_sum = 0.0f;
    for (uint32_t key = threadIdx.x; key < patches; key += blockDim.x) {
        const bool valid = positions[size_t(key) * 2] >= 0 &&
                           positions[size_t(key) * 2 + 1] >= 0;
        if (valid) local_sum += expf(__half2float(row_scores[key]) - maximum);
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inverse_sum = reduction[0] > 0.0f
        ? 1.0f / reduction[0] : 0.0f;
    for (uint32_t key = threadIdx.x; key < patches; key += blockDim.x) {
        const bool valid = positions[size_t(key) * 2] >= 0 &&
                           positions[size_t(key) * 2 + 1] >= 0;
        const float probability = valid
            ? expf(__half2float(row_scores[key]) - maximum) * inverse_sum
            : 0.0f;
        row_scores[key] = __float2half(probability);
    }
}

__global__ void gemma4_vision_head_to_token_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    uint32_t patches,
    uint32_t heads,
    uint32_t head_dim) {
    const size_t total = size_t(patches) * heads * head_dim;
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    for (; index < total; index += stride) {
        const uint32_t channel = index % head_dim;
        const uint32_t head = (index / head_dim) % heads;
        const uint32_t patch = index / (size_t(head_dim) * heads);
        output[index] = input[(size_t(head) * patches + patch) * head_dim + channel];
    }
}

__global__ void gemma4_vision_pool3x3_fp32_kernel(
    const half* __restrict__ hidden,
    float* __restrict__ pooled,
    uint32_t patch_columns,
    uint32_t patch_rows,
    uint32_t hidden_size) {
    const uint32_t pooled_columns = patch_columns / 3;
    const uint32_t pooled_rows = patch_rows / 3;
    const size_t total = size_t(pooled_columns) * pooled_rows * hidden_size;
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    for (; index < total; index += stride) {
        const uint32_t channel = index % hidden_size;
        const uint32_t pooled_patch = index / hidden_size;
        const uint32_t pooled_x = pooled_patch % pooled_columns;
        const uint32_t pooled_y = pooled_patch / pooled_columns;
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t dy = 0; dy < 3; ++dy) {
            #pragma unroll
            for (uint32_t dx = 0; dx < 3; ++dx) {
                const uint32_t x = pooled_x * 3 + dx;
                const uint32_t y = pooled_y * 3 + dy;
                const size_t source =
                    (size_t(y) * patch_columns + x) * hidden_size + channel;
                sum += __half2float(hidden[source]);
            }
        }
        // Upstream casts the FP32 average back to the visual dtype before its
        // FP32 sqrt(hidden) multiplication.
        const half average = __float2half(sum * (1.0f / 9.0f));
        pooled[index] = __half2float(average) * sqrtf(float(hidden_size));
    }
}

__global__ void fp32_to_fp16_kernel(
    const float* __restrict__ input,
    half* __restrict__ output,
    size_t elements) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    for (; index < elements; index += stride) {
        output[index] = __float2half(input[index]);
    }
}

} // namespace

void launch_gemma4_patch_input_fp16(
    const float* input,
    half* output,
    size_t elements,
    cudaStream_t stream) {
    if (elements == 0) return;
    constexpr uint32_t block = 256;
    const uint32_t grid = static_cast<uint32_t>((elements + block - 1) / block);
    gemma4_patch_input_fp16_kernel<<<grid, block, 0, stream>>>(
        input, output, elements);
}

void launch_gemma4_add_xy_position_fp16(
    half* hidden,
    const half* table,
    const int32_t* positions,
    uint32_t patches,
    uint32_t hidden_size,
    uint32_t position_embedding_size,
    cudaStream_t stream) {
    const size_t elements = size_t(patches) * hidden_size;
    if (elements == 0) return;
    constexpr uint32_t block = 256;
    const uint32_t grid = static_cast<uint32_t>((elements + block - 1) / block);
    gemma4_add_xy_position_fp16_kernel<<<grid, block, 0, stream>>>(
        hidden, table, positions, patches, hidden_size,
        position_embedding_size);
}

void launch_clamp_tensor_bounds_fp16(
    const half* input,
    const half* minimum,
    const half* maximum,
    half* output,
    size_t elements,
    cudaStream_t stream) {
    if (elements == 0) return;
    constexpr uint32_t block = 256;
    uint32_t grid = static_cast<uint32_t>((elements + block - 1) / block);
    grid = min(grid, 65535u);
    clamp_tensor_bounds_fp16_kernel<<<grid, block, 0, stream>>>(
        input, minimum, maximum, output, elements);
}

void launch_gemma4_vision_norm_rope_transpose_fp16(
    const half* input,
    const half* weight,
    const int32_t* positions,
    half* output_head_major,
    uint32_t patches,
    uint32_t heads,
    uint32_t head_dim,
    float eps,
    float theta,
    bool apply_rope,
    cudaStream_t stream) {
    if (patches == 0 || heads == 0 || head_dim == 0) return;
    uint32_t block = 1;
    while (block < head_dim) block <<= 1;
    block = min(block, 256u);
    const size_t shared = block * sizeof(float) + head_dim * sizeof(half);
    gemma4_vision_norm_rope_transpose_fp16_kernel<<<
        patches * heads, block, shared, stream>>>(
        input, weight, positions, output_head_major, patches, heads,
        head_dim, eps, theta, apply_rope);
}

bool launch_gemma4_vision_attention_fp16(
    const half* q_head_major,
    const half* k_head_major,
    const half* v_head_major,
    const int32_t* positions,
    half* scores,
    half* output_head_major,
    uint32_t patches,
    uint32_t heads,
    uint32_t head_dim,
    float scale,
    cudaStream_t stream) {
    if (patches == 0 || heads == 0 || head_dim == 0) return false;
    cublasHandle_t handle = cublas_handle_for_stream(stream);
    if (!handle) return false;
    const half alpha = __float2half(scale);
    const half one = __float2half(1.0f);
    const half zero = __float2half(0.0f);
    const long long head_stride = static_cast<long long>(patches) * head_dim;
    const long long score_stride = static_cast<long long>(patches) * patches;

    // Row-major Q @ K.T. cuBLAS sees the transposed column-major expression
    // K @ Q.T, which has the same byte layout as the desired row-major score.
    cublasStatus_t status = cublasHgemmStridedBatched(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int>(patches), static_cast<int>(patches),
        static_cast<int>(head_dim), &alpha,
        k_head_major, static_cast<int>(head_dim), head_stride,
        q_head_major, static_cast<int>(head_dim), head_stride,
        &zero, scores, static_cast<int>(patches), score_stride,
        static_cast<int>(heads));
    if (status != CUBLAS_STATUS_SUCCESS) return false;

    constexpr uint32_t softmax_block = 256;
    gemma4_vision_softmax_fp16_kernel<<<
        heads * patches, softmax_block,
        softmax_block * sizeof(float), stream>>>(
        scores, positions, patches);

    // Row-major probabilities @ V, expressed as V.T @ probabilities.T in
    // cuBLAS column-major form. The result bytes are [heads, patches, dim].
    status = cublasHgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(head_dim), static_cast<int>(patches),
        static_cast<int>(patches), &one,
        v_head_major, static_cast<int>(head_dim), head_stride,
        scores, static_cast<int>(patches), score_stride,
        &zero, output_head_major, static_cast<int>(head_dim), head_stride,
        static_cast<int>(heads));
    return status == CUBLAS_STATUS_SUCCESS;
}

void launch_gemma4_vision_head_to_token_fp16(
    const half* input_head_major,
    half* output_token_major,
    uint32_t patches,
    uint32_t heads,
    uint32_t head_dim,
    cudaStream_t stream) {
    const size_t elements = size_t(patches) * heads * head_dim;
    if (elements == 0) return;
    constexpr uint32_t block = 256;
    uint32_t grid = static_cast<uint32_t>((elements + block - 1) / block);
    grid = min(grid, 65535u);
    gemma4_vision_head_to_token_fp16_kernel<<<grid, block, 0, stream>>>(
        input_head_major, output_token_major, patches, heads, head_dim);
}

void launch_gemma4_vision_pool3x3_fp32(
    const half* hidden,
    float* pooled,
    uint32_t patch_columns,
    uint32_t patch_rows,
    uint32_t hidden_size,
    cudaStream_t stream) {
    const size_t elements =
        size_t(patch_columns / 3) * (patch_rows / 3) * hidden_size;
    if (elements == 0) return;
    constexpr uint32_t block = 256;
    uint32_t grid = static_cast<uint32_t>((elements + block - 1) / block);
    grid = min(grid, 65535u);
    gemma4_vision_pool3x3_fp32_kernel<<<grid, block, 0, stream>>>(
        hidden, pooled, patch_columns, patch_rows, hidden_size);
}

void launch_fp32_to_fp16(
    const float* input,
    half* output,
    size_t elements,
    cudaStream_t stream) {
    if (elements == 0) return;
    constexpr uint32_t block = 256;
    uint32_t grid = static_cast<uint32_t>((elements + block - 1) / block);
    grid = min(grid, 65535u);
    fp32_to_fp16_kernel<<<grid, block, 0, stream>>>(input, output, elements);
}

} // namespace kernels
} // namespace helios
