// kernels/quantize_q8.cuh
// ============================================================================
// QUANTIZE INPUT FP16 → Q8 (per-group symmetric quantization)
// ============================================================================
//
// NOTE: This header is included inside namespace helios::kernels
// Do NOT wrap in additional namespace declarations.
//

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int Q8_GROUP_SIZE = 8;  // Match HQS GROUP_SIZE

// Quantize FP16 input to Q8 in shared memory (cooperative, all threads)
__device__ __forceinline__
void quantize_input_q8(
    const half* __restrict__ input,
    int8_t* s_values,
    float*  s_scales,
    float*  s_group_sums,
    int K,
    int block_size
) {
    int num_groups = K / Q8_GROUP_SIZE;
    
    for (int g = threadIdx.x; g < num_groups; g += block_size) {
        int base = g * Q8_GROUP_SIZE;
        
        float vals[8];
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            vals[i] = __half2float(input[base + i]);
            amax = fmaxf(amax, fabsf(vals[i]));
        }
        
        float scale = amax / 127.0f;
        float inv_scale = (amax > 0.0f) ? 127.0f / amax : 0.0f;
        s_scales[g] = scale;
        
        int group_sum = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int q = __float2int_rn(vals[i] * inv_scale);
            q = max(-127, min(127, q));
            s_values[base + i] = static_cast<int8_t>(q);
            group_sum += q;
        }
        s_group_sums[g] = static_cast<float>(group_sum);
    }
}
