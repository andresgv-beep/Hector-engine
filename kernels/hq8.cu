// HQ8K prefill and embedding: dequantize to FP16, cuBLAS with FP32 accumulation, and
// row lookup. The decode GEMV and its tuner live in matmul_hqs_compact.cu.
#include "kernels.hpp"
#include "cublas_context.hpp"
#include "hq8_gemv.cuh"
#include "per_device.hpp"
#include <stdexcept>
namespace helios { namespace kernels {
namespace {
void checked(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }

// One thread per 8 weights, the same split as the GEMV lanes.
__global__ void dequant_hq8(const uint8_t* weights, half* output, int K, int N) {
    const int row = blockIdx.x;
    const int first = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
    if (row >= N || first >= K) return;
    const uint8_t* block = weights + (size_t(row) * (K / 256) + first / 256) * hq8::BLOCK_BYTES;
    const int in_block = first % 256;
    const float step = hq8::step_of(block, in_block / 32);
    const uint2 codes = *reinterpret_cast<const uint2*>(block + hq8::HEADER_BYTES + in_block);
    uint32_t out[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t word = i < 2 ? codes.x : codes.y;
        const int shift = (i & 1) * 16;
        const half lo = __float2half(step * float(int8_t(word >> shift)));
        const half hi = __float2half(step * float(int8_t(word >> (shift + 8))));
        out[i] = uint32_t(__half_as_ushort(lo)) | (uint32_t(__half_as_ushort(hi)) << 16);
    }
    *reinterpret_cast<uint4*>(output + size_t(row) * K + first) = make_uint4(out[0], out[1], out[2], out[3]);
}

__global__ void embedding_hq8(const int32_t* __restrict__ indices, const uint8_t* __restrict__ table,
                              half* __restrict__ output, int vocab, int dim) {
    const int token = blockIdx.x;
    const int row = indices[token];
    half* dst = output + size_t(token) * dim;
    const size_t blocks_per_row = size_t(dim) / 256;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        if (row < 0 || row >= vocab) { dst[d] = __float2half(0.0f); continue; }
        const uint8_t* block = table + (size_t(row) * blocks_per_row + d / 256) * hq8::BLOCK_BYTES;
        const int k = d % 256;
        dst[d] = __float2half(hq8::step_of(block, k / 32) * float(int8_t(block[hq8::HEADER_BYTES + k])));
    }
}
} // anonymous namespace

void launch_dequant_hq8k(const uint8_t* w, half* y, int K, int N, cudaStream_t s) {
    if (K <= 0 || N <= 0) return;
    if (K % 256) throw std::runtime_error("hq8k requires K divisible by 256");
    dim3 grid(N, (K / 8 + 255) / 256);
    dequant_hq8<<<grid, 256, 0, s>>>(w, y, K, N);
}

// Prefill keeps the 8-bit precision: FP32 accumulation, not cublasHgemm.
void launch_matmul_hq8k_cublas(const half* x, const uint8_t* w, half* y, int M, int K, int N, cudaStream_t s) {
    struct Scratch { half* data = nullptr; size_t capacity = 0; };
    static PerDevice<Scratch> scratches;
    Scratch& current = scratches.current();
    const size_t count = size_t(N) * K;
    if (count > current.capacity) {
        if (current.data) checked(cudaFree(current.data));
        current.data = nullptr; current.capacity = 0;
        checked(cudaMalloc(&current.data, count * sizeof(half))); current.capacity = count;
    }
    const auto handle = cublas_handle_for_stream(s);
    if (!handle) throw std::runtime_error("hq8k cuBLAS handle unavailable");
    launch_dequant_hq8k(w, current.data, K, N, s);
    const float alpha = 1.f, beta = 0.f;
    const auto result = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                                     current.data, CUDA_R_16F, K, x, CUDA_R_16F, K, &beta,
                                     y, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (result != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("hq8k cublasGemmEx failed");
}

void launch_embedding_hq8k(const int32_t* indices, const uint8_t* table, half* output,
                           int batch_size, int seq_len, int vocab_size, int dim, cudaStream_t stream) {
    if (dim % 256) throw std::runtime_error("hq8k embedding requires dim divisible by 256");
    const int tokens = batch_size * seq_len;
    if (tokens > 0) embedding_hq8<<<tokens, std::min(256, dim), 0, stream>>>(indices, table, output, vocab_size, dim);
}
} } // helios::kernels
