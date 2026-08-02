// kernels/linear.cu
// ============================================================================
// LINEAR KERNELS - Matmul FP16, Embedding
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "hqs_common.cuh"

namespace helios {
namespace kernels {
// Los kernels propios de GEMV/GEMM en fp16 (gemv_fp16_v3_kernel y
// matmul_fp16_kernel) se retiraron el 2026-08-02: estaban definidos pero NUNCA
// se lanzaban — launch_matmul_fp16 delega siempre en cuBLAS sin mirar M.
// Ademas, ningun modelo de produccion hace un MATMUL con pesos fp16: auditado
// con tools/quant_bench/audit_matmul_dtypes.cu, 0 de 253 en Qwen3-4B, 0 de 253
// en Qwen3-8B y 0 de 277 en Gemma 4 E2B. Estan en el historial si algun dia
// hace falta un camino fp16 propio.

// Forward declaration of cuBLAS version
void launch_matmul_fp16_cublas(const half*, const half*, half*, int, int, int, cudaStream_t);

void launch_matmul_fp16(
    const half* A, const half* B, half* C,
    int M, int K, int N,
    cudaStream_t stream
) {
    launch_matmul_fp16_cublas(A, B, C, M, K, N, stream);
}

// ============================================================================
// EMBEDDING KERNEL
// ============================================================================

__global__ void embedding_fp16_kernel(
    const int32_t* __restrict__ indices,
    const half* __restrict__ table,
    half* __restrict__ output,
    int total_tokens,
    int vocab_size,
    int dim
) {
    int token_idx = blockIdx.x;
    int dim_offset = threadIdx.x;
    
    if (token_idx >= total_tokens) return;
    
    int idx = indices[token_idx];
    
    // size_t obligatorio: la tabla PLE de Gemma 4 son 262144x8960 = 2.35e9
    // elementos, por encima de INT32_MAX. Con int, idx*dim desbordaba a
    // negativo para cualquier token por encima de ~239.600 — y 245237, el
    // marcador de espacio, sale en la segunda posicion de cualquier frase.
    // El resultado era un puntero salvaje y un acceso ilegal a memoria.
    if (idx < 0 || idx >= vocab_size) {
        for (int d = dim_offset; d < dim; d += blockDim.x) {
            output[size_t(token_idx) * dim + d] = __float2half(0.0f);
        }
        return;
    }
    
    const half* src = table + size_t(idx) * dim;
    half* dst = output + size_t(token_idx) * dim;
    
    for (int d = dim_offset; d < dim; d += blockDim.x) {
        dst[d] = src[d];
    }
}

void launch_embedding_fp16(
    const int32_t* indices,
    const half* table,
    half* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    int dim,
    cudaStream_t stream
) {
    int total_tokens = batch_size * seq_len;
    int block_size = min(256, dim);
    
    embedding_fp16_kernel<<<total_tokens, block_size, 0, stream>>>(
        indices, table, output, total_tokens, vocab_size, dim
    );
}

__global__ void embedding_hq51k_kernel(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ table,
    half* __restrict__ output,
    int total_tokens,
    int vocab_size,
    int dim
) {
    using namespace hqs;
    int token = blockIdx.x;
    if (token >= total_tokens) return;
    int row = indices[token];
    half* dst = output + size_t(token) * dim;
    if (row < 0 || row >= vocab_size) {
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            dst[d] = __float2half(0.0f);
        }
        return;
    }

    const size_t blocks_per_row = (size_t(dim) + SUPER_BLOCK_SIZE - 1) /
                                  SUPER_BLOCK_SIZE;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const size_t block_index = size_t(row) * blocks_per_row +
                                   size_t(d) / SUPER_BLOCK_SIZE;
        const int in_block = d % SUPER_BLOCK_SIZE;
        const int group = in_block / GROUP_SIZE;
        const int in_group = in_block % GROUP_SIZE;
        const uint8_t* block = table + block_index * HQ51K_BLOCK_SIZE;

        float min_f = 0.0f;
        float scoeff = 0.0f;
        decode_compact_group(block, group, min_f, scoeff, 1.0f / Q_MAX_5BIT);
        const uint8_t* payload = block + COMPACT_HEADER_SIZE + group * 5;
        uint64_t packed = uint64_t(payload[0]) |
                          (uint64_t(payload[1]) << 8) |
                          (uint64_t(payload[2]) << 16) |
                          (uint64_t(payload[3]) << 24) |
                          (uint64_t(payload[4]) << 32);
        uint32_t q = uint32_t((packed >> (in_group * 5)) & 0x1f);
        dst[d] = __float2half(fmaf(float(q), scoeff, min_f));
    }
}

void launch_embedding_hq51k(
    const int32_t* indices,
    const uint8_t* table,
    half* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    int dim,
    cudaStream_t stream
) {
    if (batch_size <= 0 || seq_len <= 0 || vocab_size <= 0 || dim <= 0) return;
    int total_tokens = batch_size * seq_len;
    int block_size = min(256, dim);
    embedding_hq51k_kernel<<<total_tokens, block_size, 0, stream>>>(
        indices, table, output, total_tokens, vocab_size, dim
    );
}

__global__ void embedding_hq62k_kernel(
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ table,
    half* __restrict__ output,
    int total_tokens,
    int vocab_size,
    int dim
) {
    using namespace hqs;
    int token = blockIdx.x;
    if (token >= total_tokens) return;
    int row = indices[token];
    half* dst = output + size_t(token) * dim;
    if (row < 0 || row >= vocab_size) {
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            dst[d] = __float2half(0.0f);
        }
        return;
    }

    const size_t blocks_per_row = (size_t(dim) + SUPER_BLOCK_SIZE - 1) /
                                  SUPER_BLOCK_SIZE;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const size_t block_index = size_t(row) * blocks_per_row +
                                   size_t(d) / SUPER_BLOCK_SIZE;
        const int in_block = d % SUPER_BLOCK_SIZE;
        const int group = in_block / GROUP_SIZE;
        const int in_group = in_block % GROUP_SIZE;
        const uint8_t* block = table + block_index * HQ62K_BLOCK_SIZE;

        float min_f = 0.0f;
        float scoeff = 0.0f;
        decode_hq62k_group(block, group, min_f, scoeff);
        const uint8_t* payload = block + HQ62K_HEADER_SIZE + group * 6;
        uint64_t packed = uint64_t(payload[0]) |
                          (uint64_t(payload[1]) << 8) |
                          (uint64_t(payload[2]) << 16) |
                          (uint64_t(payload[3]) << 24) |
                          (uint64_t(payload[4]) << 32) |
                          (uint64_t(payload[5]) << 40);
        uint32_t q = uint32_t((packed >> (in_group * 6)) & 0x3f);
        dst[d] = __float2half(fmaf(float(q), scoeff, min_f));
    }
}

void launch_embedding_hq62k(
    const int32_t* indices,
    const uint8_t* table,
    half* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    int dim,
    cudaStream_t stream
) {
    if (batch_size <= 0 || seq_len <= 0 || vocab_size <= 0 || dim <= 0) return;
    int total_tokens = batch_size * seq_len;
    int block_size = min(256, dim);
    embedding_hq62k_kernel<<<total_tokens, block_size, 0, stream>>>(
        indices, table, output, total_tokens, vocab_size, dim
    );
}

} // namespace kernels
} // namespace helios
