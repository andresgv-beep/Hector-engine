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

// ============================================================================
// GEMV FP16 — V3: shared input + multi-row + half2 vectorized
// ============================================================================
// Block: 256 threads = 8 warps
// Each warp: ROWS_PER_WARP output elements
// Input loaded to shared memory cooperatively
// half2 vectorized loads for 2x bandwidth

constexpr int FP16_ROWS_PER_WARP = 4;
constexpr int FP16_GEMV_WARPS = 8;
constexpr int FP16_ROWS_PER_BLOCK = FP16_GEMV_WARPS * FP16_ROWS_PER_WARP;  // 32
constexpr int FP16_GEMV_BLOCK = FP16_GEMV_WARPS * 32;

// Chunk size for shared memory input tiles
constexpr int FP16_INPUT_CHUNK = 256;  // Match block size for clean cooperative load

__global__ void gemv_fp16_v3_kernel(
    const half* __restrict__ A,   // [1, K]
    const half* __restrict__ B,   // [N, K] - stored transposed
    half* __restrict__ C,         // [1, N]
    int K, int N
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    const int base_row = blockIdx.x * FP16_ROWS_PER_BLOCK + warp_id * FP16_ROWS_PER_WARP;
    
    __shared__ float s_input[FP16_INPUT_CHUNK];
    
    float acc[FP16_ROWS_PER_WARP] = {0.0f};
    
    const int num_chunks = (K + FP16_INPUT_CHUNK - 1) / FP16_INPUT_CHUNK;
    
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int chunk_base = chunk * FP16_INPUT_CHUNK;
        
        // Cooperative load: 256 threads load 256 elements
        {
            const int k_idx = chunk_base + threadIdx.x;
            s_input[threadIdx.x] = (k_idx < K) ? __half2float(A[k_idx]) : 0.0f;
        }
        __syncthreads();
        
        // Each warp processes ROWS_PER_WARP rows
        const int chunk_len = min(FP16_INPUT_CHUNK, K - chunk_base);
        
        #pragma unroll
        for (int r = 0; r < FP16_ROWS_PER_WARP; r++) {
            const int row = base_row + r;
            if (row >= N) break;
            
            const half* B_row = B + row * K + chunk_base;
            
            // Each lane processes elements strided by 32
            for (int i = lane_id; i < chunk_len; i += 32) {
                acc[r] += __half2float(B_row[i]) * s_input[i];
            }
        }
        
        __syncthreads();
    }
    
    // Warp reduction for each row
    #pragma unroll
    for (int r = 0; r < FP16_ROWS_PER_WARP; r++) {
        const int row = base_row + r;
        if (row >= N) break;
        
        float val = acc[r];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        
        if (lane_id == 0) {
            C[row] = __float2half(val);
        }
    }
}

// ============================================================================
// GEMM FP16 — M>1, improved tiling with float accumulation
// ============================================================================

constexpr int FP16_TILE = 32;

__global__ void matmul_fp16_kernel(
    const half* __restrict__ A,   // [M, K]
    const half* __restrict__ B,   // [N, K] - stored transposed
    half* __restrict__ C,         // [M, N]
    int M, int K, int N
) {
    __shared__ float As[FP16_TILE][FP16_TILE];
    __shared__ float Bs[FP16_TILE][FP16_TILE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * FP16_TILE + ty;
    int col = bx * FP16_TILE + tx;
    
    float acc = 0.0f;
    
    int num_tiles = (K + FP16_TILE - 1) / FP16_TILE;
    
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * FP16_TILE + tx;
        int b_k = t * FP16_TILE + ty;
        
        As[ty][tx] = (row < M && a_col < K) ? __half2float(A[row * K + a_col]) : 0.0f;
        Bs[ty][tx] = (col < N && b_k < K) ? __half2float(B[col * K + b_k]) : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < FP16_TILE; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = __float2half(acc);
    }
}

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
        decode_compact_group(block, group, min_f, scoeff, 1.0f / HQ5K_Q_MAX);
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

} // namespace kernels
} // namespace helios
