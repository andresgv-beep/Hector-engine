// kernels/matmul_cublas.cu
// ============================================================================
// MATMUL via cuBLAS — Dequant HQ4K/HQ5K → FP16 buffer → cuBLAS HGEMM
// ============================================================================
//
// Strategy:
//   1. Dequant kernel: HQ4K/HQ5K → FP16 (bandwidth-bound, fast)
//   2. cuBLAS cublasHgemm: FP16 × FP16 with tensor cores
//
// The dequant is cheap (~0.25ms for 50MB) because it's pure memory I/O.
// cuBLAS uses tensor cores for the actual matmul → 4-8x faster than CUDA cores.
//
// Memory: one persistent FP16 scratch buffer, allocated on first use.
// The buffer is sized for the largest weight matrix encountered.

#include "hqs_common.cuh"
#include "cublas_context.hpp"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>

namespace helios {
namespace kernels {

// ============================================================================
// GLOBAL STATE — cuBLAS handle + dequant buffer
// ============================================================================

static cublasHandle_t g_cublas_handle = nullptr;
static half* g_dequant_buffer = nullptr;
static size_t g_dequant_buffer_size = 0;  // in elements

static void ensure_cublas() {
    if (!g_cublas_handle) {
        cublasCreate(&g_cublas_handle);
        // Enable tensor cores explicitly
        cublasSetMathMode(g_cublas_handle, CUBLAS_DEFAULT_MATH);
    }
}

cublasHandle_t cublas_handle_for_stream(cudaStream_t stream) {
    ensure_cublas();
    if (!g_cublas_handle ||
        cublasSetStream(g_cublas_handle, stream) != CUBLAS_STATUS_SUCCESS) {
        return nullptr;
    }
    return g_cublas_handle;
}

static void ensure_dequant_buffer(size_t num_elements, cudaStream_t stream) {
    if (num_elements > g_dequant_buffer_size) {
        if (g_dequant_buffer) cudaFree(g_dequant_buffer);
        // Round up to 1M elements for headroom
        size_t alloc_elements = ((num_elements + 1048575) / 1048576) * 1048576;
        cudaMalloc(&g_dequant_buffer, alloc_elements * sizeof(half));
        g_dequant_buffer_size = alloc_elements;
    }
}

// ============================================================================
// DEQUANT KERNELS — HQ4K/HQ5K → FP16
// ============================================================================
// Each thread dequantizes one group (8 elements) of one row.
// Grid: (num_groups_per_row, N)  — one thread per group per row
// Block: 256 threads (8 groups handled by 8 threads in a warp-ish pattern)
//
// Output layout: [N, K] in FP16 (row-major, same layout as weight matrix)

__global__ void dequant_hq4k_kernel(
    const uint8_t* __restrict__ weights,  // [N, K] in HQ4K superblocks
    half* __restrict__ output,            // [N, K] in FP16
    int K, int N
) {
    using namespace hqs;
    
    const int row = blockIdx.x;  /* filas en x: sin limite 65535 (lm_head: 152k filas) */
    const int group_global = blockIdx.y * blockDim.x + threadIdx.x;
    
    const int num_superblocks = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    const int total_groups = num_superblocks * NUM_GROUPS;
    
    if (row >= N || group_global >= total_groups) return;
    
    const int sb = group_global / NUM_GROUPS;
    const int group_in_sb = group_global % NUM_GROUPS;
    
    const int sb_base_k = sb * SUPER_BLOCK_SIZE;
    const int weight_sb_idx = row * num_superblocks + sb;
    const uint8_t* block_ptr = weights + weight_sb_idx * HQ4K_BLOCK_SIZE;
    
    // Vectorized header load
    const uint32_t* header32 = reinterpret_cast<const uint32_t*>(block_ptr);
    uint32_t hdr = header32[group_in_sb];
    float min_f = __half2float(__ushort_as_half(hdr & 0xFFFF));
    float scale_over_qmax = __half2float(__ushort_as_half(hdr >> 16)) * (1.0f / HQ4K_Q_MAX);
    
    // Vectorized payload load  
    const uint32_t* payload32 = reinterpret_cast<const uint32_t*>(block_ptr + HEADER_SIZE);
    uint32_t packed = payload32[group_in_sb];
    
    const int k_base = sb_base_k + group_in_sb * GROUP_SIZE;
    half* out_ptr = output + row * K + k_base;
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t byte_val = (packed >> (i * 8)) & 0xFF;
        float w0 = min_f + float((byte_val >> 4) & 0x0F) * scale_over_qmax;
        float w1 = min_f + float(byte_val & 0x0F) * scale_over_qmax;
        
        const int k0 = k_base + i * 2;
        const int k1 = k0 + 1;
        if (k0 < K) out_ptr[i * 2] = __float2half(w0);
        if (k1 < K) out_ptr[i * 2 + 1] = __float2half(w1);
    }
}

__global__ void dequant_hq5k_kernel(
    const uint8_t* __restrict__ weights,
    half* __restrict__ output,
    int K, int N
) {
    using namespace hqs;
    
    const int row = blockIdx.x;  /* filas en x: sin limite 65535 (lm_head: 152k filas) */
    const int group_global = blockIdx.y * blockDim.x + threadIdx.x;
    
    const int num_superblocks = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    const int total_groups = num_superblocks * NUM_GROUPS;
    
    if (row >= N || group_global >= total_groups) return;
    
    const int sb = group_global / NUM_GROUPS;
    const int group_in_sb = group_global % NUM_GROUPS;
    
    const int sb_base_k = sb * SUPER_BLOCK_SIZE;
    const int weight_sb_idx = row * num_superblocks + sb;
    const uint8_t* block_ptr = weights + weight_sb_idx * HQ5K_BLOCK_SIZE;
    
    const uint32_t* header32 = reinterpret_cast<const uint32_t*>(block_ptr);
    uint32_t hdr = header32[group_in_sb];
    float min_f = __half2float(__ushort_as_half(hdr & 0xFFFF));
    float scale_over_qmax = __half2float(__ushort_as_half(hdr >> 16)) * (1.0f / HQ5K_Q_MAX);
    
    const uint8_t* payload = block_ptr + HEADER_SIZE + group_in_sb * 5;
    uint64_t bits = uint64_t(payload[0])
                  | (uint64_t(payload[1]) << 8)
                  | (uint64_t(payload[2]) << 16)
                  | (uint64_t(payload[3]) << 24)
                  | (uint64_t(payload[4]) << 32);
    
    const int k_base = sb_base_k + group_in_sb * GROUP_SIZE;
    half* out_ptr = output + row * K + k_base;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t q = (bits >> (i * 5)) & 0x1F;
        float w = min_f + float(q) * scale_over_qmax;
        const int k_idx = k_base + i;
        if (k_idx < K) out_ptr[i] = __float2half(w);
    }
}

// ============================================================================
// DEQUANT KERNELS — HQ4.1K/HQ5.1K (header compacto de 40B) → FP16
// ============================================================================

__global__ void dequant_hq41k_kernel(
    const uint8_t* __restrict__ weights,
    half* __restrict__ output,
    int K, int N
) {
    using namespace hqs;

    const int row = blockIdx.x;  /* filas en x: sin limite 65535 (lm_head: 152k filas) */
    const int group_global = blockIdx.y * blockDim.x + threadIdx.x;

    const int num_superblocks = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    const int total_groups = num_superblocks * NUM_GROUPS;
    if (row >= N || group_global >= total_groups) return;

    const int sb = group_global / NUM_GROUPS;
    const int group_in_sb = group_global % NUM_GROUPS;

    const int sb_base_k = sb * SUPER_BLOCK_SIZE;
    const uint8_t* block_ptr = weights +
        ((size_t)row * num_superblocks + sb) * HQ41K_BLOCK_SIZE;

    float min_f, scoeff;
    decode_compact_group(block_ptr, group_in_sb, min_f, scoeff, 1.0f / HQ4K_Q_MAX);

    const uint32_t* payload32 = reinterpret_cast<const uint32_t*>(
        block_ptr + COMPACT_HEADER_SIZE);
    uint32_t packed = payload32[group_in_sb];

    const int k_base = sb_base_k + group_in_sb * GROUP_SIZE;
    half* out_ptr = output + (size_t)row * K + k_base;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t byte_val = (packed >> (i * 8)) & 0xFF;
        float w0 = fmaf(float((byte_val >> 4) & 0x0F), scoeff, min_f);
        float w1 = fmaf(float(byte_val & 0x0F), scoeff, min_f);
        const int k0 = k_base + i * 2;
        if (k0 < K)     out_ptr[i * 2]     = __float2half(w0);
        if (k0 + 1 < K) out_ptr[i * 2 + 1] = __float2half(w1);
    }
}

__global__ void dequant_hq51k_kernel(
    const uint8_t* __restrict__ weights,
    half* __restrict__ output,
    int K, int N
) {
    using namespace hqs;

    const int row = blockIdx.x;  /* filas en x: sin limite 65535 (lm_head: 152k filas) */
    const int group_global = blockIdx.y * blockDim.x + threadIdx.x;

    const int num_superblocks = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    const int total_groups = num_superblocks * NUM_GROUPS;
    if (row >= N || group_global >= total_groups) return;

    const int sb = group_global / NUM_GROUPS;
    const int group_in_sb = group_global % NUM_GROUPS;

    const int sb_base_k = sb * SUPER_BLOCK_SIZE;
    const uint8_t* block_ptr = weights +
        ((size_t)row * num_superblocks + sb) * HQ51K_BLOCK_SIZE;

    float min_f, scoeff;
    decode_compact_group(block_ptr, group_in_sb, min_f, scoeff, 1.0f / HQ5K_Q_MAX);

    const uint8_t* payload = block_ptr + COMPACT_HEADER_SIZE + group_in_sb * 5;
    uint64_t bits = uint64_t(payload[0])
                  | (uint64_t(payload[1]) << 8)
                  | (uint64_t(payload[2]) << 16)
                  | (uint64_t(payload[3]) << 24)
                  | (uint64_t(payload[4]) << 32);

    const int k_base = sb_base_k + group_in_sb * GROUP_SIZE;
    half* out_ptr = output + (size_t)row * K + k_base;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float w = fmaf(float((bits >> (i * 5)) & 0x1F), scoeff, min_f);
        const int k_idx = k_base + i;
        if (k_idx < K) out_ptr[i] = __float2half(w);
    }
}

// ============================================================================
// LAUNCH: Dequant → cuBLAS
// ============================================================================
// 
// Operation: output[M,N] = input[M,K] @ weights[N,K]^T
//
// cuBLAS uses column-major, but we store row-major.
// Trick: C = A × B^T in row-major = C^T = B × A^T in column-major
// So we call: cublasHgemm(N, M, K, B_dequant, A_input, C_output)
//   with B_dequant being [N, K] row-major = [K, N] column-major
//   and A_input being [M, K] row-major = [K, M] column-major
//   and C_output being [M, N] row-major = [N, M] column-major
//

void launch_matmul_hq4k_cublas(
    const half* input,
    const uint8_t* weights,
    half* output,
    int M, int K, int N,
    cudaStream_t stream
) {
    ensure_cublas();
    cublasSetStream(g_cublas_handle, stream);  // ALWAYS set (even NULL = default stream)
    
    // Ensure buffer fits [N, K]
    ensure_dequant_buffer((size_t)N * K, stream);
    
    // Step 1: Dequant HQ4K → FP16 buffer
    const int num_superblocks = (K + 255) / 256;
    const int total_groups = num_superblocks * 32;
    dim3 grid_dq(N, (total_groups + 255) / 256);
    dim3 block_dq(256);
    dequant_hq4k_kernel<<<grid_dq, block_dq, 0, stream>>>(
        weights, g_dequant_buffer, K, N
    );
    
    // Step 2: cuBLAS HGEMM — FP16 (faster than GemmEx for GEMV on Ada)
    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(0.0f);
    
    cublasHgemm(g_cublas_handle,
        CUBLAS_OP_T,    // B^T (dequanted weights)
        CUBLAS_OP_N,    // A (input)
        N, M, K,        // dims
        &alpha_h,
        g_dequant_buffer, K,
        input, K,
        &beta_h,
        output, N
    );
}

void launch_matmul_hq5k_cublas(
    const half* input,
    const uint8_t* weights,
    half* output,
    int M, int K, int N,
    cudaStream_t stream
) {
    ensure_cublas();
    cublasSetStream(g_cublas_handle, stream);  // ALWAYS set (even NULL = default stream)
    
    ensure_dequant_buffer((size_t)N * K, stream);
    
    const int num_superblocks = (K + 255) / 256;
    const int total_groups = num_superblocks * 32;
    dim3 grid_dq(N, (total_groups + 255) / 256);
    dim3 block_dq(256);
    dequant_hq5k_kernel<<<grid_dq, block_dq, 0, stream>>>(
        weights, g_dequant_buffer, K, N
    );
    
    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(0.0f);
    
    cublasHgemm(g_cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha_h,
        g_dequant_buffer, K,
        input, K,
        &beta_h,
        output, N
    );
}

void launch_matmul_hq41k_cublas(
    const half* input,
    const uint8_t* weights,
    half* output,
    int M, int K, int N,
    cudaStream_t stream
) {
    ensure_cublas();
    cublasSetStream(g_cublas_handle, stream);
    ensure_dequant_buffer((size_t)N * K, stream);

    const int num_superblocks = (K + 255) / 256;
    const int total_groups = num_superblocks * 32;
    dim3 grid_dq(N, (total_groups + 255) / 256);
    dequant_hq41k_kernel<<<grid_dq, 256, 0, stream>>>(
        weights, g_dequant_buffer, K, N
    );

    // Hgemm (acumulacion fp16) A PROPOSITO, no por descuido. En las GeForce
    // NVIDIA capa a la MITAD el fp16 con acumulacion fp32 en los tensor cores
    // — segmentacion de producto: en datacenter va a velocidad plena. Este
    // camino solo se usa en PREFILL por lotes (M >= COMPACT_GEMM_THRESHOLD),
    // donde acumular en fp32 costaba 2x de tiempo sin ganancia de calidad
    // medible (ver tools/quant_bench/prefill_logits_gemma4.cu). El decode va
    // por los kernels propios de matmul_hqs_compact.cu, que SI acumulan en
    // fp32 y no pasan por aqui.
    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(0.0f);
    cublasHgemm(g_cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha_h,
        g_dequant_buffer, K,
        input, K,
        &beta_h,
        output, N
    );
}

void launch_matmul_hq51k_cublas(
    const half* input,
    const uint8_t* weights,
    half* output,
    int M, int K, int N,
    cudaStream_t stream
) {
    ensure_cublas();
    cublasSetStream(g_cublas_handle, stream);
    ensure_dequant_buffer((size_t)N * K, stream);

    const int num_superblocks = (K + 255) / 256;
    const int total_groups = num_superblocks * 32;
    dim3 grid_dq(N, (total_groups + 255) / 256);
    dequant_hq51k_kernel<<<grid_dq, 256, 0, stream>>>(
        weights, g_dequant_buffer, K, N
    );

    // Hgemm (acumulacion fp16) A PROPOSITO, no por descuido. En las GeForce
    // NVIDIA capa a la MITAD el fp16 con acumulacion fp32 en los tensor cores
    // — segmentacion de producto: en datacenter va a velocidad plena. Este
    // camino solo se usa en PREFILL por lotes (M >= COMPACT_GEMM_THRESHOLD),
    // donde acumular en fp32 costaba 2x de tiempo sin ganancia de calidad
    // medible (ver tools/quant_bench/prefill_logits_gemma4.cu). El decode va
    // por los kernels propios de matmul_hqs_compact.cu, que SI acumulan en
    // fp32 y no pasan por aqui.
    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(0.0f);
    cublasHgemm(g_cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha_h,
        g_dequant_buffer, K,
        input, K,
        &beta_h,
        output, N
    );
}

void launch_matmul_fp16_cublas(
    const half* A, const half* B, half* C,
    int M, int K, int N,
    cudaStream_t stream
) {
    ensure_cublas();
    cublasSetStream(g_cublas_handle, stream);  // ALWAYS set

    // GemmEx con CUBLAS_COMPUTE_32F, no Hgemm: entradas y salida en fp16 pero
    // ACUMULACION en fp32. Hgemm acumula en fp16 y con K de miles y las
    // activaciones grandes de Gemma 4 (normas de +236) ese redondeo por paso
    // era la fuente principal del desvio del motor frente a la implementacion
    // oficial con el modelo SIN cuantizar. Los kernels propios de este repo
    // (gemv_fp16_v3, matmul_fp16_kernel, y todos los HQS) ya acumulaban en
    // fp32; cublas era el unico camino que no.
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasGemmEx(g_cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16F, K,
        A, CUDA_R_16F, K,
        &beta,
        C, CUDA_R_16F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
}

// ============================================================================
// DEBUG: Access dequant buffer
// ============================================================================

half* get_dequant_buffer() {
    return g_dequant_buffer;
}

// ============================================================================
// CLEANUP
// ============================================================================

void cleanup_cublas() {
    if (g_dequant_buffer) { cudaFree(g_dequant_buffer); g_dequant_buffer = nullptr; }
    if (g_cublas_handle) { cublasDestroy(g_cublas_handle); g_cublas_handle = nullptr; }
    g_dequant_buffer_size = 0;
}

} // namespace kernels
} // namespace helios
