#include <string>
#include <cstdint>
// tools/bench_gemv.cu
// ============================================================================
// Micro-banco del GEMV cuantizado — aislado, reproducible, sin conversación
// ============================================================================
//
// Motivación (medido con nsys el 2026-07-31): gemv_hq41k se lleva el 83% del
// tiempo de decode y va a 212 GB/s sobre un techo real de 371 GB/s (57%).
// Curiosamente el HQ5.1K va a 250 GB/s leyendo MÁS bytes por peso: el problema
// no es el volumen, es cómo lee el kernel de 4 bits.
//
// Este banco mide variantes con las dimensiones REALES de los modelos, sin
// ruido de térmica ni de sesión. Regla: ninguna variante entra al motor sin
// ganar aquí de forma repetible.
//
// Uso:  nvcc -O3 -arch=native -I../kernels -o bench_gemv bench_gemv.cu
//       ./bench_gemv
//
// NO toca kernels de producción: incluye copias locales para experimentar.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Constantes del formato compacto (espejo de hqs_common.cuh)
// ---------------------------------------------------------------------------
constexpr int SUPER_BLOCK_SIZE = 256;
constexpr int GROUP_SIZE = 8;
constexpr int NUM_GROUPS = 32;
constexpr int COMPACT_HEADER_SIZE = 40;
constexpr int HQ4K_PAYLOAD = 128;
constexpr int HQ41K_BLOCK_SIZE = COMPACT_HEADER_SIZE + HQ4K_PAYLOAD;  // 168
constexpr float HQ4K_Q_MAX = 15.0f;

// ---------------------------------------------------------------------------
// VARIANTE A — la de producción hoy (staging del bloque entero en shared)
// ---------------------------------------------------------------------------
template<int WPR, int RPB>
__global__ void gemv_A(const half* __restrict__ input,
                       const uint8_t* __restrict__ weights,
                       half* __restrict__ output, int K, int N) {
    constexpr int BLK_WORDS = HQ41K_BLOCK_SIZE / 4;  // 42
    extern __shared__ half s_input[];
    {
        const int BS = WPR * RPB * 32;
        const float4* src = reinterpret_cast<const float4*>(input);
        float4* dst = reinterpret_cast<float4*>(s_input);
        const int n_vec = K / 8;
        for (int i = threadIdx.x; i < n_vec; i += BS) dst[i] = src[i];
    }
    __syncthreads();

    const int tpr = WPR * 32;
    const int row_group = threadIdx.x / tpr;
    const int local = threadIdx.x % tpr;
    const int warp_in_group = local / 32;
    const int lane = local % 32;
    const int row = blockIdx.x * RPB + row_group;
    if (row >= N) return;

    float* s_partial = reinterpret_cast<float*>(s_input + K);
    const int warp_in_block = threadIdx.x / 32;
    uint32_t* s_blk = reinterpret_cast<uint32_t*>(s_partial + RPB * WPR)
                    + warp_in_block * BLK_WORDS;

    const int total_sb = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    const uint8_t* row_w = weights + (size_t)row * total_sb * HQ41K_BLOCK_SIZE;
    float acc = 0.0f;

    for (int sb = warp_in_group; sb < total_sb; sb += WPR) {
        const uint32_t* blk32 = reinterpret_cast<const uint32_t*>(
            row_w + (size_t)sb * HQ41K_BLOCK_SIZE);
        s_blk[lane] = blk32[lane];
        if (lane < BLK_WORDS - 32) s_blk[32 + lane] = blk32[32 + lane];
        __syncwarp();

        uint32_t h0 = s_blk[0], h1 = s_blk[1];
        float d_scale  = __half2float(__ushort_as_half(h0 & 0xFFFF));
        float d_min    = __half2float(__ushort_as_half(h0 >> 16));
        float min_base = __half2float(__ushort_as_half(h1 & 0xFFFF));
        const uint8_t* sb8 = reinterpret_cast<const uint8_t*>(s_blk);
        uint8_t sp = sb8[8 + (lane >> 1)], mp = sb8[24 + (lane >> 1)];
        uint8_t q_s = (lane & 1) ? (sp & 0x0F) : (sp >> 4);
        uint8_t q_m = (lane & 1) ? (mp & 0x0F) : (mp >> 4);
        float scoeff = d_scale * float(q_s) * (1.0f/15.0f) * (1.0f/HQ4K_Q_MAX);
        float min_f  = fmaf(d_min, float(q_m) * (1.0f/15.0f), min_base);

        uint32_t packed = s_blk[10 + lane];
        const int k_base = sb * SUPER_BLOCK_SIZE + lane * GROUP_SIZE;
        const half2* in2 = reinterpret_cast<const half2*>(s_input + k_base);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint8_t b = (packed >> (i * 8)) & 0xFF;
            float w0 = fmaf(float((b >> 4) & 0x0F), scoeff, min_f);
            float w1 = fmaf(float(b & 0x0F), scoeff, min_f);
            float2 x = __half22float2(in2[i]);
            acc = fmaf(w0, x.x, acc);
            acc = fmaf(w1, x.y, acc);
        }
        __syncwarp();
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, o);
    if (lane == 0) s_partial[row_group * WPR + warp_in_group] = acc;
    __syncthreads();
    if (warp_in_group == 0 && lane < WPR) {
        float v = s_partial[row_group * WPR + lane];
        #pragma unroll
        for (int o = WPR / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFF, v, o);
        if (lane == 0) output[row] = __float2half(v);
    }
}

// ---------------------------------------------------------------------------
// VARIANTE B — SIN staging en shared: cada lane lee sus datos directo de global
// Hipótesis: el staging + 2 __syncwarp por superbloque cuesta más de lo que
// ahorra; las lecturas ya son coalescadas de por sí.
// ---------------------------------------------------------------------------
template<int WPR, int RPB>
__global__ void gemv_B(const half* __restrict__ input,
                       const uint8_t* __restrict__ weights,
                       half* __restrict__ output, int K, int N) {
    extern __shared__ half s_input[];
    {
        const int BS = WPR * RPB * 32;
        const float4* src = reinterpret_cast<const float4*>(input);
        float4* dst = reinterpret_cast<float4*>(s_input);
        const int n_vec = K / 8;
        for (int i = threadIdx.x; i < n_vec; i += BS) dst[i] = src[i];
    }
    __syncthreads();

    const int tpr = WPR * 32;
    const int row_group = threadIdx.x / tpr;
    const int local = threadIdx.x % tpr;
    const int warp_in_group = local / 32;
    const int lane = local % 32;
    const int row = blockIdx.x * RPB + row_group;
    if (row >= N) return;

    float* s_partial = reinterpret_cast<float*>(s_input + K);
    const int total_sb = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    const uint8_t* row_w = weights + (size_t)row * total_sb * HQ41K_BLOCK_SIZE;
    float acc = 0.0f;

    for (int sb = warp_in_group; sb < total_sb; sb += WPR) {
        const uint8_t* blk = row_w + (size_t)sb * HQ41K_BLOCK_SIZE;
        const uint32_t* b32 = reinterpret_cast<const uint32_t*>(blk);

        // Header: los 3 fp16 los lee todo el warp (broadcast desde caché L1)
        uint32_t h0 = b32[0], h1 = b32[1];
        float d_scale  = __half2float(__ushort_as_half(h0 & 0xFFFF));
        float d_min    = __half2float(__ushort_as_half(h0 >> 16));
        float min_base = __half2float(__ushort_as_half(h1 & 0xFFFF));
        uint8_t sp = blk[8 + (lane >> 1)], mp = blk[24 + (lane >> 1)];
        uint8_t q_s = (lane & 1) ? (sp & 0x0F) : (sp >> 4);
        uint8_t q_m = (lane & 1) ? (mp & 0x0F) : (mp >> 4);
        float scoeff = d_scale * float(q_s) * (1.0f/15.0f) * (1.0f/HQ4K_Q_MAX);
        float min_f  = fmaf(d_min, float(q_m) * (1.0f/15.0f), min_base);

        uint32_t packed = b32[10 + lane];   // coalescado: 32 lanes x 4B seguidos
        const int k_base = sb * SUPER_BLOCK_SIZE + lane * GROUP_SIZE;
        const half2* in2 = reinterpret_cast<const half2*>(s_input + k_base);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint8_t b = (packed >> (i * 8)) & 0xFF;
            float w0 = fmaf(float((b >> 4) & 0x0F), scoeff, min_f);
            float w1 = fmaf(float(b & 0x0F), scoeff, min_f);
            float2 x = __half22float2(in2[i]);
            acc = fmaf(w0, x.x, acc);
            acc = fmaf(w1, x.y, acc);
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, o);
    if (lane == 0) s_partial[row_group * WPR + warp_in_group] = acc;
    __syncthreads();
    if (warp_in_group == 0 && lane < WPR) {
        float v = s_partial[row_group * WPR + lane];
        #pragma unroll
        for (int o = WPR / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFF, v, o);
        if (lane == 0) output[row] = __float2half(v);
    }
}

// ---------------------------------------------------------------------------
// Banco
// ---------------------------------------------------------------------------
struct Shape { const char* name; int K, N; };

// CRÍTICO: rotar entre COPIAS de la matriz para que los datos lleguen SIEMPRE
// fríos de VRAM. Sin esto se mide la L2 (~32 MB) y salen cifras imposibles
// (600 GB/s sobre un techo de 371) que no representan el decode real, donde
// cada matriz se lee una sola vez por token.
static double bench(void (*launch)(const half*, const uint8_t*, half*, int, int),
                    const half* in, const uint8_t* w, half* out,
                    int K, int N, size_t bytes, int copies) {
    for (int i = 0; i < 3; i++) launch(in, w, out, K, N);
    cudaDeviceSynchronize();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int IT = 30;
    cudaEventRecord(a);
    for (int i = 0; i < IT; i++)
        launch(in, w + (size_t)(i % copies) * bytes, out, K, N);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    return (double)bytes * IT / (ms / 1000.0) / 1e9;   // GB/s
}

#define MK_LAUNCH(NAME, KERNEL, WPR, RPB)                                     \
static void NAME(const half* in, const uint8_t* w, half* out, int K, int N) { \
    int nb = (N + RPB - 1) / RPB;                                             \
    size_t smem = K * sizeof(half) + RPB * WPR * sizeof(float)                \
                + (size_t)(WPR * RPB) * HQ41K_BLOCK_SIZE;                     \
    KERNEL<WPR, RPB><<<nb, WPR * RPB * 32, smem, 0>>>(in, w, out, K, N);      \
}

MK_LAUNCH(a_2x8, gemv_A, 2, 8)
MK_LAUNCH(a_4x4, gemv_A, 4, 4)
MK_LAUNCH(b_2x8, gemv_B, 2, 8)
MK_LAUNCH(b_4x4, gemv_B, 4, 4)
MK_LAUNCH(b_1x16, gemv_B, 1, 16)
MK_LAUNCH(b_2x16, gemv_B, 2, 16)

int main() {
    // Dimensiones reales: (K=hidden, N=salida) de las matmuls que dominan
    std::vector<Shape> shapes = {
        {"4B mlp gate/up  (2560->9728)", 2560,  9728},
        {"4B mlp down     (9728->2560)", 9728,  2560},
        {"4B attn q       (2560->4096)", 2560,  4096},
        {"8B mlp gate/up  (4096->12288)",4096, 12288},
        {"8B mlp down    (12288->4096)",12288,  4096},
        {"4B attn k/v    (2560->1024) PEQ", 2560, 1024},
        {"8B attn k/v    (4096->1024) PEQ", 4096, 1024},
        {"4B attn o      (4096->2560)", 4096, 2560},
    };

    printf("Micro-banco GEMV HQ4.1K — GB/s (mayor es mejor)\n");
    printf("%-30s %8s %8s %8s %8s %8s %8s\n", "forma",
           "A 2x8*", "A 4x4", "B 2x8", "B 4x4", "B 1x16", "B 2x16");
    printf("%s\n", std::string(90, '-').c_str());

    for (auto& s : shapes) {
        int sb = (s.K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
        size_t wbytes = (size_t)s.N * sb * HQ41K_BLOCK_SIZE;
        // Copias suficientes para superar de largo la L2 (~32 MB)
        int copies = (int)((256ull << 20) / wbytes) + 1;
        half *in, *out; uint8_t* w;
        cudaMalloc(&in, s.K * sizeof(half));
        cudaMalloc(&out, s.N * sizeof(half));
        if (cudaMalloc(&w, wbytes * copies) != cudaSuccess) { printf("sin VRAM\n"); return 1; }
        cudaMemset(in, 0x3c, s.K * sizeof(half));
        cudaMemset(w, 0x11, wbytes * copies);

        printf("%-30s", s.name);
        for (auto fn : {a_2x8, a_4x4, b_2x8, b_4x4, b_1x16, b_2x16})
            printf(" %8.0f", bench(fn, in, w, out, s.K, s.N, wbytes, copies));
        printf("\n");

        cudaFree(in); cudaFree(out); cudaFree(w);
    }
    printf("\n* A 2x8 = variante actual en producción. Techo medido: 371 GB/s\n");

    // -----------------------------------------------------------------------
    // PRUEBA DECISIVA: ¿cuánto cuesta que los kernels sean DEPENDIENTES?
    // En el banco de arriba las iteraciones son independientes y la GPU solapa
    // el drenaje de una con el arranque de la siguiente. En el decode real hay
    // ~250 kernels ENCADENADOS por token: cada uno espera al anterior.
    // -----------------------------------------------------------------------
    printf("\n--- Independiente vs encadenado (variante de producción A 2x8) ---\n");
    for (auto& s2 : shapes) {
        int sb = (s2.K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
        size_t wb = (size_t)s2.N * sb * HQ41K_BLOCK_SIZE;
        int copies = (int)((256ull << 20) / wb) + 1;
        half *in, *out; uint8_t* w;
        cudaMalloc(&in, s2.K * sizeof(half));
        cudaMalloc(&out, s2.N * sizeof(half));
        if (cudaMalloc(&w, wb * copies) != cudaSuccess) break;
        cudaMemset(in, 0x3c, s2.K * sizeof(half));
        cudaMemset(w, 0x11, wb * copies);

        double indep = bench(a_2x8, in, w, out, s2.K, s2.N, wb, copies);

        // Encadenado: sincronizar entre lanzamientos simula la dependencia
        for (int i = 0; i < 3; i++) a_2x8(in, w, out, s2.K, s2.N);
        cudaDeviceSynchronize();
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        const int IT = 30;
        cudaEventRecord(a);
        for (int i = 0; i < IT; i++) {
            a_2x8(in, w + (size_t)(i % copies) * wb, out, s2.K, s2.N);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        double chained = (double)wb * IT / (ms / 1000.0) / 1e9;

        printf("%-30s indep %5.0f GB/s | encadenado %5.0f GB/s  (%.0f%% de perdida)\n",
               s2.name, indep, chained, 100.0 * (indep - chained) / indep);
        cudaFree(in); cudaFree(out); cudaFree(w);
    }
    return 0;
}
