// Test de muerte del GEMV batcheado: decodificar cada tile de pesos UNA vez y
// aplicarlo a M vectores de entrada. Si M=2 cuesta <=1,10x que M=1, la lectura
// de pesos se amortiza de verdad y la especulativa vuelve a la mesa.
#include "kernels/hqs_common.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
using namespace helios::hqs;

template<int WPR, int RPB, int M>
__global__ void gemv_batch(const half* __restrict__ input,
                           const uint8_t* __restrict__ weights,
                           half* __restrict__ output, int K, int N) {

    extern __shared__ half s_in[];              // M vectores consecutivos
    {
        const int BS = WPR * RPB * 32;
        const float4* src = reinterpret_cast<const float4*>(input);
        float4* dst = reinterpret_cast<float4*>(s_in);
        const int n_vec = (K * M) / 8;
        for (int i = threadIdx.x; i < n_vec; i += BS) dst[i] = src[i];
        for (int i = n_vec * 8 + threadIdx.x; i < K * M; i += BS) s_in[i] = input[i];
    }
    __syncthreads();

    const int tpr = WPR * 32;
    const int row_group = threadIdx.x / tpr;
    const int local = threadIdx.x % tpr;
    const int warp_in = local / 32, lane = local % 32;
    const int row_raw = blockIdx.x * RPB + row_group;
    const bool activo = (row_raw < N);
    const int row = activo ? row_raw : 0;   // fila valida para no leer fuera

    float* s_part = reinterpret_cast<float*>(s_in + K * M);
    const int total_sb = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    const uint8_t* rw = weights + (size_t)row * total_sb * HQ41K_BLOCK_SIZE;
    float acc[M];
    #pragma unroll
    for (int m = 0; m < M; m++) acc[m] = 0.0f;

    for (int sb = warp_in; sb < total_sb; sb += WPR) {
        const int base_k = sb * SUPER_BLOCK_SIZE;
        const uint32_t* b32 = reinterpret_cast<const uint32_t*>(rw + (size_t)sb * HQ41K_BLOCK_SIZE);
        uint32_t h0 = b32[0], h1 = b32[1];
        float d_scale = __half2float(__ushort_as_half(h0 & 0xFFFF));
        float d_min   = __half2float(__ushort_as_half(h0 >> 16));
        float mbase   = __half2float(__ushort_as_half(h1 & 0xFFFF));
        const uint8_t* b8 = reinterpret_cast<const uint8_t*>(b32);
        uint8_t sp = b8[8 + (lane >> 1)], mp = b8[24 + (lane >> 1)];
        uint8_t q_s = (lane & 1) ? (sp & 0x0F) : (sp >> 4);
        uint8_t q_m = (lane & 1) ? (mp & 0x0F) : (mp >> 4);
        float scoeff = d_scale * float(q_s) * (1.0f/15.0f) * (1.0f/HQ4K_Q_MAX);
        float min_f  = fmaf(d_min, float(q_m) * (1.0f/15.0f), mbase);
        uint32_t packed = b32[10 + lane];        // se lee UNA vez para los M
        const int k_base = base_k + lane * GROUP_SIZE;
        if (k_base + GROUP_SIZE <= K) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint8_t bv = (packed >> (i*8)) & 0xFF;
                float w0 = fmaf(float((bv >> 4) & 0x0F), scoeff, min_f);
                float w1 = fmaf(float(bv & 0x0F), scoeff, min_f);
                #pragma unroll
                for (int m = 0; m < M; m++) {
                    const half2* in2 = reinterpret_cast<const half2*>(s_in + m*K + k_base);
                    float2 x = __half22float2(in2[i]);
                    acc[m] = fmaf(w0, x.x, acc[m]);
                    acc[m] = fmaf(w1, x.y, acc[m]);
                }
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < M; m++) {
        float a = acc[m];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xFFFFFFFF, a, o);
        if (lane == 0) s_part[row_group * WPR + warp_in] = a;
        __syncthreads();
        if (warp_in == 0 && lane < WPR) {
            float v = s_part[row_group * WPR + lane];
            #pragma unroll
            // Mascara SOLO de los lanes que entran en la rama. Con
            // 0xFFFFFFFF el shfl espera a 32 lanes de los que solo llegan WPR
            // y se cuelga (Volta+ con planificacion independiente).
            constexpr unsigned MASK = (WPR >= 32) ? 0xFFFFFFFFu : ((1u << WPR) - 1u);
            for (int o = WPR/2; o > 0; o >>= 1) v += __shfl_down_sync(MASK, v, o);
            if (lane == 0 && activo) output[m * N + row] = __float2half(v);
        }
        __syncthreads();
    }
}

template<int M>
double medir(const half* in, const std::vector<uint8_t*>& w, half* out, int K, int N) {
    constexpr int WPR = 2, RPB = 8, BLOCK = WPR*RPB*32;
    size_t smem = (size_t)K*M*sizeof(half) + RPB*WPR*sizeof(float);
    if (smem > 100000) return -1;
    printf("    [M=%d smem=%zu B] lanzando...\n", M, smem); fflush(stdout);
    cudaFuncSetAttribute(gemv_batch<WPR,RPB,M>, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376);
    int nb = (N + RPB - 1) / RPB;
    for (int i = 0; i < 30; i++) gemv_batch<WPR,RPB,M><<<nb,BLOCK,smem>>>(in,w[0],out,K,N);
    printf("    [M=%d] lanzado, sincronizando...\n", M); fflush(stdout);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("  (M=%d fallo: %s)\n", M, cudaGetErrorString(cudaGetLastError())); return -1; }
    double best = 1e9;
    for (int r = 0; r < 3; r++) {
        cudaEvent_t a,z; cudaEventCreate(&a); cudaEventCreate(&z);
        const int IT = 150; cudaEventRecord(a);
        for (int i = 0; i < IT; i++) gemv_batch<WPR,RPB,M><<<nb,BLOCK,smem>>>(in,w[i%w.size()],out,K,N);
        cudaEventRecord(z); cudaEventSynchronize(z);
        float ms; cudaEventElapsedTime(&ms,a,z); best = fmin(best, ms*1000.0/IT);
    }
    return best;
}

int main() {
    const int K = 2560, N = 9728;
    size_t bytes = (size_t)K*N/256*168;
    std::vector<uint8_t*> w(6);
    for (auto& p : w) { cudaMalloc(&p, bytes); cudaMemset(p, 0x11, bytes); }
    half *in, *out;
    cudaMalloc(&in, (size_t)K*8*2); cudaMalloc(&out, (size_t)N*8*2);
    cudaMemset(in, 0x3c, (size_t)K*8*2);
    printf("  GEMV batcheado, K=%d N=%d (mlp gate de Qwen3-4B)\n\n", K, N);
    double t1 = medir<1>(in,w,out,K,N);
    printf("  %3s %11s %10s %s\n", "M", "tiempo", "vs M=1", "por token");
    printf("  %3d %9.1f us %9s %8.1f us\n", 1, t1, "1.00x", t1);
    double ts[9] = {0};
    ts[1]=t1; ts[2]=medir<2>(in,w,out,K,N); ts[3]=medir<3>(in,w,out,K,N);
    ts[4]=medir<4>(in,w,out,K,N); ts[6]=medir<6>(in,w,out,K,N); ts[8]=medir<8>(in,w,out,K,N);
    for (int m : {2,3,4,6,8}) {
        if (ts[m] < 0) { printf("  %3d  (no cabe en shared)\n", m); continue; }
        printf("  %3d %9.1f us %8.2fx %8.1f us\n", m, ts[m], ts[m]/t1, ts[m]/m);
    }
    printf("\n  barrera: M=2 <= 1.10x\n");
    return 0;
}
