// Micro-banco del GEMV HQ5.1K usado por lm_head.
// No toca producción: compara el staging actual con lecturas globales directas.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int SUPER_BLOCK_SIZE = 256;
constexpr int GROUP_SIZE = 8;
constexpr int HQ51K_BLOCK_SIZE = 200;
constexpr float HQ5K_Q_MAX = 31.0f;

template<bool DIRECT, int WPR, int RPB>
__global__ void gemv_hq51(
    const half* __restrict__ input,
    const uint8_t* __restrict__ weights,
    half* __restrict__ output,
    int K, int N
) {
    constexpr int BLK_WORDS = HQ51K_BLOCK_SIZE / 4;
    extern __shared__ half s_input[];
    {
        const int block_size = WPR * RPB * 32;
        const float4* src = reinterpret_cast<const float4*>(input);
        float4* dst = reinterpret_cast<float4*>(s_input);
        const int n_vec = K / 8;
        for (int i = threadIdx.x; i < n_vec; i += block_size) dst[i] = src[i];
    }
    __syncthreads();

    const int threads_per_row = WPR * 32;
    const int row_group = threadIdx.x / threads_per_row;
    const int local_tid = threadIdx.x % threads_per_row;
    const int warp_in_group = local_tid / 32;
    const int lane = local_tid % 32;
    const int row = blockIdx.x * RPB + row_group;
    if (row >= N) return;

    float* s_partial = reinterpret_cast<float*>(s_input + K);
    const int warp_in_block = threadIdx.x / 32;
    uint32_t* s_blk = reinterpret_cast<uint32_t*>(s_partial + RPB * WPR)
                    + warp_in_block * BLK_WORDS;

    const int total_sb = (K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    const uint8_t* row_weights =
        weights + (size_t)row * total_sb * HQ51K_BLOCK_SIZE;
    float acc = 0.0f;

    for (int sb = warp_in_group; sb < total_sb; sb += WPR) {
        const uint32_t* global_blk = reinterpret_cast<const uint32_t*>(
            row_weights + (size_t)sb * HQ51K_BLOCK_SIZE);
        const uint32_t* blk = global_blk;
        if constexpr (!DIRECT) {
            s_blk[lane] = global_blk[lane];
            if (lane < BLK_WORDS - 32) s_blk[32 + lane] = global_blk[32 + lane];
            __syncwarp();
            blk = s_blk;
        }

        uint32_t h0 = blk[0];
        uint32_t h1 = blk[1];
        float d_scale = __half2float(__ushort_as_half(h0 & 0xFFFF));
        float d_min = __half2float(__ushort_as_half(h0 >> 16));
        float min_base = __half2float(__ushort_as_half(h1 & 0xFFFF));

        const uint8_t* blk8 = reinterpret_cast<const uint8_t*>(blk);
        uint8_t sp = blk8[8 + (lane >> 1)];
        uint8_t mp = blk8[24 + (lane >> 1)];
        uint8_t q_s = (lane & 1) ? (sp & 0x0F) : (sp >> 4);
        uint8_t q_m = (lane & 1) ? (mp & 0x0F) : (mp >> 4);
        float scoeff = d_scale * float(q_s) * (1.0f / 15.0f) *
                       (1.0f / HQ5K_Q_MAX);
        float min_f = fmaf(d_min, float(q_m) * (1.0f / 15.0f), min_base);

        const int byte_off = 40 + lane * 5;
        const int word = byte_off >> 2;
        const int shift = (byte_off & 3) * 8;
        uint32_t p0 = blk[word];
        uint32_t p1 = blk[word + 1];
        uint32_t lo = __funnelshift_r(p0, p1, shift);
        uint32_t hi = p1 >> shift;
        const int k_base = sb * SUPER_BLOCK_SIZE + lane * GROUP_SIZE;

        #pragma unroll
        for (int i = 0; i < 7; i++) {
            uint32_t q = __funnelshift_r(lo, hi, i * 5) & 0x1F;
            acc = fmaf(fmaf(float(q), scoeff, min_f),
                       __half2float(s_input[k_base + i]), acc);
        }
        uint32_t q7 = (hi >> 3) & 0x1F;
        acc = fmaf(fmaf(float(q7), scoeff, min_f),
                   __half2float(s_input[k_base + 7]), acc);

        if constexpr (!DIRECT) __syncwarp();
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (lane == 0) s_partial[row_group * WPR + warp_in_group] = acc;
    __syncthreads();
    if (warp_in_group == 0 && lane < WPR) {
        float value = s_partial[row_group * WPR + lane];
        #pragma unroll
        for (int offset = WPR / 2; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xFFFFFFFF, value, offset);
        if (lane == 0) output[row] = __float2half(value);
    }
}

using LaunchFn = void (*)(const half*, const uint8_t*, half*, int, int);

#define MAKE_LAUNCHER(NAME, DIRECT, WPR, RPB)                                 \
static void NAME(const half* in, const uint8_t* weights, half* out,           \
                 int K, int N) {                                              \
    int blocks = (N + RPB - 1) / RPB;                                         \
    size_t smem = K * sizeof(half) + RPB * WPR * sizeof(float);                \
    if (!DIRECT) smem += (size_t)(WPR * RPB) * HQ51K_BLOCK_SIZE;               \
    gemv_hq51<DIRECT, WPR, RPB><<<blocks, WPR * RPB * 32, smem>>>(             \
        in, weights, out, K, N);                                               \
}

MAKE_LAUNCHER(staged_4x4, false, 4, 4)
MAKE_LAUNCHER(staged_2x8, false, 2, 8)
MAKE_LAUNCHER(staged_4x1, false, 4, 1)
MAKE_LAUNCHER(direct_4x4, true, 4, 4)
MAKE_LAUNCHER(direct_2x8, true, 2, 8)
MAKE_LAUNCHER(direct_4x1, true, 4, 1)
MAKE_LAUNCHER(direct_1x16, true, 1, 16)

struct Shape { const char* name; int K, N; };
struct Sample { double gbps, us; };

static Sample bench(LaunchFn launch, const half* input, const uint8_t* weights,
                    half* output, int K, int N, size_t bytes, int copies) {
    for (int i = 0; i < 3; i++)
        launch(input, weights + (size_t)(i % copies) * bytes, output, K, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr int iterations = 20;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++)
        launch(input, weights + (size_t)(i % copies) * bytes, output, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return {
        (double)bytes * iterations / (ms / 1000.0) / 1e9,
        (double)ms * 1000.0 / iterations
    };
}

int main() {
    const std::vector<Shape> shapes = {
        {"Qwen3-4B lm_head 2560->151936", 2560, 151936},
        {"Qwen3-8B lm_head 4096->151936", 4096, 151936},
    };
    const char* names[] = {
        "staged 4x4*", "staged 2x8*", "staged 4x1*", "direct 4x4",
        "direct 2x8", "direct 4x1", "direct 1x16"
    };
    LaunchFn variants[] = {
        staged_4x4, staged_2x8, staged_4x1, direct_4x4,
        direct_2x8, direct_4x1, direct_1x16
    };

    std::printf("Micro-banco GEMV HQ5.1K — lm_head\n");
    std::printf("* = configuración candidata del autoajuste de producción\n");
    for (const Shape& shape : shapes) {
        int superblocks = (shape.K + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
        size_t bytes = (size_t)shape.N * superblocks * HQ51K_BLOCK_SIZE;
        int copies = bytes < (512ull << 20) ? 2 : 1;

        half* input = nullptr;
        half* output = nullptr;
        uint8_t* weights = nullptr;
        cudaMalloc(&input, shape.K * sizeof(half));
        cudaMalloc(&output, shape.N * sizeof(half));
        cudaError_t status = cudaMalloc(&weights, bytes * copies);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "cudaMalloc fallo: %s\n", cudaGetErrorString(status));
            return 1;
        }
        cudaMemset(input, 0x3c, shape.K * sizeof(half));
        cudaMemset(weights, 0x11, bytes * copies);

        std::printf("\n%s — %.2f MiB por matriz\n",
                    shape.name, bytes / 1048576.0);
        for (size_t i = 0; i < sizeof(variants) / sizeof(variants[0]); i++) {
            Sample sample = bench(variants[i], input, weights, output,
                                  shape.K, shape.N, bytes, copies);
            std::printf("  %-14s %7.0f GB/s  %8.2f us\n",
                        names[i], sample.gbps, sample.us);
        }
        cudaFree(input);
        cudaFree(output);
        cudaFree(weights);
    }
    return 0;
}
