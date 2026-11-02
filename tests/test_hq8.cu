// HQ8K kernel check on synthetic blocks, independent of model files. d, every u8 step
// and every int8 code are random, so all header bytes and code positions are used. The
// reference decoder follows the layout documented in kernels/hq8_gemv.cuh.
#include "kernels.hpp"
#include "hq8_gemv.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>
using namespace helios::kernels;

static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }

static std::vector<uint8_t> random_blocks(size_t blocks, std::mt19937& rng) {
    std::vector<uint8_t> out(blocks * hq8::BLOCK_BYTES, 0);
    std::uniform_int_distribution<int> code(-127, 127), step(1, 255);
    std::uniform_real_distribution<float> scale(1e-4f, 4e-4f);
    for (size_t b = 0; b < blocks; ++b) {
        uint8_t* p = out.data() + b * hq8::BLOCK_BYTES;
        const uint16_t d = __half_as_ushort(__float2half(scale(rng)));
        p[0] = d & 0xff; p[1] = d >> 8;
        for (int g = 0; g < 8; ++g) p[2 + g] = uint8_t(step(rng));
        for (int i = 0; i < 256; ++i) p[hq8::HEADER_BYTES + i] = uint8_t(int8_t(code(rng)));
    }
    return out;
}

static float reference_weight(const uint8_t* block, int k) {
    const float d = __half2float(__ushort_as_half(uint16_t(block[0] | (block[1] << 8))));
    return d * float(block[2 + k / 32]) / 255.0f * float(int8_t(block[hq8::HEADER_BYTES + k]));
}

static int check_shape(int K, int N, std::mt19937& rng) {
    const auto host = random_blocks(size_t(N) * K / 256, rng);
    std::vector<float> w(size_t(N) * K);
    for (int r = 0; r < N; ++r)
        for (int k = 0; k < K; ++k)
            w[size_t(r) * K + k] = reference_weight(host.data() + (size_t(r) * (K / 256) + k / 256) * hq8::BLOCK_BYTES, k % 256);
    const int M = 12;                                   // >= COMPACT_GEMM_THRESHOLD: also the cuBLAS path
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<half> x(size_t(M) * K);
    for (auto& v : x) v = __float2half(nd(rng));
    std::vector<double> ref(size_t(M) * N);
    for (int m = 0; m < M; ++m)
        for (int r = 0; r < N; ++r) {
            double s = 0;
            for (int k = 0; k < K; ++k) s += double(w[size_t(r) * K + k]) * double(__half2float(x[size_t(m) * K + k]));
            ref[size_t(m) * N + r] = s;
        }
    uint8_t* dw; half *dx, *dy, *dd;
    ck(cudaMalloc(&dw, host.size())); ck(cudaMalloc(&dx, x.size() * 2));
    ck(cudaMalloc(&dy, size_t(M) * N * 2)); ck(cudaMalloc(&dd, w.size() * 2));
    ck(cudaMemcpy(dw, host.data(), host.size(), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    int failures = 0;

    // Dequant: the FP16 of the reference value, at most one ulp apart.
    launch_dequant_hq8k(dw, dd, K, N); ck(cudaDeviceSynchronize());
    std::vector<half> deq(w.size()); ck(cudaMemcpy(deq.data(), dd, deq.size() * 2, cudaMemcpyDeviceToHost));
    size_t off = 0;
    for (size_t i = 0; i < w.size(); ++i) {
        const float got = __half2float(deq[i]), want = __half2float(__float2half(w[i]));
        if (std::fabs(got - want) > std::fabs(want) * 1e-3f + 6e-8f) ++off;
    }
    printf("K=%d N=%d dequant: %zu values beyond 1 ulp\n", K, N, off);
    failures += off != 0;

    auto rel_error = [&](int rows) {
        std::vector<half> y(size_t(rows) * N); ck(cudaMemcpy(y.data(), dy, y.size() * 2, cudaMemcpyDeviceToHost));
        double num = 0, den = 0;
        for (size_t i = 0; i < y.size(); ++i) { const double e = __half2float(y[i]) - ref[i]; num += e * e; den += ref[i] * ref[i]; }
        return std::sqrt(num / den);
    };
    using Fn = void(*)(const half*, const uint8_t*, half*, int, int, cudaStream_t);
    const struct { const char* name; Fn fn; } variants[] = {
        {"1x8u4", hq8::launch_gemv<1,8,4>}, {"2x8u2", hq8::launch_gemv<2,8,2>},
        {"2x4u2", hq8::launch_gemv<2,4,2>}, {"4x2u2", hq8::launch_gemv<4,2,2>}, {"1x4u8", hq8::launch_gemv<1,4,8>}};
    for (const auto& v : variants) {
        v.fn(dx, dw, dy, K, N, nullptr); ck(cudaDeviceSynchronize());
        const double e = rel_error(1);
        printf("K=%d N=%d GEMV %-6s rel error %.2e\n", K, N, v.name, e);
        failures += !(e < 1e-3);
    }
    launch_matmul_hq8k(dx, dw, dy, 1, K, N); ck(cudaDeviceSynchronize());
    const double e1 = rel_error(1);
    launch_matmul_hq8k(dx, dw, dy, M, K, N); ck(cudaDeviceSynchronize());
    const double em = rel_error(M);
    printf("K=%d N=%d dispatch M=1 %.2e, M=%d (cuBLAS FP32) %.2e\n", K, N, e1, M, em);
    failures += !(e1 < 1e-3) + !(em < 1e-3);

    // Embedding: each looked-up row equals the reference row rounded to FP16; out of range gives zeros.
    const int32_t ids_host[4] = {0, N - 1, N / 2, N};
    int32_t* ids; ck(cudaMalloc(&ids, sizeof(ids_host)));
    ck(cudaMemcpy(ids, ids_host, sizeof(ids_host), cudaMemcpyHostToDevice));
    launch_embedding_hq8k(ids, dw, dd, 1, 4, N, K); ck(cudaDeviceSynchronize());
    std::vector<half> rows(size_t(4) * K); ck(cudaMemcpy(rows.data(), dd, rows.size() * 2, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (int t = 0; t < 4; ++t)
        for (int k = 0; k < K; ++k) {
            const float want = ids_host[t] < N ? __half2float(__float2half(w[size_t(ids_host[t]) * K + k])) : 0.f;
            const float got = __half2float(rows[size_t(t) * K + k]);
            if (std::fabs(got - want) > std::fabs(want) * 1e-3f + 6e-8f) ++bad;
        }
    printf("K=%d N=%d embedding: %zu values off\n", K, N, bad);
    failures += bad != 0;
    cudaFree(ids); cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(dd);
    return failures;
}

int main() try {
    setenv("HELIOS_TUNE_NOCACHE", "1", 1);             // never touch the user's tune.cache
    std::mt19937 rng(20261102);
    int failures = 0;
    failures += check_shape(2816, 1024, rng);           // attention q/k/v-like
    failures += check_shape(2304, 2816, rng);           // padded shared down
    failures += check_shape(20480, 128, rng);           // K beyond one staged input chunk
    printf(failures ? "FAILED (%d)\n" : "OK\n", failures);
    return failures ? 1 : 0;
} catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
