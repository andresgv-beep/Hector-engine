// HQS v4 (hq44k_g16) kernel check on synthetic blocks, independent of model files.
// Every block field is random (d, dmin, 5-bit scales and mins, 4-bit codes), so all
// bit positions of the header are exercised. The reference decoder below follows the
// on-disk layout documented in kernels/hqs_v4_gemv.cuh.
#include "kernels.hpp"
#include "hqs_v4_gemv.cuh"
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
    std::vector<uint8_t> out(blocks * hqs_v4::BLOCK_BYTES, 0);
    std::uniform_int_distribution<int> byte(0, 255), five(0, 31);
    std::uniform_real_distribution<float> scale(1e-4f, 4e-3f);
    for (size_t b = 0; b < blocks; ++b) {
        uint8_t* p = out.data() + b * hqs_v4::BLOCK_BYTES;
        const uint16_t d = __half_as_ushort(__float2half(scale(rng)));
        const uint16_t dmin = __half_as_ushort(__float2half(scale(rng) * 4.f));
        p[0] = d & 0xff; p[1] = d >> 8; p[2] = dmin & 0xff; p[3] = dmin >> 8;
        for (int g = 0; g < 32; ++g) {                 // 16 scales then 16 mins, 5 bits each
            const unsigned v = five(rng), bit = g * 5;
            for (int k = 0; k < 5; ++k)
                if (v >> k & 1) p[4 + (bit + k) / 8] |= uint8_t(1u << ((bit + k) % 8));
        }
        for (int i = 0; i < 128; ++i) p[hqs_v4::HEADER_BYTES + i] = uint8_t(byte(rng));
    }
    return out;
}

static float reference_weight(const uint8_t* block, int k) {
    auto field = [&](int bit) {
        const unsigned w = unsigned(block[4 + bit / 8]) | (unsigned(block[5 + bit / 8]) << 8);
        return (w >> (bit % 8)) & 31u;
    };
    auto half_at = [&](int o) { return __half2float(__ushort_as_half(uint16_t(block[o] | (block[o + 1] << 8)))); };
    const int g = k / 16;
    const float a = half_at(0) * float(field(g * 5)), m = half_at(2) * float(field(80 + g * 5));
    const uint8_t c = block[hqs_v4::HEADER_BYTES + k / 2];
    return a * float(k & 1 ? c >> 4 : c & 15) - m;
}

static int check_shape(int K, int N, std::mt19937& rng) {
    const size_t blocks = size_t(N) * K / 256;
    const auto host = random_blocks(blocks, rng);
    std::vector<float> w(size_t(N) * K);
    for (int r = 0; r < N; ++r)
        for (int k = 0; k < K; ++k)
            w[size_t(r) * K + k] = reference_weight(host.data() + (size_t(r) * (K / 256) + k / 256) * hqs_v4::BLOCK_BYTES, k % 256);
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

    // Dequant: the FP16 of the reference value, at most one ulp apart (FMA contraction may differ).
    launch_dequant_hq44k(dw, dd, K, N); ck(cudaDeviceSynchronize());
    std::vector<half> deq(w.size()); ck(cudaMemcpy(deq.data(), dd, deq.size() * 2, cudaMemcpyDeviceToHost));
    size_t off = 0;
    for (size_t i = 0; i < w.size(); ++i) {
        const float got = __half2float(deq[i]), want = __half2float(__float2half(w[i]));
        const float ulp = std::fabs(want) * 1e-3f + 6e-8f;   // FP16 has 10 mantissa bits
        if (std::fabs(got - want) > ulp) ++off;
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
        {"v2_1x8u4", hqs_v4::launch_gemv2<1,8,4>}, {"v2_2x8u2", hqs_v4::launch_gemv2<2,8,2>},
        {"v2_2x4u2", hqs_v4::launch_gemv2<2,4,2>}, {"v2_4x2u2", hqs_v4::launch_gemv2<4,2,2>},
        {"v1_1x4", hqs_v4::launch_gemv<1,4>}};
    for (const auto& v : variants) {
        v.fn(dx, dw, dy, K, N, nullptr); ck(cudaDeviceSynchronize());
        const double e = rel_error(1);
        printf("K=%d N=%d GEMV %-9s rel error %.2e\n", K, N, v.name, e);
        failures += !(e < 1e-3);
    }
    launch_matmul_hq44k(dx, dw, dy, 1, K, N); ck(cudaDeviceSynchronize());
    const double e1 = rel_error(1);
    launch_matmul_hq44k(dx, dw, dy, M, K, N); ck(cudaDeviceSynchronize());
    const double em = rel_error(M);
    printf("K=%d N=%d dispatch M=1 %.2e, M=%d (cuBLAS) %.2e\n", K, N, e1, M, em);
    // El prefill cuantizado usa cublasHgemm (acumulación fp16) a propósito, igual que
    // hq41k/hq42k: ver matmul_cublas.cu. Su error esperado es de orden 1e-3.
    failures += !(e1 < 1e-3) + !(em < 1e-2);
    cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(dd);
    return failures;
}

int main() try {
    setenv("HELIOS_TUNE_NOCACHE", "1", 1);             // never touch the user's tune.cache
    std::mt19937 rng(20260928);
    int failures = 0;
    failures += check_shape(3840, 1536, rng);           // gate/up-like: K=3840
    failures += check_shape(15360, 384, rng);           // down-like: K=15360
    failures += check_shape(20480, 128, rng);           // K beyond one staged input chunk
    printf(failures ? "FAILED (%d)\n" : "OK\n", failures);
    return failures ? 1 : 0;
} catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
