// HQS v4 (hq44k_g16) kernel check and microbenchmark on a real matrix.
// Writes the fixed input, every GEMV variant and the dequantized matrix for an
// independent NumPy check, then times v4 against HQ43-G16 (vx3) and HQ42 (stable)
// on the same matrix, rotating copies so the weights do not live in L2.
#include "kernels.hpp"
#include "hqs_v4_gemv.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
using namespace helios::kernels;

static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static std::vector<uint8_t> read_at(const char* path, size_t offset, size_t bytes) {
    std::ifstream f(path, std::ios::binary); std::vector<uint8_t> b(bytes);
    f.seekg(offset); f.read(reinterpret_cast<char*>(b.data()), bytes);
    if (!f) throw std::runtime_error(std::string("read ") + path);
    return b;
}
template<class T> static void dump(const std::string& path, const T* dev, size_t count) {
    std::vector<T> h(count); ck(cudaMemcpy(h.data(), dev, count * sizeof(T), cudaMemcpyDeviceToHost));
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(h.data()), count * sizeof(T));
}

// Median over rounds of the mean time per GEMV, cycling through `copies` weight buffers.
template<class F> static float time_rotating(F launch, int copies, cudaStream_t s) {
    for (int i = 0; i < copies; ++i) launch(i);
    ck(cudaStreamSynchronize(s));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    std::vector<float> rounds;
    for (int r = 0; r < 7; ++r) {
        cudaEventRecord(a, s);
        for (int i = 0; i < 64; ++i) launch(i % copies);
        cudaEventRecord(b, s); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b); rounds.push_back(ms * 1000.f / 64.f);
    }
    std::sort(rounds.begin(), rounds.end());
    return rounds[rounds.size() / 2];
}

int main(int argc, char** argv) try {
    if (argc != 10) {
        fprintf(stderr, "test_hqs_v4 K N v4.bin v4_off hq43.hnf hq43_off hq42.hnf hq42_off out_dir\n");
        return 2;
    }
    const int K = atoi(argv[1]), N = atoi(argv[2]);
    const std::string out = argv[9];
    const size_t blocks = size_t(N) * K / 256;
    const auto v4 = read_at(argv[3], std::stoull(argv[4]), blocks * 152);
    const auto h43 = read_at(argv[5], std::stoull(argv[6]), blocks * 140);
    const auto h42 = read_at(argv[7], std::stoull(argv[8]), blocks * 152);

    std::vector<half> x(K); std::mt19937 rng(1234); std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = __float2half(nd(rng));
    cudaStream_t s; ck(cudaStreamCreate(&s));
    half *dx, *dy, *dw; ck(cudaMalloc(&dx, K * 2)); ck(cudaMalloc(&dy, N * 2)); ck(cudaMalloc(&dw, size_t(N) * K * 2));
    ck(cudaMemcpy(dx, x.data(), K * 2, cudaMemcpyHostToDevice));
    dump(out + "/x.f16", dx, K);

    // Enough copies for >= 384 MiB of weights per format (L2 on this GPU is 48 MiB).
    const int copies = int((384ull << 20) / v4.size()) + 1;
    auto upload = [&](const std::vector<uint8_t>& h) {
        std::vector<uint8_t*> d(copies);
        for (auto& p : d) { ck(cudaMalloc(&p, h.size())); ck(cudaMemcpy(p, h.data(), h.size(), cudaMemcpyHostToDevice)); }
        return d;
    };
    const auto d4 = upload(v4), d43 = upload(h43), d42 = upload(h42);

    using Fn = void(*)(const half*, const uint8_t*, half*, int, int, cudaStream_t);
    const Fn variants[] = {hqs_v4::launch_gemv<1,4>, hqs_v4::launch_gemv<2,8>, hqs_v4::launch_gemv<2,4>, hqs_v4::launch_gemv<1,8>};
    const char* names[] = {"1x4", "2x8", "2x4", "1x8"};
    for (int i = 0; i < 4; ++i) {
        variants[i](dx, d4[0], dy, K, N, s); ck(cudaStreamSynchronize(s));
        dump(out + "/y_v4_" + names[i] + ".f16", dy, N);
    }
    launch_dequant_hq44k(d4[0], dw, K, N, s); ck(cudaStreamSynchronize(s));
    dump(out + "/w_v4.f16", dw, size_t(N) * K);
    launch_matmul_hqs_v3(dx, d43[0], dy, 1, K, N, 4, 16, s); ck(cudaStreamSynchronize(s)); dump(out + "/y_hq43.f16", dy, N);
    launch_matmul_hq42k(dx, d42[0], dy, 1, K, N, s); ck(cudaStreamSynchronize(s)); dump(out + "/y_hq42.f16", dy, N);

    printf("{\"K\":%d,\"N\":%d,\"copies\":%d", K, N, copies);
    for (int i = 0; i < 4; ++i)
        printf(",\"v4_%s_us\":%.2f", names[i], time_rotating([&](int c) { variants[i](dx, d4[c], dy, K, N, s); }, copies, s));
    printf(",\"v4_tuned_us\":%.2f", time_rotating([&](int c) { launch_matmul_hq44k(dx, d4[c], dy, 1, K, N, s); }, copies, s));
    printf(",\"hq43_g16_us\":%.2f", time_rotating([&](int c) { launch_matmul_hqs_v3(dx, d43[c], dy, 1, K, N, 4, 16, s); }, copies, s));
    printf(",\"hq42_us\":%.2f", time_rotating([&](int c) { launch_matmul_hq42k(dx, d42[c], dy, 1, K, N, s); }, copies, s));
    printf(",\"v4_bytes\":%zu,\"hq43_bytes\":%zu,\"hq42_bytes\":%zu}\n", v4.size(), h43.size(), h42.size());
    ck(cudaGetLastError());
    return 0;
} catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
