// Differential tests for causal cached prefill, including the physical KV ring.
#include "kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace helios::kernels;
static void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { ck(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
};
struct Stream {
    cudaStream_t s;
    Stream() { ck(cudaStreamCreate(&s)); }
    ~Stream() { cudaStreamDestroy(s); }
};
static float random_value(uint32_t& seed) {
    seed = 1664525u * seed + 1013904223u;
    return float((seed >> 8) & 65535) / 32768.f - 1.f;
}

static void run(int heads, int kvh, int hd, int window, int slots,
                int seq, int past, bool peaked, bool bench) {
    Stream stream;
    const size_t qn = size_t(seq) * heads * hd, kn = size_t(slots) * kvh * hd;
    std::vector<half> q(qn), k(kn), v(kn), ref(qn), opt(qn);
    uint32_t seed = 97 + hd + kvh + seq;
    for (auto& x : q) x = __float2half(random_value(seed) * (peaked ? 8.f : .25f));
    for (auto& x : k) x = __float2half(random_value(seed) * 2.f);
    for (auto& x : v) x = __float2half(random_value(seed));
    Device<half> dq(qn), dk(kn), dv(kn), dr(qn), dout(qn);
    ck(cudaMemcpy(dq.p, q.data(), qn * 2, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dk.p, k.data(), kn * 2, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dv.p, v.data(), kn * 2, cudaMemcpyHostToDevice));
    const float scale = peaked ? 1.f : 1.f / std::sqrt(float(hd));
    auto launch = [&](bool coalesced) {
        auto fn = coalesced ? launch_attention_prefill_cached_coalesced_fp16
                            : launch_attention_prefill_cached_fp16;
        fn(dq.p, dk.p, dv.p, coalesced ? dout.p : dr.p, seq, past,
           heads, kvh, hd, 16384, scale, window, stream.s, slots);
    };
    // Graph capture/replay must work without allocations or host cache state.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ck(cudaStreamBeginCapture(stream.s, cudaStreamCaptureModeGlobal));
    launch(false); launch(true);
    ck(cudaStreamEndCapture(stream.s, &graph));
    ck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    ck(cudaGraphLaunch(executable, stream.s));
    ck(cudaStreamSynchronize(stream.s));
    ck(cudaGraphExecDestroy(executable)); ck(cudaGraphDestroy(graph));
    ck(cudaMemcpy(ref.data(), dr.p, qn * 2, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(opt.data(), dout.p, qn * 2, cudaMemcpyDeviceToHost));
    if (std::memcmp(ref.data(), opt.data(), qn * 2))
        throw std::runtime_error("prefill differs from reference bits");
    for (auto x : opt) if (!std::isfinite(__half2float(x)))
        throw std::runtime_error("nonfinite prefill output");

    // Independent FP64 CPU attention: start/middle/end queries, first/last head.
    double error = 0;
    for (int qi : {0, seq / 2, seq - 1}) for (int h : {0, heads - 1}) {
        const int end = past + qi + 1;
        const int first = window ? std::max(0, end - window) : 0;
        const int kh = h / (heads / kvh);
        std::vector<double> scores(end - first);
        double maximum = -INFINITY;
        for (int p = first; p < end; ++p) {
            double dot = 0;
            for (int d = 0; d < hd; ++d)
                dot += double(__half2float(q[(size_t(qi) * heads + h) * hd + d])) *
                    __half2float(k[(size_t(p % slots) * kvh + kh) * hd + d]);
            scores[p - first] = dot * scale;
            maximum = std::max(maximum, dot * scale);
        }
        double denom = 0;
        for (auto& s : scores) { s = std::exp(s - maximum); denom += s; }
        for (int d = 0; d < hd; ++d) {
            double value = 0;
            for (int p = first; p < end; ++p)
                value += scores[p - first] * __half2float(v[(size_t(p % slots) * kvh + kh) * hd + d]);
            error = std::max(error, std::fabs(value / denom -
                __half2float(opt[(size_t(qi) * heads + h) * hd + d])));
        }
    }
    if (error >= .001) throw std::runtime_error("prefill CPU error >= 0.001");
    std::printf("PASS h=%d kvh=%d hd=%d window=%d slots=%d seq=%d past=%d peaked=%d bitwise=1 cpu_max=%.8g\n",
                heads, kvh, hd, window, slots, seq, past, peaked, error);
    if (bench) {
        cudaEvent_t begin, end;
        ck(cudaEventCreate(&begin)); ck(cudaEventCreate(&end));
        float samples[2][6];
        for (int mode = 0; mode < 2; ++mode) for (int i = 0; i < 3; ++i) launch(mode);
        for (int pair = 0; pair < 6; ++pair) for (int order = 0; order < 2; ++order) {
            const int mode = order ^ (pair & 1); // AB/BA, same process and inputs
            ck(cudaEventRecord(begin, stream.s));
            for (int i = 0; i < 5; ++i) launch(mode);
            ck(cudaEventRecord(end, stream.s)); ck(cudaEventSynchronize(end));
            ck(cudaEventElapsedTime(&samples[mode][pair], begin, end));
            samples[mode][pair] /= 5.f;
        }
        for (auto& s : samples) std::sort(s, s + 6);
        const float a = (samples[0][2] + samples[0][3]) / 2;
        const float b = (samples[1][2] + samples[1][3]) / 2;
        std::printf("BENCH hd=%d kvh=%d window=%d seq=%d past=%d ref_ms=%.6f opt_ms=%.6f speedup=%.4f\n",
                    hd, kvh, window, seq, past, a, b, a / b);
        ck(cudaEventDestroy(begin)); ck(cudaEventDestroy(end));
    }
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    try {
        const bool e4b_only = argc > 1 && std::strcmp(argv[1], "--bench-e4b") == 0;
        const bool bench = e4b_only || (argc > 1 && std::strcmp(argv[1], "--bench") == 0);
        if (!e4b_only) {
            for (int hd : {256, 512}) for (int past : {0, 1023, 1535, 6144}) {
                const int window = hd == 256 ? 1024 : 0;
                const int slots = window ? 1536 : 16384;
                for (int seq : {7, 512})
                    run(16, hd == 256 ? 8 : 1, hd, window, slots, seq, past, false, bench);
                run(16, hd == 256 ? 8 : 1, hd, window, slots, 33, past, true, false);
            }
            run(4, 4, 64, 0, 1024, 1, 0, true, false);
            run(8, 2, 128, 512, 640, 128, 639, true, false);
            run(4, 1, 96, 64, 128, 65, 64, false, false);
        }
        for (int hd : {256, 512}) for (int past : {0, 511, 1023, 6144}) {
            const int window = hd == 256 ? 512 : 0;
            const int slots = window ? 1024 : 16384;
            for (int seq : {7, 512})
                run(8, 2, hd, window, slots, seq, past, false, bench);
            run(8, 2, hd, window, slots, 33, past, true, false);
        }
        // A multimodal caller may submit the entire 6144-query batch at once.
        run(8, 2, 256, 512, 6656, 6144, 1023, false, false);
        run(8, 2, 512, 0, 16384, 6144, 1023, false, false);
        std::printf("PASS: %d prefill cases; exact GPU parity, CPU FP64 reference, graph replay\n",
                    e4b_only ? 26 : 53);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
    }
}
