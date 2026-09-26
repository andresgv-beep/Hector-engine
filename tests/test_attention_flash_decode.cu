// Flash decode against an FP64 CPU reference and the current decode kernels.
// One CUDA graph per geometry is captured once and replayed with every length,
// as the engine does; ring windows wrap past the physical slots.
#include "kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace helios::kernels;
static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template <class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { ck(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
};
struct Graph {
    cudaGraph_t g = nullptr; cudaGraphExec_t x = nullptr;
    ~Graph() { if (x) cudaGraphExecDestroy(x); if (g) cudaGraphDestroy(g); }
    void finish(cudaStream_t s) { ck(cudaStreamEndCapture(s, &g)); ck(cudaGraphInstantiate(&x, g, nullptr, nullptr, 0)); }
};
static float rnd(uint32_t& s) { s = 1664525u * s + 1013904223u; return float((s >> 8) & 65535) / 32768.f - 1.f; }

struct Geometry { const char* name; int heads, kvh, hd, window, slots; bool peaked; };

static void reference(const std::vector<half>& q, const std::vector<half>& k, const std::vector<half>& v,
                      std::vector<double>& out, const Geometry& g, int len, float scale) {
    const int first = g.window > 0 ? std::max(0, len - g.window) : 0;
    std::vector<double> s(len);
    for (int h = 0; h < g.heads; ++h) {
        const int kh = h / (g.heads / g.kvh);
        double mx = -1e300;
        for (int p = first; p < len; ++p) {
            const size_t base = (size_t(p % g.slots) * g.kvh + kh) * g.hd;
            double dot = 0;
            for (int d = 0; d < g.hd; ++d)
                dot += double(__half2float(q[size_t(h) * g.hd + d])) * __half2float(k[base + d]);
            s[p] = dot * scale; mx = std::max(mx, s[p]);
        }
        double sum = 0;
        for (int p = first; p < len; ++p) { s[p] = std::exp(s[p] - mx); sum += s[p]; }
        for (int d = 0; d < g.hd; ++d) {
            double acc = 0;
            for (int p = first; p < len; ++p)
                acc += s[p] * __half2float(v[(size_t(p % g.slots) * g.kvh + kh) * g.hd + d]);
            out[size_t(h) * g.hd + d] = acc / sum;
        }
    }
}

static float time_graph(cudaStream_t s, const Graph& graph, int launches) {
    cudaEvent_t a, b; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
    for (int i = 0; i < 3; ++i) ck(cudaGraphLaunch(graph.x, s));
    ck(cudaEventRecord(a, s));
    for (int i = 0; i < 10; ++i) ck(cudaGraphLaunch(graph.x, s));
    ck(cudaEventRecord(b, s)); ck(cudaEventSynchronize(b));
    float ms; ck(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms * 1000.f / (10 * launches);
}

static bool run(const Geometry& g, bool bench, bool check) {
    cudaStream_t st; ck(cudaStreamCreate(&st));
    const int splits = 64, min_chunk = 64;
    const size_t qn = size_t(g.heads) * g.hd, kn = size_t(g.slots) * g.kvh * g.hd;
    std::vector<half> q(qn), k(kn), v(kn), out_ref(qn), out_fd(qn);
    uint32_t seed = 7 + g.hd * 3 + g.kvh;
    for (auto& x : q) x = __float2half(rnd(seed) * (g.peaked ? 8.f : 0.5f));
    for (auto& x : k) x = __float2half(rnd(seed) * 2.f);
    for (auto& x : v) x = __float2half(rnd(seed));
    Device<half> dq(qn), dk(kn), dv(kn), dr(qn), df(qn);
    Device<int32_t> dn(1);
    Device<float> work(attention_flash_decode_workspace_bytes(g.heads, g.hd, splits) / sizeof(float));
    ck(cudaMemcpy(dq.p, q.data(), qn * 2, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dk.p, k.data(), kn * 2, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dv.p, v.data(), kn * 2, cudaMemcpyHostToDevice));
    const float scale = g.peaked ? 1.f : 1.f / std::sqrt(float(g.hd));
    auto old_kernel = [&] {
        launch_attention_cached_fp16_dp(dq.p, dk.p, dv.p, dr.p, 1, dn.p, g.heads, g.kvh, g.hd,
                                        g.slots, scale, g.window, st, g.slots);
    };
    auto flash = [&] {
        launch_attention_flash_decode_dp(dq.p, dk.p, dv.p, df.p, work.p, dn.p, g.heads, g.kvh, g.hd,
                                         scale, g.window, g.slots, splits, min_chunk, st);
    };
    Graph both;
    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal)); old_kernel(); flash(); both.finish(st);
    Graph t_old, t_fd;
    if (bench) {
        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal)); for (int i = 0; i < 20; ++i) old_kernel(); t_old.finish(st);
        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal)); for (int i = 0; i < 20; ++i) flash(); t_fd.finish(st);
    }
    bool ok = true;
    std::vector<int> lengths = {1, 2, 17, 63, 64, 65, 127, 255, 256, 257, 1023, 1024, 1025, 1537, 4096, 8192, 12000, 16384};
    if (g.window == 0) lengths.erase(std::remove_if(lengths.begin(), lengths.end(), [&](int x) { return x > g.slots; }), lengths.end());
    else for (int extra : {g.slots + 1, 3 * g.slots + 77}) lengths.push_back(extra);
    std::vector<double> exact(qn);
    for (int len : lengths) {
        ck(cudaMemcpy(dn.p, &len, 4, cudaMemcpyHostToDevice));
        ck(cudaMemset(work.p, 0xff, attention_flash_decode_workspace_bytes(g.heads, g.hd, splits)));
        ck(cudaGraphLaunch(both.x, st)); ck(cudaStreamSynchronize(st));
        float us_old = 0, us_fd = 0;
        if (bench) { us_old = time_graph(st, t_old, 20); us_fd = time_graph(st, t_fd, 20); }
        if (!check) {
            std::printf("%-22s len=%6d old=%8.1fus flash=%8.1fus  x%.2f\n", g.name, len, us_old, us_fd, us_old / us_fd);
            continue;
        }
        ck(cudaMemcpy(out_ref.data(), dr.p, qn * 2, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(out_fd.data(), df.p, qn * 2, cudaMemcpyDeviceToHost));
        reference(q, k, v, exact, g, len, scale);
        double e_old = 0, e_fd = 0;
        size_t differ = 0;
        for (size_t i = 0; i < qn; ++i) differ += std::memcmp(&out_fd[i], &out_ref[i], 2) != 0;
        for (size_t i = 0; i < qn; ++i) {
            const double f = __half2float(out_fd[i]);
            if (!std::isfinite(f)) { e_fd = INFINITY; break; }
            e_fd = std::max(e_fd, std::fabs(f - exact[i]));
            e_old = std::max(e_old, std::fabs(__half2float(out_ref[i]) - exact[i]));
        }
        // FP16 output: half an ulp near 1 is ~4.9e-4; allow a few ulps and never
        // more than twice the current kernel's own error plus that slack.
        const bool pass = e_fd <= std::max(2e-3, 2 * e_old + 1e-3);
        ok &= pass;
        std::printf("%-22s len=%6d err_old=%.2e err_flash=%.2e differ=%zu/%zu %s", g.name, len, e_old, e_fd, differ, qn, pass ? "OK" : "FAIL");
        if (bench) std::printf("  old=%7.1fus flash=%7.1fus x%.2f", us_old, us_fd, us_old / us_fd);
        std::printf("\n");
    }
    cudaStreamDestroy(st);
    return ok;
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]).find("bench") != std::string::npos;
    const bool check = !(argc > 1 && std::string(argv[1]) == "bench-only");
    const Geometry geometries[] = {
        {"gemma12b-local",  16, 8, 256, 1024, 1536, false},
        {"gemma12b-global", 16, 1, 512, 0, 16384, false},
        {"e4b-local",        8, 2, 256, 512, 1024, false},
        {"e4b-global-kv1",   8, 1, 512, 0, 16384, false},
        {"e4b-global-kv2",   8, 2, 512, 0, 16384, false},
        {"qwen3-8b",        32, 8, 128, 0, 16384, false},
        {"mha-hd64",         8, 8, 64, 0, 4096, false},
        {"gqa4-hd96",        8, 2, 96, 0, 4096, false},
        {"peaked-hd512",    16, 1, 512, 0, 16384, true},
    };
    bool ok = true;
    for (const auto& g : geometries) ok &= run(g, bench, check);
    std::printf(ok ? "ALL OK\n" : "FAILURES\n");
    return ok ? 0 : 1;
}
