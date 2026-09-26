// GEMM prefill attention against an FP64 CPU reference and the reference
// kernel: GQA/MQA, HD 128/256/512, causal chunks after long pasts and sliding
// windows whose ring has wrapped several times. "bench" also times both paths.
#include "kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
using namespace helios::kernels;

static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static float rnd(uint32_t& s) { s = 1664525u * s + 1013904223u; return float((s >> 8) & 65535) / 32768.f - 1.f; }

struct Geometry { const char* name; int heads, kvh, hd, window, slots; };

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "bench";
    const Geometry geometries[] = {
        {"qwen3-8b", 32, 8, 128, 0, 16896},
        {"gemma12b-local", 16, 8, 256, 1024, 1536},
        {"gemma12b-global", 16, 1, 512, 0, 16896},
        {"e4b-local", 8, 2, 256, 512, 1024},
        {"e4b-global", 8, 2, 512, 0, 16896},
        {"mha-hd64", 8, 8, 64, 0, 4608},
    };
    struct Case { int past, seq; };
    const Case cases[] = {{0, 12}, {0, 512}, {500, 12}, {1536, 512}, {4096, 512}, {7000, 300}, {15872, 512}};
    bool ok = true;
    cudaStream_t st; ck(cudaStreamCreate(&st));
    for (const auto& g : geometries) for (const auto& c : cases) {
        if (g.window == 0 && c.past + c.seq > g.slots) continue;
        const size_t qn = size_t(c.seq) * g.heads * g.hd, kn = size_t(g.slots) * g.kvh * g.hd;
        std::vector<half> q(qn), k(kn), v(kn), o_ref(qn), o_gemm(qn);
        uint32_t seed = 3 + g.hd + c.past;
        for (auto& x : q) x = __float2half(rnd(seed) * 0.5f);
        for (auto& x : k) x = __float2half(rnd(seed) * 2.f);
        for (auto& x : v) x = __float2half(rnd(seed));
        half *dq, *dk, *dv, *dr, *dg;
        ck(cudaMalloc(&dq, qn * 2)); ck(cudaMalloc(&dk, kn * 2)); ck(cudaMalloc(&dv, kn * 2));
        ck(cudaMalloc(&dr, qn * 2)); ck(cudaMalloc(&dg, qn * 2));
        ck(cudaMemcpy(dq, q.data(), qn * 2, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(dk, k.data(), kn * 2, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(dv, v.data(), kn * 2, cudaMemcpyHostToDevice));
        const float scale = 1.f / std::sqrt(float(g.hd));
        auto run_ref = [&] { launch_attention_prefill_cached_fp16(dq, dk, dv, dr, c.seq, c.past, g.heads, g.kvh, g.hd, g.slots, scale, g.window, st, g.slots); };
        auto run_gemm = [&] {
            if (!launch_attention_prefill_gemm_fp16(dq, dk, dv, dg, c.seq, c.past, g.heads, g.kvh, g.hd, g.slots, scale, g.window, st, g.slots))
                throw std::runtime_error("gemm prefill launch failed");
        };
        run_ref(); run_gemm(); ck(cudaStreamSynchronize(st));
        ck(cudaMemcpy(o_ref.data(), dr, qn * 2, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(o_gemm.data(), dg, qn * 2, cudaMemcpyDeviceToHost));
        // FP64 reference on sampled queries and heads.
        double e_ref = 0, e_gemm = 0;
        std::vector<double> s(c.past + c.seq);
        for (int qi : {0, c.seq / 2, c.seq - 1}) for (int h = 0; h < g.heads; h += std::max(1, g.heads / 5)) {
            const int pos = c.past + qi, first = g.window > 0 ? std::max(0, pos + 1 - g.window) : 0;
            const int kh = h / (g.heads / g.kvh);
            double mx = -1e300;
            for (int p = first; p <= pos; ++p) {
                double d = 0;
                const size_t kb = (size_t(p % g.slots) * g.kvh + kh) * g.hd;
                for (int j = 0; j < g.hd; ++j) d += double(__half2float(q[(size_t(qi) * g.heads + h) * g.hd + j])) * __half2float(k[kb + j]);
                s[p] = d * scale; mx = std::max(mx, s[p]);
            }
            double sum = 0; for (int p = first; p <= pos; ++p) { s[p] = std::exp(s[p] - mx); sum += s[p]; }
            for (int j = 0; j < g.hd; ++j) {
                double a = 0;
                for (int p = first; p <= pos; ++p) a += s[p] * __half2float(v[(size_t(p % g.slots) * g.kvh + kh) * g.hd + j]);
                a /= sum;
                const size_t i = (size_t(qi) * g.heads + h) * g.hd + j;
                e_ref = std::max(e_ref, std::fabs(__half2float(o_ref[i]) - a));
                e_gemm = std::max(e_gemm, std::fabs(__half2float(o_gemm[i]) - a));
            }
        }
        // FP16 probabilities in P V add ~1e-3 relative on top of the output rounding.
        const bool pass = std::isfinite(e_gemm) && e_gemm <= std::max(3e-3, 3 * e_ref);
        ok &= pass;
        std::printf("%-16s past=%5d seq=%3d err_ref=%.2e err_gemm=%.2e %s", g.name, c.past, c.seq, e_ref, e_gemm, pass ? "OK" : "FAIL");
        if (bench) {
            cudaEvent_t a, b; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
            float t_ref, t_gemm;
            ck(cudaEventRecord(a, st)); for (int i = 0; i < 3; ++i) run_ref(); ck(cudaEventRecord(b, st)); ck(cudaEventSynchronize(b)); ck(cudaEventElapsedTime(&t_ref, a, b));
            ck(cudaEventRecord(a, st)); for (int i = 0; i < 3; ++i) run_gemm(); ck(cudaEventRecord(b, st)); ck(cudaEventSynchronize(b)); ck(cudaEventElapsedTime(&t_gemm, a, b));
            std::printf("  ref=%8.2fms gemm=%7.2fms x%.1f", t_ref / 3, t_gemm / 3, t_ref / t_gemm);
            cudaEventDestroy(a); cudaEventDestroy(b);
        }
        std::printf("\n");
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dr); cudaFree(dg);
    }
    // Gemma 4 12B image blocks: queries inside [begin, end) also see later keys
    // of the block, windowed like any other key. FP64 reference only: the
    // reference kernel has no block.
    struct Block { const char* name; int heads, kvh, hd, window, slots, past, seq, begin, end; };
    const Block blocks[] = {
        {"gemma12b-local", 16, 8, 256, 1024, 1536, 0, 300, 5, 285},
        {"gemma12b-local", 16, 8, 256, 1024, 1536, 1400, 400, 1410, 1690},
        {"window-cuts-img", 16, 8, 256, 64, 1536, 200, 300, 220, 500},
        {"gemma12b-global", 16, 1, 512, 0, 16896, 100, 300, 110, 390},
    };
    for (const auto& c : blocks) {
        const size_t qn = size_t(c.seq) * c.heads * c.hd, kn = size_t(c.slots) * c.kvh * c.hd;
        std::vector<half> q(qn), k(kn), v(kn), o(qn);
        uint32_t seed = 11 + c.hd + c.past;
        for (auto& x : q) x = __float2half(rnd(seed) * 0.5f);
        for (auto& x : k) x = __float2half(rnd(seed) * 2.f);
        for (auto& x : v) x = __float2half(rnd(seed));
        half *dq, *dk, *dv, *dg;
        ck(cudaMalloc(&dq, qn * 2)); ck(cudaMalloc(&dk, kn * 2)); ck(cudaMalloc(&dv, kn * 2)); ck(cudaMalloc(&dg, qn * 2));
        ck(cudaMemcpy(dq, q.data(), qn * 2, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(dk, k.data(), kn * 2, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(dv, v.data(), kn * 2, cudaMemcpyHostToDevice));
        const float scale = 1.f / std::sqrt(float(c.hd));
        if (!launch_attention_prefill_gemm_fp16(dq, dk, dv, dg, c.seq, c.past, c.heads, c.kvh, c.hd, c.slots,
                                                scale, c.window, st, c.slots, c.begin, c.end))
            throw std::runtime_error("bidirectional gemm prefill launch failed");
        ck(cudaStreamSynchronize(st));
        ck(cudaMemcpy(o.data(), dg, qn * 2, cudaMemcpyDeviceToHost));
        const int last = c.past + c.seq - 1;
        auto visible = [&](int pos, int p) {
            const bool in = pos >= c.begin && pos < c.end && p >= c.begin && p < c.end;
            return (p <= pos || in) && (c.window <= 0 || p > pos - c.window);
        };
        double err = 0, reach = 0;
        std::vector<double> sc(last + 1);
        // Before, first, middle and last token of the image, and after it.
        for (int pos : {c.begin - 1, c.begin, (c.begin + c.end) / 2, c.end - 1, c.end, last}) {
            if (pos < c.past || pos > last) continue;
            for (int h = 0; h < c.heads; h += std::max(1, c.heads / 5)) {
                const int qi = pos - c.past, kh = h / (c.heads / c.kvh);
                double mx = -1e300;
                for (int p = 0; p <= last; ++p) {
                    if (!visible(pos, p)) continue;
                    double d = 0;
                    const size_t kb = (size_t(p % c.slots) * c.kvh + kh) * c.hd;
                    for (int j = 0; j < c.hd; ++j) d += double(__half2float(q[(size_t(qi) * c.heads + h) * c.hd + j])) * __half2float(k[kb + j]);
                    sc[p] = d * scale; mx = std::max(mx, sc[p]);
                    if (p > pos) reach = std::max(reach, double(p - pos));
                }
                double sum = 0;
                for (int p = 0; p <= last; ++p) if (visible(pos, p)) { sc[p] = std::exp(sc[p] - mx); sum += sc[p]; }
                for (int j = 0; j < c.hd; ++j) {
                    double a = 0;
                    for (int p = 0; p <= last; ++p) if (visible(pos, p)) a += sc[p] * __half2float(v[(size_t(p % c.slots) * c.kvh + kh) * c.hd + j]);
                    const size_t i = (size_t(qi) * c.heads + h) * c.hd + j;
                    err = std::max(err, std::fabs(__half2float(o[i]) - a / sum));
                }
            }
        }
        const bool pass = std::isfinite(err) && err <= 3e-3 && reach > 0;
        ok &= pass;
        std::printf("%-16s past=%5d seq=%3d image=[%d,%d) looks_ahead=%.0f err_gemm=%.2e %s\n", c.name, c.past, c.seq,
                    c.begin, c.end, reach, err, pass ? "OK" : "FAIL");
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dg);
    }
    // A block outside the chunk would read keys that are not written yet.
    ok &= !launch_attention_prefill_gemm_fp16(nullptr, nullptr, nullptr, nullptr, 10, 100, 16, 8, 256, 1536,
                                              1.f, 1024, st, 1536, 90, 105);

    cudaStreamDestroy(st);
    std::printf(ok ? "ALL OK\n" : "FAILURES\n");
    return ok ? 0 : 1;
}
