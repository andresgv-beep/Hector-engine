// The tuned HQ GEMV variants must agree bit for bit: autotuning chooses speed,
// never the result. The tune sidecar is read once per process, so the test
// runs itself once per forced variant and compares the outputs.
#include "kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>
#include <vector>
using namespace helios::kernels;

namespace {
struct Format {
    const char* name; int block; bool symmetric;
    void (*launch)(const half*, const uint8_t*, half*, int, int, int, cudaStream_t);
};
const Format kFormats[] = {{"hq41k", 168, false, launch_matmul_hq41k}, {"hq51k", 200, false, launch_matmul_hq51k},
                           {"hq42k", 152, true, launch_matmul_hq42k}, {"hq52k", 184, true, launch_matmul_hq52k}};
// Odd/even superblock counts, a partial last superblock and several input chunks.
const int kK[] = {256, 1000, 2560, 3840, 4096, 15360, 16384, 16640, 17152, 33000};
const int kN[] = {1, 3, 64, 513};

uint32_t seed = 2026;
float rnd() { seed = 1664525u * seed + 1013904223u; return float((seed >> 8) & 65535) / 65536.f; }

void put_half(uint8_t* p, float v) { const half h = __float2half(v); std::memcpy(p, &h, 2); }

int run_variant(char variant, const char* out_path) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 2;
    {
        std::ofstream t(std::string(std::getenv("HELIOS_HOME")) + "/tune.cache");
        t << "# helios tune.cache v1 " << prop.name << "\n";
        for (const auto& f : kFormats) for (int K : kK) for (int N : kN)
            t << f.name << ' ' << K << ' ' << N << ' ' << variant << "\n";
    }
    FILE* out = std::fopen(out_path, "wb");
    for (const auto& f : kFormats) for (int K : kK) for (int N : kN) {
        const int sb = (K + 255) / 256;
        std::vector<uint8_t> w(size_t(N) * sb * f.block);
        for (auto& b : w) b = uint8_t(rnd() * 256);
        for (size_t b = 0; b < size_t(N) * sb; ++b) {
            uint8_t* blk = w.data() + b * f.block;
            if (f.symmetric) put_half(blk, 0.001f + 0.02f * rnd());
            else { put_half(blk, 0.001f + 0.05f * rnd()); put_half(blk + 2, 0.02f * rnd()); put_half(blk + 4, -0.03f * rnd()); }
        }
        std::vector<half> x(K);
        for (auto& v : x) v = __float2half(2.f * rnd() - 1.f);
        half *dx, *dy; uint8_t* dw;
        cudaMalloc(&dx, K * 2); cudaMalloc(&dy, N * 2); cudaMalloc(&dw, w.size());
        cudaMemcpy(dx, x.data(), K * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
        f.launch(dx, dw, dy, 1, K, N, nullptr);
        std::vector<half> y(N);
        cudaMemcpy(y.data(), dy, N * 2, cudaMemcpyDeviceToHost);
        std::fwrite(y.data(), 2, N, out);
        cudaFree(dx); cudaFree(dy); cudaFree(dw);
    }
    std::fclose(out);
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : 1;
}

std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 3) return run_variant(argv[1][0], argv[2]);
    char dir[] = "/tmp/helios-gemv-variants-XXXXXX";
    if (!mkdtemp(dir)) return 2;
    std::string outputs[3];
    for (int v = 0; v < 3; ++v) {
        const char letter = char('A' + v);
        const std::string home = std::string(dir) + "/" + letter;
        const std::string cmd = "mkdir -p '" + home + "' && HELIOS_HOME='" + home + "' '" + argv[0] + "' " +
                                letter + " '" + home + "/out.bin'";
        if (std::system(cmd.c_str()) != 0) { std::fprintf(stderr, "FAIL: variant %c did not run\n", letter); return 1; }
        outputs[v] = slurp(home + "/out.bin");
    }
    std::system((std::string("rm -rf '") + dir + "'").c_str());
    const bool ok = !outputs[0].empty() && outputs[0] == outputs[1] && outputs[1] == outputs[2];
    std::printf("%s: GEMV variants A/B/C %s over %zu output bytes\n", ok ? "PASS" : "FAIL",
                ok ? "bit-identical" : "differ", outputs[0].size());
    return ok ? 0 : 1;
}
