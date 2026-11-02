// GPTQ sobre rejilla fija (src/gptq.hpp) contra una referencia de CPU en FP64.
// Uso: test_gptq [carpeta]. Con carpeta compara además con una matriz real exportada por
// la referencia de NumPy: dims.txt (filas columnas amortiguación), w.f64, h.f64, a.f32,
// m.f32 y codes.u8.
#include "gptq.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace helios::gptq;

static double snap(double w, double a, double m, double& q) {
    q = a > 0 ? std::fmin(std::fmax(std::nearbyint((w + m) / a), 0.0), 15.0) : 0.0;
    return a * q - m;
}

// La misma recurrencia que gptq_codes() de gptq_full.py, sin bloques diferidos.
static std::vector<uint8_t> reference(std::vector<double> w, std::vector<double> h, const std::vector<float>& a,
                                      const std::vector<float>& m, int rows, int cols, double damp) {
    double mean = 0;
    for (int i = 0; i < cols; ++i) { double& d = h[size_t(i) * cols + i]; if (d == 0) d = 1; mean += d; }
    mean /= cols;
    for (int i = 0; i < cols; ++i) h[size_t(i) * cols + i] += damp * mean;
    // L L' = H, H⁻¹ = L⁻ᵀ L⁻¹, y U superior con U'U = H⁻¹.
    std::vector<double> l(size_t(cols) * cols, 0.0);
    for (int j = 0; j < cols; ++j) {
        double s = h[size_t(j) * cols + j];
        for (int k = 0; k < j; ++k) s -= l[size_t(j) * cols + k] * l[size_t(j) * cols + k];
        l[size_t(j) * cols + j] = std::sqrt(s);
        for (int i = j + 1; i < cols; ++i) {
            double t = h[size_t(i) * cols + j];
            for (int k = 0; k < j; ++k) t -= l[size_t(i) * cols + k] * l[size_t(j) * cols + k];
            l[size_t(i) * cols + j] = t / l[size_t(j) * cols + j];
        }
    }
    std::vector<double> linv(size_t(cols) * cols, 0.0);
    for (int c = 0; c < cols; ++c)
        for (int i = c; i < cols; ++i) {
            double s = i == c ? 1.0 : 0.0;
            for (int k = c; k < i; ++k) s -= l[size_t(i) * cols + k] * linv[size_t(k) * cols + c];
            linv[size_t(i) * cols + c] = s / l[size_t(i) * cols + i];
        }
    std::vector<double> inv(size_t(cols) * cols, 0.0);
    for (int i = 0; i < cols; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = 0;
            for (int k = i; k < cols; ++k) s += linv[size_t(k) * cols + i] * linv[size_t(k) * cols + j];
            inv[size_t(i) * cols + j] = inv[size_t(j) * cols + i] = s;
        }
    // Cholesky de la inversa: G G' = H⁻¹, U = G'.
    std::vector<double> g(size_t(cols) * cols, 0.0);
    for (int j = 0; j < cols; ++j) {
        double s = inv[size_t(j) * cols + j];
        for (int k = 0; k < j; ++k) s -= g[size_t(j) * cols + k] * g[size_t(j) * cols + k];
        g[size_t(j) * cols + j] = std::sqrt(s);
        for (int i = j + 1; i < cols; ++i) {
            double t = inv[size_t(i) * cols + j];
            for (int k = 0; k < j; ++k) t -= g[size_t(i) * cols + k] * g[size_t(j) * cols + k];
            g[size_t(i) * cols + j] = t / g[size_t(j) * cols + j];
        }
    }
    auto u = [&](int i, int j) { return g[size_t(j) * cols + i]; };
    std::vector<uint8_t> codes(size_t(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        double* row = w.data() + size_t(r) * cols;
        for (int i = 0; i < cols; ++i) {
            double q;
            const double value = snap(row[i], a[size_t(r) * cols + i], m[size_t(r) * cols + i], q);
            codes[size_t(r) * cols + i] = uint8_t(q);
            const double e = (row[i] - value) / u(i, i);
            for (int j = i; j < cols; ++j) row[j] -= e * u(i, j);
        }
    }
    return codes;
}

template <class T>
static std::vector<T> read_file(const std::string& path, size_t count) {
    std::vector<T> out(count);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(out.data()), std::streamsize(count * sizeof(T)));
    if (!f) throw std::runtime_error("no leo " + path);
    return out;
}

static int compare(const char* name, const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, double min_equal) {
    size_t equal = 0;
    for (size_t i = 0; i < got.size(); ++i) equal += got[i] == want[i];
    const double share = double(equal) / got.size();
    std::printf("%-34s %.5f %% de códigos iguales\n", name, share * 100);
    return share >= min_equal ? 0 : 1;
}

int main(int argc, char** argv) try {
    int failures = 0;
    std::mt19937 rng(20261102);
    std::normal_distribution<double> nd(0.0, 1.0);
    const int rows = 96, cols = 384;
    std::vector<double> w(size_t(rows) * cols);
    for (auto& v : w) v = nd(rng) * 0.02;
    std::vector<float> a(w.size()), m(w.size());
    for (int r = 0; r < rows; ++r)
        for (int g = 0; g < cols / 16; ++g) {
            double lo = 1e9, hi = -1e9;
            for (int i = 0; i < 16; ++i) { const double v = w[size_t(r) * cols + g * 16 + i]; lo = std::fmin(lo, v); hi = std::fmax(hi, v); }
            for (int i = 0; i < 16; ++i) { a[size_t(r) * cols + g * 16 + i] = float((hi - lo) / 15); m[size_t(r) * cols + g * 16 + i] = float(-lo); }
        }
    // Columnas de cero en una fila: paso nulo, código 0.
    for (int c = 0; c < 16; ++c) a[size_t(5) * cols + c] = 0.f;
    Quantizer quantizer(4096, 2816);

    // 1. H = identidad: cada código es el redondeo independiente.
    std::vector<double> eye(size_t(cols) * cols, 0.0);
    for (int i = 0; i < cols; ++i) eye[size_t(i) * cols + i] = 1.0;
    std::vector<uint8_t> rtn(w.size());
    for (size_t i = 0; i < w.size(); ++i) { double q; snap(w[i], a[i], m[i], q); rtn[i] = uint8_t(q); }
    failures += compare("H identidad frente a redondeo", quantizer.run_host(w, eye, a, m, rows, cols, 1.0), rtn, 1.0);

    // 2. H densa con columnas muertas frente a la referencia de CPU.
    const int samples = 300;
    std::vector<double> x(size_t(samples) * cols);
    for (int s = 0; s < samples; ++s)
        for (int c = 0; c < cols; ++c) x[size_t(s) * cols + c] = (c % 97 == 3) ? 0.0 : nd(rng) * (0.2 + 3.0 * c / cols);
    std::vector<double> h(size_t(cols) * cols, 0.0);
    for (int i = 0; i < cols; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = 0;
            for (int k = 0; k < samples; ++k) s += x[size_t(k) * cols + i] * x[size_t(k) * cols + j];
            h[size_t(i) * cols + j] = h[size_t(j) * cols + i] = s;
        }
    for (double damp : {0.01, 1.0}) {
        const auto got = quantizer.run_host(w, h, a, m, rows, cols, damp);
        failures += compare(damp < 0.1 ? "H densa, amortiguación 0,01" : "H densa, amortiguación 1",
                            got, reference(w, h, a, m, rows, cols, damp), 0.999);
    }

    // 3. Empaquetado HQ44: leer la rejilla y los códigos y volver a escribirlos.
    std::vector<uint8_t> blocks(size_t(2) * 3 * HQ44_BLOCK_BYTES);
    for (auto& b : blocks) b = uint8_t(rng());
    for (size_t b = 0; b < blocks.size(); b += HQ44_BLOCK_BYTES) { blocks[b + 1] = 0x20; blocks[b + 3] = 0x18; }
    std::vector<uint8_t> codes(size_t(2) * 768), back(codes.size());
    hq44_codes(blocks.data(), 2, 768, codes.data());
    std::vector<uint8_t> logical(size_t(2) * 704), copy = blocks;
    for (auto& c : logical) c = uint8_t(rng() & 15);
    hq44_set_codes(copy.data(), logical.data(), 2, 768, 704);
    hq44_codes(copy.data(), 2, 768, back.data());
    size_t wrong = 0;
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 768; ++c)
            wrong += back[size_t(r) * 768 + c] != (c < 704 ? logical[size_t(r) * 704 + c] : codes[size_t(r) * 768 + c]);
    for (size_t b = 0; b < blocks.size(); b += HQ44_BLOCK_BYTES)
        for (int i = 0; i < 24; ++i) wrong += copy[b + i] != blocks[b + i];
    std::printf("%-34s %zu errores\n", "empaquetado HQ44", wrong);
    failures += wrong != 0;
    std::vector<float> ga(size_t(2) * 768), gm(ga.size());
    hq44_grid(blocks.data(), 2, 768, ga.data(), gm.data());
    failures += !(std::isfinite(ga[0]) && std::isfinite(gm[767]));

    // 4. Opcional: matriz real exportada por la referencia de NumPy.
    if (argc > 1) {
        const std::string dir = argv[1];
        int rr, cc; double damp;
        std::ifstream dims(dir + "/dims.txt");
        if (!(dims >> rr >> cc >> damp)) throw std::runtime_error("dims.txt");
        const size_t n = size_t(rr) * cc;
        const auto got = quantizer.run_host(read_file<double>(dir + "/w.f64", n), read_file<double>(dir + "/h.f64", size_t(cc) * cc),
                                            read_file<float>(dir + "/a.f32", n), read_file<float>(dir + "/m.f32", n), rr, cc, damp);
        failures += compare("matriz real frente a NumPy", got, read_file<uint8_t>(dir + "/codes.u8", n), 0.999);
    }
    std::printf(failures ? "FAILED (%d)\n" : "OK\n", failures);
    return failures ? 1 : 0;
} catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
