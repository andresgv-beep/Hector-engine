// Gated DeltaNet: los dos kernels contra un cálculo en CPU en doble precisión, y la propiedad
// de la que depende que prefill y decode coincidan: procesar los tokens de golpe o por partes
// deja el mismo resultado y el mismo estado.
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace helios::kernels;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { std::cerr << "FAIL: " << message << std::endl; std::exit(1); }
}
void ok(cudaError_t error, const std::string& where) {
    require(error == cudaSuccess, where + ": " + cudaGetErrorString(error));
}

template <typename T> struct Device {
    T* p = nullptr; size_t n = 0;
    explicit Device(size_t count) : n(count) { ok(cudaMalloc(&p, count * sizeof(T)), "cudaMalloc"); ok(cudaMemset(p, 0, count * sizeof(T)), "memset"); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T>& v) { require(v.size() == n, "tamaño"); ok(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice), "subir"); }
    std::vector<T> get() const { std::vector<T> v(n); ok(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "bajar"); return v; }
};

// Los valores pasan por FP16 antes de usarse en la CPU: la referencia ve lo mismo que la GPU.
std::vector<half> halves(std::mt19937& rng, size_t n, float scale, std::vector<double>* seen) {
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<half> v(n);
    seen->resize(n);
    for (size_t i = 0; i < n; ++i) { v[i] = __float2half(d(rng)); (*seen)[i] = double(__half2float(v[i])); }
    return v;
}
std::vector<double> as_double(const std::vector<half>& v) {
    std::vector<double> d(v.size());
    for (size_t i = 0; i < v.size(); ++i) d[i] = double(__half2float(v[i]));
    return d;
}
double silu(double x) { return x / (1.0 + std::exp(-x)); }

double worst(const std::vector<half>& got, const std::vector<double>& want, const std::string& what) {
    require(got.size() == want.size(), what + ": tamaños distintos");
    double scale = 0.0, error = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        require(std::isfinite(__half2float(got[i])), what + ": valor no finito");
        scale = std::max(scale, std::abs(want[i]));
        error = std::max(error, std::abs(double(__half2float(got[i])) - want[i]));
    }
    return error / std::max(scale, 1e-9);
}

struct Shape { int L, nk, nv, dk, dv, K; };

void conv_case(const Shape& s, std::mt19937& rng) {
    const int C = 2 * s.nk * s.dk + s.nv * s.dv;
    std::vector<double> x, w;
    const auto xh = halves(rng, size_t(s.L) * C, 1.0f, &x);
    const auto wh = halves(rng, size_t(C) * s.K, 0.5f, &w);
    // Referencia: la convolución sobre la secuencia entera, con ceros antes del primer token.
    std::vector<double> want(size_t(s.L) * C);
    for (int t = 0; t < s.L; ++t) for (int c = 0; c < C; ++c) {
        double sum = 0.0;
        for (int j = 0; j < s.K; ++j) {
            const int tau = t - (s.K - 1) + j;
            if (tau >= 0) sum += w[size_t(c) * s.K + j] * x[size_t(tau) * C + c];
        }
        want[size_t(t) * C + c] = silu(sum);
    }
    Device<half> dx(xh.size()), dw(wh.size()), dout(xh.size());
    dx.put(xh); dw.put(wh);
    // De golpe.
    Device<float> state(size_t(C) * (s.K - 1));
    launch_causal_conv_silu(dx.p, dw.p, state.p, dout.p, s.L, C, s.K, nullptr);
    ok(cudaDeviceSynchronize(), "conv");
    const double whole = worst(dout.get(), want, "convolución de golpe");
    require(whole < 2e-3, "convolución de golpe: error relativo " + std::to_string(whole));
    const auto state_whole = state.get();
    // Por partes: 1 token, luego 2, luego el resto. Cubre partes más cortas que la memoria.
    Device<float> state2(size_t(C) * (s.K - 1));
    Device<half> dout2(xh.size());
    int at = 0;
    for (int part : {1, 2, s.L - 3}) {
        if (part <= 0) continue;
        launch_causal_conv_silu(dx.p + size_t(at) * C, dw.p, state2.p, dout2.p + size_t(at) * C, part, C, s.K, nullptr);
        at += part;
    }
    ok(cudaDeviceSynchronize(), "conv por partes");
    require(at == s.L, "las partes no suman L");
    const auto a = dout.get(), b = dout2.get();
    for (size_t i = 0; i < a.size(); ++i) {
        require(__half2float(a[i]) == __half2float(b[i]), "convolución: por partes no da lo mismo que de golpe");
    }
    const auto state_parts = state2.get();
    for (size_t i = 0; i < state_whole.size(); ++i) require(state_whole[i] == state_parts[i], "convolución: el estado final difiere");
    // El estado final son las K-1 últimas entradas.
    for (int c = 0; c < C; ++c) for (int j = 0; j < s.K - 1; ++j) {
        const int tau = s.L - (s.K - 1) + j;
        const double want_state = tau >= 0 ? x[size_t(tau) * C + c] : 0.0;
        require(std::abs(double(state_whole[size_t(c) * (s.K - 1) + j]) - want_state) < 1e-6, "convolución: el estado no son las últimas entradas");
    }
    std::cout << "  convolución L=" << s.L << " C=" << C << " K=" << s.K << ": error " << whole << std::endl;
}

void delta_case(const Shape& s, std::mt19937& rng) {
    const int key_dim = s.nk * s.dk, value_dim = s.nv * s.dv, C = 2 * key_dim + value_dim;
    const double eps = 1e-6;
    std::vector<double> mixed, z, a, b, a_log, dt_bias, norm_w;
    const auto mixed_h = halves(rng, size_t(s.L) * C, 1.0f, &mixed);
    const auto z_h = halves(rng, size_t(s.L) * value_dim, 1.0f, &z);
    const auto a_h = halves(rng, size_t(s.L) * s.nv, 1.5f, &a);
    const auto b_h = halves(rng, size_t(s.L) * s.nv, 1.5f, &b);
    const auto a_log_h = halves(rng, s.nv, 0.7f, &a_log);
    const auto dt_h = halves(rng, s.nv, 0.7f, &dt_bias);
    auto norm_h = halves(rng, s.dv, 0.2f, &norm_w);
    for (int j = 0; j < s.dv; ++j) norm_h[j] = __float2half(1.0f + __half2float(norm_h[j]));
    norm_w = as_double(norm_h);

    // Referencia: modeling_qwen3_5.torch_recurrent_gated_delta_rule y la norma con puerta.
    std::vector<double> S(size_t(s.nv) * s.dk * s.dv, 0.0), want(size_t(s.L) * value_dim);
    for (int t = 0; t < s.L; ++t) for (int h = 0; h < s.nv; ++h) {
        const int kh = h / (s.nv / s.nk);
        std::vector<double> q(s.dk), k(s.dk);
        double qq = 0.0, kk = 0.0;
        for (int i = 0; i < s.dk; ++i) {
            q[i] = mixed[size_t(t) * C + kh * s.dk + i]; k[i] = mixed[size_t(t) * C + key_dim + kh * s.dk + i];
            qq += q[i] * q[i]; kk += k[i] * k[i];
        }
        for (int i = 0; i < s.dk; ++i) { q[i] = q[i] / std::sqrt(qq + 1e-6) / std::sqrt(double(s.dk)); k[i] = k[i] / std::sqrt(kk + 1e-6); }
        const double pre = a[size_t(t) * s.nv + h] + dt_bias[h];
        const double decay = std::exp(-std::exp(a_log[h]) * std::log1p(std::exp(pre)));
        const double beta = 1.0 / (1.0 + std::exp(-b[size_t(t) * s.nv + h]));
        double* Sh = S.data() + size_t(h) * s.dk * s.dv;
        std::vector<double> o(s.dv);
        double ss = 0.0;
        for (int j = 0; j < s.dv; ++j) {
            double remembered = 0.0;
            for (int i = 0; i < s.dk; ++i) remembered += Sh[size_t(i) * s.dv + j] * decay * k[i];
            const double delta = (mixed[size_t(t) * C + 2 * key_dim + h * s.dv + j] - remembered) * beta;
            for (int i = 0; i < s.dk; ++i) {
                Sh[size_t(i) * s.dv + j] = Sh[size_t(i) * s.dv + j] * decay + k[i] * delta;
                o[j] += Sh[size_t(i) * s.dv + j] * q[i];
            }
            ss += o[j] * o[j];
        }
        for (int j = 0; j < s.dv; ++j) {
            want[size_t(t) * value_dim + h * s.dv + j] =
                o[j] / std::sqrt(ss / s.dv + eps) * norm_w[j] * silu(z[size_t(t) * value_dim + h * s.dv + j]);
        }
    }

    Device<half> dm(mixed_h.size()), dz(z_h.size()), da(a_h.size()), db(b_h.size()), dal(s.nv), ddt(s.nv), dn(s.dv);
    dm.put(mixed_h); dz.put(z_h); da.put(a_h); db.put(b_h); dal.put(a_log_h); ddt.put(dt_h); dn.put(norm_h);
    Device<float> state(S.size());
    Device<half> dout(want.size());
    launch_gated_delta_rule(dm.p, dz.p, da.p, db.p, dal.p, ddt.p, dn.p, state.p, dout.p,
                            s.L, s.nk, s.nv, s.dk, s.dv, float(eps), nullptr);
    ok(cudaDeviceSynchronize(), "regla delta");
    const double whole = worst(dout.get(), want, "regla delta de golpe");
    require(whole < 3e-3, "regla delta de golpe: error relativo " + std::to_string(whole));
    const auto got_state = state.get();
    double state_scale = 0.0, state_error = 0.0;
    for (size_t i = 0; i < S.size(); ++i) {
        state_scale = std::max(state_scale, std::abs(S[i]));
        state_error = std::max(state_error, std::abs(double(got_state[i]) - S[i]));
    }
    require(state_error / state_scale < 1e-4, "regla delta: el estado final se aparta " + std::to_string(state_error / state_scale));

    // De uno en uno, como en decode: mismo resultado y mismo estado, bit a bit.
    Device<float> state2(S.size());
    Device<half> dout2(want.size());
    for (int t = 0; t < s.L; ++t) {
        launch_gated_delta_rule(dm.p + size_t(t) * C, dz.p + size_t(t) * value_dim, da.p + size_t(t) * s.nv, db.p + size_t(t) * s.nv,
                                dal.p, ddt.p, dn.p, state2.p, dout2.p + size_t(t) * value_dim,
                                1, s.nk, s.nv, s.dk, s.dv, float(eps), nullptr);
    }
    ok(cudaDeviceSynchronize(), "regla delta token a token");
    const auto x = dout.get(), y = dout2.get();
    for (size_t i = 0; i < x.size(); ++i) require(__half2float(x[i]) == __half2float(y[i]), "regla delta: token a token no da lo mismo que de golpe");
    const auto state_steps = state2.get();
    for (size_t i = 0; i < got_state.size(); ++i) require(got_state[i] == state_steps[i], "regla delta: el estado token a token difiere");
    std::cout << "  regla delta L=" << s.L << " nk=" << s.nk << " nv=" << s.nv << " dk=" << s.dk << " dv=" << s.dv
              << ": error " << whole << ", estado " << state_error / state_scale << std::endl;
}

} // namespace

int main() {
    std::mt19937 rng(20261116);
    // Las formas de la familia: 0.8B (16 y 16 cabezas), 35B (16 de clave para 32 de valor), y una
    // pequeña y asimétrica para que un índice cruzado no pase desapercibido.
    for (const Shape& s : {Shape{9, 16, 16, 128, 128, 4}, Shape{6, 16, 32, 128, 128, 4}, Shape{11, 2, 6, 24, 40, 3}}) {
        conv_case(s, rng);
        delta_case(s, rng);
    }
    bool rejected = false;
    try { launch_gated_delta_rule(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, 3, 4, 8, 8, 1e-6f, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "cabezas de valor que no son múltiplo de las de clave deben rechazarse");
    std::cout << "DeltaNet: bien" << std::endl;
    return 0;
}
