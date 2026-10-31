#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace helios::kernels;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void cuda_require(cudaError_t error, const std::string& where) {
    require(error == cudaSuccess, where + ": " + cudaGetErrorString(error));
}

template <typename T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    cuda_require(cudaMalloc(&device, host.size() * sizeof(T)), "cudaMalloc");
    cuda_require(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return device;
}

template <typename T>
std::vector<T> download(const T* device, size_t count) {
    cuda_require(cudaDeviceSynchronize(), "synchronize");
    std::vector<T> host(count);
    cuda_require(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return host;
}

std::vector<half> halves(const std::vector<float>& values) {
    std::vector<half> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) result[i] = __float2half(values[i]);
    return result;
}

std::vector<float> floats(const std::vector<half>& values) {
    std::vector<float> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) result[i] = __half2float(values[i]);
    return result;
}

template <typename T>
std::vector<T> read(std::ifstream& file, size_t count) {
    std::vector<T> values(count);
    file.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    require(bool(file), "fixture truncado");
    return values;
}

double relative_error(const std::vector<float>& got, const std::vector<float>& want) {
    require(got.size() == want.size(), "tamaños distintos");
    double difference = 0, reference = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        difference += (double(got[i]) - want[i]) * (double(got[i]) - want[i]);
        reference += double(want[i]) * want[i];
    }
    return std::sqrt(difference / reference);
}

// Probabilidades iguales: gana el id menor, también cuando el empate cruza el puesto k.
void test_ties_use_the_lowest_id() {
    const std::vector<half> logits = halves({1, 3, 3, 0, 3, 3});
    const std::vector<half> scale = halves({1, 1, 1, 1, 1, 1});
    half* dl = upload(logits);
    half* ds = upload(scale);
    int32_t* di = nullptr;
    float* dw = nullptr;
    cuda_require(cudaMalloc(&di, 3 * sizeof(int32_t)), "indices");
    cuda_require(cudaMalloc(&dw, 3 * sizeof(float)), "weights");
    launch_moe_route(dl, ds, di, dw, 1, 6, 3, 0);
    const auto indices = download(di, 3);
    const auto weights = download(dw, 3);
    require(indices[0] == 1 && indices[1] == 2 && indices[2] == 4, "regla de empates");
    for (float weight : weights) require(std::fabs(weight - 1.0f / 3.0f) < 1e-6f, "pesos renormalizados");
    for (void* p : {static_cast<void*>(dl), static_cast<void*>(ds), static_cast<void*>(di), static_cast<void*>(dw)}) {
        cudaFree(p);
    }
}

void test_reference(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "no abro " + path);
    const auto head = read<int32_t>(file, 6);
    require(head[0] == 0x4D4F4531, "magic del fixture");
    const int L = head[1], E = head[2], K = head[3], H = head[4], I = head[5];
    const auto want_indices = read<int32_t>(file, size_t(L) * K);
    const auto residual = read<float>(file, size_t(L) * H);
    const auto router_scale = read<float>(file, H);
    const auto proj = read<float>(file, size_t(E) * H);
    const auto expert_scale = read<float>(file, E);
    const auto input = read<float>(file, size_t(L) * H);
    const auto gate_up = read<float>(file, size_t(E) * 2 * I * H);
    const auto down = read<float>(file, size_t(E) * H * I);
    const auto want_logits = read<float>(file, size_t(L) * E);
    const auto want_weights = read<float>(file, size_t(L) * K);
    const auto want_output = read<float>(file, size_t(L) * H);

    // Lo que hace Gemma4TextRouter antes de elegir, con los kernels que ya existen.
    half* dresidual = upload(halves(residual));
    half* dnorm = nullptr, *dscaled = nullptr, *dlogits = nullptr;
    cuda_require(cudaMalloc(&dnorm, size_t(L) * H * sizeof(half)), "norm");
    cuda_require(cudaMalloc(&dscaled, size_t(L) * H * sizeof(half)), "scaled");
    cuda_require(cudaMalloc(&dlogits, size_t(L) * E * sizeof(half)), "logits");
    launch_rmsnorm_no_weight_fp16(dresidual, dnorm, L, H, 1e-6f, 0);
    std::vector<float> full_scale(size_t(L) * H);
    for (int l = 0; l < L; ++l) {
        for (int h = 0; h < H; ++h) full_scale[size_t(l) * H + h] = router_scale[h] * std::pow(float(H), -0.5f);
    }
    half* dfull = upload(halves(full_scale));
    launch_mul_fp16(dnorm, dfull, dscaled, size_t(L) * H, 0);
    half* dproj = upload(halves(proj));
    launch_matmul_fp16_cublas(dscaled, dproj, dlogits, L, H, E, 0);
    const double logits_error = relative_error(floats(download(dlogits, size_t(L) * E)), want_logits);
    std::cout << "  logits: error relativo " << logits_error << std::endl;
    require(logits_error < 2e-3, "logits del router");

    half* dexpert_scale = upload(halves(expert_scale));
    int32_t* dindices = nullptr;
    float* dweights = nullptr;
    cuda_require(cudaMalloc(&dindices, size_t(L) * K * sizeof(int32_t)), "indices");
    cuda_require(cudaMalloc(&dweights, size_t(L) * K * sizeof(float)), "weights");
    launch_moe_route(dlogits, dexpert_scale, dindices, dweights, L, E, K, 0);
    const auto indices = download(dindices, size_t(L) * K);
    for (size_t i = 0; i < indices.size(); ++i) require(indices[i] == want_indices[i], "expertos elegidos");
    const double weights_error = relative_error(download(dweights, size_t(L) * K), want_weights);
    std::cout << "  pesos: error relativo " << weights_error << std::endl;
    require(weights_error < 2e-3, "pesos del router");

    // Con los pesos exactos: así el error que se mide es solo el de los expertos.
    float* dexact = upload(want_weights);
    half* dinput = upload(halves(input));
    half* dgate_up = upload(halves(gate_up));
    half* ddown = upload(halves(down));
    half* doutput = nullptr;
    cuda_require(cudaMalloc(&doutput, size_t(L) * H * sizeof(half)), "output");
    cudaStream_t stream;
    cuda_require(cudaStreamCreate(&stream), "stream");
    run_moe_experts_reference(dinput, dindices, dexact, dgate_up, ddown, doutput, L, E, K, H, I, stream);
    const double output_error = relative_error(floats(download(doutput, size_t(L) * H)), want_output);
    std::cout << "  expertos: error relativo " << output_error << std::endl;
    require(output_error < 2e-3, "salida de los expertos");
    cudaStreamDestroy(stream);
}

}  // namespace

int main(int argc, char** argv) {
    require(argc == 2, "uso: test_gemma4_moe fixture.bin");
    test_ties_use_the_lowest_id();
    test_reference(argv[1]);
    std::cout << "Gemma 4 MoE OK" << std::endl;
    return 0;
}
