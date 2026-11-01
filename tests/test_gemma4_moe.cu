#include "kernels.hpp"
#include "hqs_v4_gemv.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
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
T* device_array(size_t count) {
    T* device = nullptr;
    cuda_require(cudaMalloc(&device, count * sizeof(T)), "cudaMalloc");
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
    run_moe_experts_reference(dinput, dindices, dexact, dgate_up, ddown, doutput, L, E, K, H, I, I, stream);
    const double output_error = relative_error(floats(download(doutput, size_t(L) * H)), want_output);
    std::cout << "  expertos: error relativo " << output_error << std::endl;
    require(output_error < 2e-3, "salida de los expertos");
    cudaStreamDestroy(stream);
}

// El grafo se graba una vez y solo cambian los datos: si el kernel siguiera a los
// argumentos capturados, las repeticiones no coincidirían con la referencia.
void test_graph_replay_follows_the_router(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "no abro " + path);
    const auto head = read<int32_t>(file, 6);
    const int L = head[1], E = head[2], K = head[3], H = head[4], I = head[5];
    read<int32_t>(file, size_t(L) * K);
    read<float>(file, size_t(L) * H);
    read<float>(file, H);
    read<float>(file, size_t(E) * H);
    const auto expert_scale = read<float>(file, E);
    const auto input = read<float>(file, size_t(L) * H);
    const auto gate_up = read<float>(file, size_t(E) * 2 * I * H);
    const auto down = read<float>(file, size_t(E) * H * I);

    half* dscale = upload(halves(expert_scale));
    half* dgate_up = upload(halves(gate_up));
    half* ddown = upload(halves(down));
    half* dlogits = device_array<half>(E);
    half* dinput = device_array<half>(H);
    int32_t* dindices = device_array<int32_t>(K);
    float* dweights = device_array<float>(K);
    half* s1 = device_array<half>(size_t(K) * 2 * I);
    half* s2 = device_array<half>(size_t(K) * I);
    half* s3 = device_array<half>(size_t(K) * H);
    half* doutput = device_array<half>(H);
    half* dreference = device_array<half>(H);

    cudaStream_t stream, capture;
    cuda_require(cudaStreamCreate(&stream), "stream");
    cuda_require(cudaStreamCreate(&capture), "capture stream");
    cuda_require(cudaStreamBeginCapture(capture, cudaStreamCaptureModeThreadLocal), "begin capture");
    launch_moe_route(dlogits, dscale, dindices, dweights, 1, E, K, capture);
    launch_moe_experts_decode(dinput, dindices, dweights, dgate_up, false, ddown, false, s1, s2, s3, doutput,
                              K, H, I, I, capture);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cuda_require(cudaStreamEndCapture(capture, &graph), "end capture");
    cuda_require(cudaGraphInstantiate(&exec, graph, 0), "instantiate");

    std::vector<std::vector<int32_t>> seen;
    for (int token = 0; token < L; ++token) {
        std::vector<float> logits(E);
        for (int e = 0; e < E; ++e) logits[e] = std::sin(float(e * (token + 3)) * 0.7f) * 4.0f;
        const auto host_logits = halves(logits);
        const std::vector<half> row = halves(std::vector<float>(input.begin() + size_t(token) * H,
                                                                input.begin() + size_t(token + 1) * H));
        cuda_require(cudaMemcpy(dlogits, host_logits.data(), E * sizeof(half), cudaMemcpyHostToDevice), "logits");
        cuda_require(cudaMemcpy(dinput, row.data(), H * sizeof(half), cudaMemcpyHostToDevice), "input");
        cuda_require(cudaGraphLaunch(exec, stream), "graph launch");
        cuda_require(cudaStreamSynchronize(stream), "graph sync");
        seen.push_back(download(dindices, K));
        const auto got = floats(download(doutput, H));
        run_moe_experts_reference(dinput, dindices, dweights, dgate_up, ddown, dreference, 1, E, K, H, I, I, stream);
        const double error = relative_error(got, floats(download(dreference, H)));
        require(error < 1e-3, "replay frente a la referencia, token " + std::to_string(token));
    }
    bool changed = false;
    for (size_t i = 1; i < seen.size(); ++i) changed = changed || seen[i] != seen[0];
    require(changed, "la prueba debe cambiar de expertos entre repeticiones");
    std::cout << "  replay: " << seen.size() << " tokens con expertos distintos en un grafo grabado" << std::endl;
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(capture);
    cudaStreamDestroy(stream);
}

// Superbloques HQ44 al azar, con la cola de down llena de pesos no nulos: solo el
// cero de la entrada puede anular esas columnas.
std::vector<uint8_t> random_hq44(std::mt19937& rng, size_t blocks) {
    std::vector<uint8_t> bytes(blocks * hqs_v4::BLOCK_BYTES);
    for (uint8_t& byte : bytes) byte = uint8_t(rng());
    const uint16_t d = __half_as_ushort(__float2half(0.002f));
    const uint16_t dmin = __half_as_ushort(__float2half(0.003f));
    for (size_t block = 0; block < blocks; ++block) {
        std::memcpy(&bytes[block * hqs_v4::BLOCK_BYTES], &d, sizeof(d));
        std::memcpy(&bytes[block * hqs_v4::BLOCK_BYTES + 2], &dmin, sizeof(dmin));
    }
    return bytes;
}

void test_nan_logits_keep_valid_ids() {
    constexpr int E = 16, K = 4;
    const std::vector<float> nans(E, std::nanf(""));
    const std::vector<float> scale(E, 1.0f);
    half* dlogits = upload(halves(nans));
    half* dscale = upload(halves(scale));
    int32_t* dindices = device_array<int32_t>(K);
    float* dweights = device_array<float>(K);
    launch_moe_route(dlogits, dscale, dindices, dweights, 1, E, K, nullptr);
    for (int32_t id : download(dindices, K)) require(id >= 0 && id < E, "id válido con logits NaN");
    std::cout << "  logits NaN: ids validos" << std::endl;
}

void test_hq44_decode_follows_the_indices() {
    constexpr int E = 12, K = 4, H = 512, I = 176, P = 256;
    constexpr int warps = 1, rows_per_block = 8, unroll = 4;
    std::mt19937 rng(11);
    const size_t gate_up_stride = size_t(2) * I * (H / 256) * hqs_v4::BLOCK_BYTES;
    const size_t down_stride = size_t(H) * (P / 256) * hqs_v4::BLOCK_BYTES;
    uint8_t* dgate_up = upload(random_hq44(rng, size_t(E) * 2 * I * (H / 256)));
    uint8_t* ddown = upload(random_hq44(rng, size_t(E) * H * (P / 256)));

    std::vector<float> row(H);
    for (float& value : row) value = float(int(rng() % 2001) - 1000) * 1.0e-3f;
    half* dinput = upload(halves(row));
    int32_t* dindices = device_array<int32_t>(K);
    float* dweights = device_array<float>(K);
    half* s1 = device_array<half>(size_t(K) * 2 * I);
    half* s2 = device_array<half>(size_t(K) * P);
    half* s3 = device_array<half>(size_t(K) * H);
    half* doutput = device_array<half>(H);
    half* projected = device_array<half>(2 * I);
    half* activated = device_array<half>(P);
    half* expert_row = device_array<half>(H);

    cudaStream_t stream, capture;
    cuda_require(cudaStreamCreate(&stream), "stream");
    cuda_require(cudaStreamCreate(&capture), "capture stream");
    cuda_require(cudaStreamBeginCapture(capture, cudaStreamCaptureModeThreadLocal), "begin capture");
    launch_moe_experts_decode(dinput, dindices, dweights, dgate_up, true, ddown, true, s1, s2, s3, doutput,
                              K, H, I, P, capture);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cuda_require(cudaStreamEndCapture(capture, &graph), "end capture");
    cuda_require(cudaGraphInstantiate(&exec, graph, 0), "instantiate");

    const std::vector<std::vector<int32_t>> choices{{0, 3, 7, 11}, {11, 2, 5, 1}, {4, 9, 0, 6}};
    const std::vector<float> weights{0.4f, 0.3f, 0.2f, 0.1f};
    cuda_require(cudaMemcpy(dweights, weights.data(), K * sizeof(float), cudaMemcpyHostToDevice), "weights");
    for (const auto& ids : choices) {
        cuda_require(cudaMemcpy(dindices, ids.data(), K * sizeof(int32_t), cudaMemcpyHostToDevice), "indices");
        cuda_require(cudaGraphLaunch(exec, stream), "graph launch");
        cuda_require(cudaStreamSynchronize(stream), "graph sync");
        const auto hidden = download(s2, size_t(K) * P);
        for (int k = 0; k < K; ++k) {
            for (int c = I; c < P; ++c) {
                require(__half_as_ushort(hidden[size_t(k) * P + c]) == 0, "cola de relleno a cero");
            }
        }

        // La misma cuenta con el GEMV sin indexar, un experto cada vez.
        std::vector<float> want(H, 0.0f);
        for (int k = 0; k < K; ++k) {
            hqs_v4::launch_gemv2<warps, rows_per_block, unroll>(
                dinput, dgate_up + ids[k] * gate_up_stride, projected, H, 2 * I, stream);
            launch_moe_gate_up_mul(projected, activated, 1, I, P, stream);
            hqs_v4::launch_gemv2<warps, rows_per_block, unroll>(
                activated, ddown + ids[k] * down_stride, expert_row, P, H, stream);
            const auto got_row = floats(download(s3 + size_t(k) * H, H));
            const auto want_row = floats(download(expert_row, H));
            require(got_row == want_row, "fila del experto " + std::to_string(ids[k]));
            for (int c = 0; c < H; ++c) want[c] += want_row[c] * weights[k];
        }
        require(relative_error(floats(download(doutput, H)), want) < 1e-3, "suma ponderada HQ44");
    }
    std::cout << "  HQ44: " << choices.size() << " repeticiones identicas al GEMV por experto" << std::endl;
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(capture);
    cudaStreamDestroy(stream);
}

}  // namespace

int main(int argc, char** argv) {
    require(argc == 2, "uso: test_gemma4_moe fixture.bin");
    test_ties_use_the_lowest_id();
    test_reference(argv[1]);
    test_graph_replay_follows_the_router(argv[1]);
    test_nan_logits_keep_valid_ids();
    test_hq44_decode_follows_the_indices();
    std::cout << "Gemma 4 MoE OK" << std::endl;
    return 0;
}
