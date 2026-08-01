#include "gemma4_vision_runner.hpp"
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void cuda_require(cudaError_t status, const std::string& operation) {
    require(status == cudaSuccess,
            operation + ": " + cudaGetErrorString(status));
}

template <typename T>
struct NpyArray {
    std::vector<size_t> shape;
    std::vector<T> values;
};

uint32_t little_u32(const uint8_t* bytes) {
    return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
           (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
}

template <typename T>
NpyArray<T> read_npy(const std::string& path,
                     const std::string& expected_descr) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "cannot open NPY: " + path);
    uint8_t prefix[12]{};
    input.read(reinterpret_cast<char*>(prefix), 10);
    require(input.gcount() == 10 &&
            std::memcmp(prefix, "\x93NUMPY", 6) == 0,
            "invalid NPY prefix: " + path);
    const uint8_t major = prefix[6];
    size_t header_length = 0;
    if (major == 1) {
        header_length = size_t(prefix[8]) | (size_t(prefix[9]) << 8);
    } else if (major == 2 || major == 3) {
        input.read(reinterpret_cast<char*>(prefix + 10), 2);
        require(input.gcount() == 2, "truncated NPY v2 header length");
        header_length = little_u32(prefix + 8);
    } else {
        require(false, "unsupported NPY version");
    }

    std::string header(header_length, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    require(input.gcount() == static_cast<std::streamsize>(header.size()),
            "truncated NPY header");
    require(header.find("'descr': '" + expected_descr + "'") !=
                std::string::npos,
            "unexpected NPY dtype in " + path);
    require(header.find("'fortran_order': False") != std::string::npos,
            "Fortran-order NPY is unsupported");

    const size_t shape_key = header.find("'shape':");
    const size_t open = header.find('(', shape_key);
    const size_t close = header.find(')', open);
    require(shape_key != std::string::npos && open != std::string::npos &&
            close != std::string::npos,
            "NPY shape is missing");
    NpyArray<T> result;
    size_t cursor = open + 1;
    while (cursor < close) {
        while (cursor < close &&
               (header[cursor] == ' ' || header[cursor] == ',')) ++cursor;
        if (cursor >= close) break;
        size_t end = cursor;
        while (end < close && header[end] >= '0' && header[end] <= '9') ++end;
        require(end > cursor, "invalid NPY shape");
        result.shape.push_back(static_cast<size_t>(
            std::stoull(header.substr(cursor, end - cursor))));
        cursor = end;
    }

    size_t elements = 1;
    for (size_t dimension : result.shape) {
        require(dimension != 0 &&
                elements <= std::numeric_limits<size_t>::max() / dimension,
                "NPY shape overflows");
        elements *= dimension;
    }
    result.values.resize(elements);
    input.read(reinterpret_cast<char*>(result.values.data()),
               static_cast<std::streamsize>(elements * sizeof(T)));
    require(input.gcount() ==
                static_cast<std::streamsize>(elements * sizeof(T)),
            "truncated NPY payload");
    return result;
}

void test_patch_kernels() {
    const std::vector<float> input{0.0f, 0.25f, 0.5f, 1.0f};
    float* d_input = nullptr;
    half* d_scaled = nullptr;
    cuda_require(cudaMalloc(&d_input, input.size() * sizeof(float)),
                 "allocate patch input");
    cuda_require(cudaMalloc(&d_scaled, input.size() * sizeof(half)),
                 "allocate scaled patch input");
    cuda_require(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float),
                            cudaMemcpyHostToDevice),
                 "upload patch input");
    helios::kernels::launch_gemma4_patch_input_fp16(
        d_input, d_scaled, input.size());
    std::vector<half> scaled(input.size());
    cuda_require(cudaMemcpy(scaled.data(), d_scaled,
                            scaled.size() * sizeof(half),
                            cudaMemcpyDeviceToHost),
                 "download scaled patch input");
    const float expected_scaled[] = {-1.0f, -0.5f, 0.0f, 1.0f};
    for (size_t i = 0; i < scaled.size(); ++i) {
        require(__half2float(scaled[i]) == expected_scaled[i],
                "patch input scaling differs from upstream");
    }
    cudaFree(d_input);
    cudaFree(d_scaled);

    constexpr uint32_t patches = 3;
    constexpr uint32_t hidden = 2;
    constexpr uint32_t table_size = 4;
    std::vector<half> hidden_values(patches * hidden);
    for (size_t i = 0; i < hidden_values.size(); ++i) {
        hidden_values[i] = __float2half(float(i));
    }
    std::vector<half> table(2 * table_size * hidden);
    for (uint32_t coordinate = 0; coordinate < table_size; ++coordinate) {
        for (uint32_t channel = 0; channel < hidden; ++channel) {
            table[size_t(coordinate) * hidden + channel] =
                __float2half(float(coordinate * 10 + channel));
            table[(size_t(table_size) + coordinate) * hidden + channel] =
                __float2half(float(100 + coordinate * 10 + channel));
        }
    }
    const std::vector<int32_t> positions{1, 2, -1, -1, 3, 0};
    half* d_hidden = nullptr;
    half* d_table = nullptr;
    int32_t* d_positions = nullptr;
    cuda_require(cudaMalloc(&d_hidden, hidden_values.size() * sizeof(half)),
                 "allocate synthetic hidden");
    cuda_require(cudaMalloc(&d_table, table.size() * sizeof(half)),
                 "allocate synthetic position table");
    cuda_require(cudaMalloc(&d_positions,
                            positions.size() * sizeof(int32_t)),
                 "allocate synthetic positions");
    cuda_require(cudaMemcpy(d_hidden, hidden_values.data(),
                            hidden_values.size() * sizeof(half),
                            cudaMemcpyHostToDevice), "upload synthetic hidden");
    cuda_require(cudaMemcpy(d_table, table.data(), table.size() * sizeof(half),
                            cudaMemcpyHostToDevice), "upload position table");
    cuda_require(cudaMemcpy(d_positions, positions.data(),
                            positions.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice), "upload positions");
    helios::kernels::launch_gemma4_add_xy_position_fp16(
        d_hidden, d_table, d_positions, patches, hidden, table_size);
    cuda_require(cudaMemcpy(hidden_values.data(), d_hidden,
                            hidden_values.size() * sizeof(half),
                            cudaMemcpyDeviceToHost),
                 "download positioned hidden");
    cudaFree(d_hidden);
    cudaFree(d_table);
    cudaFree(d_positions);

    const float expected[] = {130.0f, 133.0f, 2.0f, 3.0f, 134.0f, 137.0f};
    for (size_t i = 0; i < hidden_values.size(); ++i) {
        require(__half2float(hidden_values[i]) == expected[i],
                "learned XY embedding or padding behavior differs");
    }
    std::cout << "PASS: Gemma 4 patch scaling and learned XY kernels"
              << std::endl;
}

void compare_boundary(const std::string& label,
                      const std::vector<float>& actual,
                      const std::vector<float>& reference,
                      double maximum_nrmse,
                      double minimum_correlation) {
    require(actual.size() == reference.size(),
            label + " element count differs");
    double sum_actual = 0.0;
    double sum_reference = 0.0;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    float max_absolute_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), label + " contains non-finite value");
        sum_actual += actual[i];
        sum_reference += reference[i];
        const double difference = double(actual[i]) - reference[i];
        squared_error += difference * difference;
        squared_reference += double(reference[i]) * reference[i];
        max_absolute_error = std::max(max_absolute_error,
                                      std::fabs(actual[i] - reference[i]));
    }
    const double mean_actual = sum_actual / actual.size();
    const double mean_reference = sum_reference / actual.size();
    double covariance = 0.0;
    double variance_actual = 0.0;
    double variance_reference = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i] - mean_actual;
        const double r = reference[i] - mean_reference;
        covariance += a * r;
        variance_actual += a * a;
        variance_reference += r * r;
    }
    const double nrmse = std::sqrt(squared_error / squared_reference);
    const double correlation = covariance /
        std::sqrt(variance_actual * variance_reference);
    std::cout << std::setprecision(10)
              << "PASS: " << label
              << " NRMSE=" << nrmse
              << " correlation=" << correlation
              << " max_abs=" << max_absolute_error << std::endl;
    require(nrmse <= maximum_nrmse, label + " exceeds V0 NRMSE barrier");
    require(correlation >= minimum_correlation,
            label + " falls below V0 correlation barrier");
}

void test_real_frontier(const std::string& hnf_path,
                        const std::string& oracle_dir) {
    const auto patches = read_npy<float>(oracle_dir + "/patches.npy", "<f4");
    const auto positions = read_npy<int64_t>(oracle_dir + "/positions.npy", "<i8");
    const auto reference = read_npy<float>(
        oracle_dir + "/patch_embedder.npy", "<f4");
    const auto layer00 = read_npy<float>(oracle_dir + "/layer00.npy", "<f4");
    const auto layer15 = read_npy<float>(oracle_dir + "/layer15.npy", "<f4");
    const auto pooler = read_npy<float>(oracle_dir + "/pooler.npy", "<f4");
    const auto projected = read_npy<float>(
        oracle_dir + "/projected.npy", "<f4");
    require(patches.shape == std::vector<size_t>({1, 2520, 768}),
            "unexpected V0 patch shape");
    require(positions.shape == std::vector<size_t>({1, 2520, 2}),
            "unexpected V0 position shape");
    require(reference.shape == std::vector<size_t>({1, 2520, 768}),
            "unexpected V0 patch-embedder shape");
    require(layer00.shape == std::vector<size_t>({1, 2520, 768}),
            "unexpected V0 layer-0 shape");
    require(layer15.shape == std::vector<size_t>({1, 2520, 768}),
            "unexpected V0 layer-15 shape");
    require(pooler.shape == std::vector<size_t>({1, 280, 768}),
            "unexpected V0 pooler shape");
    require(projected.shape == std::vector<size_t>({280, 1536}),
            "unexpected V0 projected shape");

    helios::Gemma4VisionPreprocessResult input;
    input.max_patches = static_cast<uint32_t>(patches.shape[1]);
    input.real_patches = input.max_patches;
    input.patch_values = static_cast<uint32_t>(patches.shape[2]);
    input.pixel_values = patches.values;
    input.position_ids.reserve(positions.values.size());
    for (int64_t coordinate : positions.values) {
        require(coordinate >= std::numeric_limits<int32_t>::min() &&
                coordinate <= std::numeric_limits<int32_t>::max(),
                "V0 coordinate is outside int32");
        input.position_ids.push_back(static_cast<int32_t>(coordinate));
    }

    helios::Engine engine;
    helios::HnfLoader loader;
    require(loader.open(hnf_path),
            "cannot open visual HNF: " + loader.last_error());
    size_t free_before_load = 0;
    size_t total_vram = 0;
    cuda_require(cudaMemGetInfo(&free_before_load, &total_vram),
                 "sample VRAM before visual load");
    require(loader.load_block(helios::BLOCK_VISION, engine),
            "cannot load visual block: " + loader.last_error());
    size_t free_after_load = 0;
    cuda_require(cudaMemGetInfo(&free_after_load, &total_vram),
                 "sample VRAM after visual load");

    std::vector<float> actual;
    std::vector<float> actual_layer00;
    std::vector<float> actual_layer15;
    std::vector<float> actual_pooler;
    std::vector<float> actual_projected;
    size_t free_runner_peak = 0;
    {
        helios::Gemma4VisionRunner runner(engine, loader);
        std::string error;
        require(runner.run_patch_embedder(input, &error),
                "patch embedder failed: " + error);
        require(runner.copy_patch_embeddings(actual, &error),
                "patch boundary download failed: " + error);
        require(runner.run_encoder_layer(0, &error),
                "visual layer 0 failed: " + error);
        require(runner.copy_hidden_states(actual_layer00, &error),
                "layer-0 boundary download failed: " + error);
        for (uint32_t layer = 1; layer < 16; ++layer) {
            require(runner.run_encoder_layer(layer, &error),
                    "visual layer " + std::to_string(layer) +
                    " failed: " + error);
        }
        require(runner.copy_hidden_states(actual_layer15, &error),
                "layer-15 boundary download failed: " + error);
        require(runner.run_pooler(&error), "visual pooler failed: " + error);
        require(runner.copy_pooled_states(actual_pooler, &error),
                "pooler boundary download failed: " + error);
        require(runner.run_projector(&error),
                "visual projector failed: " + error);
        require(runner.copy_projected_states(actual_projected, &error),
                "projected boundary download failed: " + error);
        cuda_require(cudaMemGetInfo(&free_runner_peak, &total_vram),
                     "sample VRAM with visual runner");
    }
    require(loader.unload_block(helios::BLOCK_VISION, engine),
            "cannot unload visual block");
    compare_boundary("real Gemma 4 patch frontier", actual, reference.values,
                     0.005, 0.99999);
    compare_boundary("real Gemma 4 layer-0 frontier", actual_layer00,
                     layer00.values, 0.015, 0.9998);
    compare_boundary("real Gemma 4 layer-15 frontier", actual_layer15,
                     layer15.values, 0.030, 0.9990);
    compare_boundary("real Gemma 4 pooler frontier", actual_pooler,
                     pooler.values, 0.030, 0.9990);
    compare_boundary("real Gemma 4 projected frontier", actual_projected,
                     projected.values, 0.030, 0.9990);
    const size_t weight_vram = free_before_load > free_after_load
        ? free_before_load - free_after_load : 0;
    const size_t runner_vram = free_after_load > free_runner_peak
        ? free_after_load - free_runner_peak : 0;
    std::cout << "PASS: real Gemma 4 runner VRAM"
              << " weights=" << weight_vram
              << " scratch_and_cublas=" << runner_vram << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    test_patch_kernels();
    if (argc == 3) {
        test_real_frontier(argv[1], argv[2]);
    } else if (argc != 1) {
        std::cerr << "usage: " << argv[0] << " [vision.hnf oracle_dir]"
                  << std::endl;
        return 2;
    }
    return 0;
}
