// Gemma 4 12B unified image/audio embedders against the Python oracle
// (tools/gemma4_unified_mm_oracle.py). Without arguments only the weight-free
// preprocessing contract runs; with `mm.hnf oracle_dir` every stage is compared.
#include "gemma4_unified_embedder.hpp"

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
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

template <typename T>
struct NpyArray {
    std::vector<size_t> shape;
    std::vector<T> values;
};

template <typename T>
NpyArray<T> read_npy(const std::string& path, const std::string& descr) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "cannot open NPY: " + path);
    uint8_t prefix[12]{};
    input.read(reinterpret_cast<char*>(prefix), 10);
    require(input.gcount() == 10 && std::memcmp(prefix, "\x93NUMPY", 6) == 0,
            "invalid NPY prefix: " + path);
    size_t header_length = size_t(prefix[8]) | (size_t(prefix[9]) << 8);
    if (prefix[6] != 1) {
        input.read(reinterpret_cast<char*>(prefix + 10), 2);
        header_length = size_t(prefix[8]) | (size_t(prefix[9]) << 8) |
                        (size_t(prefix[10]) << 16) | (size_t(prefix[11]) << 24);
    }
    std::string header(header_length, '\0');
    input.read(header.data(), std::streamsize(header.size()));
    require(header.find("'descr': '" + descr + "'") != std::string::npos,
            "unexpected NPY dtype in " + path);
    require(header.find("'fortran_order': False") != std::string::npos,
            "Fortran-order NPY is unsupported");
    const size_t open = header.find('(', header.find("'shape':"));
    const size_t close = header.find(')', open);
    NpyArray<T> result;
    size_t cursor = open + 1, elements = 1;
    while (cursor < close) {
        while (cursor < close && (header[cursor] == ' ' || header[cursor] == ',')) ++cursor;
        if (cursor >= close) break;
        size_t end = cursor;
        while (end < close && header[end] >= '0' && header[end] <= '9') ++end;
        result.shape.push_back(std::stoull(header.substr(cursor, end - cursor)));
        elements *= result.shape.back();
        cursor = end;
    }
    result.values.resize(elements);
    input.read(reinterpret_cast<char*>(result.values.data()),
               std::streamsize(elements * sizeof(T)));
    require(input.gcount() == std::streamsize(elements * sizeof(T)),
            "truncated NPY payload: " + path);
    return result;
}

void compare(const std::string& label, const std::vector<float>& actual,
             const std::vector<float>& reference, double max_nrmse,
             double min_correlation) {
    require(actual.size() == reference.size(), label + " size mismatch");
    double error = 0, norm = 0, sa = 0, sr = 0;
    float max_abs = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), label + " is not finite");
        const double d = double(actual[i]) - reference[i];
        error += d * d;
        norm += double(reference[i]) * reference[i];
        sa += actual[i];
        sr += reference[i];
        max_abs = std::max(max_abs, std::fabs(actual[i] - reference[i]));
    }
    const double ma = sa / actual.size(), mr = sr / actual.size();
    double cov = 0, va = 0, vr = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i] - ma, r = reference[i] - mr;
        cov += a * r;
        va += a * a;
        vr += r * r;
    }
    const double nrmse = std::sqrt(error / norm);
    const double correlation = cov / std::sqrt(va * vr);
    std::cout << std::setprecision(8) << "PASS: " << label << " NRMSE=" << nrmse
              << " correlation=" << correlation << " max_abs=" << max_abs << std::endl;
    require(nrmse <= max_nrmse, label + " exceeds NRMSE barrier");
    require(correlation >= min_correlation, label + " falls below correlation barrier");
}

std::vector<float> device_rows(const half* device, size_t count) {
    std::vector<half> host(count);
    require(cudaMemcpy(host.data(), device, count * sizeof(half),
                       cudaMemcpyDeviceToHost) == cudaSuccess, "copy embeddings");
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) values[i] = __half2float(host[i]);
    return values;
}

// The unified 3x3 merge is a raster split into 48 px blocks: check the
// geometry and ordering the adapter relies on, without any weights.
void test_preprocess_contract() {
    helios::Gemma4UnifiedVisionSpec spec;
    spec.model_patch_size = 48;
    spec.patch_values = 48 * 48 * 3;
    spec.max_soft_tokens = 280;
    const auto config = helios::gemma4_unified_preprocess_config(spec);
    std::vector<uint8_t> rgb(960 * 672 * 3);
    for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = uint8_t(i * 31 + (i >> 7));
    helios::Gemma4VisionPreprocessResult result;
    std::string error;
    require(helios::gemma4_vision_preprocess_rgb({rgb.data(), 960, 672, 0}, config,
                                                 result, &error), error);
    require(result.real_patches == 280 && result.patch_values == 6912 &&
            result.patch_columns == 20 && result.patch_rows == 14,
            "960x672 must give 20x14 blocks of 48 px");
    // Block (x=3, y=2), row 5, column 7, green: HWC raster inside the block.
    const size_t block = 2 * 20 + 3;
    const size_t pixel = (size_t(2 * 48 + 5) * 960 + (3 * 48 + 7)) * 3 + 1;
    require(result.position_ids[block * 2] == 3 && result.position_ids[block * 2 + 1] == 2,
            "block positions are (x, y) in raster order");
    require(std::fabs(result.pixel_values[block * 6912 + (5 * 48 + 7) * 3 + 1] -
                      rgb[pixel] / 255.0f) < 1e-7f,
            "pixels are HWC inside each block and rescaled to [0, 1]");
    std::cout << "PASS: unified preprocessing contract" << std::endl;
}

void test_real(const std::string& hnf_path, const std::string& oracle) {
    helios::Engine engine;
    helios::HnfLoader loader;
    require(loader.open(hnf_path), "cannot open " + hnf_path + ": " + loader.last_error());
    require(loader.load_block(helios::BLOCK_VISION, engine),
            "cannot load vision block: " + loader.last_error());
    require(loader.load_block(helios::BLOCK_AUDIO, engine),
            "cannot load audio block: " + loader.last_error());
    helios::Gemma4UnifiedEmbedder embedder(engine, loader);
    require(embedder.has_vision() && embedder.has_audio(), "embedder specs did not validate");
    const uint32_t hidden = embedder.vision().hidden;
    half* output = nullptr;
    require(cudaMalloc(&output, size_t(embedder.vision().max_soft_tokens) * hidden *
                                    sizeof(half)) == cudaSuccess, "allocate output");
    std::string error;

    for (const char* name : {"aligned_960x672", "resized_1234x567"}) {
        const std::string base = oracle + "/" + name;
        const auto rgb = read_npy<uint8_t>(base + ".rgb8.npy", "|u1");
        const auto pixels = read_npy<float>(base + ".pixel_values.npy", "<f4");
        const auto positions = read_npy<int32_t>(base + ".positions.npy", "<i4");
        const auto patch = read_npy<float>(base + ".after_patch_ln2.npy", "<f4");
        const auto pos = read_npy<float>(base + ".after_pos_norm.npy", "<f4");
        const auto embeddings = read_npy<float>(base + ".embeddings.npy", "<f4");
        const uint32_t rows = uint32_t(positions.shape[0]);

        // 1. Our preprocessing against the official processor.
        helios::Gemma4VisionPreprocessResult ours;
        require(helios::gemma4_vision_preprocess_rgb(
                    {rgb.values.data(), uint32_t(rgb.shape[1]), uint32_t(rgb.shape[0]), 0},
                    helios::gemma4_unified_preprocess_config(embedder.vision()), ours, &error),
                error);
        require(ours.real_patches == rows, std::string(name) + " soft-token count");
        require(std::equal(positions.values.begin(), positions.values.end(),
                           ours.position_ids.begin()), std::string(name) + " positions");
        float pixel_error = 0;
        for (size_t i = 0; i < pixels.values.size(); ++i) {
            pixel_error = std::max(pixel_error, std::fabs(pixels.values[i] - ours.pixel_values[i]));
        }
        std::cout << "PASS: " << name << " preprocessing max_abs=" << pixel_error << std::endl;
        // One 8-bit step: the resize rounds to uint8 before rescaling.
        require(pixel_error <= 1.0f / 255.0f + 1e-6f, std::string(name) + " pixels");

        // 2. The embedder on the oracle's own patches, stage by stage.
        helios::Gemma4VisionPreprocessResult reference;
        reference.real_patches = rows;
        reference.max_patches = rows;
        reference.patch_values = uint32_t(pixels.shape[1]);
        reference.pixel_values = pixels.values;
        reference.position_ids = positions.values;
        require(embedder.embed_image(reference, output, &error), error);
        std::vector<float> stage;
        require(embedder.copy_stage(helios::Gemma4UnifiedEmbedder::Stage::PatchLn2, stage, &error), error);
        compare(std::string(name) + " after patch LN2", stage, patch.values, 5e-3, 0.99998);
        require(embedder.copy_stage(helios::Gemma4UnifiedEmbedder::Stage::PosNorm, stage, &error), error);
        compare(std::string(name) + " after position LN", stage, pos.values, 5e-3, 0.99998);
        compare(std::string(name) + " embeddings", device_rows(output, size_t(rows) * hidden),
                embeddings.values, 5e-3, 0.99998);

        // 3. End to end from RGB8 with our own preprocessing.
        require(embedder.embed_image(ours, output, &error), error);
        compare(std::string(name) + " embeddings from RGB8",
                device_rows(output, size_t(rows) * hidden), embeddings.values, 1e-2, 0.9999);
    }

    const auto frames = read_npy<float>(oracle + "/audio.frames.npy", "<f4");
    const auto audio = read_npy<float>(oracle + "/audio.embeddings.npy", "<f4");
    const uint32_t tokens = uint32_t(frames.shape[0]);
    require(frames.shape[1] == embedder.audio().samples_per_token, "audio frame width");
    require(embedder.embed_audio(frames.values.data(), tokens, output, &error), error);
    compare("audio embeddings", device_rows(output, size_t(tokens) * hidden),
            audio.values, 5e-3, 0.99998);
    cudaFree(output);
}

} // namespace

int main(int argc, char** argv) {
    test_preprocess_contract();
    if (argc == 3) {
        test_real(argv[1], argv[2]);
    } else if (argc != 1) {
        std::cerr << "usage: " << argv[0] << " [gemma4_12b_mm.hnf oracle_dir]" << std::endl;
        return 2;
    }
    return 0;
}
