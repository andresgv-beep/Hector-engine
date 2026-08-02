#include "gemma4_kv_cache.hpp"
#include "gemma4_multimodal.hpp"
#include "graph_builder.hpp"
#include "hnf_loader.hpp"
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_require(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

std::vector<int32_t> parse_tokens(const std::string& value) {
    std::vector<int32_t> tokens;
    std::stringstream stream(value);
    std::string part;
    while (std::getline(stream, part, ',')) {
        if (!part.empty()) tokens.push_back(std::stoi(part));
    }
    require(!tokens.empty(), "empty token list");
    return tokens;
}

struct NpyF32 {
    std::vector<size_t> shape;
    std::vector<float> values;
};

NpyF32 read_npy_f32(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "cannot open NPY: " + path);
    uint8_t prefix[12]{};
    input.read(reinterpret_cast<char*>(prefix), 10);
    require(input.gcount() == 10 && std::memcmp(prefix, "\x93NUMPY", 6) == 0,
            "invalid NPY header");
    size_t header_size = 0;
    if (prefix[6] == 1) {
        header_size = size_t(prefix[8]) | (size_t(prefix[9]) << 8);
    } else if (prefix[6] == 2 || prefix[6] == 3) {
        input.read(reinterpret_cast<char*>(prefix + 10), 2);
        require(input.gcount() == 2, "truncated NPY header length");
        header_size = size_t(prefix[8]) | (size_t(prefix[9]) << 8) |
                      (size_t(prefix[10]) << 16) | (size_t(prefix[11]) << 24);
    } else {
        throw std::runtime_error("unsupported NPY version");
    }
    std::string header(header_size, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    require(input.gcount() == static_cast<std::streamsize>(header.size()) &&
            header.find("'descr': '<f4'") != std::string::npos &&
            header.find("'fortran_order': False") != std::string::npos,
            "NPY must be little-endian contiguous float32");

    const size_t key = header.find("'shape':");
    const size_t open = header.find('(', key);
    const size_t close = header.find(')', open);
    require(key != std::string::npos && open != std::string::npos &&
            close != std::string::npos, "NPY shape is missing");
    NpyF32 result;
    for (size_t cursor = open + 1; cursor < close;) {
        while (cursor < close && (header[cursor] == ' ' || header[cursor] == ','))
            ++cursor;
        if (cursor >= close) break;
        size_t end = cursor;
        while (end < close && header[end] >= '0' && header[end] <= '9') ++end;
        require(end > cursor, "invalid NPY shape");
        result.shape.push_back(std::stoull(header.substr(cursor, end - cursor)));
        cursor = end;
    }
    size_t elements = 1;
    for (size_t dimension : result.shape) {
        require(dimension != 0 &&
                elements <= std::numeric_limits<size_t>::max() / dimension,
                "NPY shape overflow");
        elements *= dimension;
    }
    result.values.resize(elements);
    input.read(reinterpret_cast<char*>(result.values.data()),
               static_cast<std::streamsize>(elements * sizeof(float)));
    require(input.gcount() == static_cast<std::streamsize>(elements * sizeof(float)),
            "truncated NPY payload");
    return result;
}

void upload(helios::Engine& engine, const std::string& name,
            const std::vector<int32_t>& values,
            const std::vector<uint32_t>& shape) {
    void* device = engine.tensors().allocate_and_register(
        name, shape, helios::dtype::INT32());
    cuda_require(cudaMemcpy(device, values.data(), values.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice), "upload INT32 tensor");
}

std::vector<float> download_logits(helios::Engine& engine, uint32_t vocab) {
    std::vector<half> fp16(vocab);
    cuda_require(cudaMemcpy(fp16.data(), engine.tensors().at("_s.logits").ptr,
                            fp16.size() * sizeof(half), cudaMemcpyDeviceToHost),
                 "download logits");
    std::vector<float> result(vocab);
    for (size_t i = 0; i < result.size(); ++i) result[i] = __half2float(fp16[i]);
    return result;
}

void write_logits(const std::string& prefix, uint32_t step,
                  const std::vector<float>& logits) {
    const std::string path = prefix + ".step" + std::to_string(step) + ".bin";
    std::ofstream output(path, std::ios::binary);
    require(static_cast<bool>(output), "cannot create " + path);
    output.write(reinterpret_cast<const char*>(logits.data()),
                 static_cast<std::streamsize>(logits.size() * sizeof(float)));
    require(static_cast<bool>(output), "cannot write " + path);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: gemma4_multimodal_dump <combined.hnf> <projected.npy> "
                     "<id,id,... with IMAGE> <greedy_steps> <output_prefix>\n";
        return 2;
    }
    try {
        const auto raw_tokens = parse_tokens(argv[3]);
        const uint32_t greedy_steps = static_cast<uint32_t>(std::stoul(argv[4]));
        const auto projected = read_npy_f32(argv[2]);
        require(projected.shape.size() == 2 &&
                projected.shape[0] <= std::numeric_limits<uint32_t>::max() &&
                projected.shape[1] <= std::numeric_limits<uint32_t>::max(),
                "projected visual embeddings must be [tokens, hidden]");

        helios::HnfLoader loader;
        require(loader.open(argv[1]), "cannot open HNF: " + loader.last_error());
        require(loader.has_gemma4_config(), "HNF must contain GM4X");
        const auto config = loader.config();
        require(projected.shape[1] == config.hidden_size(),
                "projected visual width differs from text hidden size");
        helios::Gemma4VisionConfig vision;
        if (loader.has_gemma4_vision_config()) {
            vision = loader.gemma4_vision_config();
        } else {
            // Text-only FP16 HNF is useful as the V5 control that isolates HQS.
            // These are the fixed special IDs of the certified E2B checkpoint;
            // production multimodal HNFs must still carry GM4V themselves.
            vision.max_soft_tokens = 280;
            vision.image_token_id = 258880;
            vision.boi_token_id = 255999;
            vision.eoi_token_id = 258882;
            vision.pad_token_id = 0;
            std::cerr << "[control] text-only HNF: using certified E2B token IDs\n";
        }
        const auto plan = helios::make_gemma4_multimodal_token_plan(
            raw_tokens, vision, config.vocab_size(),
            static_cast<uint32_t>(projected.shape[0]));
        require(plan.canonical_ids.size() <= std::numeric_limits<uint32_t>::max(),
                "multimodal sequence exceeds uint32");
        const uint32_t sequence = static_cast<uint32_t>(plan.canonical_ids.size());
        require(sequence <= std::numeric_limits<uint32_t>::max() - greedy_steps,
                "KV length overflow");

        helios::EngineConfig engine_config;
        engine_config.scratch_pool.auto_fraction = 0.0f;
        helios::Engine engine(engine_config);
        helios::kernels::register_all_kernels(engine);
        require(loader.load_block(helios::BLOCK_TEXT_MODEL, engine),
                "cannot load text block: " + loader.last_error());

        std::vector<half> projected_fp16(projected.values.size());
        for (size_t i = 0; i < projected.values.size(); ++i) {
            projected_fp16[i] = __float2half(projected.values[i]);
        }
        void* image_device = engine.tensors().allocate_and_register(
            "g4.mm.image", {static_cast<uint32_t>(projected.shape[0]),
                            static_cast<uint32_t>(projected.shape[1])},
            helios::dtype::FP16());
        cuda_require(cudaMemcpy(image_device, projected_fp16.data(),
                                projected_fp16.size() * sizeof(half),
                                cudaMemcpyHostToDevice), "upload projected vision");
        upload(engine, "g4.mm.embedding_tokens", plan.embedding_ids,
               {1, sequence});
        upload(engine, "g4.mm.ple_tokens", plan.ple_identity_ids,
               {1, sequence});
        upload(engine, "g4.mm.positions", plan.image_positions,
               {static_cast<uint32_t>(plan.image_positions.size())});
        upload(engine, "g4.mm.decode_token", {0}, {1, 1});

        const auto& gemma = loader.gemma4_config();
        helios::GraphBuilder builder;
        const auto arch = builder.detect_architecture(engine, "text", config);
        builder.allocate_gemma4_scratch(engine, config, gemma, arch, 1, sequence);
        helios::Gemma4KVCache cache;
        const uint32_t max_cache = sequence + greedy_steps;
        require(cache.allocate(gemma, config.num_key_value_heads(), 1, max_cache),
                "cannot allocate multimodal KV cache");
        cache.register_tensors(engine, "_g4mmkv");

        const helios::Gemma4MultimodalInputNames names{
            "g4.mm.embedding_tokens", "g4.mm.ple_tokens",
            "g4.mm.image", "g4.mm.positions"};
        engine.execute(builder.build_gemma4_multimodal_forward_cached(
            engine, config, gemma, arch, names, 1, sequence,
            {"_g4mmkv", 0, max_cache}));
        engine.sync();

        std::vector<int32_t> generated;
        for (uint32_t step = 0; step <= greedy_steps; ++step) {
            const auto logits = download_logits(engine, config.vocab_size());
            write_logits(argv[5], step, logits);
            if (step == greedy_steps) break;
            const int32_t next = static_cast<int32_t>(
                std::max_element(logits.begin(), logits.end()) - logits.begin());
            generated.push_back(next);
            cuda_require(cudaMemcpy(engine.tensors().at("g4.mm.decode_token").ptr,
                                    &next, sizeof(next), cudaMemcpyHostToDevice),
                         "upload greedy token");
            engine.execute(builder.build_gemma4_forward_cached(
                engine, config, gemma, arch, "g4.mm.decode_token", 1, 1,
                {"_g4mmkv", sequence + step, max_cache}));
            engine.sync();
        }

        std::ofstream tokens_output(std::string(argv[5]) + ".tokens.txt");
        for (size_t i = 0; i < generated.size(); ++i) {
            if (i) tokens_output << ',';
            tokens_output << generated[i];
        }
        tokens_output << '\n';
        std::cout << "sequence=" << sequence << " visual="
                  << plan.soft_token_count << " greedy=";
        for (int32_t token : generated) std::cout << ' ' << token;
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
