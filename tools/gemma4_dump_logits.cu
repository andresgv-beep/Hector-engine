#include "gemma4_kv_cache.hpp"
#include "graph_builder.hpp"
#include "hnf_loader.hpp"
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<int32_t> parse_tokens(const std::string& value) {
    std::vector<int32_t> tokens;
    std::stringstream stream(value);
    std::string part;
    while (std::getline(stream, part, ',')) {
        if (!part.empty()) tokens.push_back(std::stoi(part));
    }
    if (tokens.empty()) throw std::invalid_argument("empty token list");
    return tokens;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_require(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: gemma4_dump_logits <model.hnf> <id,id,...> <output.bin>\n";
        return 1;
    }
    try {
        const auto tokens = parse_tokens(argv[2]);
        helios::HnfLoader loader;
        require(loader.open(argv[1]), "cannot open HNF: " + loader.last_error());
        require(loader.has_gemma4_config(), "missing GM4X");

        helios::Engine engine;
        helios::kernels::register_all_kernels(engine);
        require(loader.load_block(helios::BLOCK_TEXT_MODEL, engine), "cannot load text block");

        const auto config = loader.config();
        const auto& gemma = loader.gemma4_config();
        helios::GraphBuilder builder;
        const auto arch = builder.detect_architecture(engine, "text", config);
        builder.allocate_gemma4_scratch(
            engine, config, gemma, arch, 1, static_cast<uint32_t>(tokens.size()));

        helios::Gemma4KVCache cache;
        require(cache.allocate(gemma, config.num_key_value_heads(), 1,
                               static_cast<uint32_t>(tokens.size())),
                "cannot allocate KV");
        cache.register_tensors(engine, "_g4kv");
        void* token_ptr = engine.tensors().allocate_and_register(
            "g4.dump.tokens", {1, static_cast<uint32_t>(tokens.size())},
            helios::dtype::INT32());
        cuda_require(cudaMemcpy(token_ptr, tokens.data(), tokens.size() * sizeof(int32_t),
                                cudaMemcpyHostToDevice), "copy tokens");

        const helios::KVCacheParams params{
            "_g4kv", 0, static_cast<uint32_t>(tokens.size())};
        engine.execute(builder.build_gemma4_forward_cached(
            engine, config, gemma, arch, "g4.dump.tokens", 1,
            static_cast<uint32_t>(tokens.size()), params));
        engine.sync();

        std::vector<half> gpu_logits(config.vocab_size());
        cuda_require(cudaMemcpy(gpu_logits.data(), engine.tensors().at("_s.logits").ptr,
                                gpu_logits.size() * sizeof(half), cudaMemcpyDeviceToHost),
                     "copy logits");
        std::vector<float> logits(gpu_logits.size());
        for (size_t i = 0; i < logits.size(); ++i) logits[i] = __half2float(gpu_logits[i]);

        std::ofstream output(argv[3], std::ios::binary);
        require(static_cast<bool>(output), "cannot create output");
        output.write(reinterpret_cast<const char*>(logits.data()),
                     static_cast<std::streamsize>(logits.size() * sizeof(float)));
        require(static_cast<bool>(output), "cannot write output");
        std::cout << "Wrote " << logits.size() << " float32 logits to " << argv[3] << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
