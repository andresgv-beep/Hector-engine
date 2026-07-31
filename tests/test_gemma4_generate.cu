#include "chat_template.hpp"
#include "gemma4_kv_cache.hpp"
#include "graph_builder.hpp"
#include "hnf_loader.hpp"
#include "htf_tokenizer.hpp"
#include "kernels.hpp"
#include "sampler.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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
    if (argc < 2 || argc > 5) {
        std::cerr << "Usage: test_gemma4_generate <gemma4.hnf> [prompt] [max_tokens] [temperature]\n";
        return 1;
    }

    try {
        const std::string path = argv[1];
        const std::string user_prompt = argc >= 3
            ? argv[2]
            : "Responde en una frase: ¿Cuál es la capital de Francia?";
        const uint32_t max_tokens = argc >= 4
            ? static_cast<uint32_t>(std::stoul(argv[3]))
            : 32;
        const float temperature = argc >= 5 ? std::stof(argv[4]) : 0.0f;
        require(max_tokens > 0, "max_tokens must be positive");
        require(temperature >= 0.0f, "temperature must be non-negative");

        helios::HnfLoader loader;
        require(loader.open(path), "cannot open HNF: " + loader.last_error());
        require(loader.has_gemma4_config(), "HNF has no Gemma 4 GM4X config");
        const auto* tokenizer = loader.tokenizer("text");
        require(tokenizer != nullptr, "HNF has no text tokenizer");

        std::vector<int32_t> prompt_ids;
        const bool raw_completion = std::getenv("HELIOS_GEMMA4_RAW") != nullptr;
        if (raw_completion) {
            prompt_ids = tokenizer->encode(user_prompt, true, false);
        } else {
            std::vector<helios::ChatMessage> messages;
            if (const char* system = std::getenv("HELIOS_GEMMA4_SYSTEM")) {
                if (*system != '\0') messages.push_back({"system", system});
            }
            messages.push_back({"user", user_prompt});
            const std::string formatted = helios::format_gemma4_chat(messages);
            prompt_ids = tokenizer->encode(formatted, false, false);
        }
        require(!prompt_ids.empty() && prompt_ids.front() == 2,
                "prompt must start with BOS");

        helios::Engine engine;
        helios::kernels::register_all_kernels(engine);
        require(loader.load_block(helios::BLOCK_TEXT_MODEL, engine),
                "cannot load Gemma 4 text weights");

        const auto config = loader.config();
        const auto& gemma = loader.gemma4_config();
        helios::GraphBuilder builder;
        const auto arch = builder.detect_architecture(engine, "text", config);
        builder.allocate_gemma4_scratch(
            engine, config, gemma, arch, 1, static_cast<uint32_t>(prompt_ids.size()));

        const uint32_t max_cache_len = static_cast<uint32_t>(prompt_ids.size()) + max_tokens;
        helios::Gemma4KVCache cache;
        require(cache.allocate(gemma, config.num_key_value_heads(), 1, max_cache_len),
                "cannot allocate heterogeneous KV cache");
        cache.register_tensors(engine, "_g4kv");

        void* token_ptr = engine.tensors().allocate_and_register(
            "g4.generate.tokens",
            {1, static_cast<uint32_t>(prompt_ids.size())},
            helios::dtype::INT32());
        cuda_require(cudaMemcpy(token_ptr, prompt_ids.data(),
                                prompt_ids.size() * sizeof(int32_t),
                                cudaMemcpyHostToDevice),
                     "copy prompt tokens");

        const auto prefill_start = std::chrono::steady_clock::now();
        const helios::KVCacheParams prefill_cache{"_g4kv", 0, max_cache_len};
        engine.execute(builder.build_gemma4_forward_cached(
            engine, config, gemma, arch, "g4.generate.tokens", 1,
            static_cast<uint32_t>(prompt_ids.size()), prefill_cache));
        engine.sync();
        const auto prefill_end = std::chrono::steady_clock::now();

        auto& token_info = engine.tensors().at("g4.generate.tokens");
        token_info.shape = {1, 1};
        helios::Sampler sampler;
        sampler.set_seed(7);
        helios::SamplingConfig sampling = helios::SamplingConfig::deterministic();
        if (temperature > 0.0f) {
            sampling.temperature = temperature;
            sampling.top_k = 64;
            sampling.top_p = 0.95f;
        }
        std::vector<int32_t> generated;
        generated.reserve(max_tokens);
        const int32_t eos = tokenizer->eos_token_id().value_or(1);
        const int32_t end_turn = tokenizer->token_to_id("<turn|>").value_or(106);

        const auto decode_start = std::chrono::steady_clock::now();
        for (uint32_t step = 0; step < max_tokens; ++step) {
            const auto& logits = engine.tensors().at("_s.logits");
            const int32_t next = sampler.sample(
                static_cast<const half*>(logits.ptr),
                static_cast<int>(config.vocab_size()), sampling);
            require(next >= 0 && static_cast<uint32_t>(next) < config.vocab_size(),
                    "sampler returned an invalid token");
            if (next == eos || next == end_turn) break;
            generated.push_back(next);

            cuda_require(cudaMemcpy(token_ptr, &next, sizeof(next), cudaMemcpyHostToDevice),
                         "copy decode token");
            const helios::KVCacheParams decode_cache{
                "_g4kv", static_cast<uint32_t>(prompt_ids.size()) + step, max_cache_len};
            engine.execute(builder.build_gemma4_forward_cached(
                engine, config, gemma, arch, "g4.generate.tokens", 1, 1, decode_cache));
            engine.sync();
        }
        const auto decode_end = std::chrono::steady_clock::now();

        require(!generated.empty(), "greedy generation produced no text tokens");
        const std::string output = tokenizer->decode(generated);
        require(!output.empty(), "generated tokens decode to empty text");

        const double prefill_ms = std::chrono::duration<double, std::milli>(
            prefill_end - prefill_start).count();
        const double decode_s = std::chrono::duration<double>(
            decode_end - decode_start).count();
        std::cout << "Prompt tokens: " << prompt_ids.size() << '\n';
        std::cout << "Generated tokens: " << generated.size() << '\n';
        std::cout << "Mode: " << (raw_completion ? "raw completion" : "chat") << '\n';
        std::cout << "Sampling: temperature=" << temperature
                  << " top_k=" << sampling.top_k
                  << " top_p=" << sampling.top_p << '\n';
        std::cout << "Prefill: " << prefill_ms << " ms\n";
        if (decode_s > 0.0) {
            std::cout << "Decode: " << generated.size() / decode_s << " tok/s\n";
        }
        std::cout << "Output: " << output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
