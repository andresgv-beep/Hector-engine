#include "gemma4_kv_cache.hpp"
#include "graph_builder.hpp"
#include "hnf_loader.hpp"
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
        if (part.empty()) throw std::invalid_argument("empty token in token list");
        size_t consumed = 0;
        const long parsed = std::stol(part, &consumed);
        if (consumed != part.size() || parsed < 0 ||
            parsed > std::numeric_limits<int32_t>::max()) {
            throw std::invalid_argument("invalid token: " + part);
        }
        tokens.push_back(static_cast<int32_t>(parsed));
    }
    if (tokens.size() < 2) {
        throw std::invalid_argument("at least two tokens are required");
    }
    return tokens;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_require(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

struct PositionMetrics {
    uint32_t argmax = 0;
    double nll = 0.0;
    double entropy = 0.0;
    double top1_probability = 0.0;
    double margin = 0.0;
};

PositionMetrics measure(const std::vector<half>& logits, uint32_t target) {
    require(target < logits.size(), "target token is outside vocabulary");

    uint32_t best_index = 0;
    double best = -std::numeric_limits<double>::infinity();
    double second = -std::numeric_limits<double>::infinity();
    for (uint32_t i = 0; i < logits.size(); ++i) {
        const double value = static_cast<double>(__half2float(logits[i]));
        require(std::isfinite(value), "non-finite logit");
        if (value > best) {
            second = best;
            best = value;
            best_index = i;
        } else if (value > second) {
            second = value;
        }
    }

    double sum_exp = 0.0;
    double weighted_logit = 0.0;
    for (const half raw : logits) {
        const double value = static_cast<double>(__half2float(raw));
        const double weight = std::exp(value - best);
        sum_exp += weight;
        weighted_logit += weight * value;
    }
    require(sum_exp > 0.0 && std::isfinite(sum_exp), "invalid softmax sum");

    const double log_z = best + std::log(sum_exp);
    const double target_logit = static_cast<double>(__half2float(logits[target]));
    return PositionMetrics{
        best_index,
        log_z - target_logit,
        log_z - weighted_logit / sum_exp,
        1.0 / sum_exp,
        best - second,
    };
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: gemma4_calibrate <model.hnf> <id,id,...> [positions.tsv]\n";
        return 1;
    }

    try {
        const auto tokens = parse_tokens(argv[2]);

        helios::EngineConfig engine_config;
        engine_config.scratch_pool.auto_fraction = 0.0f;
        engine_config.scratch_pool.min_size_bytes = 0;
        helios::Engine engine(engine_config);
        helios::kernels::register_all_kernels(engine);

        helios::HnfLoader loader;
        require(loader.open(argv[1]), "cannot open HNF: " + loader.last_error());
        require(loader.has_gemma4_config(), "missing GM4X");
        require(loader.load_block(helios::BLOCK_TEXT_MODEL, engine),
                "cannot load text block: " + loader.last_error());

        const auto config = loader.config();
        const auto& gemma = loader.gemma4_config();
        for (const int32_t token : tokens) {
            require(static_cast<uint32_t>(token) < config.vocab_size(),
                    "input token is outside vocabulary");
        }

        helios::GraphBuilder builder;
        const auto arch = builder.detect_architecture(engine, "text", config);
        builder.allocate_gemma4_scratch(engine, config, gemma, arch, 1, 1);

        helios::Gemma4KVCache cache;
        const uint32_t capacity = static_cast<uint32_t>(tokens.size());
        require(cache.allocate(gemma, config.num_key_value_heads(), 1, capacity),
                "cannot allocate KV");
        cache.register_tensors(engine, "_g4cal");

        void* token_ptr = engine.tensors().allocate_and_register(
            "g4.calibrate.token", {1, 1}, helios::dtype::INT32());
        const uint32_t vocab_size = config.vocab_size();
        std::vector<half> logits(vocab_size);

        std::ofstream rows;
        if (argc == 4) {
            rows.open(argv[3]);
            require(static_cast<bool>(rows), "cannot create positions TSV");
            rows << "position\tinput_token\ttarget_token\targmax\tnll\tentropy"
                    "\ttop1_probability\tmargin\n";
            rows << std::setprecision(10);
        }

        double total_nll = 0.0;
        double total_entropy = 0.0;
        double total_top1_probability = 0.0;
        uint64_t next_token_hits = 0;
        const uint32_t predictions = capacity - 1;

        for (uint32_t position = 0; position < predictions; ++position) {
            cuda_require(cudaMemcpy(token_ptr, &tokens[position], sizeof(int32_t),
                                    cudaMemcpyHostToDevice),
                         "copy input token");
            const helios::KVCacheParams params{"_g4cal", position, capacity};
            engine.execute(builder.build_gemma4_forward_cached(
                engine, config, gemma, arch, "g4.calibrate.token", 1, 1, params));
            engine.sync();

            cuda_require(cudaMemcpy(logits.data(), engine.tensors().at("_s.logits").ptr,
                                    logits.size() * sizeof(half), cudaMemcpyDeviceToHost),
                         "copy logits");
            const uint32_t target = static_cast<uint32_t>(tokens[position + 1]);
            const auto metrics = measure(logits, target);
            total_nll += metrics.nll;
            total_entropy += metrics.entropy;
            total_top1_probability += metrics.top1_probability;
            next_token_hits += metrics.argmax == target;

            if (rows) {
                rows << position << '\t' << tokens[position] << '\t' << target << '\t'
                     << metrics.argmax << '\t' << metrics.nll << '\t' << metrics.entropy
                     << '\t' << metrics.top1_probability << '\t' << metrics.margin << '\n';
            }
            if (position % 200 == 0) {
                std::cerr << '\r' << position << '/' << predictions << std::flush;
            }
        }
        std::cerr << '\r' << predictions << '/' << predictions << '\n';
        require(!rows || static_cast<bool>(rows), "cannot write positions TSV");

        const double count = static_cast<double>(predictions);
        const double mean_nll = total_nll / count;
        std::cout << std::fixed << std::setprecision(8)
                  << "tokens=" << tokens.size() << '\n'
                  << "predictions=" << predictions << '\n'
                  << "mean_nll=" << mean_nll << '\n'
                  << "perplexity=" << std::exp(mean_nll) << '\n'
                  << "mean_entropy=" << total_entropy / count << '\n'
                  << "mean_top1_probability=" << total_top1_probability / count << '\n'
                  << "next_token_accuracy="
                  << static_cast<double>(next_token_hits) / count << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
