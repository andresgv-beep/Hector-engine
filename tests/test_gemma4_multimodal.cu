#include "gemma4_multimodal.hpp"
#include "engine.hpp"
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void cuda_require(cudaError_t error, const char* where) {
    require(error == cudaSuccess,
            std::string(where) + ": " + cudaGetErrorString(error));
}

helios::Gemma4VisionConfig config() {
    helios::Gemma4VisionConfig result;
    result.max_soft_tokens = 280;
    result.image_token_id = 18;
    result.boi_token_id = 19;
    result.eoi_token_id = 20;
    result.pad_token_id = 0;
    return result;
}

template <typename Fn>
void require_rejected(Fn&& fn, const char* message) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}

void test_token_plan() {
    const auto vision = config();
    const std::vector<int32_t> text{2, 5, 7, 9};
    const auto plain = helios::make_gemma4_multimodal_token_plan(
        text, vision, 32, 0);
    require(!plain.has_image() && plain.canonical_ids == text &&
            plain.embedding_ids == text && plain.ple_identity_ids == text &&
            plain.image_positions.empty(),
            "text-only token plan must be byte-identical");

    const auto image = helios::make_gemma4_multimodal_token_plan(
        {2, 5, 18, 9}, vision, 32, 3);
    require(image.canonical_ids ==
                std::vector<int32_t>({2, 5, 19, 18, 18, 18, 20, 9}),
            "canonical BOI + IMAGE*N + EOI expansion");
    require(image.embedding_ids ==
                std::vector<int32_t>({2, 5, 19, 0, 0, 0, 20, 9}) &&
            image.ple_identity_ids == image.embedding_ids,
            "main and PLE identity lookups must use PAD at visual slots");
    require(image.image_positions == std::vector<int32_t>({3, 4, 5}) &&
            image.soft_token_count == 3,
            "visual row positions must match the expanded sequence");

    require_rejected([&] {
        (void)helios::make_gemma4_multimodal_token_plan(
            {2, 18, 18}, vision, 32, 3);
    }, "multiple image placeholders must be rejected");
    require_rejected([&] {
        (void)helios::make_gemma4_multimodal_token_plan(
            {2, 18}, vision, 32, 281);
    }, "too many visual tokens must be rejected");
    require_rejected([&] {
        (void)helios::make_gemma4_multimodal_token_plan(
            {2, 18}, vision, 32, 0);
    }, "placeholder without visual output must be rejected");
    require_rejected([&] {
        (void)helios::make_gemma4_multimodal_token_plan(
            {2, 7}, vision, 32, 3);
    }, "visual output without placeholder must be rejected");
    std::cout << "PASS: canonical Gemma 4 V5 token plan" << std::endl;
}

void test_scatter_rows() {
    helios::EngineConfig engine_config;
    engine_config.scratch_pool.auto_fraction = 0.0f;
    helios::Engine engine(engine_config);
    helios::kernels::register_all_kernels(engine);

    const std::vector<float> base_values{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
    };
    const std::vector<float> row_values{101, 102, 103, 201, 202, 203};
    std::vector<half> base(base_values.size());
    std::vector<half> rows(row_values.size());
    for (size_t i = 0; i < base.size(); ++i) base[i] = __float2half(base_values[i]);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = __float2half(row_values[i]);
    const std::vector<int32_t> positions{1, 3};

    void* output = engine.tensors().allocate_and_register(
        "scatter.output", {1, 4, 3}, helios::dtype::FP16());
    void* source = engine.tensors().allocate_and_register(
        "scatter.rows", {2, 3}, helios::dtype::FP16());
    void* indices = engine.tensors().allocate_and_register(
        "scatter.positions", {2}, helios::dtype::INT32());
    cuda_require(cudaMemcpy(output, base.data(), base.size() * sizeof(half),
                            cudaMemcpyHostToDevice), "upload scatter base");
    cuda_require(cudaMemcpy(source, rows.data(), rows.size() * sizeof(half),
                            cudaMemcpyHostToDevice), "upload scatter rows");
    cuda_require(cudaMemcpy(indices, positions.data(),
                            positions.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice), "upload scatter positions");

    helios::CommandBuffer commands;
    commands.add_scatter_rows("scatter.output", "scatter.rows", "scatter.positions");
    engine.execute(commands);
    engine.sync();

    std::vector<half> actual(base.size());
    cuda_require(cudaMemcpy(actual.data(), output, actual.size() * sizeof(half),
                            cudaMemcpyDeviceToHost), "download scatter result");
    const std::vector<float> expected{
        1, 2, 3,
        101, 102, 103,
        7, 8, 9,
        201, 202, 203,
    };
    for (size_t i = 0; i < actual.size(); ++i) {
        require(__half2float(actual[i]) == expected[i],
                "scatter row numerical mismatch");
    }
    std::cout << "PASS: FP16 visual row substitution" << std::endl;
}

} // namespace

int main() {
    test_token_plan();
    test_scatter_rows();
    std::cout << "Gemma 4 multimodal V5 primitives: passed" << std::endl;
    return 0;
}
