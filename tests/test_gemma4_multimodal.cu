#include "gemma4_multimodal.hpp"
#include "multimodal_adapter.hpp"
#include "chat_template.hpp"
#include "gemma4_kv_cache.hpp"
#include "model_capabilities.hpp"
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

void test_generic_attachment_contract() {
    std::vector<uint8_t> pixels(4 * 2 * 3, 127);
    helios::MultimodalTurnInput turn;
    turn.formatted_token_ids = {2, 18, 9};
    turn.attachments.push_back({
        helios::AttachmentKind::ImageRgb8,
        pixels.data(), pixels.size(), "image/rgb8",
        4, 2, 12, 0, 0});
    const helios::MultimodalAdapterLimits limits{
        helios::attachment_kind_bit(helios::AttachmentKind::ImageRgb8),
        1, 512, 1024};
    std::string error;
    require(helios::validate_multimodal_turn(turn, limits, &error),
            "valid generic RGB8 attachment: " + error);

    turn.attachments.front().byte_size--;
    require(!helios::validate_multimodal_turn(turn, limits, &error) &&
            error.find("shorter") != std::string::npos,
            "truncated RGB8 attachment must be rejected");
    turn.attachments.front().byte_size++;
    turn.attachments.front().kind = helios::AttachmentKind::AudioPcmF32;
    require(!helios::validate_multimodal_turn(turn, limits, &error) &&
            error.find("unsupported") != std::string::npos,
            "adapter mask must reject an unsupported modality");
    std::cout << "PASS: generic borrowed-attachment contract" << std::endl;
}

void test_real_persistent_adapter(const std::string& path) {
    setenv("HELIOS_EMBED_MMAP", "1", 1);
    setenv("HELIOS_VISION_MMAP", "1", 1);

    cudaStream_t stream = nullptr;
    cuda_require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "create adapter test stream");
    helios::EngineConfig engine_config;
    engine_config.scratch_pool.auto_fraction = 0.0f;
    engine_config.stream = stream;
    helios::Engine engine(engine_config);
    helios::kernels::register_all_kernels(engine);

    helios::HnfLoader loader;
    require(loader.open(path), "open combined HNF: " + loader.last_error());
    require(loader.load_block(helios::BLOCK_TEXT_MODEL, engine),
            "load text block: " + loader.last_error());
    const auto* tokenizer = loader.tokenizer("text");
    require(tokenizer, "combined HNF tokenizer");

    constexpr uint32_t kContext = 1024;
    helios::Gemma4KVCache cache;
    require(cache.allocate(loader.gemma4_config(),
                           loader.config().num_key_value_heads(), 1, kContext),
            "allocate persistent adapter KV");
    cache.register_tensors(engine, "_mm_adapter_test_kv");

    helios::GraphBuilder graph;
    const auto text_arch = graph.detect_architecture(
        engine, "text", loader.config());
    graph.allocate_gemma4_scratch(
        engine, loader.config(), loader.gemma4_config(), text_arch, 1, 512);

    const auto capabilities = helios::inspect_model_capabilities(loader);
    const auto* vision = capabilities.find(helios::ModelModality::Vision);
    require(vision && vision->status == helios::AdapterStatus::RuntimeReady,
            "real HNF must resolve a runtime vision adapter");
    std::string error;
    auto adapter = helios::create_multimodal_adapter(
        vision->adapter_id, engine, loader, graph, text_arch, 512, &error);
    require(adapter != nullptr, "create persistent adapter: " + error);

    std::vector<uint8_t> pixels(96 * 72 * 3);
    for (uint32_t y = 0; y < 72; ++y) {
        for (uint32_t x = 0; x < 96; ++x) {
            const size_t p = (size_t(y) * 96 + x) * 3;
            pixels[p + 0] = static_cast<uint8_t>(x * 255 / 95);
            pixels[p + 1] = static_cast<uint8_t>(y * 255 / 71);
            pixels[p + 2] = 63;
        }
    }
    const helios::TurnAttachment image{
        helios::AttachmentKind::ImageRgb8,
        pixels.data(), pixels.size(), "image/rgb8",
        96, 72, 96 * 3, 0, 0};

    auto first_ids = tokenizer->encode(helios::format_gemma4_chat(
        {{"user", "<|image|>\nDescribe la imagen."}}), false, false);
    helios::MultimodalTurnInput first{first_ids, {image}};
    helios::MultimodalPrefillResult first_result;
    require(adapter->prefill(
                first, {"_mm_adapter_test_kv", 0, kContext},
                first_result, &error),
            "first persistent image prefill: " + error);
    require(first_result.attachment_count == 1 &&
            first_result.injected_tokens > 0 &&
            first_result.sequence_tokens > first_ids.size(),
            "first image prefill result");
    cache.advance(first_result.sequence_tokens);

    const std::string second_fragment =
        "<turn|>\n<|turn>user\n<|image|>\n"
        "¿Qué colores dominan?<turn|>\n<|turn>model\n";
    auto second_ids = tokenizer->encode(second_fragment, false, false);
    helios::MultimodalTurnInput second{second_ids, {image}};
    helios::MultimodalPrefillResult second_result;
    require(adapter->prefill(
                second, {"_mm_adapter_test_kv", cache.position(), kContext},
                second_result, &error),
            "second persistent image prefill: " + error);
    require(second_result.injected_tokens == first_result.injected_tokens,
            "adapter must be reusable at a non-zero KV position");
    require(engine.tensors().exists("_s.logits"),
            "multimodal prefill must produce decoder logits");

    adapter.reset();
    cudaStreamDestroy(stream);
    std::cout << "PASS: real persistent adapter at two KV positions" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    test_token_plan();
    test_scatter_rows();
    test_generic_attachment_contract();
    if (argc > 1) test_real_persistent_adapter(argv[1]);
    std::cout << "Gemma 4 multimodal V5 primitives: passed" << std::endl;
    return 0;
}
