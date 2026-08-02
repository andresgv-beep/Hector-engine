#include "gemma4_multimodal.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace helios {
namespace {

void validate_token(int32_t token, uint32_t vocab_size, const char* what) {
    if (token < 0 || static_cast<uint32_t>(token) >= vocab_size) {
        throw std::invalid_argument(std::string("Gemma 4 ") + what +
                                    " is outside the text vocabulary");
    }
}

} // namespace

Gemma4MultimodalTokenPlan make_gemma4_multimodal_token_plan(
    const std::vector<int32_t>& input_ids,
    const Gemma4VisionConfig& vision,
    uint32_t vocab_size,
    uint32_t soft_token_count) {
    if (input_ids.empty() || vocab_size == 0) {
        throw std::invalid_argument("Gemma 4 multimodal input must not be empty");
    }
    validate_token(vision.image_token_id, vocab_size, "IMAGE token");
    validate_token(vision.boi_token_id, vocab_size, "BOI token");
    validate_token(vision.eoi_token_id, vocab_size, "EOI token");
    validate_token(vision.pad_token_id, vocab_size, "PAD token");
    if (vision.image_token_id == vision.boi_token_id ||
        vision.image_token_id == vision.eoi_token_id ||
        vision.boi_token_id == vision.eoi_token_id) {
        throw std::invalid_argument("Gemma 4 multimodal special tokens must be distinct");
    }
    for (int32_t token : input_ids) validate_token(token, vocab_size, "input token");

    const size_t placeholders = static_cast<size_t>(std::count(
        input_ids.begin(), input_ids.end(), vision.image_token_id));
    if (soft_token_count == 0) {
        if (placeholders != 0) {
            throw std::invalid_argument(
                "Gemma 4 IMAGE placeholder requires visual soft tokens");
        }
        Gemma4MultimodalTokenPlan plan;
        plan.canonical_ids = input_ids;
        plan.embedding_ids = input_ids;
        plan.ple_identity_ids = input_ids;
        return plan;
    }
    if (vision.max_soft_tokens == 0 || soft_token_count > vision.max_soft_tokens) {
        throw std::invalid_argument("Gemma 4 visual soft-token count exceeds GM4V");
    }
    if (placeholders != 1) {
        throw std::invalid_argument(
            "Gemma 4 V5 requires exactly one IMAGE placeholder");
    }
    if (input_ids.size() > std::numeric_limits<size_t>::max() - soft_token_count - 1) {
        throw std::overflow_error("Gemma 4 multimodal sequence length overflow");
    }
    if (input_ids.size() + soft_token_count + 1 >
        static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::overflow_error("Gemma 4 multimodal positions exceed INT32");
    }

    Gemma4MultimodalTokenPlan plan;
    const size_t output_size = input_ids.size() + soft_token_count + 1;
    plan.canonical_ids.reserve(output_size);
    plan.embedding_ids.reserve(output_size);
    plan.ple_identity_ids.reserve(output_size);
    plan.image_positions.reserve(soft_token_count);
    plan.soft_token_count = soft_token_count;

    for (int32_t token : input_ids) {
        if (token != vision.image_token_id) {
            plan.canonical_ids.push_back(token);
            plan.embedding_ids.push_back(token);
            plan.ple_identity_ids.push_back(token);
            continue;
        }

        plan.canonical_ids.push_back(vision.boi_token_id);
        plan.embedding_ids.push_back(vision.boi_token_id);
        plan.ple_identity_ids.push_back(vision.boi_token_id);
        for (uint32_t i = 0; i < soft_token_count; ++i) {
            plan.image_positions.push_back(
                static_cast<int32_t>(plan.canonical_ids.size()));
            plan.canonical_ids.push_back(vision.image_token_id);
            plan.embedding_ids.push_back(vision.pad_token_id);
            plan.ple_identity_ids.push_back(vision.pad_token_id);
        }
        plan.canonical_ids.push_back(vision.eoi_token_id);
        plan.embedding_ids.push_back(vision.eoi_token_id);
        plan.ple_identity_ids.push_back(vision.eoi_token_id);
    }
    return plan;
}

} // namespace helios
