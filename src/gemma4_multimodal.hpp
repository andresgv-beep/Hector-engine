#pragma once

#include "hnf_loader.hpp"

#include <cstdint>
#include <vector>

namespace helios {

// Canonical single-image sequence consumed by Gemma 4. `canonical_ids` keeps
// BOI + IMAGE * N + EOI for positions/masking. The two decoder lookups use
// PAD at IMAGE positions, but remain separate because upstream computes PLE
// identity before visual substitution and PLE context afterwards.
struct Gemma4MultimodalTokenPlan {
    std::vector<int32_t> canonical_ids;
    std::vector<int32_t> embedding_ids;
    std::vector<int32_t> ple_identity_ids;
    std::vector<int32_t> image_positions;
    uint32_t soft_token_count = 0;

    bool has_image() const { return soft_token_count != 0; }
};

// Expands exactly one IMAGE placeholder into BOI + IMAGE * soft_token_count +
// EOI. With soft_token_count == 0, an image-free input is copied byte for byte.
// V5 deliberately supports one image per prompt; ambiguous/already-expanded
// sequences fail instead of silently producing a wrong PLE identity stream.
Gemma4MultimodalTokenPlan make_gemma4_multimodal_token_plan(
    const std::vector<int32_t>& input_ids,
    const Gemma4VisionConfig& vision,
    uint32_t vocab_size,
    uint32_t soft_token_count);

} // namespace helios
