#pragma once

#include "command.hpp"
#include "hnf_loader.hpp"

#include <string>

namespace helios {

// Tensor names used by the isolated PLE preparation path. `main_embeddings`
// must contain the already-scaled main token embeddings, exactly as consumed
// by the Gemma 4 decoder. `ple` is used in-place for the token-identity lookup
// and the final combined [batch, sequence, layers * ple_hidden] result.
struct Gemma4PlePreparationNames {
    std::string input_tokens = "input_tokens";
    std::string main_embeddings = "_scratch.hidden";
    std::string ple = "_scratch.gemma4_ple";
    std::string context = "_scratch.gemma4_ple_context";

    std::string token_embedding_weight = "text.ple.token_embedding.weight";
    std::string model_projection_weight = "text.ple.model_projection.weight";
    std::string projection_norm_weight = "text.ple.projection_norm.weight";
};

// `ple_segment` is a zero-copy view of 256 consecutive values from the packed
// prepared PLE tensor. The graph allocator creates one such view per layer.
struct Gemma4PleLayerNames {
    std::string hidden = "_scratch.hidden";
    std::string ple_segment = "_scratch.gemma4_ple_layer";
    std::string gate = "_scratch.gemma4_ple_gate";
    std::string projected = "_scratch.gemma4_ple_projected";
};

// Appends the complete pre-layer PLE preparation:
//   token_lookup * sqrt(ple_hidden)
//   context = rmsnorm_segments(main_embeddings @ projection.T / sqrt(hidden))
//   ple = (token_identity + context) / sqrt(2)
// The context RMSNorm is explicitly segmented by ple_hidden rather than by the
// packed layers * ple_hidden width.
void append_gemma4_ple_preparation(
    CommandBuffer& commands,
    const ModelConfig& model,
    const Gemma4Config& gemma,
    const Gemma4PlePreparationNames& names = {});

// Appends the PLE residual branch at the end of one decoder layer, including
// the layer_scalar required after the residual addition.
void append_gemma4_ple_layer_injection(
    CommandBuffer& commands,
    const ModelConfig& model,
    const Gemma4Config& gemma,
    uint32_t layer_index,
    const Gemma4PleLayerNames& names = {});

} // namespace helios
