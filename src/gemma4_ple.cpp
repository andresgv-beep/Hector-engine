#include "gemma4_ple.hpp"

#include <cmath>
#include <stdexcept>

namespace helios {

void append_gemma4_ple_preparation(
    CommandBuffer& commands,
    const ModelConfig& model,
    const Gemma4Config& gemma,
    const Gemma4PlePreparationNames& names) {
    const uint32_t hidden = model.hidden_size();
    const uint32_t layers = model.num_hidden_layers();
    const uint32_t ple_hidden = gemma.ple_hidden_size;

    if (hidden == 0 || layers == 0 || ple_hidden == 0) {
        throw std::invalid_argument("Gemma 4 PLE dimensions must be non-zero");
    }
    if (gemma.layers.size() != layers) {
        throw std::invalid_argument("Gemma 4 PLE layer count differs from GM4X");
    }
    if (!gemma.has_flag(GEMMA4_EXT_FLAG_PLE)) {
        throw std::invalid_argument("Gemma 4 GM4X does not advertise PLE");
    }

    commands.add_embedding(names.ple, names.input_tokens,
                           names.token_embedding_weight);
    commands.add_scale(names.ple, names.ple,
                       std::sqrt(static_cast<float>(ple_hidden)));

    commands.add_matmul(names.context, names.main_embeddings,
                        names.model_projection_weight);
    commands.add_scale(names.context, names.context,
                       1.0f / std::sqrt(static_cast<float>(hidden)));
    commands.add_rmsnorm(names.context, names.context,
                         names.projection_norm_weight,
                         model.rms_norm_eps(), ple_hidden);

    commands.add_add(names.ple, names.context, names.ple);
    commands.add_scale(names.ple, names.ple, 1.0f / std::sqrt(2.0f));
}

void append_gemma4_ple_layer_injection(
    CommandBuffer& commands,
    const ModelConfig& model,
    const Gemma4Config& gemma,
    uint32_t layer_index,
    const Gemma4PleLayerNames& names) {
    if (model.hidden_size() == 0 || gemma.ple_hidden_size == 0) {
        throw std::invalid_argument("Gemma 4 PLE dimensions must be non-zero");
    }
    if (gemma.layers.size() != model.num_hidden_layers() ||
        layer_index >= gemma.layers.size()) {
        throw std::invalid_argument("Gemma 4 PLE layer index is outside GM4X");
    }
    if (!gemma.has_flag(GEMMA4_EXT_FLAG_PLE) ||
        !gemma.has_flag(GEMMA4_EXT_FLAG_LAYER_SCALAR)) {
        throw std::invalid_argument("Gemma 4 GM4X lacks PLE or layer_scalar");
    }

    const std::string prefix = "text.layer" + std::to_string(layer_index) + '.';
    commands.add_matmul(names.gate, names.hidden,
                        prefix + "ple.input_gate.weight");
    commands.add_gelu(names.gate, names.gate);
    commands.add_mul(names.gate, names.gate, names.ple_segment);
    commands.add_matmul(names.projected, names.gate,
                        prefix + "ple.projection.weight");
    commands.add_rmsnorm(names.projected, names.projected,
                         prefix + "ln_ple_post.weight",
                         model.rms_norm_eps());
    commands.add_add(names.hidden, names.hidden, names.projected);
    commands.add_mul_scalar_tensor(names.hidden, names.hidden,
                                   prefix + "layer_scalar");
}

} // namespace helios
