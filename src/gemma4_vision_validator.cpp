#include "gemma4_vision_validator.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace helios {
namespace {

using Shape = std::vector<uint32_t>;

void add_error(Gemma4VisionValidationReport& report,
               const std::string& message) {
    report.errors.push_back(message);
}

bool checked_add(uint64_t a, uint64_t b, uint64_t& result) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    result = a + b;
    return true;
}

bool checked_mul(uint64_t a, uint64_t b, uint64_t& result) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    result = a * b;
    return true;
}

std::string shape_string(const Shape& shape) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) out << ',';
        out << shape[i];
    }
    out << ']';
    return out.str();
}

float half_to_float(uint16_t half) {
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 0x1fu;
    uint32_t mantissa = half & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffu;
            const uint32_t fp32_exp = static_cast<uint32_t>(127 - 14 - shift);
            bits = sign | (fp32_exp << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

} // namespace

Gemma4VisionValidationReport validate_gemma4_vision_tensors(
    const HnfLoader& loader) {
    Gemma4VisionValidationReport report;

    if (!loader.has_config_for_block(BLOCK_VISION) ||
        loader.config_for_block(BLOCK_VISION).get<std::string>(
            "encoder_type", "") != "gemma4") {
        add_error(report, "vision encoder is not gemma4");
        return report;
    }
    if (!loader.has_gemma4_vision_config()) {
        add_error(report, "GM4V configuration is missing");
        return report;
    }

    const ModelConfig& base = loader.config_for_block(BLOCK_VISION);
    const Gemma4VisionConfig& config = loader.gemma4_vision_config();
    const uint32_t layers = base.num_hidden_layers();
    const uint32_t hidden = base.hidden_size();
    const uint32_t heads = base.num_attention_heads();
    const uint32_t intermediate = base.intermediate_size();
    const uint32_t channels = base.get<uint32_t>("num_channels", 0);
    const uint32_t patch = config.patch_size;

    uint64_t patch_width64 = 0;
    uint64_t max_patches64 = 0;
    if (layers == 0 || hidden == 0 || heads == 0 || intermediate == 0 ||
        channels == 0 || patch == 0 || config.head_dim == 0 ||
        config.position_embedding_size == 0 || config.projection_dim == 0 ||
        !checked_mul(patch, patch, patch_width64) ||
        !checked_mul(patch_width64, channels, patch_width64) ||
        !checked_mul(config.max_soft_tokens, config.pooling_kernel_size,
                     max_patches64) ||
        !checked_mul(max_patches64, config.pooling_kernel_size,
                     max_patches64) ||
        patch_width64 > std::numeric_limits<uint32_t>::max() ||
        max_patches64 > std::numeric_limits<uint32_t>::max()) {
        add_error(report, "Gemma 4 vision geometry is invalid or overflows");
        return report;
    }
    if (static_cast<uint64_t>(heads) * config.head_dim != hidden ||
        config.num_key_value_heads != heads) {
        add_error(report, "Gemma 4 vision attention geometry is inconsistent");
        return report;
    }
    report.max_patches = static_cast<uint32_t>(max_patches64);
    report.max_soft_tokens = config.max_soft_tokens;

    const BlockEntry& block = loader.block(BLOCK_VISION);
    uint64_t block_end = 0;
    if (block.size == 0 || !checked_add(block.offset, block.size, block_end)) {
        add_error(report, "vision block is absent or its range overflows");
        return report;
    }

    std::unordered_map<std::string, const TensorEntry*> by_name;
    std::vector<const TensorEntry*> vision_tensors;
    for (const TensorEntry& tensor : loader.tensors()) {
        if (tensor.block != "vision") continue;
        ++report.tensor_count;
        vision_tensors.push_back(&tensor);
        if (!by_name.emplace(tensor.name, &tensor).second) {
            add_error(report, "duplicate vision tensor: " + tensor.name);
        }
        if (!checked_add(report.weight_bytes, tensor.size,
                         report.weight_bytes)) {
            add_error(report, "vision weight byte total overflows");
        }
        uint64_t tensor_end = 0;
        if (tensor.offset < block.offset ||
            !checked_add(tensor.offset, tensor.size, tensor_end) ||
            tensor_end > block_end) {
            add_error(report, "tensor lies outside vision block: " + tensor.name);
        }
    }

    std::sort(vision_tensors.begin(), vision_tensors.end(),
              [](const TensorEntry* a, const TensorEntry* b) {
                  return a->offset < b->offset;
              });
    for (size_t i = 1; i < vision_tensors.size(); ++i) {
        uint64_t previous_end = 0;
        if (checked_add(vision_tensors[i - 1]->offset,
                        vision_tensors[i - 1]->size, previous_end) &&
            previous_end > vision_tensors[i]->offset) {
            add_error(report, "overlapping vision tensors: " +
                              vision_tensors[i - 1]->name + " and " +
                              vision_tensors[i]->name);
        }
    }

    std::unordered_set<std::string> expected_names;
    expected_names.reserve(3 + static_cast<size_t>(layers) * 41);
    std::unordered_map<std::string, float> clamp_values;

    auto expect = [&](const std::string& name, const Shape& shape,
                      bool is_clamp = false) {
        expected_names.insert(name);
        const auto found = by_name.find(name);
        if (found == by_name.end()) {
            add_error(report, "missing vision tensor: " + name);
            return;
        }
        const TensorEntry& tensor = *found->second;
        if (tensor.shape != shape) {
            add_error(report, "shape mismatch for " + name + ": got " +
                              shape_string(tensor.shape) + ", expected " +
                              shape_string(shape));
        }
        uint64_t numel = 1;
        for (uint32_t dimension : tensor.shape) {
            if (dimension == 0 || !checked_mul(numel, dimension, numel)) {
                add_error(report, "invalid shape for " + name);
                return;
            }
        }
        uint64_t expected_bytes = 0;
        if (tensor.dtype != "fp16" || !checked_mul(numel, 2, expected_bytes) ||
            tensor.size != expected_bytes) {
            add_error(report, "Gemma 4 vision tensor must be exact FP16: " + name);
        }
        if (is_clamp) {
            ++report.clamp_count;
            uint16_t bits = 0;
            if (tensor.size != sizeof(bits) ||
                !loader.read_tensor_data(tensor, &bits, sizeof(bits))) {
                add_error(report, "cannot read learned clamp: " + name);
                return;
            }
            const float value = half_to_float(bits);
            if (!std::isfinite(value)) {
                add_error(report, "non-finite learned clamp: " + name);
                return;
            }
            clamp_values.emplace(name, value);
        }
    };

    const uint32_t patch_width = static_cast<uint32_t>(patch_width64);
    expect("vision.patch_embed.input_proj.weight", {hidden, patch_width});
    expect("vision.patch_embed.position_embedding.weight",
           {2, config.position_embedding_size, hidden});
    expect("vision.projector.weight", {config.projection_dim, hidden});

    const char* attention_projections[] = {"q", "k", "v", "o"};
    const char* mlp_projections[] = {"gate", "up", "down"};
    const char* clamp_fields[] = {"input_min", "input_max",
                                  "output_min", "output_max"};
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = "vision.layer" + std::to_string(layer) + '.';
        expect(prefix + "ln_attn_in.weight", {hidden});
        expect(prefix + "ln_attn_post.weight", {hidden});
        expect(prefix + "ln_mlp_in.weight", {hidden});
        expect(prefix + "ln_mlp_post.weight", {hidden});
        expect(prefix + "attn.q_norm.weight", {config.head_dim});
        expect(prefix + "attn.k_norm.weight", {config.head_dim});

        for (const char* projection : attention_projections) {
            const std::string linear = prefix + "attn." + projection + "_proj.";
            expect(linear + "weight", {hidden, hidden});
            for (const char* field : clamp_fields) {
                expect(linear + field, {}, true);
            }
        }
        for (const char* projection : mlp_projections) {
            const std::string linear = prefix + "mlp." + projection + '.';
            const bool down = std::string(projection) == "down";
            expect(linear + "weight", down ? Shape{hidden, intermediate}
                                            : Shape{intermediate, hidden});
            for (const char* field : clamp_fields) {
                expect(linear + field, {}, true);
            }
        }
    }

    for (uint32_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = "vision.layer" + std::to_string(layer) + '.';
        std::vector<std::string> linears;
        for (const char* projection : attention_projections) {
            linears.push_back(prefix + "attn." + projection + "_proj.");
        }
        for (const char* projection : mlp_projections) {
            linears.push_back(prefix + "mlp." + projection + '.');
        }
        for (const std::string& linear : linears) {
            const auto input_min = clamp_values.find(linear + "input_min");
            const auto input_max = clamp_values.find(linear + "input_max");
            const auto output_min = clamp_values.find(linear + "output_min");
            const auto output_max = clamp_values.find(linear + "output_max");
            if (input_min != clamp_values.end() && input_max != clamp_values.end() &&
                input_min->second > input_max->second) {
                add_error(report, "inverted input clamp range: " + linear);
            }
            if (output_min != clamp_values.end() && output_max != clamp_values.end() &&
                output_min->second > output_max->second) {
                add_error(report, "inverted output clamp range: " + linear);
            }
        }
    }

    for (const auto& item : by_name) {
        if (expected_names.count(item.first) == 0) {
            add_error(report, "unexpected vision tensor: " + item.first);
        }
    }
    const size_t expected_count = 3 + static_cast<size_t>(layers) * 41;
    if (report.tensor_count != expected_count) {
        add_error(report, "vision tensor count mismatch: got " +
                          std::to_string(report.tensor_count) + ", expected " +
                          std::to_string(expected_count));
    }
    if (report.clamp_count != static_cast<size_t>(layers) * 7 * 4) {
        add_error(report, "learned clamp count mismatch: got " +
                          std::to_string(report.clamp_count));
    }
    if (report.weight_bytes != block.size) {
        add_error(report, "vision tensor bytes do not exactly fill the block");
    }

    // Conservative high-water estimate for the future FP16 forward. Attention
    // scores are FP32 and softmax is assumed in-place; all other terms are
    // FP16. This is a planning bound, not a reservation made by V2.
    uint64_t elements = 0;
    uint64_t term = 0;
    auto add_product = [&](std::initializer_list<uint64_t> factors,
                           uint64_t element_bytes) {
        uint64_t product = 1;
        for (uint64_t factor : factors) {
            if (!checked_mul(product, factor, product)) {
                add_error(report, "vision scratch budget overflows");
                return;
            }
        }
        if (!checked_mul(product, element_bytes, term) ||
            !checked_add(elements, term, elements)) {
            add_error(report, "vision scratch budget overflows");
        }
    };
    add_product({report.max_patches, patch_width}, 2);             // patches
    add_product({4, report.max_patches, hidden}, 2);               // hidden streams
    add_product({3, report.max_patches, hidden}, 2);               // Q/K/V
    add_product({heads, report.max_patches, report.max_patches}, 4); // scores
    add_product({2, report.max_patches, intermediate}, 2);         // gate/up
    add_product({report.max_soft_tokens,
                 static_cast<uint64_t>(hidden) + config.projection_dim}, 2);
    report.scratch_upper_bound_bytes = elements;

    return report;
}

} // namespace helios
