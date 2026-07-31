#include "gemma4_validator.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace helios {
namespace {

using Shape = std::vector<uint32_t>;

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

bool tensor_numel(const TensorEntry& tensor, uint64_t& numel) {
    numel = 1;
    if (tensor.shape.empty()) return false;
    for (uint32_t dim : tensor.shape) {
        if (dim == 0 || !checked_mul(numel, dim, numel)) return false;
    }
    return true;
}

bool expected_storage_bytes(const TensorEntry& tensor, uint64_t& bytes) {
    uint64_t numel = 0;
    if (!tensor_numel(tensor, numel)) return false;
    if (tensor.dtype == "fp16" || tensor.dtype == "bf16") {
        return checked_mul(numel, 2, bytes);
    }
    if (tensor.dtype == "fp32") {
        return checked_mul(numel, 4, bytes);
    }
    if (tensor.dtype == "hq41k" || tensor.dtype == "hq51k") {
        const uint64_t block_bytes = tensor.dtype == "hq41k" ? 168 : 200;
        const uint64_t blocks = (numel + 255) / 256;
        return checked_mul(blocks, block_bytes, bytes);
    }
    return false;
}

void add_error(Gemma4ValidationReport& report, const std::string& message) {
    report.errors.push_back(message);
}

} // namespace

Gemma4ValidationReport validate_gemma4_tensors(
    const HnfLoader& loader,
    uint32_t budget_batch_size,
    uint32_t budget_sequence_length) {
    Gemma4ValidationReport report;
    report.budget_batch_size = budget_batch_size;
    report.budget_sequence_length = budget_sequence_length;

    if (loader.config().arch() != "gemma4") {
        add_error(report, "model architecture is not gemma4");
        return report;
    }
    if (!loader.has_gemma4_config()) {
        add_error(report, "GM4X per-layer configuration is missing");
        return report;
    }

    const ModelConfig& model = loader.config();
    const Gemma4Config& gemma = loader.gemma4_config();
    const uint32_t layers = model.num_hidden_layers();
    const uint32_t hidden = model.hidden_size();
    const uint32_t vocab = model.vocab_size();
    const uint32_t heads = model.num_attention_heads();
    const uint32_t kv_heads = model.num_key_value_heads();
    const uint32_t ple_hidden = gemma.ple_hidden_size;

    if (layers == 0 || hidden == 0 || vocab == 0 || heads == 0 || kv_heads == 0) {
        add_error(report, "base Gemma 4 dimensions contain zero values");
        return report;
    }
    if (gemma.layers.size() != layers) {
        add_error(report, "GM4X layer count differs from the base config");
        return report;
    }

    std::unordered_map<std::string, const TensorEntry*> by_name;
    by_name.reserve(loader.tensors().size());
    std::vector<const TensorEntry*> text_tensors;
    text_tensors.reserve(loader.tensors().size());
    bool compact_profile = false;

    const BlockEntry& text_block = loader.block(BLOCK_TEXT_MODEL);
    uint64_t text_end = 0;
    if (!checked_add(text_block.offset, text_block.size, text_end)) {
        add_error(report, "text block range overflows uint64");
        return report;
    }

    for (const TensorEntry& tensor : loader.tensors()) {
        if (tensor.block != "text_model") {
            add_error(report, "unexpected non-text tensor in Gemma 4 manifest: " + tensor.name);
            continue;
        }
        text_tensors.push_back(&tensor);
        report.tensor_count++;
        report.dtype_counts[tensor.dtype]++;
        compact_profile |= tensor.dtype == "hq41k" || tensor.dtype == "hq51k";

        if (!by_name.emplace(tensor.name, &tensor).second) {
            add_error(report, "duplicate tensor name: " + tensor.name);
        }
        if (!checked_add(report.tensor_bytes, tensor.size, report.tensor_bytes)) {
            add_error(report, "tensor byte total overflows uint64");
        }
        report.largest_tensor_bytes = std::max(report.largest_tensor_bytes, tensor.size);
        if (tensor.name == "text.ple.token_embedding.weight") {
            report.ple_embedding_bytes = tensor.size;
        }

        uint64_t tensor_end = 0;
        if (tensor.offset < text_block.offset ||
            !checked_add(tensor.offset, tensor.size, tensor_end) || tensor_end > text_end) {
            add_error(report, "tensor range is outside text block: " + tensor.name);
        }

        uint64_t storage_bytes = 0;
        if (!expected_storage_bytes(tensor, storage_bytes)) {
            add_error(report, "unsupported dtype or invalid shape: " + tensor.name);
        } else if (storage_bytes != tensor.size) {
            add_error(report, "stored byte size mismatch for " + tensor.name);
        }
    }

    std::sort(text_tensors.begin(), text_tensors.end(),
              [](const TensorEntry* a, const TensorEntry* b) {
                  return a->offset < b->offset;
              });
    for (size_t i = 1; i < text_tensors.size(); ++i) {
        uint64_t previous_end = 0;
        if (checked_add(text_tensors[i - 1]->offset, text_tensors[i - 1]->size,
                        previous_end) && previous_end > text_tensors[i]->offset) {
            add_error(report, "overlapping tensors: " + text_tensors[i - 1]->name +
                              " and " + text_tensors[i]->name);
        }
    }

    std::unordered_set<std::string> expected_names;
    expected_names.reserve(6 + static_cast<size_t>(layers) * 17);

    auto expect = [&](const std::string& name, const Shape& shape,
                      const std::string& compact_dtype) {
        expected_names.insert(name);
        const auto found = by_name.find(name);
        if (found == by_name.end()) {
            add_error(report, "missing tensor: " + name);
            return;
        }
        const TensorEntry& tensor = *found->second;
        if (tensor.shape != shape) {
            add_error(report, "shape mismatch for " + name + ": got " +
                              shape_string(tensor.shape) + ", expected " +
                              shape_string(shape));
        }
        const std::string expected_dtype = compact_profile ? compact_dtype : "fp16";
        if (tensor.dtype != expected_dtype) {
            add_error(report, "dtype mismatch for " + name + ": got " + tensor.dtype +
                              ", expected " + expected_dtype);
        }
    };

    const uint64_t ple_width64 = static_cast<uint64_t>(layers) * ple_hidden;
    if (ple_width64 > std::numeric_limits<uint32_t>::max()) {
        add_error(report, "PLE projection width exceeds uint32");
        return report;
    }
    const uint32_t ple_width = static_cast<uint32_t>(ple_width64);

    expect("text.token_embedding.weight", {vocab, hidden}, "fp16");
    expect("text.ple.token_embedding.weight", {vocab, ple_width}, "hq51k");
    expect("text.ple.model_projection.weight", {ple_width, hidden}, "hq51k");
    expect("text.ple.projection_norm.weight", {ple_hidden}, "fp16");
    expect("text.final_norm.weight", {hidden}, "fp16");
    expect("text.lm_head.weight", {vocab, hidden}, "hq51k");

    for (uint32_t layer_index = 0; layer_index < layers; ++layer_index) {
        const Gemma4LayerConfig& layer = gemma.layers[layer_index];
        const std::string prefix = "text.layer" + std::to_string(layer_index) + '.';
        const uint64_t q_width64 = static_cast<uint64_t>(heads) * layer.head_dim;
        const uint64_t kv_width64 = static_cast<uint64_t>(kv_heads) * layer.head_dim;
        if (q_width64 > std::numeric_limits<uint32_t>::max() ||
            kv_width64 > std::numeric_limits<uint32_t>::max()) {
            add_error(report, "attention width exceeds uint32 in layer " +
                              std::to_string(layer_index));
            continue;
        }
        const uint32_t q_width = static_cast<uint32_t>(q_width64);
        const uint32_t kv_width = static_cast<uint32_t>(kv_width64);
        const uint32_t intermediate = layer.intermediate_size;

        expect(prefix + "attn.q_proj.weight", {q_width, hidden}, "hq51k");
        expect(prefix + "attn.k_proj.weight", {kv_width, hidden}, "hq51k");
        expect(prefix + "attn.v_proj.weight", {kv_width, hidden}, "hq51k");
        expect(prefix + "attn.o_proj.weight", {hidden, q_width}, "hq51k");
        expect(prefix + "attn.q_norm.weight", {layer.head_dim}, "fp16");
        expect(prefix + "attn.k_norm.weight", {layer.head_dim}, "fp16");
        expect(prefix + "mlp.gate.weight", {intermediate, hidden}, "hq41k");
        expect(prefix + "mlp.up.weight", {intermediate, hidden}, "hq41k");
        expect(prefix + "mlp.down.weight", {hidden, intermediate}, "hq41k");
        expect(prefix + "ln_attn_in.weight", {hidden}, "fp16");
        expect(prefix + "ln_attn_post.weight", {hidden}, "fp16");
        expect(prefix + "ln_mlp_in.weight", {hidden}, "fp16");
        expect(prefix + "ln_mlp_post.weight", {hidden}, "fp16");
        expect(prefix + "ln_ple_post.weight", {hidden}, "fp16");
        expect(prefix + "ple.input_gate.weight", {ple_hidden, hidden}, "hq51k");
        expect(prefix + "ple.projection.weight", {hidden, ple_hidden}, "hq51k");
        expect(prefix + "layer_scalar", {1}, "fp16");
    }

    for (const auto& item : by_name) {
        if (expected_names.count(item.first) == 0) {
            add_error(report, "unexpected tensor: " + item.first);
        }
    }
    const size_t expected_count = 6 + static_cast<size_t>(layers) * 17;
    if (report.tensor_count != expected_count) {
        add_error(report, "tensor count mismatch: got " +
                          std::to_string(report.tensor_count) + ", expected " +
                          std::to_string(expected_count));
    }

    // Safe, pre-allocation budgets for the current core scratch layout and a
    // non-shared per-layer KV cache. The latter is intentionally an upper bound.
    uint32_t max_head_dim = 0;
    uint32_t max_intermediate = 0;
    uint64_t sum_head_dims = 0;
    for (const Gemma4LayerConfig& layer : gemma.layers) {
        max_head_dim = std::max(max_head_dim, layer.head_dim);
        max_intermediate = std::max(max_intermediate, layer.intermediate_size);
        checked_add(sum_head_dims, layer.head_dim, sum_head_dims);
    }

    uint64_t per_position = 5ull * hidden +
        2ull * heads * max_head_dim + 2ull * kv_heads * max_head_dim +
        4ull * max_intermediate;
    uint64_t scratch_elements = 0;
    uint64_t batch_positions = 0;
    if (checked_mul(budget_batch_size, budget_sequence_length, batch_positions) &&
        checked_mul(batch_positions, per_position, scratch_elements)) {
        uint64_t logits_elements = 0;
        checked_mul(budget_batch_size, vocab, logits_elements);
        checked_add(scratch_elements, logits_elements, scratch_elements);
        checked_mul(scratch_elements, 2, report.core_scratch_bytes);
    } else {
        add_error(report, "scratch budget overflows uint64");
    }

    uint64_t ple_workspace_elements = 0;
    if (checked_mul(batch_positions, ple_width, ple_workspace_elements) &&
        checked_mul(ple_workspace_elements, 2, ple_workspace_elements) &&
        checked_mul(ple_workspace_elements, 2, report.ple_workspace_bytes)) {
        // Two FP16 buffers: packed token/result and contextual projection.
    } else {
        add_error(report, "PLE workspace budget overflows uint64");
    }

    uint64_t kv_elements = 0;
    if (checked_mul(sum_head_dims, kv_heads, kv_elements) &&
        checked_mul(kv_elements, 2, kv_elements) && // K and V
        checked_mul(kv_elements, budget_batch_size, kv_elements) &&
        checked_mul(kv_elements, budget_sequence_length, kv_elements) &&
        checked_mul(kv_elements, 2, report.kv_cache_upper_bound_bytes)) {
        // Computed successfully.
    } else {
        add_error(report, "KV cache budget overflows uint64");
    }

    return report;
}

} // namespace helios
