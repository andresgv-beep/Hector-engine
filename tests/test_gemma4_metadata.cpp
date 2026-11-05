// Gemma 4 metadata contract tests. This executable deliberately avoids Engine
// and CUDA allocations: it validates HNF/GM4X parsing only.

#include "hnf_loader.hpp"
#include "hnf10_test_writer.hpp"
#include "gemma4_validator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using namespace helios;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

uint64_t align32(uint64_t value) {
    return (value + 31u) & ~uint64_t{31u};
}

template <typename T>
void write_at(std::vector<uint8_t>& buffer, size_t offset, const T& value) {
    require(offset <= buffer.size() && sizeof(T) <= buffer.size() - offset,
            "internal mock buffer overflow");
    std::memcpy(buffer.data() + offset, &value, sizeof(T));
}

std::vector<uint8_t> make_gemma4_hints(bool corrupt_record_size) {
    constexpr uint32_t extension_offset =
        sizeof(ExecutionHintsBin) + sizeof(TextModelConfigBin);
    constexpr uint32_t layer_count = 2;
    constexpr uint32_t extension_size =
        sizeof(Gemma4ExtensionHeaderBin) + layer_count * sizeof(Gemma4LayerConfigBin);

    std::vector<uint8_t> result(extension_offset + extension_size, 0);

    ExecutionHintsBin hints{};
    hints.magic = HINTS_MAGIC;
    hints.version_major = 1;
    hints.version_minor = 1;
    hints.text_offset = sizeof(ExecutionHintsBin);
    hints.num_text_models = 1;
    hints.flags = 1;
    std::memcpy(hints.reserved, &extension_offset, sizeof(extension_offset));
    std::memcpy(hints.reserved + 4, &extension_size, sizeof(extension_size));
    std::memcpy(hints.reserved + 8, "GM4X", 4);
    write_at(result, 0, hints);

    TextModelConfigBin text{};
    text.rope_theta = 10000.0f;
    text.rope_scaling_factor = 1.0f;
    text.partial_rotary_factor = 1.0f;
    text.rms_norm_eps = 1.0e-6f;
    text.num_hidden_layers = layer_count;
    text.hidden_size = 1536;
    text.intermediate_size = 6144;
    text.vocab_size = 262144;
    text.max_position_embeddings = 32768;
    text.num_attention_heads = 8;
    text.num_key_value_heads = 1;
    text.head_dim = 256;
    text.attention_type = ATTN_MQA;
    text.arch = ARCH_GEMMA4;
    text.dtype = DTYPE_BF16;
    text.mlp_type = MLP_GEGLU;
    text.norm_type = NORM_RMSNORM;
    text.flags = CFG_FLAG_USE_QK_NORM | CFG_FLAG_TIE_EMBEDDINGS;
    write_at(result, hints.text_offset, text);

    Gemma4ExtensionHeaderBin extension{};
    std::memcpy(extension.magic, "GM4X", 4);
    extension.version = 1;
    extension.layer_record_size = corrupt_record_size
        ? sizeof(Gemma4LayerConfigBin) - 1
        : sizeof(Gemma4LayerConfigBin);
    extension.layer_count = layer_count;
    extension.flags = GEMMA4_EXT_FLAG_PLE |
                      GEMMA4_EXT_FLAG_LAYER_SCALAR |
                      GEMMA4_EXT_FLAG_LOGIT_SOFTCAP |
                      GEMMA4_EXT_FLAG_SHARED_KV |
                      GEMMA4_EXT_FLAG_FOUR_NORM_BLOCK;
    extension.global_head_dim = 512;
    extension.ple_hidden_size = 256;
    extension.num_kv_shared_layers = 1;
    write_at(result, extension_offset, extension);

    Gemma4LayerConfigBin local{};
    local.attention_kind = 0;
    local.sliding_window = 512;
    local.head_dim = 256;
    local.intermediate_size = 6144;
    local.rope_type = ROPE_DEFAULT;
    local.rope_theta = 10000.0f;
    local.partial_rotary_factor = 1.0f;
    local.kv_share_group = -1;
    write_at(result, extension_offset + sizeof(extension), local);

    Gemma4LayerConfigBin global{};
    global.attention_kind = 1;
    global.head_dim = 512;
    global.intermediate_size = 12288;
    global.rope_type = ROPE_PROPORTIONAL;
    global.rope_theta = 1000000.0f;
    global.partial_rotary_factor = 0.25f;
    global.kv_share_group = -1;
    write_at(result, extension_offset + sizeof(extension) + sizeof(local), global);

    return result;
}

std::vector<uint8_t> make_moe_hints(const std::vector<uint8_t>& extension) {
    constexpr uint32_t extension_offset =
        sizeof(ExecutionHintsBin) + sizeof(TextModelConfigBin);
    const uint32_t extension_size = static_cast<uint32_t>(extension.size());
    uint32_t layer_count = 0;
    std::memcpy(&layer_count, extension.data() + 8, sizeof(layer_count));

    std::vector<uint8_t> result(extension_offset + extension_size, 0);

    ExecutionHintsBin hints{};
    hints.magic = HINTS_MAGIC;
    hints.version_major = 1;
    hints.version_minor = 1;
    hints.text_offset = sizeof(ExecutionHintsBin);
    hints.num_text_models = 1;
    hints.flags = 1;
    std::memcpy(hints.reserved, &extension_offset, sizeof(extension_offset));
    std::memcpy(hints.reserved + 4, &extension_size, sizeof(extension_size));
    std::memcpy(hints.reserved + 8, "GM4X", 4);
    write_at(result, 0, hints);

    TextModelConfigBin text{};
    text.rope_theta = 10000.0f;
    text.rope_scaling_factor = 1.0f;
    text.partial_rotary_factor = 1.0f;
    text.rms_norm_eps = 1.0e-6f;
    text.num_hidden_layers = layer_count;
    text.hidden_size = 2816;
    text.intermediate_size = 2112;
    text.vocab_size = 262144;
    text.max_position_embeddings = 262144;
    text.num_attention_heads = 16;
    text.num_key_value_heads = 8;
    text.head_dim = 256;
    text.attention_type = ATTN_MQA;
    text.arch = ARCH_GEMMA4;
    text.dtype = DTYPE_BF16;
    text.mlp_type = MLP_GEGLU;
    text.norm_type = NORM_RMSNORM;
    text.flags = CFG_FLAG_USE_QK_NORM | CFG_FLAG_TIE_EMBEDDINGS;
    write_at(result, hints.text_offset, text);

    std::memcpy(result.data() + extension_offset, extension.data(), extension.size());
    return result;
}

std::string write_mock_hnf(const std::string& path,
                           const std::vector<uint8_t>& binary_hints) {
    // HNFv10: un tensor de texto y la configuración binaria de la prueba; sin JSON.
    hnf10_test::Component text{1, hnf10_test::TEXT, true, "texto", {{"text.test.weight", "fp16", {1}, {0, 0}}}, {}};
    hnf10_test::Component config{2, hnf10_test::CONFIG, true, "configuración", {}, binary_hints};
    return hnf10_test::write(path, hnf10_test::build({text, config}));
}

std::string write_mock_hnf(const std::string& path, bool corrupt_record_size) {
    return write_mock_hnf(path, make_gemma4_hints(corrupt_record_size));
}

void test_valid_mock() {
    const std::string path = write_mock_hnf("/tmp/helios_gemma4_metadata.hnf", false);
    HnfLoader loader;
    require(loader.load_metadata(path), "valid GM4X mock must load: " + loader.last_error());
    require(loader.config().arch() == "gemma4", "binary arch 15 must map to gemma4");
    require(loader.has_gemma4_config(), "valid mock must expose a Gemma4Config");

    const Gemma4Config& config = loader.gemma4_config();
    require(config.version == 1, "GM4X version");
    require(config.layers.size() == 2, "GM4X layer count");
    require(config.global_head_dim == 512, "GM4X global head dimension");
    require(config.ple_hidden_size == 256, "GM4X PLE hidden size");
    require(config.num_kv_shared_layers == 1, "GM4X shared KV count");
    require(config.has_flag(GEMMA4_EXT_FLAG_PLE), "GM4X PLE flag");
    require(config.has_flag(GEMMA4_EXT_FLAG_FOUR_NORM_BLOCK), "GM4X four-norm flag");

    require(!config.layers[0].is_global_attention(), "layer 0 must be sliding");
    require(config.layers[0].sliding_window == 512, "layer 0 sliding window");
    require(config.layers[0].head_dim == 256, "layer 0 head dimension");
    require(config.layers[1].is_global_attention(), "layer 1 must be global");
    require(config.layers[1].head_dim == 512, "layer 1 head dimension");
    require(config.layers[1].intermediate_size == 12288, "layer 1 MLP width");
    require(config.layers[1].rope_type == ROPE_PROPORTIONAL, "layer 1 RoPE type");
    require(std::fabs(config.layers[1].rope_theta - 1000000.0f) < 1.0f,
            "layer 1 RoPE theta");
    require(std::fabs(config.layers[1].partial_rotary_factor - 0.25f) < 1.0e-6f,
            "layer 1 partial rotary factor");

    std::remove(path.c_str());
    std::cout << "PASS: valid synthetic GM4X contract" << std::endl;
}

void test_corrupt_mock_is_rejected() {
    const std::string path = write_mock_hnf("/tmp/helios_gemma4_metadata_bad.hnf", true);
    HnfLoader loader;
    require(!loader.load_metadata(path), "corrupt GM4X record size must be rejected");
    require(loader.last_error().find("record size") != std::string::npos,
            "corrupt GM4X must report its record-size error");
    std::remove(path.c_str());
    std::cout << "PASS: corrupt GM4X contract rejected" << std::endl;
}

std::vector<uint8_t> read_fixture(const std::string& name) {
    std::ifstream input(std::string{GM4X_V2_FIXTURES} + "/" + name, std::ios::binary);
    require(input.is_open(), "cannot open fixture " + name);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), {});
}

template <typename T>
void poke(std::vector<uint8_t>& buffer, size_t offset, T value) {
    write_at(buffer, offset, value);
}

bool load_extension(const std::vector<uint8_t>& extension, HnfLoader& loader) {
    const std::string path =
        write_mock_hnf("/tmp/helios_gemma4_moe_metadata.hnf", make_moe_hints(extension));
    const bool loaded = loader.load_metadata(path);
    std::remove(path.c_str());
    return loaded;
}

void require_rejected(const std::string& label, const std::vector<uint8_t>& extension) {
    HnfLoader loader;
    require(!load_extension(extension, loader), label + " must be rejected");
    require(loader.last_error().find("GM4X") != std::string::npos,
            label + " must fail in GM4X, not elsewhere: " + loader.last_error());
}

void test_gm4x_v2_layouts() {
    HnfLoader loader;
    require(load_extension(read_fixture("valid-layer0-bf16.bin"), loader),
            "BF16 GM4X v2 must load: " + loader.last_error());
    const Gemma4Config& config = loader.gemma4_config();
    require(config.version == 2, "GM4X v2 version");
    require(config.has_flag(GEMMA4_EXT_FLAG_MOE), "GM4X v2 MoE flag");
    require(config.moe_experts == 128 && config.moe_top_k == 8 &&
            config.moe_intermediate_size == 704, "GM4X v2 topology");

    const Gemma4LayerConfig& layer = config.layers.at(0);
    require(layer.intermediate_size == 2112, "GM4X v2 keeps the v1 record");
    require(layer.moe_gate_up.rows == 1408 && layer.moe_gate_up.logical_cols == 2816 &&
            layer.moe_gate_up.physical_cols == 2816 &&
            layer.moe_gate_up.experts == 128, "gate_up keeps its columns");
    require(layer.moe_down.rows == 2816 && layer.moe_down.logical_cols == 704 &&
            layer.moe_down.physical_cols == 768 && layer.moe_down.row_bytes == 1536 &&
            layer.moe_down.expert_stride == 4325376, "down pads to 768 columns");
    require(layer.moe_shared_down.logical_cols == 2112 &&
            layer.moe_shared_down.physical_cols == 2304 &&
            layer.moe_shared_down.experts == 1, "shared down pads to 2304 columns");
    require(layer.moe_down.expert_offset(127) == 127 * uint64_t{4325376},
            "last expert offset");

    HnfLoader hq44;
    require(load_extension(read_fixture("valid-layer0-hq44.bin"), hq44),
            "HQ44 GM4X v2 must load: " + hq44.last_error());
    const Gemma4LayerConfig& quantized = hq44.gemma4_config().layers.at(0);
    require(quantized.moe_gate_up.storage == GEMMA4_MOE_STORAGE_HQ44 &&
            quantized.moe_gate_up.row_bytes == 11 * 152 &&
            quantized.moe_gate_up.expert_stride == 11 * 152 * 1408u, "HQ44 gate_up");
    require(quantized.moe_down.row_bytes == 3 * 152 &&
            quantized.moe_down.expert_stride == 3 * 152 * 2816u, "HQ44 down");

    HnfLoader full;
    require(load_extension(read_fixture("valid-full30-bf16.bin"), full),
            "30-layer GM4X v2 must load: " + full.last_error());
    require(full.gemma4_config().layers.size() == 30, "30 GM4X v2 layers");
    require(full.gemma4_config().layers.back().moe_down.physical_cols == 768,
            "last layer down layout");
    std::cout << "PASS: GM4X v2 layouts" << std::endl;
}

void test_gm4x_v2_rejections() {
    for (const char* name : {"reject-version.bin", "reject-topk.bin", "reject-down-stride.bin",
                             "reject-physical-cols.bin", "reject-dtype.bin", "reject-reserved.bin",
                             "reject-gate-rows.bin", "reject-overflow-stride.bin",
                             "reject-truncated.bin"}) {
        require_rejected(name, read_fixture(name));
    }

    constexpr size_t layer0 = sizeof(Gemma4ExtensionV2HeaderBin);
    constexpr size_t moe = sizeof(Gemma4LayerConfigBin);
    constexpr size_t gate_up = layer0 + offsetof(Gemma4LayerConfigV2Bin, gate_up);
    constexpr size_t down = layer0 + offsetof(Gemma4LayerConfigV2Bin, down);
    constexpr size_t shared = layer0 + offsetof(Gemma4LayerConfigV2Bin, shared_down);
    const std::vector<uint8_t> valid = read_fixture("valid-layer0-hq44.bin");
    const auto mutated = [&](size_t offset, auto value) {
        std::vector<uint8_t> copy = valid;
        poke(copy, offset, value);
        return copy;
    };

    require_rejected("v2 without the MoE flag", mutated(12, uint32_t{0}));
    require_rejected("zero top-k", mutated(layer0 + moe + 4, uint32_t{0}));
    require_rejected("wrong descriptor mask", mutated(layer0 + moe + 12, uint32_t{6}));
    require_rejected("wrong descriptor count", mutated(layer0 + moe + 16, uint32_t{2}));
    require_rejected("swapped matrix role", mutated(gate_up, uint32_t{GEMMA4_MOE_ROLE_DOWN}));
    require_rejected("HQ44 with a float layout", mutated(down + 28, uint32_t{0}));
    require_rejected("physical rows differ", mutated(down + 16, uint32_t{2815}));
    require_rejected("wrong row bytes", mutated(down + 32, uint64_t{304}));
    require_rejected("shared down width", mutated(shared + 12, uint32_t{2111}));
    require_rejected("shared down per expert", mutated(shared + 24, uint32_t{128}));

    std::vector<uint8_t> uneven = read_fixture("valid-full30-bf16.bin");
    poke(uneven, layer0 + 7 * sizeof(Gemma4LayerConfigV2Bin) + moe + 4, uint32_t{4});
    require_rejected("top-k that changes between layers", uneven);

    std::vector<uint8_t> flagged = make_gemma4_hints(false);
    const size_t v1_flags = sizeof(ExecutionHintsBin) + sizeof(TextModelConfigBin) + 12;
    uint32_t flags = 0;
    std::memcpy(&flags, flagged.data() + v1_flags, sizeof(flags));
    poke(flagged, v1_flags, flags | GEMMA4_EXT_FLAG_MOE);
    const std::string path = write_mock_hnf("/tmp/helios_gemma4_v1_moe.hnf", flagged);
    HnfLoader loader;
    require(!loader.load_metadata(path), "v1 cannot carry the MoE flag");
    std::remove(path.c_str());
    std::cout << "PASS: GM4X v2 rejections" << std::endl;
}

void test_moe_tensor_contract() {
    HnfLoader loader;
    require(load_extension(read_fixture("valid-layer0-hq44.bin"), loader),
            "HQ44 GM4X v2 must load: " + loader.last_error());
    const Gemma4Config& gemma = loader.gemma4_config();
    constexpr uint32_t D = 2816, E = 128;
    const auto entry = [](const std::string& name, std::vector<uint32_t> shape, const char* dtype, uint64_t size) {
        TensorEntry tensor{};
        tensor.name = name;
        tensor.shape = std::move(shape);
        tensor.dtype = dtype;
        tensor.size = size;
        return tensor;
    };
    const std::string p = "text.layer0.";
    std::vector<TensorEntry> tensors{
        entry(p + "moe.experts.gate_up.weight", {E, 1408, D}, "hq44k_g16", E * uint64_t{2354176}),
        entry(p + "moe.experts.down.weight", {E, D, 768}, "hq44k_g16", E * uint64_t{1284096}),
        entry(p + "mlp.down.weight", {D, 2304}, "hq44k_g16", 3852288),
        entry(p + "moe.router.weight", {E, D}, "fp16", E * uint64_t{D} * 2),
        entry(p + "moe.router.input_scale", {D}, "fp16", D * 2),
        entry(p + "moe.router.expert_scale", {E}, "fp16", E * 2),
        entry(p + "ln_mlp_branch_post.weight", {D}, "fp16", D * 2),
        entry(p + "ln_moe_in.weight", {D}, "fp16", D * 2),
        entry(p + "ln_moe_branch_post.weight", {D}, "fp16", D * 2),
    };
    const auto errors = validate_gemma4_moe_tensors(gemma, D, tensors);
    for (const auto& error : errors) std::cerr << "  " << error << std::endl;
    require(errors.empty(), "tensores MoE coherentes");

    const auto rejected = [&](const std::string& label, auto&& change) {
        auto broken = tensors;
        change(broken);
        require(!validate_gemma4_moe_tensors(gemma, D, broken).empty(), label + " must be rejected");
    };
    rejected("missing expert tensor", [](auto& all) { all.erase(all.begin()); });
    rejected("unpadded down", [](auto& all) { all[1].shape[2] = 704; });
    rejected("expert dtype", [](auto& all) { all[0].dtype = "fp16"; });
    rejected("truncated expert tensor", [](auto& all) { all[1].size -= 152; });
    rejected("unpadded shared down", [](auto& all) { all[2].shape[1] = 2112; });
    rejected("quantized router", [](auto& all) { all[3].dtype = "hq44k_g16"; });
    std::cout << "PASS: tensores MoE contra GM4X v2" << std::endl;
}

void inspect_real_hnf(const std::string& path) {
    HnfLoader loader;
    require(loader.load_metadata(path), "real HNF metadata must load: " + loader.last_error());
    require(loader.config().arch() == "gemma4", "real HNF arch must be gemma4");
    require(loader.has_gemma4_config(), "real HNF must contain GM4X");

    const Gemma4Config& config = loader.gemma4_config();
    require(config.layers.size() == 35, "real HNF must contain 35 GM4X layers");
    require(config.global_head_dim == 512, "real HNF global head dimension");
    require(config.ple_hidden_size == 256, "real HNF PLE hidden size");
    require(config.num_kv_shared_layers == 20, "real HNF shared KV count");
    require(config.flags == 0x3f, "real HNF GM4X feature flags");
    require(config.layers[4].is_global_attention(), "real layer 4 must be global");
    require(config.layers[4].head_dim == 512, "real layer 4 head dimension");
    require(config.layers[4].rope_type == ROPE_PROPORTIONAL,
            "real layer 4 RoPE type");
    require(std::fabs(config.layers[4].rope_theta - 1000000.0f) < 1.0f,
            "real layer 4 RoPE theta");
    require(std::fabs(config.layers[4].partial_rotary_factor - 0.25f) < 1.0e-6f,
            "real layer 4 partial rotary factor");
    require(config.layers[15].intermediate_size == 12288,
            "real layer 15 MLP width");
    const bool shared_embedding =
        loader.config().get<bool>("tie_word_embeddings", false) &&
        std::none_of(loader.tensors().begin(), loader.tensors().end(),
                     [](const TensorEntry& tensor) {
                         return tensor.name == "text.lm_head.weight";
                     });
    require(loader.tensors().size() == (shared_embedding ? 600u : 601u),
            "real HNF tensor count");

    const Gemma4ValidationReport validation =
        validate_gemma4_tensors(loader, 1, 512);
    if (!validation.ok()) {
        for (const std::string& error : validation.errors) {
            std::cerr << "  validation: " << error << std::endl;
        }
    }
    require(validation.ok(), "real HNF tensor contract must validate");

    size_t fp16 = 0;
    size_t hq41k = 0;
    size_t hq51k = 0;
    for (const TensorEntry& tensor : loader.tensors()) {
        fp16 += tensor.dtype == "fp16";
        hq41k += tensor.dtype == "hq41k";
        hq51k += tensor.dtype == "hq51k";
    }

    std::cout << "PASS: real Gemma 4 HNF metadata" << std::endl;
    std::cout << "  tensors=" << loader.tensors().size()
              << " fp16=" << fp16
              << " hq41k=" << hq41k
              << " hq51k=" << hq51k << std::endl;
    std::cout << "  weights=" << validation.tensor_bytes
              << " ple=" << validation.ple_embedding_bytes
              << " scratch_1x512=" << validation.core_scratch_bytes
              << " ple_workspace_1x512=" << validation.ple_workspace_bytes
              << " kv_upper_1x512=" << validation.kv_cache_upper_bound_bytes
              << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    test_valid_mock();
    test_corrupt_mock_is_rejected();
    test_gm4x_v2_layouts();
    test_gm4x_v2_rejections();
    test_moe_tensor_contract();
    if (argc == 2) inspect_real_hnf(argv[1]);
    if (argc > 2) {
        std::cerr << "Usage: test_gemma4_metadata [gemma4.hnf]" << std::endl;
        return 2;
    }
    return 0;
}
