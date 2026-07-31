// Gemma 4 metadata contract tests. This executable deliberately avoids Engine
// and CUDA allocations: it validates HNF/GM4X parsing only.

#include "hnf_loader.hpp"
#include "gemma4_validator.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
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

std::string write_mock_hnf(const std::string& path, bool corrupt_record_size) {
    const std::vector<uint8_t> binary_hints = make_gemma4_hints(corrupt_record_size);
    const std::string json_hints = R"({
        "text_enabled": true,
        "text": {
            "arch": "gemma4",
            "num_hidden_layers": 2,
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "vocab_size": 262144,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256
        }
    })";

    const uint64_t text_offset = align32(HNF_HEADER_SIZE + HNF_BLOCK_TABLE_SIZE);
    const uint64_t text_size = 2;
    const uint64_t json_offset = align32(text_offset + text_size);
    const uint64_t binary_offset = align32(json_offset + json_hints.size());
    const uint64_t manifest_offset = align32(binary_offset + binary_hints.size());

    const std::string manifest = std::string{R"({"version":"9.0","tensors":[{
        "name":"text.test.weight","dtype":"fp16","block":"text_model",
        "shape":[1],"offset":)"} + std::to_string(text_offset) +
        R"(,"size":2}]})";

    HnfHeader header{};
    std::memcpy(header.magic, HNF_MAGIC, sizeof(HNF_MAGIC));
    header.version_major = 9;
    header.version_minor = 1;
    header.flags = HNF_HAS_EXEC_HINTS_BIN;
    header.block_count = HNF_BLOCK_COUNT;
    header.header_size = HNF_HEADER_SIZE;
    header.block_table_offset = HNF_HEADER_SIZE;
    header.manifest_offset = manifest_offset;
    header.manifest_size = manifest.size();
    header.file_size = manifest_offset + manifest.size();

    std::array<BlockEntry, HNF_BLOCK_COUNT> blocks{};
    blocks[BLOCK_TEXT_MODEL] = {BLOCK_TEXT_MODEL, BLOCK_TEXT_MODEL,
                                text_offset, text_size, 0};
    blocks[BLOCK_EXEC_HINTS] = {BLOCK_EXEC_HINTS, BLOCK_EXEC_HINTS,
                               json_offset, json_hints.size(), 0};
    blocks[BLOCK_EXEC_HINTS_BIN] = {BLOCK_EXEC_HINTS_BIN, BLOCK_EXEC_HINTS_BIN,
                                   binary_offset, binary_hints.size(), 0};

    std::vector<uint8_t> file(header.file_size, 0);
    write_at(file, 0, header);
    std::memcpy(file.data() + header.block_table_offset, blocks.data(),
                sizeof(BlockEntry) * blocks.size());
    file[text_offset] = 0;
    file[text_offset + 1] = 0;
    std::memcpy(file.data() + json_offset, json_hints.data(), json_hints.size());
    std::memcpy(file.data() + binary_offset, binary_hints.data(), binary_hints.size());
    std::memcpy(file.data() + manifest_offset, manifest.data(), manifest.size());

    std::ofstream output(path, std::ios::binary);
    require(output.is_open(), "cannot create mock HNF");
    output.write(reinterpret_cast<const char*>(file.data()), file.size());
    require(output.good(), "cannot write mock HNF");
    return path;
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
    require(loader.tensors().size() == 601, "real HNF tensor count");

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
              << " kv_upper_1x512=" << validation.kv_cache_upper_bound_bytes
              << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    test_valid_mock();
    test_corrupt_mock_is_rejected();
    if (argc == 2) inspect_real_hnf(argv[1]);
    if (argc > 2) {
        std::cerr << "Usage: test_gemma4_metadata [gemma4.hnf]" << std::endl;
        return 2;
    }
    return 0;
}
