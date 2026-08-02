// Gemma 4 vision metadata/inventory test. The synthetic path needs no GPU;
// passing a real HNF optionally exercises the 659-tensor load/unload cycle.

#include "gemma4_vision_validator.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

std::vector<uint8_t> make_gm4v_hints(bool corrupt_record_size) {
    constexpr uint32_t vision_offset = sizeof(ExecutionHintsBin);
    constexpr uint32_t extension_offset =
        sizeof(ExecutionHintsBin) + sizeof(VisionModelConfigBin);
    std::vector<uint8_t> result(
        extension_offset + sizeof(Gemma4VisionExtensionBin), 0);

    ExecutionHintsBin hints{};
    hints.magic = HINTS_MAGIC;
    hints.version_major = 1;
    hints.version_minor = 1;
    hints.vision_offset = vision_offset;
    hints.num_vision_models = 1;
    hints.flags = 0x0002;
    const uint32_t extension_size = sizeof(Gemma4VisionExtensionBin);
    std::memcpy(hints.reserved + 12, &extension_offset, sizeof(extension_offset));
    std::memcpy(hints.reserved + 16, &extension_size, sizeof(extension_size));
    write_at(result, 0, hints);

    VisionModelConfigBin vision{};
    vision.encoder_type = VISION_ENCODER_GEMMA4;
    vision.patch_size = 16;
    vision.hidden_size = 768;
    vision.num_hidden_layers = 16;
    vision.num_attention_heads = 12;
    vision.intermediate_size = 3072;
    vision.num_channels = 3;
    vision.layer_norm_eps = 1.0e-6f;
    vision.projection_dim = 1536;
    vision.projector_type = 0;
    vision.num_image_tokens = 280;
    vision.image_token_id = 258880;
    write_at(result, vision_offset, vision);

    Gemma4VisionExtensionBin extension{};
    std::memcpy(extension.magic, "GM4V", 4);
    extension.version = 1;
    extension.record_size = corrupt_record_size
        ? sizeof(Gemma4VisionExtensionBin) - 1
        : sizeof(Gemma4VisionExtensionBin);
    extension.flags = 0xfdfd;
    extension.num_key_value_heads = 12;
    extension.head_dim = 64;
    extension.position_embedding_size = 10240;
    extension.pooling_kernel_size = 3;
    extension.max_soft_tokens = 280;
    extension.projection_dim = 1536;
    extension.patch_size = 16;
    extension.image_token_id = 258880;
    extension.boi_token_id = 255999;
    extension.eoi_token_id = 258882;
    extension.pad_token_id = 0;
    extension.rope_theta = 100.0f;
    extension.attention_scale = 1.0f;
    extension.rms_norm_eps = 1.0e-6f;
    extension.rescale_factor = 1.0f / 255.0f;
    extension.resize_multiple = 48;
    extension.resample = 3;
    extension.activation = 1;
    extension.patch_order = 1;
    extension.position_order = 1;
    write_at(result, extension_offset, extension);
    return result;
}

std::string write_mock_hnf(const std::string& path, bool corrupt_record_size) {
    const std::vector<uint8_t> binary_hints =
        make_gm4v_hints(corrupt_record_size);
    const uint64_t vision_offset = align32(HNF_HEADER_SIZE + HNF_BLOCK_TABLE_SIZE);
    const uint64_t vision_size = 2;
    const uint64_t hints_offset = align32(vision_offset + vision_size);
    const uint64_t manifest_offset = align32(hints_offset + binary_hints.size());
    const std::string manifest = std::string{R"({"version":"9.1","tensors":[{
        "name":"vision.scalar","dtype":"fp16","block":"vision",
        "shape":[],"offset":)"} + std::to_string(vision_offset) +
        R"(,"size":2}]})";

    HnfHeader header{};
    std::memcpy(header.magic, HNF_MAGIC, sizeof(HNF_MAGIC));
    header.version_major = 9;
    header.version_minor = 1;
    header.flags = HNF_HAS_VISION | HNF_HAS_EXEC_HINTS_BIN | HNF_IS_MULTIMODAL;
    header.block_count = HNF_BLOCK_COUNT;
    header.header_size = HNF_HEADER_SIZE;
    header.block_table_offset = HNF_HEADER_SIZE;
    header.manifest_offset = manifest_offset;
    header.manifest_size = manifest.size();
    header.file_size = manifest_offset + manifest.size();

    std::array<BlockEntry, HNF_BLOCK_COUNT> blocks{};
    blocks[BLOCK_VISION] = {BLOCK_VISION, BLOCK_VISION,
                            vision_offset, vision_size, 0};
    blocks[BLOCK_EXEC_HINTS_BIN] = {BLOCK_EXEC_HINTS_BIN, BLOCK_EXEC_HINTS_BIN,
                                   hints_offset, binary_hints.size(), 0};

    std::vector<uint8_t> file(header.file_size, 0);
    write_at(file, 0, header);
    std::memcpy(file.data() + header.block_table_offset, blocks.data(),
                sizeof(BlockEntry) * blocks.size());
    const uint16_t one = 0x3c00;
    std::memcpy(file.data() + vision_offset, &one, sizeof(one));
    std::memcpy(file.data() + hints_offset, binary_hints.data(),
                binary_hints.size());
    std::memcpy(file.data() + manifest_offset, manifest.data(), manifest.size());

    std::ofstream output(path, std::ios::binary);
    require(output.is_open(), "cannot create GM4V mock HNF");
    output.write(reinterpret_cast<const char*>(file.data()), file.size());
    require(output.good(), "cannot write GM4V mock HNF");
    return path;
}

void test_synthetic_contract() {
    const std::string valid_path =
        write_mock_hnf("/tmp/helios_gm4v_metadata.hnf", false);
    HnfLoader loader;
    require(loader.load_metadata(valid_path),
            "valid synthetic GM4V must load: " + loader.last_error());
    require(loader.has_gemma4_vision_config(), "GM4V config must be exposed");
    require(loader.config_for_block(BLOCK_VISION).get<std::string>(
                "encoder_type", "") == "gemma4",
            "encoder type 4 must map to gemma4");
    const Gemma4VisionConfig& config = loader.gemma4_vision_config();
    require(config.version == 1, "GM4V version");
    require(config.flags == 0xfdfd, "GM4V flags");
    require(config.max_patches() == 2520, "GM4V patch budget");
    require(config.position_embedding_size == 10240, "GM4V position table");
    require(config.projection_dim == 1536, "GM4V projection width");
    require(std::fabs(config.rope_theta - 100.0f) < 1.0e-6f,
            "GM4V RoPE theta");
    std::remove(valid_path.c_str());

    const std::string invalid_path =
        write_mock_hnf("/tmp/helios_gm4v_metadata_bad.hnf", true);
    HnfLoader invalid;
    require(!invalid.load_metadata(invalid_path),
            "corrupt GM4V record size must be rejected");
    require(invalid.last_error().find("version or record size") != std::string::npos,
            "corrupt GM4V must report its record size");
    std::remove(invalid_path.c_str());
    std::cout << "PASS: synthetic GM4V contract and rejection" << std::endl;
}

void inspect_real_hnf(const std::string& path, bool exercise_load,
                      bool expect_mmap) {
    HnfLoader metadata;
    require(metadata.load_metadata(path),
            "real vision HNF metadata must load: " + metadata.last_error());
    const Gemma4VisionValidationReport report =
        validate_gemma4_vision_tensors(metadata);
    if (!report.ok()) {
        for (const std::string& error : report.errors) {
            std::cerr << "  validation: " << error << std::endl;
        }
    }
    require(report.ok(), "real Gemma 4 vision contract must validate");
    require(report.tensor_count == 659, "real vision tensor count");
    require(report.clamp_count == 448, "real learned clamp count");
    require(report.weight_bytes == 337089408, "real vision weight bytes");
    require(report.max_patches == 2520, "real maximum patch count");

    std::cout << "PASS: real Gemma 4 vision metadata" << std::endl;
    std::cout << "  tensors=" << report.tensor_count
              << " clamps=" << report.clamp_count
              << " weights=" << report.weight_bytes
              << " scratch_upper=" << report.scratch_upper_bound_bytes
              << std::endl;

    if (!exercise_load) return;
    if (expect_mmap) setenv("HELIOS_VISION_MMAP", "1", 1);
    else unsetenv("HELIOS_VISION_MMAP");
    HnfLoader loader;
    require(loader.open(path), "real vision HNF must open: " + loader.last_error());
    Engine engine;
    size_t free_before = 0;
    size_t total_vram = 0;
    require(cudaMemGetInfo(&free_before, &total_vram) == cudaSuccess,
            "cannot sample VRAM before vision load");
    require(loader.load_block(BLOCK_VISION, engine),
            "real vision block must load: " + loader.last_error());
    size_t free_loaded = 0;
    require(cudaMemGetInfo(&free_loaded, &total_vram) == cudaSuccess,
            "cannot sample VRAM after vision load");
    require(loader.is_block_loaded(BLOCK_VISION), "vision load state");
    require(loader.block_tensor_count(BLOCK_VISION) == 659,
            "loaded vision tensor count");
    require(loader.block_vram_usage(BLOCK_VISION) ==
                (expect_mmap ? 0 : report.weight_bytes),
            "loaded vision VRAM byte count");
    require(loader.block_state(BLOCK_VISION).scratch_budget_bytes ==
                report.scratch_upper_bound_bytes,
            "pre-allocation scratch budget must be retained");
    require(engine.tensors().count() == 659, "registry vision tensor count");
    if (expect_mmap) {
        size_t mapped = 0;
        for (const TensorEntry* entry : loader.tensors_for_block(BLOCK_VISION)) {
            const TensorInfo* tensor = engine.tensors().get(entry->name);
            require(tensor != nullptr, "mapped vision tensor missing");
            if (tensor->file_mapped) ++mapped;
        }
        require(mapped == report.tensor_count,
                "every visual tensor must remain file-backed");
        require(engine.tensors().owned_bytes() >= report.weight_bytes &&
                engine.tensors().owned_bytes() <= report.weight_bytes + 4096,
                "registry must own one page-aligned visual mapping");
    } else {
        require(engine.tensors().owned_bytes() == report.weight_bytes,
                "registry must account for the contiguous vision allocation");
    }
    require(loader.unload_block(BLOCK_VISION, engine), "vision block unload");
    size_t free_unloaded = 0;
    require(cudaMemGetInfo(&free_unloaded, &total_vram) == cudaSuccess,
            "cannot sample VRAM after vision unload");
    require(!loader.is_block_loaded(BLOCK_VISION), "vision unload state");
    require(engine.tensors().count() == 0, "vision unload must empty registry");
    const size_t observed_load = free_before > free_loaded
        ? free_before - free_loaded : 0;
    const size_t unrecovered = free_before > free_unloaded
        ? free_before - free_unloaded : 0;
    std::cout << "PASS: real Gemma 4 vision "
              << (expect_mmap ? "mmap" : "VRAM") << " load/unload"
              << std::endl;
    std::cout << "  observed_vram=" << observed_load
              << " unrecovered_after_unload=" << unrecovered << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    test_synthetic_contract();
    if (argc >= 2) {
        const bool exercise_load = argc == 3 &&
            (std::string(argv[2]) == "--load" ||
             std::string(argv[2]) == "--mmap");
        const bool expect_mmap = argc == 3 &&
            std::string(argv[2]) == "--mmap";
        inspect_real_hnf(argv[1], exercise_load, expect_mmap);
    }
    if (argc > 3 || (argc == 3 && std::string(argv[2]) != "--load" &&
                     std::string(argv[2]) != "--mmap")) {
        std::cerr << "Usage: test_gemma4_vision_metadata [vision.hnf]"
                     " [--load|--mmap]"
                  << std::endl;
        return 2;
    }
    return 0;
}
