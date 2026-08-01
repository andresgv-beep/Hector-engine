#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace helios {

struct Gemma4RgbView {
    const uint8_t* data = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    size_t row_stride_bytes = 0; // 0 means tightly packed width * 3
};

struct Gemma4VisionPreprocessConfig {
    uint32_t patch_size = 16;
    uint32_t pooling_kernel_size = 3;
    uint32_t max_soft_tokens = 280;
    float rescale_factor = 1.0f / 255.0f;
};

struct Gemma4VisionPreprocessResult {
    uint32_t source_width = 0;
    uint32_t source_height = 0;
    uint32_t resized_width = 0;
    uint32_t resized_height = 0;
    uint32_t patch_columns = 0;
    uint32_t patch_rows = 0;
    uint32_t real_patches = 0;
    uint32_t max_patches = 0;
    uint32_t soft_tokens = 0;
    uint32_t patch_values = 0;

    // [max_patches, patch_size * patch_size * 3], HWC inside every patch.
    std::vector<float> pixel_values;
    // [max_patches, 2], ordered (x, y); padding is (-1, -1).
    std::vector<int32_t> position_ids;
};

// Exact geometry used by the pinned Gemma 4 processor. The result always fits
// max_soft_tokens * pooling_kernel_size^2 patches and each side is a multiple
// of patch_size * pooling_kernel_size.
bool gemma4_vision_target_size(
    uint32_t source_width,
    uint32_t source_height,
    const Gemma4VisionPreprocessConfig& config,
    uint32_t& target_width,
    uint32_t& target_height,
    std::string* error = nullptr);

// CPU preprocessing from an already decoded RGB8 image. PNG/JPEG decoding is
// deliberately outside this API. Resize is PyTorch/Pillow-compatible Keys
// bicubic with antialiasing and uint8 rounding before FP32 rescaling.
bool gemma4_vision_preprocess_rgb(
    const Gemma4RgbView& image,
    const Gemma4VisionPreprocessConfig& config,
    Gemma4VisionPreprocessResult& result,
    std::string* error = nullptr);

} // namespace helios
