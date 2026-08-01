#pragma once

#include "hnf_loader.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace helios {

struct Gemma4VisionValidationReport {
    std::vector<std::string> errors;
    size_t tensor_count = 0;
    size_t clamp_count = 0;
    uint64_t weight_bytes = 0;
    uint64_t scratch_upper_bound_bytes = 0;
    uint32_t max_patches = 0;
    uint32_t max_soft_tokens = 0;

    bool ok() const { return errors.empty(); }
};

// Validates the complete Gemma 4 vision contract without allocating VRAM.
// The scratch figure is a conservative V4 planning bound: one FP32 attention
// matrix (softmax in place) plus patch/hidden/QKV/MLP/projected FP16 buffers.
Gemma4VisionValidationReport validate_gemma4_vision_tensors(
    const HnfLoader& loader);

} // namespace helios
