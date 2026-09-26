#pragma once

// Gemma 4 12B «unified»: image and audio enter the decoder without a tower.
//
//   image: merged 48x48 patch -> LN -> Dense+bias -> LN -> + XY position -> LN
//          -> RMSNorm (no scale) -> Linear         (Gemma4UnifiedVisionEmbedder)
//   audio: 640-sample frame -> RMSNorm (no scale) -> Linear
//                                                  (Gemma4UnifiedMultimodalEmbedder)
//
// The weights live in a modality-only HNF (vision block 0x1, audio block 0x2)
// next to the text HNF. Both blocks must already be loaded into the engine.

#include "engine.hpp"
#include "gemma4_vision_preprocess.hpp"
#include "hnf_loader.hpp"

#include <cuda_fp16.h>
#include <cstdint>
#include <string>
#include <vector>

namespace helios {

struct Gemma4UnifiedVisionSpec {
    uint32_t model_patch_size = 0;   // 48: 3x3 teacher patches of 16 px
    uint32_t patch_values = 0;       // 48 * 48 * 3
    uint32_t posemb_size = 0;        // rows of the factorized XY table
    uint32_t max_soft_tokens = 0;    // 280
    uint32_t hidden = 0;             // decoder hidden size
    float rms_norm_eps = 1e-6f;
};

struct Gemma4UnifiedAudioSpec {
    uint32_t samples_per_token = 0;  // 640 = 40 ms at 16 kHz
    uint32_t sampling_rate = 0;
    uint32_t max_tokens = 0;         // 750 = 30 s
    uint32_t hidden = 0;
    float rms_norm_eps = 1e-6f;
};

// Read and validate the JSON hints of a modality-only HNF. False, with a
// reason, when the block is absent or is not the encoder-free variant.
bool gemma4_unified_vision_spec(const HnfLoader& loader,
                                Gemma4UnifiedVisionSpec& spec,
                                std::string* error = nullptr);
bool gemma4_unified_audio_spec(const HnfLoader& loader,
                               Gemma4UnifiedAudioSpec& spec,
                               std::string* error = nullptr);

// The unified processor's 3x3 merge of 16 px patches is exactly a raster split
// into 48 px blocks, HWC inside each block, with the block's (x, y). The
// resize budget is also the same (sides multiple of 48, at most 280 blocks),
// so the E4B preprocessing runs unchanged with these parameters.
Gemma4VisionPreprocessConfig gemma4_unified_preprocess_config(
    const Gemma4UnifiedVisionSpec& spec);

class Gemma4UnifiedEmbedder {
public:
    enum class Stage { PatchLn2, PosNorm };

    Gemma4UnifiedEmbedder(Engine& engine, const HnfLoader& loader);
    ~Gemma4UnifiedEmbedder();
    Gemma4UnifiedEmbedder(const Gemma4UnifiedEmbedder&) = delete;
    Gemma4UnifiedEmbedder& operator=(const Gemma4UnifiedEmbedder&) = delete;

    bool has_vision() const { return vision_ok_; }
    bool has_audio() const { return audio_ok_; }
    const Gemma4UnifiedVisionSpec& vision() const { return vision_; }
    const Gemma4UnifiedAudioSpec& audio() const { return audio_; }

    // Writes [real_patches, hidden] FP16 into device memory owned by the
    // caller. Padding rows of the preprocessing result are not embedded.
    bool embed_image(const Gemma4VisionPreprocessResult& input, half* output,
                     std::string* error = nullptr);

    // frames: [tokens, samples_per_token] FP32 on the host, already framed
    // and zero-padded. Writes [tokens, hidden] FP16 into device memory.
    bool embed_audio(const float* frames, uint32_t tokens, half* output,
                     std::string* error = nullptr);

    // Intermediate image stages of the last embed_image, for verification.
    bool copy_stage(Stage stage, std::vector<float>& values,
                    std::string* error = nullptr) const;

private:
    bool reserve(size_t input_values, size_t rows, std::string* error);
    const half* weight(const char* name, std::string* error) const;
    bool scaled_bias(const half* bias, std::string* error);

    Engine& engine_;
    Gemma4UnifiedVisionSpec vision_{};
    Gemma4UnifiedAudioSpec audio_{};
    bool vision_ok_ = false;
    bool audio_ok_ = false;

    uint32_t rows_ = 0;
    size_t input_capacity_ = 0;
    size_t row_capacity_ = 0;
    float* d_input_f32_ = nullptr;
    half* d_input_ = nullptr;
    half* d_input_norm_ = nullptr;
    int32_t* d_positions_ = nullptr;
    half* d_patch_ = nullptr;   // after LN2
    half* d_pos_ = nullptr;     // after position LN
    half* d_scratch_ = nullptr;
    half* d_dense_bias_ = nullptr;  // patch Dense bias, pre-scaled
};

} // namespace helios
