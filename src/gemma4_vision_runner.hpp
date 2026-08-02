#pragma once

#include "gemma4_vision_preprocess.hpp"
#include "hnf_loader.hpp"
#include "mapped_weight_pipeline.hpp"

#include <cuda_fp16.h>

#include <cstdint>
#include <string>
#include <vector>

namespace helios {

// Independent execution path for the Gemma 4 visual tower. It deliberately
// does not append commands to the text GraphBuilder: vision is executed once
// per image, while the decoder graph is replayed once per generated token.
//
// V4 grows this class one verified numerical frontier at a time. The current
// frontier owns the reusable upload/FP16 buffers and implements the official
// patch projection plus learned 2-D position embedding.
class Gemma4VisionRunner {
public:
    Gemma4VisionRunner(Engine& engine, HnfLoader& loader);
    ~Gemma4VisionRunner();

    Gemma4VisionRunner(const Gemma4VisionRunner&) = delete;
    Gemma4VisionRunner& operator=(const Gemma4VisionRunner&) = delete;
    Gemma4VisionRunner(Gemma4VisionRunner&&) = delete;
    Gemma4VisionRunner& operator=(Gemma4VisionRunner&&) = delete;

    // Runs:
    //   input = fp16(2 * (pixel_values - 0.5))
    //   hidden = input @ input_proj.weight.T
    //   hidden += position[x] + position[y]
    // Padding coordinates (-1,-1) contribute no position embedding, matching
    // upstream. Work is enqueued on EngineConfig::stream.
    bool run_patch_embedder(const Gemma4VisionPreprocessResult& input,
                            std::string* error = nullptr);

    // Executes exactly the next visual encoder layer. Layers must be called in
    // order so every numerical boundary remains observable and testable.
    bool run_encoder_layer(uint32_t layer_index,
                           std::string* error = nullptr);

    // Final visual boundaries. Pooling remains FP32 for the sqrt(hidden)
    // scaling; projection returns FP16 embeddings in the text hidden width.
    bool run_pooler(std::string* error = nullptr);
    bool run_projector(std::string* error = nullptr);

    // Device output [patch_count, hidden_size], valid until the next run or
    // destruction. This is the input of visual layer 0 in the next V4 step.
    const half* patch_embeddings_device() const { return d_hidden_; }
    const half* hidden_states_device() const { return d_hidden_; }
    uint32_t patch_count() const { return patch_count_; }
    uint32_t hidden_size() const { return hidden_size_; }
    uint32_t num_layers() const { return num_layers_; }
    uint32_t completed_layers() const { return completed_layers_; }
    uint32_t soft_token_count() const { return soft_token_count_; }
    uint32_t projection_size() const { return projection_size_; }
    const half* projected_states_device() const {
        return projector_complete_ ? d_projected_ : nullptr;
    }

    // Numerical-test/debug boundary. Converts the FP16 device output to FP32
    // on the host and synchronizes the runner stream.
    bool copy_patch_embeddings(std::vector<float>& output,
                               std::string* error = nullptr) const;
    bool copy_hidden_states(std::vector<float>& output,
                            std::string* error = nullptr) const;
    bool copy_pooled_states(std::vector<float>& output,
                            std::string* error = nullptr) const;
    bool copy_projected_states(std::vector<float>& output,
                               std::string* error = nullptr) const;

private:
    bool validate_contract(std::string* error) const;
    bool reserve(std::string* error);
    bool reserve_layer_scratch(std::string* error);
    bool clipped_linear(const std::string& prefix,
                        const half* input,
                        half* output,
                        uint32_t input_width,
                        uint32_t output_width,
                        std::string* error);
    const half* fp16_tensor(const std::string& name,
                            std::string* error) const;
    void release();

    Engine& engine_;
    HnfLoader& loader_;
    MappedWeightPipeline weight_pipeline_;
    uint32_t max_patches_ = 0;
    uint32_t patch_width_ = 0;
    uint32_t hidden_size_ = 0;
    uint32_t position_embedding_size_ = 0;
    uint32_t patch_count_ = 0;
    uint32_t num_layers_ = 0;
    uint32_t heads_ = 0;
    uint32_t head_dim_ = 0;
    uint32_t intermediate_size_ = 0;
    uint32_t completed_layers_ = 0;
    uint32_t real_patches_ = 0;
    uint32_t patch_columns_ = 0;
    uint32_t patch_rows_ = 0;
    uint32_t soft_token_count_ = 0;
    uint32_t projection_size_ = 0;
    bool pooler_complete_ = false;
    bool projector_complete_ = false;

    float* d_pixel_values_ = nullptr;
    half* d_patch_input_ = nullptr;
    int32_t* d_position_ids_ = nullptr;
    half* d_hidden_ = nullptr;

    // Reusable layer scratch. Only the attention score matrix is large;
    // everything else is recycled between attention and MLP branches.
    half* d_norm_ = nullptr;
    half* d_linear_input_ = nullptr;
    half* d_projection_ = nullptr;
    half* d_q_head_ = nullptr;
    half* d_k_head_ = nullptr;
    half* d_v_head_ = nullptr;
    half* d_attention_head_ = nullptr;
    half* d_attention_scores_ = nullptr;
    half* d_gate_ = nullptr;
    half* d_up_ = nullptr;
    float* d_pooled_fp32_ = nullptr;
    half* d_pooled_fp16_ = nullptr;
    half* d_projected_ = nullptr;
};

} // namespace helios
