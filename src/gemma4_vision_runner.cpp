#include "gemma4_vision_runner.hpp"

#include "kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <sstream>

namespace helios {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

bool cuda_ok(cudaError_t status, const char* operation, std::string* error) {
    if (status == cudaSuccess) return true;
    std::ostringstream message;
    message << operation << ": " << cudaGetErrorString(status);
    return fail(error, message.str());
}

bool shape_is(const TensorInfo* tensor,
              std::initializer_list<uint32_t> expected) {
    return tensor && tensor->shape == std::vector<uint32_t>(expected);
}

} // namespace

Gemma4VisionRunner::Gemma4VisionRunner(Engine& engine,
                                       HnfLoader& loader)
    : engine_(engine), loader_(loader) {
    if (!loader_.has_gemma4_vision_config()) return;
    const Gemma4VisionConfig& vision = loader_.gemma4_vision_config();
    const ModelConfig& base = loader_.config_for_block(BLOCK_VISION);
    max_patches_ = vision.max_patches();
    hidden_size_ = base.hidden_size();
    position_embedding_size_ = vision.position_embedding_size;
    num_layers_ = base.num_hidden_layers();
    heads_ = base.num_attention_heads();
    head_dim_ = vision.head_dim;
    intermediate_size_ = base.intermediate_size();
    projection_size_ = vision.projection_dim;

    const uint64_t channels = base.get<uint32_t>("num_channels", 0);
    const uint64_t patch = vision.patch_size;
    const uint64_t width = channels * patch * patch;
    if (width <= std::numeric_limits<uint32_t>::max()) {
        patch_width_ = static_cast<uint32_t>(width);
    }
}

Gemma4VisionRunner::~Gemma4VisionRunner() {
    release();
    (void)loader_.release_mapped_staging(BLOCK_VISION, engine_);
}

bool Gemma4VisionRunner::validate_contract(std::string* error) const {
    if (!loader_.has_gemma4_vision_config()) {
        return fail(error, "GM4V configuration is missing");
    }
    if (!loader_.is_block_loaded(BLOCK_VISION)) {
        return fail(error, "Gemma 4 vision block is not loaded");
    }
    if (max_patches_ == 0 || patch_width_ == 0 || hidden_size_ == 0 ||
        position_embedding_size_ == 0) {
        return fail(error, "Gemma 4 vision patch geometry is invalid");
    }

    const TensorInfo* projection = engine_.tensors().get(
        "vision.patch_embed.input_proj.weight");
    const TensorInfo* positions = engine_.tensors().get(
        "vision.patch_embed.position_embedding.weight");
    if (!shape_is(projection, {hidden_size_, patch_width_}) ||
        projection->dtype != dtype::FP16()) {
        return fail(error, "vision patch projection is absent or not exact FP16");
    }
    if (!shape_is(positions,
                  {2, position_embedding_size_, hidden_size_}) ||
        positions->dtype != dtype::FP16()) {
        return fail(error, "vision XY position table is absent or not exact FP16");
    }
    return true;
}

bool Gemma4VisionRunner::reserve(std::string* error) {
    if (d_hidden_) return true;
    const size_t patch_elements = size_t(max_patches_) * patch_width_;
    const size_t hidden_elements = size_t(max_patches_) * hidden_size_;

    if (!cuda_ok(cudaMalloc(&d_pixel_values_, patch_elements * sizeof(float)),
                 "allocate visual FP32 upload", error) ||
        !cuda_ok(cudaMalloc(&d_patch_input_, patch_elements * sizeof(half)),
                 "allocate visual FP16 patches", error) ||
        !cuda_ok(cudaMalloc(&d_position_ids_,
                            size_t(max_patches_) * 2 * sizeof(int32_t)),
                 "allocate visual positions", error) ||
        !cuda_ok(cudaMalloc(&d_hidden_, hidden_elements * sizeof(half)),
                 "allocate visual patch embeddings", error)) {
        release();
        return false;
    }
    return true;
}

void Gemma4VisionRunner::release() {
    if (d_projected_) cudaFree(d_projected_);
    if (d_pooled_fp16_) cudaFree(d_pooled_fp16_);
    if (d_pooled_fp32_) cudaFree(d_pooled_fp32_);
    if (d_up_) cudaFree(d_up_);
    if (d_gate_) cudaFree(d_gate_);
    if (d_attention_scores_) cudaFree(d_attention_scores_);
    if (d_attention_head_) cudaFree(d_attention_head_);
    if (d_v_head_) cudaFree(d_v_head_);
    if (d_k_head_) cudaFree(d_k_head_);
    if (d_q_head_) cudaFree(d_q_head_);
    if (d_projection_) cudaFree(d_projection_);
    if (d_linear_input_) cudaFree(d_linear_input_);
    if (d_norm_) cudaFree(d_norm_);
    if (d_hidden_) cudaFree(d_hidden_);
    if (d_position_ids_) cudaFree(d_position_ids_);
    if (d_patch_input_) cudaFree(d_patch_input_);
    if (d_pixel_values_) cudaFree(d_pixel_values_);
    d_hidden_ = nullptr;
    d_position_ids_ = nullptr;
    d_patch_input_ = nullptr;
    d_pixel_values_ = nullptr;
    d_norm_ = nullptr;
    d_linear_input_ = nullptr;
    d_projection_ = nullptr;
    d_q_head_ = nullptr;
    d_k_head_ = nullptr;
    d_v_head_ = nullptr;
    d_attention_head_ = nullptr;
    d_attention_scores_ = nullptr;
    d_gate_ = nullptr;
    d_up_ = nullptr;
    d_pooled_fp32_ = nullptr;
    d_pooled_fp16_ = nullptr;
    d_projected_ = nullptr;
    patch_count_ = 0;
    completed_layers_ = 0;
    real_patches_ = 0;
    patch_columns_ = 0;
    patch_rows_ = 0;
    soft_token_count_ = 0;
    pooler_complete_ = false;
    projector_complete_ = false;
}

bool Gemma4VisionRunner::run_patch_embedder(
    const Gemma4VisionPreprocessResult& input,
    std::string* error) {
    if (error) error->clear();
    if (!validate_contract(error)) return false;
    if (input.max_patches != max_patches_ ||
        input.patch_values != patch_width_ ||
        input.real_patches > input.max_patches ||
        input.pixel_values.size() != size_t(max_patches_) * patch_width_ ||
        input.position_ids.size() != size_t(max_patches_) * 2) {
        return fail(error, "preprocessed image does not match GM4V patch geometry");
    }
    uint32_t valid_positions = 0;
    int32_t maximum_x = -1;
    int32_t maximum_y = -1;
    for (uint32_t patch = 0; patch < max_patches_; ++patch) {
        const int32_t x = input.position_ids[size_t(patch) * 2];
        const int32_t y = input.position_ids[size_t(patch) * 2 + 1];
        const bool padding = x < 0 || y < 0;
        if (padding) {
            if (x != -1 || y != -1) {
                return fail(error, "vision padding coordinates must be (-1,-1)");
            }
        } else if (uint32_t(x) >= position_embedding_size_ ||
                   uint32_t(y) >= position_embedding_size_) {
            return fail(error, "vision position coordinate exceeds learned table");
        } else {
            ++valid_positions;
            maximum_x = std::max(maximum_x, x);
            maximum_y = std::max(maximum_y, y);
        }
    }
    const uint32_t columns = static_cast<uint32_t>(maximum_x + 1);
    const uint32_t rows = static_cast<uint32_t>(maximum_y + 1);
    const uint32_t pooling = loader_.gemma4_vision_config().pooling_kernel_size;
    if (valid_positions != input.real_patches || columns == 0 || rows == 0 ||
        uint64_t(columns) * rows != valid_positions || pooling == 0 ||
        columns % pooling != 0 || rows % pooling != 0) {
        return fail(error, "vision positions do not form a poolable rectangle");
    }
    for (uint32_t patch = 0; patch < valid_positions; ++patch) {
        if (input.position_ids[size_t(patch) * 2] != int32_t(patch % columns) ||
            input.position_ids[size_t(patch) * 2 + 1] != int32_t(patch / columns)) {
            return fail(error, "vision positions are not canonical row-major XY");
        }
    }
    const uint32_t soft_tokens =
        (columns / pooling) * (rows / pooling);
    if (input.soft_tokens != 0 && input.soft_tokens != soft_tokens) {
        return fail(error, "preprocessor soft-token count differs from positions");
    }
    if (!reserve(error)) return false;

    const cudaStream_t stream = engine_.config().stream;
    if (!loader_.stage_mapped_prefix(
            BLOCK_VISION, engine_, "vision.patch_embed.", stream)) {
        return fail(error, loader_.last_error());
    }
    struct RestoreMappedPatch {
        HnfLoader& loader;
        Engine& engine;
        ~RestoreMappedPatch() {
            (void)loader.unstage_mapped_prefix(BLOCK_VISION, engine);
        }
    } restore_mapped{loader_, engine_};
    const size_t patch_elements = size_t(max_patches_) * patch_width_;
    if (!cuda_ok(cudaMemcpyAsync(d_pixel_values_, input.pixel_values.data(),
                                 patch_elements * sizeof(float),
                                 cudaMemcpyHostToDevice, stream),
                 "upload visual patches", error) ||
        !cuda_ok(cudaMemcpyAsync(d_position_ids_, input.position_ids.data(),
                                 size_t(max_patches_) * 2 * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream),
                 "upload visual positions", error)) {
        return false;
    }

    kernels::launch_gemma4_patch_input_fp16(
        d_pixel_values_, d_patch_input_, patch_elements, stream);
    const TensorInfo& projection = engine_.tensors().at(
        "vision.patch_embed.input_proj.weight");
    kernels::launch_matmul_fp16(
        d_patch_input_, static_cast<const half*>(projection.ptr), d_hidden_,
        static_cast<int>(max_patches_), static_cast<int>(patch_width_),
        static_cast<int>(hidden_size_), stream);
    const TensorInfo& positions = engine_.tensors().at(
        "vision.patch_embed.position_embedding.weight");
    kernels::launch_gemma4_add_xy_position_fp16(
        d_hidden_, static_cast<const half*>(positions.ptr), d_position_ids_,
        max_patches_, hidden_size_, position_embedding_size_, stream);

    if (!cuda_ok(cudaGetLastError(), "launch Gemma 4 patch embedder", error)) {
        return false;
    }
    patch_count_ = max_patches_;
    completed_layers_ = 0;
    real_patches_ = valid_positions;
    patch_columns_ = columns;
    patch_rows_ = rows;
    soft_token_count_ = soft_tokens;
    pooler_complete_ = false;
    projector_complete_ = false;
    return true;
}

const half* Gemma4VisionRunner::fp16_tensor(
    const std::string& name,
    std::string* error) const {
    const TensorInfo* tensor = engine_.tensors().get(name);
    if (!tensor || tensor->dtype != dtype::FP16() || !tensor->ptr) {
        fail(error, "missing exact-FP16 vision tensor: " + name);
        return nullptr;
    }
    return static_cast<const half*>(tensor->ptr);
}

bool Gemma4VisionRunner::reserve_layer_scratch(std::string* error) {
    if (d_norm_) return true;
    if (heads_ == 0 || head_dim_ == 0 || heads_ * head_dim_ != hidden_size_ ||
        intermediate_size_ == 0) {
        return fail(error, "Gemma 4 visual layer geometry is invalid");
    }
    const size_t hidden_elements = size_t(max_patches_) * hidden_size_;
    const size_t intermediate_elements =
        size_t(max_patches_) * intermediate_size_;
    const size_t linear_elements =
        std::max(hidden_elements, intermediate_elements);
    const size_t score_elements =
        size_t(heads_) * max_patches_ * max_patches_;

    auto allocate_half = [&](half** pointer, size_t elements,
                             const char* label) {
        return cuda_ok(cudaMalloc(pointer, elements * sizeof(half)),
                       label, error);
    };
    if (!allocate_half(&d_norm_, hidden_elements, "allocate visual norm") ||
        !allocate_half(&d_linear_input_, linear_elements,
                       "allocate visual clipped input") ||
        !allocate_half(&d_projection_, hidden_elements,
                       "allocate visual projection") ||
        !allocate_half(&d_q_head_, hidden_elements, "allocate visual Q") ||
        !allocate_half(&d_k_head_, hidden_elements, "allocate visual K") ||
        !allocate_half(&d_v_head_, hidden_elements, "allocate visual V") ||
        !allocate_half(&d_attention_head_, hidden_elements,
                       "allocate visual attention output") ||
        !allocate_half(&d_attention_scores_, score_elements,
                       "allocate visual attention scores") ||
        !allocate_half(&d_gate_, intermediate_elements,
                       "allocate visual MLP gate") ||
        !allocate_half(&d_up_, intermediate_elements,
                       "allocate visual MLP up")) {
        release();
        return false;
    }
    return true;
}

bool Gemma4VisionRunner::clipped_linear(
    const std::string& prefix,
    const half* input,
    half* output,
    uint32_t input_width,
    uint32_t output_width,
    std::string* error) {
    const half* weight = fp16_tensor(prefix + "weight", error);
    const half* input_min = fp16_tensor(prefix + "input_min", error);
    const half* input_max = fp16_tensor(prefix + "input_max", error);
    const half* output_min = fp16_tensor(prefix + "output_min", error);
    const half* output_max = fp16_tensor(prefix + "output_max", error);
    if (!weight || !input_min || !input_max || !output_min || !output_max) {
        return false;
    }
    const TensorInfo* weight_info = engine_.tensors().get(prefix + "weight");
    if (!shape_is(weight_info, {output_width, input_width})) {
        return fail(error, "visual linear shape mismatch: " + prefix + "weight");
    }
    const cudaStream_t stream = engine_.config().stream;
    kernels::launch_clamp_tensor_bounds_fp16(
        input, input_min, input_max, d_linear_input_,
        size_t(max_patches_) * input_width, stream);
    kernels::launch_matmul_fp16(
        d_linear_input_, weight, output,
        static_cast<int>(max_patches_), static_cast<int>(input_width),
        static_cast<int>(output_width), stream);
    kernels::launch_clamp_tensor_bounds_fp16(
        output, output_min, output_max, output,
        size_t(max_patches_) * output_width, stream);
    return cuda_ok(cudaGetLastError(), "launch visual clipped linear", error);
}

bool Gemma4VisionRunner::run_encoder_layer(
    uint32_t layer_index,
    std::string* error) {
    if (error) error->clear();
    if (!d_hidden_ || patch_count_ == 0) {
        return fail(error, "run the Gemma 4 patch embedder before its encoder");
    }
    if (layer_index != completed_layers_ || layer_index >= num_layers_) {
        return fail(error, "Gemma 4 visual layers must execute once and in order");
    }
    if (!validate_contract(error) || !reserve_layer_scratch(error)) return false;

    const std::string prefix =
        "vision.layer" + std::to_string(layer_index) + '.';
    const float eps = loader_.gemma4_vision_config().rms_norm_eps;
    const float theta = loader_.gemma4_vision_config().rope_theta;
    const float attention_scale =
        loader_.gemma4_vision_config().attention_scale;
    const cudaStream_t stream = engine_.config().stream;
    const size_t hidden_elements = size_t(max_patches_) * hidden_size_;

    if (!loader_.stage_mapped_prefix(BLOCK_VISION, engine_, prefix, stream)) {
        return fail(error, loader_.last_error());
    }
    struct RestoreMappedLayer {
        HnfLoader& loader;
        Engine& engine;
        ~RestoreMappedLayer() {
            (void)loader.unstage_mapped_prefix(BLOCK_VISION, engine);
        }
    } restore_mapped{loader_, engine_};

    const half* attn_in = fp16_tensor(prefix + "ln_attn_in.weight", error);
    const half* q_norm = fp16_tensor(prefix + "attn.q_norm.weight", error);
    const half* k_norm = fp16_tensor(prefix + "attn.k_norm.weight", error);
    if (!attn_in || !q_norm || !k_norm) return false;
    kernels::launch_rmsnorm_fp16(
        d_hidden_, attn_in, d_norm_, max_patches_, hidden_size_, eps, stream);

    struct Projection {
        const char* name;
        const half* norm;
        half* head_major;
        bool rope;
    } projections[] = {
        {"q_proj.", q_norm, d_q_head_, true},
        {"k_proj.", k_norm, d_k_head_, true},
        {"v_proj.", nullptr, d_v_head_, false},
    };
    for (const Projection& projection : projections) {
        if (!clipped_linear(prefix + "attn." + projection.name,
                            d_norm_, d_projection_, hidden_size_, hidden_size_,
                            error)) {
            return false;
        }
        kernels::launch_gemma4_vision_norm_rope_transpose_fp16(
            d_projection_, projection.norm, d_position_ids_,
            projection.head_major, max_patches_, heads_, head_dim_, eps,
            theta, projection.rope, stream);
    }

    if (!kernels::launch_gemma4_vision_attention_fp16(
            d_q_head_, d_k_head_, d_v_head_, d_position_ids_,
            d_attention_scores_, d_attention_head_, max_patches_, heads_,
            head_dim_, attention_scale, stream)) {
        return fail(error, "cuBLAS rejected Gemma 4 visual attention");
    }
    kernels::launch_gemma4_vision_head_to_token_fp16(
        d_attention_head_, d_projection_, max_patches_, heads_, head_dim_,
        stream);
    if (!clipped_linear(prefix + "attn.o_proj.", d_projection_, d_norm_,
                        hidden_size_, hidden_size_, error)) {
        return false;
    }
    const half* attn_post = fp16_tensor(prefix + "ln_attn_post.weight", error);
    if (!attn_post) return false;
    kernels::launch_rmsnorm_fp16(
        d_norm_, attn_post, d_projection_, max_patches_, hidden_size_, eps,
        stream);
    kernels::launch_add_fp16(
        d_hidden_, d_projection_, d_hidden_, hidden_elements, stream);

    const half* mlp_in = fp16_tensor(prefix + "ln_mlp_in.weight", error);
    if (!mlp_in) return false;
    kernels::launch_rmsnorm_fp16(
        d_hidden_, mlp_in, d_norm_, max_patches_, hidden_size_, eps, stream);
    if (!clipped_linear(prefix + "mlp.gate.", d_norm_, d_gate_,
                        hidden_size_, intermediate_size_, error) ||
        !clipped_linear(prefix + "mlp.up.", d_norm_, d_up_,
                        hidden_size_, intermediate_size_, error)) {
        return false;
    }
    kernels::launch_gelu_mul_fp16(
        d_gate_, d_up_, d_gate_,
        size_t(max_patches_) * intermediate_size_, stream);
    if (!clipped_linear(prefix + "mlp.down.", d_gate_, d_projection_,
                        intermediate_size_, hidden_size_, error)) {
        return false;
    }
    const half* mlp_post = fp16_tensor(prefix + "ln_mlp_post.weight", error);
    if (!mlp_post) return false;
    kernels::launch_rmsnorm_fp16(
        d_projection_, mlp_post, d_norm_, max_patches_, hidden_size_, eps,
        stream);
    kernels::launch_add_fp16(
        d_hidden_, d_norm_, d_hidden_, hidden_elements, stream);

    if (!cuda_ok(cudaGetLastError(), "launch Gemma 4 visual layer", error)) {
        return false;
    }
    ++completed_layers_;
    return true;
}

bool Gemma4VisionRunner::run_pooler(std::string* error) {
    if (error) error->clear();
    if (completed_layers_ != num_layers_) {
        return fail(error, "pooling requires all Gemma 4 visual layers");
    }
    if (pooler_complete_) {
        return fail(error, "Gemma 4 visual pooler already ran");
    }
    if (loader_.gemma4_vision_config().pooling_kernel_size != 3 ||
        soft_token_count_ == 0) {
        return fail(error, "unsupported Gemma 4 visual pooling geometry");
    }
    const size_t pooled_capacity =
        size_t(loader_.gemma4_vision_config().max_soft_tokens) * hidden_size_;
    if (!d_pooled_fp32_ &&
        !cuda_ok(cudaMalloc(&d_pooled_fp32_, pooled_capacity * sizeof(float)),
                 "allocate visual FP32 pool", error)) {
        return false;
    }
    if (!d_pooled_fp16_ &&
        !cuda_ok(cudaMalloc(&d_pooled_fp16_, pooled_capacity * sizeof(half)),
                 "allocate visual FP16 pool", error)) {
        return false;
    }
    kernels::launch_gemma4_vision_pool3x3_fp32(
        d_hidden_, d_pooled_fp32_, patch_columns_, patch_rows_, hidden_size_,
        engine_.config().stream);
    if (!cuda_ok(cudaGetLastError(), "launch Gemma 4 visual pooler", error)) {
        return false;
    }
    pooler_complete_ = true;
    return true;
}

bool Gemma4VisionRunner::run_projector(std::string* error) {
    if (error) error->clear();
    if (!pooler_complete_) {
        return fail(error, "run the Gemma 4 visual pooler before projection");
    }
    if (projector_complete_) {
        return fail(error, "Gemma 4 visual projector already ran");
    }
    const cudaStream_t stream = engine_.config().stream;
    if (!loader_.stage_mapped_prefix(
            BLOCK_VISION, engine_, "vision.projector.", stream)) {
        return fail(error, loader_.last_error());
    }
    struct RestoreMappedProjector {
        HnfLoader& loader;
        Engine& engine;
        ~RestoreMappedProjector() {
            (void)loader.unstage_mapped_prefix(BLOCK_VISION, engine);
        }
    } restore_mapped{loader_, engine_};
    const half* weight = fp16_tensor("vision.projector.weight", error);
    const TensorInfo* info = engine_.tensors().get("vision.projector.weight");
    if (!weight || !shape_is(info, {projection_size_, hidden_size_})) {
        return fail(error, "Gemma 4 visual projector shape mismatch");
    }
    const size_t pooled_elements = size_t(soft_token_count_) * hidden_size_;
    const size_t projected_elements =
        size_t(soft_token_count_) * projection_size_;
    if (!d_projected_) {
        const size_t projected_capacity =
            size_t(loader_.gemma4_vision_config().max_soft_tokens) *
            projection_size_;
        if (!cuda_ok(cudaMalloc(&d_projected_,
                                projected_capacity * sizeof(half)),
                     "allocate projected visual embeddings", error)) {
            return false;
        }
    }
    // std_bias=0 and std_scale=1 are fixed non-persistent upstream buffers for
    // this checkpoint. Casting after the FP32 pool therefore is sufficient.
    kernels::launch_fp32_to_fp16(
        d_pooled_fp32_, d_pooled_fp16_, pooled_elements, stream);
    kernels::launch_rmsnorm_no_weight_fp16(
        d_pooled_fp16_, d_pooled_fp16_, soft_token_count_, hidden_size_,
        loader_.gemma4_vision_config().rms_norm_eps, stream);
    kernels::launch_matmul_fp16(
        d_pooled_fp16_, weight, d_projected_, soft_token_count_, hidden_size_,
        projection_size_, stream);
    if (!cuda_ok(cudaGetLastError(), "launch Gemma 4 visual projector", error)) {
        return false;
    }
    projector_complete_ = true;
    return true;
}

bool Gemma4VisionRunner::copy_patch_embeddings(
    std::vector<float>& output,
    std::string* error) const {
    if (error) error->clear();
    if (completed_layers_ != 0) {
        return fail(error, "patch boundary was replaced by encoder output");
    }
    return copy_hidden_states(output, error);
}

bool Gemma4VisionRunner::copy_hidden_states(
    std::vector<float>& output,
    std::string* error) const {
    if (error) error->clear();
    if (!d_hidden_ || patch_count_ == 0) {
        return fail(error, "Gemma 4 patch embedder has not run");
    }
    const size_t elements = size_t(patch_count_) * hidden_size_;
    std::vector<half> fp16_output(elements);
    const cudaStream_t stream = engine_.config().stream;
    if (!cuda_ok(cudaMemcpyAsync(fp16_output.data(), d_hidden_,
                                 elements * sizeof(half),
                                 cudaMemcpyDeviceToHost, stream),
                 "download visual patch embeddings", error) ||
        !cuda_ok(cudaStreamSynchronize(stream),
                 "synchronize visual patch embeddings", error)) {
        return false;
    }
    output.resize(elements);
    for (size_t i = 0; i < elements; ++i) {
        output[i] = __half2float(fp16_output[i]);
    }
    return true;
}

bool Gemma4VisionRunner::copy_pooled_states(
    std::vector<float>& output,
    std::string* error) const {
    if (error) error->clear();
    if (!pooler_complete_ || !d_pooled_fp32_) {
        return fail(error, "Gemma 4 visual pooler has not run");
    }
    const size_t elements = size_t(soft_token_count_) * hidden_size_;
    output.resize(elements);
    const cudaStream_t stream = engine_.config().stream;
    return cuda_ok(cudaMemcpyAsync(output.data(), d_pooled_fp32_,
                                   elements * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream),
                   "download visual pool", error) &&
           cuda_ok(cudaStreamSynchronize(stream),
                   "synchronize visual pool", error);
}

bool Gemma4VisionRunner::copy_projected_states(
    std::vector<float>& output,
    std::string* error) const {
    if (error) error->clear();
    if (!projector_complete_ || !d_projected_) {
        return fail(error, "Gemma 4 visual projector has not run");
    }
    const size_t elements = size_t(soft_token_count_) * projection_size_;
    std::vector<half> fp16_output(elements);
    const cudaStream_t stream = engine_.config().stream;
    if (!cuda_ok(cudaMemcpyAsync(fp16_output.data(), d_projected_,
                                 elements * sizeof(half),
                                 cudaMemcpyDeviceToHost, stream),
                 "download projected visual embeddings", error) ||
        !cuda_ok(cudaStreamSynchronize(stream),
                 "synchronize projected visual embeddings", error)) {
        return false;
    }
    output.resize(elements);
    for (size_t i = 0; i < elements; ++i) {
        output[i] = __half2float(fp16_output[i]);
    }
    return true;
}

} // namespace helios
