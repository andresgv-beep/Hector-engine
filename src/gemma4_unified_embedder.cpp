#include "gemma4_unified_embedder.hpp"

#include "../kernels/cublas_context.hpp"
#include "../kernels/kernels.hpp"

#include <cuda_runtime.h>
#include <sstream>

namespace helios {
namespace {

// torch.nn.LayerNorm keeps its default eps: the config's 1e-6 belongs to the
// scale-free RMSNorm before the projection, not to the three LayerNorms.
constexpr float kLayerNormEps = 1e-5f;

// The raw patch Dense reaches 1.3e5 on ordinary images (RMS ~1.7e4), past
// FP16's 65504. LN2 right after it is scale invariant (eps is negligible at
// that variance), so the GEMM writes (xW + b) / 64 and the result is the same.
constexpr float kDenseScale = 1.0f / 64.0f;

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

uint32_t positive(const ModelConfig& config, const char* key) {
    const int64_t value = config.get<int64_t>(key, 0);
    return value > 0 && value <= int64_t(UINT32_MAX) ? uint32_t(value) : 0u;
}

int32_t token(const ModelConfig& config, const char* key) {
    const int64_t value = config.get<int64_t>(key, -1);
    return value >= 0 && value <= INT32_MAX ? int32_t(value) : -1;
}

template <typename T>
void release(T*& pointer) {
    if (pointer) cudaFree(pointer);
    pointer = nullptr;
}

} // namespace

bool gemma4_unified_vision_spec(const HnfLoader& loader,
                                Gemma4UnifiedVisionSpec& spec,
                                std::string* error) {
    spec = {};
    if (!loader.has_config_for_block(BLOCK_VISION)) {
        return fail(error, "HNF has no vision hints");
    }
    const ModelConfig& config = loader.config_for_block(BLOCK_VISION);
    if (config.get<std::string>("arch", "") != "gemma4_unified_vision") {
        return fail(error, "vision block is not the Gemma 4 unified embedder");
    }
    spec.model_patch_size = positive(config, "model_patch_size");
    spec.patch_values = positive(config, "patch_dim");
    spec.posemb_size = positive(config, "mm_posemb_size");
    spec.max_soft_tokens = positive(config, "num_soft_tokens");
    spec.hidden = positive(config, "text_hidden_size");
    spec.rms_norm_eps = config.get<float>("rms_norm_eps", 1e-6f);
    spec.image_token_id = token(config, "image_token_id");
    spec.boi_token_id = token(config, "boi_token_id");
    spec.eoi_token_id = token(config, "eoi_token_id");
    spec.pad_token_id = token(config, "pad_token_id");
    if (spec.image_token_id < 0 || spec.boi_token_id < 0 || spec.eoi_token_id < 0 ||
        spec.pad_token_id < 0) {
        return fail(error, "Gemma 4 unified vision hints lack the image token ids");
    }
    if (!spec.model_patch_size || !spec.posemb_size || !spec.max_soft_tokens ||
        !spec.hidden || spec.patch_values !=
            spec.model_patch_size * spec.model_patch_size * 3 ||
        positive(config, "mm_embed_dim") != spec.hidden ||
        positive(config, "output_proj_dims") != spec.hidden ||
        !(spec.rms_norm_eps > 0.0f)) {
        return fail(error, "Gemma 4 unified vision hints are incomplete or inconsistent");
    }
    return true;
}

bool gemma4_unified_audio_spec(const HnfLoader& loader,
                               Gemma4UnifiedAudioSpec& spec,
                               std::string* error) {
    spec = {};
    if (!loader.has_config_for_block(BLOCK_AUDIO)) {
        return fail(error, "HNF has no audio hints");
    }
    const ModelConfig& config = loader.config_for_block(BLOCK_AUDIO);
    if (config.get<std::string>("arch", "") != "gemma4_unified_audio") {
        return fail(error, "audio block is not the Gemma 4 unified embedder");
    }
    spec.samples_per_token = positive(config, "samples_per_token");
    spec.sampling_rate = positive(config, "sampling_rate");
    spec.max_tokens = positive(config, "max_tokens");
    spec.hidden = positive(config, "text_hidden_size");
    spec.rms_norm_eps = config.get<float>("rms_norm_eps", 1e-6f);
    const uint32_t ms = positive(config, "ms_per_token");
    spec.audio_token_id = token(config, "audio_token_id");
    spec.boa_token_id = token(config, "boa_token_id");
    spec.eoa_token_id = token(config, "eoa_token_id");
    spec.pad_token_id = token(config, "pad_token_id");
    if (spec.audio_token_id < 0 || spec.boa_token_id < 0 || spec.eoa_token_id < 0 ||
        spec.pad_token_id < 0) {
        return fail(error, "Gemma 4 unified audio hints lack the audio token ids");
    }
    if (!spec.samples_per_token || !spec.sampling_rate || !spec.max_tokens ||
        !spec.hidden || !ms ||
        uint64_t(spec.sampling_rate) * ms / 1000 != spec.samples_per_token ||
        !(spec.rms_norm_eps > 0.0f)) {
        return fail(error, "Gemma 4 unified audio hints are incomplete or inconsistent");
    }
    return true;
}

Gemma4VisionPreprocessConfig gemma4_unified_preprocess_config(
    const Gemma4UnifiedVisionSpec& spec) {
    Gemma4VisionPreprocessConfig config;
    config.patch_size = spec.model_patch_size;
    config.pooling_kernel_size = 1;
    config.max_soft_tokens = spec.max_soft_tokens;
    config.rescale_factor = 1.0f / 255.0f;
    return config;
}

Gemma4UnifiedEmbedder::Gemma4UnifiedEmbedder(Engine& engine, const HnfLoader& loader)
    : engine_(engine) {
    vision_ok_ = gemma4_unified_vision_spec(loader, vision_) &&
                 loader.is_block_loaded(BLOCK_VISION);
    audio_ok_ = gemma4_unified_audio_spec(loader, audio_) &&
                loader.is_block_loaded(BLOCK_AUDIO);
}

Gemma4UnifiedEmbedder::~Gemma4UnifiedEmbedder() {
    release(d_input_f32_);
    release(d_input_);
    release(d_input_norm_);
    release(d_positions_);
    release(d_patch_);
    release(d_pos_);
    release(d_scratch_);
    release(d_dense_bias_);
}

const half* Gemma4UnifiedEmbedder::weight(const char* name, std::string* error) const {
    const TensorInfo* tensor = engine_.tensors().get(name);
    if (!tensor || !tensor->ptr || tensor->dtype != dtype::FP16()) {
        fail(error, std::string("missing FP16 embedder tensor ") + name);
        return nullptr;
    }
    return static_cast<const half*>(tensor->ptr);
}

bool Gemma4UnifiedEmbedder::reserve(size_t input_values, size_t rows, std::string* error) {
    if (input_values > input_capacity_) {
        release(d_input_f32_);
        release(d_input_);
        release(d_input_norm_);
        input_capacity_ = 0;
        if (!cuda_ok(cudaMalloc(&d_input_f32_, input_values * sizeof(float)),
                     "allocate embedder input", error) ||
            !cuda_ok(cudaMalloc(&d_input_, input_values * sizeof(half)),
                     "allocate embedder FP16 input", error) ||
            !cuda_ok(cudaMalloc(&d_input_norm_, input_values * sizeof(half)),
                     "allocate embedder normed input", error)) {
            return false;
        }
        input_capacity_ = input_values;
    }
    const uint32_t hidden = vision_ok_ ? vision_.hidden : audio_.hidden;
    if (rows > row_capacity_) {
        release(d_positions_);
        release(d_patch_);
        release(d_pos_);
        release(d_scratch_);
        row_capacity_ = 0;
        const size_t values = rows * hidden;
        if (!cuda_ok(cudaMalloc(&d_positions_, rows * 2 * sizeof(int32_t)),
                     "allocate embedder positions", error) ||
            !cuda_ok(cudaMalloc(&d_patch_, values * sizeof(half)),
                     "allocate embedder patch stage", error) ||
            !cuda_ok(cudaMalloc(&d_pos_, values * sizeof(half)),
                     "allocate embedder position stage", error) ||
            !cuda_ok(cudaMalloc(&d_scratch_, values * sizeof(half)),
                     "allocate embedder scratch", error)) {
            return false;
        }
        row_capacity_ = rows;
    }
    return true;
}

bool Gemma4UnifiedEmbedder::scaled_bias(const half* bias, std::string* error) {
    if (d_dense_bias_) return true;
    std::vector<half> host(vision_.hidden);
    if (!cuda_ok(cudaMemcpy(host.data(), bias, host.size() * sizeof(half),
                            cudaMemcpyDeviceToHost), "read patch Dense bias", error)) {
        return false;
    }
    for (half& value : host) value = __float2half(__half2float(value) * kDenseScale);
    return cuda_ok(cudaMalloc(&d_dense_bias_, host.size() * sizeof(half)),
                   "allocate scaled patch bias", error) &&
           cuda_ok(cudaMemcpy(d_dense_bias_, host.data(), host.size() * sizeof(half),
                              cudaMemcpyHostToDevice), "upload scaled patch bias", error);
}

bool Gemma4UnifiedEmbedder::embed_image(const Gemma4VisionPreprocessResult& input,
                                        half* output, std::string* error) {
    if (error) error->clear();
    if (!vision_ok_) return fail(error, "unified vision embedder is not loaded");
    const uint32_t rows = input.real_patches;
    const uint32_t width = vision_.patch_values;
    if (rows == 0 || rows > vision_.max_soft_tokens || input.patch_values != width ||
        input.pixel_values.size() < size_t(rows) * width ||
        input.position_ids.size() < size_t(rows) * 2 || !output) {
        return fail(error, "image patches do not match the unified vision contract");
    }
    const half* ln1_w = weight("vision.patch_ln1.weight", error);
    const half* ln1_b = weight("vision.patch_ln1.bias", error);
    const half* dense = weight("vision.patch_dense.weight", error);
    const half* dense_b = weight("vision.patch_dense.bias", error);
    const half* ln2_w = weight("vision.patch_ln2.weight", error);
    const half* ln2_b = weight("vision.patch_ln2.bias", error);
    const half* table = weight("vision.pos_embedding", error);
    const half* pos_w = weight("vision.pos_norm.weight", error);
    const half* pos_b = weight("vision.pos_norm.bias", error);
    const half* projection = weight("vision.mm_projection.weight", error);
    if (!ln1_w || !ln1_b || !dense || !dense_b || !ln2_w || !ln2_b || !table ||
        !pos_w || !pos_b || !projection) {
        return false;
    }
    if (!reserve(size_t(rows) * width, rows, error)) return false;

    const cudaStream_t stream = engine_.config().stream;
    const int hidden = int(vision_.hidden);
    const size_t values = size_t(rows) * width;
    const size_t hidden_values = size_t(rows) * vision_.hidden;
    if (!cuda_ok(cudaMemcpyAsync(d_input_f32_, input.pixel_values.data(),
                                 values * sizeof(float), cudaMemcpyHostToDevice, stream),
                 "upload image patches", error) ||
        !cuda_ok(cudaMemcpyAsync(d_positions_, input.position_ids.data(),
                                 size_t(rows) * 2 * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream),
                 "upload image positions", error)) {
        return false;
    }
    kernels::launch_fp32_to_fp16(d_input_f32_, d_input_, values, stream);
    kernels::launch_layernorm_fp16(d_input_, ln1_w, ln1_b, d_input_norm_,
                                   int(rows), int(width), kLayerNormEps, stream);
    if (!scaled_bias(dense_b, error)) return false;
    cublasHandle_t handle = kernels::cublas_handle_for_stream(stream);
    const float zero = 0.0f;
    // Row-major [rows, width] x [hidden, width]^T, FP32 accumulation.
    if (!handle || cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden, int(rows), int(width),
                                &kDenseScale, dense, CUDA_R_16F, int(width),
                                d_input_norm_, CUDA_R_16F, int(width), &zero,
                                d_scratch_, CUDA_R_16F, hidden, CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
        return fail(error, "unified vision patch Dense failed");
    }
    kernels::launch_add_bias_fp16(d_scratch_, d_dense_bias_, d_scratch_,
                                  hidden_values, vision_.hidden, stream);
    kernels::launch_layernorm_fp16(d_scratch_, ln2_w, ln2_b, d_patch_,
                                   int(rows), hidden, kLayerNormEps, stream);
    // The position add is in place; keep the LN2 stage intact for checking.
    if (!cuda_ok(cudaMemcpyAsync(d_scratch_, d_patch_, hidden_values * sizeof(half),
                                 cudaMemcpyDeviceToDevice, stream),
                 "copy patch stage", error)) {
        return false;
    }
    kernels::launch_gemma4_unified_pos_add_fp16(table, d_positions_, d_scratch_,
                                                int(rows), hidden,
                                                int(vision_.posemb_size), stream);
    kernels::launch_layernorm_fp16(d_scratch_, pos_w, pos_b, d_pos_,
                                   int(rows), hidden, kLayerNormEps, stream);
    kernels::launch_rmsnorm_no_weight_fp16(d_pos_, d_scratch_, int(rows), hidden,
                                           vision_.rms_norm_eps, stream);
    kernels::launch_matmul_fp16(d_scratch_, projection, output,
                                int(rows), hidden, hidden, stream);
    if (!cuda_ok(cudaGetLastError(), "launch unified vision embedder", error)) {
        return false;
    }
    rows_ = rows;
    return cuda_ok(cudaStreamSynchronize(stream), "run unified vision embedder", error);
}

bool Gemma4UnifiedEmbedder::embed_audio(const float* frames, uint32_t tokens,
                                        half* output, std::string* error) {
    if (error) error->clear();
    if (!audio_ok_) return fail(error, "unified audio embedder is not loaded");
    if (!frames || !output || tokens == 0 || tokens > audio_.max_tokens) {
        return fail(error, "audio frames do not match the unified audio contract");
    }
    const half* projection = weight("audio.mm_projection.weight", error);
    if (!projection) return false;
    const uint32_t width = audio_.samples_per_token;
    const size_t values = size_t(tokens) * width;
    if (!reserve(values, tokens, error)) return false;

    const cudaStream_t stream = engine_.config().stream;
    if (!cuda_ok(cudaMemcpyAsync(d_input_f32_, frames, values * sizeof(float),
                                 cudaMemcpyHostToDevice, stream),
                 "upload audio frames", error)) {
        return false;
    }
    kernels::launch_fp32_to_fp16(d_input_f32_, d_input_, values, stream);
    kernels::launch_rmsnorm_no_weight_fp16(d_input_, d_input_norm_, int(tokens), int(width),
                                           audio_.rms_norm_eps, stream);
    kernels::launch_matmul_fp16(d_input_norm_, projection, output,
                                int(tokens), int(width), int(audio_.hidden), stream);
    if (!cuda_ok(cudaGetLastError(), "launch unified audio embedder", error)) {
        return false;
    }
    return cuda_ok(cudaStreamSynchronize(stream), "run unified audio embedder", error);
}

bool Gemma4UnifiedEmbedder::copy_stage(Stage stage, std::vector<float>& values,
                                       std::string* error) const {
    if (rows_ == 0) return fail(error, "no image has been embedded yet");
    const half* source = stage == Stage::PatchLn2 ? d_patch_ : d_pos_;
    const size_t count = size_t(rows_) * vision_.hidden;
    std::vector<half> host(count);
    if (!cuda_ok(cudaMemcpy(host.data(), source, count * sizeof(half),
                            cudaMemcpyDeviceToHost),
                 "copy embedder stage", error)) {
        return false;
    }
    values.resize(count);
    for (size_t i = 0; i < count; ++i) values[i] = __half2float(host[i]);
    return true;
}

} // namespace helios
