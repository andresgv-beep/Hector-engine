#include "multimodal_adapter.hpp"

#include "gemma4_multimodal.hpp"
#include "gemma4_vision_preprocess.hpp"
#include "gemma4_vision_runner.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <limits>
#include <sstream>

namespace helios {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

bool checked_image_size(const TurnAttachment& attachment,
                        size_t& required, std::string* error) {
    if (!attachment.data || attachment.width == 0 || attachment.height == 0) {
        return fail(error, "RGB8 attachment has no pixels or geometry");
    }
    if (attachment.width > std::numeric_limits<size_t>::max() / 3) {
        return fail(error, "RGB8 row width overflows this platform");
    }
    const size_t packed = size_t(attachment.width) * 3;
    const size_t stride = attachment.row_stride_bytes
        ? attachment.row_stride_bytes : packed;
    if (stride < packed ||
        attachment.height > std::numeric_limits<size_t>::max() / stride) {
        return fail(error, "RGB8 stride or image size is invalid");
    }
    required = stride * attachment.height;
    if (attachment.byte_size < required) {
        return fail(error, "RGB8 payload is shorter than its geometry");
    }
    return true;
}

bool cuda_ok(cudaError_t status, const char* operation, std::string* error) {
    if (status == cudaSuccess) return true;
    std::ostringstream message;
    message << operation << ": " << cudaGetErrorString(status);
    return fail(error, message.str());
}

class Gemma4VisionAdapter final : public MultimodalAdapter {
public:
    Gemma4VisionAdapter(Engine& engine, HnfLoader& loader,
                        GraphBuilder& graph, const ArchDescriptor& text_arch,
                        uint32_t max_prefill_tokens)
        : engine_(engine), loader_(loader), graph_(graph), text_arch_(text_arch),
          max_prefill_tokens_(max_prefill_tokens) {}

    ~Gemma4VisionAdapter() override {
        if (loaded_vision_here_ && loader_.is_block_loaded(BLOCK_VISION)) {
            loader_.unload_block(BLOCK_VISION, engine_);
        }
    }

    const char* id() const override { return "helios.gemma4.vision.v1"; }

    MultimodalAdapterLimits limits() const override {
        return {
            attachment_kind_bit(AttachmentKind::ImageRgb8),
            1,
            max_prefill_tokens_,
            size_t{300} * 1024 * 1024,
        };
    }

    bool prefill(const MultimodalTurnInput& turn, const KVCacheParams& cache,
                 MultimodalPrefillResult& result,
                 std::string* error) override {
        result = {};
        if (error) error->clear();
        if (!validate_multimodal_turn(turn, limits(), error)) return false;
        if (!loader_.has_gemma4_config() ||
            !loader_.has_gemma4_vision_config() ||
            loader_.config().arch() != "gemma4") {
            return fail(error, "Gemma 4 vision adapter metadata is incomplete");
        }
        if (turn.attachments.size() != 1 ||
            turn.attachments.front().kind != AttachmentKind::ImageRgb8) {
            return fail(error, "Gemma 4 vision v1 requires exactly one RGB8 image");
        }

        try {
            const TurnAttachment& attachment = turn.attachments.front();
            const Gemma4VisionConfig& vision = loader_.gemma4_vision_config();
            Gemma4VisionPreprocessConfig preprocess_config;
            preprocess_config.patch_size = vision.patch_size;
            preprocess_config.pooling_kernel_size = vision.pooling_kernel_size;
            preprocess_config.max_soft_tokens = vision.max_soft_tokens;
            preprocess_config.rescale_factor = vision.rescale_factor;

            Gemma4VisionPreprocessResult preprocessed;
            if (!gemma4_vision_preprocess_rgb(
                    {static_cast<const uint8_t*>(attachment.data),
                     attachment.width, attachment.height,
                     attachment.row_stride_bytes},
                    preprocess_config, preprocessed, error)) {
                return false;
            }

            const Gemma4MultimodalTokenPlan plan =
                make_gemma4_multimodal_token_plan(
                    turn.formatted_token_ids, vision,
                    loader_.config().vocab_size(), preprocessed.soft_tokens);
            if (plan.canonical_ids.size() > max_prefill_tokens_) {
                return fail(error, "expanded multimodal turn exceeds prefill capacity");
            }
            if (cache.max_cache_len == 0 ||
                cache.cache_position > cache.max_cache_len ||
                plan.canonical_ids.size() >
                    cache.max_cache_len - cache.cache_position) {
                return fail(error, "multimodal turn does not fit the KV cache");
            }
            if (!ensure_buffers(error) || !run_vision(preprocessed, error)) {
                return false;
            }

            const uint32_t sequence =
                static_cast<uint32_t>(plan.canonical_ids.size());
            TensorInfo& embedding = engine_.tensors().at(kEmbeddingTokens);
            TensorInfo& ple = engine_.tensors().at(kPleTokens);
            TensorInfo& positions = engine_.tensors().at(kImagePositions);
            TensorInfo& image = engine_.tensors().at(kImageEmbeddings);
            embedding.shape = {1, sequence};
            ple.shape = {1, sequence};
            positions.shape = {preprocessed.soft_tokens};
            image.shape = {preprocessed.soft_tokens,
                           loader_.config().hidden_size()};

            const cudaStream_t stream = engine_.config().stream;
            if (!cuda_ok(cudaMemcpyAsync(
                    embedding.ptr, plan.embedding_ids.data(),
                    plan.embedding_ids.size() * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream),
                    "upload multimodal embedding ids", error) ||
                !cuda_ok(cudaMemcpyAsync(
                    ple.ptr, plan.ple_identity_ids.data(),
                    plan.ple_identity_ids.size() * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream),
                    "upload multimodal PLE ids", error) ||
                !cuda_ok(cudaMemcpyAsync(
                    positions.ptr, plan.image_positions.data(),
                    plan.image_positions.size() * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream),
                    "upload multimodal image positions", error)) {
                return false;
            }

            const Gemma4MultimodalInputNames names{
                kEmbeddingTokens, kPleTokens, kImageEmbeddings, kImagePositions};
            engine_.execute(graph_.build_gemma4_multimodal_forward_cached(
                engine_, loader_.config(), loader_.gemma4_config(), text_arch_,
                names, 1, sequence, cache));
            engine_.sync();

            result.sequence_tokens = sequence;
            result.injected_tokens = preprocessed.soft_tokens;
            result.attachment_count = 1;
            return true;
        } catch (const std::exception& exception) {
            return fail(error, exception.what());
        }
    }

private:
    bool ensure_buffers(std::string* error) {
        const bool has_embedding = engine_.tensors().exists(kEmbeddingTokens);
        const bool has_ple = engine_.tensors().exists(kPleTokens);
        const bool has_positions = engine_.tensors().exists(kImagePositions);
        const bool has_image = engine_.tensors().exists(kImageEmbeddings);
        if (has_embedding && has_ple && has_positions && has_image) return true;
        if (has_embedding || has_ple || has_positions || has_image) {
            return fail(error, "Gemma 4 multimodal buffers are partially registered");
        }
        const uint32_t visual_tokens =
            loader_.gemma4_vision_config().max_soft_tokens;
        const uint32_t hidden = loader_.config().hidden_size();
        if (max_prefill_tokens_ == 0 || visual_tokens == 0 || hidden == 0) {
            return fail(error, "Gemma 4 multimodal buffer geometry is invalid");
        }
        try {
            engine_.tensors().allocate_and_register(
                kEmbeddingTokens, {1, max_prefill_tokens_}, dtype::INT32());
            engine_.tensors().allocate_and_register(
                kPleTokens, {1, max_prefill_tokens_}, dtype::INT32());
            engine_.tensors().allocate_and_register(
                kImagePositions, {visual_tokens}, dtype::INT32());
            engine_.tensors().allocate_and_register(
                kImageEmbeddings, {visual_tokens, hidden}, dtype::FP16());
            return true;
        } catch (const std::exception& exception) {
            engine_.tensors().remove(kImageEmbeddings);
            engine_.tensors().remove(kImagePositions);
            engine_.tensors().remove(kPleTokens);
            engine_.tensors().remove(kEmbeddingTokens);
            return fail(error, exception.what());
        }
    }

    bool run_vision(const Gemma4VisionPreprocessResult& input,
                    std::string* error) {
        const bool was_loaded = loader_.is_block_loaded(BLOCK_VISION);
        if (!was_loaded && !loader_.load_block(BLOCK_VISION, engine_)) {
            return fail(error, "cannot load vision block: " + loader_.last_error());
        }
        const bool keep_mapped = [] {
            const char* value = std::getenv("HELIOS_VISION_MMAP");
            return value && value[0] == '1';
        }();
        loaded_vision_here_ = !was_loaded && keep_mapped;

        bool ok = false;
        {
            Gemma4VisionRunner runner(engine_, loader_);
            ok = runner.run_patch_embedder(input, error);
            for (uint32_t layer = 0; ok && layer < runner.num_layers(); ++layer) {
                ok = runner.run_encoder_layer(layer, error);
            }
            if (ok) ok = runner.run_pooler(error);
            if (ok) ok = runner.run_projector(error);
            if (ok && (runner.soft_token_count() != input.soft_tokens ||
                       runner.projection_size() != loader_.config().hidden_size())) {
                ok = fail(error, "visual output differs from text projection contract");
            }
            if (ok) {
                const size_t bytes = size_t(input.soft_tokens) *
                    runner.projection_size() * sizeof(half);
                ok = cuda_ok(cudaMemcpyAsync(
                    engine_.tensors().at(kImageEmbeddings).ptr,
                    runner.projected_states_device(), bytes,
                    cudaMemcpyDeviceToDevice, engine_.config().stream),
                    "persist multimodal visual embeddings", error);
                if (ok) engine_.sync();
            }
        }

        if (!was_loaded && !keep_mapped) {
            if (!loader_.unload_block(BLOCK_VISION, engine_)) {
                return fail(error, "cannot unload transient vision block");
            }
        }
        return ok;
    }

    static constexpr const char* kEmbeddingTokens = "_mm.g4v1.embedding_tokens";
    static constexpr const char* kPleTokens = "_mm.g4v1.ple_tokens";
    static constexpr const char* kImagePositions = "_mm.g4v1.image_positions";
    static constexpr const char* kImageEmbeddings = "_mm.g4v1.image_embeddings";

    Engine& engine_;
    HnfLoader& loader_;
    GraphBuilder& graph_;
    const ArchDescriptor& text_arch_;
    uint32_t max_prefill_tokens_ = 0;
    bool loaded_vision_here_ = false;
};

} // namespace

bool validate_multimodal_turn(const MultimodalTurnInput& turn,
                              const MultimodalAdapterLimits& limits,
                              std::string* error) {
    if (error) error->clear();
    if (turn.formatted_token_ids.empty()) {
        return fail(error, "multimodal turn has no formatted tokens");
    }
    if (limits.max_prefill_tokens == 0 ||
        turn.formatted_token_ids.size() > limits.max_prefill_tokens) {
        return fail(error, "multimodal turn exceeds token capacity");
    }
    if (turn.attachments.empty() ||
        turn.attachments.size() > limits.max_attachments_per_turn) {
        return fail(error, "multimodal attachment count is unsupported");
    }
    for (const TurnAttachment& attachment : turn.attachments) {
        if ((limits.supported_attachment_mask &
             attachment_kind_bit(attachment.kind)) == 0) {
            return fail(error, "attachment kind is unsupported by this adapter");
        }
        if (attachment.byte_size == 0 ||
            attachment.byte_size > limits.max_attachment_bytes) {
            return fail(error, "attachment payload size is unsupported");
        }
        if (attachment.kind == AttachmentKind::ImageRgb8) {
            if (!attachment.media_type.empty() &&
                attachment.media_type != "image/rgb8") {
                return fail(error, "RGB8 attachment has a conflicting media type");
            }
            size_t required = 0;
            if (!checked_image_size(attachment, required, error)) return false;
        }
    }
    return true;
}

std::unique_ptr<MultimodalAdapter> create_multimodal_adapter(
    const std::string& adapter_id, Engine& engine, HnfLoader& loader,
    GraphBuilder& graph, const ArchDescriptor& text_architecture,
    uint32_t max_prefill_tokens, std::string* error) {
    if (error) error->clear();
    if (adapter_id == "helios.gemma4.vision.v1") {
        if (!loader.has_gemma4_config() ||
            !loader.has_gemma4_vision_config()) {
            fail(error, "Gemma 4 vision adapter requested for incompatible HNF");
            return nullptr;
        }
        return std::make_unique<Gemma4VisionAdapter>(
            engine, loader, graph, text_architecture, max_prefill_tokens);
    }
    fail(error, "unknown or unavailable multimodal adapter: " + adapter_id);
    return nullptr;
}

} // namespace helios
