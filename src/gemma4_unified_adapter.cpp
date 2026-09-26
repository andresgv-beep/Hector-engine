// Gemma 4 12B «unified» multimodal adapter: one image or one audio clip per
// turn, embedded without a tower and scattered into the decoder prefill.
#include "multimodal_adapter.hpp"

#include "gemma4_multimodal.hpp"
#include "gemma4_unified_embedder.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
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

class Gemma4UnifiedAdapter final : public MultimodalAdapter {
public:
    Gemma4UnifiedAdapter(Engine& engine, HnfLoader& text, HnfLoader& modality,
                         GraphBuilder& graph, const ArchDescriptor& arch,
                         uint32_t max_prefill_tokens)
        : engine_(engine), text_(text), graph_(graph), arch_(arch),
          max_prefill_tokens_(max_prefill_tokens),
          embedder_(std::make_unique<Gemma4UnifiedEmbedder>(engine, modality)) {}

    const char* id() const override { return "helios.gemma4u.multimodal.v1"; }

    // Image blocks are bidirectional only among their own soft tokens; the
    // text before and after the attachment stays causal.
    bool prefix_is_text() const override { return true; }

    bool ready() const { return embedder_->has_vision() || embedder_->has_audio(); }

    MultimodalAdapterLimits limits() const override {
        uint32_t mask = 0;
        if (embedder_->has_vision()) mask |= attachment_kind_bit(AttachmentKind::ImageRgb8);
        if (embedder_->has_audio()) mask |= attachment_kind_bit(AttachmentKind::AudioPcmF32);
        return {mask, 1, max_prefill_tokens_, size_t{300} * 1024 * 1024};
    }

    int32_t marker_token(AttachmentKind kind) const override {
        if (kind == AttachmentKind::ImageRgb8 && embedder_->has_vision()) {
            return embedder_->vision().image_token_id;
        }
        if (kind == AttachmentKind::AudioPcmF32 && embedder_->has_audio()) {
            return embedder_->audio().audio_token_id;
        }
        return -1;
    }

    bool prefill(const MultimodalTurnInput& turn, const KVCacheParams& cache,
                 MultimodalPrefillResult& result, std::string* error) override {
        result = {};
        if (error) error->clear();
        if (!validate_multimodal_turn(turn, limits(), error)) return false;
        if (!text_.has_gemma4_config() || text_.config().arch() != "gemma4") {
            return fail(error, "Gemma 4 unified adapter needs a Gemma 4 text model");
        }
        try {
            if (!ensure_buffers(error)) return false;
            const TurnAttachment& attachment = turn.attachments.front();
            half* embeddings = static_cast<half*>(engine_.tensors().at(kEmbeddings).ptr);
            // The E4B token plan works for any marker: it only needs the ids.
            Gemma4VisionConfig tokens;
            uint32_t rows = 0;
            const bool image = attachment.kind == AttachmentKind::ImageRgb8;
            if (image) {
                const Gemma4UnifiedVisionSpec& spec = embedder_->vision();
                Gemma4VisionPreprocessResult patches;
                if (!gemma4_vision_preprocess_rgb(
                        {static_cast<const uint8_t*>(attachment.data), attachment.width,
                         attachment.height, attachment.row_stride_bytes},
                        gemma4_unified_preprocess_config(spec), patches, error) ||
                    !embedder_->embed_image(patches, embeddings, error)) {
                    return false;
                }
                rows = patches.real_patches;
                tokens.image_token_id = spec.image_token_id;
                tokens.boi_token_id = spec.boi_token_id;
                tokens.eoi_token_id = spec.eoi_token_id;
                tokens.pad_token_id = spec.pad_token_id;
                tokens.max_soft_tokens = spec.max_soft_tokens;
            } else {
                const Gemma4UnifiedAudioSpec& spec = embedder_->audio();
                if (attachment.sample_rate != spec.sampling_rate || attachment.channels != 1 ||
                    attachment.byte_size % sizeof(float) != 0) {
                    std::ostringstream message;
                    message << "audio must be mono PCM F32 at " << spec.sampling_rate << " Hz";
                    return fail(error, message.str());
                }
                const size_t samples = attachment.byte_size / sizeof(float);
                const size_t frame = spec.samples_per_token;
                rows = static_cast<uint32_t>((samples + frame - 1) / frame);
                if (rows == 0 || rows > spec.max_tokens) {
                    std::ostringstream message;
                    message << "audio must last between one frame and "
                            << spec.max_tokens * frame / spec.sampling_rate << " s";
                    return fail(error, message.str());
                }
                // The feature extractor zero-pads the last frame and masks
                // nothing inside it: the padded tail is part of the token.
                std::vector<float> frames(size_t(rows) * frame, 0.0f);
                std::copy_n(static_cast<const float*>(attachment.data), samples, frames.begin());
                if (!embedder_->embed_audio(frames.data(), rows, embeddings, error)) return false;
                tokens.image_token_id = spec.audio_token_id;
                tokens.boi_token_id = spec.boa_token_id;
                tokens.eoi_token_id = spec.eoa_token_id;
                tokens.pad_token_id = spec.pad_token_id;
                tokens.max_soft_tokens = spec.max_tokens;
            }

            const Gemma4MultimodalTokenPlan plan = make_gemma4_multimodal_token_plan(
                turn.formatted_token_ids, tokens, text_.config().vocab_size(), rows);
            const uint32_t sequence = static_cast<uint32_t>(plan.canonical_ids.size());
            if (sequence > max_prefill_tokens_) {
                return fail(error, "expanded multimodal turn exceeds prefill capacity");
            }
            if (cache.max_cache_len == 0 || cache.cache_position > cache.max_cache_len ||
                sequence > cache.max_cache_len - cache.cache_position) {
                return fail(error, "multimodal turn does not fit the KV cache");
            }

            TensorInfo& embedding_ids = engine_.tensors().at(kEmbeddingTokens);
            TensorInfo& ple_ids = engine_.tensors().at(kPleTokens);
            TensorInfo& positions = engine_.tensors().at(kPositions);
            TensorInfo& rows_tensor = engine_.tensors().at(kEmbeddings);
            embedding_ids.shape = {1, sequence};
            ple_ids.shape = {1, sequence};
            positions.shape = {rows};
            rows_tensor.shape = {rows, text_.config().hidden_size()};
            const cudaStream_t stream = engine_.config().stream;
            if (!cuda_ok(cudaMemcpyAsync(embedding_ids.ptr, plan.embedding_ids.data(),
                                         plan.embedding_ids.size() * sizeof(int32_t),
                                         cudaMemcpyHostToDevice, stream),
                         "upload multimodal embedding ids", error) ||
                !cuda_ok(cudaMemcpyAsync(ple_ids.ptr, plan.ple_identity_ids.data(),
                                         plan.ple_identity_ids.size() * sizeof(int32_t),
                                         cudaMemcpyHostToDevice, stream),
                         "upload multimodal identity ids", error) ||
                !cuda_ok(cudaMemcpyAsync(positions.ptr, plan.image_positions.data(),
                                         plan.image_positions.size() * sizeof(int32_t),
                                         cudaMemcpyHostToDevice, stream),
                         "upload multimodal positions", error)) {
                return false;
            }
            // One image's soft tokens are contiguous; audio stays causal.
            Gemma4BidirectionalBlock block;
            if (image) {
                block.begin = static_cast<uint32_t>(plan.image_positions.front());
                block.end = static_cast<uint32_t>(plan.image_positions.back()) + 1;
            }
            const Gemma4MultimodalInputNames names{kEmbeddingTokens, kPleTokens,
                                                   kEmbeddings, kPositions};
            engine_.execute(graph_.build_gemma4_multimodal_forward_cached(
                engine_, text_.config(), text_.gemma4_config(), arch_, names, 1,
                sequence, cache, block));
            engine_.sync();
            result.sequence_tokens = sequence;
            result.injected_tokens = rows;
            result.attachment_count = 1;
            return true;
        } catch (const std::exception& exception) {
            return fail(error, exception.what());
        }
    }

private:
    bool ensure_buffers(std::string* error) {
        if (engine_.tensors().exists(kEmbeddings)) return true;
        uint32_t rows = 0;
        if (embedder_->has_vision()) rows = std::max(rows, embedder_->vision().max_soft_tokens);
        if (embedder_->has_audio()) rows = std::max(rows, embedder_->audio().max_tokens);
        const uint32_t hidden = text_.config().hidden_size();
        if (rows == 0 || hidden == 0 || max_prefill_tokens_ == 0) {
            return fail(error, "Gemma 4 unified buffer geometry is invalid");
        }
        try {
            engine_.tensors().allocate_and_register(kEmbeddingTokens, {1, max_prefill_tokens_}, dtype::INT32());
            engine_.tensors().allocate_and_register(kPleTokens, {1, max_prefill_tokens_}, dtype::INT32());
            engine_.tensors().allocate_and_register(kPositions, {rows}, dtype::INT32());
            engine_.tensors().allocate_and_register(kEmbeddings, {rows, hidden}, dtype::FP16());
            return true;
        } catch (const std::exception& exception) {
            engine_.tensors().remove(kEmbeddings);
            engine_.tensors().remove(kPositions);
            engine_.tensors().remove(kPleTokens);
            engine_.tensors().remove(kEmbeddingTokens);
            return fail(error, exception.what());
        }
    }

    static constexpr const char* kEmbeddingTokens = "_mm.g4u1.embedding_tokens";
    static constexpr const char* kPleTokens = "_mm.g4u1.identity_tokens";
    static constexpr const char* kPositions = "_mm.g4u1.positions";
    static constexpr const char* kEmbeddings = "_mm.g4u1.embeddings";

    Engine& engine_;
    HnfLoader& text_;
    GraphBuilder& graph_;
    const ArchDescriptor& arch_;
    uint32_t max_prefill_tokens_ = 0;
    std::unique_ptr<Gemma4UnifiedEmbedder> embedder_;
};

} // namespace

std::unique_ptr<MultimodalAdapter> create_gemma4_unified_adapter(
    Engine& engine, HnfLoader& text_loader, HnfLoader& modality_loader,
    GraphBuilder& graph, const ArchDescriptor& text_architecture,
    uint32_t max_prefill_tokens, std::string* error) {
    if (error) error->clear();
    if (!text_loader.has_gemma4_config() || text_loader.config().arch() != "gemma4") {
        if (error) *error = "the text model is not Gemma 4";
        return nullptr;
    }
    std::string reason;
    Gemma4UnifiedVisionSpec vision;
    Gemma4UnifiedAudioSpec audio;
    const bool has_vision = gemma4_unified_vision_spec(modality_loader, vision, &reason);
    const bool has_audio = gemma4_unified_audio_spec(modality_loader, audio, &reason);
    if (!has_vision && !has_audio) {
        if (error) *error = "modality HNF has neither unified vision nor audio: " + reason;
        return nullptr;
    }
    const uint32_t hidden = text_loader.config().hidden_size();
    if ((has_vision && vision.hidden != hidden) || (has_audio && audio.hidden != hidden)) {
        if (error) *error = "modality HNF was converted for another hidden size";
        return nullptr;
    }
    for (const auto& [present, block, name] :
         {std::tuple{has_vision, BLOCK_VISION, "vision"}, std::tuple{has_audio, BLOCK_AUDIO, "audio"}}) {
        if (present && !modality_loader.is_block_loaded(block) &&
            !modality_loader.load_block(block, engine)) {
            if (error) *error = std::string("cannot load the ") + name + " block: " +
                                modality_loader.last_error();
            return nullptr;
        }
    }
    auto adapter = std::make_unique<Gemma4UnifiedAdapter>(
        engine, text_loader, modality_loader, graph, text_architecture, max_prefill_tokens);
    if (!adapter->ready()) {
        if (error) *error = "unified embedder did not validate after loading";
        return nullptr;
    }
    return adapter;
}

} // namespace helios
