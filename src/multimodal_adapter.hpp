#pragma once

#include "graph_builder.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace helios {

enum class AttachmentKind : uint8_t {
    ImageRgb8 = 0,
    AudioPcmF32 = 1,
};

constexpr uint32_t attachment_kind_bit(AttachmentKind kind) {
    return uint32_t{1} << static_cast<uint8_t>(kind);
}

// Borrowed payload valid for the duration of MultimodalAdapter::prefill().
// Geometry fields are interpreted by kind. Encoded PNG/JPEG decoding remains
// outside the engine boundary; image adapters receive decoded RGB8.
struct TurnAttachment {
    AttachmentKind kind = AttachmentKind::ImageRgb8;
    const void* data = nullptr;
    size_t byte_size = 0;
    std::string media_type;

    uint32_t width = 0;
    uint32_t height = 0;
    size_t row_stride_bytes = 0;

    uint32_t sample_rate = 0;
    uint32_t channels = 0;
};

struct MultimodalTurnInput {
    // Already framed by the chat template and containing the architecture's
    // attachment placeholder. The adapter owns expansion/substitution.
    std::vector<int32_t> formatted_token_ids;
    std::vector<TurnAttachment> attachments;
};

struct MultimodalAdapterLimits {
    uint32_t supported_attachment_mask = 0;
    uint32_t max_attachments_per_turn = 0;
    uint32_t max_prefill_tokens = 0;
    size_t max_attachment_bytes = 0;
};

struct MultimodalPrefillResult {
    uint32_t sequence_tokens = 0;
    uint32_t injected_tokens = 0;
    uint32_t attachment_count = 0;
};

// Common validation is public so process boundaries can reject malformed
// payloads before touching CUDA or an architecture-specific adapter.
bool validate_multimodal_turn(
    const MultimodalTurnInput& turn,
    const MultimodalAdapterLimits& limits,
    std::string* error = nullptr);

class MultimodalAdapter {
public:
    virtual ~MultimodalAdapter() = default;

    virtual const char* id() const = 0;
    virtual MultimodalAdapterLimits limits() const = 0;

    // Token that marks where an attachment of this kind goes in an already
    // formatted prompt, or -1 when the kind is unsupported. Sessions count
    // these markers instead of reading one architecture's vision config.
    virtual int32_t marker_token(AttachmentKind kind) const = 0;

    // True when the tokens around a marker are plain causal text, so a session
    // may prefill them through the normal text path and hand the adapter only
    // the marker. False keeps the whole turn inside the adapter.
    virtual bool prefix_is_text() const { return false; }

    // Executes the modality encoder and the decoder prefill at
    // cache.cache_position. The owner advances its logical KV position by
    // result.sequence_tokens only after this call succeeds.
    virtual bool prefill(
        const MultimodalTurnInput& turn,
        const KVCacheParams& cache,
        MultimodalPrefillResult& result,
        std::string* error = nullptr) = 0;
};

// Factory boundary used by persistent sessions. Unknown or metadata-only
// adapter ids fail visibly; callers never switch on model architecture.
std::unique_ptr<MultimodalAdapter> create_multimodal_adapter(
    const std::string& adapter_id,
    Engine& engine,
    HnfLoader& loader,
    GraphBuilder& graph,
    const ArchDescriptor& text_architecture,
    uint32_t max_prefill_tokens,
    std::string* error = nullptr);

// Gemma 4 12B «unified»: image and audio embedders from a modality-only HNF
// (`modality_loader`, blocks 0x1/0x2) feeding the text model of `text_loader`.
// Loads whichever of the two blocks the HNF declares and keeps them resident.
std::unique_ptr<MultimodalAdapter> create_gemma4_unified_adapter(
    Engine& engine,
    HnfLoader& text_loader,
    HnfLoader& modality_loader,
    GraphBuilder& graph,
    const ArchDescriptor& text_architecture,
    uint32_t max_prefill_tokens,
    std::string* error = nullptr);

} // namespace helios
