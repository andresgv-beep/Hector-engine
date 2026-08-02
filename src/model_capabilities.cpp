#include "model_capabilities.hpp"

#include <array>

namespace helios {
namespace {

constexpr uint32_t header_flag(ModelModality modality) {
    switch (modality) {
        case ModelModality::Vision:    return HNF_HAS_VISION;
        case ModelModality::Audio:     return HNF_HAS_AUDIO;
        case ModelModality::Video:     return HNF_HAS_VIDEO;
        case ModelModality::Spatial3D: return HNF_HAS_SPATIAL;
        case ModelModality::Code:      return HNF_HAS_CODE_EXEC;
        case ModelModality::Text:
        case ModelModality::Cortex:    return 0;
    }
    return 0;
}

const char* enabled_key(ModelModality modality) {
    switch (modality) {
        case ModelModality::Text:   return "text_enabled";
        case ModelModality::Vision: return "vision_enabled";
        case ModelModality::Audio:  return "audio_enabled";
        case ModelModality::Code:   return "code_enabled";
        case ModelModality::Cortex: return "cortex_enabled";
        case ModelModality::Video:
        case ModelModality::Spatial3D: return nullptr;
    }
    return nullptr;
}

std::string modality_architecture(const HnfLoader& loader,
                                  ModelModality modality, BlockID block) {
    if (!loader.has_config_for_block(block)) return {};
    const ModelConfig& config = loader.config_for_block(block);
    if (modality == ModelModality::Vision) {
        return config.get<std::string>("encoder_type", "");
    }
    return config.arch();
}

// This is deliberately the only architecture-specific routing table in the
// generic layer. The adapter implementation itself stays in its own files.
constexpr std::array<ModalityAdapterRule, 4> kAdapterRules{{
    {"qwen", ModelModality::Text, "qwen", "helios.text.chat.v1"},
    {"qwen2", ModelModality::Text, "qwen2", "helios.text.chat.v1"},
    {"gemma4", ModelModality::Text, "gemma4", "helios.text.chat.v1"},
    {"gemma4", ModelModality::Vision, "gemma4", "helios.gemma4.vision.v1"},
}};

} // namespace

const ModalityCapability* ModelCapabilities::find(
    ModelModality modality) const {
    for (const auto& capability : modalities) {
        if (capability.modality == modality) return &capability;
    }
    return nullptr;
}

const char* model_modality_name(ModelModality modality) {
    switch (modality) {
        case ModelModality::Text:      return "text";
        case ModelModality::Vision:    return "vision";
        case ModelModality::Audio:     return "audio";
        case ModelModality::Video:     return "video";
        case ModelModality::Spatial3D: return "spatial_3d";
        case ModelModality::Code:      return "code";
        case ModelModality::Cortex:    return "cortex";
    }
    return "unknown";
}

const char* adapter_status_name(AdapterStatus status) {
    switch (status) {
        case AdapterStatus::Absent:        return "absent";
        case AdapterStatus::DeclaredOnly:  return "declared_only";
        case AdapterStatus::MetadataReady: return "metadata_ready";
        case AdapterStatus::RuntimeReady:  return "runtime_ready";
    }
    return "unknown";
}

const ModalityAdapterRule* resolve_modality_adapter(
    const std::string& text_architecture,
    ModelModality modality,
    const std::string& modality_architecture) {
    for (const auto& rule : kAdapterRules) {
        if (rule.modality == modality &&
            text_architecture == rule.text_architecture &&
            modality_architecture == rule.modality_architecture) {
            return &rule;
        }
    }
    return nullptr;
}

ModelCapabilities inspect_model_capabilities(const HnfLoader& loader) {
    ModelCapabilities result;
    result.text_architecture = loader.config().arch();

    struct Spec {
        ModelModality modality;
        BlockID block;
    };
    constexpr std::array<Spec, 7> specs{{
        {ModelModality::Text, BLOCK_TEXT_MODEL},
        {ModelModality::Vision, BLOCK_VISION},
        {ModelModality::Audio, BLOCK_AUDIO},
        {ModelModality::Video, BLOCK_VIDEO},
        {ModelModality::Spatial3D, BLOCK_SPATIAL_3D},
        {ModelModality::Code, BLOCK_CODE_EXEC},
        {ModelModality::Cortex, BLOCK_CORTEX},
    }};

    for (const Spec& spec : specs) {
        ModalityCapability capability;
        capability.modality = spec.modality;
        capability.block = spec.block;
        capability.present = loader.has_block(spec.block);

        const uint32_t hflag = header_flag(spec.modality);
        const char* key = enabled_key(spec.modality);
        capability.declared = capability.present ||
            (hflag != 0 && (loader.header().flags & hflag) != 0) ||
            (key && loader.config().get<bool>(key, false));

        if (spec.modality == ModelModality::Text) {
            capability.architecture = result.text_architecture;
            capability.configured = capability.architecture != "unknown" &&
                                    !capability.architecture.empty();
        } else {
            capability.architecture = modality_architecture(
                loader, spec.modality, spec.block);
            capability.configured = !capability.architecture.empty() &&
                                    capability.architecture != "unknown";
        }

        if (!capability.declared && !capability.present) {
            capability.status = AdapterStatus::Absent;
        } else if (!capability.present || !capability.configured) {
            capability.status = AdapterStatus::DeclaredOnly;
        } else if (const auto* rule = resolve_modality_adapter(
                       result.text_architecture, spec.modality,
                       capability.architecture)) {
            capability.adapter_id = rule->adapter_id;
            capability.status = AdapterStatus::RuntimeReady;
        } else {
            capability.status = AdapterStatus::MetadataReady;
        }

        if (capability.declared && !capability.present) {
            result.diagnostics.emplace_back(
                std::string(model_modality_name(spec.modality)) +
                " is declared but its HNF block is absent");
        }
        if (capability.present && !capability.configured &&
            spec.modality != ModelModality::Audio &&
            spec.modality != ModelModality::Video &&
            spec.modality != ModelModality::Spatial3D) {
            result.diagnostics.emplace_back(
                std::string(model_modality_name(spec.modality)) +
                " block has no architecture metadata");
        }

        if (spec.modality != ModelModality::Text && capability.present) {
            result.multimodal = true;
        }
        result.modalities.push_back(std::move(capability));
    }

    return result;
}

} // namespace helios
