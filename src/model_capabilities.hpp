#pragma once

#include "hnf_loader.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace helios {

// Stable orchestration vocabulary. Architecture-specific configs remain in
// HnfLoader; HexOS and future frontends only need this small capability view.
enum class ModelModality : uint8_t {
    Text,
    Vision,
    Audio,
    Video,
    Spatial3D,
    Code,
    Cortex,
};

enum class AdapterStatus : uint8_t {
    Absent,
    DeclaredOnly,
    MetadataReady,
    RuntimeReady,
};

struct ModalityCapability {
    ModelModality modality = ModelModality::Text;
    BlockID block = BLOCK_TEXT_MODEL;
    bool declared = false;
    bool present = false;
    bool configured = false;
    std::string architecture;
    std::string adapter_id;
    AdapterStatus status = AdapterStatus::Absent;
};

struct ModelCapabilities {
    static constexpr uint32_t SCHEMA_VERSION = 1;

    uint32_t schema_version = SCHEMA_VERSION;
    std::string text_architecture;
    bool multimodal = false;
    std::vector<ModalityCapability> modalities;
    std::vector<std::string> diagnostics;

    const ModalityCapability* find(ModelModality modality) const;
};

// A runner registers one rule. Adding another multimodal architecture does not
// change the probe protocol or HexOS: only this registry and the new adapter.
struct ModalityAdapterRule {
    const char* text_architecture;
    ModelModality modality;
    const char* modality_architecture;
    const char* adapter_id;
};

const char* model_modality_name(ModelModality modality);
const char* adapter_status_name(AdapterStatus status);

const ModalityAdapterRule* resolve_modality_adapter(
    const std::string& text_architecture,
    ModelModality modality,
    const std::string& modality_architecture);

ModelCapabilities inspect_model_capabilities(const HnfLoader& loader);

} // namespace helios
