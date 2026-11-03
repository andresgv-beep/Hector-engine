// src/gemma4_split.hpp
// ============================================================================
// GEMMA 4 EN DOS GPUS — capas enteras en cada tarjeta
// ============================================================================
//
// Un token va hacia delante en serie, así que la segunda GPU no acelera nada: da
// memoria. Por eso se reparten capas enteras y no expertos sueltos: son dos cruces
// por token en vez de dos por capa, y cada tarjeta conserva su CUDA Graph.
// El embedding y la cabeza viven en la principal.

#pragma once

#include "command.hpp"
#include "engine.hpp"
#include "graph_builder.hpp"
#include "hnf_loader.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace helios {

struct Gemma4SplitPlan {
    uint32_t boundary = 0;           // [0, boundary) en la principal
    uint64_t primary_bytes = 0;
    uint64_t secondary_bytes = 0;

    bool split(size_t layers) const { return boundary < layers; }
};

// Llena primero la principal, que es la rápida. Sin sitio en las dos, no hay plan.
std::optional<Gemma4SplitPlan> plan_gemma4_split(
    const std::vector<uint64_t>& layer_bytes, uint64_t common_bytes,
    uint64_t primary_free, uint64_t secondary_free);

// Dónde va cada cosa al abrir un Gemma 4. Tres estados distintos, no un 0 que valga para
// «cabe en la principal» y para «no cabe» (auditoría del 2026-11-03).
enum class Gemma4Fit { Single, Split, NoFit };
enum class Gemma4VisionPlace { None, Primary, Secondary };

struct Gemma4Placement {
    Gemma4Fit fit = Gemma4Fit::NoFit;
    uint32_t boundary = 0;          // Split: capas [0, boundary) en la principal; 0 es válido
    Gemma4VisionPlace vision = Gemma4VisionPlace::None;
    bool vision_dropped = false;    // el HNF trae visión, pero no cabe junto al texto
    uint64_t primary_bytes = 0, secondary_bytes = 0;
};

// El plan, sin CUDA. `vision_bytes` son los pesos más la memoria de trabajo que declara el
// HNF (0 si no trae visión); `secondary_free` es 0 con una sola GPU.
//  - Texto y torre caben en la principal: todo allí.
//  - Si no, con dos GPUs se reparte y la torre va a la secundaria. Si el texto cabría solo en
//    la principal, la última capa pasa a la secundaria para no tener un reparto vacío.
//  - Si la torre no cabe en ningún sitio, se abre solo el texto (vision_dropped).
//  - Si ni el texto cabe: NoFit, antes de cargar nada.
Gemma4Placement plan_gemma4_placement(
    const std::vector<uint64_t>& layer_bytes, uint64_t common_bytes, uint64_t vision_bytes,
    uint64_t primary_free, uint64_t secondary_free);

// El plan con la VRAM libre de verdad (GPU 0 principal, GPU 1 secundaria si existe), con
// pesos más KV hasta `max_seq`.
Gemma4Placement gemma4_placement(const HnfLoader& loader, uint32_t max_seq, uint32_t prefill_tokens);

// Registra solo los tensores de ese tramo: los comunes viven en la principal.
bool gemma4_tensor_in_span(const TensorEntry& tensor, uint32_t first_layer, uint32_t end_layer,
                           bool common);

// detect_architecture necesita ver todas las capas, y aquí cada motor tiene las suyas.
ArchDescriptor gemma4_split_arch(uint32_t layers);

class Gemma4SplitForward {
public:
    // Las tres partes de un forward: entrada y primeras capas, resto y cabeza.
    struct Parts {
        const CommandBuffer& first;
        const CommandBuffer& rest;
        const CommandBuffer& head;
    };

    Gemma4SplitForward(Engine& primary, Engine& secondary, const std::string& hidden);
    ~Gemma4SplitForward();
    Gemma4SplitForward(const Gemma4SplitForward&) = delete;
    Gemma4SplitForward& operator=(const Gemma4SplitForward&) = delete;

    void update_device_cache_pos(int32_t cache_pos, int32_t seq_len);
    void run(const Parts& parts);
    void replay(const Parts& parts);
    bool graphs_ready() const;
    void cross(Engine& from, Engine& to);

private:
    Engine& primary_;
    Engine& secondary_;
    std::string hidden_;
    void* host_ = nullptr;
    size_t capacity_ = 0;
};

}  // namespace helios
