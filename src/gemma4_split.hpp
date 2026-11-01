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

private:
    void cross(Engine& from, Engine& to);

    Engine& primary_;
    Engine& secondary_;
    std::string hidden_;
    void* host_ = nullptr;
    size_t capacity_ = 0;
};

}  // namespace helios
