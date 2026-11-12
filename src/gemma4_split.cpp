// src/gemma4_split.cpp
#include "gemma4_split.hpp"

#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace helios {

namespace {

void check(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("Gemma 4 split ") + where + ": " +
                                 cudaGetErrorString(error));
    }
}

constexpr uint32_t PRIMARY_INPUT_GRAPH = 0, PRIMARY_HEAD_GRAPH = 1, SECONDARY_GRAPH = 0;

// Lo que no son pesos ni KV: scratch, cuBLAS, buffers de prefill y el escritorio.
constexpr uint64_t VRAM_RESERVE = 1400ull << 20;

uint64_t usable_vram(int device) {
    check(cudaSetDevice(device), "elegir GPU");
    size_t free_bytes = 0, total = 0;
    check(cudaMemGetInfo(&free_bytes, &total), "leer memoria libre");
    return free_bytes > VRAM_RESERVE ? free_bytes - VRAM_RESERVE : 0;
}

int layer_index(const std::string& name) {
    const size_t mark = name.find(".layer");
    return mark == std::string::npos ? -1 : std::atoi(name.c_str() + mark + 6);
}

}  // namespace

std::optional<Gemma4SplitPlan> plan_gemma4_split(
    const std::vector<uint64_t>& layer_bytes, uint64_t common_bytes,
    uint64_t primary_free, uint64_t secondary_free) {
    if (layer_bytes.empty() || common_bytes > primary_free) return std::nullopt;

    Gemma4SplitPlan plan;
    plan.primary_bytes = common_bytes;
    size_t layer = 0;
    while (layer < layer_bytes.size() &&
           layer_bytes[layer] <= primary_free - plan.primary_bytes) {
        plan.primary_bytes += layer_bytes[layer++];
    }
    plan.boundary = static_cast<uint32_t>(layer);
    for (; layer < layer_bytes.size(); ++layer) {
        if (layer_bytes[layer] > secondary_free - plan.secondary_bytes) return std::nullopt;
        plan.secondary_bytes += layer_bytes[layer];
    }
    return plan;
}

Gemma4Placement plan_gemma4_placement(
    const std::vector<uint64_t>& layer_bytes, uint64_t common_bytes, uint64_t vision_bytes,
    uint64_t primary_free, uint64_t secondary_free, bool split_text) {
    Gemma4Placement out;
    uint64_t text = common_bytes;
    for (uint64_t b : layer_bytes) text += b;
    const size_t layers = layer_bytes.size();
    const auto split = [&](uint64_t secondary_budget) -> std::optional<Gemma4SplitPlan> {
        if (!split_text) return std::nullopt;
        auto plan = plan_gemma4_split(layer_bytes, common_bytes, primary_free, secondary_budget);
        if (plan && !plan->split(layers)) return std::nullopt;
        return plan;
    };
    const auto take = [&](const Gemma4SplitPlan& plan, Gemma4VisionPlace vision) {
        out.fit = Gemma4Fit::Split;
        out.boundary = plan.boundary;
        out.vision = vision;
        out.primary_bytes = plan.primary_bytes;
        out.secondary_bytes = plan.secondary_bytes + (vision == Gemma4VisionPlace::Secondary ? vision_bytes : 0);
    };
    if (text <= primary_free && (vision_bytes == 0 || vision_bytes <= primary_free - text)) {
        out.fit = Gemma4Fit::Single;
        out.vision = vision_bytes ? Gemma4VisionPlace::Primary : Gemma4VisionPlace::None;
        out.primary_bytes = text + vision_bytes;
        return out;
    }
    // El texto cabe entero en la principal y la torre sola en la secundaria: cada cosa en su
    // tarjeta, sin mover capas (el decode no cruza de GPU).
    if (vision_bytes && text <= primary_free && vision_bytes <= secondary_free) {
        out.fit = Gemma4Fit::Single;
        out.vision = Gemma4VisionPlace::Secondary;
        out.primary_bytes = text;
        out.secondary_bytes = vision_bytes;
        return out;
    }
    if (vision_bytes && vision_bytes <= secondary_free) {
        if (const auto plan = split(secondary_free - vision_bytes)) {
            take(*plan, Gemma4VisionPlace::Secondary);
            return out;
        }
    }
    // Sin sitio para la torre: solo texto, en la principal si cabe y si no repartido.
    out.vision_dropped = vision_bytes != 0;
    if (text <= primary_free) {
        out.fit = Gemma4Fit::Single;
        out.primary_bytes = text;
        return out;
    }
    if (const auto plan = split(secondary_free)) take(*plan, Gemma4VisionPlace::None);
    return out;
}

Gemma4Placement gemma4_placement(const HnfLoader& loader, uint32_t max_seq, uint32_t prefill_tokens) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess) devices = 0;
    Gemma4Placement single;
    single.fit = Gemma4Fit::Single;
    if (devices < 1 || !loader.has_gemma4_config()) return single;   // fuera de Gemma 4, como siempre
    const ModelConfig& config = loader.config();
    const Gemma4Config& gemma = loader.gemma4_config();
    std::vector<uint64_t> layer_bytes(gemma.layers.size(), 0);
    uint64_t common_bytes = 0;
    for (const TensorEntry& tensor : loader.tensors()) {
        if (tensor.block != "text_model") continue;   // vision.layerN no es la capa N
        if (loader.tensor_lives_in_host_ram(tensor)) continue;   // en RAM mapeada: no gasta VRAM
        const int layer = layer_index(tensor.name);
        if (layer >= int(layer_bytes.size())) return Gemma4Placement{};
        (layer < 0 ? common_bytes : layer_bytes[size_t(layer)]) += tensor.size;
    }
    // Las capas con KV compartido no tienen KV propio: leen el de una capa anterior.
    const size_t first_shared = gemma.num_kv_shared_layers < layer_bytes.size()
        ? layer_bytes.size() - gemma.num_kv_shared_layers : layer_bytes.size();
    for (size_t i = 0; i < first_shared; ++i) {
        const Gemma4LayerConfig& layer = gemma.layers[i];
        const uint64_t slots = layer.is_global_attention() ? max_seq : layer.sliding_window + prefill_tokens;
        layer_bytes[i] += 2 * sizeof(uint16_t) * slots * layer.kv_heads_or(config.num_key_value_heads()) * layer.head_dim;
    }
    const uint64_t primary_free = usable_vram(0);
    const uint64_t secondary_free = devices >= 2 ? usable_vram(1) : 0;
    check(cudaSetDevice(0), "elegir GPU");
    return plan_gemma4_placement(layer_bytes, common_bytes, loader.block_vram_bytes(BLOCK_VISION),
                                 primary_free, secondary_free, gemma.num_kv_shared_layers == 0);
}

bool gemma4_tensor_in_span(const TensorEntry& tensor, uint32_t first_layer, uint32_t end_layer, bool common) {
    // El reparto es de capas de texto; la visión se carga aparte y entera.
    if (tensor.block != "text_model") return true;
    const int layer = layer_index(tensor.name);
    return layer < 0 ? common : layer >= int(first_layer) && layer < int(end_layer);
}

ArchDescriptor gemma4_split_arch(uint32_t layers) {
    ArchDescriptor arch;
    arch.prefix = "text";
    arch.num_layers = layers;
    arch.embedding_name = "token_embedding.weight";
    arch.final_norm_name = "final_norm.weight";
    return arch;
}

Gemma4SplitForward::Gemma4SplitForward(Engine& primary, Engine& secondary,
                                       const std::string& hidden)
    : primary_(primary), secondary_(secondary), hidden_(hidden) {
    primary_.activate();
    capacity_ = primary_.tensors().at(hidden_).size_bytes;
    if (secondary_.tensors().at(hidden_).size_bytes != capacity_) {
        throw std::invalid_argument("Gemma 4 split: hidden differs between GPUs");
    }
    // Sin P2P entre tarjetas de generaciones distintas, cruza por RAM anclada.
    check(cudaHostAlloc(&host_, capacity_, cudaHostAllocPortable), "reservar el cruce");
}

Gemma4SplitForward::~Gemma4SplitForward() {
    if (host_) cudaFreeHost(host_);
}

void Gemma4SplitForward::update_device_cache_pos(int32_t cache_pos, int32_t seq_len) {
    secondary_.activate();
    secondary_.update_device_cache_pos(cache_pos, seq_len);
    primary_.activate();
    primary_.update_device_cache_pos(cache_pos, seq_len);
}

void Gemma4SplitForward::cross(Engine& from, Engine& to) {
    const TensorInfo& source = from.tensors().at(hidden_);
    const size_t bytes = source.numel() * sizeof(uint16_t);
    if (bytes > capacity_) throw std::logic_error("Gemma 4 split: hidden exceeds its buffer");
    check(cudaMemcpyAsync(host_, source.ptr, bytes, cudaMemcpyDeviceToHost,
                          from.config().stream), "bajar hidden");
    check(cudaStreamSynchronize(from.config().stream), "esperar a la GPU de origen");
    to.activate();
    check(cudaMemcpyAsync(to.tensors().at(hidden_).ptr, host_, bytes, cudaMemcpyHostToDevice,
                          to.config().stream), "subir hidden");
}

void Gemma4SplitForward::run(const Parts& parts) {
    primary_.activate();
    primary_.execute(parts.first);
    cross(primary_, secondary_);
    secondary_.execute(parts.rest);
    cross(secondary_, primary_);
    primary_.execute(parts.head);
}

void Gemma4SplitForward::replay(const Parts& parts) {
    primary_.activate();
    primary_.execute_graph_replay(parts.first, PRIMARY_INPUT_GRAPH);
    cross(primary_, secondary_);
    secondary_.execute_graph_replay(parts.rest, SECONDARY_GRAPH);
    cross(secondary_, primary_);
    primary_.execute_graph_replay(parts.head, PRIMARY_HEAD_GRAPH);
}

bool Gemma4SplitForward::graphs_ready() const {
    return primary_.graph_ready(PRIMARY_INPUT_GRAPH) && primary_.graph_ready(PRIMARY_HEAD_GRAPH) &&
           secondary_.graph_ready(SECONDARY_GRAPH);
}

}  // namespace helios
