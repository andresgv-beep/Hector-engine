// src/gemma4_split.cpp
#include "gemma4_split.hpp"

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
