#pragma once

#include "hnf_loader.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace helios {

// Generic execution-phase pipeline for file-backed model blocks.
//
// A model adapter only names the current and next tensor prefixes. This class
// owns all storage details: the V7 single window or, when explicitly enabled,
// two windows plus the copy stream and ready/release events.
class MappedWeightPipeline {
public:
    MappedWeightPipeline(HnfLoader& loader, Engine& engine, BlockID block,
                         cudaStream_t compute_stream);
    ~MappedWeightPipeline();

    MappedWeightPipeline(const MappedWeightPipeline&) = delete;
    MappedWeightPipeline& operator=(const MappedWeightPipeline&) = delete;
    MappedWeightPipeline(MappedWeightPipeline&&) = delete;
    MappedWeightPipeline& operator=(MappedWeightPipeline&&) = delete;

    bool double_buffer_requested() const { return double_buffer_requested_; }
    bool double_buffer_active() const { return double_buffer_active_; }

private:
    friend class MappedWeightPhase;

    struct PhaseTensor {
        std::string name;
        void* source = nullptr;
        size_t relative_offset = 0;
    };

    struct PhaseLayout {
        std::string prefix;
        void* source = nullptr;
        size_t bytes = 0;
        bool mapped = false;
        std::vector<PhaseTensor> tensors;
    };

    struct Slot {
        void* ptr = nullptr;
        size_t capacity = 0;
        cudaEvent_t ready = nullptr;
        cudaEvent_t released = nullptr;
        bool has_release = false;
    };

    bool begin_phase(const std::string& prefix, std::string* error);
    bool prefetch_phase(const std::string& prefix, std::string* error);
    bool end_phase(std::string* error);
    bool collect_phase(const std::string& prefix, PhaseLayout& layout,
                       std::string* error);
    bool ensure_double_resources(std::string* error);
    bool ensure_slot(int slot, size_t bytes, std::string* error);
    bool bind_phase(const PhaseLayout& layout, int slot, std::string* error);
    bool record_active_release(std::string* error);
    bool fail(const std::string& message, std::string* error);
    void release();

    HnfLoader& loader_;
    Engine& engine_;
    BlockID block_;
    cudaStream_t compute_stream_;
    bool double_buffer_requested_ = false;
    bool double_buffer_active_ = false;
    bool phase_active_ = false;
    bool phase_is_mapped_ = false;
    bool active_release_recorded_ = false;
    int active_slot_ = -1;
    int prefetched_slot_ = -1;
    cudaStream_t copy_stream_ = nullptr;
    Slot slots_[2];
    PhaseLayout prefetched_;
    std::vector<std::pair<std::string, void*>> active_originals_;
};

// RAII boundary used by architecture adapters. Keeping the guard local makes
// every early return restore registry pointers automatically.
class MappedWeightPhase {
public:
    MappedWeightPhase(MappedWeightPipeline& pipeline,
                      const std::string& prefix,
                      std::string* error = nullptr);
    ~MappedWeightPhase();

    MappedWeightPhase(const MappedWeightPhase&) = delete;
    MappedWeightPhase& operator=(const MappedWeightPhase&) = delete;

    explicit operator bool() const { return active_; }
    bool prefetch(const std::string& next_prefix,
                  std::string* error = nullptr);
    bool close(std::string* error = nullptr);

private:
    MappedWeightPipeline& pipeline_;
    bool active_ = false;
};

} // namespace helios
