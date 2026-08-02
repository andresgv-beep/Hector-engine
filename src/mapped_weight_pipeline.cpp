#include "mapped_weight_pipeline.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <utility>

namespace helios {

MappedWeightPipeline::MappedWeightPipeline(
    HnfLoader& loader, Engine& engine, BlockID block,
    cudaStream_t compute_stream)
    : loader_(loader), engine_(engine), block_(block),
      compute_stream_(compute_stream) {
    const char* value = std::getenv("HELIOS_MMAP_DOUBLE_BUFFER");
    double_buffer_requested_ = value && value[0] == '1';
}

MappedWeightPipeline::~MappedWeightPipeline() {
    release();
}

bool MappedWeightPipeline::fail(
    const std::string& message, std::string* error) {
    loader_.last_error_ = message;
    if (error) *error = message;
    return false;
}

bool MappedWeightPipeline::collect_phase(
    const std::string& prefix, PhaseLayout& layout, std::string* error) {
    layout = PhaseLayout{};
    layout.prefix = prefix;
    const std::string target = block_name(block_);
    std::vector<const TensorEntry*> selected;
    uint64_t first_offset = std::numeric_limits<uint64_t>::max();
    uint64_t last_offset = 0;
    const TensorEntry* first_entry = nullptr;
    bool saw_mapped = false;
    bool saw_device = false;

    for (const TensorEntry& entry : loader_.tensors_) {
        if (entry.block != target || entry.name.rfind(prefix, 0) != 0) continue;
        TensorInfo* tensor = engine_.tensors().get(entry.name);
        if (!tensor || !tensor->ptr) {
            return fail("phase tensor is absent: " + entry.name, error);
        }
        saw_mapped = saw_mapped || tensor->file_mapped;
        saw_device = saw_device || !tensor->file_mapped;
        selected.push_back(&entry);
        if (entry.offset < first_offset) {
            first_offset = entry.offset;
            first_entry = &entry;
        }
        last_offset = std::max(last_offset, entry.offset + entry.size);
    }

    // A resident block needs no pipeline work. Missing prefixes remain the
    // adapter's contract error when it asks the registry for its tensors.
    if (selected.empty() || !saw_mapped) return true;
    if (saw_device) {
        return fail("mapped phase mixes storage types: " + prefix, error);
    }
    if (!first_entry || last_offset <= first_offset ||
        last_offset - first_offset > std::numeric_limits<size_t>::max()) {
        return fail("invalid mapped phase range: " + prefix, error);
    }

    TensorInfo* first_tensor = engine_.tensors().get(first_entry->name);
    if (!first_tensor || !first_tensor->file_mapped) {
        return fail("mapped phase lost its first tensor: " + prefix, error);
    }
    layout.source = first_tensor->ptr;
    layout.bytes = static_cast<size_t>(last_offset - first_offset);
    layout.mapped = true;
    layout.tensors.reserve(selected.size());
    for (const TensorEntry* entry : selected) {
        TensorInfo* tensor = engine_.tensors().get(entry->name);
        const size_t relative =
            static_cast<size_t>(entry->offset - first_offset);
        if (tensor->ptr != static_cast<uint8_t*>(layout.source) + relative) {
            return fail("mapped phase is not one contiguous file range: " +
                        prefix, error);
        }
        layout.tensors.push_back(PhaseTensor{
            entry->name,
            tensor->ptr,
            relative});
    }
    return true;
}

bool MappedWeightPipeline::ensure_double_resources(std::string* error) {
    if (copy_stream_) return true;
    cudaError_t status = cudaStreamCreateWithFlags(
        &copy_stream_, cudaStreamNonBlocking);
    if (status != cudaSuccess) {
        return fail("cannot create mapped phase copy stream: " +
                    std::string(cudaGetErrorString(status)), error);
    }
    for (Slot& slot : slots_) {
        status = cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming);
        if (status == cudaSuccess) {
            status = cudaEventCreateWithFlags(
                &slot.released, cudaEventDisableTiming);
        }
        if (status != cudaSuccess) {
            return fail("cannot create mapped phase events: " +
                        std::string(cudaGetErrorString(status)), error);
        }
    }
    double_buffer_active_ = true;
    return true;
}

bool MappedWeightPipeline::ensure_slot(
    int slot_index, size_t bytes, std::string* error) {
    Slot& slot = slots_[slot_index];
    if (slot.ptr && slot.capacity >= bytes) return true;
    if (slot.has_release) {
        const cudaError_t wait = cudaEventSynchronize(slot.released);
        if (wait != cudaSuccess) {
            return fail("cannot wait for mapped phase slot: " +
                        std::string(cudaGetErrorString(wait)), error);
        }
    }
    if (slot.ptr) cudaFree(slot.ptr);
    slot.ptr = nullptr;
    slot.capacity = 0;
    const cudaError_t allocation = cudaMalloc(&slot.ptr, bytes);
    if (allocation != cudaSuccess) {
        return fail("cannot allocate mapped phase slot: " +
                    std::string(cudaGetErrorString(allocation)), error);
    }
    slot.capacity = bytes;
    slot.has_release = false;
    return true;
}

bool MappedWeightPipeline::bind_phase(
    const PhaseLayout& layout, int slot_index, std::string* error) {
    active_originals_.clear();
    active_originals_.reserve(layout.tensors.size());
    for (const PhaseTensor& phase_tensor : layout.tensors) {
        TensorInfo* tensor = engine_.tensors().get(phase_tensor.name);
        if (!tensor || !tensor->file_mapped || !tensor->ptr) {
            for (const auto& original : active_originals_) {
                TensorInfo* restore = engine_.tensors().get(original.first);
                if (restore) restore->ptr = original.second;
            }
            active_originals_.clear();
            return fail("cannot bind mapped phase tensor: " +
                        phase_tensor.name, error);
        }
        active_originals_.emplace_back(phase_tensor.name, tensor->ptr);
        tensor->ptr = static_cast<uint8_t*>(slots_[slot_index].ptr) +
                      phase_tensor.relative_offset;
    }
    return true;
}

bool MappedWeightPipeline::begin_phase(
    const std::string& prefix, std::string* error) {
    if (phase_active_) return fail("a mapped phase is already active", error);
    if (!double_buffer_requested_) {
        if (!loader_.stage_mapped_prefix(
                block_, engine_, prefix, compute_stream_)) {
            return fail(loader_.last_error_, error);
        }
        phase_active_ = true;
        phase_is_mapped_ = !loader_.block_states_[block_].staged_tensor_ptrs.empty();
        return true;
    }

    PhaseLayout layout;
    if (!collect_phase(prefix, layout, error)) return false;
    if (!layout.mapped) {
        phase_active_ = true;
        phase_is_mapped_ = false;
        return true;
    }
    if (!ensure_double_resources(error)) return false;

    int slot_index = 0;
    if (prefetched_slot_ >= 0) {
        if (prefetched_.prefix != prefix) {
            return fail("prefetched phase order mismatch: expected " +
                        prefetched_.prefix + ", got " + prefix, error);
        }
        slot_index = prefetched_slot_;
        const cudaError_t wait = cudaStreamWaitEvent(
            compute_stream_, slots_[slot_index].ready, 0);
        if (wait != cudaSuccess) {
            return fail("cannot wait for prefetched phase: " +
                        std::string(cudaGetErrorString(wait)), error);
        }
        layout = std::move(prefetched_);
        prefetched_ = PhaseLayout{};
        prefetched_slot_ = -1;
    } else {
        if (!ensure_slot(slot_index, layout.bytes, error)) return false;
        const cudaError_t copy = cudaMemcpyAsync(
            slots_[slot_index].ptr, layout.source, layout.bytes,
            cudaMemcpyDefault, compute_stream_);
        if (copy != cudaSuccess) {
            return fail("cannot copy initial mapped phase: " +
                        std::string(cudaGetErrorString(copy)), error);
        }
    }

    if (!bind_phase(layout, slot_index, error)) return false;
    active_slot_ = slot_index;
    phase_active_ = true;
    phase_is_mapped_ = true;
    active_release_recorded_ = false;
    return true;
}

bool MappedWeightPipeline::record_active_release(std::string* error) {
    if (!phase_is_mapped_ || active_slot_ < 0 ||
        active_release_recorded_) return true;
    Slot& slot = slots_[active_slot_];
    const cudaError_t status = cudaEventRecord(slot.released, compute_stream_);
    if (status != cudaSuccess) {
        return fail("cannot record mapped phase release: " +
                    std::string(cudaGetErrorString(status)), error);
    }
    slot.has_release = true;
    active_release_recorded_ = true;
    return true;
}

bool MappedWeightPipeline::prefetch_phase(
    const std::string& prefix, std::string* error) {
    if (!phase_active_) return fail("prefetch requires an active phase", error);
    if (!double_buffer_requested_ || !phase_is_mapped_) return true;
    if (prefetched_slot_ >= 0) {
        return fail("a mapped phase is already prefetched", error);
    }
    if (!record_active_release(error)) return false;

    PhaseLayout layout;
    if (!collect_phase(prefix, layout, error)) return false;
    if (!layout.mapped) return true;
    for (const PhaseTensor& next : layout.tensors) {
        const auto overlap = std::find_if(
            active_originals_.begin(), active_originals_.end(),
            [&](const auto& active) { return active.first == next.name; });
        if (overlap != active_originals_.end()) {
            return fail("mapped phases overlap at tensor: " + next.name,
                        error);
        }
    }
    const int slot_index = 1 - active_slot_;
    if (!ensure_slot(slot_index, layout.bytes, error)) return false;
    Slot& slot = slots_[slot_index];
    if (slot.has_release) {
        const cudaError_t wait = cudaStreamWaitEvent(
            copy_stream_, slot.released, 0);
        if (wait != cudaSuccess) {
            return fail("cannot wait to reuse mapped phase slot: " +
                        std::string(cudaGetErrorString(wait)), error);
        }
    }
    const cudaError_t copy = cudaMemcpyAsync(
        slot.ptr, layout.source, layout.bytes, cudaMemcpyDefault, copy_stream_);
    if (copy != cudaSuccess) {
        return fail("cannot prefetch mapped phase: " +
                    std::string(cudaGetErrorString(copy)), error);
    }
    const cudaError_t ready = cudaEventRecord(slot.ready, copy_stream_);
    if (ready != cudaSuccess) {
        return fail("cannot record prefetched phase readiness: " +
                    std::string(cudaGetErrorString(ready)), error);
    }
    prefetched_ = std::move(layout);
    prefetched_slot_ = slot_index;
    return true;
}

bool MappedWeightPipeline::end_phase(std::string* error) {
    if (!phase_active_) return true;
    bool ok = true;
    if (!double_buffer_requested_) {
        ok = loader_.unstage_mapped_prefix(block_, engine_);
        if (!ok && error) *error = loader_.last_error_;
    } else if (phase_is_mapped_) {
        if (!record_active_release(error)) ok = false;
        for (const auto& original : active_originals_) {
            TensorInfo* tensor = engine_.tensors().get(original.first);
            if (!tensor) {
                ok = fail("cannot restore mapped phase tensor: " +
                          original.first, error);
                continue;
            }
            tensor->ptr = original.second;
        }
        active_originals_.clear();
    }
    phase_active_ = false;
    phase_is_mapped_ = false;
    active_release_recorded_ = false;
    active_slot_ = -1;
    return ok;
}

void MappedWeightPipeline::release() {
    if (phase_active_) (void)end_phase(nullptr);
    if (!double_buffer_requested_) {
        (void)loader_.release_mapped_staging(block_, engine_);
        return;
    }
    if (copy_stream_) cudaStreamSynchronize(copy_stream_);
    for (Slot& slot : slots_) {
        if (slot.has_release && slot.released) {
            cudaEventSynchronize(slot.released);
        }
        if (slot.ptr) cudaFree(slot.ptr);
        if (slot.ready) cudaEventDestroy(slot.ready);
        if (slot.released) cudaEventDestroy(slot.released);
        slot = Slot{};
    }
    if (copy_stream_) cudaStreamDestroy(copy_stream_);
    copy_stream_ = nullptr;
    prefetched_ = PhaseLayout{};
    prefetched_slot_ = -1;
    double_buffer_active_ = false;
}

MappedWeightPhase::MappedWeightPhase(
    MappedWeightPipeline& pipeline, const std::string& prefix,
    std::string* error)
    : pipeline_(pipeline), active_(pipeline_.begin_phase(prefix, error)) {}

MappedWeightPhase::~MappedWeightPhase() {
    if (active_) (void)pipeline_.end_phase(nullptr);
}

bool MappedWeightPhase::prefetch(
    const std::string& next_prefix, std::string* error) {
    return active_ && pipeline_.prefetch_phase(next_prefix, error);
}

bool MappedWeightPhase::close(std::string* error) {
    if (!active_) return true;
    active_ = false;
    return pipeline_.end_phase(error);
}

} // namespace helios
