#pragma once

#include "hnf_loader.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace helios {

struct Gemma4ValidationReport {
    std::vector<std::string> errors;
    std::unordered_map<std::string, size_t> dtype_counts;

    size_t tensor_count = 0;
    uint64_t tensor_bytes = 0;
    uint64_t ple_embedding_bytes = 0;
    uint64_t largest_tensor_bytes = 0;

    uint32_t budget_batch_size = 1;
    uint32_t budget_sequence_length = 0;
    uint64_t core_scratch_bytes = 0;
    uint64_t ple_workspace_bytes = 0;
    uint64_t kv_cache_upper_bound_bytes = 0;

    bool ok() const { return errors.empty(); }
};

// Validates the complete text tensor contract without loading weights. Scratch
// is the current graph's uniform-max core allocation. PLE workspace accounts
// separately for the packed token/result and contextual projection buffers.
Gemma4ValidationReport validate_gemma4_tensors(
    const HnfLoader& loader,
    uint32_t budget_batch_size = 1,
    uint32_t budget_sequence_length = 512);

} // namespace helios
