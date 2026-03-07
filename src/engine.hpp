// engine.hpp
// HELIOS ENGINE - Core Engine
// ============================
// Ejecuta CommandBuffers usando TensorRegistry y MemoryPool.
//
// Diseño EXTENSIBLE:
//   - Engine posee TensorRegistry (pesos del modelo)
//   - Engine posee MemoryPool (activaciones temporales)
//   - Engine ejecuta CommandBuffer command por command
//   - Kernels registrados dinámicamente por OpTypeID
//   - inputs[] resueltos a vector de TensorInfo*
//

#pragma once

#include "tensor.hpp"
#include "memory.hpp"
#include "command.hpp"

#include <functional>
#include <unordered_map>
#include <vector>
#include <chrono>

namespace helios {

// ============================================================================
// EXECUTION CONTEXT (passed to kernels)
// ============================================================================

struct ExecContext {
    TensorRegistry& tensors;
    MemoryPool& scratch;
    cudaStream_t stream;
    
    // Registry pointer for multi-output ops
    TensorRegistry* registry;
    
    // Resolved tensors for current command
    TensorInfo* output;                    // Output tensor (may be null for some ops)
    std::vector<TensorInfo*> inputs;       // Input tensors (variable count)
    
    // Access helpers
    TensorInfo* in(size_t i) const { 
        return (i < inputs.size()) ? inputs[i] : nullptr; 
    }
    size_t num_inputs() const { return inputs.size(); }
};

// ============================================================================
// KERNEL FUNCTION SIGNATURE
// ============================================================================

// All kernels have same signature for uniformity
using KernelFn = std::function<void(ExecContext& ctx, const Command& cmd)>;

// ============================================================================
// ENGINE CONFIGURATION
// ============================================================================

struct EngineConfig {
    // Memory pool settings
    MemoryPoolConfig scratch_pool;
    
    // Execution settings
    bool sync_after_each_op = false;   // Debug: sync after each kernel
    bool check_errors = true;          // Check CUDA errors
    bool enable_profiling = false;     // Measure kernel times
    
    // Default stream (nullptr = default CUDA stream)
    cudaStream_t stream = nullptr;
};

// ============================================================================
// PROFILING INFO
// ============================================================================

struct KernelProfile {
    OpTypeID op;
    std::string label;
    double time_ms;
    size_t call_count;
};

// ============================================================================
// ENGINE
// ============================================================================

class Engine {
public:
    explicit Engine(const EngineConfig& config = EngineConfig{});
    ~Engine();
    
    // No copy
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    
    // Move OK
    Engine(Engine&&) = default;
    Engine& operator=(Engine&&) = default;
    
    // ========================================
    // KERNEL REGISTRATION
    // ========================================
    
    // Register a kernel implementation for an OpTypeID
    void register_kernel(OpTypeID op, KernelFn fn);
    
    // Register by name (convenience)
    void register_kernel(const std::string& op_name, KernelFn fn);
    
    // Check if kernel is registered
    bool has_kernel(OpTypeID op) const;
    
    // ========================================
    // TENSOR MANAGEMENT
    // ========================================
    
    // Access tensor registry (for loading weights)
    TensorRegistry& tensors() { return tensors_; }
    const TensorRegistry& tensors() const { return tensors_; }
    
    // ========================================
    // EXECUTION
    // ========================================
    
    // Execute entire command buffer
    void execute(const CommandBuffer& commands);
    
    // Execute single command
    void execute_command(const Command& cmd);
    
    // CUDA Graph execution: capture on first call, replay thereafter
    // Returns true if graph was used (decode), false if fell back to normal execute
    void execute_graphed(const CommandBuffer& commands);
    
    // TRUE capture-once graph: uses device-pointer kernels
    // First call: captures graph. Subsequent calls: replay only (no re-capture).
    // Requires kernels that read cache_pos from device memory.
    void execute_graph_replay(const CommandBuffer& commands);
    
    // Invalidate captured CUDA graph (call when graph topology changes)
    void invalidate_graph();
    
    // ========================================
    // DEVICE CACHE POSITION (for CUDA Graphs)
    // ========================================
    
    // Allocate device memory for cache_pos and total_seq
    void alloc_device_cache_pos();
    
    // Update cache_pos in device memory (4 bytes async copy)
    void update_device_cache_pos(int32_t cache_pos, int32_t seq_len);
    
    // Get device pointers (for kernel registration)
    int32_t* device_cache_pos() const { return d_cache_pos_; }
    int32_t* device_total_seq() const { return d_total_seq_; }
    bool has_device_cache_pos() const { return d_cache_pos_ != nullptr; }
    
    // Is graph captured and ready for replay?
    bool graph_ready() const { return graph_valid_; }
    
    // Reset scratch memory (call between inferences)
    void reset_scratch();
    
    // Synchronize (wait for all GPU work to complete)
    void sync();
    
    // ========================================
    // PROFILING
    // ========================================
    
    // Get profiling results (if enabled)
    std::vector<KernelProfile> get_profiles() const;
    
    // Clear profiling data
    void clear_profiles();
    
    // Print profiling summary
    void print_profile_summary() const;
    
    // ========================================
    // STATE
    // ========================================
    
    const EngineConfig& config() const { return config_; }
    MemoryPool& scratch() { return scratch_; }
    
private:
    EngineConfig config_;
    TensorRegistry tensors_;
    MemoryPool scratch_;
    
    // Registered kernels by OpTypeID
    std::unordered_map<OpTypeID, KernelFn> kernels_;
    
    // Profiling data
    std::unordered_map<OpTypeID, KernelProfile> profiles_;
    
    // CUDA Graph state
    cudaGraph_t captured_graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    bool graph_valid_ = false;
    
    // Device-side cache position (for capture-once CUDA Graphs)
    int32_t* d_cache_pos_ = nullptr;    // cache_pos on device
    int32_t* d_total_seq_ = nullptr;    // cache_pos + seq_len on device
    
    // Resolve tensor names to TensorInfo pointers
    void resolve_tensors(ExecContext& ctx, const Command& cmd);
    
    // Record profiling info
    void record_profile(OpTypeID op, const std::string& label, double time_ms);
};

// ============================================================================
// STUB KERNELS (for testing without real CUDA implementations)
// ============================================================================

// Register stub kernels that just print what they would do
void register_stub_kernels(Engine& engine);

} // namespace helios
