// engine.cpp
// HELIOS ENGINE - Core Engine Implementation
// ==========================================

#include "engine.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>

namespace helios {

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

Engine::Engine(const EngineConfig& config)
    : config_(config)
    , tensors_()
    , scratch_([&config]() {
        MemoryPoolConfig pool_config = config.scratch_pool;
        pool_config.name = "engine_scratch";
        return pool_config;
    }())
{
}

Engine::~Engine() {
    // Cleanup CUDA Graph
    invalidate_graph();
    // Cleanup device cache pos
    if (d_cache_pos_) { cudaFree(d_cache_pos_); d_cache_pos_ = nullptr; }
    if (d_total_seq_) { cudaFree(d_total_seq_); d_total_seq_ = nullptr; }
}

// ============================================================================
// KERNEL REGISTRATION
// ============================================================================

void Engine::register_kernel(OpTypeID op, KernelFn fn) {
    if (op == OP_INVALID) {
        throw std::runtime_error("Cannot register kernel for invalid op");
    }
    kernels_[op] = std::move(fn);
}

void Engine::register_kernel(const std::string& op_name, KernelFn fn) {
    OpTypeID op = OpTypeRegistry::instance().get_id(op_name);
    if (op == OP_INVALID) {
        throw std::runtime_error("Unknown op: " + op_name);
    }
    register_kernel(op, std::move(fn));
}

bool Engine::has_kernel(OpTypeID op) const {
    return kernels_.find(op) != kernels_.end();
}

// ============================================================================
// EXECUTION
// ============================================================================

void Engine::execute(const CommandBuffer& commands) {
    for (const auto& cmd : commands.commands()) {
        execute_command(cmd);
    }
}

void Engine::execute_graphed(const CommandBuffer& commands) {
    // Always capture a new graph (captures current kernel arguments)
    cudaGraph_t new_graph = nullptr;
    
    cudaStreamBeginCapture(config_.stream, cudaStreamCaptureModeGlobal);
    
    for (const auto& cmd : commands.commands()) {
        auto it = kernels_.find(cmd.op);
        if (it == kernels_.end()) {
            cudaStreamEndCapture(config_.stream, &new_graph);
            if (new_graph) cudaGraphDestroy(new_graph);
            execute(commands);
            return;
        }
        
        ExecContext ctx{tensors_, scratch_, config_.stream, &tensors_, nullptr, {}};
        resolve_tensors(ctx, cmd);
        // Only launch kernel — NO profiling, NO sync, NO error checks during capture
        it->second(ctx, cmd);
    }
    
    cudaStreamEndCapture(config_.stream, &new_graph);
    
    if (!new_graph) {
        execute(commands);
        return;
    }
    
    if (graph_valid_ && graph_exec_) {
        // Try to update existing exec with new graph (avoids re-instantiation)
        cudaGraphExecUpdateResultInfo updateResult = {};
        cudaError_t err = cudaGraphExecUpdate(graph_exec_, new_graph, &updateResult);
        
        if (err == cudaSuccess && updateResult.result == cudaGraphExecUpdateSuccess) {
            // Updated in-place — fast path
            cudaGraphDestroy(new_graph);
        } else {
            // Topology changed — re-instantiate
            cudaGraphExecDestroy(graph_exec_);
            cudaGraphInstantiate(&graph_exec_, new_graph, nullptr, nullptr, 0);
            if (captured_graph_) cudaGraphDestroy(captured_graph_);
            captured_graph_ = new_graph;
        }
    } else {
        // First time — instantiate
        cudaGraphInstantiate(&graph_exec_, new_graph, nullptr, nullptr, 0);
        if (captured_graph_) cudaGraphDestroy(captured_graph_);
        captured_graph_ = new_graph;
        graph_valid_ = true;
    }
    
    // Launch
    cudaGraphLaunch(graph_exec_, config_.stream);
}

// ============================================================================
// DEVICE CACHE POSITION
// ============================================================================

void Engine::alloc_device_cache_pos() {
    if (!d_cache_pos_) {
        cudaMalloc(&d_cache_pos_, sizeof(int32_t));
        cudaMalloc(&d_total_seq_, sizeof(int32_t));
        int32_t zero = 0;
        cudaMemcpy(d_cache_pos_, &zero, sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_total_seq_, &zero, sizeof(int32_t), cudaMemcpyHostToDevice);
    }
}

void Engine::update_device_cache_pos(int32_t cache_pos, int32_t seq_len) {
    if (!d_cache_pos_) alloc_device_cache_pos();
    int32_t total = cache_pos + seq_len;
    cudaMemcpyAsync(d_cache_pos_, &cache_pos, sizeof(int32_t), 
                    cudaMemcpyHostToDevice, config_.stream);
    cudaMemcpyAsync(d_total_seq_, &total, sizeof(int32_t),
                    cudaMemcpyHostToDevice, config_.stream);
}

// ============================================================================
// CUDA GRAPH — TRUE CAPTURE-ONCE REPLAY
// ============================================================================

void Engine::execute_graph_replay(const CommandBuffer& commands) {
    if (graph_valid_ && graph_exec_) {
        // ============================================================
        // REPLAY PATH: execute non-capturable ops, then replay graph
        // ============================================================
        
        // Zero-copy splits mutate TensorInfo pointers in CPU.
        // These aren't kernel launches, so they aren't in the graph.
        // We must re-execute them before each replay so downstream
        // kernels see correct pointers. However, since the pointers
        // don't actually change between decode tokens (same fused tensor
        // address), this is just a safety measure.
        for (const auto& cmd : commands.commands()) {
            // SPLIT_QKV and SPLIT_HALF with batch_seq=1 do zero-copy (no kernel)
            auto name = std::string(op_name(cmd.op));
            if (name == "split_qkv" || name == "split_half") {
                auto it = kernels_.find(cmd.op);
                if (it != kernels_.end()) {
                    ExecContext ctx{tensors_, scratch_, config_.stream, &tensors_, nullptr, {}};
                    resolve_tensors(ctx, cmd);
                    it->second(ctx, cmd);
                }
            }
        }
        
        // Replay the captured graph
        cudaGraphLaunch(graph_exec_, config_.stream);
        return;
    }
    
    // ============================================================
    // CAPTURE PATH: first decode token — build the graph
    // ============================================================
    
    // Step 1: Execute non-capturable ops (zero-copy splits) BEFORE capture
    for (const auto& cmd : commands.commands()) {
        auto name = std::string(op_name(cmd.op));
        if (name == "split_qkv" || name == "split_half") {
            auto it = kernels_.find(cmd.op);
            if (it != kernels_.end()) {
                ExecContext ctx{tensors_, scratch_, config_.stream, &tensors_, nullptr, {}};
                resolve_tensors(ctx, cmd);
                it->second(ctx, cmd);
            }
        }
    }
    
    // Sync to ensure splits are done before capture
    cudaStreamSynchronize(config_.stream);
    // Clear any residual CUDA errors
    cudaGetLastError();
    
    // Step 2: Capture only real kernel launches
    cudaGraph_t new_graph = nullptr;
    cudaError_t cap_err = cudaStreamBeginCapture(config_.stream, cudaStreamCaptureModeGlobal);
    if (cap_err != cudaSuccess) {
        // Can't capture — fall back to normal execution
        execute(commands);
        return;
    }
    
    bool capture_ok = true;
    for (const auto& cmd : commands.commands()) {
        auto name = std::string(op_name(cmd.op));
        // Skip splits — already executed above
        if (name == "split_qkv" || name == "split_half") continue;
        
        auto it = kernels_.find(cmd.op);
        if (it == kernels_.end()) {
            capture_ok = false;
            break;
        }
        
        ExecContext ctx{tensors_, scratch_, config_.stream, &tensors_, nullptr, {}};
        resolve_tensors(ctx, cmd);
        it->second(ctx, cmd);
    }
    
    cudaStreamEndCapture(config_.stream, &new_graph);
    
    if (!new_graph || !capture_ok) {
        if (new_graph) cudaGraphDestroy(new_graph);
        // Fall back to normal execution for this token
        execute(commands);
        return;
    }
    
    // Instantiate
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (captured_graph_) cudaGraphDestroy(captured_graph_);
    
    cudaGraphInstantiate(&graph_exec_, new_graph, nullptr, nullptr, 0);
    captured_graph_ = new_graph;
    graph_valid_ = true;
    
    // Launch for first token
    cudaGraphLaunch(graph_exec_, config_.stream);
}

void Engine::invalidate_graph() {
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (captured_graph_) {
        cudaGraphDestroy(captured_graph_);
        captured_graph_ = nullptr;
    }
    graph_valid_ = false;
}

void Engine::execute_command(const Command& cmd) {
    // Find kernel
    auto it = kernels_.find(cmd.op);
    if (it == kernels_.end()) {
        throw std::runtime_error(
            "No kernel registered for op: " + std::string(op_name(cmd.op))
        );
    }
    
    // Build execution context
    ExecContext ctx{tensors_, scratch_, config_.stream, &tensors_, nullptr, {}};
    
    // Resolve tensor names to pointers
    resolve_tensors(ctx, cmd);
    
    // Execute with optional profiling
    if (config_.enable_profiling) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start, config_.stream);
        it->second(ctx, cmd);
        cudaEventRecord(stop, config_.stream);
        
        cudaEventSynchronize(stop);
        
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        record_profile(cmd.op, cmd.label, ms);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        it->second(ctx, cmd);
    }
    
    // Optional sync after each op (for debugging)
    if (config_.sync_after_each_op) {
        sync();
    }
    
    // Check for CUDA errors
    if (config_.check_errors) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                "CUDA error in " + std::string(op_name(cmd.op)) + ": " +
                cudaGetErrorString(err)
            );
        }
    }
}

void Engine::resolve_tensors(ExecContext& ctx, const Command& cmd) {
    // Resolve output tensor
    if (!cmd.output.empty()) {
        ctx.output = tensors_.get(cmd.output);
        // Note: output may be null if it needs to be allocated by the kernel
    }
    
    // Resolve input tensors
    ctx.inputs.clear();
    ctx.inputs.reserve(cmd.inputs.size());
    
    for (const auto& name : cmd.inputs) {
        TensorInfo* tensor = tensors_.get(name);
        if (!tensor) {
            throw std::runtime_error(
                "Input tensor not found: " + name + 
                " (op: " + std::string(op_name(cmd.op)) + ")"
            );
        }
        ctx.inputs.push_back(tensor);
    }
}

void Engine::reset_scratch() {
    scratch_.reset();
}

void Engine::sync() {
    cudaStreamSynchronize(config_.stream);
}

// ============================================================================
// PROFILING
// ============================================================================

void Engine::record_profile(OpTypeID op, const std::string& label, double time_ms) {
    auto& profile = profiles_[op];
    if (profile.call_count == 0) {
        profile.op = op;
        profile.label = label.empty() ? op_name(op) : label;
    }
    profile.time_ms += time_ms;
    profile.call_count++;
}

std::vector<KernelProfile> Engine::get_profiles() const {
    std::vector<KernelProfile> result;
    result.reserve(profiles_.size());
    for (const auto& [_, profile] : profiles_) {
        result.push_back(profile);
    }
    // Sort by total time descending
    std::sort(result.begin(), result.end(), 
              [](const auto& a, const auto& b) { return a.time_ms > b.time_ms; });
    return result;
}

void Engine::clear_profiles() {
    profiles_.clear();
}

void Engine::print_profile_summary() const {
    auto profiles = get_profiles();
    if (profiles.empty()) {
        std::cout << "No profiling data" << std::endl;
        return;
    }
    
    double total_ms = 0;
    for (const auto& p : profiles) {
        total_ms += p.time_ms;
    }
    
    std::cout << "=== Engine Profile ===" << std::endl;
    std::cout << std::setw(12) << "Op" 
              << std::setw(10) << "Calls"
              << std::setw(12) << "Total(ms)"
              << std::setw(12) << "Avg(ms)"
              << std::setw(8) << "%" << std::endl;
    std::cout << std::string(54, '-') << std::endl;
    
    for (const auto& p : profiles) {
        double avg = p.time_ms / p.call_count;
        double pct = (p.time_ms / total_ms) * 100.0;
        
        std::cout << std::setw(12) << op_name(p.op)
                  << std::setw(10) << p.call_count
                  << std::setw(12) << std::fixed << std::setprecision(2) << p.time_ms
                  << std::setw(12) << std::fixed << std::setprecision(3) << avg
                  << std::setw(7) << std::fixed << std::setprecision(1) << pct << "%"
                  << std::endl;
    }
    
    std::cout << std::string(54, '-') << std::endl;
    std::cout << std::setw(12) << "TOTAL"
              << std::setw(10) << ""
              << std::setw(12) << std::fixed << std::setprecision(2) << total_ms
              << std::endl;
}

// ============================================================================
// STUB KERNELS (for testing)
// ============================================================================

void register_stub_kernels(Engine& engine) {
    // Get all registered ops
    const auto& ops = OpTypeRegistry::instance().all();
    
    for (const auto& [id, info] : ops) {
        engine.register_kernel(id, [](ExecContext& ctx, const Command& cmd) {
            // Stub: just print what would happen
            std::cout << "[STUB] " << op_name(cmd.op) << ": ";
            if (!cmd.output.empty()) {
                std::cout << cmd.output << " <- ";
            }
            for (size_t i = 0; i < cmd.inputs.size(); i++) {
                if (i > 0) std::cout << ", ";
                std::cout << cmd.inputs[i];
                if (ctx.in(i)) {
                    std::cout << "[" << ctx.in(i)->numel() << "]";
                }
            }
            std::cout << std::endl;
        });
    }
}

} // namespace helios
