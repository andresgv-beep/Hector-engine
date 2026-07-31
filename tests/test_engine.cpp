// tests/test_engine.cpp
// HELIOS ENGINE - Engine Test
// ============================

#include "engine.hpp"
#include <iostream>
#include <cassert>
#include <sstream>

using namespace helios;

void test_default_scratch_is_opt_in() {
    std::cout << "Test: Default scratch is opt-in... ";
    Engine engine;
    assert(engine.scratch().capacity() == 0);
    assert(!engine.scratch().valid());
    std::cout << "PASSED" << std::endl;
}

void test_kernel_registration() {
    std::cout << "Test: Kernel registration... ";
    
    Engine engine;
    
    // Initially no kernels
    assert(!engine.has_kernel(op::MATMUL()));
    assert(!engine.has_kernel(op::SILU()));
    
    // Register by ID
    engine.register_kernel(op::MATMUL(), [](ExecContext&, const Command&) {
        // Stub
    });
    
    assert(engine.has_kernel(op::MATMUL()));
    
    // Register by name
    engine.register_kernel("silu", [](ExecContext&, const Command&) {
        // Stub
    });
    
    assert(engine.has_kernel(op::SILU()));
    
    std::cout << "PASSED" << std::endl;
}

void test_stub_kernels() {
    std::cout << "Test: Stub kernels... ";
    
    Engine engine;
    register_stub_kernels(engine);
    
    // All builtin ops should have stubs
    assert(engine.has_kernel(op::MATMUL()));
    assert(engine.has_kernel(op::RMSNORM()));
    assert(engine.has_kernel(op::ATTENTION()));
    assert(engine.has_kernel(op::SILU()));
    assert(engine.has_kernel(op::ADD()));
    
    std::cout << "PASSED" << std::endl;
}

void test_tensor_resolution() {
    std::cout << "Test: Tensor resolution... ";
    
    Engine engine;
    
    // Create some tensors
    engine.tensors().allocate_and_register("input", {4, 128}, dtype::FP16());
    engine.tensors().allocate_and_register("weight", {128, 256}, dtype::FP16());
    engine.tensors().allocate_and_register("output", {4, 256}, dtype::FP16());
    
    // Track what we receive in the kernel
    TensorInfo* received_output = nullptr;
    std::vector<TensorInfo*> received_inputs;
    
    engine.register_kernel(op::MATMUL(), [&](ExecContext& ctx, const Command& cmd) {
        received_output = ctx.output;
        received_inputs = ctx.inputs;
    });
    
    // Build and execute command
    CommandBuffer cb;
    cb.add_matmul("output", "input", "weight");
    engine.execute(cb);
    
    // Verify resolution
    assert(received_output != nullptr);
    assert(received_output->numel() == 4 * 256);
    
    assert(received_inputs.size() == 2);
    assert(received_inputs[0]->numel() == 4 * 128);
    assert(received_inputs[1]->numel() == 128 * 256);
    
    std::cout << "PASSED" << std::endl;
}

void test_execute_command_buffer() {
    std::cout << "Test: Execute command buffer... ";
    
    Engine engine;
    
    // Create tensors for a mini forward pass
    engine.tensors().allocate_and_register("x", {1, 512}, dtype::FP16());
    engine.tensors().allocate_and_register("w1", {512, 2048}, dtype::FP16());
    engine.tensors().allocate_and_register("w2", {2048, 512}, dtype::FP16());
    engine.tensors().allocate_and_register("ln_w", {512}, dtype::FP16());
    engine.tensors().allocate_and_register("h1", {1, 2048}, dtype::FP16());
    engine.tensors().allocate_and_register("h2", {1, 2048}, dtype::FP16());
    engine.tensors().allocate_and_register("h3", {1, 512}, dtype::FP16());
    engine.tensors().allocate_and_register("out", {1, 512}, dtype::FP16());
    
    // Track execution order
    std::vector<std::string> executed_ops;
    
    // Register tracking kernels
    auto make_tracker = [&](const std::string& name) {
        return [&executed_ops, name](ExecContext&, const Command&) {
            executed_ops.push_back(name);
        };
    };
    
    engine.register_kernel(op::MATMUL(), make_tracker("matmul"));
    engine.register_kernel(op::SILU(), make_tracker("silu"));
    engine.register_kernel(op::ADD(), make_tracker("add"));
    engine.register_kernel(op::RMSNORM(), make_tracker("rmsnorm"));
    
    // Build MLP forward pass
    CommandBuffer cb;
    cb.add_matmul("h1", "x", "w1");           // 1. x @ w1
    cb.add_silu("h2", "h1");                   // 2. silu(h1)
    cb.add_matmul("h3", "h2", "w2");           // 3. h2 @ w2
    cb.add_add("out", "x", "h3");              // 4. x + h3 (residual)
    cb.add_rmsnorm("out", "out", "ln_w", 1e-5f); // 5. rmsnorm
    
    // Execute
    engine.execute(cb);
    
    // Verify order
    assert(executed_ops.size() == 5);
    assert(executed_ops[0] == "matmul");
    assert(executed_ops[1] == "silu");
    assert(executed_ops[2] == "matmul");
    assert(executed_ops[3] == "add");
    assert(executed_ops[4] == "rmsnorm");
    
    std::cout << "PASSED" << std::endl;
}

void test_variable_inputs() {
    std::cout << "Test: Variable inputs in kernel... ";
    
    Engine engine;
    
    // Create 5 tensors to concatenate
    for (int i = 0; i < 5; i++) {
        engine.tensors().allocate_and_register(
            "t" + std::to_string(i), 
            {4, 64}, 
            dtype::FP16()
        );
    }
    engine.tensors().allocate_and_register("merged", {4, 320}, dtype::FP16());
    
    size_t num_inputs_received = 0;
    
    engine.register_kernel(op::CONCAT(), [&](ExecContext& ctx, const Command& cmd) {
        num_inputs_received = ctx.num_inputs();
        // Verify all inputs resolved
        for (size_t i = 0; i < ctx.num_inputs(); i++) {
            assert(ctx.in(i) != nullptr);
            assert(ctx.in(i)->numel() == 4 * 64);
        }
    });
    
    // Build concat command with 5 inputs
    CommandBuffer cb;
    cb.add_concat("merged", {"t0", "t1", "t2", "t3", "t4"}, 1);
    engine.execute(cb);
    
    assert(num_inputs_received == 5);
    
    std::cout << "PASSED" << std::endl;
}

void test_missing_tensor_error() {
    std::cout << "Test: Missing tensor error... ";
    
    Engine engine;
    register_stub_kernels(engine);
    
    // Only register input, not weight
    engine.tensors().allocate_and_register("input", {4, 128}, dtype::FP16());
    
    CommandBuffer cb;
    cb.add_matmul("output", "input", "missing_weight");
    
    bool caught_error = false;
    try {
        engine.execute(cb);
    } catch (const std::runtime_error& e) {
        caught_error = true;
        // Should mention the missing tensor name
        std::string msg = e.what();
        assert(msg.find("missing_weight") != std::string::npos);
    }
    
    assert(caught_error);
    
    std::cout << "PASSED" << std::endl;
}

void test_missing_kernel_error() {
    std::cout << "Test: Missing kernel error... ";
    
    Engine engine;
    // Don't register any kernels
    
    engine.tensors().allocate_and_register("a", {4}, dtype::FP16());
    engine.tensors().allocate_and_register("b", {4}, dtype::FP16());
    
    CommandBuffer cb;
    cb.add_add("c", "a", "b");
    
    bool caught_error = false;
    try {
        engine.execute(cb);
    } catch (const std::runtime_error& e) {
        caught_error = true;
        std::string msg = e.what();
        assert(msg.find("No kernel") != std::string::npos);
    }
    
    assert(caught_error);
    
    std::cout << "PASSED" << std::endl;
}

void test_scratch_memory() {
    std::cout << "Test: Scratch memory... ";
    
    EngineConfig config;
    config.scratch_pool.pool_size_bytes = 2 * 1024 * 1024;
    Engine engine(config);
    
    // Allocate from scratch in kernel
    void* scratch_ptr = nullptr;
    
    engine.register_kernel(op::MATMUL(), [&](ExecContext& ctx, const Command& cmd) {
        // Allocate temporary buffer
        scratch_ptr = ctx.scratch.allocate(1024 * 1024);  // 1MB
        assert(scratch_ptr != nullptr);
    });
    
    engine.tensors().allocate_and_register("a", {32, 32}, dtype::FP16());
    engine.tensors().allocate_and_register("b", {32, 32}, dtype::FP16());
    engine.tensors().allocate_and_register("c", {32, 32}, dtype::FP16());
    
    CommandBuffer cb;
    cb.add_matmul("c", "a", "b");
    
    // Execute - scratch should be allocated
    engine.execute(cb);
    assert(scratch_ptr != nullptr);
    
    // Reset scratch
    engine.reset_scratch();
    
    // Scratch pool should be back to start
    // (Next allocation would reuse same memory)
    
    std::cout << "PASSED" << std::endl;
}

void test_command_params_in_kernel() {
    std::cout << "Test: Command params in kernel... ";
    
    Engine engine;
    
    float received_eps = 0;
    
    engine.register_kernel(op::RMSNORM(), [&](ExecContext& ctx, const Command& cmd) {
        received_eps = cmd.get<float>("eps");
    });
    
    engine.tensors().allocate_and_register("x", {4, 512}, dtype::FP16());
    engine.tensors().allocate_and_register("w", {512}, dtype::FP16());
    engine.tensors().allocate_and_register("y", {4, 512}, dtype::FP16());
    
    CommandBuffer cb;
    cb.add_rmsnorm("y", "x", "w", 1e-6f);
    engine.execute(cb);
    
    assert(received_eps == 1e-6f);
    
    std::cout << "PASSED" << std::endl;
}

void test_stub_execution() {
    std::cout << "Test: Stub execution..." << std::endl;
    
    Engine engine;
    register_stub_kernels(engine);
    
    // Create tensors
    engine.tensors().allocate_and_register("hidden", {1, 2048}, dtype::FP16());
    engine.tensors().allocate_and_register("ln.weight", {2048}, dtype::FP16());
    engine.tensors().allocate_and_register("attn.qkv", {2048, 6144}, dtype::FP16());
    engine.tensors().allocate_and_register("h_norm", {1, 2048}, dtype::FP16());
    engine.tensors().allocate_and_register("qkv", {1, 6144}, dtype::FP16());
    
    // Build mini forward
    CommandBuffer cb;
    cb.add_rmsnorm("h_norm", "hidden", "ln.weight", 1e-5f);
    cb.add_matmul("qkv", "h_norm", "attn.qkv");
    
    std::cout << "  Executing with stubs:" << std::endl;
    std::cout << "  ";
    engine.execute(cb);
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "==================================" << std::endl;
    std::cout << "HELIOS ENGINE - Engine Test" << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << std::endl;
    
    // Check CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "ERROR: No CUDA devices found" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << std::endl;
    
    // Run tests
    test_default_scratch_is_opt_in();
    test_kernel_registration();
    test_stub_kernels();
    test_tensor_resolution();
    test_execute_command_buffer();
    test_variable_inputs();
    test_missing_tensor_error();
    test_missing_kernel_error();
    test_scratch_memory();
    test_command_params_in_kernel();
    test_stub_execution();
    
    std::cout << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "ALL TESTS PASSED ✓" << std::endl;
    std::cout << "==================================" << std::endl;
    
    return 0;
}
