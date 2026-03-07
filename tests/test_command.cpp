// tests/test_command.cpp
// HELIOS ENGINE - CommandBuffer Test (v2 - extensible)
// =====================================================

#include "command.hpp"
#include "optype.hpp"
#include <iostream>
#include <cassert>

using namespace helios;

void test_optype_registry() {
    std::cout << "Test: OpType registry... ";
    
    // Builtins should be registered
    assert(op::NOP() != OP_INVALID);
    assert(op::MATMUL() != OP_INVALID);
    assert(op::RMSNORM() != OP_INVALID);
    assert(op::ATTENTION() != OP_INVALID);
    
    // Lookup by name
    auto& reg = OpTypeRegistry::instance();
    assert(reg.get_id("matmul") == op::MATMUL());
    assert(reg.get_id("attention") == op::ATTENTION());
    
    // Get info
    auto* matmul_info = reg.get(op::MATMUL());
    assert(matmul_info != nullptr);
    assert(matmul_info->category == "linear");
    assert(matmul_info->min_inputs == 2);
    
    auto* attn_info = reg.get(op::ATTENTION());
    assert(attn_info != nullptr);
    assert(attn_info->category == "attention");
    assert(attn_info->min_inputs == 3);  // Q, K, V
    
    std::cout << "PASSED" << std::endl;
}

void test_optype_extensibility() {
    std::cout << "Test: OpType extensibility... ";
    
    // Register a custom op
    OpTypeID my_op = OpTypeRegistry::Builder("my_custom_fused_attn")
        .category("attention")
        .inputs(4, 5)  // 4-5 inputs
        .build();
    
    assert(my_op != OP_INVALID);
    
    // Should be findable
    auto& reg = OpTypeRegistry::instance();
    assert(reg.get_id("my_custom_fused_attn") == my_op);
    
    auto* info = reg.get(my_op);
    assert(info != nullptr);
    assert(info->category == "attention");
    assert(info->min_inputs == 4);
    
    std::cout << "PASSED" << std::endl;
}

void test_basic_command() {
    std::cout << "Test: Basic command... ";
    
    Command cmd(op::ADD(), "output");
    cmd.in({"input_a", "input_b"});
    
    assert(cmd.op == op::ADD());
    assert(cmd.output == "output");
    assert(cmd.inputs.size() == 2);
    assert(cmd.input(0) == "input_a");
    assert(cmd.input(1) == "input_b");
    
    std::cout << "PASSED" << std::endl;
}

void test_command_params() {
    std::cout << "Test: Command params... ";
    
    Command cmd(op::ATTENTION(), "attn_out");
    cmd.in({"q", "k", "v"})
       .set("num_heads", uint32_t(32))
       .set("num_kv_heads", uint32_t(8))
       .set("head_dim", uint32_t(128))
       .set("scale", 0.0883883f)
       .set("causal", true);
    
    assert(cmd.has("num_heads"));
    assert(cmd.has("causal"));
    assert(!cmd.has("nonexistent"));
    
    assert(cmd.get<uint32_t>("num_heads") == 32);
    assert(cmd.get<uint32_t>("num_kv_heads") == 8);
    assert(cmd.get<bool>("causal") == true);
    
    // Default value for missing param
    assert(cmd.get<int32_t>("missing", -1) == -1);
    
    std::cout << "PASSED" << std::endl;
}

void test_variable_inputs() {
    std::cout << "Test: Variable inputs... ";
    
    // CONCAT with many inputs
    Command cmd(op::CONCAT(), "concatenated");
    cmd.in({"tensor1", "tensor2", "tensor3", "tensor4", "tensor5"})
       .set("dim", int32_t(1));
    
    assert(cmd.num_inputs() == 5);
    assert(cmd.input(0) == "tensor1");
    assert(cmd.input(4) == "tensor5");
    
    // Out of bounds returns empty
    assert(cmd.input(10).empty());
    
    std::cout << "PASSED" << std::endl;
}

void test_command_buffer_add() {
    std::cout << "Test: CommandBuffer add... ";
    
    CommandBuffer cb;
    
    assert(cb.empty());
    assert(cb.size() == 0);
    
    cb.add_add("c", "a", "b");
    
    assert(!cb.empty());
    assert(cb.size() == 1);
    assert(cb[0].op == op::ADD());
    
    cb.add_mul("d", "c", "c");
    
    assert(cb.size() == 2);
    
    std::cout << "PASSED" << std::endl;
}

void test_convenience_builders() {
    std::cout << "Test: Convenience builders... ";
    
    CommandBuffer cb;
    
    // Test various builders
    cb.add_copy("dst", "src");
    cb.add_scale("scaled", "input", 0.5f);
    cb.add_silu("activated", "hidden");
    cb.add_rmsnorm("normed", "input", "weight", 1e-6f);
    cb.add_matmul("output", "input", "weight");
    cb.add_rope("rotated", "qk", 10000.0f, 64, 0);
    cb.add_attention("attn_out", "q", "k", "v", 32, 8, 128, true);
    cb.add_embedding("embed", "token_ids", "embed_table");
    cb.add_dequant("weights_fp", "weights_q", "hq4k");
    
    assert(cb.size() == 9);
    
    // Verify specific params
    const Command& scale_cmd = cb[1];
    assert(scale_cmd.op == op::SCALE());
    assert(scale_cmd.get<float>("scalar") == 0.5f);
    
    const Command& norm_cmd = cb[3];
    assert(norm_cmd.op == op::RMSNORM());
    assert(norm_cmd.get<float>("eps") == 1e-6f);
    
    const Command& rope_cmd = cb[5];
    assert(rope_cmd.op == op::ROPE());
    assert(rope_cmd.get<float>("theta") == 10000.0f);
    assert(rope_cmd.get<uint32_t>("dim") == 64);
    
    const Command& attn_cmd = cb[6];
    assert(attn_cmd.op == op::ATTENTION());
    assert(attn_cmd.get<uint32_t>("num_heads") == 32);
    assert(attn_cmd.get<uint32_t>("num_kv_heads") == 8);
    assert(attn_cmd.get<uint32_t>("head_dim") == 128);
    assert(attn_cmd.get<bool>("causal") == true);
    
    std::cout << "PASSED" << std::endl;
}

void test_clear_and_reserve() {
    std::cout << "Test: Clear and reserve... ";
    
    CommandBuffer cb;
    cb.reserve(100);
    
    cb.add_add("a", "b", "c");
    cb.add_add("d", "e", "f");
    
    assert(cb.size() == 2);
    
    cb.clear();
    
    assert(cb.size() == 0);
    assert(cb.empty());
    
    std::cout << "PASSED" << std::endl;
}

void test_append() {
    std::cout << "Test: Append... ";
    
    CommandBuffer cb1;
    cb1.add_add("a", "b", "c");
    cb1.add_mul("d", "e", "f");
    
    CommandBuffer cb2;
    cb2.add_silu("g", "h");
    cb2.add_rmsnorm("i", "j", "w", 1e-5f);
    
    cb1.append(cb2);
    
    assert(cb1.size() == 4);
    assert(cb1[0].op == op::ADD());
    assert(cb1[1].op == op::MUL());
    assert(cb1[2].op == op::SILU());
    assert(cb1[3].op == op::RMSNORM());
    
    std::cout << "PASSED" << std::endl;
}

void test_fluent_api() {
    std::cout << "Test: Fluent API... ";
    
    CommandBuffer cb;
    
    // Build commands with fluent API
    cb.add_op(op::MATMUL(), "h1").in({"x", "w1"});
    cb.add_op(op::SILU(), "h1_act").in({"h1"});
    cb.add_op(op::MATMUL(), "out").in({"h1_act", "w2"});
    cb.add_op(op::ADD(), "residual").in({"x", "out"});
    cb.add_op(op::RMSNORM(), "final").in({"residual", "ln_w"}).set("eps", 1e-5f);
    
    assert(cb.size() == 5);
    
    std::cout << "PASSED" << std::endl;
}

void test_print() {
    std::cout << "Test: Print..." << std::endl;
    
    CommandBuffer cb;
    
    // Build a mini transformer layer
    cb.add_rmsnorm("h_norm", "hidden", "ln1.weight", 1e-5f);
    cb.add_matmul("qkv", "h_norm", "attn.qkv.weight");
    cb.add_rope("q_rot", "q", 10000.0f, 128, 0);
    cb.add_rope("k_rot", "k", 10000.0f, 128, 0);
    cb.add_attention("attn_out", "q_rot", "k_rot", "v", 32, 8, 128, true);
    cb.add_matmul("proj", "attn_out", "attn.o.weight");
    cb.add_add("res1", "hidden", "proj");
    cb.add_rmsnorm("mlp_norm", "res1", "ln2.weight", 1e-5f);
    cb.add_matmul("gate_up", "mlp_norm", "mlp.gate_up.weight");
    cb.add_silu("gate_act", "gate");
    cb.add_mul("mlp_h", "gate_act", "up");
    cb.add_matmul("mlp_out", "mlp_h", "mlp.down.weight");
    cb.add_add("output", "res1", "mlp_out");
    
    cb.print();
    
    std::cout << "PASSED" << std::endl;
}

void test_concat_variable_inputs() {
    std::cout << "Test: Concat variable inputs... ";
    
    CommandBuffer cb;
    
    // Concat 5 tensors - not possible with fixed src0/src1/src2
    cb.add_concat("merged", {"a", "b", "c", "d", "e"}, 1);
    
    assert(cb.size() == 1);
    assert(cb[0].op == op::CONCAT());
    assert(cb[0].num_inputs() == 5);
    assert(cb[0].get<int32_t>("dim") == 1);
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "==================================" << std::endl;
    std::cout << "HELIOS ENGINE - CommandBuffer Test (v2)" << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << std::endl;
    
    // Run tests
    test_optype_registry();
    test_optype_extensibility();
    test_basic_command();
    test_command_params();
    test_variable_inputs();
    test_command_buffer_add();
    test_convenience_builders();
    test_clear_and_reserve();
    test_append();
    test_fluent_api();
    test_print();
    test_concat_variable_inputs();
    
    std::cout << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "ALL TESTS PASSED ✓" << std::endl;
    std::cout << "==================================" << std::endl;
    
    return 0;
}
