// command.cpp
// HELIOS ENGINE - Command Buffer Implementation
// =============================================

#include "command.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace helios {

// ============================================================================
// COMMAND BUFFER
// ============================================================================

CommandBuffer::CommandBuffer() {
    commands_.reserve(256);
}

void CommandBuffer::add(Command cmd) {
    commands_.push_back(std::move(cmd));
}

Command& CommandBuffer::add_op(OpTypeID op, const std::string& output) {
    commands_.emplace_back(op, output);
    return commands_.back();
}

void CommandBuffer::clear() {
    commands_.clear();
}

void CommandBuffer::reserve(size_t n) {
    commands_.reserve(n);
}

void CommandBuffer::append(const CommandBuffer& other) {
    commands_.insert(commands_.end(), 
                     other.commands_.begin(), 
                     other.commands_.end());
}

// ============================================================================
// CONVENIENCE BUILDERS
// ============================================================================

void CommandBuffer::add_copy(const std::string& dst, const std::string& src) {
    add_op(op::COPY(), dst).in({src});
}

void CommandBuffer::add_add(const std::string& dst, const std::string& a, const std::string& b) {
    add_op(op::ADD(), dst).in({a, b});
}

void CommandBuffer::add_mul(const std::string& dst, const std::string& a, const std::string& b) {
    add_op(op::MUL(), dst).in({a, b});
}

void CommandBuffer::add_bias(const std::string& dst, const std::string& input, const std::string& bias) {
    add_op(op::ADD_BIAS(), dst).in({input, bias});
}

void CommandBuffer::add_scale(const std::string& dst, const std::string& src, float scalar) {
    add_op(op::SCALE(), dst).in({src}).set("scalar", scalar);
}

void CommandBuffer::add_silu(const std::string& dst, const std::string& src) {
    add_op(op::SILU(), dst).in({src});
}

void CommandBuffer::add_gelu(const std::string& dst, const std::string& src) {
    add_op(op::GELU(), dst).in({src});
}

void CommandBuffer::add_softmax(const std::string& dst, const std::string& src, int32_t dim) {
    add_op(op::SOFTMAX(), dst).in({src}).set("dim", dim);
}

void CommandBuffer::add_rmsnorm(const std::string& dst, const std::string& src,
                                 const std::string& weight, float eps) {
    add_op(op::RMSNORM(), dst).in({src, weight}).set("eps", eps);
}
void CommandBuffer::add_add_rmsnorm(const std::string& residual_dst, const std::string& normed_dst,
                                     const std::string& a, const std::string& b,
                                     const std::string& weight, float eps) {
    auto id = OpTypeRegistry::instance().get_id("add_rmsnorm");
    add_op(id, normed_dst).in({a, b, weight}).set("eps", eps).set("output_residual", residual_dst);
}

void CommandBuffer::add_layernorm(const std::string& dst, const std::string& src,
                                   const std::string& weight, const std::string& bias,
                                   float eps) {
    add_op(op::LAYERNORM(), dst).in({src, weight, bias}).set("eps", eps);
}

void CommandBuffer::add_matmul(const std::string& dst, const std::string& a, const std::string& b) {
    add_op(op::MATMUL(), dst).in({a, b});
}

void CommandBuffer::add_matmul_t(const std::string& dst, const std::string& a, const std::string& b) {
    add_op(op::MATMUL_T(), dst).in({a, b});
}

void CommandBuffer::add_rope(const std::string& dst, const std::string& src,
                              float theta, uint32_t dim, uint32_t offset) {
    add_op(op::ROPE(), dst)
        .in({src})
        .set("theta", theta)
        .set("dim", dim)
        .set("offset", offset);
}

void CommandBuffer::add_attention(const std::string& dst,
                                   const std::string& q, const std::string& k, const std::string& v,
                                   uint32_t num_heads, uint32_t num_kv_heads, uint32_t head_dim,
                                   bool causal,
                                   uint32_t seq_q, uint32_t seq_kv) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto& cmd = add_op(op::ATTENTION(), dst);
    cmd.in({q, k, v});
    cmd.set("num_heads", num_heads);
    cmd.set("num_kv_heads", num_kv_heads);
    cmd.set("head_dim", head_dim);
    cmd.set("scale", scale);
    cmd.set("causal", causal);
    if (seq_q > 0) cmd.set("seq_q", seq_q);
    if (seq_kv > 0) cmd.set("seq_kv", seq_kv);
}

void CommandBuffer::add_attention_cached(const std::string& dst,
                                          const std::string& q,
                                          const std::string& k_cache, const std::string& v_cache,
                                          uint32_t num_heads, uint32_t num_kv_heads, uint32_t head_dim,
                                          uint32_t seq_len, uint32_t max_seq_len) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    add_op(op::ATTENTION_CACHED(), dst)
        .in({q, k_cache, v_cache})
        .set("num_heads", num_heads)
        .set("num_kv_heads", num_kv_heads)
        .set("head_dim", head_dim)
        .set("scale", scale)
        .set("seq_len", seq_len)
        .set("max_seq_len", max_seq_len);
}

void CommandBuffer::add_kv_cache_update(const std::string& k_cache, const std::string& v_cache,
                                         const std::string& k_new, const std::string& v_new,
                                         uint32_t position, uint32_t max_seq_len,
                                         uint32_t num_kv_heads, uint32_t head_dim,
                                         uint32_t seq_len) {
    add_op(op::KV_CACHE_UPDATE(), k_cache)  // Output is k_cache (in-place)
        .in({k_new, v_new, k_cache, v_cache})
        .set("position", position)
        .set("max_seq_len", max_seq_len)
        .set("num_kv_heads", num_kv_heads)
        .set("head_dim", head_dim)
        .set("seq_len", seq_len);
}

void CommandBuffer::add_dequant(const std::string& dst, const std::string& src,
                                 const std::string& dtype_name) {
    add_op(op::DEQUANT(), dst).in({src}).set("src_dtype", dtype_name);
}

void CommandBuffer::add_embedding(const std::string& dst, const std::string& indices,
                                   const std::string& table) {
    add_op(op::EMBEDDING(), dst).in({indices, table});
}

void CommandBuffer::add_concat(const std::string& dst,
                                const std::vector<std::string>& inputs, int32_t dim) {
    Command cmd(op::CONCAT(), dst);
    cmd.inputs = inputs;
    cmd.set("dim", dim);
    add(std::move(cmd));
}

void CommandBuffer::add_split_qkv(const std::string& q, const std::string& k, const std::string& v,
                                   const std::string& qkv_fused,
                                   uint32_t q_size, uint32_t k_size, uint32_t v_size) {
    // Multi-output command: output field holds first output (q)
    // Additional outputs in params
    add_op(op::SPLIT_QKV(), q)
        .in({qkv_fused})
        .set("output_k", k)
        .set("output_v", v)
        .set("q_size", q_size)
        .set("k_size", k_size)
        .set("v_size", v_size);
}

void CommandBuffer::add_split_half(const std::string& first, const std::string& second,
                                    const std::string& fused, uint32_t split_size) {
    // Split tensor in half along last dimension
    add_op(op::SPLIT_HALF(), first)
        .in({fused})
        .set("output_second", second)
        .set("split_size", split_size);
}

// ============================================================================
// DEBUG
// ============================================================================

void CommandBuffer::print() const {
    std::cout << "=== CommandBuffer (" << size() << " commands) ===" << std::endl;
    
    for (size_t i = 0; i < commands_.size(); i++) {
        const Command& cmd = commands_[i];
        
        std::cout << std::setw(4) << i << ": ";
        std::cout << std::setw(12) << std::left << op_name(cmd.op);
        
        // Output
        if (!cmd.output.empty()) {
            std::cout << cmd.output;
        }
        
        // Inputs
        std::cout << " <- ";
        for (size_t j = 0; j < cmd.inputs.size(); j++) {
            if (j > 0) std::cout << ", ";
            std::cout << cmd.inputs[j];
        }
        
        // Parameters (show a few common ones)
        if (cmd.has("eps")) {
            std::cout << " [eps=" << cmd.get<float>("eps") << "]";
        }
        if (cmd.has("theta")) {
            std::cout << " [theta=" << cmd.get<float>("theta");
            if (cmd.has("dim")) {
                std::cout << ", dim=" << cmd.get<uint32_t>("dim");
            }
            std::cout << "]";
        }
        if (cmd.has("num_heads")) {
            std::cout << " [heads=" << cmd.get<uint32_t>("num_heads");
            if (cmd.has("num_kv_heads")) {
                std::cout << ", kv=" << cmd.get<uint32_t>("num_kv_heads");
            }
            std::cout << "]";
        }
        if (cmd.has("scalar")) {
            std::cout << " [scalar=" << cmd.get<float>("scalar") << "]";
        }
        if (cmd.has("dim") && !cmd.has("theta") && !cmd.has("num_heads")) {
            std::cout << " [dim=" << cmd.get<int32_t>("dim") << "]";
        }
        
        // Label
        if (!cmd.label.empty()) {
            std::cout << " # " << cmd.label;
        }
        
        std::cout << std::endl;
    }
}

} // namespace helios
