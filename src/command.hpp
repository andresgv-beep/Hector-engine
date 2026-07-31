// command.hpp
// HELIOS ENGINE - Command Buffer
// ===============================
// Define operaciones y command buffer para ejecución diferida.
//
// Diseño EXTENSIBLE:
//   - OpTypeID: registro dinámico de operaciones
//   - inputs: vector de nombres (no limitado a 3)
//   - params: map genérico de parámetros
//

#pragma once

#include "optype.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <any>

namespace helios {

// ============================================================================
// PARAMETER VALUE (type-safe union)
// ============================================================================

using ParamValue = std::variant<
    bool,
    int32_t,
    int64_t,
    uint32_t,
    uint64_t,
    float,
    double,
    std::string,
    std::vector<int32_t>,
    std::vector<float>
>;

// ============================================================================
// COMMAND
// ============================================================================

struct Command {
    OpTypeID op;
    
    // Output tensor name
    std::string output;
    
    // Input tensor names (variable count)
    std::vector<std::string> inputs;
    
    // Named parameters (extensible)
    std::unordered_map<std::string, ParamValue> params;
    
    // Debug info
    std::string label;  // Optional name for profiling
    
    // ========================================
    // CONSTRUCTORS
    // ========================================
    
    Command() : op(OP_INVALID) {}
    
    Command(OpTypeID op_) : op(op_) {}
    
    Command(OpTypeID op_, const std::string& output_) 
        : op(op_), output(output_) {}
    
    Command(OpTypeID op_, const std::string& output_, 
            std::initializer_list<std::string> inputs_)
        : op(op_), output(output_), inputs(inputs_) {}
    
    // ========================================
    // PARAMETER ACCESS
    // ========================================
    
    // Set parameter
    template<typename T>
    Command& set(const std::string& key, T value) {
        params[key] = ParamValue(std::move(value));
        return *this;
    }
    
    // Get parameter with default
    template<typename T>
    T get(const std::string& key, T default_value = T{}) const {
        auto it = params.find(key);
        if (it == params.end()) return default_value;
        
        if (auto* val = std::get_if<T>(&it->second)) {
            return *val;
        }
        return default_value;
    }
    
    // Check if parameter exists
    bool has(const std::string& key) const {
        return params.count(key) > 0;
    }
    
    // ========================================
    // INPUT ACCESS
    // ========================================
    
    const std::string& input(size_t i) const {
        static const std::string empty;
        return (i < inputs.size()) ? inputs[i] : empty;
    }
    
    size_t num_inputs() const { return inputs.size(); }
    
    // ========================================
    // FLUENT BUILDERS
    // ========================================
    
    Command& out(const std::string& o) { output = o; return *this; }
    Command& in(const std::string& i) { inputs.push_back(i); return *this; }
    Command& in(std::initializer_list<std::string> ins) { 
        inputs.insert(inputs.end(), ins);
        return *this;
    }
    Command& named(const std::string& l) { label = l; return *this; }
};

// ============================================================================
// COMMAND BUFFER
// ============================================================================

class CommandBuffer {
public:
    CommandBuffer();
    ~CommandBuffer() = default;
    
    // Copy/move OK
    CommandBuffer(const CommandBuffer&) = default;
    CommandBuffer& operator=(const CommandBuffer&) = default;
    CommandBuffer(CommandBuffer&&) = default;
    CommandBuffer& operator=(CommandBuffer&&) = default;
    
    // ========================================
    // Building
    // ========================================
    
    // Add raw command
    void add(Command cmd);
    
    // Generic builder
    Command& add_op(OpTypeID op, const std::string& output = "");
    
    // ========================================
    // Convenience builders (common ops)
    // ========================================
    
    // Elementwise
    void add_copy(const std::string& dst, const std::string& src);
    void add_add(const std::string& dst, const std::string& a, const std::string& b);
    void add_bias(const std::string& dst, const std::string& input, const std::string& bias);
    void add_mul(const std::string& dst, const std::string& a, const std::string& b);
    void add_mul_scalar_tensor(const std::string& dst, const std::string& input,
                               const std::string& scalar);
    void add_scale(const std::string& dst, const std::string& src, float scalar);
    
    // Activations
    void add_silu(const std::string& dst, const std::string& src);
    void add_gelu(const std::string& dst, const std::string& src);
    void add_softmax(const std::string& dst, const std::string& src, int32_t dim = -1);
    void add_softcap(const std::string& dst, const std::string& src, float cap);
    
    // Normalization
    void add_rmsnorm(const std::string& dst, const std::string& src, 
                     const std::string& weight, float eps);
    void add_layernorm(const std::string& dst, const std::string& src,
                       const std::string& weight, const std::string& bias, float eps);
    void add_add_rmsnorm(const std::string& residual_dst, const std::string& normed_dst,
                         const std::string& a, const std::string& b,
                         const std::string& weight, float eps);
    
    // Linear algebra
    void add_matmul(const std::string& dst, const std::string& a, const std::string& b);
    void add_matmul_t(const std::string& dst, const std::string& a, const std::string& b);
    
    // Position encoding
    void add_rope(const std::string& dst, const std::string& src,
                  float theta, uint32_t dim, uint32_t offset = 0,
                  float partial_rotary = 1.0f, bool proportional = false);
    
    // Attention (Q, K, V as separate inputs)
    void add_attention(const std::string& dst,
                       const std::string& q, const std::string& k, const std::string& v,
                       uint32_t num_heads, uint32_t num_kv_heads, uint32_t head_dim,
                       bool causal = true,
                       uint32_t seq_q = 0, uint32_t seq_kv = 0);
    
    // Attention with KV cache (autoregressive)
    void add_attention_cached(const std::string& dst,
                              const std::string& q,
                              const std::string& k_cache, const std::string& v_cache,
                              uint32_t num_heads, uint32_t num_kv_heads, uint32_t head_dim,
                              uint32_t seq_len, uint32_t max_seq_len);
    
    // KV cache update (copy new K/V to cache at position)
    void add_kv_cache_update(const std::string& k_cache, const std::string& v_cache,
                             const std::string& k_new, const std::string& v_new,
                             uint32_t position, uint32_t max_seq_len,
                             uint32_t num_kv_heads, uint32_t head_dim,
                             uint32_t seq_len = 0);
    
    // Quantization
    void add_dequant(const std::string& dst, const std::string& src, 
                     const std::string& dtype_name);
    
    // Memory
    void add_embedding(const std::string& dst, const std::string& indices,
                       const std::string& table);
    void add_concat(const std::string& dst, 
                    const std::vector<std::string>& inputs, int32_t dim);
    
    // Split operations (for fused weights)
    void add_split_qkv(const std::string& q, const std::string& k, const std::string& v,
                       const std::string& qkv_fused,
                       uint32_t q_size, uint32_t k_size, uint32_t v_size);
    void add_split_half(const std::string& first, const std::string& second,
                        const std::string& fused, uint32_t split_size);
    
    // ========================================
    // Access
    // ========================================
    
    const std::vector<Command>& commands() const { return commands_; }
    std::vector<Command>& commands() { return commands_; }
    
    size_t size() const { return commands_.size(); }
    bool empty() const { return commands_.empty(); }
    
    const Command& operator[](size_t i) const { return commands_[i]; }
    Command& operator[](size_t i) { return commands_[i]; }
    
    // ========================================
    // Manipulation
    // ========================================
    
    void clear();
    void reserve(size_t n);
    void append(const CommandBuffer& other);
    
    // ========================================
    // Debug
    // ========================================
    
    void print() const;
    
private:
    std::vector<Command> commands_;
};

} // namespace helios
