// src/graph_builder.cpp
// ============================================================================
// HELIOS ENGINE - GraphBuilder Implementation
// ============================================================================
//
// REGLA DE ORO: No inventar nombres. Detectar desde tensores reales.
// REGLA DE PLATA: Un solo path. Si hay variante, parametrizar, no duplicar.
//
// Data flow genérico (por capa):
//
//   hidden → pre_norm → normed
//   normed → QKV projection (separado o fusionado) → +bias?
//   Q, K → RoPE (tipo desde config)
//   [cached: update KV cache]
//   Q, K, V → Attention [cached: usa K/V de cache] → attn_out
//   attn_out → o_proj → attn_proj
//   hidden + attn_proj → residual
//   residual → post_norm → normed2
//   normed2 → MLP (gated/fused_gate/simple) → mlp_out
//   residual + mlp_out → hidden (next layer)
//

#include "graph_builder.hpp"
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <iostream>

namespace helios {

// ============================================================================
// NAMING HELPERS
// ============================================================================

std::string GraphBuilder::W(const ArchDescriptor& arch, uint32_t layer,
                             const std::string& component) const {
    return arch.prefix + ".layer" + std::to_string(layer) + "." + component;
}

std::string GraphBuilder::WG(const ArchDescriptor& arch, const std::string& name) const {
    return arch.prefix + "." + name;
}

std::string GraphBuilder::S(const std::string& name) {
    return "_s." + name;
}

// ============================================================================
// ACTIVATION DISPATCH (polimórfico)
// ============================================================================

void GraphBuilder::add_activation(
    CommandBuffer& cb,
    const std::string& dst,
    const std::string& src,
    ActivationType type
) {
    switch (type) {
        case ActivationType::SILU:
            cb.add_silu(dst, src);
            break;
        case ActivationType::GELU:
        case ActivationType::GELU_NEW:
            cb.add_gelu(dst, src);
            break;
        case ActivationType::RELU:
            // TODO: add_relu cuando tengamos el kernel
            cb.add_silu(dst, src);  // Fallback temporal
            break;
    }
}

// Fused activation × mul: act(gate) * up → dst (single kernel)
void GraphBuilder::add_gated_activation(
    CommandBuffer& cb,
    const std::string& dst,
    const std::string& gate,
    const std::string& up,
    ActivationType type
) {
    OpTypeID fused_op;
    switch (type) {
        case ActivationType::SILU:
            fused_op = op_id("silu_mul");
            break;
        case ActivationType::GELU:
        case ActivationType::GELU_NEW:
            fused_op = op_id("gelu_mul");
            break;
        default:
            // Fallback to unfused
            add_activation(cb, dst, gate, type);
            cb.add_mul(dst, dst, up);
            return;
    }
    cb.add_op(fused_op, dst).in({gate, up});
}

// ============================================================================
// ARCHITECTURE DETECTION
// ============================================================================

static ActivationType parse_activation(const std::string& s) {
    if (s == "silu" || s == "swish") return ActivationType::SILU;
    if (s == "gelu") return ActivationType::GELU;
    if (s == "gelu_new" || s == "gelu_fast") return ActivationType::GELU_NEW;
    if (s == "relu") return ActivationType::RELU;
    return ActivationType::SILU;  // Safe default
}

static RoPEType parse_rope_type(const std::string& s) {
    if (s == "default" || s == "standard") return RoPEType::DEFAULT;
    if (s == "llama3") return RoPEType::LLAMA3;
    if (s == "linear") return RoPEType::LINEAR;
    if (s == "dynamic") return RoPEType::DYNAMIC;
    if (s == "yarn") return RoPEType::YARN;
    if (s == "longrope") return RoPEType::LONGROPE;
    if (s == "su") return RoPEType::SU;
    if (s == "none") return RoPEType::NONE;
    return RoPEType::DEFAULT;
}

ArchDescriptor GraphBuilder::detect_architecture(
    const Engine& engine,
    const std::string& prefix,
    const ModelConfig& config
) const {
    ArchDescriptor arch;
    arch.prefix = prefix;
    
    // ====== PHASE 1: Detect from tensors (ground truth) ======
    
    // Count layers
    for (uint32_t i = 0; i < 256; i++) {
        std::string test = prefix + ".layer" + std::to_string(i) + ".";
        bool found = false;
        for (const auto& name : engine.tensors().names()) {
            if (name.find(test) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            arch.num_layers = i;
            break;
        }
    }
    
    if (arch.num_layers == 0) {
        throw std::runtime_error("No layers found for prefix: " + prefix);
    }
    
    // Detect from layer 0
    std::string L0 = prefix + ".layer0.";
    
    // Attention: fused or separate QKV?
    arch.has_fused_qkv = engine.tensors().exists(L0 + "attn.qkv_proj.weight");
    
    // Attention bias
    arch.has_qkv_bias = arch.has_fused_qkv
        ? engine.tensors().exists(L0 + "attn.qkv_proj.bias")
        : engine.tensors().exists(L0 + "attn.q_proj.bias");
    
    // O projection bias
    arch.has_o_proj_bias = engine.tensors().exists(L0 + "attn.o_proj.bias");

    // QK-norm (Qwen3): RMSNorm por-head sobre Q y K antes de RoPE
    arch.use_qk_norm = engine.tensors().exists(L0 + "attn.q_norm.weight");
    
    // Norm names
    if (engine.tensors().exists(L0 + "ln_attn_in.weight")) {
        arch.pre_attn_norm = "ln_attn_in";
        arch.post_attn_norm = "ln_attn_out";
    } else if (engine.tensors().exists(L0 + "ln1.weight")) {
        arch.pre_attn_norm = "ln1";
        arch.post_attn_norm = "ln2";
    } else {
        throw std::runtime_error("Cannot detect norm names for: " + prefix);
    }
    
    // Norm bias (LayerNorm vs RMSNorm)
    arch.norm_has_bias = engine.tensors().exists(L0 + arch.pre_attn_norm + ".bias");
    
    // MLP structure
    if (engine.tensors().exists(L0 + "mlp.gate.weight")) {
        arch.has_gate = true;
        arch.mlp_gate_name = "gate";
        arch.mlp_up_name = "up";
        arch.mlp_down_name = "down";
    } else if (engine.tensors().exists(L0 + "mlp.gate_up.weight")) {
        arch.has_fused_gate_up = true;
        arch.has_gate = true;  // Fused gate_up IS gated
        arch.mlp_gate_name = "gate_up";
        arch.mlp_up_name = "";
        arch.mlp_down_name = "down";
    } else if (engine.tensors().exists(L0 + "mlp.fc1.weight")) {
        arch.has_gate = false;
        arch.mlp_up_name = "fc1";
        arch.mlp_down_name = "fc2";
    } else {
        throw std::runtime_error("Cannot detect MLP structure for: " + prefix);
    }
    
    // MLP bias
    if (arch.has_fused_gate_up) {
        arch.has_mlp_bias = engine.tensors().exists(L0 + "mlp.gate_up.bias");
    } else if (arch.has_gate) {
        arch.has_mlp_bias = engine.tensors().exists(L0 + "mlp.gate.bias");
    } else {
        arch.has_mlp_bias = engine.tensors().exists(L0 + "mlp.fc1.bias");
    }
    
    // Global tensors - embedding
    if (engine.tensors().exists(prefix + ".token_embedding.weight")) {
        arch.embedding_name = "token_embedding.weight";
    } else if (engine.tensors().exists(prefix + ".embed_tokens.weight")) {
        arch.embedding_name = "embed_tokens.weight";
    }
    
    // Global tensors - final norm (weight)
    if (engine.tensors().exists(prefix + ".final_norm.weight")) {
        arch.final_norm_name = "final_norm.weight";
    } else if (engine.tensors().exists(prefix + ".norm.weight")) {
        arch.final_norm_name = "norm.weight";
    }
    
    // Global tensors - final norm (bias) — DETECTADO, no inventado
    // Derivar nombre de bias del nombre de weight
    if (!arch.final_norm_name.empty()) {
        std::string bias_name = arch.final_norm_name;
        // "final_norm.weight" → "final_norm.bias", "norm.weight" → "norm.bias"
        auto pos = bias_name.rfind(".weight");
        if (pos != std::string::npos) {
            bias_name.replace(pos, 7, ".bias");
        }
        if (engine.tensors().exists(prefix + "." + bias_name)) {
            arch.final_norm_bias = bias_name;
        }
        // else: stays empty → no bias
    }
    
    // Global tensors - lm_head
    if (engine.tensors().exists(prefix + ".lm_head.weight")) {
        arch.lm_head_name = "lm_head.weight";
    }
    
    // Auto-detect compute dtype from first weight found
    {
        std::string test_weight = L0 + arch.pre_attn_norm + ".weight";
        auto* t = engine.tensors().get(test_weight);
        if (t) {
            arch.compute_dtype = t->dtype;
        }
    }
    
    // ====== PHASE 2: Enrich from ModelConfig (semantics) ======
    
    // Activation: check config, fallback to structural default
    if (config.has("mlp_activation")) {
        arch.activation = parse_activation(
            config.get<std::string>("mlp_activation", "silu")
        );
    } else if (config.has("hidden_act")) {
        // HuggingFace naming convention
        arch.activation = parse_activation(
            config.get<std::string>("hidden_act", "silu")
        );
    } else {
        // Structural default: gated → SiLU, non-gated → GELU
        arch.activation = arch.has_gate ? ActivationType::SILU : ActivationType::GELU;
    }
    
    // RoPE type
    arch.rope_type = parse_rope_type(
        config.get<std::string>("rope_type", "default")
    );
    arch.rope_theta = config.get<float>("rope_theta", 10000.0f);
    arch.rope_scaling_factor = config.get<float>("rope_scaling_factor", 1.0f);
    arch.partial_rotary_factor = config.get<float>("partial_rotary_factor", 1.0f);
    
    // Config "dtype" describes the ORIGINAL model precision (before quantization).
    // Actual compute dtype comes from the real tensors loaded in GPU.
    // Detected dtype (from norm weights) takes priority because:
    //   - Quantized weights (HQ4K/HQ5K) dequantize to FP16
    //   - Norm weights are stored in compute precision (FP16)
    //   - Config saying "bf16" just means the source model was BF16
    // Only use config dtype if we couldn't detect from tensors.
    if (arch.compute_dtype == 0 && config.has("dtype")) {
        std::string dtype_str = config.get<std::string>("dtype", "");
        if (dtype_str == "bf16") arch.compute_dtype = dtype::BF16();
        else if (dtype_str == "fp32") arch.compute_dtype = dtype::FP32();
        else if (dtype_str == "fp16") arch.compute_dtype = dtype::FP16();
    }
    
    return arch;
}

// ============================================================================
// SCRATCH ALLOCATION
// ============================================================================

void GraphBuilder::allocate_scratch(
    Engine& engine,
    const ModelConfig& config,
    const ArchDescriptor& arch,
    uint32_t max_batch,
    uint32_t max_seq
) {
    if (scratch_allocated_) {
        free_scratch(engine);
    }
    
    alloc_batch_ = max_batch;
    alloc_seq_ = max_seq;
    
    uint32_t B = max_batch;
    uint32_t L = max_seq;
    uint32_t D = config.hidden_size();
    uint32_t I = config.intermediate_size();
    uint32_t V = config.vocab_size();
    uint32_t H = config.num_attention_heads();
    uint32_t KVH = config.num_key_value_heads();
    uint32_t HD = config.head_dim();
    
    // Use model's compute dtype for scratch (not hardcoded FP16)
    DTypeID scratch_dtype = arch.compute_dtype;
    if (scratch_dtype == 0) scratch_dtype = dtype::FP16();  // Safe fallback
    
    auto alloc = [&](const std::string& name, std::vector<uint32_t> shape) {
        std::string full = S(name);
        engine.tensors().allocate_and_register(full, shape, scratch_dtype);
        scratch_names_.push_back(full);
    };
    
    // Hidden states
    alloc("hidden",   {B, L, D});
    alloc("normed",   {B, L, D});
    alloc("residual", {B, L, D});
    
    // Attention
    alloc("q", {B, L, H * HD});
    alloc("k", {B, L, KVH * HD});
    alloc("v", {B, L, KVH * HD});
    alloc("attn_out",  {B, L, H * HD});
    alloc("attn_proj", {B, L, D});
    
    // Fused QKV buffer (for architectures with fused qkv_proj)
    if (arch.has_fused_qkv) {
        alloc("qkv_fused", {B, L, H * HD + 2 * KVH * HD});
    }
    
    // MLP
    alloc("gate",     {B, L, I});
    alloc("up",       {B, L, I});
    alloc("gate_act", {B, L, I});
    alloc("mlp_h",    {B, L, I});
    alloc("mlp_out",  {B, L, D});
    
    // Fused gate_up buffer
    if (arch.has_fused_gate_up) {
        alloc("gate_up_fused", {B, L, 2 * I});
    }
    
    // Output
    alloc("logits", {B, L, V});
    
    scratch_allocated_ = true;
}

void GraphBuilder::free_scratch(Engine& engine) {
    for (const auto& name : scratch_names_) {
        if (engine.tensors().exists(name)) {
            engine.tensors().remove(name);
        }
    }
    scratch_names_.clear();
    scratch_allocated_ = false;
    alloc_batch_ = 0;
    alloc_seq_ = 0;
}

// ============================================================================
// BUILD FORWARD (UNIFICADO)
// ============================================================================
// Un solo path para cached y non-cached.
// Si cache == nullptr → prefill (full attention)
// Si cache != nullptr → autoregresivo (KV cache)

CommandBuffer GraphBuilder::build_forward(
    Engine& engine,
    const ModelConfig& config,
    const ArchDescriptor& arch,
    const std::string& input_tokens,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t position_offset,
    const KVCacheParams* cache
) {
    if (!scratch_allocated_) {
        throw std::runtime_error("GraphBuilder: call allocate_scratch() first");
    }

    // Ajustar la METADATA de forma de los scratch al seq actual (los buffers
    // siguen dimensionados a max_seq; solo cambia shape[1]). Los kernels que
    // derivan su tamaño de la forma (silu, add, rmsnorm, mul...) procesan así
    // exactamente seq filas — sin esto, un decode con scratch de 512 hacía
    // 512× trabajo en cada op elementwise (¡21 tok/s en vez de 85!).
    if (seq_len <= alloc_seq_) {
        for (const auto& name : scratch_names_) {
            TensorInfo* t = engine.tensors().get(name);
            if (t && t->shape.size() == 3) t->shape[1] = seq_len;
        }
    }

    CommandBuffer cb;
    cb.reserve(arch.num_layers * 20 + 10);
    
    // 1. Embedding
    cb.add_embedding(S("hidden"), input_tokens, WG(arch, arch.embedding_name));
    
    // 2. Transformer layers
    for (uint32_t i = 0; i < arch.num_layers; i++) {
        build_attention_block(cb, config, arch, i, batch_size, seq_len,
                            position_offset, cache);
        build_mlp_block(cb, config, arch, i, batch_size, seq_len);
    }
    
    // 3. Final norm
    if (arch.norm_has_bias && !arch.final_norm_bias.empty()) {
        cb.add_layernorm(S("normed"), S("hidden"), 
                         WG(arch, arch.final_norm_name),
                         WG(arch, arch.final_norm_bias),
                         config.rms_norm_eps());
    } else {
        cb.add_rmsnorm(S("normed"), S("hidden"),
                       WG(arch, arch.final_norm_name),
                       config.rms_norm_eps());
    }
    
    // 4. LM head (logits)
    // En prefill (seq>1) solo importa la ÚLTIMA posición: es de donde sale el
    // token siguiente. Calcular vocab×seq logits para tirar seq-1 filas es el
    // mayor desperdicio del prefill y obliga a descuantizar la matriz entera
    // (1.16 GB en un 8B). Con row_offset el lm_head del prefill es un GEMV y
    // los logits quedan SIEMPRE en la fila 0.
    {
        const std::string& w = arch.lm_head_name.empty()
            ? arch.embedding_name : arch.lm_head_name;
        cb.add_matmul(S("logits"), S("normed"), WG(arch, w));
        auto& c = cb.commands().back();
        if (seq_len > 1) {
            c.set("seq_len", (uint32_t)1);
            c.set("row_offset", (uint32_t)(seq_len - 1));
        } else {
            c.set("seq_len", seq_len);
        }
    }
    
    return cb;
}

// ============================================================================
// FORWARD PASS CON REUSE (decode optimizado)
// ============================================================================
// En decode (batch=1, seq=1), el CommandBuffer es idéntico entre tokens
// excepto por cache_position. Esta función:
//   1ª llamada: build completo + indexa qué commands tienen cache_pos params
//   N llamadas: solo actualiza esos params (O(K) donde K ~ 3*num_layers)

const CommandBuffer& GraphBuilder::build_forward_reuse(
    Engine& engine,
    const ModelConfig& config,
    const ArchDescriptor& arch,
    const std::string& input_tokens,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t position_offset,
    const KVCacheParams* cache
) {
    // Si dimensiones cambian (ej: prefill→decode) o no hay cache → rebuild
    if (cached_decode_.valid &&
        cached_decode_.batch_size == batch_size &&
        cached_decode_.seq_len == seq_len)
    {
        // Fast path: solo actualizar cache_pos params
        if (cache) {
            uint32_t pos = cache->cache_position;
            auto& cmds = cached_decode_.cb.commands();
            for (const auto& ref : cached_decode_.refs) {
                uint32_t val = (ref.kind == CachedDecode::CachePosRef::PLUS_SEQ)
                             ? (pos + seq_len) : pos;
                cmds[ref.cmd_idx].params[ref.param_name] = ParamValue(val);
            }
        }
        return cached_decode_.cb;
    }
    
    // Build completo (primera vez o cambio de dimensiones)
    cached_decode_.cb = build_forward(
        engine, config, arch, input_tokens,
        batch_size, seq_len, position_offset, cache);
    cached_decode_.batch_size = batch_size;
    cached_decode_.seq_len = seq_len;
    cached_decode_.refs.clear();
    
    // Indexar commands que contienen cache_pos-dependent params
    auto& cmds = cached_decode_.cb.commands();
    for (size_t i = 0; i < cmds.size(); i++) {
        const auto& cmd = cmds[i];
        
        // RoPE: "offset" param = cache_pos
        if (cmd.op == op::ROPE() && cmd.has("offset")) {
            cached_decode_.refs.push_back({
                i, "offset", CachedDecode::CachePosRef::DIRECT
            });
        }
        // KV_CACHE_UPDATE: "position" param = cache_pos
        else if (cmd.op == op::KV_CACHE_UPDATE() && cmd.has("position")) {
            cached_decode_.refs.push_back({
                i, "position", CachedDecode::CachePosRef::DIRECT
            });
        }
        // ATTENTION_CACHED: "seq_len" param = cache_pos + seq_len
        else if (cmd.op == op::ATTENTION_CACHED() && cmd.has("seq_len")) {
            cached_decode_.refs.push_back({
                i, "seq_len", CachedDecode::CachePosRef::PLUS_SEQ
            });
        }
    }
    
    cached_decode_.valid = true;
    return cached_decode_.cb;
}

CommandBuffer GraphBuilder::build_single_layer(
    Engine& engine,
    const ModelConfig& config,
    const ArchDescriptor& arch,
    uint32_t layer_idx,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t position_offset,
    const KVCacheParams* cache
) {
    CommandBuffer cb;
    cb.reserve(20);
    
    build_attention_block(cb, config, arch, layer_idx, batch_size, seq_len,
                        position_offset, cache);
    build_mlp_block(cb, config, arch, layer_idx, batch_size, seq_len);
    
    return cb;
}

// ============================================================================
// ATTENTION BLOCK (unificado: con y sin cache)
// ============================================================================

void GraphBuilder::build_attention_block(
    CommandBuffer& cb,
    const ModelConfig& config,
    const ArchDescriptor& arch,
    uint32_t layer_idx,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t position_offset,
    const KVCacheParams* cache
) {
    uint32_t H = config.num_attention_heads();
    uint32_t KVH = config.num_key_value_heads();
    uint32_t HD = config.head_dim();
    
    // 1. Pre-attention norm + 2. Q/K/V projections
    // In decode (seq_len=1), fuse rmsnorm + first matmul when possible
    // Fusion disabled — current fused kernel is slower due to block-wide
    // RMSNorm reduction serializing all GEMV warps. Needs redesign.
    // TODO: pre-pass rmsnorm in separate small kernel, or warp-0-only reduction.
    bool can_fuse_norm_qkv = false; // was: (seq_len == 1) && !arch.norm_has_bias && arch.has_fused_qkv;
    
    std::string norm_w = W(arch, layer_idx, arch.pre_attn_norm + ".weight");
    
    if (can_fuse_norm_qkv) {
        // FUSED: rmsnorm(hidden) + matmul(normed, qkv_weight) → qkv_fused
        // Eliminates: normed write (K bytes) + normed read (K bytes per GEMV row)
        auto rmsnorm_gemv_id = OpTypeRegistry::instance().get_id("rmsnorm_gemv");
        auto& cmd = cb.add_op(rmsnorm_gemv_id, S("qkv_fused"));
        cmd.in({S("hidden"), norm_w, W(arch, layer_idx, "attn.qkv_proj.weight")});
        cmd.set("eps", config.rms_norm_eps());
        cmd.set("seq_len", seq_len);
        
        // Still need bias if present (applied after fused kernel)
        if (arch.has_qkv_bias) {
            cb.add_bias(S("qkv_fused"), S("qkv_fused"), W(arch, layer_idx, "attn.qkv_proj.bias"));
        }
        
        // Split QKV (same as unfused path)
        uint32_t q_size = H * HD;
        uint32_t kv_size = KVH * HD;
        cb.add_split_qkv(S("q"), S("k"), S("v"), S("qkv_fused"), 
                         q_size, kv_size, kv_size);
        cb.commands().back().set("seq_len", seq_len);
        
        // Also need to write normed for potential downstream use
        // Actually no — in this architecture, normed is only used by QKV matmul
        // which we've fused. But we still need it for the O_proj and MLP paths
        // that read from "normed". Wait — O_proj reads from attn_out, not normed.
        // And MLP block reads from "residual" and "hidden". So normed is ONLY
        // used by the QKV projections. Fusion is clean!
        
    } else {
        // UNFUSED: separate rmsnorm + matmul
        if (arch.norm_has_bias) {
            std::string norm_b = W(arch, layer_idx, arch.pre_attn_norm + ".bias");
            cb.add_layernorm(S("normed"), S("hidden"), norm_w, norm_b, config.rms_norm_eps());
        } else {
            cb.add_rmsnorm(S("normed"), S("hidden"), norm_w, config.rms_norm_eps());
        }
        
        // 2. Q/K/V projections (polimórfico: fused o separado)
        if (arch.has_fused_qkv) {
            cb.add_matmul(S("qkv_fused"), S("normed"), W(arch, layer_idx, "attn.qkv_proj.weight"));
            cb.commands().back().set("seq_len", seq_len);
            
            if (arch.has_qkv_bias) {
                cb.add_bias(S("qkv_fused"), S("qkv_fused"), W(arch, layer_idx, "attn.qkv_proj.bias"));
            }
            
            uint32_t q_size = H * HD;
            uint32_t kv_size = KVH * HD;
            cb.add_split_qkv(S("q"), S("k"), S("v"), S("qkv_fused"), 
                             q_size, kv_size, kv_size);
            cb.commands().back().set("seq_len", seq_len);
        } else {
            cb.add_matmul(S("q"), S("normed"), W(arch, layer_idx, "attn.q_proj.weight"));
        cb.commands().back().set("seq_len", seq_len);
            cb.add_matmul(S("k"), S("normed"), W(arch, layer_idx, "attn.k_proj.weight"));
        cb.commands().back().set("seq_len", seq_len);
            cb.add_matmul(S("v"), S("normed"), W(arch, layer_idx, "attn.v_proj.weight"));
        cb.commands().back().set("seq_len", seq_len);
            
            if (arch.has_qkv_bias) {
                cb.add_bias(S("q"), S("q"), W(arch, layer_idx, "attn.q_proj.bias"));
                cb.add_bias(S("k"), S("k"), W(arch, layer_idx, "attn.k_proj.bias"));
                cb.add_bias(S("v"), S("v"), W(arch, layer_idx, "attn.v_proj.bias"));
            }
        }
    }
    
    // 2b/3 fusionados (Qwen3): qk_norm + rope de Q y K en UN kernel.
    // 4 lanzamientos por capa → 1 (rmsnorm(q), rmsnorm(k), rope(q), rope(k)).
    if (arch.use_qk_norm && arch.rope_type != RoPEType::NONE) {
        uint32_t rope_offset = cache ? cache->cache_position : position_offset;
        bool device_pos = (cache != nullptr) && (seq_len == 1);
        auto id = OpTypeRegistry::instance().get_id("qk_norm_rope");
        auto& c = cb.add_op(id, S("q"));
        c.in({S("q"), S("k"),
              W(arch, layer_idx, "attn.q_norm.weight"),
              W(arch, layer_idx, "attn.k_norm.weight")});
        c.set("num_heads", H);
        c.set("num_kv_heads", KVH);
        c.set("dim", HD);
        c.set("seq_len", seq_len);
        c.set("offset", rope_offset);
        c.set("eps", config.rms_norm_eps());
        c.set("theta", arch.rope_theta);
        c.set("partial_rotary", arch.partial_rotary_factor);
        c.set("rope_scaling_factor", arch.rope_scaling_factor);
        if (device_pos) c.set("device_pos", (uint32_t)1);
    }
    // 2b. QK-norm sin RoPE (poco común): RMSNorm por-head separada
    else if (arch.use_qk_norm) {
        cb.add_rmsnorm(S("q"), S("q"), W(arch, layer_idx, "attn.q_norm.weight"),
                       config.rms_norm_eps());
        cb.commands().back().set("dim", HD);
        cb.add_rmsnorm(S("k"), S("k"), W(arch, layer_idx, "attn.k_norm.weight"),
                       config.rms_norm_eps());
        cb.commands().back().set("dim", HD);
    }

    // 3. RoPE clásico (arquitecturas sin qk_norm)
    if (!arch.use_qk_norm && arch.rope_type != RoPEType::NONE) {
        uint32_t rope_offset = cache ? cache->cache_position : position_offset;
        // All RoPE config as explicit params — polimórfico, escalable.
        // Cada modelo aporta su propia combinación:
        //   - partial_rotary: fracción de head_dim a rotar (Phi=0.75, default=1.0)
        //   - rope_scaling_factor: escala posicional (DeepSeek linear=4.0, default=1.0)
        //   - theta/dim/offset: standard RoPE params
        //   - num_heads/seq_len: explícitos (no deducir de scratch shapes)
        // En decode con cache, offset se lee de device (d_cache_pos) para que el
        // command buffer sea capturable una sola vez (CUDA Graph replay).
        bool device_pos = (cache != nullptr) && (seq_len == 1);
        {
            auto& cmd_q = cb.add_op(op::ROPE(), S("q"));
            cmd_q.in({S("q")});
            cmd_q.set("theta", arch.rope_theta);
            cmd_q.set("dim", HD);
            cmd_q.set("offset", rope_offset);
            cmd_q.set("num_heads", H);
            cmd_q.set("seq_len", seq_len);
            cmd_q.set("partial_rotary", arch.partial_rotary_factor);
            cmd_q.set("rope_scaling_factor", arch.rope_scaling_factor);
            if (device_pos) cmd_q.set("device_pos", (uint32_t)1);
        }
        {
            auto& cmd_k = cb.add_op(op::ROPE(), S("k"));
            cmd_k.in({S("k")});
            cmd_k.set("theta", arch.rope_theta);
            cmd_k.set("dim", HD);
            cmd_k.set("offset", rope_offset);
            cmd_k.set("num_heads", KVH);
            cmd_k.set("seq_len", seq_len);
            cmd_k.set("partial_rotary", arch.partial_rotary_factor);
            cmd_k.set("rope_scaling_factor", arch.rope_scaling_factor);
            if (device_pos) cmd_k.set("device_pos", (uint32_t)1);
        }
    }
    
    // 4. Attention (bifurcación prefill vs decode)
    if (cache) {
        std::string k_cache = cache->kv_prefix + ".layer" + std::to_string(layer_idx) + ".k";
        std::string v_cache = cache->kv_prefix + ".layer" + std::to_string(layer_idx) + ".v";
        
        if (seq_len > 1) {
            // ============================================================
            // PREFILL con cache: cache primero, atención sobre TODO el cache
            // ============================================================
            // El orden importa: al escribir K/V nuevos al cache antes de la
            // atención, cada query nueva atiende causalmente a la historia
            // completa [0..past+q]. (El branch anterior atendía solo dentro
            // del turno nuevo — incorrecto en multi-turno.)
            cb.add_kv_cache_update(k_cache, v_cache, S("k"), S("v"),
                                   cache->cache_position, cache->max_cache_len, KVH, HD, seq_len);
            {
                float scale = 1.0f / std::sqrt(static_cast<float>(HD));
                auto pf_id = OpTypeRegistry::instance().get_id("attention_prefill_cached");
                auto& c = cb.add_op(pf_id, S("attn_out"));
                c.in({S("q"), k_cache, v_cache});
                c.set("num_heads", H);
                c.set("num_kv_heads", KVH);
                c.set("head_dim", HD);
                c.set("scale", scale);
                c.set("seq_len", seq_len);
                c.set("past_len", cache->cache_position);
                c.set("max_seq_len", cache->max_cache_len);
            }
        } else {
            // ============================================================
            // DECODE: update cache primero, luego cached attention
            // ============================================================
            // device_pos: position/total_seq se leen de device en ejecución
            // (los params horneados son solo fallback sin CUDA Graph)
            cb.add_kv_cache_update(k_cache, v_cache, S("k"), S("v"),
                                   cache->cache_position, cache->max_cache_len, KVH, HD, 1);
            cb.commands().back().set("device_pos", (uint32_t)1);

            uint32_t total_seq = cache->cache_position + seq_len;
            cb.add_attention_cached(S("attn_out"), S("q"), k_cache, v_cache,
                                    H, KVH, HD, total_seq, cache->max_cache_len);
            cb.commands().back().set("device_pos", (uint32_t)1);
        }
    } else {
        // 4b. Full attention (no cache)
        cb.add_attention(S("attn_out"), S("q"), S("k"), S("v"), H, KVH, HD, true,
                        seq_len, seq_len);
    }
    
    // 5. Output projection
    cb.add_matmul(S("attn_proj"), S("attn_out"), W(arch, layer_idx, "attn.o_proj.weight"));
    cb.commands().back().set("seq_len", seq_len);
    
    if (arch.has_o_proj_bias) {
        cb.add_bias(S("attn_proj"), S("attn_proj"), W(arch, layer_idx, "attn.o_proj.bias"));
    }
    
    // 6. Residual (deferred to fused add_rmsnorm in mlp_block)
    // cb.add_add(S("residual"), S("hidden"), S("attn_proj")); // FUSED into mlp_block
}

// ============================================================================
// MLP BLOCK (polimórfico: gated / fused_gate / simple)
// ============================================================================

void GraphBuilder::build_mlp_block(
    CommandBuffer& cb,
    const ModelConfig& config,
    const ArchDescriptor& arch,
    uint32_t layer_idx,
    uint32_t batch_size,
    uint32_t seq_len
) {
    // 1. Post-attention / Pre-MLP norm
    std::string norm_w = W(arch, layer_idx, arch.post_attn_norm + ".weight");
    
    if (arch.norm_has_bias) {
        std::string norm_b = W(arch, layer_idx, arch.post_attn_norm + ".bias");
        cb.add_layernorm(S("normed"), S("residual"), norm_w, norm_b, config.rms_norm_eps());
    } else {
        cb.add_add_rmsnorm(S("residual"), S("normed"), S("hidden"), S("attn_proj"), norm_w, config.rms_norm_eps());
    }
    
    // 2. MLP (polimórfico según estructura detectada)
    if (arch.has_fused_gate_up) {
        // Fused gate_up: single matmul then split
        cb.add_matmul(S("gate_up_fused"), S("normed"),
                      W(arch, layer_idx, "mlp." + arch.mlp_gate_name + ".weight"));
        cb.commands().back().set("seq_len", seq_len);
        
        uint32_t I = config.intermediate_size();
        cb.add_split_half(S("gate"), S("up"), S("gate_up_fused"), I);
        cb.commands().back().set("seq_len", seq_len);
        
        // Gated activation: fused act(gate) * up
        add_gated_activation(cb, S("mlp_h"), S("gate"), S("up"), arch.activation);
        cb.add_matmul(S("mlp_out"), S("mlp_h"),
                      W(arch, layer_idx, "mlp." + arch.mlp_down_name + ".weight"));
    cb.commands().back().set("seq_len", seq_len);
    } else if (arch.has_gate) {
        // Separate gate and up: gate, up, act(gate) * up, down
        cb.add_matmul(S("gate"), S("normed"), 
                      W(arch, layer_idx, "mlp." + arch.mlp_gate_name + ".weight"));
    cb.commands().back().set("seq_len", seq_len);
        cb.add_matmul(S("up"), S("normed"),
                      W(arch, layer_idx, "mlp." + arch.mlp_up_name + ".weight"));
    cb.commands().back().set("seq_len", seq_len);
        
        // Gated activation: fused act(gate) * up
        add_gated_activation(cb, S("mlp_h"), S("gate"), S("up"), arch.activation);
        cb.add_matmul(S("mlp_out"), S("mlp_h"),
                      W(arch, layer_idx, "mlp." + arch.mlp_down_name + ".weight"));
    cb.commands().back().set("seq_len", seq_len);
    } else {
        // Simple MLP: up → activation → down (CLIP, older models)
        cb.add_matmul(S("up"), S("normed"),
                      W(arch, layer_idx, "mlp." + arch.mlp_up_name + ".weight"));
    cb.commands().back().set("seq_len", seq_len);
        
        add_activation(cb, S("gate_act"), S("up"), arch.activation);
        
        cb.add_matmul(S("mlp_out"), S("gate_act"),
                      W(arch, layer_idx, "mlp." + arch.mlp_down_name + ".weight"));
    cb.commands().back().set("seq_len", seq_len);
    }
    
    // 3. Residual
    cb.add_add(S("hidden"), S("residual"), S("mlp_out"));
}

// ============================================================================
// OUTPUT ACCESS
// ============================================================================

TensorInfo* GraphBuilder::get_logits(Engine& engine) const {
    return engine.tensors().get("_s.logits");
}

TensorInfo* GraphBuilder::get_hidden(Engine& engine) const {
    return engine.tensors().get("_s.hidden");
}

// ============================================================================
// VALIDATION
// ============================================================================

std::string GraphBuilder::validate_weights(
    const Engine& engine,
    const ModelConfig& config,
    const ArchDescriptor& arch
) const {
    std::ostringstream missing;
    int count = 0;
    
    auto check = [&](const std::string& name) {
        if (!engine.tensors().exists(name)) {
            if (count > 0) missing << ", ";
            missing << name;
            count++;
        }
    };
    
    // Global
    check(WG(arch, arch.embedding_name));
    check(WG(arch, arch.final_norm_name));
    if (!arch.lm_head_name.empty()) {
        check(WG(arch, arch.lm_head_name));
    }
    
    // Per layer
    for (uint32_t i = 0; i < arch.num_layers; i++) {
        check(W(arch, i, arch.pre_attn_norm + ".weight"));
        check(W(arch, i, arch.post_attn_norm + ".weight"));
        
        // QKV
        if (arch.has_fused_qkv) {
            check(W(arch, i, "attn.qkv_proj.weight"));
        } else {
            check(W(arch, i, "attn.q_proj.weight"));
            check(W(arch, i, "attn.k_proj.weight"));
            check(W(arch, i, "attn.v_proj.weight"));
        }
        check(W(arch, i, "attn.o_proj.weight"));
        
        // MLP
        if (arch.has_fused_gate_up) {
            check(W(arch, i, "mlp." + arch.mlp_gate_name + ".weight"));
        } else if (arch.has_gate) {
            check(W(arch, i, "mlp." + arch.mlp_gate_name + ".weight"));
            check(W(arch, i, "mlp." + arch.mlp_up_name + ".weight"));
        } else {
            check(W(arch, i, "mlp." + arch.mlp_up_name + ".weight"));
        }
        check(W(arch, i, "mlp." + arch.mlp_down_name + ".weight"));
    }
    
    if (count > 0) {
        return "Missing " + std::to_string(count) + " weight(s): " + missing.str();
    }
    return "";
}

void GraphBuilder::print_scratch_info(const Engine& engine) const {
    if (!scratch_allocated_) {
        std::cout << "GraphBuilder: no scratch allocated" << std::endl;
        return;
    }
    
    size_t total = 0;
    std::cout << "Scratch tensors:" << std::endl;
    for (const auto& name : scratch_names_) {
        auto* t = engine.tensors().get(name);
        if (t) {
            std::cout << "  " << name << " " << (t->size_bytes / 1024) << " KB" << std::endl;
            total += t->size_bytes;
        }
    }
    std::cout << "  TOTAL: " << (total / 1024 / 1024) << " MB" << std::endl;
}

// ============================================================================
// WEIGHT FUSION — Runtime concatenation of separate weights
// ============================================================================
// For models with separate Q/K/V weights, create fused qkv_proj.weight tensors.
// For models with separate gate/up weights, create fused gate_up.weight tensors.
// This turns 3 matmuls + 0 splits → 1 matmul + 1 split (net: 2 fewer kernel launches per layer).

void GraphBuilder::fuse_weights(Engine& engine, ArchDescriptor& arch, const ModelConfig& config) {
    auto& reg = engine.tensors();
    uint32_t H = config.num_attention_heads();
    uint32_t KVH = config.num_key_value_heads();
    uint32_t HD = config.head_dim();
    uint32_t D = config.hidden_size();
    uint32_t I = config.intermediate_size();
    
    // ========================================================================
    // Fuse Q+K+V → qkv_proj (if separate)
    // ========================================================================
    if (!arch.has_fused_qkv) {
        uint32_t q_dim = H * HD;
        uint32_t k_dim = KVH * HD;
        uint32_t v_dim = KVH * HD;
        uint32_t fused_dim = q_dim + k_dim + v_dim;
        
        bool all_same_dtype = true;
        DTypeID qkv_dtype = 0;
        
        for (uint32_t i = 0; i < arch.num_layers; i++) {
            std::string q_name = W(arch, i, "attn.q_proj.weight");
            std::string k_name = W(arch, i, "attn.k_proj.weight");
            std::string v_name = W(arch, i, "attn.v_proj.weight");
            
            TensorInfo* q = reg.get(q_name);
            TensorInfo* k = reg.get(k_name);
            TensorInfo* v = reg.get(v_name);
            
            if (!q || !k || !v) continue;
            
            if (i == 0) qkv_dtype = q->dtype;
            if (k->dtype != qkv_dtype || v->dtype != qkv_dtype) {
                all_same_dtype = false;
                break;
            }
        }
        
        if (all_same_dtype && qkv_dtype != 0) {
            // CRITICAL: Cannot fuse quantized tensors by simple byte concatenation!
            // HQ4K/HQ5K data is organized in superblocks per-row. Concatenating
            // Q_blob + K_blob + V_blob does NOT produce a valid [Q+K+V, K] quantized tensor.
            // The dequant kernel would index into wrong rows.
            // Only fuse FP16/FP32 tensors where rows are simple contiguous memory.
            bool is_quantized = (qkv_dtype == dtype::HQ4K() || qkv_dtype == dtype::HQ5K());
            
            if (is_quantized) {
                // printf("  [FUSE] Skipping QKV fusion for quantized dtype=%s (superblock layout incompatible)\n",
                //        dtype_name(qkv_dtype));
            } else {
            // printf("  [FUSE] Fusing Q+K+V weights → qkv_proj (%d layers, dtype=%s)\n",
            //        arch.num_layers, dtype_name(qkv_dtype));
            
            for (uint32_t i = 0; i < arch.num_layers; i++) {
                std::string q_name = W(arch, i, "attn.q_proj.weight");
                std::string k_name = W(arch, i, "attn.k_proj.weight");
                std::string v_name = W(arch, i, "attn.v_proj.weight");
                std::string fused_name = W(arch, i, "attn.qkv_proj.weight");
                
                TensorInfo* q = reg.get(q_name);
                TensorInfo* k = reg.get(k_name);
                TensorInfo* v = reg.get(v_name);
                
                if (!q || !k || !v) continue;
                
                // Allocate fused tensor
                size_t fused_bytes = q->size_bytes + k->size_bytes + v->size_bytes;
                void* fused_ptr = nullptr;
                cudaMalloc(&fused_ptr, fused_bytes);
                
                // Copy Q, K, V contiguously
                uint8_t* dst = static_cast<uint8_t*>(fused_ptr);
                cudaMemcpy(dst, q->ptr, q->size_bytes, cudaMemcpyDeviceToDevice);
                dst += q->size_bytes;
                cudaMemcpy(dst, k->ptr, k->size_bytes, cudaMemcpyDeviceToDevice);
                dst += k->size_bytes;
                cudaMemcpy(dst, v->ptr, v->size_bytes, cudaMemcpyDeviceToDevice);
                
                // Register fused tensor
                TensorInfo fused;
                fused.ptr = fused_ptr;
                fused.shape = {fused_dim, (uint32_t)(q->shape.size() > 1 ? q->shape[1] : D)};
                fused.dtype = qkv_dtype;
                fused.size_bytes = fused_bytes;
                fused.owns_memory = true;
                reg.register_tensor(fused_name, fused);
            }
            
            // Fuse biases too if present
            if (arch.has_qkv_bias) {
                for (uint32_t i = 0; i < arch.num_layers; i++) {
                    std::string qb = W(arch, i, "attn.q_proj.bias");
                    std::string kb = W(arch, i, "attn.k_proj.bias");
                    std::string vb = W(arch, i, "attn.v_proj.bias");
                    std::string fb = W(arch, i, "attn.qkv_proj.bias");
                    
                    TensorInfo* q = reg.get(qb);
                    TensorInfo* k = reg.get(kb);
                    TensorInfo* v = reg.get(vb);
                    
                    if (!q || !k || !v) continue;
                    
                    size_t fused_bytes = q->size_bytes + k->size_bytes + v->size_bytes;
                    void* fused_ptr = nullptr;
                    cudaMalloc(&fused_ptr, fused_bytes);
                    
                    uint8_t* dst = static_cast<uint8_t*>(fused_ptr);
                    cudaMemcpy(dst, q->ptr, q->size_bytes, cudaMemcpyDeviceToDevice);
                    dst += q->size_bytes;
                    cudaMemcpy(dst, k->ptr, k->size_bytes, cudaMemcpyDeviceToDevice);
                    dst += k->size_bytes;
                    cudaMemcpy(dst, v->ptr, v->size_bytes, cudaMemcpyDeviceToDevice);
                    
                    TensorInfo fused;
                    fused.ptr = fused_ptr;
                    fused.shape = {fused_dim};
                    fused.dtype = q->dtype;
                    fused.size_bytes = fused_bytes;
                    fused.owns_memory = true;
                    reg.register_tensor(fb, fused);
                }
            }
            
            arch.has_fused_qkv = true;
            // printf("  [FUSE] ✓ QKV fusion complete\n");
            } // end else (non-quantized)
        }
    }
    
    // ========================================================================
    // Fuse gate+up → gate_up (if separate)
    // ========================================================================
    if (!arch.has_fused_gate_up && arch.has_gate) {
        DTypeID gate_dtype = 0;
        bool ok = true;
        
        for (uint32_t i = 0; i < arch.num_layers && ok; i++) {
            std::string g = W(arch, i, "mlp." + arch.mlp_gate_name + ".weight");
            std::string u = W(arch, i, "mlp." + arch.mlp_up_name + ".weight");
            TensorInfo* gt = reg.get(g);
            TensorInfo* ut = reg.get(u);
            if (!gt || !ut) { ok = false; break; }
            if (i == 0) gate_dtype = gt->dtype;
            if (gt->dtype != gate_dtype || ut->dtype != gate_dtype) ok = false;
        }
        
        if (ok && gate_dtype != 0) {
            bool is_quantized = (gate_dtype == dtype::HQ4K() || gate_dtype == dtype::HQ5K());
            
            if (is_quantized) {
                // printf("  [FUSE] Skipping gate+up fusion for quantized dtype=%s\n",
                //        dtype_name(gate_dtype));
            } else {
            // printf("  [FUSE] Fusing gate+up → gate_up (%d layers, dtype=%s)\n",
            //        arch.num_layers, dtype_name(gate_dtype));
            
            for (uint32_t i = 0; i < arch.num_layers; i++) {
                std::string g_name = W(arch, i, "mlp." + arch.mlp_gate_name + ".weight");
                std::string u_name = W(arch, i, "mlp." + arch.mlp_up_name + ".weight");
                std::string fused_name = W(arch, i, "mlp.gate_up.weight");
                
                TensorInfo* gt = reg.get(g_name);
                TensorInfo* ut = reg.get(u_name);
                
                if (!gt || !ut) continue;
                
                size_t fused_bytes = gt->size_bytes + ut->size_bytes;
                void* fused_ptr = nullptr;
                cudaMalloc(&fused_ptr, fused_bytes);
                
                uint8_t* dst = static_cast<uint8_t*>(fused_ptr);
                cudaMemcpy(dst, gt->ptr, gt->size_bytes, cudaMemcpyDeviceToDevice);
                dst += gt->size_bytes;
                cudaMemcpy(dst, ut->ptr, ut->size_bytes, cudaMemcpyDeviceToDevice);
                
                uint32_t K_dim = gt->shape.size() > 1 ? gt->shape[1] : D;
                TensorInfo fused;
                fused.ptr = fused_ptr;
                fused.shape = {2 * I, K_dim};
                fused.dtype = gate_dtype;
                fused.size_bytes = fused_bytes;
                fused.owns_memory = true;
                reg.register_tensor(fused_name, fused);
            }
            
            arch.has_fused_gate_up = true;
            arch.mlp_gate_name = "gate_up";
            // printf("  [FUSE] ✓ gate_up fusion complete\n");
            } // end else (non-quantized)
        }
    }
}

} // namespace helios
