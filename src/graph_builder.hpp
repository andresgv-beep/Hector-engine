// src/graph_builder.hpp
// ============================================================================
// HELIOS ENGINE - GraphBuilder
// ============================================================================
// Genera CommandBuffers para forward pass de cualquier transformer.
// 
// DISEÑO POLIMÓRFICO:
//   - ArchDescriptor detectado desde tensores reales (nunca inventa nombres)
//   - ModelConfig consultado para semántica (activación, rope_type, dtype)
//   - Un solo build path para cached y non-cached (sin duplicación)
//   - Scratch dtype parametrizado desde modelo
//
// Data flow genérico (por capa):
//   hidden → pre_norm → normed
//   normed → QKV (separado o fusionado) → +bias?
//   Q, K → RoPE (según tipo)
//   [opcional: KV cache update]
//   Q, K, V → Attention → attn_out
//   attn_out → o_proj → attn_proj
//   hidden + attn_proj → residual
//   residual → post_norm → normed2
//   normed2 → MLP (según variante) → mlp_out
//   residual + mlp_out → hidden (next layer)
//

#pragma once

#include "engine.hpp"
#include "hnf_loader.hpp"
#include <string>
#include <vector>
#include <functional>

namespace helios {

// ============================================================================
// ACTIVATION TYPE (polimórfico - leído de ModelConfig, no hardcoded)
// ============================================================================

enum class ActivationType : uint8_t {
    SILU = 0,       // SwiGLU default
    GELU,           // Standard GELU
    GELU_NEW,       // Approximate GELU (tanh)
    RELU,           // ReLU
};

// ============================================================================
// ROPE TYPE (polimórfico - leído de ModelConfig)
// ============================================================================

enum class RoPEType : uint8_t {
    DEFAULT = 0,    // Standard rotary
    LLAMA3,         // LLaMA-3 scaled
    LINEAR,         // Linear scaling
    DYNAMIC,        // Dynamic NTK
    YARN,           // YaRN
    LONGROPE,       // LongRoPE (Phi-4)
    SU,             // SuRoPE
    NONE,           // No positional encoding
};

// ============================================================================
// KV CACHE PARAMS (optional, passed to forward builder)
// ============================================================================

struct KVCacheParams {
    std::string kv_prefix;       // "_kv" → "_kv.layer{N}.k/v"
    uint32_t cache_position;     // Current write position
    uint32_t max_cache_len;      // Maximum cache length
};

// ============================================================================
// ARCHITECTURE DESCRIPTOR
// ============================================================================
// Detectado automáticamente desde los nombres de tensores en el HNF,
// enriquecido con semántica de ModelConfig.
// El GraphBuilder NO asume nada - lee lo que hay.

struct ArchDescriptor {
    std::string prefix;              // "text", "code", "vision", "cortex"
    uint32_t num_layers = 0;
    
    // --- Detectado de tensores (ground truth) ---
    
    // Attention layout
    bool has_qkv_bias = false;       // ¿Tiene q_proj.bias?
    bool has_fused_qkv = false;      // ¿Usa qkv_proj en vez de q/k/v separados?
    bool has_o_proj_bias = false;     // ¿Tiene o_proj.bias?
    bool use_qk_norm = false;        // ¿RMSNorm por-head en Q/K antes de RoPE? (Qwen3)

    // MLP layout
    bool has_gate = false;           // ¿Tiene mlp.gate? (SwiGLU)
    bool has_fused_gate_up = false;  // ¿Usa gate_up fusionado?
    bool has_mlp_bias = false;       // ¿Tiene bias en capas MLP?
    std::string mlp_up_name;         // "up" o "fc1"
    std::string mlp_down_name;       // "down" o "fc2"
    std::string mlp_gate_name;       // "gate" o "gate_up" o ""
    
    // Norm layout
    std::string pre_attn_norm;       // "ln_attn_in" o "ln1"
    std::string post_attn_norm;      // "ln_attn_out" o "ln2"
    bool norm_has_bias = false;      // LayerNorm vs RMSNorm
    
    // Global tensors (detectados, no inventados)
    std::string embedding_name;      // "token_embedding.weight" o "embed_tokens.weight"
    std::string final_norm_name;     // "final_norm.weight" o "norm.weight"
    std::string final_norm_bias;     // "final_norm.bias" o "" si no existe
    std::string lm_head_name;        // "" si tie_embeddings, "lm_head.weight" si no
    
    // --- Enriquecido desde ModelConfig ---
    
    // Activación MLP (default: SiLU para gated, GELU para non-gated)
    ActivationType activation = ActivationType::SILU;
    
    // RoPE config
    RoPEType rope_type = RoPEType::DEFAULT;
    float rope_theta = 10000.0f;
    float rope_scaling_factor = 1.0f;      // Linear/Dynamic scaling (DeepSeek=4.0)
    float partial_rotary_factor = 1.0f;    // < 1.0 para Phi-4 LongRoPE
    
    // DType para scratch tensors (match modelo)
    DTypeID compute_dtype = 0;       // 0 = auto-detect from weights
};

// ============================================================================
// CACHED DECODE STATE
// ============================================================================
// Almacena el CommandBuffer del primer decode token para reutilizarlo.
// Solo se actualizan los params que cambian (cache_pos, rope_offset, seq_len).

struct CachedDecode {
    CommandBuffer cb;
    uint32_t batch_size = 0;
    uint32_t seq_len = 0;
    bool valid = false;
    
    // Índices de los commands que contienen params dependientes de cache_pos.
    // Pre-calculados en la primera build para O(K) update donde K << total commands.
    struct CachePosRef {
        size_t cmd_idx;          // Índice en cb.commands()
        std::string param_name;  // "offset", "position", "seq_len"
        enum Kind { DIRECT, PLUS_SEQ } kind;  // DIRECT=cache_pos, PLUS_SEQ=cache_pos+seq_len
    };
    std::vector<CachePosRef> refs;
    
    void invalidate() { valid = false; refs.clear(); }
};

// ============================================================================
// GRAPH BUILDER
// ============================================================================

class GraphBuilder {
public:
    GraphBuilder() = default;
    
    // ========================================================================
    // ARCHITECTURE DETECTION
    // ========================================================================
    
    // Detectar arquitectura desde tensores cargados + enriquecer con config
    // Llamar DESPUÉS de load_block()
    ArchDescriptor detect_architecture(
        const Engine& engine,
        const std::string& prefix,
        const ModelConfig& config = ModelConfig{}
    ) const;
    
    // ========================================================================
    // SCRATCH ALLOCATION
    // ========================================================================
    
    void allocate_scratch(
        Engine& engine,
        const ModelConfig& config,
        const ArchDescriptor& arch,
        uint32_t max_batch = 1,
        uint32_t max_seq = 1
    );

    // Explicit Gemma 4 scratch. Per-layer aliases carry exact GM4X shapes
    // while sharing max-sized backing allocations sequentially.
    void allocate_gemma4_scratch(
        Engine& engine,
        const ModelConfig& config,
        const Gemma4Config& gemma,
        const ArchDescriptor& arch,
        uint32_t max_batch = 1,
        uint32_t max_seq = 1
    );
    
    void free_scratch(Engine& engine);
    
    bool is_allocated() const { return scratch_allocated_; }
    
    // ========================================================================
    // WEIGHT FUSION (runtime optimization)
    // ========================================================================
    // For models with separate Q/K/V and gate/up weights, concatenate them
    // into fused tensors at load time. This reduces 3 matmuls → 1 matmul + split.
    // Must be called AFTER detect_architecture, BEFORE allocate_scratch.
    // Modifies arch in-place to reflect the fusion.
    
    void fuse_weights(Engine& engine, ArchDescriptor& arch, const ModelConfig& config);
    
    // ========================================================================
    // FORWARD PASS (unificado)
    // ========================================================================
    
    // Forward completo: embedding → layers → final_norm → logits
    // Si cache != nullptr, usa KV cache autoregresivo
    CommandBuffer build_forward(
        Engine& engine,
        const ModelConfig& config,
        const ArchDescriptor& arch,
        const std::string& input_tokens,
        uint32_t batch_size,
        uint32_t seq_len,
        uint32_t position_offset = 0,
        const KVCacheParams* cache = nullptr
    );
    
    // ========================================================================
    // FORWARD PASS CON REUSE (decode optimizado)
    // ========================================================================
    // Primera llamada decode: build completo + cachea CB + indexa refs.
    // Siguientes: solo actualiza cache_pos params. Retorna ref (no copia).
    
    const CommandBuffer& build_forward_reuse(
        Engine& engine,
        const ModelConfig& config,
        const ArchDescriptor& arch,
        const std::string& input_tokens,
        uint32_t batch_size,
        uint32_t seq_len,
        uint32_t position_offset = 0,
        const KVCacheParams* cache = nullptr
    );
    
    // Invalidar cache de decode (llamar si cambia modelo, config, o prefill)
    void invalidate_decode_cache() { cached_decode_.invalidate(); }
    
    // Una sola capa (para testing)
    CommandBuffer build_single_layer(
        Engine& engine,
        const ModelConfig& config,
        const ArchDescriptor& arch,
        uint32_t layer_idx,
        uint32_t batch_size,
        uint32_t seq_len,
        uint32_t position_offset = 0,
        const KVCacheParams* cache = nullptr
    );

    // Gemma 4 input path: scaled main embedding plus complete packed PLE.
    CommandBuffer build_gemma4_input(
        Engine& engine,
        const ModelConfig& config,
        const Gemma4Config& gemma,
        const ArchDescriptor& arch,
        const std::string& input_tokens,
        uint32_t batch_size,
        uint32_t seq_len
    );

    // One exact non-shared Gemma 4 layer. Local attention is valid here only
    // while the supplied sequence does not exceed its sliding window; the
    // long-context mask and shared KV semantics belong to Phase 6.
    CommandBuffer build_gemma4_single_layer(
        Engine& engine,
        const ModelConfig& config,
        const Gemma4Config& gemma,
        const ArchDescriptor& arch,
        uint32_t layer_idx,
        uint32_t batch_size,
        uint32_t seq_len,
        uint32_t position_offset = 0
    );

    // Exact cached Gemma 4 layer. Non-shared layers project and update their
    // own K/V; shared layers only project Q and consume the cache alias
    // registered for their local/global source layer.
    CommandBuffer build_gemma4_layer_cached(
        Engine& engine,
        const ModelConfig& config,
        const Gemma4Config& gemma,
        const ArchDescriptor& arch,
        uint32_t layer_idx,
        uint32_t batch_size,
        uint32_t seq_len,
        const KVCacheParams& cache
    );

    // Complete cached text forward for Gemma 4: scaled embedding + PLE,
    // heterogeneous/shared-KV layers, final norm, LM head and logit softcap.
    CommandBuffer build_gemma4_forward_cached(
        Engine& engine,
        const ModelConfig& config,
        const Gemma4Config& gemma,
        const ArchDescriptor& arch,
        const std::string& input_tokens,
        uint32_t batch_size,
        uint32_t seq_len,
        const KVCacheParams& cache
    );

    // Attention-independent tail shared by normal and shared-KV layers. This
    // permits validating double-wide MLP+PLE before Phase 6 supplies shared KV.
    CommandBuffer build_gemma4_mlp_ple_tail(
        Engine& engine,
        const ModelConfig& config,
        const Gemma4Config& gemma,
        const ArchDescriptor& arch,
        uint32_t layer_idx,
        uint32_t batch_size,
        uint32_t seq_len
    );
    
    // ========================================================================
    // BACKWARD COMPAT WRAPPER
    // ========================================================================
    
    // Forward con cache (old signature → delegates to unified build_forward)
    CommandBuffer build_forward_cached(
        Engine& engine,
        const ModelConfig& config,
        const ArchDescriptor& arch,
        const std::string& input_tokens,
        uint32_t batch_size,
        uint32_t seq_len,
        const std::string& kv_prefix,
        uint32_t cache_position,
        uint32_t max_cache_len
    ) {
        KVCacheParams cache{kv_prefix, cache_position, max_cache_len};
        return build_forward(engine, config, arch, input_tokens,
                           batch_size, seq_len, cache_position, &cache);
    }
    
    // ========================================================================
    // OUTPUT ACCESS
    // ========================================================================
    
    TensorInfo* get_logits(Engine& engine) const;
    TensorInfo* get_hidden(Engine& engine) const;
    
    // ========================================================================
    // VALIDATION
    // ========================================================================
    
    std::string validate_weights(
        const Engine& engine,
        const ModelConfig& config,
        const ArchDescriptor& arch
    ) const;
    
    void print_scratch_info(const Engine& engine) const;

private:
    // ========================================================================
    // LAYER BUILDERS (un solo path, polimórfico)
    // ========================================================================
    
    // Attention block: norm → QKV → RoPE → [cache?] → attention → o_proj → residual
    // cache puede ser nullptr (no-cache) o apuntar a KVCacheParams (autoregresivo)
    void build_attention_block(
        CommandBuffer& cb,
        const ModelConfig& config,
        const ArchDescriptor& arch,
        uint32_t layer_idx,
        uint32_t batch_size,
        uint32_t seq_len,
        uint32_t position_offset,
        const KVCacheParams* cache
    );
    
    // MLP block: norm → MLP variant → residual
    void build_mlp_block(
        CommandBuffer& cb,
        const ModelConfig& config,
        const ArchDescriptor& arch,
        uint32_t layer_idx,
        uint32_t batch_size,
        uint32_t seq_len
    );

    void append_gemma4_mlp_ple_tail(
        CommandBuffer& cb,
        const ModelConfig& config,
        const Gemma4Config& gemma,
        const ArchDescriptor& arch,
        uint32_t layer_idx,
        uint32_t seq_len
    );
    
    // ========================================================================
    // ACTIVATION (polimórfico dispatch)
    // ========================================================================
    
    void add_activation(
        CommandBuffer& cb,
        const std::string& dst,
        const std::string& src,
        ActivationType type
    );
    
    // Fused activation × mul: act(gate) * up → dst
    void add_gated_activation(
        CommandBuffer& cb,
        const std::string& dst,
        const std::string& gate,
        const std::string& up,
        ActivationType type
    );
    
    // ========================================================================
    // NAMING (basado en arch descriptor, no hardcoded)
    // ========================================================================
    
    std::string W(const ArchDescriptor& arch, uint32_t layer, 
                  const std::string& component) const;
    std::string WG(const ArchDescriptor& arch, const std::string& name) const;
    static std::string S(const std::string& name);
    static std::string G4(uint32_t layer, const std::string& name);
    void set_active_scratch_shape(Engine& engine, uint32_t batch_size,
                                  uint32_t seq_len);
    
    // ========================================================================
    // STATE
    // ========================================================================
    
    bool scratch_allocated_ = false;
    uint32_t alloc_batch_ = 0;
    uint32_t alloc_seq_ = 0;
    std::vector<std::string> scratch_names_;
    std::vector<std::string> scratch_view_names_;
    
    // Decode cache para CommandBuffer reuse
    CachedDecode cached_decode_;
};

} // namespace helios
