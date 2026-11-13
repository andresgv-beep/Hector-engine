// hnf_loader.hpp
// ============================================================================
// HNF v9 LOADER - Carga modelos HELIOS Neural Format
// ============================================================================
// Lee HNF v9, carga tensores a GPU, registra en TensorRegistry.
// Los pesos HQ4.1K/HQ5.1K se cargan SIN dequantizar (fused en kernel).
//

#pragma once

#include "engine.hpp"
#include "htf_tokenizer.hpp"
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <fstream>
#include <utility>

namespace helios {

class MappedWeightPipeline;

// ============================================================================
// HNF CONSTANTS
// ============================================================================

// El contenedor es HNFv10 (src/hnf10.hpp). Sus componentes se colocan en estas ranuras
// internas de bloque, que ya no corresponden a ninguna tabla del fichero.
constexpr uint32_t HNF_BLOCK_COUNT = 16;
constexpr uint32_t HNF_ALIGNMENT = 32;

// ============================================================================
// RANURAS INTERNAS DE BLOQUE
// ============================================================================

enum BlockID : uint32_t {
    BLOCK_TEXT_MODEL      = 0x0,
    BLOCK_VISION          = 0x1,
    BLOCK_AUDIO           = 0x2,
    BLOCK_VIDEO           = 0x3,
    BLOCK_SPATIAL_3D      = 0x4,
    BLOCK_PERSONALITY     = 0x5,
    BLOCK_MEMORY          = 0x6,
    BLOCK_CORTEX          = 0x7,
    BLOCK_CODE_EXEC       = 0x8,
    BLOCK_TOKENIZER       = 0x9,   // HTF multi-domain
    BLOCK_EXEC_HINTS      = 0xA,   // hints JSON de HNFv9: ya no existe
    BLOCK_EXEC_HINTS_BIN  = 0xB,   // Binario (preferido) ← NUEVO
    BLOCK_TOOLS           = 0xC,
    BLOCK_EXPERT_ROUTER   = 0xD,
    BLOCK_RESERVED_0      = 0xE,
    BLOCK_RESERVED_1      = 0xF,
};

inline const char* block_name(uint32_t id) {
    static const char* names[16] = {
        "text_model", "vision", "audio", "video",
        "spatial_3d", "personality", "memory", "cortex",
        "code_exec", "tokenizer", "execution_hints", "exec_hints_bin",
        "tools", "expert_router", "reserved_0", "reserved_1"
    };
    return (id < 16) ? names[id] : "unknown";
}

// ============================================================================
// BINARY STRUCTURES (packed, little-endian)
// ============================================================================

#pragma pack(push, 1)

// Dónde está un componente en el fichero; `checksum` ya no se usa (los hashes son de HNFv10).
struct BlockEntry {
    uint32_t block_id;
    uint32_t block_type;
    uint64_t offset;
    uint64_t size;
    uint64_t checksum;              // XXH3-64
};
static_assert(sizeof(BlockEntry) == 32, "BlockEntry must be 32 bytes");

// ============================================================================
// EXECUTION HINTS BINARY STRUCTURES (block 0xB)
// ============================================================================
// La configuración binaria es la única fuente de configuración.

constexpr uint32_t HINTS_MAGIC = 0x48494E54;  // "HINT"

// Arch enum
enum ExecArch : uint32_t {
    ARCH_UNKNOWN = 0,
    ARCH_LLAMA = 1, ARCH_LLAMA2 = 2, ARCH_LLAMA3 = 3,
    ARCH_QWEN = 4, ARCH_QWEN2 = 5,
    ARCH_PHI3 = 6, ARCH_PHI4 = 7,
    ARCH_GEMMA = 8, ARCH_GEMMA2 = 9,
    ARCH_MISTRAL = 10, ARCH_MIXTRAL = 11,
    ARCH_DEEPSEEK = 12,
    ARCH_CLIP = 13, ARCH_SIGLIP = 14,
    ARCH_GEMMA4 = 15,
};

// DType enum
enum ExecDType : uint32_t {
    DTYPE_FP16 = 0, DTYPE_BF16 = 1, DTYPE_FP32 = 2,
};

// AttentionType enum
enum ExecAttnType : uint32_t {
    ATTN_MHA = 0, ATTN_GQA = 1, ATTN_MQA = 2,
};

// MLPType enum
enum ExecMLPType : uint32_t {
    MLP_SWIGLU = 0, MLP_SWIGLU_FUSED = 1, MLP_GEGLU = 2, MLP_GATED = 3, MLP_STANDARD = 4,
};

// NormType enum
enum ExecNormType : uint32_t {
    NORM_RMSNORM = 0, NORM_LAYERNORM = 1,
};

// RoPEType enum
enum ExecRoPEType : uint32_t {
    ROPE_DEFAULT = 0, ROPE_LLAMA3 = 1, ROPE_LINEAR = 2, ROPE_DYNAMIC = 3,
    ROPE_YARN = 4, ROPE_LONGROPE = 5, ROPE_SU = 6, ROPE_NONE = 7,
    ROPE_PROPORTIONAL = 8,
};

// Flags for TextModelConfigBin
constexpr uint32_t CFG_FLAG_ATTENTION_BIAS     = 0x0001;
constexpr uint32_t CFG_FLAG_MLP_BIAS           = 0x0002;
constexpr uint32_t CFG_FLAG_NORM_BIAS          = 0x0004;
constexpr uint32_t CFG_FLAG_USE_QK_NORM        = 0x0008;
constexpr uint32_t CFG_FLAG_PARALLEL_ATTENTION = 0x0010;
constexpr uint32_t CFG_FLAG_TIE_EMBEDDINGS     = 0x0020;

struct ExecutionHintsBin {
    uint32_t magic;                  // 0x48494E54 = "HINT"
    uint16_t version_major;
    uint16_t version_minor;
    
    uint32_t text_offset;            // Offset a TextModelConfigBin
    uint32_t vision_offset;
    uint32_t audio_offset;
    uint32_t code_offset;
    uint32_t cortex_offset;
    uint32_t spatial_offset;
    
    uint16_t num_text_models;
    uint16_t num_vision_models;
    uint16_t num_audio_models;
    uint16_t num_code_models;
    
    uint32_t flags;
    uint8_t  reserved[20];
};
static_assert(sizeof(ExecutionHintsBin) == 64, "ExecutionHintsBin must be 64 bytes");

struct TextModelConfigBin {
    // Floats (24 bytes)
    float rope_theta;
    float rope_scaling_factor;
    float partial_rotary_factor;
    float rms_norm_eps;
    float layer_norm_eps;
    float reserved_float;
    
    // Dimensions (24 bytes)
    uint32_t num_hidden_layers;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t vocab_size;
    uint32_t max_position_embeddings;
    uint32_t rope_dim;
    
    // Attention (20 bytes)
    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;
    uint32_t head_dim;
    uint32_t attention_type;         // ExecAttnType
    uint32_t qkv_layout;             // 0=separate, 1=fused
    
    // Identity + Types (16 bytes)
    uint32_t arch;                   // ExecArch
    uint32_t dtype;                  // ExecDType
    uint32_t mlp_type;               // ExecMLPType
    uint32_t mlp_activation;         // 0=silu, 1=gelu, etc.
    
    // More types (8 bytes)
    uint32_t norm_type;              // ExecNormType
    uint32_t rope_type;              // ExecRoPEType
    
    // Flags (4 bytes)
    uint32_t flags;
    
    // Reserved (32 bytes) - para total 128 bytes
    uint8_t reserved[32];
};
static_assert(sizeof(TextModelConfigBin) == 128, "TextModelConfigBin must be 128 bytes");

struct VisionModelConfigBin {
    uint32_t encoder_type;
    uint32_t image_size;
    uint32_t patch_size;
    uint32_t hidden_size;
    uint32_t num_hidden_layers;
    uint32_t num_attention_heads;
    uint32_t intermediate_size;
    uint32_t num_channels;
    float    layer_norm_eps;
    uint32_t projection_dim;
    uint32_t projector_type;
    uint32_t num_image_tokens;
    int32_t  image_token_id;
    uint32_t flags;
    uint8_t  reserved[8];
};
static_assert(sizeof(VisionModelConfigBin) == 64, "VisionModelConfigBin must be 64 bytes");

enum VisionEncoderType : uint32_t {
    VISION_ENCODER_CLIP = 0,
    VISION_ENCODER_SIGLIP = 1,
    VISION_ENCODER_VIT = 2,
    VISION_ENCODER_EVA = 3,
    VISION_ENCODER_GEMMA4 = 4,
};

// Fixed Gemma 4 vision extension. ExecutionHintsBin.reserved[12..16] stores
// its offset and reserved[16..20] its size. It intentionally does not reuse
// the GM4X text-extension pointer.
struct Gemma4VisionExtensionBin {
    char     magic[4];
    uint16_t version;
    uint16_t record_size;
    uint32_t flags;
    uint32_t num_key_value_heads;
    uint32_t head_dim;
    uint32_t position_embedding_size;
    uint32_t pooling_kernel_size;
    uint32_t max_soft_tokens;
    uint32_t projection_dim;
    uint32_t patch_size;
    int32_t  image_token_id;
    int32_t  boi_token_id;
    int32_t  eoi_token_id;
    int32_t  pad_token_id;
    float    rope_theta;
    float    attention_scale;
    float    rms_norm_eps;
    float    rescale_factor;
    uint32_t resize_multiple;
    uint32_t resample;
    uint32_t activation;
    uint32_t patch_order;
    uint32_t position_order;
    uint32_t reserved;
};
static_assert(sizeof(Gemma4VisionExtensionBin) == 96,
              "Gemma4VisionExtensionBin must be 96 bytes");

// Gemma 4 extension stored inside block 0xB. The parent hint header points to
// this structure through reserved[0..8] and carries a second "GM4X" marker in
// reserved[8..12]. All fields are little-endian on disk.
struct Gemma4ExtensionHeaderBin {
    char     magic[4];
    uint16_t version;
    uint16_t layer_record_size;
    uint32_t layer_count;
    uint32_t flags;
    uint32_t global_head_dim;
    uint32_t ple_hidden_size;
    uint32_t num_kv_shared_layers;
    uint32_t reserved;
};
static_assert(sizeof(Gemma4ExtensionHeaderBin) == 32,
              "Gemma4ExtensionHeaderBin must be 32 bytes");

struct Gemma4LayerConfigBin {
    uint32_t attention_kind;         // 0=sliding, 1=full
    uint32_t sliding_window;
    uint32_t head_dim;
    uint32_t intermediate_size;
    uint32_t rope_type;              // ExecRoPEType
    uint32_t flags;
    float    rope_theta;
    float    partial_rotary_factor;
    int32_t  kv_share_group;         // -1 until sharing groups are assigned
    // KV heads DE ESTA CAPA.  Gemma 4 «unified» mezcla GQA en las deslizantes
    // con una sola KV head en las globales.  0 = el HNF es anterior a este
    // campo y manda el valor global del TextModelConfig.
    uint32_t num_kv_heads;
};
static_assert(sizeof(Gemma4LayerConfigBin) == 40,
              "Gemma4LayerConfigBin must be 40 bytes");

// GM4X v2 describes the MoE experts. The v1 header is kept as a prefix, with
// base.reserved holding the size of this header.
struct Gemma4ExtensionV2HeaderBin {
    Gemma4ExtensionHeaderBin base;
    uint32_t descriptor_size;
    uint32_t descriptors_per_layer;
    uint64_t extension_size;
    uint32_t expert_order;           // 1 = packed by expert
    uint32_t gate_up_order;          // 1 = gate before up inside each expert
    uint64_t reserved;
};
static_assert(sizeof(Gemma4ExtensionV2HeaderBin) == 64,
              "Gemma4ExtensionV2HeaderBin must be 64 bytes");

// Physical columns are the logical ones rounded up to 256; the extra columns
// are zeros and the engine must feed zeros in the matching input tail.
struct Gemma4MoeMatrixBin {
    uint32_t role;                   // Gemma4MoeRole
    uint32_t storage;                // Gemma4MoeStorage
    uint32_t rows;
    uint32_t logical_cols;
    uint32_t physical_rows;
    uint32_t physical_cols;
    uint32_t experts;
    uint32_t layout;                 // 0=float, 1=HQ44
    uint64_t row_bytes;
    uint64_t expert_stride;
};
static_assert(sizeof(Gemma4MoeMatrixBin) == 48,
              "Gemma4MoeMatrixBin must be 48 bytes");

struct Gemma4LayerConfigV2Bin {
    Gemma4LayerConfigBin base;
    uint32_t experts;
    uint32_t top_k;
    uint32_t intermediate_size;
    uint32_t descriptor_mask;        // Bit (role - 1) set per matrix present
    uint32_t descriptor_count;
    uint32_t reserved;
    Gemma4MoeMatrixBin gate_up;
    Gemma4MoeMatrixBin down;
    Gemma4MoeMatrixBin shared_down;
};
static_assert(sizeof(Gemma4LayerConfigV2Bin) == 208,
              "Gemma4LayerConfigV2Bin must be 208 bytes");

#pragma pack(pop)

enum Gemma4MoeRole : uint32_t {
    GEMMA4_MOE_ROLE_GATE_UP = 1,
    GEMMA4_MOE_ROLE_DOWN = 2,
    GEMMA4_MOE_ROLE_SHARED_DOWN = 3,
};

enum Gemma4MoeStorage : uint32_t {
    GEMMA4_MOE_STORAGE_FP16 = 0,
    GEMMA4_MOE_STORAGE_BF16 = 1,
    GEMMA4_MOE_STORAGE_FP32 = 2,
    GEMMA4_MOE_STORAGE_HQ44 = 3,
    GEMMA4_MOE_STORAGE_HQ8 = 4,     // solo down compartido: los kernels MoE leen FP16 o HQ44
};

constexpr uint32_t GEMMA4_EXT_FLAG_PLE              = (1u << 0);
constexpr uint32_t GEMMA4_EXT_FLAG_LAYER_SCALAR     = (1u << 1);
constexpr uint32_t GEMMA4_EXT_FLAG_LOGIT_SOFTCAP    = (1u << 2);
constexpr uint32_t GEMMA4_EXT_FLAG_SHARED_KV        = (1u << 3);
constexpr uint32_t GEMMA4_EXT_FLAG_DOUBLE_WIDE_MLP  = (1u << 4);
constexpr uint32_t GEMMA4_EXT_FLAG_FOUR_NORM_BLOCK  = (1u << 5);
constexpr uint32_t GEMMA4_EXT_FLAG_MOE              = (1u << 6);

// Flags de Gemma4LayerConfigBin.flags (por CAPA, no confundir con los EXT del
// modelo).  Bit 0: la capa no tiene v_proj y reutiliza la proyeccion K como V
// (`attention_k_eq_v`).
constexpr uint32_t GEMMA4_LAYER_FLAG_K_EQ_V = (1u << 0);

constexpr uint32_t GEMMA4_VISION_FLAG_CLIPPED_LINEARS  = (1u << 0);
constexpr uint32_t GEMMA4_VISION_FLAG_STANDARDIZE      = (1u << 1);
constexpr uint32_t GEMMA4_VISION_FLAG_DYNAMIC_TOKENS   = (1u << 2);
constexpr uint32_t GEMMA4_VISION_FLAG_LEARNED_XY_POS   = (1u << 3);
constexpr uint32_t GEMMA4_VISION_FLAG_ROPE_2D          = (1u << 4);
constexpr uint32_t GEMMA4_VISION_FLAG_NONCAUSAL        = (1u << 5);
constexpr uint32_t GEMMA4_VISION_FLAG_CONVERT_RGB      = (1u << 6);
constexpr uint32_t GEMMA4_VISION_FLAG_RESCALE          = (1u << 7);
constexpr uint32_t GEMMA4_VISION_FLAG_RESIZE           = (1u << 8);
constexpr uint32_t GEMMA4_VISION_FLAG_NORMALIZE        = (1u << 9);
constexpr uint32_t GEMMA4_VISION_FLAG_ANTIALIAS        = (1u << 10);
constexpr uint32_t GEMMA4_VISION_FLAG_QK_NORM_WEIGHTED = (1u << 11);
constexpr uint32_t GEMMA4_VISION_FLAG_V_NORM_WEIGHTLESS = (1u << 12);
constexpr uint32_t GEMMA4_VISION_FLAG_POOL_NORM_WEIGHTLESS = (1u << 13);
constexpr uint32_t GEMMA4_VISION_FLAG_POOL_FP32_SQRT_HIDDEN = (1u << 14);
constexpr uint32_t GEMMA4_VISION_FLAG_INPUT_MINUS_ONE_ONE = (1u << 15);

struct Gemma4MoeMatrix {
    uint32_t storage = GEMMA4_MOE_STORAGE_FP16;
    uint32_t rows = 0;
    uint32_t logical_cols = 0;
    uint32_t physical_cols = 0;
    uint32_t experts = 0;
    uint64_t row_bytes = 0;
    uint64_t expert_stride = 0;

    uint64_t expert_offset(uint32_t expert) const {
        return uint64_t{expert} * expert_stride;
    }
};

struct Gemma4LayerConfig {
    uint32_t attention_kind = 0;
    uint32_t sliding_window = 0;
    uint32_t head_dim = 0;
    uint32_t intermediate_size = 0;
    uint32_t rope_type = ROPE_DEFAULT;
    uint32_t flags = 0;
    float rope_theta = 10000.0f;
    float partial_rotary_factor = 1.0f;
    int32_t kv_share_group = -1;
    uint32_t num_kv_heads = 0;   // 0 = usar el valor global del modelo
    // Solo con GM4X v2.
    Gemma4MoeMatrix moe_gate_up;
    Gemma4MoeMatrix moe_down;
    Gemma4MoeMatrix moe_shared_down;

    bool is_global_attention() const { return attention_kind == 1; }
    bool k_eq_v() const { return (flags & GEMMA4_LAYER_FLAG_K_EQ_V) != 0; }
    /// KV heads efectivos: los de la capa si el HNF los trae, si no el global.
    uint32_t kv_heads_or(uint32_t global) const {
        return num_kv_heads != 0 ? num_kv_heads : global;
    }
};

struct Gemma4Config {
    uint16_t version = 0;
    uint32_t flags = 0;
    uint32_t global_head_dim = 0;
    uint32_t ple_hidden_size = 0;
    uint32_t num_kv_shared_layers = 0;
    // Expertos por capa, expertos usados por token y filas de la matriz gate/up.
    uint32_t moe_experts = 0;
    uint32_t moe_top_k = 0;
    uint32_t moe_intermediate_size = 0;
    std::vector<Gemma4LayerConfig> layers;

    bool has_flag(uint32_t flag) const { return (flags & flag) != 0; }
};

struct Gemma4VisionConfig {
    uint16_t version = 0;
    uint32_t flags = 0;
    uint32_t num_key_value_heads = 0;
    uint32_t head_dim = 0;
    uint32_t position_embedding_size = 0;
    uint32_t pooling_kernel_size = 0;
    uint32_t max_soft_tokens = 0;
    uint32_t projection_dim = 0;
    uint32_t patch_size = 0;
    int32_t image_token_id = -1;
    int32_t boi_token_id = -1;
    int32_t eoi_token_id = -1;
    int32_t pad_token_id = -1;
    float rope_theta = 0.0f;
    float attention_scale = 0.0f;
    float rms_norm_eps = 0.0f;
    float rescale_factor = 0.0f;
    uint32_t resize_multiple = 0;
    uint32_t resample = 0;
    uint32_t activation = 0;
    uint32_t patch_order = 0;
    uint32_t position_order = 0;

    bool has_flag(uint32_t flag) const { return (flags & flag) != 0; }
    uint32_t max_patches() const {
        return max_soft_tokens * pooling_kernel_size * pooling_kernel_size;
    }
};

// ============================================================================
// MODEL CONFIG (from execution_hints)
// ============================================================================

// ============================================================================
// CONFIG VALUE - Generic value holder for execution hints
// ============================================================================

#include <variant>
#include <unordered_map>

using ConfigValue = std::variant<int64_t, double, std::string, bool>;

// ============================================================================
// MODEL CONFIG - Generic parameter storage
// ============================================================================
// El motor NO asume qué parámetros existen - HEXOS/HERA definen.
// Cada arquitectura (Transformer, SSM, RWKV, etc.) usa sus propios parámetros.

struct ModelConfig {
    // Generic parameter storage
    std::unordered_map<std::string, ConfigValue> params;
    
    // ========================================
    // TYPED ACCESSORS (convenience, not hardcoding)
    // ========================================
    
    template<typename T>
    T get(const std::string& key, T default_val) const {
        auto it = params.find(key);
        if (it == params.end()) return default_val;
        
        if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, int> || 
                      std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) {
            if (auto* v = std::get_if<int64_t>(&it->second)) 
                return static_cast<T>(*v);
        } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            if (auto* v = std::get_if<double>(&it->second)) 
                return static_cast<T>(*v);
            if (auto* v = std::get_if<int64_t>(&it->second)) 
                return static_cast<T>(*v);
        } else if constexpr (std::is_same_v<T, std::string>) {
            if (auto* v = std::get_if<std::string>(&it->second)) 
                return *v;
        } else if constexpr (std::is_same_v<T, bool>) {
            if (auto* v = std::get_if<bool>(&it->second)) 
                return *v;
            if (auto* v = std::get_if<int64_t>(&it->second)) 
                return *v != 0;
        }
        return default_val;
    }
    
    bool has(const std::string& key) const {
        return params.find(key) != params.end();
    }
    
    void set(const std::string& key, ConfigValue value) {
        params[key] = std::move(value);
    }
    
    // ========================================
    // COMMON ACCESSORS (for readability, NOT hardcoding)
    // These just call get() with standard names
    // ========================================
    
    // Dimensions
    uint32_t num_hidden_layers() const { return get<uint32_t>("num_hidden_layers", 0); }
    uint32_t hidden_size() const { return get<uint32_t>("hidden_size", 0); }
    uint32_t intermediate_size() const { return get<uint32_t>("intermediate_size", 0); }
    uint32_t vocab_size() const { return get<uint32_t>("vocab_size", 0); }
    
    // Attention
    uint32_t num_attention_heads() const { return get<uint32_t>("num_attention_heads", 0); }
    uint32_t num_key_value_heads() const { return get<uint32_t>("num_key_value_heads", 0); }
    uint32_t head_dim() const { return get<uint32_t>("head_dim", 128); }
    std::string attention_type() const { return get<std::string>("attention_type", "gqa"); }
    
    // Normalization
    std::string norm_type() const { return get<std::string>("norm_type", "rmsnorm"); }
    float rms_norm_eps() const { return get<float>("rms_norm_eps", 1e-6f); }
    
    // RoPE
    float rope_theta() const { return get<float>("rope_theta", 10000.0f); }
    
    // Identity
    std::string arch() const { return get<std::string>("arch", "unknown"); }
};

// ============================================================================
// TENSOR ENTRY (from manifest)
// ============================================================================

struct TensorEntry {
    std::string name;
    std::string dtype;              // "fp16", "bf16", "hq4k", "hq5k"
    std::string block;              // "text_model", "vision", etc.
    std::vector<uint32_t> shape;
    uint64_t offset;                // Offset within block
    uint64_t size;                  // Size in bytes
};

namespace detail {

// True only when an embedding tensor is used exclusively for row lookups and
// is therefore safe to place in mapped host memory. A tied main embedding
// without a separate lm_head is read in full during every decode step.
bool embedding_is_lookup_only(
    const TensorEntry& entry,
    const std::vector<TensorEntry>& manifest);

} // namespace detail

// ============================================================================
// BLOCK STATE (for tracking loaded blocks)
// ============================================================================

struct BlockState {
    bool loaded = false;
    size_t vram_bytes = 0;
    size_t tensor_count = 0;
    size_t scratch_budget_bytes = 0;
    std::vector<std::string> tensor_names;  // For unloading
    // Single-window backend retained as the zero-risk fallback of the generic
    // mapped-weight pipeline. Double buffering owns its slots in the pipeline.
    void* staging_ptr = nullptr;
    size_t staging_capacity = 0;
    std::vector<std::pair<std::string, void*>> staged_tensor_ptrs;
};

// ============================================================================
// HNF LOADER
// ============================================================================

class HnfLoader {
public:
    HnfLoader();
    ~HnfLoader();
    
    // ========================================
    // FILE OPERATIONS
    // ========================================
    
    // Open HNF file and load metadata (required before load_block)
    bool open(const std::string& path);
    
    // Close file handle
    void close();
    
    // Check if file is open
    bool is_open() const { return file_.is_open(); }
    
    // ========================================
    // BLOCK OPERATIONS (CK interface)
    // ========================================
    
    // Load specific block to GPU
    // CK calls this when switching modality
    bool load_block(BlockID block_id, Engine& engine);
    // Con el modelo partido, cada GPU sube solo los tensores de sus capas.
    void set_tensor_filter(std::function<bool(const TensorEntry&)> filter) {
        tensor_filter_ = std::move(filter);
    }
    bool load_block(const std::string& block_name, Engine& engine);
    
    // Unload specific block from GPU (free VRAM)
    // CK calls this before loading a different modality
    bool unload_block(BlockID block_id, Engine& engine);
    bool unload_block(const std::string& block_name, Engine& engine);

    // Query block state
    bool is_block_loaded(BlockID block_id) const;
    size_t block_vram_usage(BlockID block_id) const;
    size_t block_tensor_count(BlockID block_id) const;
    
    // ========================================
    // LEGACY LOADING (load all at once)
    // ========================================
    
    // Load ALL tensors into engine (not recommended for multimodal)
    bool load(const std::string& path, Engine& engine);
    
    // Load only header/manifest (for inspection)
    bool load_metadata(const std::string& path);
    
    // ========================================
    // ACCESS (after open or load_metadata)
    // ========================================
    
    // Versión mayor del contenedor abierto (10) y tamaño que declara su cabecera.
    uint32_t format_major() const { return format_major_; }
    // VRAM que pide un bloque: sus pesos más su memoria de trabajo en los máximos que
    // declara el HNF (§8 de HNFV10.md). 0 si el bloque no existe o no la declara.
    uint64_t block_vram_bytes(BlockID block) const {
        return block < HNF_BLOCK_COUNT ? block_vram_[block] : 0;
    }
    // ¿Este tensor se queda en RAM en vez de subir a la gráfica? Es la misma regla que
    // aplica la carga (tabla de embeddings de solo consulta, con HELIOS_EMBED_MMAP o
    // HELIOS_EMBED_IN_RAM). Quien reparte VRAM antes de cargar tiene que preguntarlo:
    // contarla como VRAM dejaba fuera la torre visual del E4B por 1,6 GB que no ocupa.
    static bool embed_offload_requested() {
        const char* off = getenv("HELIOS_EMBED_IN_RAM");
        const char* mmap_env = getenv("HELIOS_EMBED_MMAP");
        return (off && *off == '1') || (mmap_env && *mmap_env == '1');
    }
    bool tensor_lives_in_host_ram(const TensorEntry& entry) const {
        return embed_offload_requested() &&
               entry.name.find("token_embedding") != std::string::npos &&
               detail::embedding_is_lookup_only(entry, tensors_);
    }
    uint64_t file_size() const { return file_size_; }
    // Plantilla de chat guardada en el HNFv10, si la trae (bytes UTF-8 exactos).
    const std::string& chat_template() const { return chat_template_; }
    
    // Default config (text/primary model)
    const ModelConfig& config() const { return config_; }

    // Gemma 4 keeps authoritative per-layer geometry in the GM4X extension.
    bool has_gemma4_config() const { return has_gemma4_config_; }
    const Gemma4Config& gemma4_config() const { return gemma4_config_; }
    bool has_gemma4_vision_config() const { return has_gemma4_vision_config_; }
    const Gemma4VisionConfig& gemma4_vision_config() const {
        return gemma4_vision_config_;
    }
    const std::string& last_error() const { return last_error_; }
    
    // Config por bloque específico (para multimodal)
    const ModelConfig& config_for_block(BlockID block_id) const {
        auto it = block_configs_.find(block_id);
        if (it != block_configs_.end()) {
            return it->second;
        }
        return config_;  // Fallback to default
    }
    
    // Check if block has specific config
    bool has_config_for_block(BlockID block_id) const {
        return block_configs_.find(block_id) != block_configs_.end();
    }
    
    const std::vector<TensorEntry>& tensors() const { return tensors_; }
    
    // Block info from file
    const BlockEntry& block(uint32_t id) const { return blocks_[id]; }
    bool has_block(uint32_t id) const { return blocks_[id].size > 0; }
    
    // Block state (loaded/unloaded)
    const BlockState& block_state(BlockID id) const { return block_states_[id]; }
    
    // Get tensors for a specific block
    std::vector<const TensorEntry*> tensors_for_block(BlockID block_id) const;
    std::vector<const TensorEntry*> tensors_for_block(const std::string& block_name) const;

    // Read a tensor payload without loading it into VRAM. Validators use this
    // for scalar contracts such as Gemma 4's learned clipping bounds.
    bool read_tensor_data(const TensorEntry& entry, void* output, size_t bytes) const;
    
    // ========================================
    // TOKENIZER (multi-domain)
    // ========================================
    
    // Get tokenizer for specific domain (default: "text")
    // Loads lazily from cached HTF data
    const HTFTokenizer* tokenizer(const std::string& domain = "text") const;
    HTFTokenizer* tokenizer(const std::string& domain = "text");
    bool has_tokenizer(const std::string& domain = "text") const;
    
    // List available tokenizer domains
    std::vector<std::string> tokenizer_domains() const;
    
    // ========================================
    // DEBUG
    // ========================================
    
    void print_info() const;
    void print_tensors() const;
    void print_loaded_blocks() const;
    
private:
    friend class MappedWeightPipeline;

    // File handle (kept open for block loading)
    std::string file_path_;
    mutable std::ifstream file_;
    
    // Metadata
    BlockEntry blocks_[HNF_BLOCK_COUNT];
    uint32_t format_major_ = 0;
    uint64_t block_vram_[HNF_BLOCK_COUNT] = {};
    uint64_t file_size_ = 0;
    std::string chat_template_;
    ModelConfig config_;                                    // Default/text config
    std::unordered_map<BlockID, ModelConfig> block_configs_; // Per-block configs
    Gemma4Config gemma4_config_;
    bool has_gemma4_config_ = false;
    Gemma4VisionConfig gemma4_vision_config_;
    bool has_gemma4_vision_config_ = false;
    std::string last_error_;
    std::vector<TensorEntry> tensors_;
    
    // Tokenizer (multi-domain support)
    mutable std::unordered_map<std::string, HTFTokenizer> tokenizers_;
    std::vector<uint8_t> htf_data_;  // Cached HTF blob for lazy loading
    bool htf_loaded_ = false;
    std::function<bool(const TensorEntry&)> tensor_filter_;
    
    // Block loading state
    BlockState block_states_[HNF_BLOCK_COUNT];
    
    // Internal methods
    // HNFv10 (src/hnf10.hpp): rellena bloques, tensores y configuración desde sus componentes.
    bool open_hnf10(std::ifstream& f);
    bool read_execution_hints(std::ifstream& f);

    // Configuración binaria (TNIH, GM4X, GM4V)
    bool read_execution_hints_binary(std::ifstream& f);
    bool parse_execution_hints_binary(const uint8_t* data, size_t size);
    bool parse_gemma4_extension(const uint8_t* data, size_t size,
                                const ExecutionHintsBin& hints,
                                const TextModelConfigBin& text_config);
    bool parse_gemma4_vision_extension(const uint8_t* data, size_t size,
                                       const ExecutionHintsBin& hints,
                                       const VisionModelConfigBin& vision_config);
    // Single generic version — no duplicated apply per block
    void apply_text_config_bin(const TextModelConfigBin& cfg, ModelConfig& target);
    void apply_vision_config_bin(const VisionModelConfigBin& cfg);
    
    DTypeID dtype_from_string(const std::string& s) const;
    BlockID block_id_from_name(const std::string& name) const;
    
    // Load tensors to GPU
    bool load_tensors(std::ifstream& f, Engine& engine);
    // Fusiones opcionales de q/k/v y gate/up. HELIOS_FUSE_QKV conserva la
    // compatibilidad historica; HELIOS_FUSE_GATE_UP permite separarlas.
    void fuse_qkv_weights(Engine& engine, BlockState& state);
    void fuse_gemma4_weights(Engine& engine, BlockState& state);
    bool concat_tensors(Engine& engine, BlockState& state,
                        const std::vector<std::string>& partes,
                        const std::string& destino);

    bool load_tensor(std::ifstream& f, const TensorEntry& entry, 
                     TensorRegistry& registry);

    // Single-window backend used by MappedWeightPipeline when double buffering
    // is disabled. It is generic over block and canonical tensor prefix.
    bool stage_mapped_prefix(BlockID block_id, Engine& engine,
                             const std::string& prefix,
                             cudaStream_t stream = nullptr);
    bool unstage_mapped_prefix(BlockID block_id, Engine& engine);
    bool release_mapped_staging(BlockID block_id, Engine& engine);
    
    // Load single block's tensors
    bool load_block_tensors(BlockID block_id, Engine& engine);
    
    // Load tokenizer from BLOCK_TOKENIZER
    void load_tokenizer();
};

// ============================================================================
// CONVENIENCE FUNCTION
// ============================================================================

// Load HNF file into engine
inline bool load_hnf(const std::string& path, Engine& engine) {
    HnfLoader loader;
    return loader.load(path, engine);
}

} // namespace helios
