// hnf_loader.hpp
// ============================================================================
// HNF v9 LOADER - Carga modelos HELIOS Neural Format
// ============================================================================
// Lee HNF v9, carga tensores a GPU, registra en TensorRegistry.
// Los pesos HQ4K/HQ5K se cargan SIN dequantizar (fused en kernel).
//

#pragma once

#include "engine.hpp"
#include "htf_tokenizer.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <fstream>

namespace helios {

// ============================================================================
// HNF CONSTANTS
// ============================================================================

constexpr uint8_t HNF_MAGIC[8] = {'H', 'N', 'F', 'v', '9', 0, 0, 0};
constexpr uint32_t HNF_HEADER_SIZE = 64;
constexpr uint32_t HNF_BLOCK_TABLE_SIZE = 512;
constexpr uint32_t HNF_BLOCK_ENTRY_SIZE = 32;
constexpr uint32_t HNF_BLOCK_COUNT = 16;
constexpr uint32_t HNF_ALIGNMENT = 32;

// ============================================================================
// BLOCK IDs - HNFv9.1
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
    BLOCK_EXEC_HINTS      = 0xA,   // JSON (obligatorio)
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
// FLAGS - HNFv9.1
// ============================================================================

enum HnfFlags : uint32_t {
    HNF_HAS_VISION         = (1 << 0),
    HNF_HAS_AUDIO          = (1 << 1),
    HNF_HAS_VIDEO          = (1 << 2),
    HNF_HAS_SPATIAL        = (1 << 3),
    HNF_HAS_PERSONALITY    = (1 << 4),
    HNF_HAS_MEMORY         = (1 << 5),
    HNF_HAS_CORTEX         = (1 << 6),
    HNF_HAS_CODE_EXEC      = (1 << 7),
    HNF_HAS_TOKENIZER      = (1 << 8),   // ← NUEVO
    HNF_HAS_EXEC_HINTS_BIN = (1 << 9),   // ← NUEVO
    HNF_HAS_TOOLS          = (1 << 10),
    HNF_HAS_EXPERT_ROUTER  = (1 << 11),
    HNF_IS_MOE             = (1 << 12),
    HNF_IS_MULTIMODAL      = (1 << 13),
};

// ============================================================================
// BINARY STRUCTURES (packed, little-endian)
// ============================================================================

#pragma pack(push, 1)

struct HnfHeader {
    uint8_t  magic[8];              // "HNFv9\x00\x00\x00"
    uint16_t version_major;         // 9
    uint16_t version_minor;         // 0
    uint32_t flags;                 // Capability flags
    uint32_t block_count;           // 16
    uint32_t header_size;           // 64
    uint64_t block_table_offset;    // 64
    uint64_t manifest_offset;       // Offset del manifest JSON
    uint64_t manifest_size;         // Tamaño del manifest
    uint64_t file_size;             // Tamaño total
    uint32_t checksum;              // CRC32
    uint32_t reserved;              // 0
};
static_assert(sizeof(HnfHeader) == 64, "HnfHeader must be 64 bytes");

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
// Prioridad: Si [0xB] existe, usar binario. Fallback a JSON [0xA].

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
    uint32_t reserved;
};
static_assert(sizeof(Gemma4LayerConfigBin) == 40,
              "Gemma4LayerConfigBin must be 40 bytes");

#pragma pack(pop)

constexpr uint32_t GEMMA4_EXT_FLAG_PLE              = (1u << 0);
constexpr uint32_t GEMMA4_EXT_FLAG_LAYER_SCALAR     = (1u << 1);
constexpr uint32_t GEMMA4_EXT_FLAG_LOGIT_SOFTCAP    = (1u << 2);
constexpr uint32_t GEMMA4_EXT_FLAG_SHARED_KV        = (1u << 3);
constexpr uint32_t GEMMA4_EXT_FLAG_DOUBLE_WIDE_MLP  = (1u << 4);
constexpr uint32_t GEMMA4_EXT_FLAG_FOUR_NORM_BLOCK  = (1u << 5);

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

    bool is_global_attention() const { return attention_kind == 1; }
};

struct Gemma4Config {
    uint16_t version = 0;
    uint32_t flags = 0;
    uint32_t global_head_dim = 0;
    uint32_t ple_hidden_size = 0;
    uint32_t num_kv_shared_layers = 0;
    std::vector<Gemma4LayerConfig> layers;

    bool has_flag(uint32_t flag) const { return (flags & flag) != 0; }
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
    std::vector<std::string> tensor_names;  // For unloading
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
    bool load_block(const std::string& block_name, Engine& engine);
    
    // Unload specific block from GPU (free VRAM)
    // CK calls this before loading a different modality
    bool unload_block(BlockID block_id, Engine& engine);
    bool unload_block(const std::string& block_name, Engine& engine);
    
    // Query block state
    bool is_block_loaded(BlockID block_id) const;
    size_t block_vram_usage(BlockID block_id) const;
    size_t block_tensor_count(BlockID block_id) const;
    
    // Get total VRAM used by loaded blocks
    size_t total_vram_usage() const;
    
    // List currently loaded blocks
    std::vector<BlockID> loaded_blocks() const;
    
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
    
    const HnfHeader& header() const { return header_; }
    
    // Default config (text/primary model)
    const ModelConfig& config() const { return config_; }

    // Gemma 4 keeps authoritative per-layer geometry in the GM4X extension.
    bool has_gemma4_config() const { return has_gemma4_config_; }
    const Gemma4Config& gemma4_config() const { return gemma4_config_; }
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
    // File handle (kept open for block loading)
    std::string file_path_;
    mutable std::ifstream file_;
    
    // Metadata
    HnfHeader header_;
    BlockEntry blocks_[HNF_BLOCK_COUNT];
    ModelConfig config_;                                    // Default/text config
    std::unordered_map<BlockID, ModelConfig> block_configs_; // Per-block configs
    Gemma4Config gemma4_config_;
    bool has_gemma4_config_ = false;
    std::string last_error_;
    std::vector<TensorEntry> tensors_;
    std::string manifest_json_;
    
    // Tokenizer (multi-domain support)
    mutable std::unordered_map<std::string, HTFTokenizer> tokenizers_;
    std::vector<uint8_t> htf_data_;  // Cached HTF blob for lazy loading
    bool htf_loaded_ = false;
    
    // Block loading state
    BlockState block_states_[HNF_BLOCK_COUNT];
    
    // Internal methods
    bool read_header(std::ifstream& f);
    bool read_block_table(std::ifstream& f);
    bool read_manifest(std::ifstream& f);
    bool read_execution_hints(std::ifstream& f);
    bool parse_manifest();
    bool parse_execution_hints(const std::string& json);
    
    // Binary hints parsing (priority over JSON)
    bool read_execution_hints_binary(std::ifstream& f);
    bool parse_execution_hints_binary(const uint8_t* data, size_t size);
    bool parse_gemma4_extension(const uint8_t* data, size_t size,
                                const ExecutionHintsBin& hints,
                                const TextModelConfigBin& text_config);
    // Single generic version — no duplicated apply per block
    void apply_text_config_bin(const TextModelConfigBin& cfg, ModelConfig& target);
    void apply_vision_config_bin(const VisionModelConfigBin& cfg);
    
    DTypeID dtype_from_string(const std::string& s) const;
    BlockID block_id_from_name(const std::string& name) const;
    
    // Load tensors to GPU
    bool load_tensors(std::ifstream& f, Engine& engine);
    // Fusion opcional de q/k/v en un solo tensor (HELIOS_FUSE_QKV=1)
    void fuse_qkv_weights(Engine& engine, BlockState& state);

    bool load_tensor(std::ifstream& f, const TensorEntry& entry, 
                     TensorRegistry& registry);
    
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
