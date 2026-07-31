// htf_tokenizer.hpp
// ============================================================================
// HTF TOKENIZER v1.0 - Lee tokenizer HTF v1.2 embebido en HNF
// ============================================================================
// Implementa encode/decode sin dependencias externas.
// Soporta: BPE, SentencePiece, WordPiece, byte-level
//
// Formato HTF:
//   Header (32 bytes) + Domain entries (32 bytes each) + Domain data
//
// Polimórfico: no sabe qué modelo es, lee config del archivo.
// ============================================================================

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <optional>

namespace helios {

// ============================================================================
// HTF CONSTANTS - Soporta v1.2 (JSON) y v1.3 (Binario)
// ============================================================================

constexpr uint32_t HTF_MAGIC_V12 = 0x32465448;  // "HTF2" little-endian
constexpr uint32_t HTF_MAGIC_V13 = 0x33465448;  // "HTF3" little-endian
constexpr uint32_t HTF_MAGIC = HTF_MAGIC_V12;   // Default for compatibility
constexpr uint32_t HTF_HEADER_SIZE = 32;
constexpr uint32_t HTF_DOMAIN_ENTRY_SIZE = 32;

// HTF v1.3 Binary Config Sizes
constexpr uint32_t HTF_TEXT_CONFIG_SIZE = 32;
constexpr uint32_t HTF_VISION_CONFIG_SIZE = 64;
constexpr uint32_t HTF_AUDIO_CONFIG_SIZE = 64;
constexpr uint32_t HTF_CODE_CONFIG_SIZE = 32;

// Domain types
enum class HTFDomainType : uint8_t {
    TEXT = 0x00,
    VISION = 0x01,
    AUDIO = 0x02,
    CODE = 0x03,
    CORTEX = 0x04,
};

// Domain flags
enum HTFDomainFlags : uint8_t {
    HTF_FLAG_HAS_VOCAB = 1 << 0,
    HTF_FLAG_HAS_CODEBOOK = 1 << 1,
    HTF_FLAG_HAS_MERGES = 1 << 2,
    HTF_FLAG_IS_PRIMARY = 1 << 3,
};

// Encoding types
enum class EncodingType {
    BPE,
    SENTENCEPIECE,
    WORDPIECE,
    UNIGRAM,
    UNKNOWN,
};

// ============================================================================
// HTF STRUCTURES (packed, match file format)
// ============================================================================

#pragma pack(push, 1)

struct HTFHeader {
    uint32_t magic;           // "HTF2"
    uint16_t version;         // 0x0102 = v1.2
    uint16_t flags;
    uint8_t  num_domains;
    uint8_t  reserved[7];
    uint64_t total_size;
    uint64_t checksum;
};
static_assert(sizeof(HTFHeader) == 32, "HTFHeader must be 32 bytes");

struct HTFDomainEntry {
    uint8_t  domain_type;
    uint8_t  flags;
    uint8_t  reserved[2];
    uint32_t vocab_size;
    uint64_t data_offset;
    uint64_t data_size;
    uint64_t name_hash;
};
static_assert(sizeof(HTFDomainEntry) == 32, "HTFDomainEntry must be 32 bytes");

// ============================================================================
// HTF v1.3 BINARY CONFIG STRUCTURES
// ============================================================================

struct HTFTextConfigBin {
    int32_t  bos_token_id;
    int32_t  eos_token_id;
    int32_t  pad_token_id;
    int32_t  unk_token_id;
    uint32_t vocab_size;
    uint16_t num_added_tokens;
    uint8_t  encoding_type;   // 0=BPE, 1=SP, 2=WP, 3=Unigram
    uint8_t  flags;           // bit0: byte_level, bit1: add_prefix_space
    uint8_t  pre_tokenizer_type; // 1=split-space, merged with previous
    uint8_t  decoder_type;       // 1=replace U+2581 with ASCII space
    uint8_t  behaviour_flags;    // bit0: add BOS by default
    uint8_t  normalizer_type;    // 1=replace ASCII space with U+2581
    uint8_t  model_flags;        // bit0: byte fallback
    uint8_t  reserved[3];
};
static_assert(sizeof(HTFTextConfigBin) == 32, "HTFTextConfigBin must be 32 bytes");

#pragma pack(pop)

// ============================================================================
// DOMAIN INFO
// ============================================================================

struct HTFDomain {
    HTFDomainType type;
    uint8_t flags;
    uint32_t vocab_size;
    uint64_t data_offset;
    uint64_t data_size;
    int index;
};

// ============================================================================
// HTF TOKENIZER
// ============================================================================

class HTFTokenizer {
public:
    HTFTokenizer() = default;
    ~HTFTokenizer() = default;
    
    // ========================================
    // LOADING
    // ========================================
    
    // Load from raw bytes (HTF file content)
    bool load(const uint8_t* data, size_t size, 
              const std::string& domain = "text",
              uint32_t target_vocab_size = 0);
    
    // Load from file path
    bool load_file(const std::string& path,
                   const std::string& domain = "text",
                   uint32_t target_vocab_size = 0);
    
    // ========================================
    // ENCODE / DECODE
    // ========================================
    
    // Encode text to token IDs
    std::vector<int32_t> encode(const std::string& text, 
                                 bool add_bos = false,
                                 bool add_eos = false) const;
    
    // Decode token IDs to text
    std::string decode(const std::vector<int32_t>& ids) const;
    
    // ========================================
    // SPECIAL TOKENS
    // ========================================
    
    std::optional<int32_t> bos_token_id() const { return bos_id_; }
    std::optional<int32_t> eos_token_id() const { return eos_id_; }
    std::optional<int32_t> pad_token_id() const { return pad_id_; }
    std::optional<int32_t> unk_token_id() const { return unk_id_; }
    
    // ========================================
    // INFO
    // ========================================
    
    uint32_t vocab_size() const { return vocab_size_; }
    size_t merge_count() const { return merges_.size(); }
    size_t added_token_count() const { return added_tokens_.size(); }
    EncodingType encoding_type() const { return encoding_type_; }
    bool byte_level() const { return byte_level_; }
    bool byte_fallback() const { return byte_fallback_; }
    bool add_prefix_space() const { return add_prefix_space_; }
    bool add_bos_token_by_default() const { return add_bos_token_; }
    bool space_to_metaspace_normalizer() const { return normalizer_type_ == 1; }
    bool metaspace_decoder() const { return decoder_type_ == 1; }
    bool is_htf_v13() const { return is_htf_v13_; }
    const std::vector<HTFDomain>& available_domains() const { return domains_; }
    
    // Get token string by ID
    std::string id_to_token(int32_t id) const;
    
    // Get ID by token string
    std::optional<int32_t> token_to_id(const std::string& token) const;

private:
    // ========================================
    // PARSING
    // ========================================
    
    bool parse_header(const uint8_t* data, size_t size);
    bool parse_domains(const uint8_t* data, size_t size);
    bool parse_text_domain(const uint8_t* data, const HTFDomain& domain);
    bool parse_code_domain(const uint8_t* data, const HTFDomain& domain);
    bool parse_vocab(const uint8_t* data, size_t offset, uint32_t count);
    bool parse_merges(const uint8_t* data, size_t offset, uint32_t count);
    bool parse_config(const uint8_t* data, size_t offset, size_t size);
    
    // HTF v1.3 binary config parsing
    bool parse_text_domain_v13(const uint8_t* data, const HTFDomain& domain);
    bool parse_config_v13(const uint8_t* data, size_t offset);
    
    // Version detection
    bool is_htf_v13_ = false;
    
    // ========================================
    // ENCODING IMPLEMENTATIONS
    // ========================================
    
    std::vector<int32_t> encode_bpe(const std::string& text) const;
    std::vector<int32_t> encode_bpe_segment(const std::string& text) const;
    std::vector<int32_t> encode_sentencepiece(const std::string& text) const;
    std::vector<int32_t> encode_wordpiece(const std::string& text) const;
    
    // ========================================
    // BYTE-LEVEL HELPERS
    // ========================================
    
    void build_byte_encoder();
    void build_byte_decoder();
    std::string bytes_to_unicode(const std::string& text) const;
    std::string unicode_to_bytes(const std::string& text) const;
    
    // ========================================
    // DATA
    // ========================================
    
    // Vocab: token <-> id
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<int32_t, std::string> vocab_inv_;
    
    // BPE merges: (id1, id2) pairs in priority order
    std::vector<std::pair<int32_t, int32_t>> merges_;
    
    // Merge priority lookup: (id1, id2) -> priority
    std::unordered_map<uint64_t, int32_t> merge_priority_;
    
    // Byte-level mappings (GPT-2 style)
    std::unordered_map<uint8_t, uint32_t> byte_to_unicode_;  // byte -> codepoint
    std::unordered_map<uint32_t, uint8_t> unicode_to_byte_;  // codepoint -> byte
    
    // Config
    uint32_t vocab_size_ = 0;
    EncodingType encoding_type_ = EncodingType::BPE;
    bool byte_level_ = false;
    bool byte_fallback_ = false;
    bool add_prefix_space_ = false;
    bool add_bos_token_ = false;
    uint8_t pre_tokenizer_type_ = 0;
    uint8_t decoder_type_ = 0;
    uint8_t normalizer_type_ = 0;

    struct AddedToken {
        std::string content;
        int32_t id = -1;
        uint8_t flags = 0;
    };
    std::vector<AddedToken> added_tokens_;
    
    // Special tokens
    std::optional<int32_t> bos_id_;
    std::optional<int32_t> eos_id_;
    std::optional<int32_t> pad_id_;
    std::optional<int32_t> unk_id_;
    
    // Available domains
    std::vector<HTFDomain> domains_;
    
    // Requested domain
    std::string requested_domain_;
    uint32_t target_vocab_size_ = 0;
};

} // namespace helios
