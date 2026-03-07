// htf_tokenizer.cpp
// ============================================================================
// HTF TOKENIZER v1.0 - Implementation
// ============================================================================

#include "htf_tokenizer.hpp"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <iostream>

namespace helios {

// ============================================================================
// SIMPLE JSON HELPERS (for config parsing)
// ============================================================================

namespace {

const char* skip_ws(const char* p) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

std::string parse_string(const char*& p) {
    if (*p != '"') return "";
    p++;
    std::string result;
    while (*p && *p != '"') {
        if (*p == '\\' && *(p+1)) {
            p++;
            switch (*p) {
                case 'n': result += '\n'; break;
                case 't': result += '\t'; break;
                case 'r': result += '\r'; break;
                case '"': result += '"'; break;
                case '\\': result += '\\'; break;
                default: result += *p;
            }
        } else {
            result += *p;
        }
        p++;
    }
    if (*p == '"') p++;
    return result;
}

int64_t parse_int(const char*& p) {
    char* end;
    int64_t val = strtoll(p, &end, 10);
    p = end;
    return val;
}

bool parse_bool(const char*& p) {
    if (strncmp(p, "true", 4) == 0) { p += 4; return true; }
    if (strncmp(p, "false", 5) == 0) { p += 5; return false; }
    return false;
}

const char* find_key(const char* json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    const char* p = strstr(json, search.c_str());
    if (!p) return nullptr;
    p += search.length();
    p = skip_ws(p);
    if (*p == ':') p++;
    return skip_ws(p);
}

std::string get_string(const char* json, const std::string& key, const std::string& def = "") {
    const char* p = find_key(json, key);
    if (!p || *p != '"') return def;
    return parse_string(p);
}

int64_t get_int(const char* json, const std::string& key, int64_t def = 0) {
    const char* p = find_key(json, key);
    if (!p) return def;
    return parse_int(p);
}

bool get_bool(const char* json, const std::string& key, bool def = false) {
    const char* p = find_key(json, key);
    if (!p) return def;
    return parse_bool(p);
}

// Hash for pair<int32_t, int32_t>
inline uint64_t pair_hash(int32_t a, int32_t b) {
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint32_t>(b);
}

} // anonymous namespace

// ============================================================================
// LOADING
// ============================================================================

bool HTFTokenizer::load(const uint8_t* data, size_t size,
                        const std::string& domain,
                        uint32_t target_vocab_size) {
    if (!data || size < HTF_HEADER_SIZE) {
        std::cerr << "HTFTokenizer: Data too small" << std::endl;
        return false;
    }
    
    requested_domain_ = domain;
    target_vocab_size_ = target_vocab_size;
    
    // Parse header
    if (!parse_header(data, size)) {
        return false;
    }
    
    // Parse domains
    if (!parse_domains(data, size)) {
        return false;
    }
    
    // Build byte encoder/decoder
    build_byte_encoder();
    build_byte_decoder();
    
    return true;
}

bool HTFTokenizer::load_file(const std::string& path,
                              const std::string& domain,
                              uint32_t target_vocab_size) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "HTFTokenizer: Cannot open file: " << path << std::endl;
        return false;
    }
    
    size_t size = f.tellg();
    f.seekg(0);
    
    std::vector<uint8_t> data(size);
    f.read(reinterpret_cast<char*>(data.data()), size);
    
    return load(data.data(), size, domain, target_vocab_size);
}

// ============================================================================
// HEADER PARSING
// ============================================================================

bool HTFTokenizer::parse_header(const uint8_t* data, size_t size) {
    const HTFHeader* header = reinterpret_cast<const HTFHeader*>(data);
    
    // Check for HTF v1.3 (binary config)
    if (header->magic == HTF_MAGIC_V13) {
        is_htf_v13_ = true;
//        std::cerr << "HTFTokenizer: Detected HTF v1.3 (binary config)" << std::endl;
        return true;
    }
    
    // Check for HTF v1.2 (JSON config)
    if (header->magic == HTF_MAGIC_V12) {
        is_htf_v13_ = false;
        return true;
    }
    
    std::cerr << "HTFTokenizer: Invalid magic: 0x" << std::hex << header->magic << std::endl;
    return false;
}

bool HTFTokenizer::parse_domains(const uint8_t* data, size_t size) {
    const HTFHeader* header = reinterpret_cast<const HTFHeader*>(data);
    uint8_t num_domains = header->num_domains;
    
    domains_.clear();
    
    size_t offset = HTF_HEADER_SIZE;
    for (int i = 0; i < num_domains; i++) {
        if (offset + HTF_DOMAIN_ENTRY_SIZE > size) break;
        
        const HTFDomainEntry* entry = reinterpret_cast<const HTFDomainEntry*>(data + offset);
        
        HTFDomain domain;
        domain.type = static_cast<HTFDomainType>(entry->domain_type);
        domain.flags = entry->flags;
        domain.vocab_size = entry->vocab_size;
        domain.data_offset = entry->data_offset;
        domain.data_size = entry->data_size;
        domain.index = i;
        
        domains_.push_back(domain);
        offset += HTF_DOMAIN_ENTRY_SIZE;
    }
    
    // Find requested domain
    HTFDomainType requested_type = HTFDomainType::TEXT;
    if (requested_domain_ == "text") requested_type = HTFDomainType::TEXT;
    else if (requested_domain_ == "code") requested_type = HTFDomainType::CODE;
    else if (requested_domain_ == "cortex") requested_type = HTFDomainType::CORTEX;
    else if (requested_domain_ == "vision") requested_type = HTFDomainType::VISION;
    else if (requested_domain_ == "audio") requested_type = HTFDomainType::AUDIO;
    
    // Find matching domains
    std::vector<HTFDomain*> matching;
    for (auto& d : domains_) {
        if (d.type == requested_type) {
            matching.push_back(&d);
        }
    }
    
    // Cortex fallback: if no CORTEX, use TEXT with largest vocab
    if (matching.empty() && requested_domain_ == "cortex") {
        for (auto& d : domains_) {
            if (d.type == HTFDomainType::TEXT) {
                matching.push_back(&d);
            }
        }
    }
    
    // Code fallback: if no CODE, use TEXT
    if (matching.empty() && requested_domain_ == "code") {
        for (auto& d : domains_) {
            if (d.type == HTFDomainType::TEXT) {
                matching.push_back(&d);
            }
        }
    }
    
    if (matching.empty()) {
        std::cerr << "HTFTokenizer: Domain '" << requested_domain_ << "' not found" << std::endl;
        return false;
    }
    
    // Select domain
    HTFDomain* target = matching[0];
    
    if (matching.size() > 1) {
        // Multiple domains - select by vocab size if specified
        if (target_vocab_size_ > 0) {
            for (auto* d : matching) {
                int diff = std::abs(static_cast<int>(d->vocab_size) - static_cast<int>(target_vocab_size_));
                if (diff < 1000) {
                    target = d;
                    break;
                }
            }
        }
        // For cortex, prefer largest vocab
        if (requested_domain_ == "cortex") {
            for (auto* d : matching) {
                if (d->vocab_size > target->vocab_size) {
                    target = d;
                }
            }
        }
    }
    
    vocab_size_ = target->vocab_size;
    
    // Parse the domain data based on type
    if (target->type == HTFDomainType::TEXT || target->type == HTFDomainType::CORTEX) {
        return parse_text_domain(data, *target);
    } else if (target->type == HTFDomainType::CODE) {
        // CODE domain has simpler layout (no added_tokens)
        return parse_code_domain(data, *target);
    }
    
    // Vision/Audio domains would be handled differently
    std::cerr << "HTFTokenizer: Domain type not yet supported" << std::endl;
    return false;
}

// ============================================================================
// TEXT DOMAIN PARSING
// ============================================================================

bool HTFTokenizer::parse_text_domain(const uint8_t* data, const HTFDomain& domain) {
    // Dispatch based on version
    if (is_htf_v13_) {
        return parse_text_domain_v13(data, domain);
    }
    
    // HTF v1.2: JSON config
    // Text domain data structure:
    // [config_size:u32][config_json][vocab_count:u32][vocab...][merges_count:u32][merges...]
    
    size_t offset = domain.data_offset;
    size_t end = domain.data_offset + domain.data_size;
    
    // Read config size
    if (offset + 4 > end) return false;
    uint32_t config_size = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += 4;
    
    // Parse config JSON
    if (offset + config_size > end) return false;
    if (!parse_config(data, offset, config_size)) {
        std::cerr << "HTFTokenizer: Failed to parse config" << std::endl;
    }
    offset += config_size;
    
    // Align to 4 bytes
    offset = (offset + 3) & ~3;
    
    // Read vocab count
    if (offset + 4 > end) return false;
    uint32_t vocab_count = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += 4;
    
    // Parse vocab
    if (!parse_vocab(data, offset, vocab_count)) {
        std::cerr << "HTFTokenizer: Failed to parse vocab" << std::endl;
        return false;
    }
    
    // Skip to after vocab (need to calculate size)
    // Each vocab entry: [id:u32][len:u16][flags:u8][score:u8][token_bytes][padding]
    for (uint32_t i = 0; i < vocab_count && offset < end; i++) {
        if (offset + 8 > end) break;
        uint16_t len = *reinterpret_cast<const uint16_t*>(data + offset + 4);
        offset += 8 + len;
        offset = (offset + 3) & ~3;  // Align to 4 bytes
    }
    
    // Read merges count
    if (offset + 4 <= end) {
        uint32_t merges_count = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        
        if (merges_count > 0 && (domain.flags & HTF_FLAG_HAS_MERGES)) {
            parse_merges(data, offset, merges_count);
        }
    }
    
    return true;
}

// ============================================================================
// HTF v1.3 BINARY CONFIG PARSING
// ============================================================================

bool HTFTokenizer::parse_text_domain_v13(const uint8_t* data, const HTFDomain& domain) {
    // HTF v1.3 Text domain data structure:
    // [TextConfigBin:32][added_count:u32][added_tokens × added_count][vocab_count:u32][vocab...][merges_count:u32][merges...]
    
    size_t offset = domain.data_offset;
    size_t end = domain.data_offset + domain.data_size;
    
//    std::cerr << "HTFTokenizer: Parsing domain at offset=" << offset 
//              << ", size=" << domain.data_size << std::endl;
    
    // Parse binary config (32 bytes)
    if (offset + HTF_TEXT_CONFIG_SIZE > end) {
        std::cerr << "HTFTokenizer: Config exceeds domain bounds" << std::endl;
        return false;
    }
    if (!parse_config_v13(data, offset)) {
        std::cerr << "HTFTokenizer: Failed to parse v1.3 config" << std::endl;
        return false;
    }
    offset += HTF_TEXT_CONFIG_SIZE;
    
    // Read added_count from file (u32 after config)
    if (offset + 4 > end) {
        std::cerr << "HTFTokenizer: Added count exceeds bounds" << std::endl;
        return false;
    }
    uint32_t added_count = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += 4;
    
//    std::cerr << "HTFTokenizer: added_count=" << added_count << " (from file)" << std::endl;
    
    // Sanity check
    if (added_count > 10000) {
        std::cerr << "HTFTokenizer: Suspicious added_count=" << added_count << std::endl;
        return false;
    }
    
    // Parse added tokens — collect for later registration (after vocab parse)
    // AddedTokenEntry: [id:u32][content_len:u16][flags:u8][reserved:u8][content:var][padding to 4]
    std::vector<std::pair<std::string, int32_t>> pending_added;
    for (uint32_t i = 0; i < added_count && offset < end; i++) {
        if (offset + 8 > end) {
            std::cerr << "HTFTokenizer: Added token " << i << " exceeds bounds at offset " << offset << std::endl;
            break;
        }
        uint32_t token_id = *reinterpret_cast<const uint32_t*>(data + offset);
        uint16_t content_len = *reinterpret_cast<const uint16_t*>(data + offset + 4);
        
        // Extract token content string
        std::string token_content;
        if (content_len > 0 && offset + 8 + content_len <= end) {
            token_content = std::string(reinterpret_cast<const char*>(data + offset + 8), content_len);
        }
        
        if (i < 3 || i == added_count - 1) {
//            std::cerr << "HTFTokenizer: Added[" << i << "] id=" << token_id 
//                      << " len=" << content_len << " at offset=" << offset << std::endl;
//        } else if (i == 3) {
//            std::cerr << "HTFTokenizer: ..." << std::endl;
        }
        
        // Collect for registration after vocab parse
        if (!token_content.empty()) {
            pending_added.push_back({token_content, (int32_t)token_id});
        }
        
        offset += 8 + content_len;
        offset = (offset + 3) & ~3;  // Align to 4
    }
    
//    std::cerr << "HTFTokenizer: After added tokens, offset=" << offset << std::endl;
    
    // Pad to 8 bytes before vocab (as per converter)
    offset = (offset + 7) & ~7;
//    std::cerr << "HTFTokenizer: After padding to 8, offset=" << offset << std::endl;
    
    // Read vocab count
    if (offset + 4 > end) {
        std::cerr << "HTFTokenizer: Vocab count exceeds bounds at offset " << offset << std::endl;
        return false;
    }
    uint32_t vocab_count = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += 4;
    
//    std::cerr << "HTFTokenizer: vocab_count=" << vocab_count << " at offset=" << (offset - 4) << std::endl;
    
    // Sanity check
    if (vocab_count > 1000000) {
        std::cerr << "HTFTokenizer: Suspicious vocab_count=" << vocab_count << ", likely corrupt" << std::endl;
        return false;
    }
    
    // Parse vocab (same format as v1.2)
    if (!parse_vocab(data, offset, vocab_count)) {
        std::cerr << "HTFTokenizer: Failed to parse vocab" << std::endl;
        return false;
    }
    
    // NOW register added tokens (after vocab parse, which calls vocab_.clear())
    for (auto& [content, id] : pending_added) {
        vocab_[content] = id;
        vocab_inv_[id] = content;
    }
    
    // Skip to after vocab
    for (uint32_t i = 0; i < vocab_count && offset < end; i++) {
        if (offset + 8 > end) break;
        uint16_t len = *reinterpret_cast<const uint16_t*>(data + offset + 4);
        offset += 8 + len;
        offset = (offset + 3) & ~3;
    }
    
    // Read merges count
    if (offset + 4 <= end) {
        uint32_t merges_count = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        
//        std::cerr << "HTFTokenizer: merges_count=" << merges_count << std::endl;
        
        if (merges_count > 0 && merges_count < 1000000 && (domain.flags & HTF_FLAG_HAS_MERGES)) {
            parse_merges(data, offset, merges_count);
        }
    }
    
    // Build byte encoder if needed
    if (byte_level_) {
        build_byte_encoder();
    }
    
    return true;
}

// ============================================================================
// CODE DOMAIN PARSING (simpler layout - no added_tokens)
// ============================================================================

bool HTFTokenizer::parse_code_domain(const uint8_t* data, const HTFDomain& domain) {
    // CODE domain layout en HTF v1.3:
    // [TextDomainConfigBin:32][CodeDomainConfigBin:32][added_count:u32][added_tokens...][vocab_count:u32][vocab...]
    // 
    // Note: CODE tiene DOS configs! 64 bytes total antes de added_count
    
    size_t offset = domain.data_offset;
    size_t end = domain.data_offset + domain.data_size;
    
//    std::cerr << "HTFTokenizer: Parsing CODE domain at offset=" << offset 
//              << ", size=" << domain.data_size << std::endl;
//    
//    // Dump first 80 bytes for debug
//    std::cerr << "HTFTokenizer: CODE first 80 bytes:" << std::endl;
//    std::cerr << "  TextConfig  (0-31): ";
//    for (size_t i = 0; i < 32 && (offset + i) < end; i++) {
//        fprintf(stderr, "%02x ", data[offset + i]);
//    }
//    std::cerr << std::endl;
//    std::cerr << "  CodeConfig (32-63): ";
//    for (size_t i = 32; i < 64 && (offset + i) < end; i++) {
//        fprintf(stderr, "%02x ", data[offset + i]);
//    }
//    std::cerr << std::endl;
//    std::cerr << "  After      (64-79): ";
//    for (size_t i = 64; i < 80 && (offset + i) < end; i++) {
//        fprintf(stderr, "%02x ", data[offset + i]);
//    }
//    std::cerr << std::endl;
    
    // 1. Parse TextDomainConfigBin (32 bytes)
    if (offset + 32 > end) {
        std::cerr << "HTFTokenizer: TextConfig exceeds bounds" << std::endl;
        return false;
    }
    if (!parse_config_v13(data, offset)) {
        std::cerr << "HTFTokenizer: Failed to parse TextConfig" << std::endl;
        return false;
    }
    
    // Get num_added_tokens from TextConfig
    uint16_t num_added_tokens = *reinterpret_cast<const uint16_t*>(data + offset + 20);
    offset += 32;
    
    // 2. Skip CodeDomainConfigBin (32 bytes) - contains FIM tokens etc.
    if (offset + 32 > end) {
        std::cerr << "HTFTokenizer: CodeConfig exceeds bounds" << std::endl;
        return false;
    }
    // Could parse FIM tokens here if needed:
    // uint32_t fim_prefix = *reinterpret_cast<const uint32_t*>(data + offset + 4);
    offset += 32;
    
    // 3. Read added_count (u32)
    if (offset + 4 > end) {
        std::cerr << "HTFTokenizer: added_count exceeds bounds" << std::endl;
        return false;
    }
    uint32_t added_count = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += 4;
    
//    std::cerr << "HTFTokenizer: CODE num_added_tokens=" << num_added_tokens 
//              << " (from config), added_count=" << added_count << " (from file)" << std::endl;
    
    // Use the actual added_count from file (should match num_added_tokens)
    if (added_count > 10000) {
        std::cerr << "HTFTokenizer: Suspicious added_count=" << added_count << std::endl;
        return false;
    }
    
    // 4. Collect added tokens (register after parse_vocab which clears vocab_)
    std::vector<std::pair<std::string, int32_t>> pending_added;
    for (uint32_t i = 0; i < added_count && offset < end; i++) {
        if (offset + 8 > end) break;
        uint32_t token_id = *reinterpret_cast<const uint32_t*>(data + offset);
        uint16_t content_len = *reinterpret_cast<const uint16_t*>(data + offset + 4);
        
        if (content_len > 0 && offset + 8 + content_len <= end) {
            std::string token_content(reinterpret_cast<const char*>(data + offset + 8), content_len);
            pending_added.push_back({token_content, (int32_t)token_id});
        }
        
        offset += 8 + content_len;
        offset = (offset + 3) & ~3;
    }
    
    // 5. Pad to 8 before vocab
    offset = (offset + 7) & ~7;
    
    // 6. Read vocab_count
    if (offset + 4 > end) {
        std::cerr << "HTFTokenizer: vocab_count exceeds bounds at offset " << offset << std::endl;
        return false;
    }
    uint32_t vocab_count = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += 4;
    
//    std::cerr << "HTFTokenizer: CODE vocab_count=" << vocab_count << " at offset=" << (offset - 4) << std::endl;
    
    if (vocab_count == 0 || vocab_count > 500000) {
        std::cerr << "HTFTokenizer: Invalid vocab_count=" << vocab_count << std::endl;
        return false;
    }
    
    // 7. Parse vocab
    if (!parse_vocab(data, offset, vocab_count)) {
        std::cerr << "HTFTokenizer: Failed to parse vocab" << std::endl;
        return false;
    }
    
    // 7b. NOW register added tokens
    for (auto& [content, id] : pending_added) {
        vocab_[content] = id;
        vocab_inv_[id] = content;
    }
    
    // 8. Skip to merges
    for (uint32_t i = 0; i < vocab_count && offset < end; i++) {
        if (offset + 8 > end) break;
        uint16_t len = *reinterpret_cast<const uint16_t*>(data + offset + 4);
        offset += 8 + len;
        offset = (offset + 3) & ~3;
    }
    
    // 9. Merges
    if (offset + 4 <= end) {
        uint32_t merges_count = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
//        std::cerr << "HTFTokenizer: CODE merges_count=" << merges_count << std::endl;
        if (merges_count > 0 && merges_count < 500000 && (domain.flags & HTF_FLAG_HAS_MERGES)) {
            parse_merges(data, offset, merges_count);
        }
    }
    
    // Build byte encoder
    if (byte_level_) {
        build_byte_encoder();
    }
    
    return true;
}

bool HTFTokenizer::parse_config_v13(const uint8_t* data, size_t offset) {
    const HTFTextConfigBin* cfg = reinterpret_cast<const HTFTextConfigBin*>(data + offset);
    
    // Special tokens (-1 means not defined)
    if (cfg->bos_token_id >= 0) bos_id_ = cfg->bos_token_id;
    if (cfg->eos_token_id >= 0) eos_id_ = cfg->eos_token_id;
    if (cfg->pad_token_id >= 0) pad_id_ = cfg->pad_token_id;
    if (cfg->unk_token_id >= 0) unk_id_ = cfg->unk_token_id;
    
    vocab_size_ = cfg->vocab_size;
    
    // Encoding type
    switch (cfg->encoding_type) {
        case 0: encoding_type_ = EncodingType::BPE; break;
        case 1: encoding_type_ = EncodingType::SENTENCEPIECE; break;
        case 2: encoding_type_ = EncodingType::WORDPIECE; break;
        case 3: encoding_type_ = EncodingType::UNIGRAM; break;
        default: encoding_type_ = EncodingType::BPE; break;
    }
    
    // Flags
    byte_level_ = (cfg->flags & 0x01) != 0;
    
//    std::cerr << "HTFTokenizer: v1.3 config loaded - vocab=" << vocab_size_ 
//              << ", bos=" << (bos_id_.has_value() ? std::to_string(*bos_id_) : "none")
//              << ", eos=" << (eos_id_.has_value() ? std::to_string(*eos_id_) : "none")
//              << std::endl;
    
    return true;
}

bool HTFTokenizer::parse_vocab(const uint8_t* data, size_t offset, uint32_t count) {
    vocab_.clear();
    vocab_inv_.clear();
    vocab_.reserve(count);
    vocab_inv_.reserve(count);
    
    for (uint32_t i = 0; i < count; i++) {
        // Format: [id:u32][len:u16][flags:u8][score:u8][token_bytes]
        uint32_t id = *reinterpret_cast<const uint32_t*>(data + offset);
        uint16_t len = *reinterpret_cast<const uint16_t*>(data + offset + 4);
        // uint8_t flags = *(data + offset + 6);
        // uint8_t score = *(data + offset + 7);
        
        std::string token(reinterpret_cast<const char*>(data + offset + 8), len);
        
        vocab_[token] = id;
        vocab_inv_[id] = token;
        
        offset += 8 + len;
        offset = (offset + 3) & ~3;  // Align to 4 bytes
    }
    
    return true;
}

bool HTFTokenizer::parse_merges(const uint8_t* data, size_t offset, uint32_t count) {
    merges_.clear();
    merges_.reserve(count);
    merge_priority_.clear();
    
    for (uint32_t i = 0; i < count; i++) {
        // Format: [id1:u32][id2:u32]
        int32_t id1 = *reinterpret_cast<const int32_t*>(data + offset);
        int32_t id2 = *reinterpret_cast<const int32_t*>(data + offset + 4);
        
        merges_.emplace_back(id1, id2);
        merge_priority_[pair_hash(id1, id2)] = i;
        
        offset += 8;
    }
    
    return true;
}

bool HTFTokenizer::parse_config(const uint8_t* data, size_t offset, size_t size) {
    std::string json(reinterpret_cast<const char*>(data + offset), size);
    const char* j = json.c_str();
    
    // Encoding type
    std::string enc_type = get_string(j, "encoding_type", "bpe");
    if (enc_type == "bpe") encoding_type_ = EncodingType::BPE;
    else if (enc_type == "sentencepiece") encoding_type_ = EncodingType::SENTENCEPIECE;
    else if (enc_type == "wordpiece") encoding_type_ = EncodingType::WORDPIECE;
    else if (enc_type == "unigram") encoding_type_ = EncodingType::UNIGRAM;
    else encoding_type_ = EncodingType::UNKNOWN;
    
    // Byte-level
    byte_level_ = get_bool(j, "byte_level", false);
    
    // Special tokens
    int64_t bos = get_int(j, "bos_token_id", -1);
    int64_t eos = get_int(j, "eos_token_id", -1);
    int64_t pad = get_int(j, "pad_token_id", -1);
    int64_t unk = get_int(j, "unk_token_id", -1);
    
    if (bos >= 0) bos_id_ = static_cast<int32_t>(bos);
    if (eos >= 0) eos_id_ = static_cast<int32_t>(eos);
    if (pad >= 0) pad_id_ = static_cast<int32_t>(pad);
    if (unk >= 0) unk_id_ = static_cast<int32_t>(unk);
    
    return true;
}

// ============================================================================
// BYTE-LEVEL ENCODING (GPT-2 style)
// ============================================================================

void HTFTokenizer::build_byte_encoder() {
    byte_to_unicode_.clear();
    unicode_to_byte_.clear();
    
    // GPT-2 byte-level BPE maps bytes to Unicode characters
    // Printable ASCII (33-126) and extended Latin (161-172, 174-255) map directly
    // Other bytes (0-32, 127-160, 173) map to Unicode 256+
    
    std::vector<bool> direct(256, false);
    for (int i = 33; i < 127; i++) direct[i] = true;     // '!' to '~'
    for (int i = 161; i < 173; i++) direct[i] = true;    // Extended Latin
    for (int i = 174; i < 256; i++) direct[i] = true;    // More extended
    
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (direct[b]) {
            // Direct mapping: byte -> same codepoint
            byte_to_unicode_[static_cast<uint8_t>(b)] = static_cast<uint32_t>(b);
        } else {
            // Map to Unicode 256+ (Ġ, Ċ, etc.)
            byte_to_unicode_[static_cast<uint8_t>(b)] = 256 + n;
            n++;
        }
    }
    
    // Build reverse mapping
    for (auto& [byte, code] : byte_to_unicode_) {
        unicode_to_byte_[code] = byte;
    }
}

void HTFTokenizer::build_byte_decoder() {
    // Already built in build_byte_encoder via unicode_to_byte_
}

std::string HTFTokenizer::bytes_to_unicode(const std::string& text) const {
    std::string result;
    for (unsigned char c : text) {
        auto it = byte_to_unicode_.find(c);
        if (it != byte_to_unicode_.end()) {
            uint32_t code = it->second;
            // Encode as UTF-8
            if (code < 0x80) {
                result += static_cast<char>(code);
            } else if (code < 0x800) {
                result += static_cast<char>(0xC0 | (code >> 6));
                result += static_cast<char>(0x80 | (code & 0x3F));
            } else if (code < 0x10000) {
                result += static_cast<char>(0xE0 | (code >> 12));
                result += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                result += static_cast<char>(0x80 | (code & 0x3F));
            }
        } else {
            // Fallback: use byte directly
            result += static_cast<char>(c);
        }
    }
    return result;
}

std::string HTFTokenizer::unicode_to_bytes(const std::string& text) const {
    std::vector<uint8_t> bytes;
    
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = text[i];
        uint32_t code;
        
        // Decode UTF-8 to codepoint
        if ((c & 0x80) == 0) {
            code = c;
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            code = (c & 0x1F) << 6;
            if (i + 1 < text.size()) code |= (text[i+1] & 0x3F);
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            code = (c & 0x0F) << 12;
            if (i + 1 < text.size()) code |= (text[i+1] & 0x3F) << 6;
            if (i + 2 < text.size()) code |= (text[i+2] & 0x3F);
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            code = (c & 0x07) << 18;
            if (i + 1 < text.size()) code |= (text[i+1] & 0x3F) << 12;
            if (i + 2 < text.size()) code |= (text[i+2] & 0x3F) << 6;
            if (i + 3 < text.size()) code |= (text[i+3] & 0x3F);
            i += 4;
        } else {
            code = c;
            i += 1;
        }
        
        // Find in decoder map
        auto it = unicode_to_byte_.find(code);
        if (it != unicode_to_byte_.end()) {
            bytes.push_back(it->second);
        } else {
            // Direct byte (should not happen in well-formed input)
            bytes.push_back(static_cast<uint8_t>(code & 0xFF));
        }
    }
    
    return std::string(bytes.begin(), bytes.end());
}

// ============================================================================
// ENCODE
// ============================================================================

std::vector<int32_t> HTFTokenizer::encode(const std::string& text,
                                           bool add_bos,
                                           bool add_eos) const {
    std::vector<int32_t> ids;
    
    if (add_bos && bos_id_) {
        ids.push_back(*bos_id_);
    }
    
    std::vector<int32_t> text_ids;
    
    switch (encoding_type_) {
        case EncodingType::BPE:
            text_ids = encode_bpe(text);
            break;
        case EncodingType::SENTENCEPIECE:
        case EncodingType::UNIGRAM:
            text_ids = encode_sentencepiece(text);
            break;
        case EncodingType::WORDPIECE:
            text_ids = encode_wordpiece(text);
            break;
        default:
            text_ids = encode_bpe(text);
    }
    
    ids.insert(ids.end(), text_ids.begin(), text_ids.end());
    
    if (add_eos && eos_id_) {
        ids.push_back(*eos_id_);
    }
    
    return ids;
}

std::vector<int32_t> HTFTokenizer::encode_bpe(const std::string& text) const {
    std::vector<int32_t> ids;
    
    // Convert to byte-level if needed
    std::string processed = byte_level_ ? bytes_to_unicode(text) : text;
    
    // Tokenize each character initially
    std::vector<std::string> tokens;
    for (size_t i = 0; i < processed.size(); ) {
        // Handle UTF-8
        unsigned char c = processed[i];
        size_t char_len = 1;
        if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;
        
        tokens.push_back(processed.substr(i, char_len));
        i += char_len;
    }
    
    // Convert to IDs
    for (const auto& t : tokens) {
        auto it = vocab_.find(t);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
        } else if (unk_id_) {
            ids.push_back(*unk_id_);
        }
    }
    
    if (merges_.empty()) {
        return ids;
    }
    
    // Apply BPE merges
    while (ids.size() > 1) {
        // Find best merge
        int32_t best_priority = static_cast<int32_t>(merges_.size());
        size_t best_idx = 0;
        bool found = false;
        
        for (size_t i = 0; i < ids.size() - 1; i++) {
            uint64_t h = pair_hash(ids[i], ids[i+1]);
            auto it = merge_priority_.find(h);
            if (it != merge_priority_.end() && it->second < best_priority) {
                best_priority = it->second;
                best_idx = i;
                found = true;
            }
        }
        
        if (!found) break;
        
        // Merge
        std::string merged = vocab_inv_.at(ids[best_idx]) + vocab_inv_.at(ids[best_idx + 1]);
        auto it = vocab_.find(merged);
        if (it != vocab_.end()) {
            ids[best_idx] = it->second;
            ids.erase(ids.begin() + best_idx + 1);
        } else {
            break;
        }
    }
    
    return ids;
}

std::vector<int32_t> HTFTokenizer::encode_sentencepiece(const std::string& text) const {
    std::vector<int32_t> ids;
    
    // Greedy longest-match tokenization
    size_t i = 0;
    while (i < text.size()) {
        std::string best_match;
        size_t best_len = 0;
        
        // Try all lengths from current position
        for (size_t end = std::min(i + 50, text.size()); end > i; end--) {
            std::string substr = text.substr(i, end - i);
            
            // Handle SentencePiece space prefix
            if (i == 0 || (i > 0 && text[i-1] == ' ')) {
                std::string sp_substr = "\xE2\x96\x81" + substr;  // ▁ (U+2581)
                // Remove leading space if present
                if (!substr.empty() && substr[0] == ' ') {
                    sp_substr = "\xE2\x96\x81" + substr.substr(1);
                }
                if (vocab_.find(sp_substr) != vocab_.end()) {
                    best_match = sp_substr;
                    best_len = end - i;
                    break;
                }
            }
            
            if (vocab_.find(substr) != vocab_.end()) {
                best_match = substr;
                best_len = end - i;
                break;
            }
        }
        
        if (!best_match.empty()) {
            ids.push_back(vocab_.at(best_match));
            i += best_len;
        } else {
            // Single character or unknown
            std::string ch(1, text[i]);
            auto it = vocab_.find(ch);
            if (it != vocab_.end()) {
                ids.push_back(it->second);
            } else if (unk_id_) {
                ids.push_back(*unk_id_);
            }
            i++;
        }
    }
    
    return ids;
}

std::vector<int32_t> HTFTokenizer::encode_wordpiece(const std::string& text) const {
    std::vector<int32_t> ids;
    
    // Split by spaces
    std::vector<std::string> words;
    std::string current;
    for (char c : text) {
        if (c == ' ') {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        words.push_back(current);
    }
    
    for (size_t w = 0; w < words.size(); w++) {
        const std::string& word = words[w];
        
        // Add space token between words
        if (w > 0) {
            auto it = vocab_.find(" ");
            if (it != vocab_.end()) {
                ids.push_back(it->second);
            }
        }
        
        // Check if whole word exists
        auto it = vocab_.find(word);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
            continue;
        }
        
        // WordPiece tokenization
        size_t i = 0;
        while (i < word.size()) {
            std::string best_match;
            size_t best_len = 0;
            
            for (size_t end = word.size(); end > i; end--) {
                std::string substr = word.substr(i, end - i);
                if (i > 0) {
                    substr = "##" + substr;  // Continuation prefix
                }
                
                if (vocab_.find(substr) != vocab_.end()) {
                    best_match = substr;
                    best_len = end - i;
                    break;
                }
            }
            
            if (!best_match.empty()) {
                ids.push_back(vocab_.at(best_match));
                i += best_len;
            } else {
                if (unk_id_) {
                    ids.push_back(*unk_id_);
                }
                i++;
            }
        }
    }
    
    return ids;
}

// ============================================================================
// DECODE
// ============================================================================

std::string HTFTokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string text;
    
    for (int32_t id : ids) {
        auto it = vocab_inv_.find(id);
        if (it != vocab_inv_.end()) {
            text += it->second;
        }
    }
    
    // Byte-level decode (GPT-2 style: unicode codepoints → raw bytes)
    if (byte_level_) {
        text = unicode_to_bytes(text);
    }
    
    // SentencePiece: replace ▁ with space
    size_t pos = 0;
    const std::string sp_space = "\xE2\x96\x81";  // ▁ (U+2581)
    while ((pos = text.find(sp_space, pos)) != std::string::npos) {
        text.replace(pos, 3, " ");
        pos++;
    }
    
    return text;
}

// ============================================================================
// UTILITIES
// ============================================================================

std::string HTFTokenizer::id_to_token(int32_t id) const {
    auto it = vocab_inv_.find(id);
    if (it != vocab_inv_.end()) {
        return it->second;
    }
    return "";
}

std::optional<int32_t> HTFTokenizer::token_to_id(const std::string& token) const {
    auto it = vocab_.find(token);
    if (it != vocab_.end()) {
        return it->second;
    }
    return std::nullopt;
}

} // namespace helios
