// hnf_loader.cpp
// ============================================================================
// HNF v9.1 LOADER - Implementation (REFACTORED)
// ============================================================================
// CHANGES:
//   - Single apply_text_config_bin(cfg, target) — no duplication
//   - JSON parser processes ALL block configs (text, vision, cortex, code)
//   - Uses block_id_from_name() consistently in load_tensor()
//   - Removed apply_cortex_config_bin/apply_code_config_bin wrappers

#include "hnf_loader.hpp"
#include "gemma4_vision_validator.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ============================================================================
// MINIMAL JSON PARSER
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

double parse_number(const char*& p) {
    char* end;
    double val = strtod(p, &end);
    p = end;
    return val;
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
    if (*p == '"') { p++; return parse_int(p); }
    return parse_int(p);
}

bool get_bool(const char* json, const std::string& key, bool def = false) {
    const char* p = find_key(json, key);
    if (!p) return def;
    if (strncmp(p, "true", 4) == 0) return true;
    if (strncmp(p, "false", 5) == 0) return false;
    return def;
}

std::vector<uint32_t> get_int_array(const char* json, const std::string& key) {
    std::vector<uint32_t> result;
    const char* p = find_key(json, key);
    if (!p || *p != '[') return result;
    p++;
    while (*p && *p != ']') {
        p = skip_ws(p);
        if (*p == ']') break;
        result.push_back(static_cast<uint32_t>(parse_int(p)));
        p = skip_ws(p);
        if (*p == ',') p++;
    }
    return result;
}

// Single-pass JSON object → map
void parse_json_object_to_map(const char* json, 
                               std::unordered_map<std::string, helios::ConfigValue>& out) {
    const char* p = skip_ws(json);
    if (*p != '{') return;
    p++;
    
    while (*p && *p != '}') {
        p = skip_ws(p);
        if (*p == '}') break;
        if (*p == ',') { p++; continue; }
        if (*p != '"') { p++; continue; }
        
        std::string key = parse_string(p);
        p = skip_ws(p);
        if (*p != ':') continue;
        p++;
        p = skip_ws(p);
        
        if (*p == '"') {
            out[key] = parse_string(p);
        } else if (*p == 't') {
            out[key] = true; p += 4;
        } else if (*p == 'f') {
            out[key] = false; p += 5;
        } else if (*p == '-' || (*p >= '0' && *p <= '9')) {
            const char* start = p;
            bool is_float = false;
            while (*p && *p != ',' && *p != '}' && *p != ' ' && *p != '\n') {
                if (*p == '.' || *p == 'e' || *p == 'E') is_float = true;
                p++;
            }
            std::string num_str(start, p - start);
            if (is_float) out[key] = std::stod(num_str);
            else out[key] = (int64_t)std::stoll(num_str);
        } else if (*p == '{' || *p == '[') {
            int depth = 1;
            char open = *p, close = (open == '{') ? '}' : ']';
            p++;
            while (*p && depth > 0) {
                if (*p == '"') { p++; while (*p && *p != '"') { if (*p == '\\') p++; p++; } if (*p == '"') p++; continue; }
                if (*p == open) depth++;
                else if (*p == close) depth--;
                p++;
            }
        } else { p++; }
    }
}

} // anonymous namespace

namespace helios {

namespace detail {

namespace {

bool ends_with(const std::string& value, const char* suffix) {
    const size_t suffix_size = std::strlen(suffix);
    return value.size() >= suffix_size &&
           value.compare(value.size() - suffix_size, suffix_size, suffix) == 0;
}

} // namespace

bool embedding_is_lookup_only(
    const TensorEntry& entry,
    const std::vector<TensorEntry>& manifest) {
    if (!ends_with(entry.name, "token_embedding.weight")) {
        return false;
    }

    // PLE is always a lookup table; it never participates in the output GEMV.
    if (entry.name.find(".ple.") != std::string::npos) {
        return true;
    }

    // The main table is lookup-only only while a distinct output projection
    // exists in the same HNF block. Without it, GraphBuilder deliberately
    // reuses the embedding as lm_head and the full tensor must stay in VRAM.
    return std::any_of(manifest.begin(), manifest.end(), [&](const TensorEntry& tensor) {
        return tensor.block == entry.block && ends_with(tensor.name, "lm_head.weight");
    });
}

} // namespace detail

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

HnfLoader::HnfLoader() {
    memset(&header_, 0, sizeof(header_));
    memset(blocks_, 0, sizeof(blocks_));
}

HnfLoader::~HnfLoader() { close(); }

// ============================================================================
// FILE OPERATIONS
// ============================================================================

bool HnfLoader::open(const std::string& path) {
    close();
    config_.params.clear();
    block_configs_.clear();
    gemma4_config_ = Gemma4Config{};
    has_gemma4_config_ = false;
    gemma4_vision_config_ = Gemma4VisionConfig{};
    has_gemma4_vision_config_ = false;
    last_error_.clear();
    file_.open(path, std::ios::binary);
    if (!file_.is_open()) { std::cerr << "HnfLoader: Cannot open: " << path << std::endl; return false; }
    file_path_ = path;
    
    if (!read_header(file_) || !read_block_table(file_) || !read_manifest(file_) || !parse_manifest()) {
        close(); return false;
    }
    if (!read_execution_hints(file_)) {
        close();
        return false;
    }
    load_tokenizer();
    return true;
}

void HnfLoader::close() { if (file_.is_open()) file_.close(); file_path_.clear(); }

// ============================================================================
// TOKENIZER
// ============================================================================

void HnfLoader::load_tokenizer() {
    htf_loaded_ = false; htf_data_.clear(); tokenizers_.clear();
    const BlockEntry& block = blocks_[BLOCK_TOKENIZER];
    if (block.size == 0 || !file_.is_open()) return;
    
    htf_data_.resize(block.size);
    file_.seekg(block.offset);
    file_.read(reinterpret_cast<char*>(htf_data_.data()), block.size);
    if (!file_.good()) { htf_data_.clear(); return; }
    
    htf_loaded_ = true;
    // Trigger lazy-load of text tokenizer
    tokenizer("text");
}

HTFTokenizer* HnfLoader::tokenizer(const std::string& domain) {
    if (!htf_loaded_ || htf_data_.empty()) return nullptr;
    auto it = tokenizers_.find(domain);
    if (it != tokenizers_.end()) return &it->second;
    HTFTokenizer tok;
    if (tok.load(htf_data_.data(), htf_data_.size(), domain)) {
        tokenizers_[domain] = std::move(tok);
        return &tokenizers_[domain];
    }
    return nullptr;
}

const HTFTokenizer* HnfLoader::tokenizer(const std::string& domain) const {
    if (!htf_loaded_ || htf_data_.empty()) return nullptr;
    auto it = tokenizers_.find(domain);
    if (it != tokenizers_.end()) return &it->second;
    HTFTokenizer tok;
    if (tok.load(htf_data_.data(), htf_data_.size(), domain)) {
        tokenizers_[domain] = std::move(tok);
        return &tokenizers_[domain];
    }
    return nullptr;
}

bool HnfLoader::has_tokenizer(const std::string& domain) const {
    if (!htf_loaded_) return false;
    if (tokenizers_.count(domain)) return true;
    return tokenizer(domain) != nullptr;
}

std::vector<std::string> HnfLoader::tokenizer_domains() const {
    std::vector<std::string> domains;
    if (!htf_loaded_ || htf_data_.size() < 32) return domains;
    uint8_t num = htf_data_[8];
    size_t offset = 32;
    for (int i = 0; i < num && offset + 32 <= htf_data_.size(); i++, offset += 32) {
        switch (htf_data_[offset]) {
            case 0x00: domains.push_back("text"); break;
            case 0x01: domains.push_back("vision"); break;
            case 0x02: domains.push_back("audio"); break;
            case 0x03: domains.push_back("code"); break;
            case 0x04: domains.push_back("cortex"); break;
            default: domains.push_back("unknown"); break;
        }
    }
    return domains;
}

// ============================================================================
// BLOCK ID CONVERSION
// ============================================================================

BlockID HnfLoader::block_id_from_name(const std::string& name) const {
    if (name == "text_model") return BLOCK_TEXT_MODEL;
    if (name == "vision") return BLOCK_VISION;
    if (name == "audio") return BLOCK_AUDIO;
    if (name == "video") return BLOCK_VIDEO;
    if (name == "spatial_3d") return BLOCK_SPATIAL_3D;
    if (name == "personality") return BLOCK_PERSONALITY;
    if (name == "memory") return BLOCK_MEMORY;
    if (name == "cortex") return BLOCK_CORTEX;
    if (name == "code_exec") return BLOCK_CODE_EXEC;
    if (name == "tokenizer") return BLOCK_TOKENIZER;
    if (name == "execution_hints") return BLOCK_EXEC_HINTS;
    if (name == "exec_hints_bin") return BLOCK_EXEC_HINTS_BIN;
    if (name == "tools") return BLOCK_TOOLS;
    if (name == "expert_router") return BLOCK_EXPERT_ROUTER;
    return BLOCK_RESERVED_1;
}

// ============================================================================
// BLOCK OPERATIONS
// ============================================================================

bool HnfLoader::load_block(BlockID id, Engine& engine) {
    if (!file_.is_open()) return false;
    if (id >= HNF_BLOCK_COUNT || blocks_[id].size == 0) return false;
    if (block_states_[id].loaded) return true;

    size_t scratch_budget = 0;
    if (id == BLOCK_VISION && has_gemma4_vision_config_) {
        const Gemma4VisionValidationReport report =
            validate_gemma4_vision_tensors(*this);
        if (!report.ok()) {
            last_error_ = "invalid Gemma 4 vision block: " + report.errors.front();
            std::cerr << "HnfLoader: " << last_error_ << std::endl;
            return false;
        }
        scratch_budget = static_cast<size_t>(report.scratch_upper_bound_bytes);

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (status != cudaSuccess || report.weight_bytes > free_bytes) {
            last_error_ = "insufficient VRAM for Gemma 4 vision weights (need " +
                          std::to_string(report.weight_bytes) + ", free " +
                          std::to_string(free_bytes) + ")";
            std::cerr << "HnfLoader: " << last_error_ << std::endl;
            return false;
        }
    }

    if (!load_block_tensors(id, engine)) return false;
    block_states_[id].scratch_budget_bytes = scratch_budget;
    return true;
}

bool HnfLoader::load_block(const std::string& name, Engine& engine) {
    return load_block(block_id_from_name(name), engine);
}

bool HnfLoader::unload_block(BlockID id, Engine& engine) {
    if (id >= HNF_BLOCK_COUNT) return false;
    auto& s = block_states_[id];
    if (!s.loaded) return true;
    // Contiguous blocks place ownership on their first registered tensor, so
    // remove in reverse order and release the backing allocation last.
    for (auto it = s.tensor_names.rbegin(); it != s.tensor_names.rend(); ++it) {
        engine.tensors().remove(*it);
    }
    s = BlockState{};
    return true;
}

bool HnfLoader::unload_block(const std::string& name, Engine& engine) {
    return unload_block(block_id_from_name(name), engine);
}

bool HnfLoader::is_block_loaded(BlockID id) const { return id < HNF_BLOCK_COUNT && block_states_[id].loaded; }
size_t HnfLoader::block_vram_usage(BlockID id) const { return id < HNF_BLOCK_COUNT ? block_states_[id].vram_bytes : 0; }
size_t HnfLoader::block_tensor_count(BlockID id) const { return id < HNF_BLOCK_COUNT ? block_states_[id].tensor_count : 0; }

size_t HnfLoader::total_vram_usage() const {
    size_t t = 0; for (int i = 0; i < HNF_BLOCK_COUNT; i++) t += block_states_[i].vram_bytes; return t;
}

std::vector<BlockID> HnfLoader::loaded_blocks() const {
    std::vector<BlockID> r;
    for (int i = 0; i < HNF_BLOCK_COUNT; i++) if (block_states_[i].loaded) r.push_back(static_cast<BlockID>(i));
    return r;
}

std::vector<const TensorEntry*> HnfLoader::tensors_for_block(BlockID id) const {
    std::vector<const TensorEntry*> r;
    std::string target = block_name(id);
    for (const auto& t : tensors_) if (t.block == target) r.push_back(&t);
    return r;
}

std::vector<const TensorEntry*> HnfLoader::tensors_for_block(const std::string& name) const {
    return tensors_for_block(block_id_from_name(name));
}

bool HnfLoader::read_tensor_data(const TensorEntry& entry, void* output,
                                 size_t bytes) const {
    if (!output || bytes != entry.size || file_path_.empty()) return false;
    std::ifstream input(file_path_, std::ios::binary);
    if (!input.is_open()) return false;
    input.seekg(static_cast<std::streamoff>(entry.offset));
    input.read(static_cast<char*>(output), static_cast<std::streamsize>(bytes));
    return input.good();
}

bool HnfLoader::load_block_tensors(BlockID id, Engine& engine) {
    std::string target = block_name(id);
    auto& state = block_states_[id];
    state = BlockState{};

    // Gemma 4 vision contains 448 learned scalar clamps. One cudaMalloc per
    // tensor inflated a 321 MiB block to roughly 452 MiB on the test GPU.
    // Preserve all 659 registry entries as views into one owned allocation.
    if (id == BLOCK_VISION && has_gemma4_vision_config_) {
        const BlockEntry& block = blocks_[id];
        void* allocation = nullptr;
        const cudaError_t allocation_status =
            cudaMalloc(&allocation, static_cast<size_t>(block.size));
        if (allocation_status != cudaSuccess) {
            last_error_ = "cudaMalloc failed for contiguous Gemma 4 vision block: " +
                          std::string(cudaGetErrorString(allocation_status));
            return false;
        }

        bool owner_registered = false;
        auto rollback = [&]() {
            for (auto it = state.tensor_names.rbegin();
                 it != state.tensor_names.rend(); ++it) {
                engine.tensors().remove(*it);
            }
            if (!owner_registered) cudaFree(allocation);
            state = BlockState{};
        };

        for (const auto& entry : tensors_) {
            if (entry.block != target) continue;
            const DTypeID dtype = dtype_from_string(entry.dtype);
            size_t numel = 1;
            bool valid_shape = true;
            for (uint32_t dimension : entry.shape) {
                if (dimension == 0 ||
                    numel > std::numeric_limits<size_t>::max() / dimension) {
                    valid_shape = false;
                    break;
                }
                numel *= dimension;
            }
            const size_t expected_size = dtype == DTYPE_INVALID
                ? 0 : dtype_size(dtype, numel);
            const uint64_t relative = entry.offset - block.offset;
            if (!valid_shape || expected_size == 0 ||
                entry.size != expected_size || entry.offset < block.offset ||
                relative > block.size || entry.size > block.size - relative ||
                engine.tensors().exists(entry.name)) {
                last_error_ = "invalid tensor in contiguous Gemma 4 vision block: " +
                              entry.name;
                rollback();
                return false;
            }

            std::vector<uint8_t> host(static_cast<size_t>(entry.size));
            file_.clear();
            file_.seekg(static_cast<std::streamoff>(entry.offset));
            file_.read(reinterpret_cast<char*>(host.data()),
                       static_cast<std::streamsize>(host.size()));
            void* destination = static_cast<uint8_t*>(allocation) + relative;
            if (!file_.good() ||
                cudaMemcpy(destination, host.data(), host.size(),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                last_error_ = "failed to copy contiguous Gemma 4 vision tensor: " +
                              entry.name;
                rollback();
                return false;
            }

            TensorInfo info;
            info.ptr = destination;
            info.shape = entry.shape;
            info.dtype = dtype;
            info.size_bytes = static_cast<size_t>(entry.size);
            info.owns_memory = !owner_registered;
            info.allocation_ptr = info.owns_memory ? allocation : nullptr;
            info.allocation_size = info.owns_memory
                ? static_cast<size_t>(block.size) : 0;
            engine.tensors().register_tensor(entry.name, std::move(info));
            owner_registered = true;
            state.tensor_names.push_back(entry.name);
            state.vram_bytes += entry.size;
            ++state.tensor_count;
        }
        state.loaded = true;
        return true;
    }
    
    for (const auto& entry : tensors_) {
        if (entry.block != target) continue;
        if (!load_tensor(file_, entry, engine.tensors())) {
            for (const auto& n : state.tensor_names) engine.tensors().remove(n);
            state = BlockState{};
            return false;
        }
        state.tensor_names.push_back(entry.name);
        state.vram_bytes += entry.size;
        state.tensor_count++;
    }
    const char* fuse_qkv = getenv("HELIOS_FUSE_QKV");
    const char* fuse_gate_up = getenv("HELIOS_FUSE_GATE_UP");
    if ((fuse_qkv && fuse_qkv[0] == '1') ||
        (fuse_gate_up && fuse_gate_up[0] == '1')) {
        fuse_qkv_weights(engine, state);
    }
    state.loaded = true;
    return true;
}

// Concatena varios tensores por la dimension de salida en uno solo.
//
// Funciona a nivel de bytes porque los formatos compactos son row-major con los
// bloques a lo largo de K: cada fila de salida ocupa K/256 bloques
// consecutivos, asi que pegar las filas de unos detras de otros da exactamente
// el tensor [sum(N), K] que espera el kernel. Devuelve false sin tocar nada si
// las formas no encajan.
bool HnfLoader::concat_tensors(Engine& engine, BlockState& state,
                               const std::vector<std::string>& partes,
                               const std::string& destino) {
    auto& reg = engine.tensors();
    std::vector<const TensorInfo*> t;
    size_t bytes = 0;
    uint32_t filas = 0;
    for (const auto& n : partes) {
        const TensorInfo* p = reg.get(n);
        if (!p || p->shape.size() != 2 || p->host_mapped || p->file_mapped) return false;
        if (!t.empty() && (p->dtype != t[0]->dtype || p->shape[1] != t[0]->shape[1])) return false;
        t.push_back(p);
        bytes += p->size_bytes;
        filas += p->shape[0];
    }
    if (t.empty()) return false;

    void* fused = nullptr;
    if (cudaMalloc(&fused, bytes) != cudaSuccess) {
        std::cerr << "HnfLoader: sin VRAM para fusionar " << destino
                  << "; se sigue sin fusionar" << std::endl;
        return false;
    }
    uint8_t* dst = static_cast<uint8_t*>(fused);
    for (const auto* p : t) {
        if (cudaMemcpy(dst, p->ptr, p->size_bytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
            cudaFree(fused);
            return false;
        }
        dst += p->size_bytes;
    }

    TensorInfo info;
    info.ptr = fused;
    info.shape = {filas, t[0]->shape[1]};
    info.dtype = t[0]->dtype;
    info.size_bytes = bytes;
    info.owns_memory = true;
    info.allocation_ptr = fused;
    reg.register_tensor(destino, info);
    state.tensor_names.push_back(destino);

    // Los originales ya no los mira nadie: el grafo elige la ruta fusionada en
    // cuanto existe el tensor destino.
    for (const auto& n : partes) {
        reg.remove(n);
        auto& v = state.tensor_names;
        v.erase(std::remove(v.begin(), v.end(), n), v.end());
    }
    return true;
}

// Concatena q/k/v y gate/up de cada capa en tensores fusionados.
//
// El grafo YA sabe usarlo (arch.has_fused_qkv -> un matmul + SPLIT_QKV, que en
// decode es zero-copy). Lo unico que faltaba era producir el tensor.
//
// Se puede concatenar a nivel de bytes porque los formatos compactos son
// row-major con bloques a lo largo de K: cada fila de salida ocupa K/256
// bloques consecutivos, asi que pegar las filas de k y v detras de las de q da
// exactamente el tensor [Nq+Nk+Nv, K] que espera el kernel.
//
// Motivo: k y v son matrices pequenas (N=1024) que no llegan a llenar la GPU y
// se quedan limitadas por ocupacion. Medido en aislamiento sobre HQ4.1K:
// 3 matmuls 38,63 us frente a 33,04 us fusionados, un -14,5%.
// ============================================================================
// LOADING (legacy all-at-once)
// ============================================================================

bool HnfLoader::load(const std::string& path, Engine& engine) {
    close();
    file_path_ = path;
    config_.params.clear();
    block_configs_.clear();
    gemma4_config_ = Gemma4Config{};
    has_gemma4_config_ = false;
    gemma4_vision_config_ = Gemma4VisionConfig{};
    has_gemma4_vision_config_ = false;
    last_error_.clear();
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { file_path_.clear(); return false; }
    if (!read_header(f) || !read_block_table(f) || !read_manifest(f) || !parse_manifest()) {
        file_path_.clear();
        return false;
    }
    if (!read_execution_hints(f)) { file_path_.clear(); return false; }
    if (has_gemma4_vision_config_) {
        const Gemma4VisionValidationReport report =
            validate_gemma4_vision_tensors(*this);
        if (!report.ok()) {
            last_error_ = "invalid Gemma 4 vision block: " + report.errors.front();
            file_path_.clear();
            return false;
        }
    }
    return load_tensors(f, engine);
}

void HnfLoader::fuse_qkv_weights(Engine& engine, BlockState& state) {
    // Gemma 4 construye su grafo con una ruta explicita que pide q/k/v y
    // gate/up por nombre y no mira has_fused_qkv / has_fused_gate_up, asi que
    // fusionar le quitaria tensores que necesita. Se deja fuera hasta que esa
    // ruta soporte los tensores fusionados.
    if (has_gemma4_config()) return;

    const char* qkv_env = getenv("HELIOS_FUSE_QKV");
    const char* gate_up_env = getenv("HELIOS_FUSE_GATE_UP");
    const bool fuse_qkv = qkv_env && qkv_env[0] == '1';
    // Compatibilidad: antes HELIOS_FUSE_QKV fusionaba tambien gate/up. Un
    // HELIOS_FUSE_GATE_UP explicito permite medir y controlar ambas rutas por
    // separado sin cambiar los comandos de benchmark ya documentados.
    const bool fuse_gate_up = gate_up_env
        ? gate_up_env[0] == '1'
        : fuse_qkv;

    uint32_t qkv_fusionadas = 0;
    uint32_t gate_up_fusionadas = 0;
    for (uint32_t layer = 0;; ++layer) {
        const std::string b = "text.layer" + std::to_string(layer) + ".";
        if (!engine.tensors().exists(b + "attn.q_proj.weight") &&
            !engine.tensors().exists(b + "mlp.gate.weight")) break;

        // El orden importa: SPLIT_QKV espera q|k|v y add_split_half gate|up.
        if (fuse_qkv &&
            concat_tensors(engine, state,
                           {b + "attn.q_proj.weight", b + "attn.k_proj.weight",
                            b + "attn.v_proj.weight"}, b + "attn.qkv_proj.weight")) {
            qkv_fusionadas++;
        }
        if (fuse_gate_up &&
            concat_tensors(engine, state,
                           {b + "mlp.gate.weight", b + "mlp.up.weight"},
                           b + "mlp.gate_up.weight")) {
            gate_up_fusionadas++;
        }
    }
    if (qkv_fusionadas || gate_up_fusionadas) {
        std::cout << "  [FUSE] q/k/v=" << qkv_fusionadas
                  << " capas, gate/up=" << gate_up_fusionadas
                  << " capas" << std::endl;
    }
}


bool HnfLoader::load_metadata(const std::string& path) {
    close();
    file_path_ = path;
    config_.params.clear();
    block_configs_.clear();
    gemma4_config_ = Gemma4Config{};
    has_gemma4_config_ = false;
    gemma4_vision_config_ = Gemma4VisionConfig{};
    has_gemma4_vision_config_ = false;
    last_error_.clear();
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { file_path_.clear(); return false; }
    if (!read_header(f) || !read_block_table(f) || !read_manifest(f) || !parse_manifest()) {
        file_path_.clear();
        return false;
    }
    if (!read_execution_hints(f)) {
        file_path_.clear();
        return false;
    }
    return true;
}

// ============================================================================
// HEADER / BLOCK TABLE / MANIFEST
// ============================================================================

bool HnfLoader::read_header(std::ifstream& f) {
    f.seekg(0);
    f.read(reinterpret_cast<char*>(&header_), sizeof(HnfHeader));
    if (!f) return false;
    if (memcmp(header_.magic, HNF_MAGIC, 8) != 0) return false;
    if (header_.version_major != 9) return false;
    if (header_.block_count != HNF_BLOCK_COUNT) return false;
    if (header_.header_size != HNF_HEADER_SIZE) return false;
    return true;
}

bool HnfLoader::read_block_table(std::ifstream& f) {
    f.seekg(header_.block_table_offset);
    f.read(reinterpret_cast<char*>(blocks_), sizeof(BlockEntry) * HNF_BLOCK_COUNT);
    return f.good();
}

bool HnfLoader::read_manifest(std::ifstream& f) {
    if (header_.manifest_size == 0) return false;
    f.seekg(header_.manifest_offset);
    manifest_json_.resize(header_.manifest_size);
    f.read(&manifest_json_[0], header_.manifest_size);
    return f.good();
}

bool HnfLoader::parse_manifest() {
    tensors_.clear();
    const char* p = find_key(manifest_json_.c_str(), "tensors");
    if (!p || *p != '[') return false;
    
    p++;
    while (*p) {
        p = skip_ws(p);
        if (*p == ']') break;
        if (*p != '{') { if (*p == ',') { p++; continue; } break; }
        
        const char* obj_start = p;
        int depth = 1; p++;
        while (*p && depth > 0) { if (*p == '{') depth++; else if (*p == '}') depth--; p++; }
        
        std::string obj(obj_start, p - obj_start);
        TensorEntry entry;
        entry.name = get_string(obj.c_str(), "name");
        entry.dtype = get_string(obj.c_str(), "dtype");
        entry.block = get_string(obj.c_str(), "block");
        const char* shape = find_key(obj.c_str(), "shape");
        if (!shape || *shape != '[') {
            last_error_ = "manifest tensor is missing its shape array";
            return false;
        }
        entry.shape = get_int_array(obj.c_str(), "shape");
        entry.offset = static_cast<uint64_t>(get_int(obj.c_str(), "offset"));
        entry.size = static_cast<uint64_t>(get_int(obj.c_str(), "size"));
        if (!entry.name.empty()) tensors_.push_back(entry);
    }
    return !tensors_.empty();
}

// ============================================================================
// EXECUTION HINTS - PRIORIDAD: BINARIO [0xB] > JSON [0xA]
// ============================================================================

bool HnfLoader::read_execution_hints(std::ifstream& f) {
    const bool binary_present = blocks_[BLOCK_EXEC_HINTS_BIN].size > 0;
    const bool has_binary = read_execution_hints_binary(f);
    if (binary_present && !has_binary) {
        if (last_error_.empty()) last_error_ = "invalid binary execution hints";
        std::cerr << "HnfLoader: " << last_error_ << std::endl;
        return false;
    }
//    if (has_binary) std::cout << "HnfLoader: Binary hints loaded (O(1))" << std::endl;
    
    const BlockEntry& json_block = blocks_[BLOCK_EXEC_HINTS];
    if (json_block.size > 0) {
        f.clear(); f.seekg(json_block.offset);
        std::string json; json.resize(json_block.size);
        f.read(&json[0], json_block.size);
        if (f.good()) {
            if (!parse_execution_hints(json)) {
                last_error_ = "invalid JSON execution hints";
                return false;
            }
//            if (!has_binary) std::cout << "HnfLoader: JSON hints loaded (fallback)" << std::endl;
        }
    }
    // Legacy HNF files may have no hint blocks at all. Preserve the historical
    // fallback where graph detection derives the architecture from tensors.
    // A present-but-invalid binary block is still rejected above.
    return true;
}

bool HnfLoader::read_execution_hints_binary(std::ifstream& f) {
    const BlockEntry& block = blocks_[BLOCK_EXEC_HINTS_BIN];
    if (block.size < sizeof(ExecutionHintsBin)) return false;
    
    std::vector<uint8_t> data(block.size);
    f.clear(); f.seekg(block.offset);
    f.read(reinterpret_cast<char*>(data.data()), block.size);
    if (!f.good()) return false;
    
    return parse_execution_hints_binary(data.data(), data.size());
}

bool HnfLoader::parse_execution_hints_binary(const uint8_t* data, size_t size) {
    if (size < sizeof(ExecutionHintsBin)) {
        last_error_ = "binary execution hints header is truncated";
        return false;
    }

    ExecutionHintsBin hints{};
    std::memcpy(&hints, data, sizeof(hints));
    if (hints.magic != HINTS_MAGIC) {
        last_error_ = "binary execution hints have invalid magic";
        return false;
    }
    if (hints.version_major != 1) {
        last_error_ = "unsupported binary execution hints major version";
        return false;
    }

    auto range_fits = [size](uint32_t offset, size_t bytes) {
        const size_t start = static_cast<size_t>(offset);
        return start <= size && bytes <= size - start;
    };

    TextModelConfigBin text_config{};
    bool has_text_config = false;
    if (hints.text_offset > 0) {
        if (!range_fits(hints.text_offset, sizeof(TextModelConfigBin))) {
            last_error_ = "binary text config is outside the hints block";
            return false;
        }
        std::memcpy(&text_config, data + hints.text_offset, sizeof(text_config));
        has_text_config = true;

        if (text_config.arch == ARCH_GEMMA4 &&
            !parse_gemma4_extension(data, size, hints, text_config)) {
            return false;
        }
    } else if (hints.num_text_models != 0) {
        last_error_ = "binary hints declare a text model without a text config";
        return false;
    }
    
    // Text → config_ (default) AND block_configs_[TEXT]
    if (has_text_config) {
        apply_text_config_bin(text_config, config_);
        apply_text_config_bin(text_config, block_configs_[BLOCK_TEXT_MODEL]);
    }
    // Vision
    if (hints.vision_offset > 0) {
        if (hints.num_vision_models != 1) {
            last_error_ = "binary vision config requires exactly one vision model";
            return false;
        }
        if (!range_fits(hints.vision_offset, sizeof(VisionModelConfigBin))) {
            last_error_ = "binary vision config is outside the hints block";
            return false;
        }
        VisionModelConfigBin vision_config{};
        std::memcpy(&vision_config, data + hints.vision_offset, sizeof(vision_config));
        if (vision_config.encoder_type > VISION_ENCODER_GEMMA4) {
            last_error_ = "binary vision config has an unknown encoder type";
            return false;
        }
        if (vision_config.encoder_type == VISION_ENCODER_GEMMA4 &&
            !parse_gemma4_vision_extension(data, size, hints, vision_config)) {
            return false;
        }
        if (vision_config.encoder_type != VISION_ENCODER_GEMMA4) {
            uint32_t unexpected_offset = 0;
            uint32_t unexpected_size = 0;
            std::memcpy(&unexpected_offset, hints.reserved + 12,
                        sizeof(unexpected_offset));
            std::memcpy(&unexpected_size, hints.reserved + 16,
                        sizeof(unexpected_size));
            if (unexpected_offset != 0 || unexpected_size != 0) {
                last_error_ = "non-Gemma vision config carries an unexpected GM4V pointer";
                return false;
            }
        }
        apply_vision_config_bin(vision_config);
    } else if (hints.num_vision_models != 0) {
        last_error_ = "binary hints declare a vision model without a vision config";
        return false;
    } else {
        uint32_t unexpected_offset = 0;
        uint32_t unexpected_size = 0;
        std::memcpy(&unexpected_offset, hints.reserved + 12,
                    sizeof(unexpected_offset));
        std::memcpy(&unexpected_size, hints.reserved + 16,
                    sizeof(unexpected_size));
        if (unexpected_offset != 0 || unexpected_size != 0) {
            last_error_ = "binary hints carry GM4V without a vision config";
            return false;
        }
    }
    // Cortex → block_configs_[CORTEX] (single call, no wrapper)
    if (hints.cortex_offset > 0) {
        if (!range_fits(hints.cortex_offset, sizeof(TextModelConfigBin))) {
            last_error_ = "binary cortex config is outside the hints block";
            return false;
        }
        TextModelConfigBin cortex_config{};
        std::memcpy(&cortex_config, data + hints.cortex_offset, sizeof(cortex_config));
        apply_text_config_bin(cortex_config, block_configs_[BLOCK_CORTEX]);
    }
    // Code → block_configs_[CODE_EXEC] (single call, no wrapper)
    if (hints.code_offset > 0) {
        if (!range_fits(hints.code_offset, sizeof(TextModelConfigBin))) {
            last_error_ = "binary code config is outside the hints block";
            return false;
        }
        TextModelConfigBin code_config{};
        std::memcpy(&code_config, data + hints.code_offset, sizeof(code_config));
        apply_text_config_bin(code_config, block_configs_[BLOCK_CODE_EXEC]);
    }
    
    config_.set("text_enabled", (hints.flags & 0x0001) != 0);
    config_.set("vision_enabled", (hints.flags & 0x0002) != 0);
    config_.set("audio_enabled", (hints.flags & 0x0004) != 0);
    config_.set("code_enabled", (hints.flags & 0x0008) != 0);
    config_.set("cortex_enabled", (hints.flags & 0x0010) != 0);
    return true;
}

bool HnfLoader::parse_gemma4_extension(const uint8_t* data, size_t size,
                                       const ExecutionHintsBin& hints,
                                       const TextModelConfigBin& text_config) {
    if (std::memcmp(hints.reserved + 8, "GM4X", 4) != 0) {
        last_error_ = "Gemma 4 binary hints are missing the GM4X marker";
        return false;
    }

    uint32_t extension_offset = 0;
    uint32_t extension_size = 0;
    std::memcpy(&extension_offset, hints.reserved, sizeof(extension_offset));
    std::memcpy(&extension_size, hints.reserved + 4, sizeof(extension_size));

    const size_t start = static_cast<size_t>(extension_offset);
    const size_t bytes = static_cast<size_t>(extension_size);
    if (start > size || bytes > size - start || bytes < sizeof(Gemma4ExtensionHeaderBin)) {
        last_error_ = "GM4X extension is outside the binary hints block";
        return false;
    }

    Gemma4ExtensionHeaderBin header{};
    std::memcpy(&header, data + start, sizeof(header));
    if (std::memcmp(header.magic, "GM4X", 4) != 0) {
        last_error_ = "GM4X extension has invalid magic";
        return false;
    }
    if (header.version != 1) {
        last_error_ = "unsupported GM4X version";
        return false;
    }
    if (header.layer_record_size != sizeof(Gemma4LayerConfigBin)) {
        last_error_ = "unsupported GM4X layer record size";
        return false;
    }
    if (header.layer_count == 0 || header.layer_count != text_config.num_hidden_layers) {
        last_error_ = "GM4X layer count does not match the text config";
        return false;
    }

    const size_t record_count = static_cast<size_t>(header.layer_count);
    if (record_count > (bytes - sizeof(header)) / sizeof(Gemma4LayerConfigBin)) {
        last_error_ = "GM4X layer records are truncated";
        return false;
    }
    const size_t required_size = sizeof(header) + record_count * sizeof(Gemma4LayerConfigBin);
    if (required_size != bytes) {
        last_error_ = "GM4X extension size does not match its layer records";
        return false;
    }

    Gemma4Config parsed{};
    parsed.version = header.version;
    parsed.flags = header.flags;
    parsed.global_head_dim = header.global_head_dim;
    parsed.ple_hidden_size = header.ple_hidden_size;
    parsed.num_kv_shared_layers = header.num_kv_shared_layers;
    parsed.layers.reserve(record_count);

    for (size_t i = 0; i < record_count; ++i) {
        Gemma4LayerConfigBin record{};
        const size_t record_offset = start + sizeof(header) + i * sizeof(record);
        std::memcpy(&record, data + record_offset, sizeof(record));

        if (record.attention_kind > 1 || record.rope_type > ROPE_PROPORTIONAL ||
            record.head_dim == 0 || record.intermediate_size == 0 ||
            !std::isfinite(record.rope_theta) || record.rope_theta <= 0.0f ||
            !std::isfinite(record.partial_rotary_factor) ||
            record.partial_rotary_factor <= 0.0f) {
            last_error_ = "GM4X layer " + std::to_string(i) + " has invalid fields";
            return false;
        }

        Gemma4LayerConfig layer{};
        layer.attention_kind = record.attention_kind;
        layer.sliding_window = record.sliding_window;
        layer.head_dim = record.head_dim;
        layer.intermediate_size = record.intermediate_size;
        layer.rope_type = record.rope_type;
        layer.flags = record.flags;
        layer.rope_theta = record.rope_theta;
        layer.partial_rotary_factor = record.partial_rotary_factor;
        layer.kv_share_group = record.kv_share_group;
        parsed.layers.push_back(layer);
    }

    gemma4_config_ = std::move(parsed);
    has_gemma4_config_ = true;
    return true;
}

bool HnfLoader::parse_gemma4_vision_extension(
    const uint8_t* data, size_t size, const ExecutionHintsBin& hints,
    const VisionModelConfigBin& vision_config) {
    uint32_t extension_offset = 0;
    uint32_t extension_size = 0;
    std::memcpy(&extension_offset, hints.reserved + 12, sizeof(extension_offset));
    std::memcpy(&extension_size, hints.reserved + 16, sizeof(extension_size));

    const size_t start = static_cast<size_t>(extension_offset);
    const size_t bytes = static_cast<size_t>(extension_size);
    if (start > size || bytes > size - start ||
        bytes != sizeof(Gemma4VisionExtensionBin)) {
        last_error_ = "GM4V extension is missing, truncated, or has the wrong size";
        return false;
    }

    const auto overlaps = [](size_t a_start, size_t a_size,
                             size_t b_start, size_t b_size) {
        return a_start < b_start + b_size && b_start < a_start + a_size;
    };
    if (overlaps(start, bytes, 0, sizeof(ExecutionHintsBin)) ||
        overlaps(start, bytes, hints.vision_offset,
                 sizeof(VisionModelConfigBin)) ||
        (hints.text_offset > 0 &&
         overlaps(start, bytes, hints.text_offset, sizeof(TextModelConfigBin)))) {
        last_error_ = "GM4V extension overlaps a binary hints record";
        return false;
    }
    if (std::memcmp(hints.reserved + 8, "GM4X", 4) == 0) {
        uint32_t text_extension_offset = 0;
        uint32_t text_extension_size = 0;
        std::memcpy(&text_extension_offset, hints.reserved,
                    sizeof(text_extension_offset));
        std::memcpy(&text_extension_size, hints.reserved + 4,
                    sizeof(text_extension_size));
        if (overlaps(start, bytes, text_extension_offset,
                     text_extension_size)) {
            last_error_ = "GM4V extension overlaps the GM4X extension";
            return false;
        }
    }

    Gemma4VisionExtensionBin extension{};
    std::memcpy(&extension, data + start, sizeof(extension));
    if (std::memcmp(extension.magic, "GM4V", 4) != 0) {
        last_error_ = "GM4V extension has invalid magic";
        return false;
    }
    if (extension.version != 1 ||
        extension.record_size != sizeof(Gemma4VisionExtensionBin)) {
        last_error_ = "unsupported GM4V version or record size";
        return false;
    }

    constexpr uint32_t required_flags =
        GEMMA4_VISION_FLAG_CLIPPED_LINEARS |
        GEMMA4_VISION_FLAG_DYNAMIC_TOKENS |
        GEMMA4_VISION_FLAG_LEARNED_XY_POS |
        GEMMA4_VISION_FLAG_ROPE_2D |
        GEMMA4_VISION_FLAG_NONCAUSAL |
        GEMMA4_VISION_FLAG_CONVERT_RGB |
        GEMMA4_VISION_FLAG_RESCALE |
        GEMMA4_VISION_FLAG_RESIZE |
        GEMMA4_VISION_FLAG_ANTIALIAS |
        GEMMA4_VISION_FLAG_QK_NORM_WEIGHTED |
        GEMMA4_VISION_FLAG_V_NORM_WEIGHTLESS |
        GEMMA4_VISION_FLAG_POOL_NORM_WEIGHTLESS |
        GEMMA4_VISION_FLAG_POOL_FP32_SQRT_HIDDEN |
        GEMMA4_VISION_FLAG_INPUT_MINUS_ONE_ONE;
    constexpr uint32_t supported_flags = required_flags |
        GEMMA4_VISION_FLAG_STANDARDIZE | GEMMA4_VISION_FLAG_NORMALIZE;
    if ((extension.flags & required_flags) != required_flags ||
        (extension.flags & ~supported_flags) != 0 ||
        (extension.flags & GEMMA4_VISION_FLAG_STANDARDIZE) != 0 ||
        (extension.flags & GEMMA4_VISION_FLAG_NORMALIZE) != 0) {
        last_error_ = "GM4V flags request an unsupported vision contract";
        return false;
    }

    const auto finite_positive = [](float value) {
        return std::isfinite(value) && value > 0.0f;
    };
    const uint64_t attention_width =
        static_cast<uint64_t>(vision_config.num_attention_heads) * extension.head_dim;
    if (vision_config.hidden_size == 0 || vision_config.num_hidden_layers == 0 ||
        vision_config.num_attention_heads == 0 || vision_config.intermediate_size == 0 ||
        vision_config.num_channels != 3 || vision_config.projector_type != 0 ||
        extension.num_key_value_heads != vision_config.num_attention_heads ||
        attention_width != vision_config.hidden_size ||
        extension.position_embedding_size == 0 || extension.pooling_kernel_size == 0 ||
        extension.max_soft_tokens == 0 || extension.projection_dim == 0 ||
        extension.patch_size == 0 || extension.patch_size != vision_config.patch_size ||
        extension.projection_dim != vision_config.projection_dim ||
        extension.max_soft_tokens != vision_config.num_image_tokens ||
        extension.image_token_id != vision_config.image_token_id ||
        extension.image_token_id < 0 || extension.boi_token_id < 0 ||
        extension.eoi_token_id < 0 || extension.pad_token_id < 0 ||
        !finite_positive(extension.rope_theta) ||
        !finite_positive(extension.attention_scale) ||
        !finite_positive(extension.rms_norm_eps) ||
        !finite_positive(extension.rescale_factor) ||
        std::fabs(extension.rms_norm_eps - vision_config.layer_norm_eps) > 1.0e-12f ||
        std::fabs(extension.attention_scale - 1.0f) > 1.0e-6f ||
        extension.resize_multiple != extension.patch_size * extension.pooling_kernel_size ||
        extension.resample != 3 || extension.activation != 1 ||
        extension.patch_order != 1 ||
        extension.position_order != 1 || extension.reserved != 0) {
        last_error_ = "GM4V fields are inconsistent or unsupported";
        return false;
    }

    Gemma4VisionConfig parsed{};
    parsed.version = extension.version;
    parsed.flags = extension.flags;
    parsed.num_key_value_heads = extension.num_key_value_heads;
    parsed.head_dim = extension.head_dim;
    parsed.position_embedding_size = extension.position_embedding_size;
    parsed.pooling_kernel_size = extension.pooling_kernel_size;
    parsed.max_soft_tokens = extension.max_soft_tokens;
    parsed.projection_dim = extension.projection_dim;
    parsed.patch_size = extension.patch_size;
    parsed.image_token_id = extension.image_token_id;
    parsed.boi_token_id = extension.boi_token_id;
    parsed.eoi_token_id = extension.eoi_token_id;
    parsed.pad_token_id = extension.pad_token_id;
    parsed.rope_theta = extension.rope_theta;
    parsed.attention_scale = extension.attention_scale;
    parsed.rms_norm_eps = extension.rms_norm_eps;
    parsed.rescale_factor = extension.rescale_factor;
    parsed.resize_multiple = extension.resize_multiple;
    parsed.resample = extension.resample;
    parsed.activation = extension.activation;
    parsed.patch_order = extension.patch_order;
    parsed.position_order = extension.position_order;
    gemma4_vision_config_ = parsed;
    has_gemma4_vision_config_ = true;
    return true;
}

// ============================================================================
// APPLY CONFIG — SINGLE GENERIC VERSION
// ============================================================================

void HnfLoader::apply_text_config_bin(const TextModelConfigBin& cfg, ModelConfig& target) {
    static const char* arch_names[] = {
        "unknown","llama","llama2","llama3","qwen","qwen2",
        "phi3","phi4","gemma","gemma2","mistral","mixtral","deepseek","clip","siglip","gemma4"
    };
    const uint32_t arch_index = cfg.arch <= ARCH_GEMMA4 ? cfg.arch : ARCH_UNKNOWN;
    target.set("arch", std::string(arch_names[arch_index]));
    
    static const char* dtype_names[] = {"fp16","bf16","fp32"};
    target.set("dtype", std::string(dtype_names[std::min(cfg.dtype, 2u)]));
    
    target.set("num_hidden_layers", (int64_t)cfg.num_hidden_layers);
    target.set("hidden_size", (int64_t)cfg.hidden_size);
    target.set("intermediate_size", (int64_t)cfg.intermediate_size);
    target.set("vocab_size", (int64_t)cfg.vocab_size);
    target.set("max_position_embeddings", (int64_t)cfg.max_position_embeddings);
    target.set("num_attention_heads", (int64_t)cfg.num_attention_heads);
    target.set("num_key_value_heads", (int64_t)cfg.num_key_value_heads);
    target.set("head_dim", (int64_t)cfg.head_dim);
    
    static const char* attn_types[] = {"mha","gqa","mqa"};
    target.set("attention_type", std::string(attn_types[std::min(cfg.attention_type, 2u)]));
    
    static const char* mlp_types[] = {"swiglu","swiglu_fused","geglu","gated","standard"};
    target.set("mlp_type", std::string(mlp_types[std::min(cfg.mlp_type, 4u)]));
    
    // Map MLP type to activation
    static const char* mlp_acts[] = {"silu","silu","gelu","silu","gelu"};
    target.set("mlp_activation", std::string(mlp_acts[std::min(cfg.mlp_type, 4u)]));
    
    static const char* norm_types[] = {"rmsnorm","layernorm"};
    target.set("norm_type", std::string(norm_types[std::min(cfg.norm_type, 1u)]));
    target.set("rms_norm_eps", (double)cfg.rms_norm_eps);
    
    static const char* rope_types[] = {"default","llama3","linear","dynamic","yarn","longrope","su","none","proportional"};
    const uint32_t rope_index = cfg.rope_type <= ROPE_PROPORTIONAL ? cfg.rope_type : ROPE_DEFAULT;
    target.set("rope_type", std::string(rope_types[rope_index]));
    target.set("rope_theta", (double)cfg.rope_theta);
    target.set("rope_scaling_factor", (double)cfg.rope_scaling_factor);
    target.set("partial_rotary_factor", (double)cfg.partial_rotary_factor);
    target.set("rope_dim", (int64_t)cfg.rope_dim);
    
    target.set("attention_bias", (cfg.flags & CFG_FLAG_ATTENTION_BIAS) != 0);
    target.set("mlp_bias", (cfg.flags & CFG_FLAG_MLP_BIAS) != 0);
    target.set("norm_bias", (cfg.flags & CFG_FLAG_NORM_BIAS) != 0);
    target.set("use_qk_norm", (cfg.flags & CFG_FLAG_USE_QK_NORM) != 0);
    target.set("parallel_attention", (cfg.flags & CFG_FLAG_PARALLEL_ATTENTION) != 0);
    target.set("tie_word_embeddings", (cfg.flags & CFG_FLAG_TIE_EMBEDDINGS) != 0);
}

void HnfLoader::apply_vision_config_bin(const VisionModelConfigBin& cfg) {
    ModelConfig& vc = block_configs_[BLOCK_VISION];
    static const char* enc_types[] = {"clip","siglip","vit","eva","gemma4"};
    const uint32_t encoder = cfg.encoder_type <= VISION_ENCODER_GEMMA4
        ? cfg.encoder_type : VISION_ENCODER_CLIP;
    vc.set("encoder_type", std::string(enc_types[encoder]));
    vc.set("image_size", (int64_t)cfg.image_size);
    vc.set("patch_size", (int64_t)cfg.patch_size);
    vc.set("hidden_size", (int64_t)cfg.hidden_size);
    vc.set("num_hidden_layers", (int64_t)cfg.num_hidden_layers);
    vc.set("num_attention_heads", (int64_t)cfg.num_attention_heads);
    vc.set("intermediate_size", (int64_t)cfg.intermediate_size);
    vc.set("num_channels", (int64_t)cfg.num_channels);
    vc.set("layer_norm_eps", (double)cfg.layer_norm_eps);
    vc.set("projection_dim", (int64_t)cfg.projection_dim);
    vc.set("num_image_tokens", (int64_t)cfg.num_image_tokens);
    vc.set("image_token_id", (int64_t)cfg.image_token_id);
    if (has_gemma4_vision_config_) {
        const Gemma4VisionConfig& gemma = gemma4_vision_config_;
        vc.set("num_key_value_heads", (int64_t)gemma.num_key_value_heads);
        vc.set("head_dim", (int64_t)gemma.head_dim);
        vc.set("position_embedding_size", (int64_t)gemma.position_embedding_size);
        vc.set("pooling_kernel_size", (int64_t)gemma.pooling_kernel_size);
        vc.set("max_soft_tokens", (int64_t)gemma.max_soft_tokens);
        vc.set("rope_theta", (double)gemma.rope_theta);
        vc.set("attention_scale", (double)gemma.attention_scale);
        vc.set("rms_norm_eps", (double)gemma.rms_norm_eps);
        vc.set("resize_multiple", (int64_t)gemma.resize_multiple);
    }
}

// ============================================================================
// JSON EXECUTION HINTS — NOW PROCESSES ALL BLOCKS
// ============================================================================

bool HnfLoader::parse_execution_hints(const std::string& json) {
    const char* j = json.c_str();
    
    // Global flags
    if (get_bool(j, "text_enabled", false)) config_.set("text_enabled", true);
    if (get_bool(j, "code_enabled", false)) config_.set("code_enabled", true);
    if (get_bool(j, "vision_enabled", false)) config_.set("vision_enabled", true);
    if (get_bool(j, "cortex_enabled", false)) config_.set("cortex_enabled", true);
    
    // Structure: { "text": {...}, "vision": {...}, "cortex": {...}, "code": {...} }
    // Parse each block into its respective config
    
    struct BlockMapping {
        const char* json_key;
        BlockID block_id;
        bool is_default;  // Also write to config_ (text is default)
    };
    
    static const BlockMapping mappings[] = {
        {"text",   BLOCK_TEXT_MODEL, true},
        {"vision", BLOCK_VISION,    false},
        {"cortex", BLOCK_CORTEX,    false},
        {"code",   BLOCK_CODE_EXEC, false},
    };
    
    for (const auto& m : mappings) {
        const char* obj = find_key(j, m.json_key);
        if (!obj || *obj != '{') continue;
        
        // Parse into block-specific config
        parse_json_object_to_map(obj, block_configs_[m.block_id].params);
        
        // Text is also the default config
        if (m.is_default) {
            parse_json_object_to_map(obj, config_.params);
        }
    }
    
    return true;
}

// ============================================================================
// DTYPE CONVERSION
// ============================================================================

DTypeID HnfLoader::dtype_from_string(const std::string& s) const {
    if (s == "fp32") return dtype::FP32();
    if (s == "fp16") return dtype::FP16();
    if (s == "bf16") return dtype::BF16();
    // hq4k y hq5k (cabecera de 128 B, 8,0 y 9,0 bpw) se retiraron el
    // 2026-08-01: ningun HNF los usaba y obligaban a duplicar cada mejora del
    // cuantizador. Un fichero antiguo que los traiga da DTYPE_INVALID y el
    // loader lo rechaza con mensaje claro, en vez de malinterpretar los bytes.
    if (s == "hq31k") return dtype::HQ31K();
    if (s == "hq41k") return dtype::HQ41K();
    if (s == "hq51k") return dtype::HQ51K();
    if (s == "int8") return dtype::INT8();
    if (s == "int32") return dtype::INT32();
    return DTYPE_INVALID;
}

// ============================================================================
// TENSOR LOADING — USES block_id_from_name() CONSISTENTLY
// ============================================================================

bool HnfLoader::load_tensors(std::ifstream& f, Engine& engine) {
    for (const auto& entry : tensors_) {
        if (!load_tensor(f, entry, engine.tensors())) {
            std::cerr << "HnfLoader: Failed: " << entry.name << std::endl;
            return false;
        }
    }
    return true;
}

bool HnfLoader::load_tensor(std::ifstream& f, const TensorEntry& entry,
                            TensorRegistry& registry) {
    const DTypeID dtype = dtype_from_string(entry.dtype);
    if (dtype == DTYPE_INVALID) {
        std::cerr << "HnfLoader: Unknown dtype: " << entry.dtype
                  << " (" << entry.name << ")" << std::endl;
        return false;
    }
    size_t numel = 1;
    for (uint32_t dim : entry.shape) {
        if (dim == 0 || numel > std::numeric_limits<size_t>::max() / dim) {
            std::cerr << "HnfLoader: Invalid shape: " << entry.name << std::endl;
            return false;
        }
        numel *= dim;
    }
    const size_t expected_size = dtype_size(dtype, numel);
    if (expected_size == 0 || entry.size != expected_size) {
        std::cerr << "HnfLoader: Size mismatch: " << entry.name
                  << " manifest=" << entry.size
                  << " expected=" << expected_size << std::endl;
        return false;
    }

    // Use block_id_from_name() — single source of truth, covers all blocks
    BlockID block_id = block_id_from_name(entry.block);
    if (block_id == BLOCK_RESERVED_1) {
        std::cerr << "HnfLoader: Unknown block: " << entry.block << std::endl;
        return false;
    }
    
    const BlockEntry& block = blocks_[block_id];
    if (block.size == 0) {
        std::cerr << "HnfLoader: Block is empty: " << entry.block << std::endl;
        return false;
    }
    
    // ¿Este tensor vive en RAM? La tabla de embeddings ocupa 1.16 GB en un 8B
    // pero por token solo se lee UNA fila (8 KB). Tenerla en VRAM es malgastar
    // memoria carísima: en RAM anclada+mapeada la GPU lee esa fila por PCIe en
    // ~0.3 µs sobre los ~18.000 µs que dura un token (0.002% de coste).
    // Los pesos que SÍ se leen enteros cada token (capas, lm_head) jamás:
    // PCIe es 15× más lento que la VRAM.
    const char* off = getenv("HELIOS_EMBED_IN_RAM");
    const char* mmap_env = getenv("HELIOS_EMBED_MMAP");
    const bool mmap_requested = mmap_env && *mmap_env == '1';
    const bool offload_requested = (off && *off == '1') || mmap_requested;
    const bool embedding_candidate =
        entry.name.find("token_embedding") != std::string::npos;
    // Verifier-only escape hatch: the all-FP16 Gemma 4 control otherwise
    // misses an 8 GiB card by its distinct 768 MiB output head. Reading that
    // full matrix through HMM is intentionally slow and never enabled by the
    // production offload flag alone.
    const char* verify_head_env = getenv("HELIOS_VERIFY_LM_HEAD_MMAP");
    const bool verify_head_mmap = mmap_requested && verify_head_env &&
        *verify_head_env == '1' && entry.name.size() >= 14 &&
        entry.name.compare(entry.name.size() - 14, 14, "lm_head.weight") == 0;
    bool to_host = verify_head_mmap ||
        (offload_requested && embedding_candidate &&
         detail::embedding_is_lookup_only(entry, tensors_));
    if (offload_requested && embedding_candidate && !to_host &&
        entry.name.find(".ple.") == std::string::npos) {
        std::cout << "  [VRAM] " << entry.name
                  << " compartido con lm_head; no se puede mapear a RAM"
                  << std::endl;
    }

    void* d_ptr = nullptr;
    void* allocation_ptr = nullptr;
    size_t allocation_size = 0;
    bool file_mapped = false;
    cudaError_t err = cudaSuccess;

    // Linux HMM permite que un kernel CUDA lea directamente un mmap pageable.
    // El puntero es estable (compatible con CUDA Graph), pero solo se vuelven
    // residentes las paginas de las filas realmente consultadas.
    if (to_host && mmap_requested && !file_path_.empty()) {
        int device = 0;
        int pageable = 0;
        if (cudaGetDevice(&device) == cudaSuccess) {
            cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess,
                                   device);
        }
        if (pageable) {
            const long page_size = sysconf(_SC_PAGESIZE);
            if (page_size > 0) {
                const uint64_t abs_offset = entry.offset;
                const uint64_t page_mask = uint64_t(page_size - 1);
                const uint64_t map_offset = abs_offset & ~page_mask;
                const size_t delta = size_t(abs_offset - map_offset);
                if (entry.size <= std::numeric_limits<size_t>::max() - delta) {
                    const size_t map_size = delta + entry.size;
                    const int fd = ::open(file_path_.c_str(), O_RDONLY | O_CLOEXEC);
                    if (fd >= 0) {
                        struct stat st {};
                        const bool in_bounds = fstat(fd, &st) == 0 && st.st_size >= 0 &&
                            map_offset <= uint64_t(st.st_size) &&
                            map_size <= uint64_t(st.st_size) - map_offset;
                        if (in_bounds) {
                            void* base = mmap(nullptr, map_size, PROT_READ,
                                              MAP_PRIVATE, fd, off_t(map_offset));
                            if (base != MAP_FAILED) {
                                madvise(base, map_size, MADV_RANDOM);
                                allocation_ptr = base;
                                allocation_size = map_size;
                                d_ptr = static_cast<uint8_t*>(base) + delta;
                                file_mapped = true;
                                std::cout << "  [MMAP] " << entry.name << " ("
                                          << entry.size / (1024 * 1024)
                                          << " MB, paginas bajo demanda)" << std::endl;
                            }
                        }
                        ::close(fd);
                    }
                }
            }
        }
        if (!file_mapped) {
            std::cerr << "HnfLoader: mmap pageable no disponible para "
                      << entry.name << " — usando RAM mapeada" << std::endl;
        }
    }

    std::vector<uint8_t> host_data;
    if (!file_mapped) {
        const uint64_t abs_offset = entry.offset;
        f.clear();
        f.seekg(static_cast<std::streamoff>(abs_offset), std::ios::beg);
        if (!f.good()) {
            std::cerr << "HnfLoader: Seek failed: " << entry.name
                      << " (offset=" << abs_offset << ")" << std::endl;
            return false;
        }
        host_data.resize(entry.size);
        f.read(reinterpret_cast<char*>(host_data.data()), entry.size);
        if (!f) {
            std::cerr << "HnfLoader: Read failed: " << entry.name << std::endl;
            return false;
        }
    }

    if (to_host && !file_mapped) {
        void* h_ptr = nullptr;
        err = cudaHostAlloc(&h_ptr, entry.size, cudaHostAllocMapped);
        if (err == cudaSuccess) {
            memcpy(h_ptr, host_data.data(), entry.size);
            err = cudaHostGetDevicePointer(&d_ptr, h_ptr, 0);
        }
        if (err != cudaSuccess) {
            if (h_ptr) cudaFreeHost(h_ptr);
            std::cerr << "HnfLoader: host-mapped alloc failed para " << entry.name
                      << " — cayendo a VRAM" << std::endl;
            to_host = false;
        } else {
            allocation_ptr = h_ptr;
            std::cout << "  [RAM] " << entry.name << " ("
                      << entry.size / (1024 * 1024) << " MB fuera de VRAM)" << std::endl;
        }
    }
    if (!to_host) {
        err = cudaMalloc(&d_ptr, entry.size);
        if (err != cudaSuccess) {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            std::cerr << "HnfLoader: cudaMalloc failed for " << entry.name
                      << " (" << entry.size << " bytes, "
                      << free_bytes << " free): " << cudaGetErrorString(err)
                      << std::endl;
            return false;
        }
        allocation_ptr = d_ptr;
        err = cudaMemcpy(d_ptr, host_data.data(), entry.size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_ptr);
            std::cerr << "HnfLoader: cudaMemcpy failed" << std::endl;
            return false;
        }
    }

    TensorInfo info;
    info.ptr = d_ptr;
    info.shape = entry.shape;
    info.dtype = dtype;
    info.size_bytes = entry.size;
    info.owns_memory = true;
    info.host_mapped = to_host && !file_mapped;
    info.file_mapped = file_mapped;
    info.allocation_ptr = allocation_ptr;
    info.allocation_size = allocation_size;
    
    registry.register_tensor(entry.name, info);
    return true;
}

// ============================================================================
// DEBUG OUTPUT
// ============================================================================

void HnfLoader::print_info() const {
    std::cout << "=== HNF v" << header_.version_major << "." << header_.version_minor << " ===" << std::endl;
    std::cout << "File size: " << (header_.file_size / 1024 / 1024) << " MB" << std::endl;
    std::cout << "Flags: 0x" << std::hex << header_.flags << std::dec << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Blocks ===" << std::endl;
    for (int i = 0; i < HNF_BLOCK_COUNT; i++) {
        if (blocks_[i].size > 0) {
            std::cout << "  0x" << std::hex << i << std::dec 
                      << " " << std::setw(16) << std::left << block_name(i)
                      << " size=" << std::setw(12) << blocks_[i].size
                      << " offset=" << blocks_[i].offset << std::endl;
        }
    }
    std::cout << std::endl;
    
    // Print configs for each block that has one
    struct { BlockID id; const char* label; } configs[] = {
        {BLOCK_TEXT_MODEL, "TEXT"}, {BLOCK_CORTEX, "CORTEX"},
        {BLOCK_CODE_EXEC, "CODE"}, {BLOCK_VISION, "VISION"},
    };
    
    for (const auto& c : configs) {
        auto it = block_configs_.find(c.id);
        const ModelConfig& cfg = (c.id == BLOCK_TEXT_MODEL) ? config_ : 
                                  (it != block_configs_.end() ? it->second : config_);
        if (!cfg.has("arch") && c.id != BLOCK_TEXT_MODEL) continue;
        if (c.id != BLOCK_TEXT_MODEL && it == block_configs_.end()) continue;
        
        std::cout << "=== " << c.label << " Model Config ===" << std::endl;
        if (cfg.has("arch")) std::cout << "  Architecture: " << cfg.arch() << std::endl;
        if (cfg.has("num_hidden_layers")) std::cout << "  Layers: " << cfg.num_hidden_layers() << std::endl;
        if (cfg.has("hidden_size")) std::cout << "  Hidden: " << cfg.hidden_size() << std::endl;
        if (cfg.has("intermediate_size")) std::cout << "  Intermediate: " << cfg.intermediate_size() << std::endl;
        if (cfg.has("vocab_size")) std::cout << "  Vocab: " << cfg.vocab_size() << std::endl;
        if (cfg.has("num_attention_heads")) {
            std::cout << "  Heads: " << cfg.num_attention_heads();
            if (cfg.has("num_key_value_heads")) std::cout << " (KV: " << cfg.num_key_value_heads() << ")";
            std::cout << std::endl;
        }
        if (cfg.has("attention_type")) std::cout << "  Attention: " << cfg.attention_type() << std::endl;
        if (cfg.has("norm_type")) std::cout << "  Norm: " << cfg.norm_type() << " (eps=" << cfg.rms_norm_eps() << ")" << std::endl;
        if (cfg.has("rope_theta")) std::cout << "  RoPE: theta=" << cfg.rope_theta() << std::endl;
        if (cfg.has("mlp_type")) std::cout << "  MLP: " << cfg.get<std::string>("mlp_type", "") << std::endl;
        std::cout << std::endl;
    }
}

void HnfLoader::print_tensors() const {
    std::cout << "=== Tensors (" << tensors_.size() << ") ===" << std::endl;
    size_t total = 0;
    for (const auto& t : tensors_) {
        total += t.size;
        std::cout << "  " << std::setw(50) << std::left << t.name 
                  << " " << std::setw(6) << t.dtype << " [";
        for (size_t i = 0; i < t.shape.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << t.shape[i];
        }
        std::cout << "]  " << (t.size / 1024) << " KB" << std::endl;
    }
    std::cout << "\nTotal: " << (total / 1024 / 1024) << " MB" << std::endl;
}

void HnfLoader::print_loaded_blocks() const {
    std::cout << "=== Loaded Blocks ===" << std::endl;
    size_t total = 0;
    bool any = false;
    for (int i = 0; i < HNF_BLOCK_COUNT; i++) {
        if (block_states_[i].loaded) {
            any = true;
            std::cout << "  " << std::setw(16) << std::left << block_name(i)
                      << " tensors=" << std::setw(4) << block_states_[i].tensor_count
                      << " VRAM=" << (block_states_[i].vram_bytes / 1024 / 1024) << " MB" << std::endl;
            total += block_states_[i].vram_bytes;
        }
    }
    if (!any) std::cout << "  (none)" << std::endl;
    else std::cout << "  Total VRAM: " << (total / 1024 / 1024) << " MB" << std::endl;
}

} // namespace helios
