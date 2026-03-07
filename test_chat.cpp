// test_chat.cpp
// ============================================================================
// HELIOS ENGINE v9.1 — CHAT TEST
// ============================================================================
// Interactive chat with proper templates per model.
// Clean output: just the conversation and tok/s stats.
//
// Usage:
//   ./test_chat <hnf_path> <block> "<prompt>" [max_tokens]
//   ./test_chat ../tests/helios_core.hnf text "What is the meaning of life?"
//   ./test_chat ../tests/helios_core.hnf cortex "Hello, how are you?"
//   ./test_chat ../tests/helios_core.hnf code_exec "def fibonacci(n):"
//

#include "hnf_loader.hpp"
#include "graph_builder.hpp"
#include "sampler.hpp"
#include "kv_cache.hpp"
#include "kernels.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <optional>
#include <algorithm>

using namespace helios;

// ============================================================================
// SPECIAL TOKEN ENCODER
// ============================================================================

static std::vector<int32_t> encode_with_special_tokens(
    const HTFTokenizer& tok,
    const std::string& text,
    const std::vector<std::pair<std::string, int32_t>>& specials
) {
    auto sorted = specials;
    std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) {
        return a.first.size() > b.first.size();
    });
    
    struct Segment { std::string text; bool is_special; int32_t special_id; };
    std::vector<Segment> segments;
    
    std::string remaining = text;
    while (!remaining.empty()) {
        size_t best_pos = std::string::npos;
        size_t best_len = 0;
        int32_t best_id = 0;
        
        for (auto& [stext, sid] : sorted) {
            size_t pos = remaining.find(stext);
            if (pos != std::string::npos && (pos < best_pos || (pos == best_pos && stext.size() > best_len))) {
                best_pos = pos;
                best_len = stext.size();
                best_id = sid;
            }
        }
        
        if (best_pos == std::string::npos) {
            if (!remaining.empty()) segments.push_back({remaining, false, 0});
            break;
        }
        
        if (best_pos > 0) segments.push_back({remaining.substr(0, best_pos), false, 0});
        segments.push_back({"", true, best_id});
        remaining = remaining.substr(best_pos + best_len);
    }
    
    std::vector<int32_t> ids;
    for (auto& seg : segments) {
        if (seg.is_special) {
            ids.push_back(seg.special_id);
        } else {
            auto seg_ids = tok.encode(seg.text);
            ids.insert(ids.end(), seg_ids.begin(), seg_ids.end());
        }
    }
    return ids;
}

// ============================================================================
// CHAT TEMPLATES
// ============================================================================

struct ChatMessage { std::string role; std::string content; };
struct ChatTemplate {
    std::string name;
    std::string (*format)(const std::vector<ChatMessage>& messages);
    std::vector<std::string> stop_strings;
};

static std::string format_chatml(const std::vector<ChatMessage>& msgs) {
    std::string out;
    for (auto& m : msgs) out += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
    out += "<|im_start|>assistant\n";
    return out;
}

static std::string format_phi(const std::vector<ChatMessage>& msgs) {
    std::string out;
    for (auto& m : msgs) {
        if (m.role == "system")         out += "<|system|>\n" + m.content + "<|end|>\n";
        else if (m.role == "user")      out += "<|user|>\n" + m.content + "<|end|>\n";
        else if (m.role == "assistant") out += "<|assistant|>\n" + m.content + "<|end|>\n";
    }
    out += "<|assistant|>\n";
    return out;
}

static std::string format_deepseek(const std::vector<ChatMessage>& msgs) {
    std::string out;
    for (auto& m : msgs) {
        if (m.role == "system")         out += m.content + "\n";
        else if (m.role == "user")      out += "### Instruction:\n" + m.content + "\n";
        else if (m.role == "assistant") out += "### Response:\n" + m.content + "\n";
    }
    out += "### Response:\n";
    return out;
}

static std::string format_raw(const std::vector<ChatMessage>& msgs) {
    std::string out;
    for (auto& m : msgs) out += m.content;
    return out;
}

static ChatTemplate TEMPLATES[] = {
    {"chatml",    format_chatml,    {"<|im_end|>"}},
    {"phi",       format_phi,       {"<|end|>", "<|endoftext|>"}},
    {"deepseek",  format_deepseek,  {"<|EOT|>"}},
    {"raw",       format_raw,       {}},
};

static ChatTemplate* find_template(const std::string& name) {
    for (auto& t : TEMPLATES) { if (t.name == name) return &t; }
    return nullptr;
}

static std::string detect_template(const std::string& arch, const std::string& prefix) {
    if (prefix == "code") return "raw";
    if (arch.find("qwen") != std::string::npos) return "chatml";
    if (arch.find("phi") != std::string::npos)   return "phi";
    return "raw";
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./test_chat <hnf_path> <block> \"<prompt>\" [max_tokens]" << std::endl;
        std::cerr << "  Blocks: text, cortex, code_exec" << std::endl;
        return 1;
    }
    
    std::string hnf_path = argv[1];
    std::string block_name = argv[2];
    std::string user_prompt = argv[3];
    int max_tokens = (argc >= 5) ? std::atoi(argv[4]) : 128;
    
    // ========================================================================
    // LOAD
    // ========================================================================
    
    HnfLoader loader;
    if (!loader.open(hnf_path)) {
        std::cerr << "ERROR: Cannot open " << hnf_path << std::endl;
        return 1;
    }
    
    // Resolve prefix from block name
    std::string prefix;
    if (block_name == "text_model") prefix = "text";
    else if (block_name == "cortex") prefix = "cortex";
    else if (block_name == "code_exec") prefix = "code";
    else prefix = block_name;
    
    auto resolve_block = [](const std::string& name) -> BlockID {
        if (name == "text_model" || name == "text") return BLOCK_TEXT_MODEL;
        if (name == "cortex") return BLOCK_CORTEX;
        if (name == "code_exec" || name == "code") return BLOCK_CODE_EXEC;
        if (name == "vision") return BLOCK_VISION;
        return BLOCK_TEXT_MODEL;
    };
    
    BlockID bid = resolve_block(block_name);
    ModelConfig mutable_config = loader.config_for_block(bid);
    if (!mutable_config.has("hidden_size")) {
        mutable_config = ModelConfig(loader.config());
    }
    
    // Fix: DeepSeek rope_scaling_factor=4.0 (converter stores 1.0)
    if (prefix == "code") {
        mutable_config.set("rope_scaling_factor", ConfigValue(4.0));
    }
    ModelConfig& effective_config = mutable_config;
    
    // Get tokenizer
    HTFTokenizer* tok = nullptr;
    if (loader.has_tokenizer(prefix)) {
        tok = loader.tokenizer(prefix);
    } else if (loader.has_tokenizer("text")) {
        tok = loader.tokenizer("text");
    }
    if (!tok) {
        std::cerr << "ERROR: No tokenizer for " << prefix << std::endl;
        return 1;
    }
    
    // Engine + kernels
    EngineConfig eng_config;
    eng_config.scratch_pool.pool_size_bytes = 256 * 1024 * 1024;
    // Enable profiling via environment variable: HELIOS_PROFILE=1
    if (std::getenv("HELIOS_PROFILE") && std::string(std::getenv("HELIOS_PROFILE")) == "1") {
        eng_config.enable_profiling = true;
    }
    Engine engine(eng_config);
    kernels::register_all_kernels(engine);
    
    // Load weights
    if (!loader.load_block(bid, engine)) {
        std::cerr << "ERROR: Failed to load block " << block_name << std::endl;
        return 1;
    }
    
    // Detect arch + fuse weights
    GraphBuilder builder;
    ArchDescriptor arch = builder.detect_architecture(engine, prefix, effective_config);
    builder.fuse_weights(engine, arch, effective_config);
    
    uint32_t V = effective_config.vocab_size();
    
    // ========================================================================
    // CHAT TEMPLATE
    // ========================================================================
    
    std::string arch_str = effective_config.has("arch") ? effective_config.arch() : "unknown";
    std::string tmpl_name = detect_template(arch_str, prefix);
    ChatTemplate* tmpl = find_template(tmpl_name);
    if (!tmpl) tmpl = find_template("raw");
    
    // Build prompt
    std::vector<ChatMessage> messages;
    std::string formatted;
    
    if (tmpl_name == "raw") {
        formatted = user_prompt;
    } else {
        messages.push_back({"system", "You are a helpful assistant."});
        messages.push_back({"user", user_prompt});
        formatted = tmpl->format(messages);
    }
    
    // Resolve stop token IDs
    std::vector<int32_t> stop_ids;
    for (auto& s : tmpl->stop_strings) {
        auto id = tok->token_to_id(s);
        if (id.has_value()) stop_ids.push_back(id.value());
    }
    if (tok->eos_token_id().has_value()) {
        int32_t eos = tok->eos_token_id().value();
        if (std::find(stop_ids.begin(), stop_ids.end(), eos) == stop_ids.end())
            stop_ids.push_back(eos);
    }
    
    // Tokenize
    std::vector<int32_t> input_tokens;
    if (tmpl_name != "raw") {
        std::vector<std::pair<std::string, int32_t>> specials;
        for (auto& s : std::vector<std::string>{
            "<|im_start|>", "<|im_end|>",
            "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
            "<|endoftext|>"
        }) {
            auto id = tok->token_to_id(s);
            if (id.has_value()) specials.push_back({s, id.value()});
        }
        for (auto& s : tmpl->stop_strings) {
            auto id = tok->token_to_id(s);
            if (id.has_value()) {
                bool found = false;
                for (auto& sp : specials) { if (sp.first == s) { found = true; break; } }
                if (!found) specials.push_back({s, id.value()});
            }
        }
        input_tokens = encode_with_special_tokens(*tok, formatted, specials);
    } else {
        input_tokens = tok->encode(formatted);
        if (tok->bos_token_id().has_value()) {
            input_tokens.insert(input_tokens.begin(), tok->bos_token_id().value());
        }
    }
    
    uint32_t seq_len = input_tokens.size();
    
    // Print header
    std::cout << "\n╔══════════════════════════════════════════╗" << std::endl;
    std::cout << "║  HELIOS CHAT — " << prefix << " (" << tmpl_name << ")" << std::endl;
    std::cout << "╚══════════════════════════════════════════╝\n" << std::endl;
    std::cout << "  Prompt tokens: " << seq_len << " | Max gen: " << max_tokens << std::endl;
    std::cout << "  Stop IDs: [";
    for (size_t i = 0; i < stop_ids.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << stop_ids[i];
    }
    std::cout << "]\n" << std::endl;
    
    if (tmpl_name == "raw") {
        std::cout << ">>> " << user_prompt << std::flush;
    } else {
        std::cout << "User: " << user_prompt << "\n" << std::endl;
        std::cout << "Assistant: " << std::flush;
    }
    
    // ========================================================================
    // ALLOCATE + PREFILL
    // ========================================================================
    
    uint32_t max_cache_len = seq_len + max_tokens + 16;
    
    builder.allocate_scratch(engine, effective_config, arch, 1, max_cache_len);
    
    // KV cache
    std::string kv_prefix = "_kv";
    DTypeID cache_dtype = arch.compute_dtype ? arch.compute_dtype : dtype::FP16();
    uint32_t kv_heads = effective_config.num_key_value_heads();
    uint32_t head_dim = effective_config.head_dim();
    
    for (uint32_t layer = 0; layer < arch.num_layers; layer++) {
        std::string k_name = kv_prefix + ".layer" + std::to_string(layer) + ".k";
        std::string v_name = kv_prefix + ".layer" + std::to_string(layer) + ".v";
        engine.tensors().allocate_and_register(k_name, {1, max_cache_len, kv_heads, head_dim}, cache_dtype);
        engine.tensors().allocate_and_register(v_name, {1, max_cache_len, kv_heads, head_dim}, cache_dtype);
    }
    
    // Upload tokens
    std::string tokens_name = "_input_tokens";
    engine.tensors().allocate_and_register(tokens_name, {1, (uint32_t)seq_len}, dtype::INT32());
    cudaMemcpy(engine.tensors().get(tokens_name)->ptr, 
               input_tokens.data(), seq_len * sizeof(int32_t), 
               cudaMemcpyHostToDevice);
    
    // Prefill (with cache)
    {
        KVCacheParams cache{kv_prefix, 0, max_cache_len};
        CommandBuffer cb = builder.build_forward(
            engine, effective_config, arch, tokens_name, 1, seq_len, 0, &cache);
        engine.execute(cb);
        engine.sync();
    }
    
    // ========================================================================
    // GENERATE
    // ========================================================================
    
    std::vector<int32_t> generated;
    Sampler samp;
    samp.add_context(input_tokens);
    
    SamplingConfig samp_config = SamplingConfig::greedy();
    samp_config.repetition_penalty = 1.15f;
    
    uint32_t cache_pos = seq_len;
    
    // First token from prefill logits
    {
        TensorInfo* lg = builder.get_logits(engine);
        const half* last_logits = static_cast<const half*>(lg->ptr) + (seq_len - 1) * V;
        
        int32_t first_tok = samp.sample(last_logits, V, samp_config);
        generated.push_back(first_tok);
        samp.add_context(first_tok);
        
        std::cout << tok->decode({first_tok}) << std::flush;
    }
    
    // Autoregressive loop
    auto gen_start = std::chrono::high_resolution_clock::now();
    bool hit_stop = false;
    
    builder.invalidate_decode_cache();
    
    for (int step = 1; step < max_tokens; step++) {
        int32_t current_token = generated.back();
        
        // Check stop
        if (std::find(stop_ids.begin(), stop_ids.end(), current_token) != stop_ids.end()) {
            hit_stop = true;
            break;
        }
        
        // Upload token
        {
            auto* t = engine.tensors().get(tokens_name);
            cudaMemcpy(t->ptr, &current_token, sizeof(int32_t), cudaMemcpyHostToDevice);
            t->shape = {1, 1};
        }
        
        // Forward (CB reuse — only updates cache_pos params, no full rebuild)
        KVCacheParams cache{kv_prefix, cache_pos, max_cache_len};
        const CommandBuffer& cb = builder.build_forward_reuse(
            engine, effective_config, arch, tokens_name, 1, 1, cache_pos, &cache);
        engine.execute(cb);
        engine.sync();
        
        // Sample
        TensorInfo* lg = builder.get_logits(engine);
        int32_t next_tok = samp.sample(static_cast<const half*>(lg->ptr), V, samp_config);
        
        generated.push_back(next_tok);
        samp.add_context(next_tok);
        cache_pos++;
        
        // Streaming output
        std::cout << tok->decode({next_tok}) << std::flush;
    }
    
    auto gen_end = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(gen_end - gen_start).count();
    int decode_tokens = (int)generated.size() - 1;
    double tps = (decode_tokens > 0) ? (decode_tokens * 1000.0 / gen_ms) : 0;
    
    // ========================================================================
    // STATS
    // ========================================================================
    
    std::cout << "\n\n────────────────────────────────────────────" << std::endl;
    std::cout << "  " << generated.size() << " tokens"
              << " | " << std::fixed << std::setprecision(1) << tps << " tok/s"
              << " | " << (int)gen_ms << "ms"
              << " | " << (hit_stop ? "stopped" : "max_len")
              << std::endl;
    
    engine.print_profile_summary();
    
    return 0;
}
