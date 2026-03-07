// test_smoke.cpp
// ============================================================================
// HELIOS ENGINE v9.1 — SMOKE TEST PROGRESIVO
// ============================================================================
// Ejecutar con: ./test_smoke <ruta_al.hnf> [bloque]
//
// Fases (para si falla → sabes exactamente dónde):
//   FASE 1: Metadatos   — open, config, detect_architecture
//   FASE 2: Pesos       — load_block, validate_weights, inspect shapes
//   FASE 3: Forward     — allocate_scratch, build_forward, execute (1 token)
//   FASE 4: Generación  — loop autoregresivo con KV cache (10 tokens)
//
// Uso:
//   ./test_smoke /ruta/helios_core.hnf                → TEXT (default)
//   ./test_smoke /ruta/helios_core.hnf text_model     → TEXT
//   ./test_smoke /ruta/helios_core.hnf cortex         → CORTEX
//   ./test_smoke /ruta/helios_core.hnf code_exec      → CODE
//

#include "hnf_loader.hpp"
#include "graph_builder.hpp"
#include "sampler.hpp"
#include "kv_cache.hpp"
#include "kernels.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <chrono>

using namespace helios;

// ============================================================================
// HELPERS
// ============================================================================

void print_vram() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used = total_mem - free_mem;
    std::cout << "  VRAM: " << (used / 1024 / 1024) << " MB usado / "
              << (total_mem / 1024 / 1024) << " MB total ("
              << (free_mem / 1024 / 1024) << " MB libre)" << std::endl;
}

void print_separator(const std::string& title) {
    std::cout << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << "============================================================" << std::endl;
}

void print_ok(const std::string& msg) {
    std::cout << "  ✓ " << msg << std::endl;
}

void print_fail(const std::string& msg) {
    std::cerr << "  ✗ " << msg << std::endl;
}

void print_info(const std::string& msg) {
    std::cout << "  · " << msg << std::endl;
}

// Comprobar tensor GPU por NaN/Inf/valores extremos
struct TensorStats {
    float min_val, max_val, mean_val;
    int nan_count, inf_count;
    bool valid;
};

TensorStats check_tensor_health(TensorInfo* t) {
    TensorStats stats{0, 0, 0, 0, 0, false};
    if (!t || !t->ptr) return stats;
    
    // Solo FP16 por ahora
    size_t numel = 1;
    for (auto d : t->shape) numel *= d;
    if (numel == 0) return stats;
    
    // Copiar a CPU
    std::vector<half> host(numel);
    cudaMemcpy(host.data(), t->ptr, numel * sizeof(half), cudaMemcpyDeviceToHost);
    
    float sum = 0;
    stats.min_val = 1e10f;
    stats.max_val = -1e10f;
    
    for (size_t i = 0; i < numel; i++) {
        float v = __half2float(host[i]);
        if (std::isnan(v)) { stats.nan_count++; continue; }
        if (std::isinf(v)) { stats.inf_count++; continue; }
        sum += v;
        if (v < stats.min_val) stats.min_val = v;
        if (v > stats.max_val) stats.max_val = v;
    }
    
    size_t valid_count = numel - stats.nan_count - stats.inf_count;
    stats.mean_val = (valid_count > 0) ? (sum / valid_count) : 0;
    stats.valid = (stats.nan_count == 0 && stats.inf_count == 0);
    return stats;
}

std::string shape_str(const std::vector<uint32_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) s += ", ";
        s += std::to_string(shape[i]);
    }
    return s + "]";
}

// Map block name string to BlockID
BlockID resolve_block(const std::string& name) {
    if (name == "text_model" || name == "text") return BLOCK_TEXT_MODEL;
    if (name == "cortex") return BLOCK_CORTEX;
    if (name == "code_exec" || name == "code") return BLOCK_CODE_EXEC;
    if (name == "vision") return BLOCK_VISION;
    return BLOCK_TEXT_MODEL;
}

// Map BlockID to the prefix used in tensor names
std::string block_to_prefix(BlockID id) {
    switch (id) {
        case BLOCK_TEXT_MODEL: return "text";
        case BLOCK_CORTEX: return "cortex";
        case BLOCK_CODE_EXEC: return "code";
        case BLOCK_VISION: return "vision";
        default: return "text";
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    // --- Arguments ---
    if (argc < 2) {
        std::cout << "Uso: " << argv[0] << " <ruta.hnf> [bloque]" << std::endl;
        std::cout << "  bloque: text_model (default), cortex, code_exec, vision" << std::endl;
        return 1;
    }
    
    const char* hnf_path = argv[1];
    std::string block_name_str = (argc > 2) ? argv[2] : "text_model";
    BlockID target_block = resolve_block(block_name_str);
    std::string prefix = block_to_prefix(target_block);
    
    std::cout << std::endl;
    std::cout << "╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║          HELIOS ENGINE v9.1 — SMOKE TEST               ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;
    std::cout << "  Archivo: " << hnf_path << std::endl;
    std::cout << "  Bloque:  " << block_name_str << " (prefix: " << prefix << ")" << std::endl;
    print_vram();
    
    bool phase1_ok = false;
    bool phase2_ok = false;
    bool phase3_ok = false;
    bool phase4_ok = false;
    
    // ==================================================================
    // FASE 1: METADATOS
    // ==================================================================
    print_separator("FASE 1: METADATOS (sin GPU)");
    
    HnfLoader loader;
    
    // 1a. Open
    std::cout << std::endl << "  [1a] Abriendo HNF..." << std::endl;
    if (!loader.open(hnf_path)) {
        print_fail("No se pudo abrir el HNF");
        return 1;
    }
    print_ok("HNF abierto correctamente");
    
    // 1b. Print info
    std::cout << std::endl << "  [1b] Información del archivo:" << std::endl;
    loader.print_info();
    
    // 1c. Check config for target block
    const ModelConfig& config = loader.config_for_block(target_block);
    // If block config is empty, use default config
    const ModelConfig& effective_config = config.has("hidden_size") ? config : loader.config();
    
    std::cout << std::endl << "  [1c] Config efectiva para '" << block_name_str << "':" << std::endl;
    if (effective_config.has("arch"))
        print_info("arch: " + effective_config.arch());
    if (effective_config.has("hidden_size"))
        print_info("hidden_size: " + std::to_string(effective_config.hidden_size()));
    if (effective_config.has("num_hidden_layers"))
        print_info("layers: " + std::to_string(effective_config.num_hidden_layers()));
    if (effective_config.has("num_attention_heads"))
        print_info("heads: " + std::to_string(effective_config.num_attention_heads()) +
                   " (KV: " + std::to_string(effective_config.num_key_value_heads()) + ")");
    if (effective_config.has("vocab_size"))
        print_info("vocab: " + std::to_string(effective_config.vocab_size()));
    if (effective_config.has("intermediate_size"))
        print_info("intermediate: " + std::to_string(effective_config.intermediate_size()));
    if (effective_config.has("mlp_type"))
        print_info("mlp_type: " + effective_config.get<std::string>("mlp_type", "?"));
    if (effective_config.has("mlp_activation"))
        print_info("activation: " + effective_config.get<std::string>("mlp_activation", "?"));
    if (effective_config.has("norm_type"))
        print_info("norm: " + effective_config.norm_type() + " (eps=" + 
                   std::to_string(effective_config.rms_norm_eps()) + ")");
    if (effective_config.has("rope_type"))
        print_info("rope: " + effective_config.get<std::string>("rope_type", "default") + 
                   " (theta=" + std::to_string(effective_config.rope_theta()) + ")");
    if (effective_config.has("attention_bias"))
        print_info("attention_bias: " + std::string(effective_config.get<bool>("attention_bias", false) ? "true" : "false"));
    
    // 1d. Check tensors for block
    auto block_tensors = loader.tensors_for_block(target_block);
    if (block_tensors.empty()) {
        print_fail("No hay tensores para bloque '" + block_name_str + "'");
        return 1;
    }
    print_ok("Tensores en bloque: " + std::to_string(block_tensors.size()));
    
    // Show first 5
    int shown = 0;
    for (const auto* t : block_tensors) {
        if (shown++ >= 5) { print_info("... (" + std::to_string(block_tensors.size() - 5) + " más)"); break; }
        print_info(t->name + " " + t->dtype + " " + shape_str(t->shape) + 
                   " (" + std::to_string(t->size / 1024) + " KB)");
    }
    
    // 1e. Tokenizer check
    std::cout << std::endl << "  [1e] Tokenizer:" << std::endl;
    auto domains = loader.tokenizer_domains();
    if (domains.empty()) {
        print_info("No hay tokenizer en HNF (se necesitará externo)");
    } else {
        for (const auto& d : domains) {
            auto* tok = loader.tokenizer(d);
            if (tok) print_ok("Dominio '" + d + "': vocab=" + std::to_string(tok->vocab_size()));
            else print_fail("Dominio '" + d + "': fallo al cargar");
        }
    }
    
    // Validation
    if (!effective_config.has("hidden_size") || effective_config.hidden_size() == 0) {
        print_fail("Config incompleta — hidden_size es 0 o no existe");
        print_info("Esto probablemente significa que el HNF no tiene execution hints válidos.");
        print_info("Verifica que el conversor escribió los binary hints correctamente.");
        return 1;
    }
    
    phase1_ok = true;
    print_ok("FASE 1 COMPLETADA");
    
    // ==================================================================
    // FASE 2: CARGA DE PESOS
    // ==================================================================
    print_separator("FASE 2: CARGA DE PESOS A GPU");
    
    // 2a. Create engine
    EngineConfig eng_config;
    eng_config.scratch_pool.pool_size_bytes = 256 * 1024 * 1024;  // 256 MB
    eng_config.check_errors = true;
    eng_config.enable_profiling = true;
    
    Engine engine(eng_config);
    kernels::register_all_kernels(engine);
    print_ok("Engine creado, kernels registrados");
    print_vram();
    
    // 2b. Load block
    std::cout << std::endl << "  [2b] Cargando bloque '" << block_name_str << "'..." << std::endl;
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    if (!loader.load_block(target_block, engine)) {
        print_fail("No se pudo cargar el bloque");
        print_info("¿Hay suficiente VRAM?");
        print_vram();
        return 1;
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    print_ok("Bloque cargado en " + std::to_string((int)load_ms) + " ms");
    print_ok("Tensores en registry: " + std::to_string(engine.tensors().count()));
    print_vram();
    
    // 2c. Detect architecture
    std::cout << std::endl << "  [2c] Detectando arquitectura..." << std::endl;
    
    GraphBuilder builder;
    ArchDescriptor arch;
    
    try {
        arch = builder.detect_architecture(engine, prefix, effective_config);
        print_ok("Arquitectura detectada:");
        print_info("prefix: " + arch.prefix);
        print_info("layers: " + std::to_string(arch.num_layers));
        print_info("fused_qkv: " + std::string(arch.has_fused_qkv ? "SÍ" : "no"));
        print_info("qkv_bias: " + std::string(arch.has_qkv_bias ? "SÍ" : "no"));
        print_info("gate: " + std::string(arch.has_gate ? "SÍ" : "no"));
        print_info("fused_gate_up: " + std::string(arch.has_fused_gate_up ? "SÍ" : "no"));
        print_info("norm_has_bias: " + std::string(arch.norm_has_bias ? "SÍ (LayerNorm)" : "no (RMSNorm)"));
        print_info("pre_norm: " + arch.pre_attn_norm);
        print_info("post_norm: " + arch.post_attn_norm);
        print_info("mlp_up: " + arch.mlp_up_name);
        print_info("mlp_down: " + arch.mlp_down_name);
        print_info("mlp_gate: " + (arch.mlp_gate_name.empty() ? "(none)" : arch.mlp_gate_name));
        print_info("embedding: " + arch.embedding_name);
        print_info("final_norm: " + arch.final_norm_name);
        print_info("final_norm_bias: " + (arch.final_norm_bias.empty() ? "(none)" : arch.final_norm_bias));
        print_info("lm_head: " + (arch.lm_head_name.empty() ? "(tied)" : arch.lm_head_name));
        print_info("activation: " + std::to_string(static_cast<int>(arch.activation)));
        print_info("rope_type: " + std::to_string(static_cast<int>(arch.rope_type)));
        print_info("rope_theta: " + std::to_string(arch.rope_theta));
        print_info("compute_dtype: " + std::string(dtype_name(arch.compute_dtype)));
    } catch (const std::exception& e) {
        print_fail(std::string("detect_architecture falló: ") + e.what());
        return 1;
    }
    
    // 2d. Validate weights
    std::cout << std::endl << "  [2d] Validando pesos..." << std::endl;
    std::string missing = builder.validate_weights(engine, effective_config, arch);
    if (!missing.empty()) {
        print_fail("Pesos faltantes: " + missing);
        print_info("Esto puede indicar que el conversor no generó todos los tensores necesarios,");
        print_info("o que los nombres en el HNF no coinciden con lo que espera detect_architecture.");
        return 1;
    }
    print_ok("Todos los pesos necesarios están presentes");
    
    // 2d+. Fuse weights (Q+K+V → QKV, gate+up → gate_up) for performance
    std::cout << std::endl << "  [2d+] Fusionando pesos para rendimiento..." << std::endl;
    builder.fuse_weights(engine, arch, effective_config);
    
    // 2e. Spot-check: verify some key tensor shapes make sense
    std::cout << std::endl << "  [2e] Verificando shapes clave..." << std::endl;
    {
        auto* emb = engine.tensors().get(prefix + "." + arch.embedding_name);
        if (emb) {
            print_info("Embedding: " + shape_str(emb->shape) + " (" + std::to_string(emb->size_bytes / 1024 / 1024) + " MB)");
            if (emb->shape.size() >= 2 && emb->shape[0] == effective_config.vocab_size()) {
                print_ok("Embedding shape coincide con vocab_size=" + std::to_string(effective_config.vocab_size()));
            }
        }
        
        // Check a layer 0 weight
        auto* q = engine.tensors().get(prefix + ".layer0.attn." + 
                                        (arch.has_fused_qkv ? "qkv_proj" : "q_proj") + ".weight");
        if (q) {
            print_info("Layer0 " + std::string(arch.has_fused_qkv ? "QKV" : "Q") + 
                       " proj: " + shape_str(q->shape));
        }
    }
    
    phase2_ok = true;
    print_ok("FASE 2 COMPLETADA");
    
    // ==================================================================
    // FASE 3: FORWARD PASS (un prompt corto)
    // ==================================================================
    print_separator("FASE 3: FORWARD PASS (prefill)");
    
    // 3a. Allocate scratch
    std::cout << std::endl << "  [3a] Allocating scratch tensors..." << std::endl;
    
    uint32_t max_batch = 1;
    uint32_t max_seq = 64;  // Corto para test
    
    try {
        builder.allocate_scratch(engine, effective_config, arch, max_batch, max_seq);
        print_ok("Scratch allocated (batch=" + std::to_string(max_batch) + 
                 ", seq=" + std::to_string(max_seq) + ")");
        builder.print_scratch_info(engine);
        print_vram();
    } catch (const std::exception& e) {
        print_fail(std::string("allocate_scratch falló: ") + e.what());
        return 1;
    }
    
    // 3b. Prepare input tokens
    // Use simple token IDs — if tokenizer exists, encode "Hello"; otherwise use raw IDs
    std::vector<int32_t> input_tokens;
    
    // Select tokenizer matching the block domain
    // prefix "text" → domain "text", "cortex" → "cortex", "code" → "code"
    std::string tok_domain = prefix;
    auto* tok = loader.tokenizer(tok_domain);
    if (!tok) {
        // Fallback to text if domain-specific tokenizer unavailable
        print_info("Tokenizer para '" + tok_domain + "' no disponible, usando 'text' como fallback");
        tok = loader.tokenizer("text");
        tok_domain = "text";
    }
    if (tok) {
        std::string test_prompt = "Hello, world!";
        input_tokens = tok->encode(test_prompt);
        print_ok("Tokenized '" + test_prompt + "' → " + std::to_string(input_tokens.size()) + " tokens");
        std::string token_str = "[";
        for (size_t i = 0; i < input_tokens.size() && i < 10; i++) {
            if (i > 0) token_str += ", ";
            token_str += std::to_string(input_tokens[i]);
        }
        if (input_tokens.size() > 10) token_str += ", ...";
        token_str += "]";
        print_info("Token IDs: " + token_str);
    } else {
        // Fallback: BOS + some common token IDs
        input_tokens = {1, 9707, 29892, 3186, 29991};  // Approximate "Hello, world!"
        print_info("Sin tokenizer — usando IDs fijos: [1, 9707, 29892, 3186, 29991]");
    }
    
    uint32_t seq_len = static_cast<uint32_t>(input_tokens.size());
    if (seq_len > max_seq) {
        print_fail("Input demasiado largo (" + std::to_string(seq_len) + " > " + std::to_string(max_seq) + ")");
        return 1;
    }
    
    // Upload input tokens to GPU
    std::string tokens_tensor_name = "_input_tokens";
    {
        void* d_tokens = nullptr;
        cudaMalloc(&d_tokens, seq_len * sizeof(int32_t));
        cudaMemcpy(d_tokens, input_tokens.data(), seq_len * sizeof(int32_t), cudaMemcpyHostToDevice);
        
        TensorInfo tok_info;
        tok_info.ptr = d_tokens;
        tok_info.shape = {1, seq_len};
        tok_info.dtype = dtype::INT32();
        tok_info.size_bytes = seq_len * sizeof(int32_t);
        tok_info.owns_memory = true;
        engine.tensors().register_tensor(tokens_tensor_name, tok_info);
        print_ok("Input tokens en GPU: " + shape_str(tok_info.shape));
    }
    
    // 3c. Build forward graph
    std::cout << std::endl << "  [3c] Construyendo grafo de forward..." << std::endl;
    
    CommandBuffer cb;
    try {
        cb = builder.build_forward(engine, effective_config, arch,
                                   tokens_tensor_name, max_batch, seq_len);
        print_ok("Grafo construido: " + std::to_string(cb.size()) + " comandos");
        
        // Print first few commands
        for (size_t i = 0; i < std::min((size_t)8, cb.size()); i++) {
            const auto& cmd = cb.commands()[i];
            std::string cmd_str = "  cmd[" + std::to_string(i) + "] op=" + std::to_string(cmd.op);
            if (!cmd.output.empty()) cmd_str += " out=" + cmd.output;
            if (!cmd.inputs.empty()) cmd_str += " in=" + cmd.inputs[0];
            print_info(cmd_str);
        }
        if (cb.size() > 8) print_info("... (" + std::to_string(cb.size() - 8) + " más)");
        
    } catch (const std::exception& e) {
        print_fail(std::string("build_forward falló: ") + e.what());
        return 1;
    }
    
    // 3d. Execute forward
    std::cout << std::endl << "  [3d] Ejecutando forward pass..." << std::endl;
    
    t_start = std::chrono::high_resolution_clock::now();
    
    try {
        engine.execute(cb);
        engine.sync();
        
        t_end = std::chrono::high_resolution_clock::now();
        double fwd_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        
        print_ok("Forward ejecutado en " + std::to_string((int)fwd_ms) + " ms");
    } catch (const std::exception& e) {
        print_fail(std::string("execute falló: ") + e.what());
        
        // Check CUDA error
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            print_fail(std::string("CUDA error: ") + cudaGetErrorString(err));
        }
        return 1;
    }
    
    // 3e. Check output health
    std::cout << std::endl << "  [3e] Verificando logits..." << std::endl;
    
    TensorInfo* logits = builder.get_logits(engine);
    if (!logits) {
        print_fail("No se encontró tensor de logits");
        return 1;
    }
    
    print_info("Logits shape: " + shape_str(logits->shape));
    
    auto stats = check_tensor_health(logits);
    if (stats.nan_count > 0) print_fail("NaN detectados: " + std::to_string(stats.nan_count));
    if (stats.inf_count > 0) print_fail("Inf detectados: " + std::to_string(stats.inf_count));
    
    print_info("min=" + std::to_string(stats.min_val) + 
               " max=" + std::to_string(stats.max_val) + 
               " mean=" + std::to_string(stats.mean_val));
    
    if (!stats.valid) {
        print_fail("Logits contienen NaN/Inf — algo está mal en el forward");
        print_info("Posibles causas:");
        print_info("  - DType mismatch (modelo BF16 con kernels FP16)");
        print_info("  - RMSNorm epsilon demasiado bajo");
        print_info("  - Pesos corruptos o mal cargados");
        return 1;
    }
    
    // 3f. Sample first token
    std::cout << std::endl << "  [3f] Sampling..." << std::endl;
    
    // Get logits for last position
    uint32_t V = effective_config.vocab_size();
    const half* last_logits = static_cast<const half*>(logits->ptr) + (seq_len - 1) * V;
    
    Sampler sampler;
    int32_t greedy_token = sampler.sample_greedy(last_logits, V);
    
    std::string token_text = "(no tokenizer)";
    if (tok) {
        token_text = tok->decode({greedy_token});
    }
    print_ok("Greedy next token: " + std::to_string(greedy_token) + " → '" + token_text + "'");
    
    // Top-5
    {
        std::vector<half> host_logits(V);
        cudaMemcpy(host_logits.data(), last_logits, V * sizeof(half), cudaMemcpyDeviceToHost);
        
        std::vector<std::pair<float, int>> scores(V);
        for (uint32_t i = 0; i < V; i++) {
            scores[i] = {__half2float(host_logits[i]), (int)i};
        }
        std::partial_sort(scores.begin(), scores.begin() + 5, scores.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        
        print_info("Top-5 logits:");
        for (int i = 0; i < 5; i++) {
            std::string t = (tok) ? tok->decode({scores[i].second}) : "?";
            print_info("  #" + std::to_string(i+1) + " token=" + std::to_string(scores[i].second) + 
                       " logit=" + std::to_string(scores[i].first) + " → '" + t + "'");
        }
    }
    
    phase3_ok = true;
    print_ok("FASE 3 COMPLETADA — EL ENGINE PRODUCE LOGITS VÁLIDOS");
    
    // ==================================================================
    // FASE 4: GENERACIÓN AUTOREGRESIVA (10 tokens con KV cache)
    // ==================================================================
    print_separator("FASE 4: GENERACIÓN AUTOREGRESIVA");
    
    // 4a. Allocate KV cache
    std::cout << std::endl << "  [4a] Allocating KV cache..." << std::endl;
    
    uint32_t max_cache_len = 256;
    std::string kv_prefix = "_kv";
    
    try {
        KVCacheConfig kv_config;
        kv_config.max_seq_len = max_cache_len;
        kv_config.num_layers = arch.num_layers;
        kv_config.num_kv_heads = effective_config.num_key_value_heads();
        kv_config.head_dim = effective_config.head_dim();
        
        DTypeID cache_dtype = arch.compute_dtype ? arch.compute_dtype : dtype::FP16();
        
        // Allocate KV cache tensors in engine registry
        for (uint32_t layer = 0; layer < arch.num_layers; layer++) {
            std::string k_name = kv_prefix + ".layer" + std::to_string(layer) + ".k";
            std::string v_name = kv_prefix + ".layer" + std::to_string(layer) + ".v";
            
            std::vector<uint32_t> cache_shape = {
                1, max_cache_len, kv_config.num_kv_heads, kv_config.head_dim
            };
            
            engine.tensors().allocate_and_register(k_name, cache_shape, cache_dtype);
            engine.tensors().allocate_and_register(v_name, cache_shape, cache_dtype);
        }
        
        print_ok("KV cache allocado: " + std::to_string(arch.num_layers) + " layers × " +
                 std::to_string(max_cache_len) + " positions");
        print_vram();
    } catch (const std::exception& e) {
        print_fail(std::string("KV cache allocation falló: ") + e.what());
        print_info("Continuando sin fase 4...");
        goto summary;
    }
    
    // 4b. Prefill (process initial prompt into KV cache)
    {
        std::cout << std::endl << "  [4b] Prefill con KV cache..." << std::endl;
        
        // Re-allocate scratch for cache-compatible sizes
        builder.free_scratch(engine);
        builder.allocate_scratch(engine, effective_config, arch, 1, max_cache_len);
        
        // Re-upload tokens
        {
            auto* tok_tensor = engine.tensors().get(tokens_tensor_name);
            if (tok_tensor) {
                cudaMemcpy(tok_tensor->ptr, input_tokens.data(), 
                          seq_len * sizeof(int32_t), cudaMemcpyHostToDevice);
            }
        }
        
        KVCacheParams cache{kv_prefix, 0, max_cache_len};
        
        try {
            CommandBuffer prefill_cb = builder.build_forward(
                engine, effective_config, arch, tokens_tensor_name,
                1, seq_len, 0, &cache);
            
            engine.execute(prefill_cb);
            engine.sync();
            print_ok("Prefill completado (" + std::to_string(seq_len) + " tokens → cache)");
        } catch (const std::exception& e) {
            print_fail(std::string("Prefill falló: ") + e.what());
            goto summary;
        }
    }
    
    // 4c. Autoregressive generation
    {
        std::cout << std::endl << "  [4c] Generando tokens..." << std::endl;
        
        int gen_tokens = 30;
        uint32_t cache_pos = seq_len;  // Prefill wrote [0, seq_len)
        std::vector<int32_t> generated;
        
        // Persistent sampler — accumulates context for repetition penalty
        Sampler samp;
        samp.add_context(input_tokens);  // Feed prompt as initial context
        
        // Sampling config: greedy with repetition penalty
        SamplingConfig gen_config = SamplingConfig::greedy();
        
        // Get last logit from prefill and sample first token
        {
            TensorInfo* lg = builder.get_logits(engine);
            uint32_t V2 = effective_config.vocab_size();
            const half* last_lg = static_cast<const half*>(lg->ptr) + (seq_len - 1) * V2;
            
            int32_t first_tok = samp.sample(last_lg, V2, gen_config);
            generated.push_back(first_tok);
            samp.add_context(first_tok);
        }
        
        auto gen_start = std::chrono::high_resolution_clock::now();
        
        for (int step = 1; step < gen_tokens; step++) {
            // Upload single token
            int32_t current_token = generated.back();
            {
                auto* tok_tensor = engine.tensors().get(tokens_tensor_name);
                cudaMemcpy(tok_tensor->ptr, &current_token, sizeof(int32_t), cudaMemcpyHostToDevice);
                tok_tensor->shape = {1, 1};
            }
            
            // Build forward with cache
            KVCacheParams cache{kv_prefix, cache_pos, max_cache_len};
            
            try {
                CommandBuffer step_cb = builder.build_forward(
                    engine, effective_config, arch, tokens_tensor_name,
                    1, 1, cache_pos, &cache);
                
                engine.execute(step_cb);
                engine.sync();
            } catch (const std::exception& e) {
                print_fail(std::string("Generation step ") + std::to_string(step) + " falló: " + e.what());
                break;
            }
            
            // Sample with context-aware penalties
            TensorInfo* lg = builder.get_logits(engine);
            uint32_t V2 = effective_config.vocab_size();
            
            int32_t next_tok = samp.sample(static_cast<const half*>(lg->ptr), V2,
                                           gen_config);
            generated.push_back(next_tok);
            samp.add_context(next_tok);
            cache_pos++;
            
            // Check for EOS (use actual tokenizer EOS, not hardcoded)
            int32_t eos_id = -1;
            if (tok && tok->eos_token_id().has_value()) {
                eos_id = tok->eos_token_id().value();
            }
            if (next_tok == eos_id) break;
        }
        
        auto gen_end = std::chrono::high_resolution_clock::now();
        double gen_ms = std::chrono::duration<double, std::milli>(gen_end - gen_start).count();
        double tps = (generated.size() > 1) ? ((generated.size() - 1) * 1000.0 / gen_ms) : 0;
        
        print_ok("Generados " + std::to_string(generated.size()) + " tokens en " +
                 std::to_string((int)gen_ms) + " ms (" + std::to_string((int)tps) + " tok/s)");
        
        // Print generated tokens
        std::string id_str = "";
        for (size_t i = 0; i < generated.size(); i++) {
            if (i > 0) id_str += ", ";
            id_str += std::to_string(generated[i]);
        }
        print_info("IDs: [" + id_str + "]");
        
        // Decode if tokenizer available
        if (tok) {
            std::string decoded = tok->decode(generated);
            print_ok("Texto generado: \"" + decoded + "\"");
        }
        
        phase4_ok = true;
        print_ok("FASE 4 COMPLETADA — GENERACIÓN AUTOREGRESIVA FUNCIONA");
    }
    
    // ==================================================================
    // RESUMEN
    // ==================================================================
summary:
    print_separator("RESUMEN");
    
    std::cout << std::endl;
    std::cout << "  Fase 1 (Metadatos):   " << (phase1_ok ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "  Fase 2 (Pesos GPU):   " << (phase2_ok ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "  Fase 3 (Forward):     " << (phase3_ok ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "  Fase 4 (Generación):  " << (phase4_ok ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << std::endl;
    
    if (phase4_ok) {
        std::cout << "  🟢 ENGINE OPERATIVO — Inferencia end-to-end funcional" << std::endl;
    } else if (phase3_ok) {
        std::cout << "  🟡 Forward funciona pero la generación autoregresiva tiene problemas" << std::endl;
        std::cout << "     Revisa KV cache allocation o el path cached" << std::endl;
    } else if (phase2_ok) {
        std::cout << "  🟠 Pesos cargados pero forward falla" << std::endl;
        std::cout << "     Revisa kernels, scratch allocation, o graph construction" << std::endl;
    } else if (phase1_ok) {
        std::cout << "  🔴 Metadatos OK pero no se pueden cargar pesos" << std::endl;
        std::cout << "     Revisa VRAM, offsets en manifest, o formato de tensores" << std::endl;
    }
    
    std::cout << std::endl;
    print_vram();
    
    // Profiling
    if (phase3_ok) {
        std::cout << std::endl;
        engine.print_profile_summary();
    }
    
    return phase3_ok ? 0 : 1;
}
