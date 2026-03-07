// test_graph_builder.cpp
// Test del GraphBuilder con detección automática de arquitectura

#include "graph_builder.hpp"
#include "kernels.hpp"
#include <iostream>
#include <map>

void print_vram() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "VRAM: " << ((total_mem - free_mem) >> 20) << " MB used" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "============================================" << std::endl;
    std::cout << "HELIOS GraphBuilder Test" << std::endl;
    std::cout << "============================================" << std::endl;
    
    std::string hnf_path = "/home/andres/Escritorio/helios-engine/tests/helios_core_v10.hnf";
    if (argc > 1) hnf_path = argv[1];
    
    print_vram();
    
    // 1. Setup engine con scratch pequeño
    helios::EngineConfig ecfg;
    ecfg.scratch_pool.pool_size_bytes = 64 * 1024 * 1024;  // 64 MB
    helios::Engine engine(ecfg);
    helios::kernels::register_all_kernels(engine);
    
    std::cout << "\n>>> Loading HNF..." << std::endl;
    helios::HnfLoader loader;
    if (!loader.open(hnf_path)) {
        std::cerr << "FAIL: Cannot open " << hnf_path << std::endl;
        return 1;
    }
    
    if (!loader.load_block(helios::BLOCK_TEXT_MODEL, engine)) {
        std::cerr << "FAIL: Cannot load TEXT block" << std::endl;
        return 1;
    }
    
    const auto& config = loader.config();
    std::cout << "Model: " << config.arch() << std::endl;
    std::cout << "  layers=" << config.num_hidden_layers() 
              << " hidden=" << config.hidden_size()
              << " heads=" << config.num_attention_heads()
              << "/" << config.num_key_value_heads()
              << " head_dim=" << config.head_dim() << std::endl;
    print_vram();
    
    // 2. Detectar arquitectura
    std::cout << "\n>>> Detecting architecture..." << std::endl;
    helios::GraphBuilder gb;
    
    auto arch = gb.detect_architecture(engine, "text");
    
    std::cout << "Detected:" << std::endl;
    std::cout << "  prefix: " << arch.prefix << std::endl;
    std::cout << "  num_layers: " << arch.num_layers << std::endl;
    std::cout << "  has_qkv_bias: " << (arch.has_qkv_bias ? "yes" : "no") << std::endl;
    std::cout << "  has_fused_qkv: " << (arch.has_fused_qkv ? "yes" : "no") << std::endl;
    std::cout << "  has_gate (SwiGLU): " << (arch.has_gate ? "yes" : "no") << std::endl;
    std::cout << "  pre_attn_norm: " << arch.pre_attn_norm << std::endl;
    std::cout << "  post_attn_norm: " << arch.post_attn_norm << std::endl;
    std::cout << "  norm_has_bias: " << (arch.norm_has_bias ? "yes" : "no") << std::endl;
    std::cout << "  mlp: " << arch.mlp_gate_name << "/" << arch.mlp_up_name << "/" << arch.mlp_down_name << std::endl;
    std::cout << "  embedding: " << arch.embedding_name << std::endl;
    std::cout << "  final_norm: " << arch.final_norm_name << std::endl;
    std::cout << "  lm_head: " << (arch.lm_head_name.empty() ? "(tied)" : arch.lm_head_name) << std::endl;
    
    // 3. Validar pesos
    std::cout << "\n>>> Validating weights..." << std::endl;
    std::string err = gb.validate_weights(engine, config, arch);
    if (!err.empty()) {
        std::cerr << "WARN: " << err << std::endl;
    } else {
        std::cout << "All weights present ✓" << std::endl;
    }
    
    // 4. Allocate scratch
    std::cout << "\n>>> Allocating scratch..." << std::endl;
    gb.allocate_scratch(engine, config, arch, 1, 1);
    gb.print_scratch_info(engine);
    print_vram();
    
    // 5. Build forward (solo construir, no ejecutar)
    std::cout << "\n>>> Building forward pass..." << std::endl;
    
    // Crear input token
    int32_t token_id = 1;
    void* d_tokens;
    cudaMalloc(&d_tokens, sizeof(int32_t));
    cudaMemcpy(d_tokens, &token_id, sizeof(int32_t), cudaMemcpyHostToDevice);
    
    helios::TensorInfo tok;
    tok.ptr = d_tokens;
    tok.shape = {1, 1};
    tok.dtype = helios::dtype::INT32();
    tok.size_bytes = sizeof(int32_t);
    tok.owns_memory = false;
    engine.tensors().register_tensor("input_tokens", tok);
    
    auto cb = gb.build_forward(engine, config, arch, "input_tokens", 1, 1, 0);
    std::cout << "CommandBuffer: " << cb.size() << " commands" << std::endl;
    
    // Contar por tipo
    std::map<std::string, int> counts;
    for (size_t i = 0; i < cb.size(); i++) {
        auto* info = helios::OpTypeRegistry::instance().get(cb[i].op);
        if (info) counts[info->name]++;
    }
    std::cout << "By op type:" << std::endl;
    for (const auto& [name, count] : counts) {
        std::cout << "  " << name << ": " << count << std::endl;
    }
    
    // 6. Ejecutar con debug
    std::cout << "\n>>> Executing forward pass..." << std::endl;
    try {
        // Ejecutar comando por comando para encontrar el fallo exacto
        for (size_t i = 0; i < cb.size(); i++) {
            const auto& cmd = cb[i];
            auto* info = helios::OpTypeRegistry::instance().get(cmd.op);
            std::string opname = info ? info->name : "unknown";
            
            try {
                engine.execute_command(cmd);
                cudaError_t err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    std::cerr << "FAIL at [" << i << "] " << opname 
                              << " → " << cmd.output << " ← ";
                    for (size_t j = 0; j < cmd.inputs.size(); j++) {
                        if (j) std::cerr << ", ";
                        std::cerr << cmd.inputs[j];
                    }
                    std::cerr << std::endl;
                    std::cerr << "CUDA: " << cudaGetErrorString(err) << std::endl;
                    break;
                }
                
                // Progress cada 50
                if (i < 10 || i % 100 == 0) {
                    std::cout << "  [" << i << "] " << opname << " OK" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "FAIL at [" << i << "] " << opname 
                          << " → " << cmd.output << " ← ";
                for (size_t j = 0; j < cmd.inputs.size(); j++) {
                    if (j) std::cerr << ", ";
                    std::cerr << cmd.inputs[j];
                }
                std::cerr << std::endl;
                std::cerr << "Exception: " << e.what() << std::endl;
                break;
            }
        }
        std::cout << "Forward pass: OK ✓" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
    }
    
    // 7. Check logits
    auto* logits = gb.get_logits(engine);
    if (logits && logits->ptr) {
        std::cout << "\nLogits shape: [";
        for (size_t i = 0; i < logits->shape.size(); i++) {
            if (i) std::cout << ", ";
            std::cout << logits->shape[i];
        }
        std::cout << "]" << std::endl;
        
        // Print first few values
        std::vector<uint16_t> host(8);
        cudaMemcpy(host.data(), logits->ptr, 16, cudaMemcpyDeviceToHost);
        std::cout << "First 8 values (fp16 bits): ";
        for (int i = 0; i < 8; i++) {
            std::cout << std::hex << host[i] << " ";
        }
        std::cout << std::dec << std::endl;
    }
    
    // Cleanup
    cudaFree(d_tokens);
    gb.free_scratch(engine);
    loader.unload_block(helios::BLOCK_TEXT_MODEL, engine);
    
    std::cout << "\n============================================" << std::endl;
    std::cout << "TEST COMPLETE" << std::endl;
    std::cout << "============================================" << std::endl;
    
    return 0;
}
