// test_real_hnf.cpp
// Inspecciona un archivo HNF real y prueba load/unload por bloque

#include "hnf_loader.hpp"
#include "kernels.hpp"
#include <iostream>
#include <map>

void print_vram_usage() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "VRAM: " << ((total_mem - free_mem) / 1024 / 1024) << " MB used / "
              << (total_mem / 1024 / 1024) << " MB total" << std::endl;
}

int main(int argc, char** argv) {
    const char* path = "/home/andres/Escritorio/helios-engine/tests/helios_core_v10.hnf";
    
    if (argc > 1) {
        path = argv[1];
    }
    
    std::cout << "============================================" << std::endl;
    std::cout << "HELIOS HNF Inspector + Block Loading Test" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "File: " << path << std::endl;
    std::cout << std::endl;
    
    print_vram_usage();
    std::cout << std::endl;
    
    // Create engine with minimal scratch pool (we're just loading weights, not running inference)
    helios::EngineConfig config;
    config.scratch_pool.pool_size_bytes = 64 * 1024 * 1024;  // 64 MB only
    helios::Engine engine(config);
    helios::kernels::register_all_kernels(engine);
    
    // Open HNF file
    helios::HnfLoader loader;
    
    std::cout << "Opening HNF file..." << std::endl;
    if (!loader.open(path)) {
        std::cerr << "ERROR: Failed to open HNF" << std::endl;
        return 1;
    }
    std::cout << "SUCCESS!" << std::endl;
    std::cout << std::endl;
    
    loader.print_info();
    
    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "TENSOR SUMMARY BY BLOCK" << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Count tensors by block
    std::map<std::string, int> block_counts;
    std::map<std::string, size_t> block_sizes;
    std::map<std::string, int> dtype_counts;
    
    for (const auto& t : loader.tensors()) {
        block_counts[t.block]++;
        block_sizes[t.block] += t.size;
        dtype_counts[t.dtype]++;
    }
    
    std::cout << "By block:" << std::endl;
    for (const auto& [block, count] : block_counts) {
        std::cout << "  " << block << ": " << count << " tensors, "
                  << (block_sizes[block] / 1024 / 1024) << " MB" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "By dtype:" << std::endl;
    for (const auto& [dtype, count] : dtype_counts) {
        std::cout << "  " << dtype << ": " << count << " tensors" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "BLOCK LOADING TEST" << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Test: Load only code_exec block (smallest: 1.4GB)
    std::cout << std::endl;
    std::cout << ">>> Loading CODE_EXEC block (DeepSeek 1.3B)..." << std::endl;
    print_vram_usage();
    
    if (loader.load_block(helios::BLOCK_CODE_EXEC, engine)) {
        std::cout << "SUCCESS!" << std::endl;
        loader.print_loaded_blocks();
        print_vram_usage();
        std::cout << "Tensors in registry: " << engine.tensors().count() << std::endl;
    } else {
        std::cout << "FAILED (might not have enough VRAM)" << std::endl;
    }
    
    // Test: Unload code_exec
    std::cout << std::endl;
    std::cout << ">>> Unloading CODE_EXEC block..." << std::endl;
    
    if (loader.unload_block(helios::BLOCK_CODE_EXEC, engine)) {
        std::cout << "SUCCESS!" << std::endl;
        loader.print_loaded_blocks();
        print_vram_usage();
        std::cout << "Tensors in registry: " << engine.tensors().count() << std::endl;
    }
    
    // Test: Load vision block (smaller: 303 MB)
    std::cout << std::endl;
    std::cout << ">>> Loading VISION block (CLIP)..." << std::endl;
    print_vram_usage();
    
    if (loader.load_block(helios::BLOCK_VISION, engine)) {
        std::cout << "SUCCESS!" << std::endl;
        loader.print_loaded_blocks();
        print_vram_usage();
        std::cout << "Tensors in registry: " << engine.tensors().count() << std::endl;
        
        // List first 5 vision tensors
        std::cout << "First 5 vision tensors:" << std::endl;
        int count = 0;
        for (const auto& name : engine.tensors().names()) {
            if (count++ >= 5) break;
            auto* t = engine.tensors().get(name);
            std::cout << "  " << name << " [";
            for (size_t i = 0; i < t->shape.size(); i++) {
                if (i > 0) std::cout << ",";
                std::cout << t->shape[i];
            }
            std::cout << "]" << std::endl;
        }
    } else {
        std::cout << "FAILED" << std::endl;
    }
    
    // Cleanup
    std::cout << std::endl;
    std::cout << ">>> Unloading all..." << std::endl;
    loader.unload_block(helios::BLOCK_VISION, engine);
    loader.print_loaded_blocks();
    print_vram_usage();
    
    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "TEST COMPLETE" << std::endl;
    std::cout << "============================================" << std::endl;
    
    return 0;
}
