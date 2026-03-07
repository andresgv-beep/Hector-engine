// tests/test_hnf_loader.cpp
// ============================================================================
// HELIOS ENGINE - HNF Loader Test
// ============================================================================

#include "hnf_loader.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>

using namespace helios;

// ============================================================================
// CREATE MOCK HNF FILE
// ============================================================================

std::string create_mock_hnf(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return "Cannot create file";
    }
    
    // ========================================
    // HEADER (64 bytes)
    // ========================================
    
    HnfHeader header;
    memset(&header, 0, sizeof(header));
    
    memcpy(header.magic, "HNFv9\x00\x00\x00", 8);
    header.version_major = 9;
    header.version_minor = 0;
    header.flags = 0;
    header.block_count = 16;
    header.header_size = 64;
    header.block_table_offset = 64;
    
    // We'll calculate these after
    uint64_t block_table_end = 64 + 512;  // Header + block table
    
    // ========================================
    // PREPARE TENSOR DATA
    // ========================================
    
    // Create a small mock tensor: [4, 8] fp16 = 64 elements = 128 bytes
    std::vector<uint16_t> tensor1_data(64, 0x3C00);  // 1.0 in fp16
    
    // Create another tensor: [8] fp16 = 16 bytes
    std::vector<uint16_t> tensor2_data(8, 0x4000);   // 2.0 in fp16
    
    size_t tensor1_size = tensor1_data.size() * sizeof(uint16_t);
    size_t tensor2_size = tensor2_data.size() * sizeof(uint16_t);
    
    // ========================================
    // PREPARE EXECUTION HINTS (Block 0xA)
    // ========================================
    
    std::string exec_hints = R"({
        "text_enabled": true,
        "text": {
            "arch": "mock_test",
            "num_hidden_layers": 2,
            "hidden_size": 8,
            "intermediate_size": 32,
            "vocab_size": 100,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "attention_type": "mha",
            "mlp_type": "swiglu",
            "mlp_activation": "silu",
            "norm_type": "rmsnorm",
            "rms_norm_eps": 1e-6,
            "rope_type": "default",
            "rope_theta": 10000.0,
            "rope_dim": 4,
            "tie_word_embeddings": false,
            "max_position_embeddings": 1024
        }
    })";
    
    // ========================================
    // PREPARE MANIFEST
    // ========================================
    
    // Block 0x0 (text_model) will contain our tensors
    // Tensor 1 at offset 0, tensor 2 at offset tensor1_size
    
    std::string manifest = R"({
        "version": "9.0",
        "tensors": [
            {
                "name": "text.token_embedding.weight",
                "dtype": "fp16",
                "block": "text_model",
                "shape": [4, 8],
                "offset": 0,
                "size": )" + std::to_string(tensor1_size) + R"(
            },
            {
                "name": "text.norm.weight",
                "dtype": "fp16",
                "block": "text_model",
                "shape": [8],
                "offset": )" + std::to_string(tensor1_size) + R"(,
                "size": )" + std::to_string(tensor2_size) + R"(
            }
        ]
    })";
    
    // ========================================
    // CALCULATE OFFSETS
    // ========================================
    
    // Align to 32 bytes
    auto align32 = [](uint64_t x) { return (x + 31) & ~31ULL; };
    
    uint64_t text_block_offset = align32(block_table_end);
    uint64_t text_block_size = tensor1_size + tensor2_size;
    
    uint64_t hints_offset = align32(text_block_offset + text_block_size);
    uint64_t hints_size = exec_hints.size();
    
    uint64_t manifest_offset = align32(hints_offset + hints_size);
    uint64_t manifest_size = manifest.size();
    
    uint64_t file_size = manifest_offset + manifest_size;
    
    // Update header
    header.manifest_offset = manifest_offset;
    header.manifest_size = manifest_size;
    header.file_size = file_size;
    
    // ========================================
    // WRITE HEADER
    // ========================================
    
    f.write(reinterpret_cast<const char*>(&header), sizeof(header));
    
    // ========================================
    // WRITE BLOCK TABLE
    // ========================================
    
    BlockEntry blocks[16];
    memset(blocks, 0, sizeof(blocks));
    
    // Block 0x0: text_model
    blocks[0].block_id = 0;
    blocks[0].block_type = 0;
    blocks[0].offset = text_block_offset;
    blocks[0].size = text_block_size;
    blocks[0].checksum = 0;
    
    // Block 0xA: execution_hints
    blocks[0xA].block_id = 0xA;
    blocks[0xA].block_type = 0xA;
    blocks[0xA].offset = hints_offset;
    blocks[0xA].size = hints_size;
    blocks[0xA].checksum = 0;
    
    f.write(reinterpret_cast<const char*>(blocks), sizeof(blocks));
    
    // ========================================
    // WRITE PADDING TO TEXT BLOCK
    // ========================================
    
    uint64_t current_pos = 64 + 512;
    while (current_pos < text_block_offset) {
        char zero = 0;
        f.write(&zero, 1);
        current_pos++;
    }
    
    // ========================================
    // WRITE TENSOR DATA
    // ========================================
    
    f.write(reinterpret_cast<const char*>(tensor1_data.data()), tensor1_size);
    f.write(reinterpret_cast<const char*>(tensor2_data.data()), tensor2_size);
    
    // ========================================
    // WRITE PADDING TO HINTS
    // ========================================
    
    current_pos = text_block_offset + text_block_size;
    while (current_pos < hints_offset) {
        char zero = 0;
        f.write(&zero, 1);
        current_pos++;
    }
    
    // ========================================
    // WRITE EXECUTION HINTS
    // ========================================
    
    f.write(exec_hints.data(), exec_hints.size());
    
    // ========================================
    // WRITE PADDING TO MANIFEST
    // ========================================
    
    current_pos = hints_offset + hints_size;
    while (current_pos < manifest_offset) {
        char zero = 0;
        f.write(&zero, 1);
        current_pos++;
    }
    
    // ========================================
    // WRITE MANIFEST
    // ========================================
    
    f.write(manifest.data(), manifest.size());
    
    f.close();
    return "";
}

// ============================================================================
// TESTS
// ============================================================================

void test_header_parsing() {
    std::cout << "Test: Header parsing... ";
    
    // Create mock file
    std::string path = "/tmp/test_model.hnf";
    std::string err = create_mock_hnf(path);
    assert(err.empty());
    
    // Load metadata
    HnfLoader loader;
    bool ok = loader.load_metadata(path);
    assert(ok);
    
    // Check header
    const auto& header = loader.header();
    assert(header.version_major == 9);
    assert(header.version_minor == 0);
    assert(header.block_count == 16);
    
    std::cout << "PASSED" << std::endl;
}

void test_block_table() {
    std::cout << "Test: Block table... ";
    
    std::string path = "/tmp/test_model.hnf";
    
    HnfLoader loader;
    bool ok = loader.load_metadata(path);
    assert(ok);
    
    // Check text_model block exists
    assert(loader.has_block(BLOCK_TEXT_MODEL));
    assert(loader.block(BLOCK_TEXT_MODEL).size > 0);
    
    // Check execution_hints block exists
    assert(loader.has_block(BLOCK_EXEC_HINTS));
    assert(loader.block(BLOCK_EXEC_HINTS).size > 0);
    
    // Vision block should be empty
    assert(!loader.has_block(BLOCK_VISION));
    
    std::cout << "PASSED" << std::endl;
}

void test_manifest_parsing() {
    std::cout << "Test: Manifest parsing... ";
    
    std::string path = "/tmp/test_model.hnf";
    
    HnfLoader loader;
    bool ok = loader.load_metadata(path);
    assert(ok);
    
    // Check tensor count
    const auto& tensors = loader.tensors();
    assert(tensors.size() == 2);
    
    // Check first tensor
    assert(tensors[0].name == "text.token_embedding.weight");
    assert(tensors[0].dtype == "fp16");
    assert(tensors[0].shape.size() == 2);
    assert(tensors[0].shape[0] == 4);
    assert(tensors[0].shape[1] == 8);
    
    // Check second tensor
    assert(tensors[1].name == "text.norm.weight");
    assert(tensors[1].shape.size() == 1);
    assert(tensors[1].shape[0] == 8);
    
    std::cout << "PASSED" << std::endl;
}

void test_execution_hints() {
    std::cout << "Test: Execution hints... ";
    
    std::string path = "/tmp/test_model.hnf";
    
    HnfLoader loader;
    bool ok = loader.load_metadata(path);
    assert(ok);
    
    const auto& config = loader.config();
    
    assert(config.text_enabled == true);
    assert(config.arch == "mock_test");
    assert(config.num_hidden_layers == 2);
    assert(config.hidden_size == 8);
    assert(config.vocab_size == 100);
    assert(config.num_attention_heads == 2);
    assert(config.mlp_type == "swiglu");
    assert(config.norm_type == "rmsnorm");
    
    std::cout << "PASSED" << std::endl;
}

void test_tensor_loading() {
    std::cout << "Test: Tensor loading to GPU... ";
    
    std::string path = "/tmp/test_model.hnf";
    
    Engine engine;
    HnfLoader loader;
    bool ok = loader.load(path, engine);
    assert(ok);
    
    // Check tensors are registered
    assert(engine.tensors().exists("text.token_embedding.weight"));
    assert(engine.tensors().exists("text.norm.weight"));
    
    // Check tensor info
    auto* t1 = engine.tensors().get("text.token_embedding.weight");
    assert(t1 != nullptr);
    assert(t1->shape.size() == 2);
    assert(t1->shape[0] == 4);
    assert(t1->shape[1] == 8);
    assert(t1->numel() == 32);
    assert(t1->ptr != nullptr);
    
    auto* t2 = engine.tensors().get("text.norm.weight");
    assert(t2 != nullptr);
    assert(t2->shape.size() == 1);
    assert(t2->shape[0] == 8);
    
    std::cout << "PASSED" << std::endl;
}

void test_tensor_values() {
    std::cout << "Test: Tensor values correctness... ";
    
    std::string path = "/tmp/test_model.hnf";
    
    Engine engine;
    HnfLoader loader;
    loader.load(path, engine);
    
    auto* t1 = engine.tensors().get("text.token_embedding.weight");
    
    // Copy back to host and verify
    std::vector<uint16_t> host_data(t1->numel());
    cudaMemcpy(host_data.data(), t1->ptr, t1->size_bytes, cudaMemcpyDeviceToHost);
    
    // All values should be 0x3C00 (1.0 in fp16)
    for (size_t i = 0; i < host_data.size(); i++) {
        assert(host_data[i] == 0x3C00);
    }
    
    std::cout << "PASSED" << std::endl;
}

void test_print_info() {
    std::cout << "Test: Print info..." << std::endl;
    
    std::string path = "/tmp/test_model.hnf";
    
    HnfLoader loader;
    loader.load_metadata(path);
    
    std::cout << std::endl;
    loader.print_info();
    loader.print_tensors();
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "==================================" << std::endl;
    std::cout << "HELIOS ENGINE - HNF Loader Test" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Check CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "ERROR: No CUDA devices found" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << std::endl;
    
    // Run tests
    test_header_parsing();
    test_block_table();
    test_manifest_parsing();
    test_execution_hints();
    test_tensor_loading();
    test_tensor_values();
    test_print_info();
    
    // Cleanup
    std::remove("/tmp/test_model.hnf");
    
    std::cout << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "ALL TESTS PASSED ✓" << std::endl;
    std::cout << "==================================" << std::endl;
    
    return 0;
}
