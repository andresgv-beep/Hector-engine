// tests/test_hnf_loader.cpp
// ============================================================================
// HELIOS ENGINE - HNF Loader Test
// ============================================================================

#include "hnf_loader.hpp"
#include "model_capabilities.hpp"
#include "hnf10_test_writer.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>

using namespace helios;

// ============================================================================
// CREATE MOCK HNFv10 FILE
// ============================================================================

// Configuración binaria TNIH de un modelo de texto de juguete: 2 capas, ancho 8, vocabulario 100.
std::vector<uint8_t> mock_config() {
    ExecutionHintsBin hints{};
    hints.magic = HINTS_MAGIC;
    hints.version_major = 1;
    hints.text_offset = sizeof(ExecutionHintsBin);
    hints.num_text_models = 1;
    hints.flags = 1;                              // text_enabled
    TextModelConfigBin text{};
    text.rope_theta = 10000.0f;
    text.rope_scaling_factor = 1.0f;
    text.partial_rotary_factor = 1.0f;
    text.rms_norm_eps = 1e-6f;
    text.num_hidden_layers = 2;
    text.hidden_size = 8;
    text.intermediate_size = 32;
    text.vocab_size = 100;
    text.max_position_embeddings = 1024;
    text.rope_dim = 4;
    text.num_attention_heads = 2;
    text.num_key_value_heads = 2;
    text.head_dim = 4;
    text.attention_type = ATTN_MHA;
    text.arch = ARCH_PHI4;                         // conocida en metadatos, sin ejecución
    text.mlp_type = MLP_SWIGLU;
    text.norm_type = NORM_RMSNORM;
    std::vector<uint8_t> bytes(sizeof(hints) + sizeof(text));
    std::memcpy(bytes.data(), &hints, sizeof(hints));
    std::memcpy(bytes.data() + sizeof(hints), &text, sizeof(text));
    return bytes;
}

std::vector<uint8_t> fp16_bytes(size_t count, uint16_t value) {
    std::vector<uint8_t> bytes(count * 2);
    for (size_t i = 0; i < count; ++i) std::memcpy(bytes.data() + 2 * i, &value, 2);
    return bytes;
}

// Dos tensores FP16: [4, 8] a 1,0 y [8] a 2,0. Con `malformed_tensor_size`, el primero trae el
// doble de bytes de los que dice su forma y el índice lo rechaza.
std::string create_mock_hnf(const std::string& path, bool malformed_tensor_size = false) {
    hnf10_test::Component text{1, hnf10_test::TEXT, true, "texto", {
        {"text.token_embedding.weight", "fp16", {4, 8}, fp16_bytes(malformed_tensor_size ? 64 : 32, 0x3C00)},
        {"text.norm.weight", "fp16", {8}, fp16_bytes(8, 0x4000)},
    }, {}};
    hnf10_test::Component config{2, hnf10_test::CONFIG, true, "configuración", {}, mock_config()};
    try { hnf10_test::write(path, hnf10_test::build({text, config})); } catch (const std::exception& e) { return e.what(); }
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
    
    // HNFv10: el cargador lo identifica.
    assert(loader.format_major() == 10);
    
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
    
    // La configuración binaria es un componente; no hay hints JSON.
    assert(loader.has_block(BLOCK_EXEC_HINTS_BIN));
    assert(!loader.has_block(BLOCK_EXEC_HINTS));
    
    // Vision block should be empty
    assert(!loader.has_block(BLOCK_VISION));
    
    std::cout << "PASSED" << std::endl;
}

void test_manifest_parsing() {
    std::cout << "Test: Tensor index... ";
    
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

void test_embedding_offload_policy() {
    std::cout << "Test: Shared embedding offload policy... ";

    TensorEntry main_embedding{
        "text.token_embedding.weight", "hq51k", "text_model", {100, 8}, 0, 400};
    TensorEntry ple_embedding{
        "text.ple.token_embedding.weight", "hq51k", "text_model", {100, 8}, 400, 400};
    TensorEntry lm_head{
        "text.lm_head.weight", "hq51k", "text_model", {100, 8}, 800, 400};

    std::vector<TensorEntry> shared_manifest{main_embedding, ple_embedding};
    assert(!detail::embedding_is_lookup_only(main_embedding, shared_manifest));
    assert(detail::embedding_is_lookup_only(ple_embedding, shared_manifest));

    std::vector<TensorEntry> legacy_manifest{main_embedding, ple_embedding, lm_head};
    assert(detail::embedding_is_lookup_only(main_embedding, legacy_manifest));
    assert(detail::embedding_is_lookup_only(ple_embedding, legacy_manifest));

    TensorEntry unrelated{
        "text.layer0.mlp.up.weight", "hq51k", "text_model", {8, 8}, 1200, 200};
    assert(!detail::embedding_is_lookup_only(unrelated, legacy_manifest));

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
    assert(config.arch == "phi4");
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

void test_tensor_size_validation() {
    std::cout << "Test: Tensor size validation... ";

    const std::string path = "/tmp/test_model_bad_size.hnf";
    assert(create_mock_hnf(path, true).empty());

    Engine engine;
    HnfLoader loader;
    assert(!loader.load(path, engine));
    assert(!engine.tensors().exists("text.token_embedding.weight"));

    std::remove(path.c_str());
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

void test_model_capabilities() {
    std::cout << "Test: Generic model capabilities... ";

    HnfLoader loader;
    assert(loader.load_metadata("/tmp/test_model.hnf"));
    const ModelCapabilities model = inspect_model_capabilities(loader);
    assert(model.schema_version == 1);
    assert(model.text_architecture == "phi4");
    assert(!model.multimodal);
    const auto* text = model.find(ModelModality::Text);
    const auto* vision = model.find(ModelModality::Vision);
    assert(text && text->present && text->configured);
    assert(text->status == AdapterStatus::MetadataReady);
    assert(vision && vision->status == AdapterStatus::Absent);

    const auto* gemma4 = resolve_modality_adapter(
        "gemma4", ModelModality::Vision, "gemma4");
    assert(gemma4);
    assert(std::string(gemma4->adapter_id) == "helios.gemma4.vision.v1");
    assert(!resolve_modality_adapter(
        "qwen2", ModelModality::Vision, "gemma4"));
    const auto* qwen_text = resolve_modality_adapter(
        "qwen2", ModelModality::Text, "qwen2");
    assert(qwen_text);
    assert(std::string(qwen_text->adapter_id) == "helios.text.chat.v1");

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
    test_embedding_offload_policy();
    test_execution_hints();
    test_tensor_size_validation();
    test_tensor_loading();
    test_tensor_values();
    test_model_capabilities();
    test_print_info();
    
    // Cleanup
    std::remove("/tmp/test_model.hnf");
    
    std::cout << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "ALL TESTS PASSED ✓" << std::endl;
    std::cout << "==================================" << std::endl;
    
    return 0;
}
