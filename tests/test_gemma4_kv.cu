#include "gemma4_kv_cache.hpp"
#include "graph_builder.hpp"
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void cuda_require(cudaError_t error, const std::string& where) {
    require(error == cudaSuccess, where + ": " + cudaGetErrorString(error));
}

helios::Gemma4Config synthetic_gemma() {
    helios::Gemma4Config gemma;
    gemma.version = 1;
    gemma.flags = helios::GEMMA4_EXT_FLAG_SHARED_KV |
                  helios::GEMMA4_EXT_FLAG_PLE |
                  helios::GEMMA4_EXT_FLAG_LAYER_SCALAR |
                  helios::GEMMA4_EXT_FLAG_LOGIT_SOFTCAP;
    gemma.ple_hidden_size = 256;
    gemma.num_kv_shared_layers = 20;
    gemma.layers.resize(35);
    for (uint32_t i = 0; i < gemma.layers.size(); ++i) {
        auto& layer = gemma.layers[i];
        layer.attention_kind = i % 5 == 4 ? 1 : 0;
        layer.sliding_window = layer.is_global_attention() ? 0 : 512;
        layer.head_dim = layer.is_global_attention() ? 512 : 256;
        layer.intermediate_size = i >= 15 ? 12288 : 6144;
        layer.rope_type = helios::ROPE_PROPORTIONAL;
        layer.rope_theta = 1000000.0f;
        layer.partial_rotary_factor = layer.is_global_attention() ? 0.25f : 0.5f;
    }
    return gemma;
}

helios::ModelConfig synthetic_model() {
    helios::ModelConfig config;
    config.set("arch", std::string("gemma4"));
    config.set("num_hidden_layers", int64_t(35));
    config.set("hidden_size", int64_t(1536));
    config.set("intermediate_size", int64_t(6144));
    config.set("vocab_size", int64_t(64));
    config.set("num_attention_heads", int64_t(8));
    config.set("num_key_value_heads", int64_t(1));
    config.set("rms_norm_eps", 1.0e-6);
    config.set("hidden_act", std::string("gelu_pytorch_tanh"));
    config.set("final_logit_softcapping", 30.0);
    return config;
}

std::vector<std::string> register_detection_contract(helios::Engine& engine,
                                                      void* dummy) {
    std::vector<std::string> names;
    auto add = [&](const std::string& name) {
        engine.tensors().register_external(name, dummy, {1}, helios::dtype::FP16());
        names.push_back(name);
    };
    for (uint32_t i = 0; i < 35; ++i) {
        add("text.layer" + std::to_string(i) + ".ln_attn_in.weight");
    }
    add("text.layer0.ln_attn_post.weight");
    add("text.layer0.attn.q_norm.weight");
    add("text.layer0.mlp.gate.weight");
    add("text.token_embedding.weight");
    add("text.final_norm.weight");
    add("text.lm_head.weight");
    return names;
}

size_t count_op(const helios::CommandBuffer& commands, helios::OpTypeID op) {
    size_t count = 0;
    for (const auto& command : commands.commands()) count += command.op == op;
    return count;
}

const helios::Command& find_op(const helios::CommandBuffer& commands,
                               helios::OpTypeID op) {
    for (const auto& command : commands.commands()) {
        if (command.op == op) return command;
    }
    require(false, "required command missing: " + std::string(helios::op_name(op)));
    return commands[0];
}

bool mentions(const helios::CommandBuffer& commands, const std::string& fragment) {
    for (const auto& command : commands.commands()) {
        if (command.output.find(fragment) != std::string::npos) return true;
        for (const auto& input : command.inputs) {
            if (input.find(fragment) != std::string::npos) return true;
        }
    }
    return false;
}

void test_layout_and_aliases() {
    const auto gemma = synthetic_gemma();
    helios::Gemma4KVCache cache;
    require(cache.allocate(gemma, 1, 1, 8), "allocate heterogeneous KV cache");
    require(cache.num_layers() == 35 && cache.physical_slots() == 15,
            "only the first 15 layers own physical KV slots");
    require(cache.source_layer(13) == 13 && cache.source_layer(14) == 14,
            "source layers own their slots");
    require(cache.source_layer(15) == 13 && cache.source_layer(18) == 13,
            "shared local layers reuse layer 13");
    require(cache.source_layer(19) == 14 && cache.source_layer(34) == 14,
            "shared global layers reuse layer 14");
    require(cache.k_cache(15) == cache.k_cache(13) &&
            cache.v_cache(19) == cache.v_cache(14),
            "shared layers are true pointer aliases");
    require(cache.head_dim(13) == 256 && cache.head_dim(14) == 512,
            "local/global slot dimensions remain heterogeneous");
    require(cache.total_bytes() == 147456,
            "heterogeneous KV byte budget for 8 positions");

    const half value = __float2half(7.0f);
    cuda_require(cudaMemcpy(cache.k_cache(13), &value, sizeof(value),
                            cudaMemcpyHostToDevice), "write local source alias");
    half observed{};
    cuda_require(cudaMemcpy(&observed, cache.k_cache(15), sizeof(observed),
                            cudaMemcpyDeviceToHost), "read shared local alias");
    require(__half2float(observed) == 7.0f, "shared alias observes source writes");

    helios::Engine engine;
    cache.register_tensors(engine, "_g4kv");
    require(engine.tensors().at("_g4kv.layer13.k").shape ==
                std::vector<uint32_t>({1, 8, 1, 256}),
            "local cache tensor shape");
    require(engine.tensors().at("_g4kv.layer14.k").shape ==
                std::vector<uint32_t>({1, 8, 1, 512}),
            "global cache tensor shape");
    require(engine.tensors().at("_g4kv.layer15.k").ptr ==
                engine.tensors().at("_g4kv.layer13.k").ptr,
            "registered local tensor is an alias");
    for (uint32_t i = 0; i < 35; ++i) {
        engine.tensors().remove("_g4kv.layer" + std::to_string(i) + ".k");
        engine.tensors().remove("_g4kv.layer" + std::to_string(i) + ".v");
    }
    std::cout << "PASS: heterogeneous KV layout and shared aliases" << std::endl;
}

void test_cached_graph_contract() {
    const auto gemma = synthetic_gemma();
    const auto config = synthetic_model();
    helios::Engine engine;
    void* dummy = nullptr;
    cuda_require(cudaMalloc(&dummy, 2), "allocate graph dummy");
    const auto detection_names = register_detection_contract(engine, dummy);

    helios::GraphBuilder builder;
    const auto arch = builder.detect_architecture(engine, "text", config);
    builder.allocate_gemma4_scratch(engine, config, gemma, arch, 1, 2);
    helios::Gemma4KVCache cache;
    require(cache.allocate(gemma, 1, 1, 8), "allocate graph KV cache");
    cache.register_tensors(engine, "_g4kv");
    const helios::KVCacheParams params{"_g4kv", 3, 8};

    const auto source = builder.build_gemma4_layer_cached(
        engine, config, gemma, arch, 13, 1, 1, params);
    require(count_op(source, helios::op::KV_CACHE_UPDATE()) == 1,
            "source layer writes one KV slot");
    require(mentions(source, "layer13.attn.k_proj.weight") &&
            mentions(source, "layer13.attn.v_proj.weight"),
            "source layer computes K and V");
    const auto& local_attention = find_op(source, helios::op::ATTENTION_CACHED());
    require(local_attention.get<uint32_t>("head_dim", 0) == 256 &&
            local_attention.get<uint32_t>("window_size", 0) == 512 &&
            local_attention.get<float>("scale", 0.0f) == 1.0f,
            "local cached attention geometry, window and scale");

    const auto shared_local = builder.build_gemma4_layer_cached(
        engine, config, gemma, arch, 15, 1, 1, params);
    require(count_op(shared_local, helios::op::KV_CACHE_UPDATE()) == 0,
            "shared local layer does not overwrite source KV");
    require(!mentions(shared_local, "layer15.attn.k_proj.weight") &&
            !mentions(shared_local, "layer15.attn.v_proj.weight") &&
            mentions(shared_local, "layer15.attn.q_proj.weight"),
            "shared local layer computes Q only");
    const auto& shared_attention = find_op(shared_local, helios::op::ATTENTION_CACHED());
    require(shared_attention.inputs[1] == "_g4kv.layer15.k" &&
            engine.tensors().at(shared_attention.inputs[1]).ptr == cache.k_cache(13),
            "shared local command consumes the layer-13 alias");

    const auto shared_global = builder.build_gemma4_layer_cached(
        engine, config, gemma, arch, 19, 1, 2, params);
    const auto& global_attention = find_op(
        shared_global, helios::op::ATTENTION_PREFILL_CACHED());
    require(global_attention.get<uint32_t>("head_dim", 0) == 512 &&
            global_attention.get<uint32_t>("window_size", 99) == 0 &&
            global_attention.get<float>("scale", 0.0f) == 1.0f,
            "global shared prefill uses full HD512 attention");
    require(count_op(shared_global, helios::op::KV_CACHE_UPDATE()) == 0 &&
            engine.tensors().at(global_attention.inputs[1]).ptr == cache.k_cache(14),
            "shared global layer consumes layer-14 KV without updating it");

    engine.tensors().allocate_and_register(
        "g4.tokens", {1, 1}, helios::dtype::INT32(), true);
    const auto full = builder.build_gemma4_forward_cached(
        engine, config, gemma, arch, "g4.tokens", 1, 1, params);
    require(count_op(full, helios::op::KV_CACHE_UPDATE()) == 15,
            "complete forward updates only the 15 physical source layers");
    require(full[full.size() - 1].op == helios::op::SOFTCAP() &&
            full[full.size() - 1].get<float>("cap", 0.0f) == 30.0f,
            "complete forward finishes with the Gemma 4 logit softcap");

    builder.free_scratch(engine);
    for (uint32_t i = 0; i < 35; ++i) {
        engine.tensors().remove("_g4kv.layer" + std::to_string(i) + ".k");
        engine.tensors().remove("_g4kv.layer" + std::to_string(i) + ".v");
    }
    for (const auto& name : detection_names) engine.tensors().remove(name);
    cudaFree(dummy);
    std::cout << "PASS: source/shared local/global cached graph contract" << std::endl;
}

std::vector<half> repeated_half(size_t count, float value) {
    return std::vector<half>(count, __float2half(value));
}

void test_local_window_and_global_hd512() {
    helios::Engine engine;
    helios::kernels::register_all_kernels(engine);

    engine.tensors().allocate_and_register("attn.q", {1, 1, 1, 2}, helios::dtype::FP16());
    engine.tensors().allocate_and_register("attn.k", {1, 6, 1, 2}, helios::dtype::FP16());
    engine.tensors().allocate_and_register("attn.v", {1, 6, 1, 2}, helios::dtype::FP16());
    engine.tensors().allocate_and_register("attn.out", {1, 1, 1, 2}, helios::dtype::FP16());
    const auto zeros = repeated_half(12, 0.0f);
    std::vector<half> values(12);
    for (int pos = 0; pos < 6; ++pos) {
        values[pos * 2] = values[pos * 2 + 1] = __float2half(float(pos + 1));
    }
    cuda_require(cudaMemset(engine.tensors().at("attn.q").ptr, 0, 2 * sizeof(half)),
                 "zero local Q");
    cuda_require(cudaMemcpy(engine.tensors().at("attn.k").ptr, zeros.data(),
                            zeros.size() * sizeof(half), cudaMemcpyHostToDevice),
                 "copy local K");
    cuda_require(cudaMemcpy(engine.tensors().at("attn.v").ptr, values.data(),
                            values.size() * sizeof(half), cudaMemcpyHostToDevice),
                 "copy local V");
    helios::CommandBuffer local;
    local.add_attention_cached("attn.out", "attn.q", "attn.k", "attn.v",
                               1, 1, 2, 6, 6, 3);
    engine.execute(local);
    engine.sync();
    std::vector<half> local_out(2);
    cuda_require(cudaMemcpy(local_out.data(), engine.tensors().at("attn.out").ptr,
                            2 * sizeof(half), cudaMemcpyDeviceToHost),
                 "read local attention");
    require(std::fabs(__half2float(local_out[0]) - 5.0f) < 0.01f &&
            std::fabs(__half2float(local_out[1]) - 5.0f) < 0.01f,
            "decode local mask keeps only the last three values");

    engine.tensors().allocate_and_register("prefill.q", {1, 2, 1, 2}, helios::dtype::FP16());
    engine.tensors().allocate_and_register("prefill.out", {1, 2, 1, 2}, helios::dtype::FP16());
    cuda_require(cudaMemset(engine.tensors().at("prefill.q").ptr, 0, 4 * sizeof(half)),
                 "zero prefill Q");
    helios::CommandBuffer prefill;
    prefill.add_op(helios::op::ATTENTION_PREFILL_CACHED(), "prefill.out")
        .in({"prefill.q", "attn.k", "attn.v"})
        .set("num_heads", uint32_t{1})
        .set("num_kv_heads", uint32_t{1})
        .set("head_dim", uint32_t{2})
        .set("scale", 1.0f)
        .set("seq_len", uint32_t{2})
        .set("past_len", uint32_t{4})
        .set("max_seq_len", uint32_t{6})
        .set("window_size", uint32_t{3});
    engine.execute(prefill);
    engine.sync();
    std::vector<half> prefill_out(4);
    cuda_require(cudaMemcpy(prefill_out.data(), engine.tensors().at("prefill.out").ptr,
                            prefill_out.size() * sizeof(half), cudaMemcpyDeviceToHost),
                 "read prefill attention");
    require(std::fabs(__half2float(prefill_out[0]) - 4.0f) < 0.01f &&
            std::fabs(__half2float(prefill_out[2]) - 5.0f) < 0.01f,
            "prefill local mask slides independently for each new query");

    engine.tensors().allocate_and_register("boundary.k", {1, 514, 1, 2}, helios::dtype::FP16());
    engine.tensors().allocate_and_register("boundary.v", {1, 514, 1, 2}, helios::dtype::FP16());
    engine.tensors().allocate_and_register("boundary.out", {1, 1, 1, 2}, helios::dtype::FP16());
    std::vector<half> boundary_k(514 * 2, __float2half(0.0f));
    std::vector<half> boundary_v(514 * 2, __float2half(2.0f));
    for (size_t i = 0; i < 4; ++i) boundary_v[i] = __float2half(1000.0f);
    cuda_require(cudaMemcpy(engine.tensors().at("boundary.k").ptr, boundary_k.data(),
                            boundary_k.size() * sizeof(half), cudaMemcpyHostToDevice),
                 "copy boundary K");
    cuda_require(cudaMemcpy(engine.tensors().at("boundary.v").ptr, boundary_v.data(),
                            boundary_v.size() * sizeof(half), cudaMemcpyHostToDevice),
                 "copy boundary V");
    helios::CommandBuffer boundary;
    boundary.add_attention_cached("boundary.out", "attn.q", "boundary.k", "boundary.v",
                                  1, 1, 2, 514, 514, 512);
    engine.execute(boundary);
    engine.sync();
    std::vector<half> boundary_out(2);
    cuda_require(cudaMemcpy(boundary_out.data(), engine.tensors().at("boundary.out").ptr,
                            boundary_out.size() * sizeof(half), cudaMemcpyDeviceToHost),
                 "read 512-token boundary attention");
    require(std::fabs(__half2float(boundary_out[0]) - 2.0f) < 0.01f,
            "token 514 must exclude the two values outside the 512 window");

    engine.tensors().allocate_and_register("global.q", {1, 1, 1, 512}, helios::dtype::FP16());
    engine.tensors().allocate_and_register("global.k", {1, 2, 1, 512}, helios::dtype::FP16());
    engine.tensors().allocate_and_register("global.v", {1, 2, 1, 512}, helios::dtype::FP16());
    engine.tensors().allocate_and_register("global.out", {1, 1, 1, 512}, helios::dtype::FP16());
    cuda_require(cudaMemset(engine.tensors().at("global.q").ptr, 0, 512 * sizeof(half)),
                 "zero global Q");
    cuda_require(cudaMemset(engine.tensors().at("global.k").ptr, 0, 1024 * sizeof(half)),
                 "zero global K");
    auto global_values = repeated_half(1024, 2.0f);
    for (size_t i = 512; i < global_values.size(); ++i) {
        global_values[i] = __float2half(6.0f);
    }
    cuda_require(cudaMemcpy(engine.tensors().at("global.v").ptr, global_values.data(),
                            global_values.size() * sizeof(half), cudaMemcpyHostToDevice),
                 "copy global V");
    helios::CommandBuffer global;
    global.add_attention_cached("global.out", "global.q", "global.k", "global.v",
                                1, 1, 512, 2, 2);
    engine.execute(global);
    engine.sync();
    std::vector<half> global_out(512);
    cuda_require(cudaMemcpy(global_out.data(), engine.tensors().at("global.out").ptr,
                            global_out.size() * sizeof(half), cudaMemcpyDeviceToHost),
                 "read global attention");
    for (half value : global_out) {
        require(std::fabs(__half2float(value) - 4.0f) < 0.01f,
                "HD512 cached attention must write every dimension");
    }
    std::cout << "PASS: local prefill/decode windows (including >512) and global HD512"
              << std::endl;
}

std::vector<half> copy_logits(const helios::Engine& engine, uint32_t vocab) {
    const auto& tensor = engine.tensors().at("_s.logits");
    std::vector<half> logits(vocab);
    cuda_require(cudaMemcpy(logits.data(), tensor.ptr, logits.size() * sizeof(half),
                            cudaMemcpyDeviceToHost), "copy real logits");
    return logits;
}

void test_real_prefill_decode_consistency(const std::string& path) {
    helios::EngineConfig engine_config;
    engine_config.scratch_pool.auto_fraction = 0.0f;
    engine_config.scratch_pool.min_size_bytes = 0;
    helios::Engine engine(engine_config);
    helios::kernels::register_all_kernels(engine);
    helios::HnfLoader loader;
    require(loader.open(path), "open real Gemma 4 HNF: " + loader.last_error());
    require(loader.has_gemma4_config(), "real HNF exposes GM4X");
    require(loader.load_block(helios::BLOCK_TEXT_MODEL, engine),
            "load real Gemma 4 text weights");

    const auto config = loader.config();
    const auto& gemma = loader.gemma4_config();
    helios::GraphBuilder builder;
    const auto arch = builder.detect_architecture(engine, "text", config);
    builder.allocate_gemma4_scratch(engine, config, gemma, arch, 1, 3);

    helios::Gemma4KVCache cache;
    require(cache.allocate(gemma, config.num_key_value_heads(), 1, 8),
            "allocate real heterogeneous KV cache");
    cache.register_tensors(engine, "_g4kv");
    void* token_ptr = engine.tensors().allocate_and_register(
        "g4.real.tokens", {1, 3}, helios::dtype::INT32());
    const std::vector<int32_t> tokens{42, 43, 44};
    cuda_require(cudaMemcpy(token_ptr, tokens.data(), tokens.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice), "copy real prefill tokens");

    const helios::KVCacheParams prefill_cache{"_g4kv", 0, 8};
    const auto prefill = builder.build_gemma4_forward_cached(
        engine, config, gemma, arch, "g4.real.tokens", 1, 3, prefill_cache);
    engine.execute(prefill);
    engine.sync();
    const auto prefill_logits = copy_logits(engine, config.vocab_size());

    auto& token_info = engine.tensors().at("g4.real.tokens");
    token_info.shape = {1, 1};
    for (uint32_t position = 0; position < tokens.size(); ++position) {
        cuda_require(cudaMemcpy(token_ptr, &tokens[position], sizeof(int32_t),
                                cudaMemcpyHostToDevice), "copy real decode token");
        const helios::KVCacheParams decode_cache{"_g4kv", position, 8};
        const auto decode = builder.build_gemma4_forward_cached(
            engine, config, gemma, arch, "g4.real.tokens", 1, 1, decode_cache);
        engine.execute(decode);
        engine.sync();
    }
    const auto decode_logits = copy_logits(engine, config.vocab_size());

    float max_error = 0.0f;
    double mean_error = 0.0;
    uint32_t prefill_argmax = 0;
    uint32_t decode_argmax = 0;
    float prefill_max = -std::numeric_limits<float>::infinity();
    float decode_max = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < config.vocab_size(); ++i) {
        const float a = __half2float(prefill_logits[i]);
        const float b = __half2float(decode_logits[i]);
        require(std::isfinite(a) && std::isfinite(b), "real logits must be finite");
        const float error = std::fabs(a - b);
        max_error = std::max(max_error, error);
        mean_error += error;
        if (a > prefill_max) { prefill_max = a; prefill_argmax = i; }
        if (b > decode_max) { decode_max = b; decode_argmax = i; }
    }
    mean_error /= config.vocab_size();
    require(max_error <= 0.125f && mean_error <= 0.005 &&
            prefill_argmax == decode_argmax,
            "real prefill/decode logits diverged");

    builder.free_scratch(engine);
    for (uint32_t i = 0; i < gemma.layers.size(); ++i) {
        engine.tensors().remove("_g4kv.layer" + std::to_string(i) + ".k");
        engine.tensors().remove("_g4kv.layer" + std::to_string(i) + ".v");
    }
    std::cout << "PASS: real prefill/decode logits max=" << max_error
              << " mean=" << mean_error << " argmax=" << prefill_argmax
              << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    cuda_require(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    require(devices > 0, "CUDA device required");
    if (argc > 1) {
        test_real_prefill_decode_consistency(argv[1]);
        std::cout << "ALL REAL GEMMA 4 KV TESTS PASSED" << std::endl;
        return 0;
    }
    test_layout_and_aliases();
    test_cached_graph_contract();
    test_local_window_and_global_hd512();
    std::cout << "ALL GEMMA 4 KV TESTS PASSED" << std::endl;
    return 0;
}
