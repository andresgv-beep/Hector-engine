// El modelo partido en dos motores debe dar los logits del motor único, token a
// token, con grafos grabados. Pesos sintéticos: no prueba calidad del modelo.

#include "gemma4_kv_cache.hpp"
#include "gemma4_split.hpp"
#include "graph_builder.hpp"
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using helios::dtype::FP16;

constexpr uint32_t LAYERS = 6, BOUNDARY = 2, D = 512, HEADS = 16, LOCAL_KV = 8, VOCAB = 32;
constexpr uint32_t E = 8, K = 2, I = 128, P = 256, SHARED = 384, SHARED_P = 512;
// El prefill pasa de 32 tokens para entrar en la ruta GEMM de la atención, que tiene
// su propio espacio de trabajo por GPU.
constexpr uint32_t PREFILL = 40, DECODE = 8, MAX_SEQ = 64;

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void cuda_require(cudaError_t error, const std::string& what) {
    require(error == cudaSuccess, what + ": " + cudaGetErrorString(error));
}

helios::ModelConfig make_config() {
    helios::ModelConfig config;
    config.set("arch", std::string("gemma4"));
    config.set("num_hidden_layers", int64_t(LAYERS));
    config.set("hidden_size", int64_t(D));
    config.set("intermediate_size", int64_t(SHARED));
    config.set("vocab_size", int64_t(VOCAB));
    config.set("num_attention_heads", int64_t(HEADS));
    config.set("num_key_value_heads", int64_t(LOCAL_KV));
    config.set("rms_norm_eps", 1.0e-6);
    config.set("hidden_act", std::string("gelu_pytorch_tanh"));
    config.set("final_logit_softcapping", 30.0);
    return config;
}

helios::Gemma4Config make_gemma() {
    helios::Gemma4Config gemma;
    gemma.flags = helios::GEMMA4_EXT_FLAG_MOE | helios::GEMMA4_EXT_FLAG_LAYER_SCALAR |
                  helios::GEMMA4_EXT_FLAG_LOGIT_SOFTCAP;
    gemma.moe_experts = E;
    gemma.moe_top_k = K;
    gemma.moe_intermediate_size = I;
    gemma.layers.resize(LAYERS);
    for (uint32_t i = 0; i < LAYERS; ++i) {
        helios::Gemma4LayerConfig& layer = gemma.layers[i];
        const bool global = i % 3 == 2;
        layer.attention_kind = global ? 1 : 0;
        layer.sliding_window = global ? 0 : 16;
        layer.head_dim = global ? 512 : 256;
        layer.num_kv_heads = global ? 1 : LOCAL_KV;
        layer.intermediate_size = SHARED;
        layer.rope_type = helios::ROPE_PROPORTIONAL;
        layer.rope_theta = 10000.0f;
        layer.partial_rotary_factor = 1.0f;
        layer.moe_shared_down.physical_cols = SHARED_P;
    }
    return gemma;
}

helios::ArchDescriptor make_arch() {
    helios::ArchDescriptor arch;
    arch.prefix = "text";
    arch.num_layers = LAYERS;
    arch.embedding_name = "token_embedding.weight";
    arch.final_norm_name = "final_norm.weight";
    arch.lm_head_name = "lm_head.weight";
    return arch;
}

using Weights = std::map<std::string, std::pair<std::vector<uint32_t>, std::vector<half>>>;

const Weights& model_weights() {
    static const Weights weights = [] {
        Weights out;
        std::mt19937 rng(23);
        std::normal_distribution<float> normal(0.0f, 0.03f);
        const auto add = [&](const std::string& name, std::vector<uint32_t> shape, bool unit = false) {
            size_t count = 1;
            for (uint32_t dim : shape) count *= dim;
            std::vector<half> values(count);
            for (half& value : values) value = __float2half(unit ? 1.0f : normal(rng));
            out[name] = {std::move(shape), std::move(values)};
        };
        const auto gemma = make_gemma();
        for (uint32_t i = 0; i < LAYERS; ++i) {
            const std::string prefix = "text.layer" + std::to_string(i) + ".";
            const uint32_t HD = gemma.layers[i].head_dim, KVH = gemma.layers[i].num_kv_heads;
            for (const char* norm : {"ln_attn_in.weight", "ln_attn_post.weight", "ln_mlp_in.weight",
                                     "ln_mlp_branch_post.weight", "ln_mlp_post.weight", "ln_moe_in.weight",
                                     "ln_moe_branch_post.weight", "moe.router.input_scale"}) {
                add(prefix + norm, {D}, true);
            }
            add(prefix + "attn.q_norm.weight", {HD}, true);
            add(prefix + "attn.k_norm.weight", {HD}, true);
            add(prefix + "layer_scalar", {1}, true);
            add(prefix + "moe.router.expert_scale", {E}, true);
            add(prefix + "attn.q_proj.weight", {HEADS * HD, D});
            add(prefix + "attn.k_proj.weight", {KVH * HD, D});
            add(prefix + "attn.v_proj.weight", {KVH * HD, D});
            add(prefix + "attn.o_proj.weight", {D, HEADS * HD});
            add(prefix + "mlp.gate.weight", {SHARED, D});
            add(prefix + "mlp.up.weight", {SHARED, D});
            add(prefix + "mlp.down.weight", {D, SHARED_P});
            add(prefix + "moe.router.weight", {E, D});
            add(prefix + "moe.experts.gate_up.weight", {E, 2 * I, D});
            add(prefix + "moe.experts.down.weight", {E, D, P});
        }
        add("text.token_embedding.weight", {VOCAB, D});
        add("text.lm_head.weight", {VOCAB, D});
        add("text.final_norm.weight", {D}, true);
        return out;
    }();
    return weights;
}

// Solo sube los pesos de sus capas: si una parte tocara los de la otra, falla.
struct Device {
    helios::Engine engine;
    helios::GraphBuilder builder;
    helios::Gemma4KVCache cache;
    Device(const helios::EngineConfig& config, uint32_t first, uint32_t end, bool common,
           const helios::Gemma4Config& gemma, const helios::ArchDescriptor& arch)
        : engine(config) {
        helios::kernels::register_all_kernels(engine);
        for (const auto& [name, weight] : model_weights()) {
            const size_t mark = name.find(".layer");
            const bool ours = mark == std::string::npos
                ? common
                : [&] {
                      const uint32_t layer = uint32_t(std::atoi(name.c_str() + mark + 6));
                      return layer >= first && layer < end;
                  }();
            if (!ours) continue;
            void* device = engine.tensors().allocate_and_register(name, weight.first, FP16());
            cuda_require(cudaMemcpy(device, weight.second.data(), weight.second.size() * sizeof(half),
                                    cudaMemcpyHostToDevice), name);
        }
        const auto config_model = make_config();
        builder.allocate_gemma4_scratch(engine, config_model, gemma, arch, 1, PREFILL);
        require(cache.allocate(gemma, LOCAL_KV, 1, MAX_SEQ, PREFILL, nullptr, first, end), "KV cache");
        cache.register_tensors(engine, "_kv");
    }
};

cudaStream_t new_stream(int device) {
    cuda_require(cudaSetDevice(device), "set device");
    cudaStream_t stream = nullptr;
    cuda_require(cudaStreamCreate(&stream), "stream");
    return stream;
}

helios::EngineConfig engine_config(int device, bool flash) {
    helios::EngineConfig config;
    config.device = device;
    config.stream = new_stream(device);
    config.use_flash_decode = flash;
    return config;
}

std::vector<int32_t> token_ids(uint32_t count, uint32_t offset) {
    std::vector<int32_t> tokens(count);
    for (uint32_t i = 0; i < count; ++i) tokens[i] = int32_t((i * 7 + offset * 5 + 3) % VOCAB);
    return tokens;
}

std::vector<float> read_logits(helios::Engine& engine, helios::GraphBuilder& builder) {
    engine.sync();
    std::vector<half> raw(VOCAB);
    cuda_require(cudaMemcpy(raw.data(), builder.get_logits(engine)->ptr, VOCAB * sizeof(half),
                            cudaMemcpyDeviceToHost), "logits");
    std::vector<float> logits(VOCAB);
    for (uint32_t i = 0; i < VOCAB; ++i) logits[i] = __half2float(raw[i]);
    return logits;
}

std::vector<std::vector<float>> run_single(int device, bool flash) {
    const auto config = make_config();
    const auto gemma = make_gemma();
    const auto arch = make_arch();
    Device model(engine_config(device, flash), 0, LAYERS, true, gemma, arch);
    auto& engine = model.engine;
    void* prefill_ids = engine.tensors().allocate_and_register("ids.prefill", {1, PREFILL}, helios::dtype::INT32(), true);
    void* token_id = engine.tensors().allocate_and_register("ids.token", {1, 1}, helios::dtype::INT32(), true);

    std::vector<std::vector<float>> logits;
    const auto tokens = token_ids(PREFILL, 0);
    cuda_require(cudaMemcpy(prefill_ids, tokens.data(), PREFILL * 4, cudaMemcpyHostToDevice), "ids");
    engine.update_device_cache_pos(0, PREFILL);
    engine.execute(model.builder.build_gemma4_forward_cached(
        engine, config, gemma, arch, "ids.prefill", 1, PREFILL, {"_kv", 0, MAX_SEQ}));
    logits.push_back(read_logits(engine, model.builder));

    const auto decode = model.builder.build_gemma4_forward_cached(
        engine, config, gemma, arch, "ids.token", 1, 1, {"_kv", PREFILL, MAX_SEQ});
    for (uint32_t step = 0; step < DECODE; ++step) {
        const int32_t token = token_ids(1, step + 1)[0];
        cuda_require(cudaMemcpy(token_id, &token, 4, cudaMemcpyHostToDevice), "id");
        engine.update_device_cache_pos(int32_t(PREFILL + step), 1);
        if (step == 0) engine.execute(decode);
        else engine.execute_graph_replay(decode);
        logits.push_back(read_logits(engine, model.builder));
    }
    engine.invalidate_graph();
    return logits;
}

std::vector<std::vector<float>> run_split(int primary_id, int secondary_id, bool flash) {
    const auto config = make_config();
    const auto gemma = make_gemma();
    const auto arch = make_arch();
    Device secondary(engine_config(secondary_id, flash), BOUNDARY, LAYERS, false, gemma, arch);
    Device primary(engine_config(primary_id, flash), 0, BOUNDARY, true, gemma, arch);
    void* prefill_ids = primary.engine.tensors().allocate_and_register("ids.prefill", {1, PREFILL}, helios::dtype::INT32(), true);
    void* token_id = primary.engine.tensors().allocate_and_register("ids.token", {1, 1}, helios::dtype::INT32(), true);

    const auto parts = [&](const std::string& ids, uint32_t seq, uint32_t position, auto&& use) {
        const helios::KVCacheParams cache{"_kv", position, MAX_SEQ};
        secondary.engine.activate();
        const auto rest = secondary.builder.build_gemma4_layers_cached(
            secondary.engine, config, gemma, arch, BOUNDARY, LAYERS, 1, seq, cache);
        primary.engine.activate();
        auto first = primary.builder.build_gemma4_input(primary.engine, config, gemma, arch, ids, 1, seq);
        first.append(primary.builder.build_gemma4_layers_cached(
            primary.engine, config, gemma, arch, 0, BOUNDARY, 1, seq, cache));
        const auto head = primary.builder.build_gemma4_head(config, gemma, arch, seq);
        use(helios::Gemma4SplitForward::Parts{first, rest, head});
    };

    primary.engine.activate();
    helios::Gemma4SplitForward split(primary.engine, secondary.engine, "_s.hidden");
    std::vector<std::vector<float>> logits;
    const auto tokens = token_ids(PREFILL, 0);
    cuda_require(cudaMemcpy(prefill_ids, tokens.data(), PREFILL * 4, cudaMemcpyHostToDevice), "ids");
    split.update_device_cache_pos(0, PREFILL);
    parts("ids.prefill", PREFILL, 0, [&](const auto& forward) { split.run(forward); });
    logits.push_back(read_logits(primary.engine, primary.builder));

    parts("ids.token", 1, PREFILL, [&](const auto& forward) {
        for (uint32_t step = 0; step < DECODE; ++step) {
            primary.engine.activate();
            const int32_t token = token_ids(1, step + 1)[0];
            cuda_require(cudaMemcpy(token_id, &token, 4, cudaMemcpyHostToDevice), "id");
            split.update_device_cache_pos(int32_t(PREFILL + step), 1);
            if (step == 0) split.run(forward);
            else split.replay(forward);
            logits.push_back(read_logits(primary.engine, primary.builder));
            if (step > 0) require(split.graphs_ready(), "las tres partes deben grabarse en grafos");
        }
    });
    primary.engine.activate();
    primary.engine.invalidate_graph();
    secondary.engine.activate();
    secondary.engine.invalidate_graph();
    primary.engine.activate();
    return logits;
}

double worst_error(const std::vector<std::vector<float>>& got, const std::vector<std::vector<float>>& want) {
    double worst = 0.0;
    for (size_t step = 0; step < want.size(); ++step) {
        double diff = 0.0, norm = 0.0;
        for (size_t i = 0; i < want[step].size(); ++i) {
            const double d = double(got[step][i]) - want[step][i];
            diff += d * d;
            norm += double(want[step][i]) * want[step][i];
        }
        worst = std::max(worst, std::sqrt(diff / std::max(norm, 1e-30)));
    }
    return worst;
}

void test_plan() {
    using helios::plan_gemma4_split;
    const std::vector<uint64_t> layers(10, 100);
    auto all = plan_gemma4_split(layers, 50, 2000, 0);
    require(all && !all->split(10) && all->boundary == 10, "todo cabe en la principal");
    auto half_way = plan_gemma4_split(layers, 50, 450, 600);
    require(half_way && half_way->boundary == 4 && half_way->primary_bytes == 450 &&
            half_way->secondary_bytes == 600, "justo de sitio: el único corte que cabe");
    // Con holgura, el margen se reparte en vez de llenar la principal hasta el tope: el corte
    // 9 dejaría 50 libres en la principal y 800 en la otra; el 5 deja 450 y 400.
    auto balanced = plan_gemma4_split(layers, 50, 1000, 900);
    require(balanced && balanced->boundary == 5 && balanced->primary_bytes == 550 &&
            balanced->secondary_bytes == 500, "el margen se reparte entre las dos tarjetas");
    // A igualdad de margen, más capas en la principal: con 1050 y 1000 los cortes 5 y 6
    // empatan a 500 de margen.
    auto tie = plan_gemma4_split(layers, 50, 1150, 1000);
    require(tie && tie->boundary == 6, "a igualdad gana la principal");
    // El caso medido: 26B a 32.768 en una 4070 Ti y una 2080 Ti (MiB). Antes cortaba en 17.
    std::vector<uint64_t> real(30, 514);
    for (size_t i = 5; i < 30; i += 6) real[i] = 641;
    auto medido = plan_gemma4_split(real, 748, 9798, 9255 - 1407);
    require(medido && medido->boundary == 16, "el 26B a 32k corta en 16 y no llena la principal");
    require(!plan_gemma4_split(layers, 50, 450, 500), "sin sitio no hay plan");
    require(!plan_gemma4_split(layers, 500, 450, 2000), "lo común debe caber en la principal");
    std::cout << "PASS: plan de reparto" << std::endl;
}

// Los tres estados y la visión (auditoría del 2026-11-03). Capas de 100, común 50.
void test_placement() {
    using namespace helios;
    const std::vector<uint64_t> layers(10, 100);
    // Texto (1050) y torre (200) caben en la principal: todo allí, sin reparto.
    auto p = plan_gemma4_placement(layers, 50, 200, 1300, 0);
    require(p.fit == Gemma4Fit::Single && p.vision == Gemma4VisionPlace::Primary, "todo en la principal");
    // Una sola GPU donde cabe el texto pero no la torre: solo texto, sin error.
    p = plan_gemma4_placement(layers, 50, 200, 1100, 0);
    require(p.fit == Gemma4Fit::Single && p.vision == Gemma4VisionPlace::None && p.vision_dropped,
            "una GPU sin sitio para la torre abre solo el texto");
    // Dos GPUs, el texto cabe solo en la principal pero no con la torre: la torre va sola a la
    // secundaria y el texto no se reparte. Vale aunque en la secundaria no quepa además una
    // capa (250 libres: torre 200 sí, torre + capa 300 no).
    p = plan_gemma4_placement(layers, 50, 200, 1100, 250);
    require(p.fit == Gemma4Fit::Single && p.vision == Gemma4VisionPlace::Secondary && !p.vision_dropped,
            "texto entero en la principal y torre sola en la secundaria");
    // Reparto normal con la torre en la secundaria. Corta en 3 y no en 4: con 4 la principal
    // quedaría sin margen (450 de 450) y la secundaria con 200; con 3 quedan 100 y 100.
    p = plan_gemma4_placement(layers, 50, 200, 450, 1000);
    require(p.fit == Gemma4Fit::Split && p.boundary == 3 && p.vision == Gemma4VisionPlace::Secondary,
            "reparto con torre en la secundaria");
    // Todas las capas en la secundaria (boundary 0) es un reparto válido, no «no repartir».
    p = plan_gemma4_placement(layers, 50, 0, 120, 1200);
    require(p.fit == Gemma4Fit::Split && p.boundary == 0, "boundary 0 es válido");
    // La torre no cabe en la secundaria: reparto solo de texto.
    p = plan_gemma4_placement(layers, 50, 900, 450, 700);
    require(p.fit == Gemma4Fit::Split && p.vision == Gemma4VisionPlace::None && p.vision_dropped,
            "sin sitio para la torre se reparte solo el texto");
    // Ni el texto cabe entre las dos: NoFit, no un 0 que se confunda con «cabe».
    require(plan_gemma4_placement(layers, 50, 0, 450, 500).fit == Gemma4Fit::NoFit, "no cabe");
    require(plan_gemma4_placement(layers, 50, 0, 900, 0).fit == Gemma4Fit::NoFit, "no cabe en una GPU");
    // KV compartido (E4B): nunca reparte el texto, pero la torre sí puede ir a la secundaria.
    p = plan_gemma4_placement(layers, 50, 200, 1100, 250, false);
    require(p.fit == Gemma4Fit::Single && p.vision == Gemma4VisionPlace::Secondary, "E4B: torre sola en la secundaria");
    require(plan_gemma4_placement(layers, 50, 0, 450, 2000, false).fit == Gemma4Fit::NoFit,
            "E4B que no cabe en la principal: no se reparte");
    std::cout << "PASS: colocación de texto y visión" << std::endl;
}

}  // namespace

int main() {
    test_plan();
    test_placement();
    int devices = 0;
    cuda_require(cudaGetDeviceCount(&devices), "device count");
    for (bool flash : {false, true}) {
        const auto single = run_single(0, flash);
        const double same = worst_error(run_split(0, 0, flash), single);
        std::cout << "  flash=" << flash << " partido en la misma GPU: error " << same << std::endl;
        require(same == 0.0, "partir en dos motores no debe cambiar ningún logit");
        if (devices < 2) {
            std::cout << "  una sola GPU: se omite el cruce real" << std::endl;
            continue;
        }
        // Dos tarjetas no redondean igual: el listón es lo que ya difieren sin partir.
        const double hardware = worst_error(run_single(1, flash), single);
        const double cross = worst_error(run_split(0, 1, flash), single);
        std::cout << "  flash=" << flash << " entre GPUs sin partir: error " << hardware
                  << "; partido: error " << cross << std::endl;
        require(cross <= 1.5 * hardware + 1e-3, "logits entre GPUs");
    }
    cudaSetDevice(0);
    std::cout << "Gemma 4 split OK" << std::endl;
    return 0;
}
