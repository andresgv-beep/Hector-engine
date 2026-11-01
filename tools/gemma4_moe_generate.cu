// Generación de Gemma 4 MoE con las capas repartidas entre las GPUs visibles.
// La sesión de chat aún no conoce el reparto: esto lo prueba con el modelo real.
//
// uso: gemma4_moe_generate modelo.hnf "pregunta" [--max N] [--ctx N] [--boundary N]
//      gemma4_moe_generate modelo.hnf --prompts fichero --out dir/ [--max N]
//      gemma4_moe_generate modelo.hnf --capture corpus/ --out dir/ [--limit N] [--seed N]
//
// --prompts responde una pregunta por línea y guarda cada conversación como .ids.
// --capture acumula las activaciones de down por experto (ver
// helios_convert_v9.1/docs/GEMMA4_MOE_CALIBRATION.md): hay dos mitades por documento
// y uno de cada cinco documentos se reserva para la muestra de control.
//
// Con CUDA_DEVICE_ORDER=PCI_BUS_ID y CUDA_VISIBLE_DEVICES=1,0 la principal es la 4070 Ti.

#include "chat_template.hpp"
#include "gemma4_kv_cache.hpp"
#include "gemma4_split.hpp"
#include "graph_builder.hpp"
#include "hnf_loader.hpp"
#include "kernels.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace helios;

namespace {

constexpr uint32_t PREFILL_CHUNK = 512;
constexpr uint64_t VRAM_RESERVE = 900ull << 20;
const char* const HIDDEN = "_s.hidden";

void check(cudaError_t error, const std::string& what) {
    if (error != cudaSuccess) throw std::runtime_error(what + ": " + cudaGetErrorString(error));
}

// -1 para los tensores comunes.
int layer_of(const std::string& name) {
    const size_t mark = name.find(".layer");
    return mark == std::string::npos ? -1 : std::atoi(name.c_str() + mark + 6);
}

uint64_t free_vram(int device) {
    check(cudaSetDevice(device), "elegir GPU");
    size_t free_bytes = 0, total = 0;
    check(cudaMemGetInfo(&free_bytes, &total), "memoria libre");
    return free_bytes;
}

EngineConfig engine_config(int device) {
    check(cudaSetDevice(device), "elegir GPU");
    EngineConfig config;
    config.device = device;
    check(cudaStreamCreate(&config.stream), "crear stream");
    config.use_split_attention = true;
    config.use_coalesced_prefill = true;
    config.use_flash_decode = true;
    config.flash_decode_min_seq = 1024;
    config.use_gemm_prefill = true;
    config.use_flash_prefill = true;
    return config;
}

struct Device {
    Engine engine;
    HnfLoader loader;
    GraphBuilder builder;
    Gemma4KVCache cache;
    uint32_t first, end;

    Device(int device, const std::string& path, uint32_t first_layer, uint32_t end_layer, bool common,
           const ArchDescriptor& arch, uint32_t max_seq)
        : engine(engine_config(device)), first(first_layer), end(end_layer) {
        kernels::register_all_kernels(engine);
        if (!loader.open(path)) throw std::runtime_error(loader.last_error());
        loader.set_tensor_filter([=](const TensorEntry& tensor) {
            const int layer = layer_of(tensor.name);
            return layer < 0 ? common : layer >= int(first_layer) && layer < int(end_layer);
        });
        if (!loader.load_block(BLOCK_TEXT_MODEL, engine)) throw std::runtime_error(loader.last_error());
        const ModelConfig& config = loader.config();
        builder.allocate_gemma4_scratch(engine, config, loader.gemma4_config(), arch, 1, PREFILL_CHUNK);
        if (!cache.allocate(loader.gemma4_config(), config.num_key_value_heads(), 1, max_seq, PREFILL_CHUNK,
                            nullptr, first_layer, end_layer)) {
            throw std::runtime_error("no pude reservar la caché KV");
        }
        cache.register_tensors(engine, "_kv");
    }
};

ArchDescriptor make_arch(uint32_t layers) {
    // detect_architecture necesita ver todas las capas, y aquí cada motor tiene las suyas.
    ArchDescriptor arch;
    arch.prefix = "text";
    arch.num_layers = layers;
    arch.embedding_name = "token_embedding.weight";
    arch.final_norm_name = "final_norm.weight";
    return arch;
}

int32_t argmax(const std::vector<half>& logits) {
    int32_t best = 0;
    float top = -INFINITY;
    for (size_t i = 0; i < logits.size(); ++i) {
        const float value = __half2float(logits[i]);
        if (value > top) { top = value; best = int32_t(i); }
    }
    return best;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

template <typename T>
void write_raw(const std::filesystem::path& path, const std::vector<T>& values) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(T)));
    if (!file) throw std::runtime_error("no escribo " + path.string());
}

template <typename T>
std::vector<T> read_tensor(Engine& engine, const char* name, size_t count) {
    engine.sync();
    std::vector<T> values(count);
    check(cudaMemcpy(values.data(), engine.tensors().at(name).ptr, count * sizeof(T), cudaMemcpyDeviceToHost), name);
    return values;
}

std::vector<float> read_fp16(Engine& engine, const char* name, size_t count) {
    const auto raw = read_tensor<half>(engine, name, count);
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) values[i] = __half2float(raw[i]);
    return values;
}

// Acumuladores de una mitad del corpus. S, T y las columnas son [capa, experto, 704].
struct Moments {
    uint32_t layers, experts, inner, shared;
    std::vector<double> S, T, U, V, shared_S;
    std::vector<uint64_t> n, shared_n;

    Moments(uint32_t layer_count, uint32_t E, uint32_t I, uint32_t shared_cols)
        : layers(layer_count), experts(E), inner(I), shared(shared_cols),
          S(size_t(layer_count) * E * I, 0.0), T(S.size(), 0.0),
          U(size_t(layer_count) * E, 0.0), V(U.size(), 0.0),
          shared_S(size_t(layer_count) * shared_cols, 0.0), n(U.size(), 0), shared_n(layer_count, 0) {}

    void add_expert(uint32_t layer, int32_t expert, const float* a, float weight) {
        const size_t slot = size_t(layer) * experts + size_t(expert);
        const double w2 = double(weight) * weight;
        double* s = S.data() + slot * inner;
        double* t = T.data() + slot * inner;
        for (uint32_t j = 0; j < inner; ++j) {
            const double square = double(a[j]) * a[j];
            s[j] += square;
            t[j] += w2 * square;
        }
        U[slot] += w2;
        V[slot] += w2 * w2;
        ++n[slot];
    }

    void add_shared(uint32_t layer, const float* a) {
        double* s = shared_S.data() + size_t(layer) * shared;
        for (uint32_t j = 0; j < shared; ++j) s[j] += double(a[j]) * a[j];
        ++shared_n[layer];
    }

    void save(const std::filesystem::path& dir) const {
        std::filesystem::create_directories(dir);
        write_raw(dir / "S.f64", S);
        write_raw(dir / "T.f64", T);
        write_raw(dir / "U.f64", U);
        write_raw(dir / "V.f64", V);
        write_raw(dir / "n.u64", n);
        write_raw(dir / "shared_S.f64", shared_S);
        write_raw(dir / "shared_n.u64", shared_n);
    }
};

struct ControlLayer {
    std::vector<float> residual, weights, logits, down_input;
    std::vector<int32_t> ids;
};

struct Document {
    std::string name;
    std::vector<int32_t> ids;
};

std::vector<Document> read_corpus(const std::filesystem::path& dir, const HTFTokenizer& tokenizer,
                                  uint32_t max_tokens) {
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) paths.push_back(entry.path());
    std::sort(paths.begin(), paths.end());
    const int32_t bos = tokenizer.bos_token_id().value_or(2);
    std::vector<Document> documents;
    for (const auto& path : paths) {
        std::ifstream file(path, std::ios::binary);
        Document document{path.filename().string(), {}};
        if (path.extension() == ".ids") {
            const auto bytes = std::filesystem::file_size(path);
            document.ids.resize(bytes / sizeof(int32_t));
            file.read(reinterpret_cast<char*>(document.ids.data()), std::streamsize(bytes));
        } else if (path.extension() == ".txt") {
            std::stringstream text;
            text << file.rdbuf();
            document.ids = tokenizer.encode(format_gemma4_chat({{"user", text.str()}}), false, false);
        } else {
            continue;
        }
        if (document.ids.empty()) continue;
        if (document.ids.front() != bos) document.ids.insert(document.ids.begin(), bos);
        if (document.ids.size() > max_tokens) document.ids.resize(max_tokens);
        documents.push_back(std::move(document));
    }
    return documents;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "uso: gemma4_moe_generate modelo.hnf \"pregunta\" [--max N] [--ctx N] [--boundary N]\n";
        return 2;
    }
    const std::string path = argv[1];
    std::string prompt, prompts_file, corpus_dir, out_dir;
    uint32_t max_tokens = 128, max_seq = 4096, seed = 26;
    uint64_t limit = 64000;
    int forced_boundary = -1;
    int first_flag = 2;
    if (std::strncmp(argv[2], "--", 2) != 0) {
        prompt = argv[2];
        first_flag = 3;
    }
    for (int i = first_flag; i + 1 < argc; i += 2) {
        const std::string flag = argv[i];
        if (flag == "--max") max_tokens = uint32_t(std::atoi(argv[i + 1]));
        else if (flag == "--ctx") max_seq = uint32_t(std::atoi(argv[i + 1]));
        else if (flag == "--boundary") forced_boundary = std::atoi(argv[i + 1]);
        else if (flag == "--prompts") prompts_file = argv[i + 1];
        else if (flag == "--capture") corpus_dir = argv[i + 1];
        else if (flag == "--out") out_dir = argv[i + 1];
        else if (flag == "--limit") limit = std::strtoull(argv[i + 1], nullptr, 10);
        else if (flag == "--seed") seed = uint32_t(std::atoi(argv[i + 1]));
    }
    try {
        HnfLoader meta;
        if (!meta.load_metadata(path)) throw std::runtime_error(meta.last_error());
        const Gemma4Config& gemma = meta.gemma4_config();
        const ModelConfig& config = meta.config();
        const uint32_t layers = config.num_hidden_layers();

        // Pesos más KV de cada capa, para que el reparto no se quede corto al reservarlos.
        std::vector<uint64_t> layer_bytes(layers, 0);
        uint64_t common_bytes = 0;
        for (const TensorEntry& tensor : meta.tensors()) {
            const int layer = layer_of(tensor.name);
            (layer < 0 ? common_bytes : layer_bytes[layer]) += tensor.size;
        }
        for (uint32_t i = 0; i < layers; ++i) {
            const Gemma4LayerConfig& layer = gemma.layers[i];
            const uint64_t slots = layer.is_global_attention() ? max_seq : layer.sliding_window + PREFILL_CHUNK;
            layer_bytes[i] += 2 * sizeof(half) * slots * layer.kv_heads_or(config.num_key_value_heads()) * layer.head_dim;
        }

        int devices = 0;
        check(cudaGetDeviceCount(&devices), "contar GPUs");
        const uint64_t first_free = free_vram(0) - VRAM_RESERVE;
        const uint64_t second_free = devices > 1 ? free_vram(1) - VRAM_RESERVE : 0;
        const auto plan = plan_gemma4_split(layer_bytes, common_bytes, first_free, second_free);
        if (!plan && forced_boundary < 0) throw std::runtime_error("el modelo no cabe en las GPUs visibles");
        const uint32_t boundary = forced_boundary >= 0 ? uint32_t(forced_boundary) : plan->boundary;
        if (boundary == 0 || boundary >= layers || devices < 2) {
            throw std::runtime_error("esta herramienta necesita capas en dos GPUs");
        }
        std::cerr << "reparto: capas 0-" << boundary - 1 << " en GPU 0, " << boundary << "-" << layers - 1
                  << " en GPU 1" << std::endl;

        const ArchDescriptor arch = make_arch(layers);
        auto start = std::chrono::steady_clock::now();
        Device secondary(1, path, boundary, layers, false, arch, max_seq);
        Device primary(0, path, 0, boundary, true, arch, max_seq);
        std::cerr << "pesos en VRAM en " << seconds_since(start) << " s" << std::endl;

        const HTFTokenizer* tokenizer = primary.loader.tokenizer("text");
        if (!tokenizer) throw std::runtime_error("el HNF no trae tokenizer");
        const auto turn_end = tokenizer->token_to_id("<turn|>");
        const int32_t eos = tokenizer->eos_token_id().value_or(1);
        primary.engine.activate();
        void* tokens = primary.engine.tensors().allocate_and_register("ids", {1, PREFILL_CHUNK}, dtype::INT32(), true);
        Gemma4SplitForward split(primary.engine, secondary.engine, HIDDEN);

        const auto build = [&](uint32_t seq, uint32_t position, auto&& use) {
            primary.engine.tensors().at("ids").shape = {1, seq};
            const KVCacheParams cache{"_kv", position, max_seq};
            secondary.engine.activate();
            const auto rest = secondary.builder.build_gemma4_layers_cached(
                secondary.engine, config, gemma, arch, boundary, layers, 1, seq, cache);
            primary.engine.activate();
            auto first = primary.builder.build_gemma4_input(primary.engine, config, gemma, arch, "ids", 1, seq);
            first.append(primary.builder.build_gemma4_layers_cached(
                primary.engine, config, gemma, arch, 0, boundary, 1, seq, cache));
            const auto head = primary.builder.build_gemma4_head(config, gemma, arch, seq);
            use(Gemma4SplitForward::Parts{first, rest, head});
        };
        std::vector<half> logits(config.vocab_size());
        const auto next_token = [&]() {
            primary.engine.activate();
            primary.engine.sync();
            check(cudaMemcpy(logits.data(), primary.builder.get_logits(primary.engine)->ptr,
                             logits.size() * sizeof(half), cudaMemcpyDeviceToHost), "leer logits");
            return argmax(logits);
        };

        const auto reset_graphs = [&]() {
            secondary.engine.activate();
            secondary.engine.invalidate_graph();
            primary.engine.activate();
            primary.engine.invalidate_graph();
        };
        const auto generate = [&](const std::vector<int32_t>& ids, bool show) {
            reset_graphs();
            const auto prefill_start = std::chrono::steady_clock::now();
            uint32_t position = 0;
            int32_t next = 0;
            while (position < ids.size()) {
                const uint32_t count = std::min(PREFILL_CHUNK, uint32_t(ids.size()) - position);
                check(cudaMemcpy(tokens, ids.data() + position, count * sizeof(int32_t), cudaMemcpyHostToDevice), "tokens");
                split.update_device_cache_pos(int32_t(position), int32_t(count));
                build(count, position, [&](const Gemma4SplitForward::Parts& parts) { split.run(parts); });
                position += count;
                next = next_token();
            }
            if (show) std::cerr << "prefill: " << seconds_since(prefill_start) * 1000 << " ms" << std::endl;

            std::vector<double> token_ms;
            std::vector<int32_t> output;
            build(1, position, [&](const Gemma4SplitForward::Parts& parts) {
                for (uint32_t step = 0; step < max_tokens; ++step) {
                    if (next == eos || (turn_end && next == *turn_end) || position >= max_seq) break;
                    output.push_back(next);
                    if (show) std::cout << tokenizer->decode({next}) << std::flush;
                    const auto token_start = std::chrono::steady_clock::now();
                    primary.engine.activate();
                    check(cudaMemcpy(tokens, &next, sizeof(next), cudaMemcpyHostToDevice), "token");
                    split.update_device_cache_pos(int32_t(position), 1);
                    if (step == 0) split.run(parts);
                    else split.replay(parts);
                    next = next_token();
                    ++position;
                    token_ms.push_back(seconds_since(token_start) * 1000);
                }
            });
            if (show) {
                std::cout << std::endl;
                if (token_ms.size() > 4) {
                    std::vector<double> sorted(token_ms.begin() + 2, token_ms.end());
                    std::sort(sorted.begin(), sorted.end());
                    const double median = sorted[sorted.size() / 2];
                    std::cerr << output.size() << " tokens, mediana " << median << " ms, " << 1000 / median
                              << " tok/s, grafos " << split.graphs_ready() << std::endl;
                }
            }
            return output;
        };

        if (!prompts_file.empty()) {
            std::filesystem::create_directories(out_dir);
            std::ifstream file(prompts_file);
            std::string line;
            int index = 0;
            size_t total = 0;
            start = std::chrono::steady_clock::now();
            while (std::getline(file, line)) {
                if (line.empty()) continue;
                auto ids = tokenizer->encode(format_gemma4_chat({{"user", line}}), false, false);
                const auto output = generate(ids, false);
                ids.insert(ids.end(), output.begin(), output.end());
                if (turn_end) ids.push_back(*turn_end);
                char name[64];
                std::snprintf(name, sizeof(name), "generated-%04d.ids", index++);
                write_raw(std::filesystem::path(out_dir) / name, ids);
                total += ids.size();
            }
            std::cerr << index << " conversaciones, " << total << " tokens en " << seconds_since(start) << " s" << std::endl;
        } else if (!corpus_dir.empty()) {
            const uint32_t E = gemma.moe_experts, K = gemma.moe_top_k, I = gemma.moe_intermediate_size;
            const uint32_t H = config.hidden_size();
            const uint32_t SHARED = gemma.layers[0].intermediate_size;
            const uint32_t PHYSICAL = gemma.layers[0].moe_down.physical_cols;
            const uint32_t SHARED_PHYSICAL = gemma.layers[0].moe_shared_down.physical_cols;
            const std::vector<uint32_t> control_layers{0, 5, 15, 29};
            constexpr uint32_t CONTROL_TOKENS = 256;

            auto documents = read_corpus(corpus_dir, *tokenizer, max_seq - 1);
            std::mt19937 rng(seed);
            std::shuffle(documents.begin(), documents.end(), rng);
            std::vector<const Document*> train, held_out;
            for (size_t i = 0; i < documents.size(); ++i) (i % 5 == 4 ? held_out : train).push_back(&documents[i]);
            // Tokens de control repartidos por los documentos reservados.
            uint64_t held_tokens = 0;
            for (const Document* document : held_out) held_tokens += document->ids.size();
            const uint64_t control_step = std::max<uint64_t>(1, held_tokens / CONTROL_TOKENS);

            // Una sola construcción: la posición va en el dispositivo y 1024 elige flash.
            const KVCacheParams cache{"_kv", 1024, max_seq};
            primary.engine.tensors().at("ids").shape = {1, 1};
            secondary.engine.activate();
            std::vector<CommandBuffer> layer_commands(layers);
            for (uint32_t l = boundary; l < layers; ++l) {
                layer_commands[l] = secondary.builder.build_gemma4_layer_cached(
                    secondary.engine, config, gemma, arch, l, 1, 1, cache);
            }
            primary.engine.activate();
            const auto input = primary.builder.build_gemma4_input(primary.engine, config, gemma, arch, "ids", 1, 1);
            for (uint32_t l = 0; l < boundary; ++l) {
                layer_commands[l] = primary.builder.build_gemma4_layer_cached(
                    primary.engine, config, gemma, arch, l, 1, 1, cache);
            }

            Moments half_a(layers, E, I, SHARED), half_b(layers, E, I, SHARED);
            std::map<uint32_t, ControlLayer> control;
            std::ofstream control_index;
            std::filesystem::create_directories(out_dir);
            control_index.open(std::filesystem::path(out_dir) / "control-tokens.tsv");
            control_index << "document\tposition\ttoken\n";

            // Pasa un token por todas las capas sin cabeza: el texto ya está dado.
            const auto run_token = [&](int32_t token, uint32_t position, Moments* moments, bool keep_control) {
                primary.engine.activate();
                check(cudaMemcpy(tokens, &token, sizeof(token), cudaMemcpyHostToDevice), "token");
                split.update_device_cache_pos(int32_t(position), 1);
                primary.engine.execute(input);
                for (uint32_t l = 0; l < layers; ++l) {
                    if (l == boundary) split.cross(primary.engine, secondary.engine);
                    Device& device = l < boundary ? primary : secondary;
                    const bool sample = keep_control &&
                        std::find(control_layers.begin(), control_layers.end(), l) != control_layers.end();
                    for (const Command& command : layer_commands[l].commands()) {
                        // La primera rmsnorm de la MLP lee el residual tras la atención.
                        if (sample && command.op == op::RMSNORM() && command.output == "_s.normed" &&
                            command.inputs.size() > 1 && command.inputs[1].find("ln_mlp_in") != std::string::npos) {
                            const auto residual = read_fp16(device.engine, HIDDEN, H);
                            auto& all = control[l].residual;
                            all.insert(all.end(), residual.begin(), residual.end());
                        }
                        device.engine.execute_command(command);
                        if (moments && command.op == op::PAD_COLUMNS()) {
                            const auto h = read_fp16(device.engine, "_s.g4.mlp_h_padded", SHARED_PHYSICAL);
                            moments->add_shared(l, h.data());
                        }
                        if (command.op != op::MOE_EXPERTS_DECODE() || (!moments && !sample)) continue;
                        const auto ids = read_tensor<int32_t>(device.engine, "_s.g4.moe_indices", K);
                        const auto weights = read_tensor<float>(device.engine, "_s.g4.moe_weights", K);
                        const auto hidden = read_fp16(device.engine, "_s.g4.moe_hidden", size_t(K) * PHYSICAL);
                        if (moments) {
                            for (uint32_t k = 0; k < K; ++k) {
                                moments->add_expert(l, ids[k], hidden.data() + size_t(k) * PHYSICAL, weights[k]);
                            }
                        }
                        if (sample) {
                            ControlLayer& layer = control[l];
                            layer.ids.insert(layer.ids.end(), ids.begin(), ids.end());
                            layer.weights.insert(layer.weights.end(), weights.begin(), weights.end());
                            const auto logits = read_tensor<float>(device.engine, "_s.g4.moe_logits", E);
                            layer.logits.insert(layer.logits.end(), logits.begin(), logits.end());
                            for (uint32_t k = 0; k < K; ++k) {
                                const float* a = hidden.data() + size_t(k) * PHYSICAL;
                                layer.down_input.insert(layer.down_input.end(), a, a + I);
                            }
                        }
                    }
                }
                secondary.engine.activate();
                secondary.engine.sync();
                primary.engine.activate();
            };

            uint64_t processed = 0;
            start = std::chrono::steady_clock::now();
            for (size_t i = 0; i < train.size() && processed < limit; ++i) {
                Moments* moments = i % 2 == 0 ? &half_a : &half_b;
                for (uint32_t position = 0; position < train[i]->ids.size() && processed < limit; ++position) {
                    run_token(train[i]->ids[position], position, moments, false);
                    ++processed;
                    if (processed % 2000 == 0) {
                        std::cerr << processed << " tokens, " << processed / seconds_since(start) << " tok/s" << std::endl;
                    }
                }
            }
            const double capture_seconds = seconds_since(start);
            half_a.save(std::filesystem::path(out_dir) / "half-a");
            half_b.save(std::filesystem::path(out_dir) / "half-b");

            uint64_t seen = 0, kept = 0;
            for (const Document* document : held_out) {
                for (uint32_t position = 0; position < document->ids.size(); ++position, ++seen) {
                    const bool keep = seen % control_step == 0 && kept < CONTROL_TOKENS;
                    run_token(document->ids[position], position, nullptr, keep);
                    if (keep) {
                        control_index << document->name << '\t' << position << '\t' << document->ids[position] << '\n';
                        ++kept;
                    }
                }
            }
            for (const auto& [layer, data] : control) {
                const auto dir = std::filesystem::path(out_dir) / ("control-layer-" + std::to_string(layer));
                std::filesystem::create_directories(dir);
                write_raw(dir / "residual.f32", data.residual);
                write_raw(dir / "expert_ids.i32", data.ids);
                write_raw(dir / "expert_weights.f32", data.weights);
                write_raw(dir / "router_logits.f32", data.logits);
                write_raw(dir / "down_input.f32", data.down_input);
            }
            std::ofstream documents_file(std::filesystem::path(out_dir) / "documents.tsv");
            documents_file << "document\ttokens\tsplit\n";
            for (size_t i = 0; i < train.size(); ++i) {
                documents_file << train[i]->name << '\t' << train[i]->ids.size() << '\t' << (i % 2 == 0 ? "half-a" : "half-b") << '\n';
            }
            for (const Document* document : held_out) {
                documents_file << document->name << '\t' << document->ids.size() << "\tcontrol\n";
            }
            std::cerr << processed << " tokens de ajuste en " << capture_seconds << " s, " << kept
                      << " estados de control en " << control.size() << " capas" << std::endl;
        } else {
            generate(tokenizer->encode(format_gemma4_chat({{"user", prompt}}), false, false), true);
        }

        secondary.engine.activate();
        secondary.engine.invalidate_graph();
        primary.engine.activate();
        primary.engine.invalidate_graph();
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
