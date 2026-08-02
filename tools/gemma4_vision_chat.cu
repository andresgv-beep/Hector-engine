#include "chat_template.hpp"
#include "gemma4_kv_cache.hpp"
#include "gemma4_multimodal.hpp"
#include "gemma4_vision_preprocess.hpp"
#include "gemma4_vision_runner.hpp"
#include "graph_builder.hpp"
#include "hnf_loader.hpp"
#include "kernels.hpp"
#include "sampler.hpp"

#include <png.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct RgbImage {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> pixels;
};

struct CudaStreamOwner {
    cudaStream_t value = nullptr;
    CudaStreamOwner() {
        const cudaError_t status =
            cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("create V6 CUDA stream: ") +
                                     cudaGetErrorString(status));
        }
    }
    ~CudaStreamOwner() {
        if (value) cudaStreamDestroy(value);
    }
    CudaStreamOwner(const CudaStreamOwner&) = delete;
    CudaStreamOwner& operator=(const CudaStreamOwner&) = delete;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_require(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double mib(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

double delta_mib(size_t current, size_t baseline) {
    return (static_cast<double>(current) - static_cast<double>(baseline)) /
           (1024.0 * 1024.0);
}

size_t used_vram() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cuda_require(cudaMemGetInfo(&free_bytes, &total_bytes), "sample VRAM");
    return total_bytes - free_bytes;
}

RgbImage decode_png_rgb(const std::string& path) {
    std::ifstream signature(path, std::ios::binary);
    require(signature.good(), "no se puede abrir la imagen: " + path);
    png_byte header[8]{};
    signature.read(reinterpret_cast<char*>(header), sizeof(header));
    require(signature.gcount() == static_cast<std::streamsize>(sizeof(header)) &&
            png_sig_cmp(header, 0, sizeof(header)) == 0,
            "V6 acepta PNG; formato no reconocido: " + path);

    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    require(png_image_begin_read_from_file(&image, path.c_str()) != 0,
            "PNG inválido: " + std::string(image.message));
    struct Guard {
        png_image* image;
        ~Guard() { png_image_free(image); }
    } guard{&image};

    constexpr uint64_t kMaxPixels = 100'000'000;
    require(image.width != 0 && image.height != 0 &&
            uint64_t(image.width) * image.height <= kMaxPixels,
            "PNG vacío o superior al límite de 100 megapíxeles");
    image.format = PNG_FORMAT_RGBA;
    const png_alloc_size_t rgba_size = PNG_IMAGE_SIZE(image);
    require(rgba_size <= std::numeric_limits<size_t>::max(),
            "PNG demasiado grande para esta plataforma");
    std::vector<uint8_t> rgba(static_cast<size_t>(rgba_size));
    require(png_image_finish_read(&image, nullptr, rgba.data(), 0, nullptr) != 0,
            "no se pudo decodificar PNG: " + std::string(image.message));

    const size_t pixel_count = size_t(image.width) * image.height;
    RgbImage result;
    result.width = image.width;
    result.height = image.height;
    result.pixels.resize(pixel_count * 3);
    for (size_t i = 0; i < pixel_count; ++i) {
        result.pixels[i * 3 + 0] = rgba[i * 4 + 0];
        result.pixels[i * 3 + 1] = rgba[i * 4 + 1];
        result.pixels[i * 3 + 2] = rgba[i * 4 + 2];
    }
    return result;
}

void upload_i32(helios::Engine& engine, const std::string& name,
                const std::vector<int32_t>& values,
                const std::vector<uint32_t>& shape) {
    require(!values.empty(), "no se puede subir un tensor INT32 vacío");
    void* device = engine.tensors().allocate_and_register(
        name, shape, helios::dtype::INT32());
    cuda_require(cudaMemcpy(device, values.data(), values.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice), "upload INT32 tensor");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        std::cerr << "Uso: " << argv[0]
                  << " <modelo-combinado.hnf> <imagen.png> <prompt>"
                     " [max_tokens=64] [temperatura=0]\n";
        return 2;
    }

    try {
        const std::string hnf_path = argv[1];
        const std::string image_path = argv[2];
        const std::string prompt = argv[3];
        const char* followup_env = std::getenv("HELIOS_VISION_FOLLOWUP");
        const std::string followup = followup_env ? followup_env : "";
        const char* vision_mmap_env = std::getenv("HELIOS_VISION_MMAP");
        const bool vision_mmap = vision_mmap_env && vision_mmap_env[0] == '1';
        const char* double_buffer_env =
            std::getenv("HELIOS_MMAP_DOUBLE_BUFFER");
        const bool double_buffer =
            double_buffer_env && double_buffer_env[0] == '1';
        uint32_t vision_repeats = 1;
        if (const char* repeat_env = std::getenv("HELIOS_VISION_REPEAT")) {
            const uint64_t requested_repeats = std::stoull(repeat_env);
            require(requested_repeats >= 1 && requested_repeats <= 16,
                    "HELIOS_VISION_REPEAT debe estar entre 1 y 16");
            vision_repeats = static_cast<uint32_t>(requested_repeats);
        }
        const uint64_t requested_tokens = argc >= 5 ? std::stoull(argv[4]) : 64;
        require(requested_tokens <= std::numeric_limits<uint32_t>::max(),
                "max_tokens desborda uint32");
        const uint32_t max_tokens = static_cast<uint32_t>(requested_tokens);
        const float temperature = argc >= 6 ? std::stof(argv[5]) : 0.0f;
        require(!prompt.empty(), "el prompt no puede estar vacío");
        require(max_tokens > 0 && max_tokens <= 4096,
                "max_tokens debe estar entre 1 y 4096");
        require(temperature >= 0.0f && temperature <= 5.0f,
                "temperatura fuera de rango [0,5]");

        const auto total_start = Clock::now();
        const auto decode_start = Clock::now();
        const RgbImage image = decode_png_rgb(image_path);
        const auto decode_end = Clock::now();

        helios::HnfLoader loader;
        require(loader.open(hnf_path), "no se puede abrir HNF: " + loader.last_error());
        require(loader.has_gemma4_config() && loader.has_gemma4_vision_config(),
                "V6 necesita un HNF combinado con GM4X y GM4V");
        const auto* tokenizer = loader.tokenizer("text");
        require(tokenizer != nullptr, "el HNF combinado no contiene tokenizer HTF");
        const auto& vision_config = loader.gemma4_vision_config();
        require(tokenizer->token_to_id("<|image|>") == vision_config.image_token_id,
                "tokenizer y GM4V discrepan en IMAGE");

        helios::Gemma4VisionPreprocessConfig preprocess_config;
        preprocess_config.patch_size = vision_config.patch_size;
        preprocess_config.pooling_kernel_size = vision_config.pooling_kernel_size;
        preprocess_config.max_soft_tokens = vision_config.max_soft_tokens;
        preprocess_config.rescale_factor = vision_config.rescale_factor;
        helios::Gemma4VisionPreprocessResult preprocessed;
        std::string error;
        const auto preprocess_start = Clock::now();
        require(helios::gemma4_vision_preprocess_rgb(
                    {image.pixels.data(), image.width, image.height, size_t(image.width) * 3},
                    preprocess_config, preprocessed, &error),
                "preprocesado visual: " + error);
        const auto preprocess_end = Clock::now();

        // Token geometry is known after V3, before executing the tower. This
        // lets V7 reserve the real heterogeneous KV while vision runs, proving
        // that no decoder state needs to be evicted for a later image.
        const uint32_t expected_visual_tokens = preprocessed.soft_tokens;
        const std::string user_content = "<|image|>\n" + prompt;
        const std::string formatted = helios::format_gemma4_chat(
            {{"user", user_content}});
        const std::vector<int32_t> raw_ids = tokenizer->encode(
            formatted, false, false);
        const auto plan = helios::make_gemma4_multimodal_token_plan(
            raw_ids, vision_config, loader.config().vocab_size(),
            expected_visual_tokens);
        require(plan.canonical_ids.size() <= std::numeric_limits<uint32_t>::max(),
                "secuencia multimodal demasiado larga");
        const uint32_t sequence = static_cast<uint32_t>(plan.canonical_ids.size());
        std::vector<int32_t> followup_ids;
        if (!followup.empty()) {
            const std::string fragment =
                "<turn|>\n<|turn>user\n" + followup +
                "<turn|>\n<|turn>model\n";
            followup_ids = tokenizer->encode(fragment, false, false);
            require(!followup_ids.empty() &&
                    std::find(followup_ids.begin(), followup_ids.end(),
                              vision_config.image_token_id) == followup_ids.end(),
                    "el segundo turno V6 no admite otra imagen");
        }
        const auto model_config = loader.config();
        const auto& gemma = loader.gemma4_config();
        const uint64_t cache_required = uint64_t(sequence) +
            uint64_t(max_tokens) * (followup.empty() ? 1u : 2u) +
            followup_ids.size();
        uint64_t cache_capacity = cache_required;
        if (const char* context_env = std::getenv("HELIOS_CTX")) {
            const uint64_t requested_context = std::stoull(context_env);
            const uint64_t model_limit = model_config.get<uint32_t>(
                "max_position_embeddings", 0);
            require(requested_context >= 1 &&
                    (!model_limit || requested_context <= model_limit),
                    "HELIOS_CTX está fuera del límite del modelo");
            cache_capacity = std::max(cache_capacity, requested_context);
        }
        require(cache_capacity <= std::numeric_limits<uint32_t>::max(),
                "contexto multimodal demasiado largo");
        const uint32_t max_cache = static_cast<uint32_t>(cache_capacity);
        require(max_cache <= static_cast<uint32_t>(std::numeric_limits<int32_t>::max()),
                "contexto multimodal excede el contador INT32 del decode");

        CudaStreamOwner stream;
        helios::EngineConfig engine_config;
        engine_config.scratch_pool.auto_fraction = 0.0f;
        engine_config.stream = stream.value;
        helios::Engine engine(engine_config);
        helios::kernels::register_all_kernels(engine);
        const size_t vram_initial = used_vram();
        helios::Gemma4KVCache cache;
        bool cache_allocated = false;
        auto ensure_cache = [&]() {
            if (cache_allocated) return;
            require(cache.allocate(gemma, model_config.num_key_value_heads(),
                                   1, max_cache),
                    "no se pudo reservar el KV heterogéneo");
            cache.register_tensors(engine, "_g4v6kv");
            cache_allocated = true;
        };

        // V7 mmap proves the final residency model: text is loaded first and
        // remains untouched while the GPU executes the file-backed tower.
        // The legacy V6 control deliberately preserves its old ordering.
        if (!std::getenv("HELIOS_EMBED_MMAP")) {
            setenv("HELIOS_EMBED_MMAP", "1", 0);
        }
        bool text_loaded = false;
        size_t vram_text_weights = vram_initial;
        if (vision_mmap) {
            require(loader.load_block(helios::BLOCK_TEXT_MODEL, engine),
                    "no se puede cargar texto antes de visión: " +
                    loader.last_error());
            text_loaded = true;
            vram_text_weights = used_vram();
            ensure_cache();
        }

        const size_t vram_before_vision = used_vram();
        const auto vision_load_start = Clock::now();
        require(loader.load_block(helios::BLOCK_VISION, engine),
                "no se puede cargar visión: " + loader.last_error());
        const auto vision_load_end = Clock::now();
        const size_t vram_vision_weights = used_vram();
        uint32_t visual_tokens = 0;
        size_t vram_vision_peak = 0;
        std::vector<double> tower_times;
        tower_times.reserve(vision_repeats);
        for (uint32_t repeat = 0; repeat < vision_repeats; ++repeat) {
            const auto tower_start = Clock::now();
            {
                helios::Gemma4VisionRunner runner(engine, loader);
                require(runner.run_patch_embedder(preprocessed, &error),
                        "patch embedder: " + error);
                for (uint32_t layer = 0; layer < runner.num_layers(); ++layer) {
                    require(runner.run_encoder_layer(layer, &error),
                            "capa visual " + std::to_string(layer) + ": " + error);
                }
                require(runner.run_pooler(&error), "pooler visual: " + error);
                require(runner.run_projector(&error), "proyector visual: " + error);
                engine.sync();
                visual_tokens = runner.soft_token_count();
                require(visual_tokens == preprocessed.soft_tokens &&
                        runner.projection_size() == loader.config().hidden_size(),
                        "la salida visual no coincide con el contrato de texto");
                if (repeat + 1 == vision_repeats) {
                    void* persistent = engine.tensors().allocate_and_register(
                        "g4.v6.image", {visual_tokens, runner.projection_size()},
                        helios::dtype::FP16());
                    cuda_require(cudaMemcpyAsync(
                        persistent, runner.projected_states_device(),
                        size_t(visual_tokens) * runner.projection_size() * sizeof(half),
                        cudaMemcpyDeviceToDevice, engine.config().stream),
                        "persist visual projection");
                    engine.sync();
                }
                vram_vision_peak = std::max(vram_vision_peak, used_vram());
            }
            const auto tower_end = Clock::now();
            tower_times.push_back(milliseconds(tower_start, tower_end));
        }
        if (!vision_mmap) {
            require(loader.unload_block(helios::BLOCK_VISION, engine),
                    "no se pudo descargar el bloque visual");
        }
        const size_t vram_after_vision = used_vram();

        if (!text_loaded) {
            require(loader.load_block(helios::BLOCK_TEXT_MODEL, engine),
                    "no se puede cargar texto: " + loader.last_error());
            text_loaded = true;
            vram_text_weights = used_vram();
        }
        ensure_cache();

        upload_i32(engine, "g4.v6.embedding_tokens", plan.embedding_ids,
                   {1, sequence});
        upload_i32(engine, "g4.v6.ple_tokens", plan.ple_identity_ids,
                   {1, sequence});
        upload_i32(engine, "g4.v6.positions", plan.image_positions,
                   {visual_tokens});
        upload_i32(engine, "g4.v6.decode_token", {0}, {1, 1});
        if (!followup_ids.empty()) {
            upload_i32(engine, "g4.v6.followup", followup_ids,
                       {1, static_cast<uint32_t>(followup_ids.size())});
        }

        helios::GraphBuilder builder;
        const auto arch = builder.detect_architecture(engine, "text", model_config);
        const uint32_t scratch_sequence = std::max(
            sequence, static_cast<uint32_t>(followup_ids.size()));
        builder.allocate_gemma4_scratch(
            engine, model_config, gemma, arch, 1, scratch_sequence);
        const size_t vram_ready = used_vram();

        const helios::Gemma4MultimodalInputNames names{
            "g4.v6.embedding_tokens", "g4.v6.ple_tokens",
            "g4.v6.image", "g4.v6.positions"};
        const auto prefill_start = Clock::now();
        engine.execute(builder.build_gemma4_multimodal_forward_cached(
            engine, model_config, gemma, arch, names, 1, sequence,
            {"_g4v6kv", 0, max_cache}));
        engine.sync();
        const auto prefill_end = Clock::now();
        const size_t vram_prefill_peak = used_vram();

        helios::Sampler sampler;
        sampler.set_seed(7);
        helios::SamplingConfig sampling = helios::SamplingConfig::deterministic();
        if (temperature > 0.0f) {
            sampling.temperature = temperature;
            sampling.top_k = 64;
            sampling.top_p = 0.95f;
        }
        const int32_t eos = tokenizer->eos_token_id().value_or(1);
        const int32_t end_turn = tokenizer->token_to_id("<turn|>").value_or(106);
        auto& decode_tensor = engine.tensors().at("g4.v6.decode_token");
        helios::CommandBuffer decode_commands;
        bool decode_built = false;
        auto generate_turn = [&](uint32_t& cache_position) {
            std::vector<int32_t> generated;
            generated.reserve(max_tokens);
            for (uint32_t step = 0; step < max_tokens; ++step) {
                const auto& logits = engine.tensors().at("_s.logits");
                const int32_t next = sampler.sample(
                    static_cast<const half*>(logits.ptr),
                    static_cast<int>(model_config.vocab_size()), sampling);
                require(next >= 0 &&
                        static_cast<uint32_t>(next) < model_config.vocab_size(),
                        "sampler devolvió un token inválido");
                if (next == eos || next == end_turn) break;
                generated.push_back(next);
                cuda_require(cudaMemcpyAsync(
                    decode_tensor.ptr, &next, sizeof(next), cudaMemcpyHostToDevice,
                    engine.config().stream), "upload decode token");
                engine.update_device_cache_pos(
                    static_cast<int32_t>(cache_position), 1);
                if (!decode_built) {
                    decode_commands = builder.build_gemma4_forward_cached(
                        engine, model_config, gemma, arch,
                        "g4.v6.decode_token", 1, 1,
                        {"_g4v6kv", cache_position, max_cache});
                    engine.execute(decode_commands);
                    decode_built = true;
                } else {
                    engine.execute_graph_replay(decode_commands);
                }
                engine.sync();
                ++cache_position;
            }
            require(!generated.empty(), "la generación terminó sin texto");
            return generated;
        };

        uint32_t cache_position = sequence;
        const auto generation_start = Clock::now();
        const std::vector<int32_t> generated = generate_turn(cache_position);
        const auto generation_end = Clock::now();
        const std::string output = tokenizer->decode(generated);

        double followup_prefill_ms = 0.0;
        double followup_decode_seconds = 0.0;
        std::vector<int32_t> followup_generated;
        std::string followup_output;
        if (!followup_ids.empty()) {
            const auto followup_prefill_start = Clock::now();
            engine.execute(builder.build_gemma4_forward_cached(
                engine, model_config, gemma, arch, "g4.v6.followup", 1,
                static_cast<uint32_t>(followup_ids.size()),
                {"_g4v6kv", cache_position, max_cache}));
            engine.sync();
            const auto followup_prefill_end = Clock::now();
            followup_prefill_ms = milliseconds(
                followup_prefill_start, followup_prefill_end);
            cache_position += static_cast<uint32_t>(followup_ids.size());
            const auto followup_generation_start = Clock::now();
            followup_generated = generate_turn(cache_position);
            const auto followup_generation_end = Clock::now();
            followup_decode_seconds = std::chrono::duration<double>(
                followup_generation_end - followup_generation_start).count();
            followup_output = tokenizer->decode(followup_generated);
        }

        const double generation_seconds =
            std::chrono::duration<double>(generation_end - generation_start).count();
        std::cout << "\n=== GEMMA 4 VISION "
                  << (vision_mmap
                          ? (double_buffer ? "V8 MMAP/DOUBLE BUFFER"
                                           : "V7 MMAP/STAGING")
                          : "V6 VRAM")
                  << " ===\n"
                  << "Imagen: " << image.width << 'x' << image.height
                  << " -> " << preprocessed.resized_width << 'x'
                  << preprocessed.resized_height << '\n'
                  << "Patches: " << preprocessed.real_patches << '/'
                  << preprocessed.max_patches << " · soft tokens: "
                  << visual_tokens << " · secuencia: " << sequence << '\n'
                  << "Decode PNG: " << milliseconds(decode_start, decode_end) << " ms\n"
                  << "Preprocesado: " << milliseconds(preprocess_start, preprocess_end)
                  << " ms\n"
                  << "Carga visión: "
                  << milliseconds(vision_load_start, vision_load_end) << " ms\n"
                  << "Torre visual:";
        for (size_t i = 0; i < tower_times.size(); ++i) {
            std::cout << (i == 0 ? " " : " · ")
                      << (i == 0 ? "fría " : "repetición " + std::to_string(i + 1) + " ")
                      << tower_times[i] << " ms";
        }
        std::cout << "\n"
                  << "Prefill: " << milliseconds(prefill_start, prefill_end) << " ms\n"
                  << "Decode: " << generated.size() << " tokens";
        if (generation_seconds > 0.0) {
            std::cout << " · " << generated.size() / generation_seconds << " tok/s";
        }
        if (!followup_ids.empty()) {
            std::cout << "\nSegundo prefill: " << followup_prefill_ms << " ms"
                      << " · decode: " << followup_generated.size() << " tokens";
            if (followup_decode_seconds > 0.0) {
                std::cout << " · "
                          << followup_generated.size() / followup_decode_seconds
                          << " tok/s";
            }
        }
        std::cout << "\nResidencia: texto "
                  << (vision_mmap ? "antes de visión" : "después de visión")
                  << " · visión " << (vision_mmap ? "HNF/RAM" : "VRAM")
                  << "\nVRAM base: " << mib(vram_initial)
                  << " MiB · texto: +"
                  << delta_mib(vram_text_weights, vram_initial)
                  << " MiB · pesos visión: +"
                  << delta_mib(vram_vision_weights, vram_before_vision)
                  << " MiB · pico torre: +"
                  << delta_mib(vram_vision_peak, vram_before_vision)
                  << " MiB\nVRAM tras torre: +"
                  << delta_mib(vram_after_vision, vram_initial)
                  << " MiB · listo: +" << delta_mib(vram_ready, vram_initial)
                  << " MiB · prefill: +" << delta_mib(vram_prefill_peak, vram_initial)
                  << " MiB\nTiempo total: " << milliseconds(total_start, Clock::now())
                  << " ms\n\nRespuesta:\n" << output << '\n';
        if (!followup_ids.empty()) {
            std::cout << "\nSegundo turno (" << followup << "):\n"
                      << followup_output << '\n';
        }
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "ERROR V6: " << exception.what() << '\n';
        return 1;
    }
}
