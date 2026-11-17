// kernels/qwen35_vision.cu
// ============================================================================
// La torre visual de la familia Qwen3.5 (src/qwen35_vision.hpp)
// ============================================================================
//
// Las lineales y la atención completa son las de siempre (cuBLAS, y la atención de la torre
// de Gemma). Aquí solo van las dos piezas propias: el reparto de qkv en cabezas con la
// rotación en dos ejes, y el sesgo con su activación. Y el recorrido de la torre, que lee los
// pesos del fichero capa a capa.

#include "kernels.hpp"
#include "qwen35_vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <stdexcept>

namespace helios {

namespace {

constexpr int BLOCK = 256;
unsigned blocks_for(size_t count) { return unsigned((count + BLOCK - 1) / BLOCK); }

enum Activation { NONE = 0, GELU_TANH = 1, GELU = 2 };

// x += sesgo, y la activación si la hay. La torre usa la GELU aproximada en sus bloques y la
// exacta en la fusión.
__global__ void bias_act_kernel(half* __restrict__ x, const half* __restrict__ bias, size_t count, int dim, int activation) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = __half2float(x[i]) + __half2float(bias[i % dim]);
    if (activation == GELU_TANH) {
        v = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    } else if (activation == GELU) {
        v = 0.5f * v * (1.0f + erff(v * 0.7071067812f));
    }
    x[i] = __float2half(v);
}

// qkv sale por token como [3, cabezas, dim]. La atención los quiere por cabeza, [cabezas,
// tokens, dim], y con consulta y clave rotadas. La rotación es en dos ejes sobre toda la
// dimensión: de cada mitad, el primer cuarto gira con la fila del parche y el segundo con
// su columna, con las mismas frecuencias.
__global__ void split_rope_kernel(
    const half* __restrict__ qkv, const int32_t* __restrict__ row_column,
    half* __restrict__ q, half* __restrict__ k, half* __restrict__ v,
    int tokens, int heads, int dim, float theta
) {
    const int half_dim = dim / 2, quarter = dim / 4;
    const size_t at = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= size_t(tokens) * heads * half_dim) return;
    const int i = int(at % half_dim);
    const int head = int((at / half_dim) % heads);
    const int token = int(at / (size_t(half_dim) * heads));
    const bool by_row = i < quarter;
    const int frequency = by_row ? i : i - quarter;
    const float angle = float(row_column[2 * token + (by_row ? 0 : 1)]) *
                        powf(theta, -float(2 * frequency) / float(half_dim));
    const float c = cosf(angle), s = sinf(angle);
    const size_t out = (size_t(head) * tokens + token) * dim;
    const half* src = qkv + (size_t(token) * 3 * heads + head) * dim;
    half* const rotated[2] = {q, k};
    for (int part = 0; part < 2; ++part) {
        const half* in = src + size_t(part) * heads * dim;
        const float x0 = __half2float(in[i]), x1 = __half2float(in[i + half_dim]);
        rotated[part][out + i] = __float2half(x0 * c - x1 * s);
        rotated[part][out + i + half_dim] = __float2half(x1 * c + x0 * s);
    }
    const half* in = src + size_t(2) * heads * dim;
    v[out + i] = in[i];
    v[out + i + half_dim] = in[i + half_dim];
}

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("torre visual, ") + what + ": " + cudaGetErrorString(error));
}

template <typename T>
struct Device {
    T* data = nullptr;
    Device() = default;
    explicit Device(size_t count) { reserve(count); }
    void reserve(size_t count) {
        if (cudaMalloc(&data, std::max<size_t>(count, 1) * sizeof(T)) != cudaSuccess) {
            data = nullptr;
            throw std::runtime_error("torre visual: sin memoria de GPU para mirar la imagen");
        }
    }
    ~Device() { if (data) cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

}  // namespace

bool Qwen35VisionConfig::read(const ModelConfig& config) {
    if (config.get<std::string>("vision_family", "") != "qwen3_5") return false;
    const auto number = [&](const char* key) { return uint32_t(config.get<int64_t>(key, 0)); };
    depth = number("vision_depth");
    hidden = number("vision_hidden_size");
    heads = number("vision_num_heads");
    intermediate = number("vision_intermediate_size");
    patch = number("vision_patch_size");
    merge = number("vision_spatial_merge_size");
    temporal = number("vision_temporal_patch_size");
    out_hidden = number("vision_out_hidden_size");
    grid_side = uint32_t(std::lround(std::sqrt(double(number("vision_num_position_embeddings")))));
    image_token = int32_t(config.get<int64_t>("image_token_id", -1));
    start_token = int32_t(config.get<int64_t>("vision_start_token_id", -1));
    end_token = int32_t(config.get<int64_t>("vision_end_token_id", -1));
    if (const char* limit = getenv("HELIOS_VISION_MAX_PARCHES")) {
        const int requested = std::atoi(limit);
        if (requested > 0) max_patches = uint32_t(requested);
    }
    if (merge) max_patches -= max_patches % (merge * merge);
    return depth && hidden && heads && hidden % heads == 0 && (hidden / heads) % 4 == 0 && intermediate && patch &&
           merge && temporal && out_hidden && grid_side && image_token >= 0 && max_patches >= merge * merge;
}

bool qwen35_vision_target_size(uint32_t width, uint32_t height, const Qwen35VisionConfig& config,
                               uint32_t& target_width, uint32_t& target_height, std::string* error) {
    const auto fail = [&](const char* why) { if (error) *error = why; return false; };
    if (width == 0 || height == 0) return fail("imagen sin tamaño");
    if (std::max(width, height) / std::min(width, height) > 200) return fail("imagen demasiado alargada");
    const double factor = double(config.patch) * config.merge;
    const double min_pixels = 256.0 * 256.0;
    const double max_pixels = double(config.max_patches) * config.patch * config.patch;
    double h = std::max(factor, std::round(height / factor) * factor);
    double w = std::max(factor, std::round(width / factor) * factor);
    if (h * w > max_pixels) {
        const double beta = std::sqrt(double(height) * width / max_pixels);
        h = std::max(factor, std::floor(height / beta / factor) * factor);
        w = std::max(factor, std::floor(width / beta / factor) * factor);
    } else if (h * w < min_pixels) {
        const double beta = std::sqrt(min_pixels / (double(height) * width));
        h = std::ceil(height * beta / factor) * factor;
        w = std::ceil(width * beta / factor) * factor;
    }
    target_width = uint32_t(w);
    target_height = uint32_t(h);
    if (uint64_t(target_width / config.patch) * (target_height / config.patch) > config.max_patches) {
        return fail("la imagen no cabe en el tope de parches");
    }
    return true;
}

Qwen35VisionRunner::Qwen35VisionRunner(const HnfLoader& loader, const Qwen35VisionConfig& config)
    : loader_(loader), config_(config) {
    // Que estén los tensores antes de que llegue la primera imagen.
    tensor("vision.patch_embed.weight");
    tensor("vision.pos_embed.weight");
    tensor("vision.merger.fc2.weight");
    tensor("vision.layer" + std::to_string(config_.depth - 1) + ".mlp.fc2.weight");
}

const TensorEntry& Qwen35VisionRunner::tensor(const std::string& name) const {
    for (const TensorEntry& entry : loader_.tensors()) {
        if (entry.block == "vision" && entry.name == name) return entry;
    }
    throw std::runtime_error("el HNF no trae el tensor de visión " + name);
}

bool Qwen35VisionRunner::run(const Gemma4RgbView& image, half* output, Qwen35ImageGrid& grid,
                             cudaStream_t stream, std::string* error) {
    try {
        const Qwen35VisionConfig& c = config_;
        uint32_t width = 0, height = 0;
        if (!qwen35_vision_target_size(image.width, image.height, c, width, height, error)) return false;
        std::vector<uint8_t> pixels;
        if (!resize_rgb_bicubic(image, width, height, pixels)) {
            if (error) *error = "no pude redimensionar la imagen";
            return false;
        }
        const uint32_t gh = height / c.patch, gw = width / c.patch, m = c.merge, P = c.patch;
        const uint32_t n = gh * gw, W = c.hidden, H = c.heads, D = W / H, I = c.intermediate, PV = c.patch_values();
        const uint32_t tokens = n / (m * m), MW = W * m * m;
        grid.rows = gh / m;
        grid.columns = gw / m;

        // Parches en orden de bloques de fusión; dentro de cada uno, canal, tiempo, fila y
        // columna, que es como viene aplanada la convolución. Y la fila y columna de cada uno.
        std::vector<half> patches(size_t(n) * PV);
        std::vector<int32_t> row_column(size_t(n) * 2);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t in_block = i % (m * m), block = i / (m * m);
            const uint32_t row = (block / (gw / m)) * m + in_block / m, column = (block % (gw / m)) * m + in_block % m;
            row_column[2 * i] = int32_t(row);
            row_column[2 * i + 1] = int32_t(column);
            half* out = patches.data() + size_t(i) * PV;
            for (uint32_t channel = 0; channel < 3; ++channel)
                for (uint32_t y = 0; y < P; ++y)
                    for (uint32_t x = 0; x < P; ++x) {
                        const uint8_t byte = pixels[(size_t(row * P + y) * width + column * P + x) * 3 + channel];
                        const half value = __float2half((float(byte) / 255.0f - 0.5f) / 0.5f);
                        for (uint32_t t = 0; t < c.temporal; ++t) out[((channel * c.temporal + t) * P + y) * P + x] = value;
                    }
        }

        // La posición aprendida es una rejilla cuadrada; se interpola a la de esta imagen.
        std::vector<half> learned;
        {
            const TensorEntry& table = tensor("vision.pos_embed.weight");
            std::vector<half> grid_table(table.size / sizeof(half));
            if (!loader_.read_tensor_data(table, grid_table.data(), table.size)) throw std::runtime_error("no leo la posición aprendida");
            const uint32_t side = c.grid_side;
            const auto taps = [&](uint32_t index, uint32_t size, uint32_t tap[2], float weight[2]) {
                const float source = float(index) * float(side - 1) / float(std::max<uint32_t>(size - 1, 1));
                const float floor = std::floor(source);
                for (int a = 0; a < 2; ++a) {
                    tap[a] = uint32_t(std::min<float>(std::max<float>(floor + float(a), 0.0f), float(side - 1)));
                    weight[a] = std::max(0.0f, 1.0f - std::fabs(source - (floor + float(a))));
                }
            };
            learned.resize(size_t(n) * W);
            std::vector<float> sum(W);
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t rt[2], ct[2];
                float rw[2], cw[2];
                taps(uint32_t(row_column[2 * i]), gh, rt, rw);
                taps(uint32_t(row_column[2 * i + 1]), gw, ct, cw);
                std::fill(sum.begin(), sum.end(), 0.0f);
                for (int a = 0; a < 2; ++a)
                    for (int b = 0; b < 2; ++b) {
                        const float weight = rw[a] * cw[b];
                        if (weight == 0.0f) continue;
                        const half* source = grid_table.data() + (size_t(rt[a]) * side + ct[b]) * W;
                        for (uint32_t d = 0; d < W; ++d) sum[d] += weight * __half2float(source[d]);
                    }
                for (uint32_t d = 0; d < W; ++d) learned[size_t(i) * W + d] = __float2half(sum[d]);
            }
        }

        // Memoria de trabajo de esta imagen. Se suelta al salir.
        Device<half> d_patches(size_t(n) * PV), x(size_t(n) * W), normed(size_t(n) * W), wide(size_t(n) * std::max(3 * W, I));
        Device<half> q(size_t(n) * W), k(size_t(n) * W), v(size_t(n) * W), attended(size_t(n) * W), temp(size_t(n) * W);
        Device<half> scores(size_t(H) * n * n), merged(size_t(tokens) * MW);
        Device<int32_t> d_row_column(size_t(n) * 2), valid(size_t(n) * 2);
        check(cudaMemcpy(d_patches.data, patches.data(), patches.size() * sizeof(half), cudaMemcpyHostToDevice), "subir parches");
        check(cudaMemcpy(temp.data, learned.data(), learned.size() * sizeof(half), cudaMemcpyHostToDevice), "subir posiciones");
        check(cudaMemcpy(d_row_column.data, row_column.data(), row_column.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "subir rejilla");
        check(cudaMemset(valid.data, 0, size_t(n) * 2 * sizeof(int32_t)), "marcar parches");

        // Los pesos pasan por una ventana de la tarjeta: los de un paso entero cada vez.
        size_t window_bytes = 0;
        std::map<std::string, size_t> layer_bytes;
        for (const TensorEntry& entry : loader_.tensors()) {
            if (entry.block != "vision") continue;
            const size_t dot = entry.name.find('.', 7);            // tras «vision.»
            layer_bytes[entry.name.substr(0, dot)] += (entry.size + 15) & ~size_t(15);
        }
        for (const auto& [group, bytes] : layer_bytes) window_bytes = std::max(window_bytes, bytes);
        Device<uint8_t> window(window_bytes);
        std::vector<uint8_t> host(window_bytes);
        std::map<std::string, const half*> loaded;
        // Carga en la ventana los tensores cuyo nombre empieza por `prefix`. Antes espera a
        // que acabe lo que estuviera usando los anteriores.
        const auto load = [&](const std::string& prefix) {
            check(cudaStreamSynchronize(stream), "esperar a la capa anterior");
            loaded.clear();
            size_t used = 0;
            for (const TensorEntry& entry : loader_.tensors()) {
                if (entry.block != "vision" || entry.name.compare(0, prefix.size(), prefix) != 0) continue;
                if (!loader_.read_tensor_data(entry, host.data() + used, entry.size)) throw std::runtime_error("no leo " + entry.name);
                loaded[entry.name.substr(prefix.size())] = reinterpret_cast<const half*>(window.data + used);
                used += (entry.size + 15) & ~size_t(15);
            }
            if (used == 0) throw std::runtime_error("el HNF no trae " + prefix);
            check(cudaMemcpy(window.data, host.data(), used, cudaMemcpyHostToDevice), "subir pesos de la torre");
        };
        const auto weight = [&](const char* name) {
            const auto found = loaded.find(name);
            if (found == loaded.end()) throw std::runtime_error(std::string("falta el tensor de visión ") + name);
            return found->second;
        };
        // salida[filas, out] = entrada[filas, in] · pesoᵀ + sesgo, y la activación.
        const auto linear = [&](const half* input, const char* name, half* out, uint32_t rows, uint32_t in, uint32_t outs, int activation) {
            kernels::launch_matmul_fp16_cublas(input, weight((std::string(name) + ".weight").c_str()), out, int(rows), int(in), int(outs), stream);
            bias_act_kernel<<<blocks_for(size_t(rows) * outs), BLOCK, 0, stream>>>(
                out, weight((std::string(name) + ".bias").c_str()), size_t(rows) * outs, int(outs), activation);
        };

        load("vision.patch_embed.");
        kernels::launch_matmul_fp16_cublas(d_patches.data, weight("weight"), x.data, int(n), int(PV), int(W), stream);
        bias_act_kernel<<<blocks_for(size_t(n) * W), BLOCK, 0, stream>>>(x.data, weight("bias"), size_t(n) * W, int(W), NONE);
        kernels::launch_add_fp16(x.data, temp.data, x.data, size_t(n) * W, stream);

        const float scale = 1.0f / std::sqrt(float(D));
        for (uint32_t layer = 0; layer < c.depth; ++layer) {
            load("vision.layer" + std::to_string(layer) + ".");
            kernels::launch_layernorm_fp16(x.data, weight("ln1.weight"), weight("ln1.bias"), normed.data, int(n), int(W), 1e-6f, stream);
            linear(normed.data, "attn.qkv", wide.data, n, W, 3 * W, NONE);
            split_rope_kernel<<<blocks_for(size_t(n) * H * (D / 2)), BLOCK, 0, stream>>>(
                wide.data, d_row_column.data, q.data, k.data, v.data, int(n), int(H), int(D), 10000.0f);
            if (!kernels::launch_gemma4_vision_attention_fp16(q.data, k.data, v.data, valid.data, scores.data, attended.data,
                                                              n, H, D, scale, stream)) {
                throw std::runtime_error("falló la atención de la torre visual");
            }
            kernels::launch_gemma4_vision_head_to_token_fp16(attended.data, temp.data, n, H, D, stream);
            linear(temp.data, "attn.proj", normed.data, n, W, W, NONE);
            kernels::launch_add_fp16(x.data, normed.data, x.data, size_t(n) * W, stream);
            kernels::launch_layernorm_fp16(x.data, weight("ln2.weight"), weight("ln2.bias"), normed.data, int(n), int(W), 1e-6f, stream);
            linear(normed.data, "mlp.fc1", wide.data, n, W, I, GELU_TANH);
            linear(wide.data, "mlp.fc2", temp.data, n, I, W, NONE);
            kernels::launch_add_fp16(x.data, temp.data, x.data, size_t(n) * W, stream);
        }

        // La fusión: los cuatro parches de cada bloque van seguidos, así que juntarlos es
        // leer la misma memoria como [bloques, 4·ancho].
        load("vision.merger.");
        kernels::launch_layernorm_fp16(x.data, weight("norm.weight"), weight("norm.bias"), normed.data, int(n), int(W), 1e-6f, stream);
        linear(normed.data, "fc1", merged.data, tokens, MW, MW, GELU);
        linear(merged.data, "fc2", output, tokens, MW, c.out_hidden, NONE);
        check(cudaStreamSynchronize(stream), "terminar la torre");
        if (keep_output) {
            std::vector<half> raw(size_t(tokens) * c.out_hidden);
            check(cudaMemcpy(raw.data(), output, raw.size() * sizeof(half), cudaMemcpyDeviceToHost), "bajar la salida");
            last_output_.assign(raw.size(), 0.0f);
            for (size_t i = 0; i < raw.size(); ++i) last_output_[i] = __half2float(raw[i]);
        }
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

}  // namespace helios
