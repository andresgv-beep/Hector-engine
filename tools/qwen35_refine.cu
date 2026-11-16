// Refina con GPTQ las matrices HQ44 de la MLP de un HNF de la familia Qwen3.5.
//
// uso: qwen35_refine entrada.hnf salida.hnf --model checkpoint/ --calibration textos/
//                    [--tokens 8192] [--damp 1] [--layers N]
//      qwen35_refine entrada.hnf --tokenize fichero.txt
//
// La cuantización del conversor redondea cada peso por su cuenta. GPTQ (src/gptq.hpp) elige
// los códigos de una matriz entera mirando lo que de verdad le entra: dentro de la rejilla
// que ya trae el HNF (mismas escalas y mínimos), reparte el error de cada columna sobre las
// que faltan para que la salida de la matriz, y no cada peso, se parezca a la original.
//
// Capa a capa, en la GPU 0 y con una sola capa cargada cada vez: pasa por ella los textos
// de calibración (los estados de entrada salen de la capa anterior ya refinada), toma la
// entrada de la MLP y refina lo que esté en HQ44:
//
//   mlp.gate, mlp.up   con la entrada de la MLP. En un MoE son el experto compartido.
//   mlp.down           con las activaciones silu(gate)·up de su propio gate/up refinado.
//   moe.experts.*      cada experto con los tokens que el enrutador le manda.
//
// Lo que no esté en HQ44 se deja como está, y la atención y la capa lineal no se tocan.
// Después vuelve a pasar los textos con los pesos nuevos para dar la entrada a la capa
// siguiente.
//
// Solo cambian los bytes de los códigos, en las mismas posiciones: el tamaño y el layout
// del HNF no cambian. La salida no se crea aquí: `hnf10 begin-edit --from entrada --to
// salida --touch .m` la copia y la deja «en curso»; al terminar, `hnf10 seal salida
// --record salida.refine/record.json`. Las capas se cargan siempre de la entrada.
//
// La calibración son ficheros .txt, que se parten en documentos de MAX_DOC tokens con el
// tokenizer del propio HNF. Cada documento empieza con el estado de las capas lineales a
// cero. --tokenize escribe los ids de un texto, para preparar comparaciones.
#include "engine.hpp"
#include "gptq.hpp"
#include "graph_builder.hpp"
#include "hnf10.hpp"
#include "hnf_loader.hpp"
#include "htf_tokenizer.hpp"
#include "kernels.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace helios;
namespace fs = std::filesystem;

namespace {

constexpr uint32_t MAX_DOC = 512;
const char* const STATE_PREFIX = "_ref";

void check(cudaError_t error, const std::string& what) {
    if (error != cudaSuccess) throw std::runtime_error(what + ": " + cudaGetErrorString(error));
}
void check(cublasStatus_t status, const std::string& what) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(what + ": cuBLAS " + std::to_string(int(status)));
}
double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
std::string read_text(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("no leo " + path.string());
    std::ostringstream out;
    out << f.rdbuf();
    return out.str();
}
void pread_all(int fd, void* out, size_t bytes, uint64_t offset) {
    char* at = static_cast<char*>(out);
    while (bytes) {
        const ssize_t n = ::pread(fd, at, bytes, off_t(offset));
        if (n <= 0) throw std::runtime_error("lectura corta");
        at += n; bytes -= size_t(n); offset += uint64_t(n);
    }
}
void pwrite_all(int fd, const void* data, size_t bytes, uint64_t offset) {
    const char* at = static_cast<const char*>(data);
    while (bytes) {
        const ssize_t n = ::pwrite(fd, at, bytes, off_t(offset));
        if (n <= 0) throw std::runtime_error("escritura corta");
        at += n; bytes -= size_t(n); offset += uint64_t(n);
    }
}

// ---------------------------------------------------------------------------
// Los pesos originales, de los safetensors. Solo hace falta encontrar un tensor por su
// nombre: de la cabecera JSON se saca su dtype y dónde empieza, sin analizarla entera.
class Checkpoint {
public:
    explicit Checkpoint(const fs::path& dir) {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() != ".safetensors") continue;
            Shard shard;
            shard.path = entry.path();
            std::ifstream f(shard.path, std::ios::binary);
            uint64_t length = 0;
            f.read(reinterpret_cast<char*>(&length), 8);
            shard.header.resize(length);
            f.read(shard.header.data(), std::streamsize(length));
            if (!f) throw std::runtime_error("cabecera ilegible en " + shard.path.string());
            shard.base = 8 + length;
            shards_.push_back(std::move(shard));
        }
        if (shards_.empty()) throw std::runtime_error("no hay .safetensors en " + dir.string());
    }

    // `count` valores desde el elemento `first` del tensor `name`, en FP64.
    std::vector<double> read(const std::string& name, uint64_t first, size_t count) const {
        const std::string key = "\"" + name + "\":{";
        for (const Shard& shard : shards_) {
            const size_t at = shard.header.find(key);
            if (at == std::string::npos) continue;
            const size_t end = shard.header.find('}', at);
            const std::string entry = shard.header.substr(at, end - at);
            const std::string dtype = quoted(entry, "\"dtype\":\"");
            const size_t offsets = entry.find("\"data_offsets\":[");
            if (dtype.empty() || offsets == std::string::npos) throw std::runtime_error("cabecera rara para " + name);
            const uint64_t begin = std::strtoull(entry.c_str() + offsets + 16, nullptr, 10);
            const size_t width = dtype == "F32" ? 4 : 2;
            if (dtype != "F32" && dtype != "F16" && dtype != "BF16") throw std::runtime_error(name + ": dtype " + dtype);
            std::vector<uint8_t> raw(count * width);
            const int fd = ::open(shard.path.c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("no abro " + shard.path.string());
            pread_all(fd, raw.data(), raw.size(), shard.base + begin + first * width);
            ::close(fd);
            std::vector<double> out(count);
            for (size_t i = 0; i < count; ++i) {
                if (dtype == "F32") { float v; std::memcpy(&v, raw.data() + 4 * i, 4); out[i] = v; }
                else if (dtype == "BF16") {
                    uint16_t h; std::memcpy(&h, raw.data() + 2 * i, 2);
                    const uint32_t bits = uint32_t(h) << 16;
                    float v; std::memcpy(&v, &bits, 4); out[i] = v;
                } else { half h; std::memcpy(&h, raw.data() + 2 * i, 2); out[i] = __half2float(h); }
            }
            return out;
        }
        throw std::runtime_error("el checkpoint no trae " + name);
    }

private:
    struct Shard { fs::path path; std::string header; uint64_t base = 0; };
    static std::string quoted(const std::string& text, const std::string& key) {
        const size_t at = text.find(key);
        if (at == std::string::npos) return {};
        const size_t start = at + key.size();
        return text.substr(start, text.find('"', start) - start);
    }
    std::vector<Shard> shards_;
};

EngineConfig engine_config() {
    check(cudaSetDevice(0), "elegir GPU");
    EngineConfig config;
    config.device = 0;
    check(cudaStreamCreate(&config.stream), "crear stream");
    return config;
}

int layer_of(const std::string& name) {
    const size_t at = name.find(".layer");
    if (at == std::string::npos) return -1;
    return std::atoi(name.c_str() + at + 6);
}

// Lo que hay que tener cargado para detectar la arquitectura sin subir el modelo: las dos
// capas en las que mira el detector (la 0 y la primera de atención) enteras, y de las
// demás solo los vectores, que bastan para contarlas.
ArchDescriptor probe_architecture(const std::string& path, std::string& layer_types) {
    Engine engine(engine_config());
    kernels::register_all_kernels(engine);
    HnfLoader loader;
    if (!loader.open(path)) throw std::runtime_error(loader.last_error());
    layer_types = loader.config().get<std::string>("layer_types", "");
    if (layer_types.find('l') == std::string::npos) throw std::runtime_error("no es un HNF de la familia Qwen3.5");
    const int attention = int(layer_types.find('a'));
    loader.set_tensor_filter([=](const TensorEntry& tensor) {
        if (tensor.block != "text_model") return false;
        const int layer = layer_of(tensor.name);
        if (layer < 0) return tensor.shape.size() < 2;
        return layer == 0 || layer == attention || tensor.shape.size() < 2;
    });
    if (!loader.load_block(BLOCK_TEXT_MODEL, engine)) throw std::runtime_error(loader.last_error());
    ArchDescriptor arch = GraphBuilder().detect_architecture(engine, "text", loader.config());
    // El detector busca el embedding y la cabeza entre los tensores; aquí no están.
    if (arch.embedding_name.empty()) arch.embedding_name = "token_embedding.weight";
    return arch;
}

// Una capa de texto en la GPU, o solo el embedding si `input`.
struct LayerDevice {
    Engine engine;
    HnfLoader loader;
    GraphBuilder builder;
    std::vector<TensorInfo*> state;      // lo que se pone a cero al empezar cada documento

    LayerDevice(const std::string& path, uint32_t layer, bool input, const ArchDescriptor& arch) : engine(engine_config()) {
        kernels::register_all_kernels(engine);
        if (!loader.open(path)) throw std::runtime_error(loader.last_error());
        loader.set_tensor_filter([=](const TensorEntry& tensor) {
            if (tensor.block != "text_model") return false;
            const int at = layer_of(tensor.name);
            if (input) return at < 0 && tensor.name.find("token_embedding") != std::string::npos;
            return at == int(layer);
        });
        if (!loader.load_block(BLOCK_TEXT_MODEL, engine)) throw std::runtime_error(loader.last_error());
        const ModelConfig& config = loader.config();
        builder.allocate_scratch(engine, config, arch, 1, MAX_DOC);
        engine.tensors().allocate_and_register("ids", {1, 1}, dtype::INT32());
        if (input) return;
        const auto zeroed = [&](const std::string& name, std::vector<uint32_t> shape, DTypeID type) {
            engine.tensors().allocate_and_register(name, shape, type);
            state.push_back(engine.tensors().get(name));
        };
        if (arch.is_linear(layer)) {
            zeroed(GraphBuilder::linear_state_name(STATE_PREFIX, layer),
                   {arch.lin_value_heads, arch.lin_key_dim, arch.lin_value_dim}, dtype::FP32());
            zeroed(GraphBuilder::linear_conv_name(STATE_PREFIX, layer), {arch.lin_channels(), arch.lin_kernel - 1}, dtype::FP32());
        } else {
            const std::string base = std::string(STATE_PREFIX) + ".layer" + std::to_string(layer);
            zeroed(base + ".k", {1, MAX_DOC, config.num_key_value_heads(), config.head_dim()}, dtype::FP16());
            zeroed(base + ".v", {1, MAX_DOC, config.num_key_value_heads(), config.head_dim()}, dtype::FP16());
        }
    }

    // Los scratch de rango 3 siguen la longitud del documento que se está pasando.
    void set_rows(uint32_t rows) {
        for (const std::string& name : engine.tensors().names()) {
            if (name.rfind("_s.", 0) != 0 || name == "_s.logits") continue;
            TensorInfo* t = engine.tensors().get(name);
            if (t && t->shape.size() == 3) t->shape[1] = rows;
        }
    }
};

struct Calibration {
    std::vector<std::vector<int32_t>> documents;
    size_t tokens = 0;
};

uint64_t fingerprint(const Calibration& c) {
    uint64_t h = 1469598103934665603ull;
    for (const auto& document : c.documents)
        for (int32_t id : document)
            for (int b = 0; b < 4; ++b) { h ^= uint8_t(uint32_t(id) >> (8 * b)); h *= 1099511628211ull; }
    return h;
}

// H = XᵀX en el triángulo superior (orden de columnas) de una X [n, cols] ya en la GPU.
void hessian(cublasHandle_t blas, const double* d_x, size_t n, int cols, double* d_h) {
    check(cudaMemset(d_h, 0, size_t(cols) * cols * sizeof(double)), "H a cero");
    const double one = 1.0, zero = 0.0;
    // X fila a fila es Xᵀ en orden de columnas [cols, n]: H = (Xᵀ)(Xᵀ)ᵀ.
    check(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, cols, int(n), &one, d_x, cols, &zero, d_h, cols), "syrk");
}

// Una matriz HQ44 [rows, cols] dentro de un tensor del HNF.
struct Matrix {
    std::string name, source;
    uint64_t offset = 0, size = 0;      // del tensor entero en el fichero
    uint32_t experts = 0, rows = 0, cols = 0;
    uint64_t stride() const { return uint64_t(rows) * cols / 256 * gptq::HQ44_BLOCK_BYTES; }
};

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 3) {
        std::cerr << "uso: qwen35_refine entrada.hnf salida.hnf --model checkpoint/ --calibration textos/ "
                     "[--tokens 8192] [--damp 1] [--layers N]\n"
                     "     qwen35_refine entrada.hnf --tokenize fichero.txt\n";
        return 2;
    }
    const fs::path input_path = argv[1];
    if (std::string(argv[2]) == "--tokenize") {
        if (argc != 4) throw std::runtime_error("--tokenize pide un fichero");
        HnfLoader loader;
        if (!loader.open(input_path.string())) throw std::runtime_error(loader.last_error());
        const HTFTokenizer* tokenizer = loader.tokenizer("text");
        if (!tokenizer) throw std::runtime_error("el HNF no trae tokenizer");
        for (int32_t id : tokenizer->encode(read_text(argv[3]), false, false)) std::cout << id << ' ';
        std::cout << std::endl;
        return 0;
    }
    const fs::path output_path = argv[2];
    fs::path model_dir, calibration_dir;
    double damp = 1.0;
    uint32_t layer_limit = 0;
    size_t token_limit = 8192;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string flag = argv[i];
        if (flag == "--model") model_dir = argv[i + 1];
        else if (flag == "--calibration") calibration_dir = argv[i + 1];
        else if (flag == "--damp") damp = std::atof(argv[i + 1]);
        else if (flag == "--layers") layer_limit = uint32_t(std::atoi(argv[i + 1]));
        else if (flag == "--tokens") token_limit = size_t(std::atoll(argv[i + 1]));
        else throw std::runtime_error("opción desconocida " + flag);
    }
    if (model_dir.empty() || calibration_dir.empty()) throw std::runtime_error("faltan --model o --calibration");
    const fs::path work_dir = output_path.string() + ".refine";
    fs::create_directories(work_dir);
    const auto start = std::chrono::steady_clock::now();

    hnf10::Contents index;
    {
        std::ifstream in(input_path, std::ios::binary);
        std::string error;
        if (!hnf10::read(in, index, error)) throw std::runtime_error(error);
        if (!fs::exists(output_path)) throw std::runtime_error("crea antes la salida con: hnf10 begin-edit --from entrada --to salida --touch .m");
        std::ifstream out(output_path, std::ios::binary);
        hnf10::Contents edited;
        if (!hnf10::is_hnf10(out) || !hnf10::read(out, edited, error, true)) throw std::runtime_error("salida: " + error);
        if (edited.state != 2) throw std::runtime_error("la salida no está en edición: crea una con hnf10 begin-edit");
        if (edited.tensors.size() != index.tensors.size()) throw std::runtime_error("la salida no es una copia de la entrada");
        for (size_t i = 0; i < edited.tensors.size(); ++i)
            if (edited.tensors[i].name != index.tensors[i].name || edited.tensors[i].offset != index.tensors[i].offset ||
                edited.tensors[i].size != index.tensors[i].size)
                throw std::runtime_error("la salida no tiene el índice de la entrada: " + edited.tensors[i].name);
    }
    // La matriz HQ44 con ese nombre, o una vacía (rows == 0) si no está o va en otro formato.
    const auto matrix_named = [&](uint32_t layer, const std::string& part) {
        const std::string name = "text.layer" + std::to_string(layer) + "." + part + ".weight";
        Matrix m;
        for (const auto& t : index.tensors) {
            if (t.name != name || t.dtype != "hq44k_g16") continue;
            const bool packed = t.shape.size() == 3;
            if (!packed && t.shape.size() != 2) throw std::runtime_error(name + ": forma inesperada");
            m.name = name; m.source = t.source; m.offset = t.offset; m.size = t.size;
            m.experts = packed ? t.shape[0] : 1;
            m.rows = t.shape[packed ? 1 : 0];
            m.cols = t.shape[packed ? 2 : 1];
            if (t.logical_last != m.cols) throw std::runtime_error(name + ": columnas de relleno, que esta herramienta no trata");
            if (m.stride() * m.experts != m.size) throw std::runtime_error(name + ": tamaño inesperado");
        }
        return m;
    };

    std::string layer_types;
    const ArchDescriptor arch = probe_architecture(input_path.string(), layer_types);
    HnfLoader meta;
    if (!meta.open(input_path.string())) throw std::runtime_error(meta.last_error());
    const ModelConfig config = meta.config();
    const uint32_t layers = config.num_hidden_layers(), H = config.hidden_size();
    const uint32_t end_layer = layer_limit ? std::min(layer_limit, layers) : layers;
    const uint32_t E = arch.moe_experts, K = arch.moe_top_k;
    const bool silu = arch.activation == ActivationType::SILU;
    if (!silu) throw std::runtime_error("la MLP de esta familia es SiLU; el HNF dice otra cosa");
    const Checkpoint checkpoint(model_dir);

    // Calibración: cada .txt, en documentos de MAX_DOC tokens.
    Calibration calibration;
    {
        const HTFTokenizer* tokenizer = meta.tokenizer("text");
        if (!tokenizer) throw std::runtime_error("el HNF no trae tokenizer");
        std::vector<fs::path> files;
        for (const auto& entry : fs::directory_iterator(calibration_dir))
            if (entry.path().extension() == ".txt") files.push_back(entry.path());
        std::sort(files.begin(), files.end());
        // Por turnos entre ficheros, para que el tope de tokens no deje fuera a los últimos.
        std::vector<std::vector<int32_t>> ids;
        for (const auto& p : files) ids.push_back(tokenizer->encode(read_text(p), false, false));
        for (size_t part = 0; calibration.tokens < token_limit; ++part) {
            bool any = false;
            for (const auto& all : ids) {
                if (part * MAX_DOC >= all.size() || calibration.tokens >= token_limit) continue;
                const size_t first = part * MAX_DOC;
                const size_t n = std::min({size_t(MAX_DOC), all.size() - first, token_limit - calibration.tokens});
                if (n < 16) continue;
                calibration.documents.emplace_back(all.begin() + long(first), all.begin() + long(first + n));
                calibration.tokens += n;
                any = true;
            }
            if (!any) break;
        }
        if (calibration.documents.empty()) throw std::runtime_error("no hay .txt con texto en " + calibration_dir.string());
    }
    const size_t T = calibration.tokens;
    std::cerr << calibration.documents.size() << " documentos, " << T << " tokens de calibración; " << layers
              << " capas (" << layer_types << ")" << std::endl;

    // Embedding de la calibración.
    std::vector<half> hidden(T * H);
    {
        LayerDevice device(input_path.string(), 0, true, arch);
        Engine& engine = device.engine;
        const CommandBuffer embed = device.builder.build_embedding_only(arch, "ids");
        size_t row = 0;
        for (const auto& document : calibration.documents)
            for (int32_t id : document) {
                check(cudaMemcpy(engine.tensors().at("ids").ptr, &id, 4, cudaMemcpyHostToDevice), "token");
                engine.execute(embed);
                engine.sync();
                check(cudaMemcpy(hidden.data() + row++ * H, engine.tensors().at("_s.hidden").ptr, H * 2, cudaMemcpyDeviceToHost), "embedding");
            }
    }

    const int fd = ::open(output_path.c_str(), O_RDWR);
    if (fd < 0) throw std::runtime_error("no abro " + output_path.string());
    std::ofstream log(work_dir / "refine.log", std::ios::app);
    log << "inicio damp " << damp << " documentos " << calibration.documents.size() << " tokens " << T << '\n';
    uint64_t refined_total = 0;

    for (uint32_t layer = 0; layer < end_layer; ++layer) {
        const auto layer_start = std::chrono::steady_clock::now();
        LayerDevice device(input_path.string(), layer, false, arch);
        Engine& engine = device.engine;
        const KVCacheParams cache{STATE_PREFIX, 0, MAX_DOC};

        // Pasa los documentos por la capa. Con `capture`, guarda la entrada de la MLP y, en
        // un MoE, a qué expertos manda el enrutador cada token.
        std::vector<half> mlp_input;
        std::vector<int32_t> routes;
        const auto run_layer = [&](bool capture, std::vector<half>& out) {
            out.resize(T * H);
            if (capture) { mlp_input.resize(T * H); if (E) routes.resize(T * K); }
            size_t row = 0;
            for (const auto& document : calibration.documents) {
                const uint32_t n = uint32_t(document.size());
                for (TensorInfo* t : device.state) check(cudaMemset(t->ptr, 0, t->size_bytes), "estado a cero");
                device.set_rows(n);
                check(cudaMemcpy(engine.tensors().at("_s.hidden").ptr, hidden.data() + row * H, size_t(n) * H * 2, cudaMemcpyHostToDevice), "subir estados");
                engine.update_device_cache_pos(0, n);
                CommandBuffer commands = device.builder.build_layer_only(config, arch, layer, 1, n, cache);
                engine.execute(commands);
                engine.sync();
                check(cudaMemcpy(out.data() + row * H, engine.tensors().at("_s.hidden").ptr, size_t(n) * H * 2, cudaMemcpyDeviceToHost), "bajar estados");
                if (capture) {
                    check(cudaMemcpy(mlp_input.data() + row * H, engine.tensors().at("_s.normed").ptr, size_t(n) * H * 2, cudaMemcpyDeviceToHost), "bajar entrada de la MLP");
                    if (E) check(cudaMemcpy(routes.data() + row * K, engine.tensors().at("_s.moe_indices").ptr, size_t(n) * K * 4, cudaMemcpyDeviceToHost), "bajar enrutado");
                }
                row += n;
            }
        };
        std::vector<half> next;
        run_layer(true, next);

        const Matrix gate = matrix_named(layer, "mlp.gate"), up = matrix_named(layer, "mlp.up"), down = matrix_named(layer, "mlp.down");
        const Matrix expert_gate_up = matrix_named(layer, "moe.experts.gate_up"), expert_down = matrix_named(layer, "moe.experts.down");
        uint32_t max_rows = 1, max_cols = H;
        for (const Matrix* m : {&gate, &up, &down, &expert_gate_up, &expert_down}) {
            max_rows = std::max(max_rows, m->rows); max_cols = std::max(max_cols, m->cols);
        }
        const uint32_t I = std::max(gate.rows, up.rows);
        const uint32_t EI = expert_gate_up.rows / 2;
        const uint32_t widest_act = std::max({H, I, 2 * EI});
        gptq::Quantizer quantizer{int(max_rows), int(max_cols)};
        const cublasHandle_t blas = quantizer.cublas();
        double *d_x = nullptr, *d_act = nullptr, *d_h = nullptr, *d_w = nullptr, *d_wq = nullptr;
        float *d_a = nullptr, *d_m = nullptr;
        uint8_t* d_codes = nullptr;
        const size_t matrix_max = size_t(max_rows) * max_cols;
        check(cudaMalloc(&d_x, T * H * sizeof(double)), "X");
        check(cudaMalloc(&d_act, T * size_t(widest_act) * sizeof(double)), "activaciones");
        check(cudaMalloc(&d_h, size_t(max_cols) * max_cols * sizeof(double)), "H");
        check(cudaMalloc(&d_w, matrix_max * sizeof(double)), "W");
        check(cudaMalloc(&d_wq, matrix_max * sizeof(double)), "Wq");
        check(cudaMalloc(&d_a, matrix_max * sizeof(float)), "a");
        check(cudaMalloc(&d_m, matrix_max * sizeof(float)), "m");
        check(cudaMalloc(&d_codes, matrix_max), "códigos");

        // Refina una matriz [rows, cols] con la H que hay en d_h y deja en `wq` sus valores
        // nuevos. `blocks` son sus bytes HQ44, que salen con los códigos cambiados.
        const auto refine = [&](const Matrix& m, uint32_t expert, uint8_t* blocks, std::vector<double>* wq) {
            const size_t count = size_t(m.rows) * m.cols;
            std::vector<float> a(count), mins(count);
            gptq::hq44_grid(blocks, int(m.rows), int(m.cols), a.data(), mins.data());
            const std::vector<double> w = checkpoint.read(m.source, uint64_t(expert) * count, count);
            check(cudaMemcpy(d_w, w.data(), count * 8, cudaMemcpyHostToDevice), "subir W");
            check(cudaMemcpy(d_a, a.data(), count * 4, cudaMemcpyHostToDevice), "subir a");
            check(cudaMemcpy(d_m, mins.data(), count * 4, cudaMemcpyHostToDevice), "subir m");
            quantizer.run_device(d_w, d_h, d_a, d_m, int(m.rows), int(m.cols), damp, d_codes);
            std::vector<uint8_t> codes(count);
            check(cudaMemcpy(codes.data(), d_codes, count, cudaMemcpyDeviceToHost), "bajar códigos");
            gptq::hq44_set_codes(blocks, codes.data(), int(m.rows), int(m.cols), int(m.cols));
            if (wq) {
                wq->resize(count);
                for (size_t i = 0; i < count; ++i) (*wq)[i] = double(a[i] * float(codes[i]) - mins[i]);
            }
            ++refined_total;
        };
        // G[n, rows] = X · Wqᵀ, de la X [n, cols] que hay en d_x.
        const auto project = [&](const std::vector<double>& wq, size_t n, uint32_t rows, uint32_t cols) {
            check(cudaMemcpy(d_wq, wq.data(), wq.size() * 8, cudaMemcpyHostToDevice), "subir Wq");
            const double one = 1.0, zero = 0.0;
            check(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, int(rows), int(n), int(cols), &one,
                              d_wq, int(cols), d_x, int(cols), &zero, d_act, int(rows)), "proyectar");
            std::vector<double> g(n * rows);
            check(cudaMemcpy(g.data(), d_act, g.size() * 8, cudaMemcpyDeviceToHost), "bajar proyección");
            return g;
        };
        const auto upload_rows = [&](const std::vector<double>& x) {
            check(cudaMemcpy(d_x, x.data(), x.size() * 8, cudaMemcpyHostToDevice), "subir X");
        };
        const auto silu_of = [](double v) { return v / (1.0 + std::exp(-v)); };
        // Los bytes de un tensor del fichero, y de vuelta al fichero y al motor.
        const auto load_bytes = [&](const Matrix& m) {
            std::vector<uint8_t> bytes(m.size);
            pread_all(fd, bytes.data(), m.size, m.offset);
            std::vector<uint8_t> probe(std::min<size_t>(m.size, 4096));
            check(cudaMemcpy(probe.data(), engine.tensors().at(m.name).ptr, probe.size(), cudaMemcpyDeviceToHost), "comprobar layout");
            if (std::memcmp(probe.data(), bytes.data(), probe.size()) != 0)
                throw std::runtime_error(m.name + ": el motor no guarda los mismos bytes que el fichero");
            return bytes;
        };
        const auto store_bytes = [&](const Matrix& m, const std::vector<uint8_t>& bytes) {
            pwrite_all(fd, bytes.data(), m.size, m.offset);
            check(cudaMemcpy(engine.tensors().at(m.name).ptr, bytes.data(), m.size, cudaMemcpyHostToDevice), "pesos nuevos al motor");
        };

        uint32_t refined_here = 0;
        // La MLP densa (el experto compartido, en un MoE).
        std::vector<double> x(T * H);
        for (size_t i = 0; i < x.size(); ++i) x[i] = __half2float(mlp_input[i]);
        if (gate.rows || up.rows || down.rows) {
            std::vector<double> gate_q, up_q;
            upload_rows(x);
            for (const Matrix* m : {&gate, &up}) {
                if (!m->rows) continue;
                std::vector<uint8_t> bytes = load_bytes(*m);
                hessian(blas, d_x, T, int(H), d_h);
                refine(*m, 0, bytes.data(), m == &gate ? &gate_q : &up_q);
                store_bytes(*m, bytes);
                ++refined_here;
            }
            if (down.rows) {
                // La entrada de down es silu(gate)·up. De lo refinado si lo está; si gate o up
                // van en otro formato, de sus pesos originales.
                const uint32_t inner = down.cols;
                const auto values = [&](const Matrix& m, const std::vector<double>& refined_values, const char* part) {
                    if (m.rows) return refined_values;
                    for (const auto& t : index.tensors)
                        if (t.name == "text.layer" + std::to_string(layer) + ".mlp." + part + ".weight")
                            return checkpoint.read(t.source, 0, size_t(inner) * H);
                    throw std::runtime_error(std::string("falta mlp.") + part);
                };
                const std::vector<double> g = project(values(gate, gate_q, "gate"), T, inner, H);
                const std::vector<double> u = project(values(up, up_q, "up"), T, inner, H);
                std::vector<double> act(T * inner);
                for (size_t i = 0; i < act.size(); ++i) act[i] = silu_of(g[i]) * u[i];
                check(cudaMemcpy(d_act, act.data(), act.size() * 8, cudaMemcpyHostToDevice), "subir activaciones");
                std::vector<uint8_t> bytes = load_bytes(down);
                hessian(blas, d_act, T, int(inner), d_h);
                refine(down, 0, bytes.data(), nullptr);
                store_bytes(down, bytes);
                ++refined_here;
            }
        }
        // Los expertos, cada uno con sus tokens.
        uint32_t idle = 0;
        if (expert_gate_up.rows && expert_down.rows) {
            std::vector<std::vector<uint32_t>> rows_of(E);
            for (size_t t = 0; t < T; ++t)
                for (uint32_t k = 0; k < K; ++k) {
                    const int32_t e = routes[t * K + k];
                    if (e < 0 || uint32_t(e) >= E) throw std::runtime_error("enrutado fuera de rango");
                    rows_of[size_t(e)].push_back(uint32_t(t));
                }
            std::vector<uint8_t> gate_up_bytes = load_bytes(expert_gate_up), down_bytes = load_bytes(expert_down);
            for (uint32_t e = 0; e < E; ++e) {
                const size_t n = rows_of[e].size();
                if (n == 0) { ++idle; continue; }
                std::vector<double> xe(n * H);
                for (size_t r = 0; r < n; ++r) std::copy_n(x.begin() + long(size_t(rows_of[e][r]) * H), H, xe.begin() + long(r * H));
                upload_rows(xe);
                hessian(blas, d_x, n, int(H), d_h);
                std::vector<double> wq;
                refine(expert_gate_up, e, gate_up_bytes.data() + e * expert_gate_up.stride(), &wq);
                const std::vector<double> g = project(wq, n, 2 * EI, H);
                std::vector<double> act(n * EI);
                for (size_t r = 0; r < n; ++r)
                    for (uint32_t j = 0; j < EI; ++j) act[r * EI + j] = silu_of(g[r * 2 * EI + j]) * g[r * 2 * EI + EI + j];
                check(cudaMemcpy(d_act, act.data(), act.size() * 8, cudaMemcpyHostToDevice), "subir activaciones");
                hessian(blas, d_act, n, int(EI), d_h);
                refine(expert_down, e, down_bytes.data() + e * expert_down.stride(), nullptr);
                refined_here += 2;
            }
            store_bytes(expert_gate_up, gate_up_bytes);
            store_bytes(expert_down, down_bytes);
        }
        cudaFree(d_x); cudaFree(d_act); cudaFree(d_h); cudaFree(d_w); cudaFree(d_wq);
        cudaFree(d_a); cudaFree(d_m); cudaFree(d_codes);
        if (::fsync(fd) != 0) throw std::runtime_error("fsync falló");

        // La capa vuelve a pasar con los pesos nuevos: es la entrada de la siguiente.
        if (refined_here) run_layer(false, next);
        hidden.swap(next);

        std::ostringstream line;
        line << "capa " << layer << " (" << (arch.is_linear(layer) ? "lineal" : "atención") << "): " << refined_here
             << " matrices refinadas";
        if (E) line << ", " << idle << " expertos sin tokens";
        line << ", " << seconds_since(layer_start) << " s";
        std::cerr << line.str() << std::endl;
        log << line.str() << '\n';
        log.flush();
    }
    ::close(fd);
    {
        char print[17];
        std::snprintf(print, sizeof(print), "%016llx", static_cast<unsigned long long>(fingerprint(calibration)));
        std::ofstream(work_dir / "record.json")
            << "{\n \"tool\": \"qwen35_refine\",\n \"mode\": \"gptq\",\n"
            << " \"input\": {\"path\": \"" << input_path.string() << "\", \"bytes\": " << fs::file_size(input_path) << "},\n"
            << " \"layers\": [0, " << end_layer << "],\n \"matrices\": " << refined_total << ",\n"
            << " \"damp\": " << damp << ",\n \"grid\": \"hnf_existing_headers\",\n \"arithmetic\": \"fp64 cuSOLVER/cuBLAS, block 128\",\n"
            << " \"calibration\": {\"documents\": " << calibration.documents.size() << ", \"tokens\": " << T
            << ", \"max_tokens_per_document\": " << MAX_DOC << ", \"ids_fnv1a64\": \"" << print << "\"},\n"
            << " \"complete\": " << (end_layer == layers ? "true" : "false") << "\n}\n";
    }
    std::cerr << refined_total << " matrices refinadas en " << seconds_since(start) << " s" << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
}
