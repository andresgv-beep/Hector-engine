// Refina con GPTQ los expertos HQ44 de un HNF de Gemma 4 MoE ya formado por el conversor.
//
// uso: hnf_refine entrada.hnf salida.hnf --model checkpoint/ --calibration casos/
//                 [--damp 1] [--work carpeta/] [--layers N]
//
// Capa a capa, en la GPU 0: carga solo esa capa, pasa por ella todos los tokens de
// calibración (los estados de entrada salen de la capa anterior ya refinada), toma la
// entrada de los expertos y su enrutado, y para cada experto aplica GPTQ (src/gptq.hpp)
// dentro de la rejilla que ya trae el HNF: mismas escalas y mínimos, códigos nuevos.
// down usa las activaciones de su propio gate/up refinado. Después vuelve a pasar los
// tokens con los expertos nuevos para obtener la entrada de la capa siguiente.
//
// Solo cambian los bytes de los códigos de los expertos, en las mismas posiciones: el
// tamaño y el layout del HNF no cambian. El manifiesto no se toca; lo sella el conversor.
// Reanudable: en --work quedan el estado tras cada capa terminada y un registro.
#include "gemma4_kv_cache.hpp"
#include "gemma4_split.hpp"
#include "gptq.hpp"
#include "graph_builder.hpp"
#include "hnf_loader.hpp"
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
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace helios;
namespace fs = std::filesystem;

namespace {

constexpr uint32_t PREFILL_CHUNK = 512, MAX_FORCING = 512, MAX_SEQ = 4096;
// Como en gemma4_moe_generate: 1024 elige el camino flash y la posición va en el dispositivo.
constexpr uint32_t CACHE_BUILD_POSITION = 1024;
const char* const HIDDEN = "_s.hidden";

void check(cudaError_t error, const std::string& what) {
    if (error != cudaSuccess) throw std::runtime_error(what + ": " + cudaGetErrorString(error));
}
void check(cublasStatus_t status, const std::string& what) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(what + ": cuBLAS " + std::to_string(int(status)));
}
double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// ---------------------------------------------------------------------------
// JSON mínimo: lo justo para el manifiesto del HNF y las cabeceras de safetensors.
struct Json {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    double number = 0;
    bool boolean = false;
    std::string string;
    std::vector<Json> array;
    std::vector<std::pair<std::string, Json>> object;

    const Json* find(const std::string& key) const {
        for (const auto& [k, v] : object) if (k == key) return &v;
        return nullptr;
    }
    const Json& at(const std::string& key) const {
        const Json* v = find(key);
        if (!v) throw std::runtime_error("JSON sin clave " + key);
        return *v;
    }
    uint64_t u64() const { return uint64_t(number); }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : s_(text) {}
    Json parse() { Json v = value(); space(); return v; }
private:
    const std::string& s_;
    size_t i_ = 0;
    void space() { while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_; }
    [[noreturn]] void fail() { throw std::runtime_error("JSON mal formado cerca de " + std::to_string(i_)); }
    char peek() { space(); if (i_ >= s_.size()) fail(); return s_[i_]; }
    void expect(char c) { if (peek() != c) fail(); ++i_; }
    std::string str() {
        expect('"');
        std::string out;
        while (i_ < s_.size() && s_[i_] != '"') {
            char c = s_[i_++];
            if (c != '\\') { out += c; continue; }
            if (i_ >= s_.size()) fail();
            c = s_[i_++];
            switch (c) {
                case 'n': out += '\n'; break; case 't': out += '\t'; break; case 'r': out += '\r'; break;
                case 'b': out += '\b'; break; case 'f': out += '\f'; break;
                case 'u': {
                    if (i_ + 4 > s_.size()) fail();
                    const unsigned code = unsigned(std::stoul(s_.substr(i_, 4), nullptr, 16));
                    i_ += 4;
                    if (code < 0x80) out += char(code);
                    else if (code < 0x800) { out += char(0xC0 | (code >> 6)); out += char(0x80 | (code & 63)); }
                    else { out += char(0xE0 | (code >> 12)); out += char(0x80 | ((code >> 6) & 63)); out += char(0x80 | (code & 63)); }
                    break;
                }
                default: out += c;
            }
        }
        if (i_ >= s_.size()) fail();
        ++i_;
        return out;
    }
    Json value() {
        Json v;
        const char c = peek();
        if (c == '{') {
            v.type = Json::Object; ++i_;
            if (peek() == '}') { ++i_; return v; }
            for (;;) {
                std::string key = str();
                expect(':');
                v.object.emplace_back(std::move(key), value());
                if (peek() == ',') { ++i_; continue; }
                expect('}'); return v;
            }
        }
        if (c == '[') {
            v.type = Json::Array; ++i_;
            if (peek() == ']') { ++i_; return v; }
            for (;;) {
                v.array.push_back(value());
                if (peek() == ',') { ++i_; continue; }
                expect(']'); return v;
            }
        }
        if (c == '"') { v.type = Json::String; v.string = str(); return v; }
        if (s_.compare(i_, 4, "true") == 0) { v.type = Json::Bool; v.boolean = true; i_ += 4; return v; }
        if (s_.compare(i_, 5, "false") == 0) { v.type = Json::Bool; i_ += 5; return v; }
        if (s_.compare(i_, 4, "null") == 0) { i_ += 4; return v; }
        char* end = nullptr;
        v.type = Json::Number;
        v.number = std::strtod(s_.c_str() + i_, &end);
        if (end == s_.c_str() + i_) fail();
        i_ = size_t(end - s_.c_str());
        return v;
    }
};

std::string read_text(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("no abro " + path.string());
    std::stringstream s; s << f.rdbuf(); return s.str();
}

void pread_all(int fd, void* out, size_t bytes, uint64_t offset) {
    auto* p = static_cast<char*>(out);
    while (bytes) {
        const ssize_t n = ::pread(fd, p, bytes, off_t(offset));
        if (n <= 0) throw std::runtime_error("pread falló");
        p += n; bytes -= size_t(n); offset += uint64_t(n);
    }
}
void pwrite_all(int fd, const void* data, size_t bytes, uint64_t offset) {
    auto* p = static_cast<const char*>(data);
    while (bytes) {
        const ssize_t n = ::pwrite(fd, p, bytes, off_t(offset));
        if (n <= 0) throw std::runtime_error("pwrite falló");
        p += n; bytes -= size_t(n); offset += uint64_t(n);
    }
}

Json read_manifest(const fs::path& hnf) {
    std::ifstream f(hnf, std::ios::binary);
    if (!f) throw std::runtime_error("no abro " + hnf.string());
    uint64_t where[2];
    f.seekg(32);
    f.read(reinterpret_cast<char*>(where), sizeof(where));
    std::string text(where[1], '\0');
    f.seekg(std::streamoff(where[0]));
    f.read(text.data(), std::streamsize(text.size()));
    if (!f) throw std::runtime_error("manifiesto ilegible");
    return JsonParser(text).parse();
}

// Un tensor de expertos: dónde están sus bytes en el HNF y de dónde sale en el checkpoint.
struct ExpertTensor {
    uint64_t offset = 0, size = 0;
    uint32_t experts = 0, rows = 0, logical = 0, physical = 0;
    fs::path shard;
    uint64_t source_offset = 0;   // primer byte BF16 del experto 0 en el shard
    uint64_t stride() const { return uint64_t(rows) * physical / 256 * gptq::HQ44_BLOCK_BYTES; }
};

class Checkpoint {
public:
    explicit Checkpoint(const fs::path& dir) : dir_(dir) {
        const Json index = JsonParser(read_text(dir / "model.safetensors.index.json")).parse();
        for (const auto& [name, file] : index.at("weight_map").object) shard_of_[name] = file.string;
    }
    // Cabecera del shard y posición del tensor `source`.
    void locate(const std::string& source, ExpertTensor& t) {
        const auto file = shard_of_.find(source);
        if (file == shard_of_.end()) throw std::runtime_error("el checkpoint no trae " + source);
        t.shard = dir_ / file->second;
        auto& header = headers_[file->second];
        if (header.type == Json::Null) {
            std::ifstream f(t.shard, std::ios::binary);
            uint64_t length = 0;
            f.read(reinterpret_cast<char*>(&length), 8);
            std::string text(length, '\0');
            f.read(text.data(), std::streamsize(length));
            if (!f) throw std::runtime_error("cabecera ilegible en " + t.shard.string());
            header = JsonParser(text).parse();
            bases_[file->second] = 8 + length;
        }
        const Json& entry = header.at(source);
        if (entry.at("dtype").string != "BF16") throw std::runtime_error(source + " no es BF16");
        const auto& shape = entry.at("shape").array;
        if (shape.size() != 3 || shape[0].u64() != t.experts || shape[1].u64() != t.rows || shape[2].u64() != t.logical)
            throw std::runtime_error(source + ": forma distinta de la del manifiesto");
        t.source_offset = bases_[file->second] + entry.at("data_offsets").array[0].u64();
    }
private:
    fs::path dir_;
    std::map<std::string, std::string> shard_of_;
    std::map<std::string, Json> headers_;
    std::map<std::string, uint64_t> bases_;
};

std::vector<double> read_original(const ExpertTensor& t, uint32_t expert) {
    const size_t count = size_t(t.rows) * t.logical;
    std::vector<uint16_t> raw(count);
    const int fd = ::open(t.shard.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("no abro " + t.shard.string());
    pread_all(fd, raw.data(), count * 2, t.source_offset + uint64_t(expert) * count * 2);
    ::close(fd);
    std::vector<double> out(count);
    for (size_t i = 0; i < count; ++i) {
        const uint32_t bits = uint32_t(raw[i]) << 16;
        float v;
        std::memcpy(&v, &bits, 4);
        out[i] = v;
    }
    return out;
}

// La misma gelu de tanh que la referencia de NumPy, con sus constantes FP32.
double gelu(double x) {
    const double c = double(0.79788456f), k = double(0.044715f);
    return 0.5 * x * (1.0 + std::tanh(c * (x + k * x * x * x)));
}

EngineConfig engine_config() {
    check(cudaSetDevice(0), "elegir GPU");
    EngineConfig config;
    config.device = 0;
    check(cudaStreamCreate(&config.stream), "crear stream");
    config.use_split_attention = true;
    config.use_coalesced_prefill = true;
    config.use_flash_decode = true;
    config.flash_decode_min_seq = 1024;
    config.use_gemm_prefill = true;
    config.use_flash_prefill = true;
    return config;
}

// Una sola capa de texto en la GPU 0, o la entrada (embedding) si `input`.
struct LayerDevice {
    Engine engine;
    HnfLoader loader;
    GraphBuilder builder;
    Gemma4KVCache cache;

    LayerDevice(const std::string& path, uint32_t layer, bool input, const ArchDescriptor& arch)
        : engine(engine_config()) {
        kernels::register_all_kernels(engine);
        if (!loader.open(path)) throw std::runtime_error(loader.last_error());
        const uint32_t first = layer, end = input ? layer : layer + 1;
        loader.set_tensor_filter([=](const TensorEntry& tensor) { return gemma4_tensor_in_span(tensor, first, end, input); });
        if (!loader.load_block(BLOCK_TEXT_MODEL, engine)) throw std::runtime_error(loader.last_error());
        builder.allocate_gemma4_scratch(engine, loader.config(), loader.gemma4_config(), arch, 1, PREFILL_CHUNK);
        if (!input) {
            if (!cache.allocate(loader.gemma4_config(), loader.config().num_key_value_heads(), 1, MAX_SEQ, PREFILL_CHUNK,
                                nullptr, first, end)) {
                throw std::runtime_error("no pude reservar la caché KV");
            }
            cache.register_tensors(engine, "_kv");
        }
    }
};

struct Calibration {
    std::vector<std::string> names;
    std::vector<std::vector<int32_t>> ids;
    size_t tokens = 0;
};

Calibration read_calibration(const fs::path& dir) {
    Calibration c;
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        const auto p = entry.path();
        if (p.extension() == ".ids" && p.stem().extension() == ".forcing") files.push_back(p);
    }
    std::sort(files.begin(), files.end());
    for (const auto& p : files) {
        std::vector<int32_t> ids(fs::file_size(p) / 4);
        std::ifstream f(p, std::ios::binary);
        f.read(reinterpret_cast<char*>(ids.data()), std::streamsize(ids.size() * 4));
        if (ids.size() > MAX_FORCING) ids.resize(MAX_FORCING);
        c.tokens += ids.size();
        c.names.push_back(p.filename().string());
        c.ids.push_back(std::move(ids));
    }
    if (c.ids.empty()) throw std::runtime_error("no hay casos .forcing.ids en " + dir.string());
    return c;
}

template <typename T>
void write_file(const fs::path& path, const std::vector<T>& values) {
    const fs::path part = path.string() + ".part";
    {
        std::ofstream f(part, std::ios::binary);
        f.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(T)));
        if (!f) throw std::runtime_error("no escribo " + part.string());
    }
    fs::rename(part, path);
}

template <typename T>
std::vector<T> read_file(const fs::path& path) {
    std::vector<T> values(fs::file_size(path) / sizeof(T));
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(values.data()), std::streamsize(values.size() * sizeof(T)));
    if (!f) throw std::runtime_error("no leo " + path.string());
    return values;
}

// H = XᵀX en el triángulo superior (orden de columnas) de una X de host [n, cols].
void hessian(cublasHandle_t blas, const std::vector<double>& x, size_t n, int cols, double* d_x, double* d_h) {
    check(cudaMemcpy(d_x, x.data(), n * cols * sizeof(double), cudaMemcpyHostToDevice), "subir X");
    check(cudaMemset(d_h, 0, size_t(cols) * cols * sizeof(double)), "H a cero");
    const double one = 1.0, zero = 0.0;
    // X de host fila a fila es Xᵀ en orden de columnas [cols, n]: H = Xᵀ·X = (Xᵀ)(Xᵀ)ᵀ.
    check(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, cols, int(n), &one, d_x, cols, &zero, d_h, cols), "syrk");
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 3) {
        std::cerr << "uso: hnf_refine entrada.hnf salida.hnf --model checkpoint/ --calibration casos/ "
                     "[--damp 1] [--work carpeta/] [--layers N]\n";
        return 2;
    }
    const fs::path input_path = argv[1], output_path = argv[2];
    fs::path model_dir, calibration_dir, work_dir;
    double damp = 1.0;
    uint32_t layer_limit = 0;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string flag = argv[i];
        if (flag == "--model") model_dir = argv[i + 1];
        else if (flag == "--calibration") calibration_dir = argv[i + 1];
        else if (flag == "--damp") damp = std::atof(argv[i + 1]);
        else if (flag == "--work") work_dir = argv[i + 1];
        else if (flag == "--layers") layer_limit = uint32_t(std::atoi(argv[i + 1]));
        else throw std::runtime_error("opción desconocida " + flag);
    }
    if (model_dir.empty() || calibration_dir.empty()) throw std::runtime_error("faltan --model o --calibration");
    if (work_dir.empty()) work_dir = output_path.string() + ".refine";
    fs::create_directories(work_dir);
    const auto start = std::chrono::steady_clock::now();

    // Salida: copia de la entrada, que nunca se modifica.
    if (!fs::exists(output_path)) {
        std::cerr << "copiando " << input_path << " -> " << output_path << std::endl;
        const fs::path part = output_path.string() + ".part";
        fs::copy_file(input_path, part, fs::copy_options::overwrite_existing);
        fs::rename(part, output_path);
    }
    if (fs::file_size(output_path) != fs::file_size(input_path)) throw std::runtime_error("la salida no tiene el tamaño de la entrada");

    const Json manifest = read_manifest(output_path);
    std::map<std::string, const Json*> tensors, sources;
    for (const Json& t : manifest.at("tensors").array) tensors[t.at("name").string] = &t;
    for (const Json& s : manifest.at("source_tensors").array) sources[s.at("canonical").string] = &s;

    HnfLoader meta;
    if (!meta.open(output_path.string())) throw std::runtime_error(meta.last_error());
    const ModelConfig& config = meta.config();
    const Gemma4Config& gemma = meta.gemma4_config();
    const uint32_t layers = config.num_hidden_layers(), H = config.hidden_size();
    const uint32_t E = gemma.moe_experts, K = gemma.moe_top_k, I = gemma.moe_intermediate_size;
    const uint32_t end_layer = layer_limit ? std::min(layer_limit, layers) : layers;
    const ArchDescriptor arch = gemma4_split_arch(layers);

    Checkpoint checkpoint(model_dir);
    auto expert_tensor = [&](uint32_t layer, const char* kind) {
        const std::string name = "text.layer" + std::to_string(layer) + ".moe.experts." + kind + ".weight";
        const Json& t = *tensors.at(name);
        const Json& s = *sources.at(name);
        if (t.at("dtype").string != "hq44k_g16") throw std::runtime_error(name + " no es HQ44");
        ExpertTensor out;
        out.offset = t.at("offset").u64();
        out.size = t.at("size").u64();
        const auto& logical = s.at("logical_shape").array;
        const auto& physical = s.at("physical_shape").array;
        out.experts = uint32_t(logical[0].u64());
        out.rows = uint32_t(logical[1].u64());
        out.logical = uint32_t(logical[2].u64());
        out.physical = uint32_t(physical[2].u64());
        if (out.stride() * out.experts != out.size) throw std::runtime_error(name + ": tamaño inesperado");
        checkpoint.locate(s.at("source").string, out);
        return out;
    };

    const Calibration calibration = read_calibration(calibration_dir);
    std::cerr << calibration.names.size() << " casos, " << calibration.tokens << " tokens de calibración" << std::endl;

    // Reanudar desde la última capa terminada.
    uint32_t first_layer = 0;
    std::vector<half> hidden;
    for (uint32_t l = end_layer; l-- > 0;) {
        if (fs::exists(work_dir / ("hidden-after-" + std::to_string(l) + ".f16"))) {
            first_layer = l + 1;
            hidden = read_file<half>(work_dir / ("hidden-after-" + std::to_string(l) + ".f16"));
            break;
        }
    }
    if (first_layer == 0) {
        LayerDevice device(output_path.string(), 0, true, arch);
        Engine& engine = device.engine;
        int32_t* tokens = static_cast<int32_t*>(engine.tensors().allocate_and_register("ids", {1, PREFILL_CHUNK}, dtype::INT32(), true));
        engine.tensors().at("ids").shape = {1, 1};
        const CommandBuffer embed = device.builder.build_gemma4_input(engine, config, gemma, arch, "ids", 1, 1);
        hidden.resize(calibration.tokens * H);
        size_t row = 0;
        for (const auto& document : calibration.ids)
            for (int32_t id : document) {
                check(cudaMemcpy(tokens, &id, 4, cudaMemcpyHostToDevice), "token");
                engine.execute(embed);
                engine.sync();
                check(cudaMemcpy(hidden.data() + row++ * H, engine.tensors().at(HIDDEN).ptr, H * 2, cudaMemcpyDeviceToHost), "embedding");
            }
        std::cerr << "embedding de la calibración en " << seconds_since(start) << " s" << std::endl;
    } else {
        std::cerr << "reanudo en la capa " << first_layer << std::endl;
    }
    if (hidden.size() != calibration.tokens * H) throw std::runtime_error("estado de calibración incompleto");

    const int fd = ::open(output_path.c_str(), O_RDWR);
    if (fd < 0) throw std::runtime_error("no abro la salida para escribir");
    std::ofstream log(work_dir / "refine.log", std::ios::app);
    log << "inicio capa " << first_layer << " damp " << damp << " casos " << calibration.names.size()
        << " tokens " << calibration.tokens << '\n';

    for (uint32_t layer = first_layer; layer < end_layer; ++layer) {
        const auto layer_start = std::chrono::steady_clock::now();
        const ExpertTensor gate_up = expert_tensor(layer, "gate_up"), down = expert_tensor(layer, "down");
        if (gate_up.logical != H || gate_up.rows != 2 * I || down.rows != H || down.logical != I || gate_up.experts != E)
            throw std::runtime_error("geometría de expertos inesperada");

        LayerDevice device(output_path.string(), layer, false, arch);
        Engine& engine = device.engine;
        const KVCacheParams cache{"_kv", CACHE_BUILD_POSITION, MAX_SEQ};
        const CommandBuffer commands = device.builder.build_gemma4_layer_cached(engine, config, gemma, arch, layer, 1, 1, cache);
        half* d_hidden = static_cast<half*>(engine.tensors().at(HIDDEN).ptr);

        // Pasa todos los tokens por la capa; con `capture`, guarda la entrada de los expertos.
        std::vector<half> moe_input;
        std::vector<int32_t> routes;
        auto run_layer = [&](bool capture, std::vector<half>* out) {
            if (capture) { moe_input.resize(calibration.tokens * H); routes.resize(calibration.tokens * K); }
            if (out) out->resize(calibration.tokens * H);
            size_t row = 0;
            for (const auto& document : calibration.ids) {
                for (uint32_t position = 0; position < document.size(); ++position, ++row) {
                    check(cudaMemcpy(d_hidden, hidden.data() + row * H, H * 2, cudaMemcpyHostToDevice), "subir estado");
                    engine.update_device_cache_pos(int32_t(position), 1);
                    engine.execute(commands);
                    engine.sync();
                    if (capture) {
                        check(cudaMemcpy(moe_input.data() + row * H, engine.tensors().at("_s.normed").ptr, H * 2, cudaMemcpyDeviceToHost), "entrada MoE");
                        check(cudaMemcpy(routes.data() + row * K, engine.tensors().at("_s.g4.moe_indices").ptr, K * 4, cudaMemcpyDeviceToHost), "enrutado");
                    }
                    if (out) check(cudaMemcpy(out->data() + row * H, d_hidden, H * 2, cudaMemcpyDeviceToHost), "bajar estado");
                }
            }
        };
        run_layer(true, nullptr);
        const double capture_seconds = seconds_since(layer_start);

        // Tokens de cada experto.
        std::vector<std::vector<uint32_t>> rows_of(E);
        for (size_t t = 0; t < calibration.tokens; ++t)
            for (uint32_t k = 0; k < K; ++k) {
                const int32_t e = routes[t * K + k];
                if (e < 0 || uint32_t(e) >= E) throw std::runtime_error("enrutado fuera de rango");
                rows_of[size_t(e)].push_back(uint32_t(t));
            }
        size_t max_rows_routed = 0;
        for (const auto& r : rows_of) max_rows_routed = std::max(max_rows_routed, r.size());

        // Bytes actuales de los expertos: la rejilla de la que parte GPTQ.
        std::vector<uint8_t> gate_up_bytes(gate_up.size), down_bytes(down.size);
        pread_all(fd, gate_up_bytes.data(), gate_up.size, gate_up.offset);
        pread_all(fd, down_bytes.data(), down.size, down.offset);
        {
            // El motor guarda los expertos con los mismos bytes que el fichero.
            std::vector<uint8_t> probe(gate_up.stride());
            const std::string name = "text.layer" + std::to_string(layer) + ".moe.experts.gate_up.weight";
            check(cudaMemcpy(probe.data(), engine.tensors().at(name).ptr, probe.size(), cudaMemcpyDeviceToHost), "comprobar layout");
            if (std::memcmp(probe.data(), gate_up_bytes.data(), probe.size()) != 0)
                throw std::runtime_error("el layout de los expertos en el motor no es el del fichero");
        }

        gptq::Quantizer quantizer(int(std::max(gate_up.rows, down.rows)), int(H));
        double *d_x = nullptr, *d_h = nullptr, *d_w = nullptr, *d_g = nullptr, *d_wq = nullptr;
        float *d_a = nullptr, *d_m = nullptr;
        uint8_t* d_codes = nullptr;
        const size_t matrix = size_t(gate_up.rows) * gate_up.logical;
        check(cudaMalloc(&d_x, std::max<size_t>(max_rows_routed, 1) * H * sizeof(double)), "X");
        check(cudaMalloc(&d_h, size_t(H) * H * sizeof(double)), "H");
        check(cudaMalloc(&d_w, matrix * sizeof(double)), "W");
        check(cudaMalloc(&d_wq, matrix * sizeof(double)), "Wq");
        check(cudaMalloc(&d_g, std::max<size_t>(max_rows_routed, 1) * gate_up.rows * sizeof(double)), "G");
        check(cudaMalloc(&d_a, matrix * sizeof(float)), "a");
        check(cudaMalloc(&d_m, matrix * sizeof(float)), "m");
        check(cudaMalloc(&d_codes, matrix), "códigos");

        uint64_t refined = 0, untouched = 0;
        for (uint32_t e = 0; e < E; ++e) {
            const auto& rows = rows_of[e];
            if (rows.empty()) { ++untouched; continue; }
            const size_t n = rows.size();
            std::vector<double> x(n * H);
            for (size_t r = 0; r < n; ++r)
                for (uint32_t c = 0; c < H; ++c) x[r * H + c] = __half2float(moe_input[size_t(rows[r]) * H + c]);

            // gate/up.
            uint8_t* blocks = gate_up_bytes.data() + e * gate_up.stride();
            std::vector<float> a(matrix), m(matrix);
            gptq::hq44_grid(blocks, int(gate_up.rows), int(gate_up.physical), a.data(), m.data());
            std::vector<double> w = read_original(gate_up, e);
            hessian(quantizer.cublas(), x, n, int(H), d_x, d_h);
            check(cudaMemcpy(d_w, w.data(), matrix * 8, cudaMemcpyHostToDevice), "subir W");
            check(cudaMemcpy(d_a, a.data(), matrix * 4, cudaMemcpyHostToDevice), "subir a");
            check(cudaMemcpy(d_m, m.data(), matrix * 4, cudaMemcpyHostToDevice), "subir m");
            quantizer.run_device(d_w, d_h, d_a, d_m, int(gate_up.rows), int(H), damp, d_codes);
            std::vector<uint8_t> codes(matrix);
            check(cudaMemcpy(codes.data(), d_codes, matrix, cudaMemcpyDeviceToHost), "bajar códigos");
            gptq::hq44_set_codes(blocks, codes.data(), int(gate_up.rows), int(gate_up.physical), int(gate_up.logical));

            // Activaciones de down con el gate/up refinado, como la referencia: valores FP32.
            std::vector<double> wq(matrix);
            for (size_t i = 0; i < matrix; ++i) wq[i] = double(a[i] * float(codes[i]) - m[i]);
            check(cudaMemcpy(d_wq, wq.data(), matrix * 8, cudaMemcpyHostToDevice), "subir Wq");
            {
                // G[n, rows] = X · Wqᵀ; en orden de columnas, Gᵀ[rows, n] = Wq'·X' con Wq' = Wq en columnas.
                const double one = 1.0, zero = 0.0;
                check(cublasDgemm(quantizer.cublas(), CUBLAS_OP_T, CUBLAS_OP_N, int(gate_up.rows), int(n), int(H), &one,
                                  d_wq, int(H), d_x, int(H), &zero, d_g, int(gate_up.rows)), "G");
            }
            std::vector<double> g(n * gate_up.rows), act(n * I);
            check(cudaMemcpy(g.data(), d_g, g.size() * 8, cudaMemcpyDeviceToHost), "bajar G");
            for (size_t r = 0; r < n; ++r)
                for (uint32_t j = 0; j < I; ++j)
                    act[r * I + j] = gelu(g[r * gate_up.rows + j]) * g[r * gate_up.rows + I + j];

            // down.
            const size_t down_matrix = size_t(down.rows) * down.logical;
            uint8_t* down_blocks = down_bytes.data() + e * down.stride();
            std::vector<float> pa(size_t(down.rows) * down.physical), pm(pa.size()), da(down_matrix), dm(down_matrix);
            gptq::hq44_grid(down_blocks, int(down.rows), int(down.physical), pa.data(), pm.data());
            for (uint32_t r = 0; r < down.rows; ++r)
                for (uint32_t c = 0; c < down.logical; ++c) {
                    da[size_t(r) * down.logical + c] = pa[size_t(r) * down.physical + c];
                    dm[size_t(r) * down.logical + c] = pm[size_t(r) * down.physical + c];
                }
            std::vector<double> wd = read_original(down, e);
            hessian(quantizer.cublas(), act, n, int(I), d_x, d_h);
            check(cudaMemcpy(d_w, wd.data(), down_matrix * 8, cudaMemcpyHostToDevice), "subir W down");
            check(cudaMemcpy(d_a, da.data(), down_matrix * 4, cudaMemcpyHostToDevice), "subir a down");
            check(cudaMemcpy(d_m, dm.data(), down_matrix * 4, cudaMemcpyHostToDevice), "subir m down");
            quantizer.run_device(d_w, d_h, d_a, d_m, int(down.rows), int(I), damp, d_codes);
            std::vector<uint8_t> down_codes(down_matrix);
            check(cudaMemcpy(down_codes.data(), d_codes, down_matrix, cudaMemcpyDeviceToHost), "bajar códigos down");
            gptq::hq44_set_codes(down_blocks, down_codes.data(), int(down.rows), int(down.physical), int(down.logical));
            ++refined;
        }
        cudaFree(d_x); cudaFree(d_h); cudaFree(d_w); cudaFree(d_wq); cudaFree(d_g);
        cudaFree(d_a); cudaFree(d_m); cudaFree(d_codes);
        const double gptq_seconds = seconds_since(layer_start) - capture_seconds;

        // Al fichero y al motor, y la capa vuelve a pasar con los expertos nuevos.
        pwrite_all(fd, gate_up_bytes.data(), gate_up.size, gate_up.offset);
        pwrite_all(fd, down_bytes.data(), down.size, down.offset);
        if (::fsync(fd) != 0) throw std::runtime_error("fsync falló");
        const std::string prefix = "text.layer" + std::to_string(layer) + ".moe.experts.";
        check(cudaMemcpy(engine.tensors().at(prefix + "gate_up.weight").ptr, gate_up_bytes.data(), gate_up.size, cudaMemcpyHostToDevice), "expertos gate/up");
        check(cudaMemcpy(engine.tensors().at(prefix + "down.weight").ptr, down_bytes.data(), down.size, cudaMemcpyHostToDevice), "expertos down");
        std::vector<half> next;
        run_layer(false, &next);
        hidden.swap(next);
        write_file(work_dir / ("hidden-after-" + std::to_string(layer) + ".f16"), hidden);
        if (layer > 0) fs::remove(work_dir / ("hidden-after-" + std::to_string(layer - 1) + ".f16"));

        std::ostringstream line;
        line << "capa " << layer << ": " << refined << " expertos refinados, " << untouched << " sin tokens; captura "
             << capture_seconds << " s, GPTQ " << gptq_seconds << " s, total " << seconds_since(layer_start) << " s";
        std::cerr << line.str() << std::endl;
        log << line.str() << '\n';
        log.flush();
    }
    ::close(fd);
    log << "fin capa " << end_layer << " en " << seconds_since(start) << " s\n";
    std::cerr << "listo en " << seconds_since(start) << " s" << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
}
