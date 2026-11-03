// Refina con GPTQ los expertos HQ44 de un HNF de Gemma 4 MoE ya formado por el conversor.
//
// uso: hnf_refine entrada.hnf salida.hnf --model checkpoint/ --calibration casos/
//                 [--damp 1] [--work carpeta/] [--layers N]
//      hnf_refine entrada.hnf carpeta/ --importance --calibration casos/ [--layers N]
//
// --importance no refina: pasa la calibración por el HNF tal cual, que no se copia ni se
// modifica, y escribe en carpeta/ la importancia HQ44 de cada experto, la suma de x² por
// columna de la entrada de gate/up y de down, que lee `gemma4-moe-experimental
// --importance`. Los expertos con pocos tokens se mezclan con la media de su capa como
// si esta aportara 64 tokens, igual que la referencia de CPU (Hq44Importance).
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
//
// Con un HNFv10 la salida no se crea aquí: `hnf10 begin-edit --from entrada --to salida` la
// copia y la deja «en curso», las capas se cargan de la entrada (completa) y los códigos se
// escriben en la salida; al terminar, `hnf10 seal salida --record trabajo/record.json`.
// Reanudable: en --work quedan el estado tras cada capa terminada y un registro.
#include "gemma4_kv_cache.hpp"
#include "gemma4_split.hpp"
#include "gptq.hpp"
#include "graph_builder.hpp"
#include "hnf10.hpp"
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
#include <deque>
#include <fstream>
#include <future>
#include <thread>
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

// FNV-1a de 64 bits de los IDs de calibración, en orden: identifica el corpus en el registro.
uint64_t fingerprint(const Calibration& c) {
    uint64_t h = 1469598103934665603ull;
    for (const auto& document : c.ids)
        for (int32_t id : document)
            for (int b = 0; b < 4; ++b) { h ^= uint8_t(uint32_t(id) >> (8 * b)); h *= 1099511628211ull; }
    return h;
}

std::string hex64(uint64_t v) { char b[17]; std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(v)); return b; }

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
    bool importance_only = false;
    for (int i = 3; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--importance") { importance_only = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("falta el valor de " + flag);
        ++i;
        if (flag == "--model") model_dir = argv[i];
        else if (flag == "--calibration") calibration_dir = argv[i];
        else if (flag == "--damp") damp = std::atof(argv[i]);
        else if (flag == "--work") work_dir = argv[i];
        else if (flag == "--layers") layer_limit = uint32_t(std::atoi(argv[i]));
        else throw std::runtime_error("opción desconocida " + flag);
    }
    if (calibration_dir.empty() || (!importance_only && model_dir.empty())) throw std::runtime_error("faltan --model o --calibration");
    if (work_dir.empty()) work_dir = importance_only ? output_path / "estado" : fs::path(output_path.string() + ".refine");
    fs::create_directories(work_dir);
    const auto start = std::chrono::steady_clock::now();
    bool v10 = false;
    { std::ifstream probe(input_path, std::ios::binary); v10 = hnf10::is_hnf10(probe); }
    // El HNF que se carga: con v9, la copia que se refina; con v10 y al medir importancia, la
    // entrada tal cual (la salida v10 está «en curso» y el cargador no la acepta, con razón).
    const fs::path model_path = (importance_only || v10) ? input_path : output_path;
    hnf10::Contents index10;
    if (v10) {
        std::ifstream in(input_path, std::ios::binary);
        std::string error;
        if (!hnf10::read(in, index10, error)) throw std::runtime_error(error);
        if (!importance_only) {
            if (!fs::exists(output_path)) throw std::runtime_error("con HNFv10, crea antes la salida con: hnf10 begin-edit --from entrada --to salida");
            std::ifstream out(output_path, std::ios::binary);
            hnf10::Contents edited;
            if (!hnf10::is_hnf10(out) || !hnf10::read(out, edited, error, true)) throw std::runtime_error("salida: " + error);
            if (edited.state != 2) throw std::runtime_error("la salida no está en edición: crea una con hnf10 begin-edit");
            if (edited.tensors.size() != index10.tensors.size()) throw std::runtime_error("la salida no es una copia de la entrada");
            for (size_t i = 0; i < edited.tensors.size(); ++i)
                if (edited.tensors[i].name != index10.tensors[i].name || edited.tensors[i].offset != index10.tensors[i].offset ||
                    edited.tensors[i].size != index10.tensors[i].size)
                    throw std::runtime_error("la salida no tiene el índice de la entrada: " + edited.tensors[i].name);
        }
    }

    // Salida v9: copia de la entrada, que nunca se modifica.
    if (!v10 && !importance_only && !fs::exists(output_path)) {
        std::cerr << "copiando " << input_path << " -> " << output_path << std::endl;
        const fs::path part = output_path.string() + ".part";
        fs::copy_file(input_path, part, fs::copy_options::overwrite_existing);
        fs::rename(part, output_path);
    }
    if (!importance_only && fs::file_size(output_path) != fs::file_size(input_path))
        throw std::runtime_error("la salida no tiene el tamaño de la entrada");

    Json manifest;
    std::map<std::string, const Json*> tensors, sources;
    if (!v10) {
        manifest = read_manifest(model_path);
        for (const Json& t : manifest.at("tensors").array) tensors[t.at("name").string] = &t;
        for (const Json& s : manifest.at("source_tensors").array) sources[s.at("canonical").string] = &s;
    }

    HnfLoader meta;
    if (!meta.open(model_path.string())) throw std::runtime_error(meta.last_error());
    const ModelConfig& config = meta.config();
    const Gemma4Config& gemma = meta.gemma4_config();
    const uint32_t layers = config.num_hidden_layers(), H = config.hidden_size();
    const uint32_t E = gemma.moe_experts, K = gemma.moe_top_k, I = gemma.moe_intermediate_size;
    const uint32_t end_layer = layer_limit ? std::min(layer_limit, layers) : layers;
    const ArchDescriptor arch = gemma4_split_arch(layers);

    std::unique_ptr<Checkpoint> checkpoint;
    if (!importance_only) checkpoint = std::make_unique<Checkpoint>(model_dir);
    auto expert_tensor = [&](uint32_t layer, const char* kind) {
        const std::string name = "text.layer" + std::to_string(layer) + ".moe.experts." + kind + ".weight";
        if (v10) {
            for (const auto& t : index10.tensors) {
                if (t.name != name) continue;
                if (t.dtype != "hq44k_g16" || t.shape.size() != 3 || t.experts != t.shape[0]) throw std::runtime_error(name + " no es un HQ44 de expertos");
                ExpertTensor out;
                out.offset = t.offset;
                out.size = t.size;
                out.experts = t.shape[0];
                out.rows = t.shape[1];
                out.physical = t.shape[2];
                out.logical = uint32_t(t.logical_last);
                if (out.stride() * out.experts != out.size) throw std::runtime_error(name + ": tamaño inesperado");
                if (checkpoint) checkpoint->locate(t.source, out);
                return out;
            }
            throw std::runtime_error("HNFv10 sin " + name);
        }
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
        if (checkpoint) checkpoint->locate(s.at("source").string, out);
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
        LayerDevice device(model_path.string(), 0, true, arch);
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

    // Se leen y escriben los códigos en la salida (v9: también es lo que se carga; v10: la copia
    // en edición). Al medir importancia, solo se lee la entrada.
    const fs::path written = importance_only ? input_path : output_path;
    const int fd = ::open(written.c_str(), importance_only ? O_RDONLY : O_RDWR);
    if (fd < 0) throw std::runtime_error("no abro " + written.string());
    std::ofstream log(work_dir / (importance_only ? "importance.log" : "refine.log"), std::ios::app);
    log << "inicio capa " << first_layer << " damp " << damp << " casos " << calibration.names.size()
        << " tokens " << calibration.tokens << '\n';

    for (uint32_t layer = first_layer; layer < end_layer; ++layer) {
        const auto layer_start = std::chrono::steady_clock::now();
        const ExpertTensor gate_up = expert_tensor(layer, "gate_up"), down = expert_tensor(layer, "down");
        if (gate_up.logical != H || gate_up.rows != 2 * I || down.rows != H || down.logical != I || gate_up.experts != E)
            throw std::runtime_error("geometría de expertos inesperada");

        LayerDevice device(model_path.string(), layer, false, arch);
        Engine& engine = device.engine;
        const KVCacheParams cache{"_kv", CACHE_BUILD_POSITION, MAX_SEQ};
        const CommandBuffer commands = device.builder.build_gemma4_layer_cached(engine, config, gemma, arch, layer, 1, 1, cache);
        half* d_hidden = static_cast<half*>(engine.tensors().at(HIDDEN).ptr);
        const half* d_normed = static_cast<const half*>(engine.tensors().at("_s.normed").ptr);
        const int32_t* d_indices = static_cast<const int32_t*>(engine.tensors().at("_s.g4.moe_indices").ptr);
        // Entrada de down de cada ranura, [K, columnas físicas], tal como la lee el kernel.
        const half* d_slot_hidden = static_cast<const half*>(engine.tensors().at("_s.g4.moe_hidden").ptr);
        const uint32_t P = down.physical;
        const cudaStream_t stream = engine.config().stream;
        const size_t T = calibration.tokens;
        half *d_states_in = nullptr, *d_states_out = nullptr, *d_moe_input = nullptr;
        int32_t* d_routes = nullptr;
        check(cudaMalloc(&d_states_in, T * H * 2), "estados de entrada");
        check(cudaMalloc(&d_states_out, T * H * 2), "estados de salida");
        check(cudaMalloc(&d_moe_input, T * H * 2), "entradas MoE");
        check(cudaMalloc(&d_routes, T * K * 4), "enrutados");
        half* d_down_input = nullptr;
        if (importance_only) check(cudaMalloc(&d_down_input, T * K * P * 2), "entradas de down");
        // Posición y longitud de cada token, ya en la GPU: copiarlas de dispositivo a
        // dispositivo no sincroniza el stream, a diferencia de update_device_cache_pos.
        int32_t* d_positions = nullptr;
        {
            std::vector<int32_t> positions(2 * T);
            size_t row = 0;
            for (const auto& document : calibration.ids)
                for (uint32_t position = 0; position < document.size(); ++position, ++row) {
                    positions[2 * row] = int32_t(position);
                    positions[2 * row + 1] = int32_t(position) + 1;
                }
            check(cudaMalloc(&d_positions, positions.size() * 4), "posiciones");
            check(cudaMemcpy(d_positions, positions.data(), positions.size() * 4, cudaMemcpyHostToDevice), "subir posiciones");
        }
        engine.alloc_device_cache_pos();
        int32_t* d_cache_pos = engine.device_cache_pos();
        int32_t* d_total_seq = engine.device_total_seq();

        // Pasa todos los tokens por la capa, uno a uno y con KV como en decode, con la
        // capa capturada en un CUDA Graph. Las entradas y salidas se quedan en la GPU
        // hasta el final de la pasada; con `capture`, también la entrada de los expertos.
        std::vector<half> moe_input;
        std::vector<int32_t> routes;
        auto run_layer = [&](bool capture, std::vector<half>* out) {
            check(cudaMemcpy(d_states_in, hidden.data(), T * H * 2, cudaMemcpyHostToDevice), "subir estados");
            size_t row = 0;
            for (const auto& document : calibration.ids) {
                for (uint32_t position = 0; position < document.size(); ++position, ++row) {
                    check(cudaMemcpyAsync(d_hidden, d_states_in + row * H, H * 2, cudaMemcpyDeviceToDevice, stream), "estado");
                    check(cudaMemcpyAsync(d_cache_pos, d_positions + 2 * row, 4, cudaMemcpyDeviceToDevice, stream), "posición");
                    check(cudaMemcpyAsync(d_total_seq, d_positions + 2 * row + 1, 4, cudaMemcpyDeviceToDevice, stream), "longitud");
                    engine.execute_graph_replay(commands);
                    if (capture) {
                        check(cudaMemcpyAsync(d_moe_input + row * H, d_normed, H * 2, cudaMemcpyDeviceToDevice, stream), "entrada MoE");
                        check(cudaMemcpyAsync(d_routes + row * K, d_indices, K * 4, cudaMemcpyDeviceToDevice, stream), "enrutado");
                        if (d_down_input)
                            check(cudaMemcpyAsync(d_down_input + row * K * P, d_slot_hidden, K * P * 2, cudaMemcpyDeviceToDevice, stream), "entrada de down");
                    }
                    if (out) check(cudaMemcpyAsync(d_states_out + row * H, d_hidden, H * 2, cudaMemcpyDeviceToDevice, stream), "salida");
                }
            }
            check(cudaStreamSynchronize(stream), "pasada");
            if (capture) {
                moe_input.resize(T * H); routes.resize(T * K);
                check(cudaMemcpy(moe_input.data(), d_moe_input, T * H * 2, cudaMemcpyDeviceToHost), "bajar entradas MoE");
                check(cudaMemcpy(routes.data(), d_routes, T * K * 4, cudaMemcpyDeviceToHost), "bajar enrutados");
            }
            if (out) {
                out->resize(T * H);
                check(cudaMemcpy(out->data(), d_states_out, T * H * 2, cudaMemcpyDeviceToHost), "bajar estados");
            }
        };
        if (importance_only) {
            std::vector<half> next, down_input(T * K * P);
            run_layer(true, &next);
            check(cudaMemcpy(down_input.data(), d_down_input, down_input.size() * 2, cudaMemcpyDeviceToHost), "bajar entradas de down");
            cudaFree(d_down_input); cudaFree(d_moe_input); cudaFree(d_routes);
            cudaFree(d_states_in); cudaFree(d_states_out); cudaFree(d_positions);
            // Sumas de x² por experto y columna, en FP64, y tokens de cada experto.
            std::vector<double> sum_gu(size_t(E) * H, 0.0), sum_dn(size_t(E) * I, 0.0);
            std::vector<uint64_t> count(E, 0);
            for (size_t t = 0; t < T; ++t)
                for (uint32_t k = 0; k < K; ++k) {
                    const int32_t e = routes[t * K + k];
                    if (e < 0 || uint32_t(e) >= E) throw std::runtime_error("enrutado fuera de rango");
                    ++count[size_t(e)];
                    double* g = sum_gu.data() + size_t(e) * H;
                    const half* x = moe_input.data() + t * H;
                    for (uint32_t c = 0; c < H; ++c) { const double v = __half2float(x[c]); g[c] += v * v; }
                    double* d = sum_dn.data() + size_t(e) * I;
                    const half* a = down_input.data() + (t * K + k) * P;
                    for (uint32_t c = 0; c < I; ++c) { const double v = __half2float(a[c]); d[c] += v * v; }
                }
            // (S + 64·media de la capa) / (n + 64), rellenado a las columnas físicas.
            constexpr double SMOOTHING = 64.0;
            uint64_t tokens = 0;
            for (uint64_t c : count) tokens += c;
            auto export_kind = [&](const std::vector<double>& sum, uint32_t cols, uint32_t physical, const char* kind) {
                std::vector<double> mean(cols, 0.0);
                for (uint32_t e = 0; e < E; ++e)
                    for (uint32_t c = 0; c < cols; ++c) mean[c] += sum[size_t(e) * cols + c];
                for (double& v : mean) v /= double(std::max<uint64_t>(tokens, 1));
                std::vector<float> out(size_t(E) * physical, 0.0f);
                for (uint32_t e = 0; e < E; ++e)
                    for (uint32_t c = 0; c < cols; ++c)
                        out[size_t(e) * physical + c] = float((sum[size_t(e) * cols + c] + SMOOTHING * mean[c]) / (double(count[e]) + SMOOTHING));
                write_file(output_path / ("layer" + std::to_string(layer) + "." + kind + ".f32"), out);
            };
            export_kind(sum_gu, H, gate_up.physical, "gate_up");
            export_kind(sum_dn, I, down.physical, "down");
            hidden.swap(next);
            write_file(work_dir / ("hidden-after-" + std::to_string(layer) + ".f16"), hidden);
            if (layer > 0) fs::remove(work_dir / ("hidden-after-" + std::to_string(layer - 1) + ".f16"));
            uint32_t sparse = 0;
            for (uint64_t c : count) sparse += c < 64;
            std::ostringstream line;
            line << "capa " << layer << ": importancia de " << E << " expertos, " << sparse << " con menos de 64 tokens, "
                 << seconds_since(layer_start) << " s";
            std::cerr << line.str() << std::endl;
            log << line.str() << '\n';
            log.flush();
            continue;
        }
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

        // Lo que cada experto necesita del host: sus tokens en FP64, los pesos originales y
        // las dos rejillas. Se prepara en hilos varios expertos por delante de la GPU. Solo
        // lee cabeceras de bloques, que GPTQ no toca, mientras la GPU escribe códigos de otro.
        const size_t down_matrix = size_t(down.rows) * down.logical;
        struct Prepared {
            std::vector<double> x, w, wd;
            std::vector<float> a, m, da, dm;
        };
        auto prepare = [&](uint32_t e) {
            Prepared p;
            const auto& rows = rows_of[e];
            p.x.resize(rows.size() * H);
            for (size_t r = 0; r < rows.size(); ++r)
                for (uint32_t c = 0; c < H; ++c) p.x[r * H + c] = __half2float(moe_input[size_t(rows[r]) * H + c]);
            p.a.resize(matrix); p.m.resize(matrix);
            gptq::hq44_grid(gate_up_bytes.data() + e * gate_up.stride(), int(gate_up.rows), int(gate_up.physical), p.a.data(), p.m.data());
            p.w = read_original(gate_up, e);
            std::vector<float> pa(size_t(down.rows) * down.physical), pm(pa.size());
            gptq::hq44_grid(down_bytes.data() + e * down.stride(), int(down.rows), int(down.physical), pa.data(), pm.data());
            p.da.resize(down_matrix); p.dm.resize(down_matrix);
            for (uint32_t r = 0; r < down.rows; ++r)
                for (uint32_t c = 0; c < down.logical; ++c) {
                    p.da[size_t(r) * down.logical + c] = pa[size_t(r) * down.physical + c];
                    p.dm[size_t(r) * down.logical + c] = pm[size_t(r) * down.physical + c];
                }
            p.wd = read_original(down, e);
            return p;
        };
        std::vector<uint32_t> busy;
        for (uint32_t e = 0; e < E; ++e) if (!rows_of[e].empty()) busy.push_back(e);
        constexpr size_t AHEAD = 4;
        std::deque<std::future<Prepared>> ready;
        size_t launched = 0;
        auto launch = [&]() { if (launched < busy.size()) { const uint32_t e = busy[launched++]; ready.push_back(std::async(std::launch::async, prepare, e)); } };
        for (size_t i = 0; i < AHEAD; ++i) launch();
        const unsigned workers = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
        auto parallel_rows = [&](size_t count, auto body) {
            std::vector<std::thread> pool;
            for (unsigned t = 0; t < workers; ++t)
                pool.emplace_back([&, t] { for (size_t r = t; r < count; r += workers) body(r); });
            for (auto& thread : pool) thread.join();
        };

        uint64_t refined = 0;
        const uint64_t untouched = E - busy.size();
        for (const uint32_t e : busy) {
            Prepared p = ready.front().get();
            ready.pop_front();
            launch();
            const size_t n = rows_of[e].size();
            std::vector<double>& x = p.x;
            std::vector<float>& a = p.a;
            std::vector<float>& m = p.m;

            // gate/up.
            uint8_t* blocks = gate_up_bytes.data() + e * gate_up.stride();
            const std::vector<double>& w = p.w;
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
            parallel_rows(gate_up.rows, [&](size_t r) {
                for (size_t i = r * H; i < (r + 1) * H; ++i) wq[i] = double(a[i] * float(codes[i]) - m[i]);
            });
            check(cudaMemcpy(d_wq, wq.data(), matrix * 8, cudaMemcpyHostToDevice), "subir Wq");
            {
                // G[n, rows] = X · Wqᵀ; en orden de columnas, Gᵀ[rows, n] = Wq'·X' con Wq' = Wq en columnas.
                const double one = 1.0, zero = 0.0;
                check(cublasDgemm(quantizer.cublas(), CUBLAS_OP_T, CUBLAS_OP_N, int(gate_up.rows), int(n), int(H), &one,
                                  d_wq, int(H), d_x, int(H), &zero, d_g, int(gate_up.rows)), "G");
            }
            std::vector<double> g(n * gate_up.rows), act(n * I);
            check(cudaMemcpy(g.data(), d_g, g.size() * 8, cudaMemcpyDeviceToHost), "bajar G");
            parallel_rows(n, [&](size_t r) {
                for (uint32_t j = 0; j < I; ++j)
                    act[r * I + j] = gelu(g[r * gate_up.rows + j]) * g[r * gate_up.rows + I + j];
            });

            // down.
            uint8_t* down_blocks = down_bytes.data() + e * down.stride();
            const std::vector<float>& da = p.da;
            const std::vector<float>& dm = p.dm;
            const std::vector<double>& wd = p.wd;
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
        cudaFree(d_moe_input); cudaFree(d_routes);
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
        cudaFree(d_states_in); cudaFree(d_states_out); cudaFree(d_positions);
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
    // Registro para `hnf-seal` del conversor (refinado) o para quien lea la importancia.
    {
        std::ostringstream record;
        record << "{\n \"tool\": \"hnf_refine\",\n \"mode\": \"" << (importance_only ? "importance" : "gptq") << "\",\n"
               << " \"input\": {\"path\": \"" << input_path.string() << "\", \"bytes\": " << fs::file_size(input_path) << "},\n"
               << " \"layers\": [0, " << end_layer << "],\n";
        if (importance_only) record << " \"smoothing_tokens\": 64,\n \"statistic\": \"sum x^2 per column of gate_up and down input, fp16 activations from the engine\",\n";
        else record << " \"damp\": " << damp << ",\n \"grid\": \"hnf_existing_headers\",\n \"arithmetic\": \"fp64 cuSOLVER/cuBLAS, block 128\",\n";
        record << " \"calibration\": {\"cases\": " << calibration.names.size() << ", \"tokens\": " << calibration.tokens
               << ", \"max_tokens_per_case\": " << MAX_FORCING << ", \"ids_fnv1a64\": \"" << hex64(fingerprint(calibration)) << "\"},\n"
               << " \"complete\": " << (end_layer == layers ? "true" : "false") << "\n}\n";
        const fs::path where = importance_only ? output_path / "record.json" : work_dir / "record.json";
        std::ofstream(where) << record.str();
    }
    log << "fin capa " << end_layer << " en " << seconds_since(start) << " s\n";
    std::cerr << "listo en " << seconds_since(start) << " s" << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
}
