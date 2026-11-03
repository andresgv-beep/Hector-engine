// Lector HNFv10 (src/hnf10.cpp) con ficheros escritos por el escritor de Rust
// (helios_convert_v10, `hnf10 fixtures`): mismo formato visto desde los dos lados.
// uso: test_hnf10 tests/data/hnf10
#include "hnf10.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace helios;
namespace fs = std::filesystem;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FALLO %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

static bool read(const fs::path& path, hnf10::Contents& out, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    out = hnf10::Contents{};
    return hnf10::is_hnf10(f) && hnf10::read(f, out, error);
}

static fs::path damaged(const fs::path& from, const char* name, uint64_t offset, size_t bytes) {
    const fs::path to = fs::temp_directory_path() / (std::string("hnf10-") + name + ".hnf");
    fs::copy_file(from, to, fs::copy_options::overwrite_existing);
    std::fstream f(to, std::ios::binary | std::ios::in | std::ios::out);
    std::vector<char> buf(bytes);
    f.seekg(std::streamoff(offset));
    f.read(buf.data(), std::streamsize(bytes));
    for (char& c : buf) c = char(~c);
    f.seekp(std::streamoff(offset));
    f.write(buf.data(), std::streamsize(bytes));
    return to;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "uso: test_hnf10 carpeta\n"); return 2; }
    const fs::path dir = argv[1], small = dir / "small.hnf";
    hnf10::Contents c;
    std::string error;

    // El fichero sellado: manda la cabecera B, secuencia 2.
    CHECK(read(small, c, error));
    CHECK(c.sequence == 2);
    CHECK(c.components.size() == 4);
    if (c.components.size() == 4) {
        CHECK(c.components[0].kind == hnf10::TEXT && c.components[0].required && c.components[0].tensor_count == 2);
        CHECK(c.components[1].kind == hnf10::TOKENIZER && c.components[1].size == 14);
        CHECK(c.components[2].kind == hnf10::CONFIG && c.components[2].name == "configuración");
        CHECK(c.components[3].kind == hnf10::TEMPLATE && !c.components[3].required);
    }
    CHECK(c.tensors.size() == 2);
    if (c.tensors.size() == 2) {
        CHECK(c.tensors[0].name == "text.layer0.attn.q_proj.weight" && c.tensors[0].layer == 0 && c.tensors[0].experts == 0);
        CHECK((c.tensors[0].shape == std::vector<uint32_t>{16, 32}) && c.tensors[0].offset == 4096 && c.tensors[0].size == 1024);
        CHECK(c.tensors[1].layer == 1 && c.tensors[1].experts == 4 && (c.tensors[1].shape == std::vector<uint32_t>{4, 16, 16}));
    }
    CHECK(c.properties.size() == 1);
    if (c.properties.size() == 1)
        CHECK(c.properties[0].component == 3 && c.properties[0].key == "final_logit_softcapping" &&
              c.properties[0].type == 3 && c.properties[0].real == 30.0);

    // Una cabecera «en curso» rechaza el fichero aunque la anterior esté completa.
    CHECK(!read(dir / "in_progress.hnf", c, error));
    CHECK(error.find("en curso") != std::string::npos);

    // Cabecera B rota: vuelve a la A, secuencia 1.
    CHECK(read(damaged(small, "b-rota", 512 + 100, 4), c, error) && c.sequence == 1);
    // Las dos rotas: ilegible.
    const fs::path both = damaged(damaged(small, "a-rota", 100, 4), "ambas", 512 + 100, 4);
    CHECK(!read(both, c, error));
    // Metadatos de B dañados: su hash no cuadra y vuelve a la A.
    {
        std::ifstream f(small, std::ios::binary);
        uint8_t header[512];
        f.seekg(512);
        f.read(reinterpret_cast<char*>(header), 512);
        uint64_t offset = 0;
        std::memcpy(&offset, header + 32, 8);
        CHECK(read(damaged(small, "meta-b", offset + 40, 1), c, error) && c.sequence == 1);
    }
    // No es HNFv10.
    {
        const fs::path other = fs::temp_directory_path() / "hnf10-otro.hnf";
        std::ofstream(other, std::ios::binary) << "HNFv9\0\0\0 y nada más";
        std::ifstream f(other, std::ios::binary);
        CHECK(!hnf10::is_hnf10(f));
    }

    std::printf(failures ? "FAILED (%d)\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
