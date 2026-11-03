// HNFv10: lector del contenedor (helios_convert_v10/docs/HNFV10.md).
//
// Comprueba siempre lo barato: CRC de las dos cabeceras, xxh3-128 de los metadatos, estado
// «en curso», estructura (payloads y tensores dentro de su sitio, sin solapes) y el hash
// compuesto de cada componente. No lee pesos: su hash por tensor lo comprueba
// `hnf10 verify` del conversor.
#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace helios { namespace hnf10 {

constexpr char MAGIC[8] = {'H', 'N', 'F', 'v', '1', '0', '\0', '\0'};

enum Kind : uint32_t { TEXT = 1, VISION = 2, AUDIO = 3, VIDEO = 4, TOKENIZER = 16, CONFIG = 17, TEMPLATE = 18 };

struct Component {
    uint32_t id = 0, kind = 0;
    bool required = false;
    std::string name;
    uint64_t offset = 0, size = 0;
    uint32_t first_tensor = 0, tensor_count = 0;
};

struct Tensor {
    std::string name, dtype;
    uint32_t component = 0;
    int32_t layer = -1;
    std::vector<uint32_t> shape;
    uint64_t offset = 0, size = 0;
    uint32_t experts = 0;
};

struct Property {
    uint32_t component = 0;
    std::string key;
    uint8_t type = 0;   // 1 bool, 2 entero, 3 real, 4 cadena, 5 bytes
    bool boolean = false;
    int64_t integer = 0;
    double real = 0.0;
    std::string text;
};

struct Contents {
    uint64_t sequence = 0, file_size = 0;
    std::vector<Component> components;
    std::vector<Tensor> tensors;
    std::vector<Property> properties;
    const Component* find(uint32_t kind) const {
        for (const auto& c : components) if (c.kind == kind) return &c;
        return nullptr;
    }
};

// ¿Empieza el fichero por el magic de HNFv10?
bool is_hnf10(std::ifstream& file);

// Lee y valida. Devuelve false con el motivo en `error`.
bool read(std::ifstream& file, Contents& out, std::string& error);

} } // helios::hnf10
