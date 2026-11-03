// Escritor HNFv10 mínimo para las pruebas del motor: mismo formato que el escritor de Rust
// (helios_convert_v10, src/file.rs y src/metadata.rs), para fabricar en la prueba ficheros con
// hints binarios hechos a medida. Solo cabecera A (secuencia 1, completa), sin procedencia ni
// fórmulas de memoria.
#pragma once

#include <zlib.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hnf10_test {

namespace detail {
// libxxhash del sistema, la misma ABI que xxhash.h 0.8 (ver src/hnf10.cpp).
extern "C" {
struct Hash128 { uint64_t low64, high64; };
Hash128 XXH3_128bits(const void* data, size_t length);
}
}  // namespace detail

inline std::vector<uint8_t> xxh3_128(const void* data, size_t length) {
    const detail::Hash128 h = detail::XXH3_128bits(data, length);
    std::vector<uint8_t> out(16);
    std::memcpy(out.data(), &h.low64, 8);
    std::memcpy(out.data() + 8, &h.high64, 8);
    return out;
}

enum Kind : uint32_t { TEXT = 1, VISION = 2, TOKENIZER = 16, CONFIG = 17, TEMPLATE = 18 };

struct Tensor {
    std::string name, dtype;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> bytes;
    int32_t layer = -1;
    uint32_t experts = 0;
};

struct Component {
    uint32_t id = 0, kind = 0;
    bool required = true;
    std::string name;
    std::vector<Tensor> tensors;     // componentes de pesos
    std::vector<uint8_t> bytes;      // tokenizer, configuración, plantilla
};

struct RealProperty { uint32_t component; std::string key; double value; };

class Out {
public:
    std::vector<uint8_t> b;
    void u8(uint8_t v) { b.push_back(v); }
    void u32(uint32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void f64(double v) { raw(&v, 8); }
    void raw(const void* p, size_t n) { const auto* c = static_cast<const uint8_t*>(p); b.insert(b.end(), c, c + n); }
};

inline uint64_t align(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// Devuelve el fichero completo.
inline std::vector<uint8_t> build(const std::vector<Component>& components, const std::vector<RealProperty>& properties = {}) {
    std::vector<uint8_t> file(4096, 0);
    std::string strings;
    auto ref = [&](Out& o, const std::string& s) { o.u32(uint32_t(strings.size())); o.u32(uint32_t(s.size())); strings += s; };
    struct Placed { uint64_t offset, size; std::vector<std::pair<uint64_t, std::vector<uint8_t>>> tensors; std::vector<uint8_t> hash; };
    std::vector<Placed> placed;
    for (const auto& c : components) {
        Placed p;
        p.offset = align(file.size(), 4096);
        file.resize(p.offset, 0);
        if (c.tensors.empty()) {
            file.insert(file.end(), c.bytes.begin(), c.bytes.end());
            p.hash = xxh3_128(c.bytes.data(), c.bytes.size());
        } else {
            for (const auto& t : c.tensors) {
                file.resize(align(file.size(), 32), 0);
                p.tensors.push_back({file.size(), xxh3_128(t.bytes.data(), t.bytes.size())});
                file.insert(file.end(), t.bytes.begin(), t.bytes.end());
            }
        }
        p.size = file.size() - p.offset;
        placed.push_back(std::move(p));
    }
    Out comps, tensors, props, empty;
    comps.u32(uint32_t(components.size()));
    uint32_t first = 0, total = 0;
    for (const auto& c : components) total += uint32_t(c.tensors.size());
    tensors.u32(total);
    for (size_t i = 0; i < components.size(); ++i) {
        const auto& c = components[i];
        Out composite;
        for (size_t k = 0; k < c.tensors.size(); ++k) {
            const auto& t = c.tensors[k];
            const auto& [offset, hash] = placed[i].tensors[k];
            ref(tensors, t.name); tensors.u32(c.id); tensors.u32(uint32_t(t.layer)); ref(tensors, t.dtype);
            tensors.u32(uint32_t(t.shape.size())); tensors.u32(0);
            for (size_t d = 0; d < 4; ++d) tensors.u64(d < t.shape.size() ? t.shape[d] : 0);
            tensors.u64(t.shape.back()); tensors.u64(offset); tensors.u64(t.bytes.size()); tensors.u32(t.experts); tensors.u32(0);
            tensors.raw(hash.data(), 16); ref(tensors, "prueba." + t.name); tensors.u64(0);
            composite.u64(offset); composite.u64(t.bytes.size()); composite.raw(hash.data(), 16);
        }
        comps.u32(c.id); comps.u32(c.kind); comps.u32(c.required ? 1 : 0); ref(comps, c.name);
        comps.u64(placed[i].offset); comps.u64(placed[i].size); comps.u32(4096); comps.u32(0);
        comps.u64(placed[i].size); comps.u32(0xFFFFFFFFu);
        comps.u32(first); comps.u32(uint32_t(c.tensors.size())); comps.u32(0xFFFFFFFFu);
        const auto composite_hash = xxh3_128(composite.b.data(), composite.b.size());
        comps.raw(composite_hash.data(), 16);
        if (c.tensors.empty()) comps.raw(placed[i].hash.data(), 16); else { std::vector<uint8_t> zero(16, 0); comps.raw(zero.data(), 16); }
        first += uint32_t(c.tensors.size());
    }
    props.u32(uint32_t(properties.size()));
    for (const auto& p : properties) { props.u32(p.component); ref(props, p.key); props.u8(3); props.f64(p.value); }
    empty.u32(0);
    const std::vector<std::pair<uint32_t, const std::vector<uint8_t>*>> sections = {
        {2, &comps.b}, {3, &tensors.b}, {4, &empty.b}, {5, &empty.b}, {6, &props.b}};
    std::vector<uint8_t> strings_bytes(strings.begin(), strings.end());
    Out meta;
    meta.raw("HNFM", 4); meta.u32(1); meta.u32(uint32_t(sections.size() + 1)); meta.u32(0);
    uint64_t at = 16 + 24 * (sections.size() + 1);
    meta.u32(1); meta.u32(0); meta.u64(at); meta.u64(strings_bytes.size()); at += strings_bytes.size();
    for (const auto& [type, body] : sections) { meta.u32(type); meta.u32(0); meta.u64(at); meta.u64(body->size()); at += body->size(); }
    meta.raw(strings_bytes.data(), strings_bytes.size());
    for (const auto& [type, body] : sections) meta.raw(body->data(), body->size());

    const uint64_t metadata_offset = align(file.size(), 4096);
    file.resize(metadata_offset, 0);
    file.insert(file.end(), meta.b.begin(), meta.b.end());
    std::vector<uint8_t> header(512, 0);
    std::memcpy(header.data(), "HNFv10\0\0", 8);
    const uint16_t major = 10, minor = 0;
    const uint32_t header_size = 512, state = 1;
    const uint64_t sequence = 1, size = meta.b.size(), file_size = file.size();
    std::memcpy(header.data() + 8, &major, 2);
    std::memcpy(header.data() + 10, &minor, 2);
    std::memcpy(header.data() + 12, &header_size, 4);
    std::memcpy(header.data() + 16, &sequence, 8);
    std::memcpy(header.data() + 24, &state, 4);
    std::memcpy(header.data() + 32, &metadata_offset, 8);
    std::memcpy(header.data() + 40, &size, 8);
    const auto meta_hash = xxh3_128(meta.b.data(), meta.b.size());
    std::memcpy(header.data() + 48, meta_hash.data(), 16);
    std::memcpy(header.data() + 64, &file_size, 8);
    const uint32_t crc = uint32_t(crc32(0L, header.data(), 508));
    std::memcpy(header.data() + 508, &crc, 4);
    std::memcpy(file.data(), header.data(), 512);
    return file;
}

inline std::string write(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("no creo " + path);
    out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    if (!out) throw std::runtime_error("no escribo " + path);
    return path;
}

}  // namespace hnf10_test
