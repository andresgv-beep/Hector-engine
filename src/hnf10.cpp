// HNFv10: lector del contenedor. Ver hnf10.hpp.
#include "hnf10.hpp"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

// libxxhash 0.8 del sistema, sin cabecera de desarrollo: la misma ABI que xxhash.h.
extern "C" {
struct XXH128_hash_t { uint64_t low64, high64; };
XXH128_hash_t XXH3_128bits(const void* data, size_t length);
}

namespace helios { namespace hnf10 {
namespace {

constexpr size_t HEADER = 512;
constexpr uint64_t PAYLOAD_START = 4096, ALIGN = 4096;
constexpr uint32_t OPTIONAL_SECTION = 1u << 31, NONE = 0xFFFFFFFFu;

using Hash = std::array<uint8_t, 16>;

Hash xxh3_128(const void* data, size_t length) {
    const XXH128_hash_t h = XXH3_128bits(data, length);
    Hash out;
    std::memcpy(out.data(), &h.low64, 8);
    std::memcpy(out.data() + 8, &h.high64, 8);
    return out;
}

template <typename T> T at(const uint8_t* p) { T v; std::memcpy(&v, p, sizeof(T)); return v; }

struct Header {
    uint64_t sequence = 0, metadata_offset = 0, metadata_size = 0, file_size = 0;
    uint32_t state = 0;
    Hash metadata_hash{};
};

bool parse_header(const uint8_t* b, Header& h) {
    if (std::memcmp(b, MAGIC, 8) != 0 || at<uint16_t>(b + 8) != 10 || at<uint32_t>(b + 12) != HEADER) return false;
    if (at<uint32_t>(b + 508) != uint32_t(crc32(0L, b, 508))) return false;
    h.sequence = at<uint64_t>(b + 16);
    h.state = at<uint32_t>(b + 24);
    h.metadata_offset = at<uint64_t>(b + 32);
    h.metadata_size = at<uint64_t>(b + 40);
    std::memcpy(h.metadata_hash.data(), b + 48, 16);
    h.file_size = at<uint64_t>(b + 64);
    return (h.state == 1 || h.state == 2) && h.metadata_offset >= PAYLOAD_START && h.metadata_offset % ALIGN == 0 &&
           h.metadata_offset + h.metadata_size <= h.file_size;
}

// Lector acotado de una sección, con las cadenas de STRINGS.
struct In {
    const uint8_t* b; size_t size, pos = 0;
    const uint8_t* strings; size_t strings_size;
    bool ok = true;
    const uint8_t* take(size_t n) { if (!ok || pos + n > size) { ok = false; return nullptr; } const uint8_t* p = b + pos; pos += n; return p; }
    uint8_t u8() { const uint8_t* p = take(1); return p ? *p : 0; }
    uint32_t u32() { const uint8_t* p = take(4); return p ? at<uint32_t>(p) : 0; }
    uint64_t u64() { const uint8_t* p = take(8); return p ? at<uint64_t>(p) : 0; }
    double f64() { const uint8_t* p = take(8); return p ? at<double>(p) : 0.0; }
    Hash h() { Hash v{}; const uint8_t* p = take(16); if (p) std::memcpy(v.data(), p, 16); return v; }
    std::string s() {
        const uint32_t o = u32(), l = u32();
        if (!ok || uint64_t(o) + l > strings_size) { ok = false; return {}; }
        return std::string(reinterpret_cast<const char*>(strings) + o, l);
    }
};

bool known(uint32_t kind) { return kind == TEXT || kind == VISION || kind == AUDIO || kind == VIDEO || kind == TOKENIZER || kind == CONFIG || kind == TEMPLATE; }

}  // namespace

bool is_hnf10(std::ifstream& file) {
    char magic[8] = {};
    file.clear();
    file.seekg(0);
    file.read(magic, 8);
    const bool yes = file.good() && std::memcmp(magic, MAGIC, 8) == 0;
    file.clear();
    file.seekg(0);
    return yes;
}

bool read(std::ifstream& file, Contents& out, std::string& error) {
    file.clear();
    file.seekg(0, std::ios::end);
    const uint64_t length = uint64_t(file.tellg());
    uint8_t slots[2][HEADER] = {};
    file.seekg(0);
    file.read(reinterpret_cast<char*>(slots), sizeof(slots));
    if (!file.good()) { error = "HNFv10 más corto que sus cabeceras"; return false; }

    // La válida con mayor secuencia; «en curso» rechaza el fichero aunque la otra esté completa.
    bool found = false;
    Header best;
    std::vector<uint8_t> metadata;
    for (const auto& slot : slots) {
        Header h;
        if (!parse_header(slot, h) || h.file_size > length) continue;
        std::vector<uint8_t> bytes(h.metadata_size);
        file.clear();
        file.seekg(std::streamoff(h.metadata_offset));
        file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
        if (!file.good() || xxh3_128(bytes.data(), bytes.size()) != h.metadata_hash) continue;
        if (!found || h.sequence > best.sequence) { best = h; metadata.swap(bytes); found = true; }
    }
    if (!found) { error = "HNFv10 sin ninguna cabecera válida"; return false; }
    if (best.state != 1) { error = "HNFv10 con una escritura de pesos en curso: repararlo antes de usarlo"; return false; }
    out.sequence = best.sequence;
    out.file_size = best.file_size;

    // Tabla de secciones.
    const uint8_t* m = metadata.data();
    if (metadata.size() < 16 || std::memcmp(m, "HNFM", 4) != 0 || at<uint32_t>(m + 4) != 1) { error = "metadatos HNFv10 sin magic o de otra versión"; return false; }
    const uint32_t count = at<uint32_t>(m + 8);
    if (16 + 24ull * count > metadata.size()) { error = "tabla de secciones truncada"; return false; }
    std::map<uint32_t, std::pair<const uint8_t*, size_t>> sections;
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t* e = m + 16 + 24 * i;
        const uint32_t type = at<uint32_t>(e);
        const uint64_t offset = at<uint64_t>(e + 8), size = at<uint64_t>(e + 16);
        if (offset + size > metadata.size()) { error = "sección fuera de los metadatos"; return false; }
        if (type >= 1 && type <= 6) sections[type] = {m + offset, size_t(size)};
        else if (!(type & OPTIONAL_SECTION)) { error = "sección obligatoria desconocida en HNFv10"; return false; }
    }
    for (uint32_t type : {1u, 2u, 3u, 6u}) if (!sections.count(type)) { error = "falta una sección de metadatos HNFv10"; return false; }
    const auto [strings, strings_size] = sections[1];

    In c{sections[2].first, sections[2].second, 0, strings, strings_size};
    std::vector<Hash> composites;
    for (uint32_t n = c.u32(), i = 0; i < n && c.ok; ++i) {
        Component x;
        x.id = c.u32(); x.kind = c.u32(); x.required = (c.u32() & 1) != 0; x.name = c.s();
        x.offset = c.u64(); x.size = c.u64(); const uint32_t alignment = c.u32(); c.u32();
        c.u64(); c.u32();                                   // bytes de pesos, fórmula de memoria
        x.first_tensor = c.u32(); x.tensor_count = c.u32(); c.u32();   // depende de
        composites.push_back(c.h()); c.h();
        if (!known(x.kind) && x.required) { error = "componente obligatorio de tipo desconocido: " + x.name; return false; }
        if (alignment == 0 || (alignment & (alignment - 1)) != 0 || (x.size > 0 && (x.offset < PAYLOAD_START || x.offset % alignment != 0)) ||
            x.offset + x.size > best.metadata_offset) {
            error = "payload fuera de sitio: " + x.name; return false;
        }
        out.components.push_back(x);
    }
    if (!c.ok || c.pos != c.size) { error = "sección de componentes mal formada"; return false; }

    In t{sections[3].first, sections[3].second, 0, strings, strings_size};
    std::vector<Hash> hashes;
    for (uint32_t n = t.u32(), i = 0; i < n && t.ok; ++i) {
        Tensor x;
        x.name = t.s(); x.component = t.u32(); x.layer = int32_t(t.u32()); x.dtype = t.s();
        const uint32_t rank = t.u32(); t.u32();
        for (uint32_t d = 0; d < 4; ++d) { const uint64_t v = t.u64(); if (d < rank) x.shape.push_back(uint32_t(v)); }
        t.u64();                                            // columnas lógicas
        x.offset = t.u64(); x.size = t.u64(); x.experts = t.u32(); t.u32();
        hashes.push_back(t.h()); t.s(); t.u64();            // origen y su hash
        if (rank < 1 || rank > 4) { error = "rango de tensor no admitido: " + x.name; return false; }
        out.tensors.push_back(std::move(x));
    }
    if (!t.ok || t.pos != t.size) { error = "sección de tensores mal formada"; return false; }

    // Tensores dentro de su componente, sin solapes, y compuesto de cada componente.
    for (size_t i = 0; i < out.components.size(); ++i) {
        const Component& x = out.components[i];
        if (uint64_t(x.first_tensor) + x.tensor_count > out.tensors.size()) { error = "tensores fuera del índice: " + x.name; return false; }
        std::vector<uint8_t> buf;
        for (uint32_t k = x.first_tensor; k < x.first_tensor + x.tensor_count; ++k) {
            const Tensor& y = out.tensors[k];
            if (y.component != x.id || y.offset < x.offset || y.offset + y.size > x.offset + x.size) {
                error = "tensor fuera de su componente: " + y.name; return false;
            }
            const uint8_t* o = reinterpret_cast<const uint8_t*>(&y.offset);
            const uint8_t* s = reinterpret_cast<const uint8_t*>(&y.size);
            buf.insert(buf.end(), o, o + 8);
            buf.insert(buf.end(), s, s + 8);
            buf.insert(buf.end(), hashes[k].begin(), hashes[k].end());
        }
        if (xxh3_128(buf.data(), buf.size()) != composites[i]) { error = "el compuesto no cuadra con sus tensores: " + x.name; return false; }
    }
    // Cada tensor en exactamente un rango de componente y dentro de la zona de payloads.
    std::vector<uint32_t> owners(out.tensors.size(), 0);
    for (const auto& x : out.components)
        for (uint32_t k = x.first_tensor; k < x.first_tensor + x.tensor_count; ++k) ++owners[k];
    for (size_t k = 0; k < out.tensors.size(); ++k) {
        const Tensor& y = out.tensors[k];
        if (owners[k] != 1) { error = "tensor en " + std::to_string(owners[k]) + " rangos de componente: " + y.name; return false; }
        if (y.offset < PAYLOAD_START || y.offset + y.size < y.offset || y.offset + y.size > best.metadata_offset) {
            error = "tensor fuera de la zona de payloads: " + y.name; return false;
        }
    }
    // Fórmulas de memoria: variables 1–4, exponentes 1–3, coeficientes finitos y no
    // negativos, y un máximo para cada variable que se use.
    if (sections.count(4)) {
        In f{sections[4].first, sections[4].second, 0, strings, strings_size};
        for (uint32_t n = f.u32(), i = 0; i < n && f.ok; ++i) {
            f.u64();
            const uint32_t terms = f.u32(), maxima = f.u32();
            struct Term { uint32_t variable, power; double coefficient; };
            std::vector<Term> list;
            for (uint32_t k = 0; k < terms && f.ok; ++k) { Term t{f.u32(), f.u32(), f.f64()}; list.push_back(t); }
            bool has_max[5] = {};
            for (uint32_t k = 0; k < maxima && f.ok; ++k) {
                const uint32_t v = f.u32(); f.u32(); const uint64_t max = f.u64();
                if (v < 1 || v > 4 || max == 0 || has_max[v]) { error = "fórmula de memoria con máximos no válidos"; return false; }
                has_max[v] = true;
            }
            for (const Term& t : list)
                if (t.variable < 1 || t.variable > 4 || t.power < 1 || t.power > 3 || !(t.coefficient >= 0.0) ||
                    t.coefficient == HUGE_VAL || !has_max[t.variable]) {
                    error = "fórmula de memoria no válida"; return false;
                }
        }
        if (!f.ok || f.pos != f.size) { error = "sección de memoria mal formada"; return false; }
    }
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    for (const auto& y : out.tensors) ranges.push_back({y.offset, y.offset + y.size});
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) if (ranges[i - 1].second > ranges[i].first) { error = "tensores solapados en HNFv10"; return false; }

    In p{sections[6].first, sections[6].second, 0, strings, strings_size};
    for (uint32_t n = p.u32(), i = 0; i < n && p.ok; ++i) {
        Property x;
        x.component = p.u32(); x.key = p.s(); x.type = p.u8();
        switch (x.type) {
            case 1: x.boolean = p.u8() != 0; break;
            case 2: x.integer = int64_t(p.u64()); break;
            case 3: x.real = p.f64(); break;
            case 4: x.text = p.s(); break;
            case 5: { const uint32_t l = p.u32(); p.take(l); break; }
            default: error = "propiedad de tipo desconocido: " + x.key; return false;
        }
        out.properties.push_back(std::move(x));
    }
    if (!p.ok || p.pos != p.size) { error = "sección de propiedades mal formada"; return false; }
    return true;
}

} } // helios::hnf10
