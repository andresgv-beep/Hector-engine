// src/expert_cache.hpp
// ============================================================================
// Caché de expertos de un MoE que no cabe entero en la VRAM
// ============================================================================
//
// Qwen3.6-35B-A3B son 17,8 GiB de expertos: 256 por capa, de los que cada token usa 8.
// Con dos tarjetas de 11 y 12 GB entran todos o casi, pero no queda sitio para la ventana.
// Cuáles se usan depende del texto (medido: los 224 más usados de un texto en inglés
// dejan fuera una de cada diez elecciones en español y código), así que no vale decidir
// de antemano cuáles se quedan fuera.
//
// En VRAM, cada capa tiene `slots` huecos. Un mapa por capa dice dónde está cada experto.
// Si un token elige uno que no está en VRAM, el kernel lo lee de la RAM en ese mismo
// momento: el resultado es el correcto y el grafo de decode no se entera. Al acabar el
// token, after_token() mira qué faltó y lo sube al sitio del experto que lleve más tiempo
// sin usarse.
//
// Dónde están los que no caben en VRAM:
//
//   RESERVA (lo normal). Una reserva de RAM anclada con sitio solo para los que no están en
//   VRAM: E - slots + 1 por capa (2,3 GiB en el 35B con 224 huecos). Cada experto está en
//   un único sitio. Para poder intercambiar sin memoria intermedia, un hueco de VRAM de
//   cada capa se deja libre: el que sube se copia ahí, el desalojado baja al sitio de la
//   reserva que el otro deja, y su hueco pasa a ser el libre. Una vez llenas la caché y la
//   reserva, la copia completa que dejó el cargador se suelta y el fichero ya no se toca.
//
//   COPIA COMPLETA (HELIOS_EXPERTOS_RESERVA=0, o si no se puede anclar la reserva). Todos
//   los expertos siguen en la copia del cargador y de ahí se leen. Si esa copia es el
//   fichero mapeado, cada lectura son fallos de página atendidos de 4 KiB en 4 KiB, y si el
//   fichero no cabe en la caché de disco, además van al disco: medido en el 35B, el mismo
//   turno genera a 66 tok/s así y a 85 con los expertos en RAM anclada.
//
// El prefill lee de la caché lo que está y de la RAM lo que no, y no la mueve: en una
// tanda de 1.024 tokens salen elegidos casi todos los expertos de cada capa.
//
// Hay una por motor: con el modelo repartido, cada tarjeta lleva la caché de sus capas.
#pragma once

#include "engine.hpp"
#include "graph_builder.hpp"
#include "hnf_loader.hpp"
#include "../kernels/kernels.hpp"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Por si las cabeceras del sistema son anteriores a Linux 5.4.
#ifndef MADV_PAGEOUT
#define MADV_PAGEOUT 21
#endif

namespace helios {

class ExpertCache {
public:
    // Sube a la VRAM los primeros expertos de cada capa de este motor y deja el resto en la
    // reserva. Con el motor activo. `loader` es el que cargó estos tensores: con él la
    // reserva se lee del fichero directamente, que por USB va cuatro veces más rápido que a
    // través del mapa.
    ExpertCache(Engine& engine, const ArchDescriptor& arch, uint32_t slots, const HnfLoader* loader = nullptr)
        : engine_(engine), loader_(loader) {
        const uint32_t E = arch.moe_experts;
        K_ = arch.moe_top_k;
        E_ = E;
        slots_ = std::min(slots, E);
        const char* off = std::getenv("HELIOS_EXPERTOS_RESERVA");
        reserve_ = slots_ < E && slots_ > K_ && !(off && *off == '0');
        if (slots_ < K_) throw std::invalid_argument("la caché de expertos necesita al menos top-k huecos");
        std::vector<uint32_t> indices;
        for (uint32_t l = 0; l < arch.num_layers; ++l) {
            if (engine.tensors().exists(expert_name(arch, l, "gate_up"))) indices.push_back(l);
        }
        // La reserva de todas las capas se pide antes de tocar nada: si no hay RAM que
        // anclar, se sigue con la copia completa en vez de quedarse a medias.
        std::vector<void*> pool;
        const auto t_pool = std::chrono::steady_clock::now();
        if (reserve_) {
            used_ = slots_ - 1;
            for (uint32_t l : indices) {
                for (const char* which : {"gate_up", "down"}) {
                    const TensorInfo& all = engine.tensors().at(expert_name(arch, l, which));
                    void* host = nullptr;
                    if (cudaHostAlloc(&host, all.size_bytes / E * (E - used_), cudaHostAllocMapped) != cudaSuccess) {
                        host = nullptr;
                        break;
                    }
                    pool.push_back(host);
                    reserve_bytes_ += all.size_bytes / E * (E - used_);
                }
                if (pool.size() % 2) break;
            }
            if (pool.size() != indices.size() * 2) {
                (void)cudaGetLastError();
                for (void* host : pool) cudaFreeHost(host);
                pool.clear();
                reserve_ = false;
                reserve_bytes_ = 0;
                std::cerr << "[caché de expertos] no pude anclar la reserva de RAM: sigo con la copia completa"
                          << std::endl;
            }
        }
        if (!reserve_) used_ = slots_;
        const double pool_s = seconds_since(t_pool);
        for (size_t i = 0; i < indices.size(); ++i) {
            const uint32_t l = indices[i];
            Layer layer;
            layer.index = l;
            layer.gate_up = part(expert_name(arch, l, "gate_up"), GraphBuilder::expert_cache_name(arch, l, "gate_up"),
                                 reserve_ ? pool[2 * i] : nullptr);
            layer.down = part(expert_name(arch, l, "down"), GraphBuilder::expert_cache_name(arch, l, "down"),
                              reserve_ ? pool[2 * i + 1] : nullptr);
            layer.slot_of.resize(E);
            layer.expert_in.assign(slots_, -1);
            layer.last_used.assign(slots_, 0);
            for (uint32_t s = 0; s < used_; ++s) { layer.slot_of[s] = int32_t(s); layer.expert_in[s] = int32_t(s); }
            // Los demás, en la RAM: en su sitio de la reserva o, sin ella, en el suyo de siempre.
            for (uint32_t e = used_; e < E; ++e) layer.slot_of[e] = -1 - int32_t(reserve_ ? e - used_ : e);
            layer.spare = used_;
            layers_.push_back(std::move(layer));
        }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        if (layers_.empty()) return;
        if (reserve_) {
            std::cerr << "[caché de expertos] " << layers_.size() << " capas: anclar la reserva " << pool_s
                      << " s, llenar la VRAM " << fill_s_ << " s, llenar la reserva " << reserve_s_ << " s" << std::endl;
        }
        // Mapas y registros de todas las capas, seguidos: se leen y escriben de una vez.
        check(cudaMalloc(&maps_, layers_.size() * E * sizeof(int32_t)), "reservar los mapas de la caché");
        check(cudaMalloc(&records_, layers_.size() * K_ * sizeof(int32_t)), "reservar los registros de la caché");
        check(cudaMemset(records_, 0xFF, layers_.size() * K_ * sizeof(int32_t)), "limpiar los registros");
        host_records_.resize(layers_.size() * K_);
        for (size_t i = 0; i < layers_.size(); ++i) {
            view(GraphBuilder::expert_cache_name(arch, layers_[i].index, "map"), maps_ + i * E, E);
            view(GraphBuilder::expert_cache_name(arch, layers_[i].index, "record"), records_ + i * K_, K_);
            push_map(i);
        }
    }

    ~ExpertCache() {
        engine_.activate();
        for (const std::string& name : names_) engine_.tensors().remove(name);
        for (Layer& layer : layers_) { cudaFree(layer.gate_up.cache); cudaFree(layer.down.cache); }
        if (maps_) cudaFree(maps_);
        if (records_) cudaFree(records_);
        if (tokens_) {
            std::cerr << "[caché de expertos] " << layers_.size() << " capas, " << slots_ << " de " << E_
                      << " huecos: " << tokens_ << " tokens, " << double(misses_) / double(tokens_)
                      << " lecturas de RAM por token, " << uploads_ << " expertos subidos" << std::endl;
        }
    }
    ExpertCache(const ExpertCache&) = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

    bool empty() const { return layers_.empty(); }
    uint64_t misses() const { return misses_; }
    uint64_t tokens() const { return tokens_; }
    // Bytes de RAM anclada de la reserva; 0 si se trabaja con la copia completa.
    size_t reserve_bytes() const { return reserve_bytes_; }

    // Tras un token de decode ya terminado, con el motor activo: apunta qué se usó y sube
    // lo que hubo que leer de la RAM.
    void after_token() {
        if (layers_.empty()) return;
        check(cudaMemcpy(host_records_.data(), records_, host_records_.size() * sizeof(int32_t),
                         cudaMemcpyDeviceToHost), "leer los registros de la caché");
        ++clock_;
        ++tokens_;
        for (size_t i = 0; i < layers_.size(); ++i) {
            Layer& layer = layers_[i];
            const int32_t* record = host_records_.data() + i * K_;
            // Primero lo que sí estaba: que no se desaloje en este mismo token.
            for (uint32_t k = 0; k < K_; ++k) {
                if (record[k] < 0 || (record[k] & kernels::MOE_CACHE_MISS)) continue;
                const int32_t slot = layer.slot_of[size_t(record[k])];
                if (slot >= 0) layer.last_used[size_t(slot)] = clock_;
            }
            bool changed = false;
            for (uint32_t k = 0; k < K_; ++k) {
                if (record[k] < 0 || !(record[k] & kernels::MOE_CACHE_MISS)) continue;
                const int32_t expert = record[k] & ~kernels::MOE_CACHE_MISS;
                ++misses_;
                if (uint32_t(expert) >= E_ || layer.slot_of[size_t(expert)] >= 0) continue;
                if (reserve_) swap_in(layer, expert);
                else replace(layer, expert);
                ++uploads_;
                changed = true;
            }
            if (changed) push_map(i);
        }
    }

private:
    struct Part {
        const uint8_t* all = nullptr;    // la copia completa, en RAM; nullptr con reserva
        uint8_t* reserve = nullptr;      // [E - used] en RAM anclada; nullptr sin reserva
        uint8_t* cache = nullptr;        // [slots] en VRAM
        size_t stride = 0;
    };
    struct Layer {
        uint32_t index = 0;
        Part gate_up, down;
        // slot_of[experto]: su hueco de VRAM o, si no está, -1 - (su sitio en la RAM).
        // expert_in[hueco]: el experto que lo ocupa, o -1 si es el libre.
        std::vector<int32_t> slot_of, expert_in;
        std::vector<uint64_t> last_used;
        uint32_t spare = 0;              // con reserva: el hueco de VRAM que está libre
    };

    static std::string expert_name(const ArchDescriptor& arch, uint32_t layer, const char* which) {
        return arch.prefix + ".layer" + std::to_string(layer) + ".moe.experts." + which + ".weight";
    }

    // El hueco ocupado que lleva más tiempo sin usarse.
    uint32_t oldest(const Layer& layer) const {
        uint32_t victim = slots_;
        for (uint32_t s = 0; s < slots_; ++s) {
            if (layer.expert_in[s] < 0) continue;
            if (victim == slots_ || layer.last_used[s] < layer.last_used[victim]) victim = s;
        }
        return victim;
    }

    // Sin reserva: el que falta se copia encima del más antiguo, que sigue en la copia completa.
    void replace(Layer& layer, int32_t expert) {
        const uint32_t victim = oldest(layer);
        const int32_t out = layer.expert_in[victim];
        layer.slot_of[size_t(out)] = -1 - out;
        layer.slot_of[size_t(expert)] = int32_t(victim);
        layer.expert_in[victim] = expert;
        layer.last_used[victim] = clock_;
        upload(layer.gate_up, layer.gate_up.all + size_t(expert) * layer.gate_up.stride, victim);
        upload(layer.down, layer.down.all + size_t(expert) * layer.down.stride, victim);
    }

    // Con reserva: el que falta sube al hueco libre, el más antiguo baja al sitio de la
    // reserva que el otro deja, y su hueco pasa a ser el libre. En ningún momento hay un
    // experto sin sitio, y el mapa solo cambia cuando las dos copias han terminado.
    void swap_in(Layer& layer, int32_t expert) {
        const uint32_t place = uint32_t(-1 - layer.slot_of[size_t(expert)]);
        const uint32_t target = layer.spare;
        upload(layer.gate_up, layer.gate_up.reserve + size_t(place) * layer.gate_up.stride, target);
        upload(layer.down, layer.down.reserve + size_t(place) * layer.down.stride, target);
        layer.slot_of[size_t(expert)] = int32_t(target);
        layer.expert_in[target] = expert;
        layer.last_used[target] = clock_;
        const uint32_t victim = oldest(layer);
        const int32_t out = layer.expert_in[victim];
        download(layer.gate_up, victim, place);
        download(layer.down, victim, place);
        layer.slot_of[size_t(out)] = -1 - int32_t(place);
        layer.expert_in[victim] = -1;
        layer.spare = victim;
    }

    static double seconds_since(std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    }

    static void check(cudaError_t error, const char* what) {
        if (error != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }

    Part part(const std::string& source, const std::string& cached, void* reserve) {
        const TensorInfo& all = engine_.tensors().at(source);
        if (all.shape.size() != 3 || all.shape[0] != E_) throw std::runtime_error(source + ": no son expertos empaquetados");
        if (!all.host_mapped && !all.file_mapped) {
            throw std::runtime_error(source + ": con caché de expertos, la copia completa tiene que estar en RAM");
        }
        Part p;
        p.all = static_cast<const uint8_t*>(all.ptr);
        p.stride = all.size_bytes / E_;
        check(cudaMalloc(&p.cache, p.stride * slots_), "reservar la caché de expertos");
        // El fichero mapeado va marcado como de acceso aleatorio, que es lo suyo para leer
        // después un experto suelto. Aquí se lee casi entero y de seguido, y con esa marca
        // cada página de 4 KiB se pide al disco por separado: llenar la caché del 35B
        // tardaba 226 s con un disco que lee esos 16 GiB en 45. Se quita mientras dura.
        const auto t_fill = std::chrono::steady_clock::now();
        advise(all, 0, p.stride * used_, MADV_SEQUENTIAL);
        check(cudaMemcpy(p.cache, p.all, p.stride * used_, cudaMemcpyDefault), "llenar la caché de expertos");
        advise(all, 0, p.stride * used_, MADV_RANDOM);
        fill_s_ += seconds_since(t_fill);
        TensorInfo info;
        info.ptr = p.cache;
        info.shape = {slots_, all.shape[1], all.shape[2]};
        info.dtype = all.dtype;
        info.size_bytes = p.stride * slots_;
        info.owns_memory = false;
        engine_.tensors().register_tensor(cached, info);
        names_.push_back(cached);
        if (reserve) {
            // El resto, a la reserva; y el tensor de expertos de siempre pasa a ser ella. Con
            // eso se suelta la copia completa (el mapa del fichero o sus 18 GiB anclados) y
            // los kernels, que leen de ese tensor lo que no está en VRAM, leen de la reserva.
            const auto t_reserve = std::chrono::steady_clock::now();
            const size_t bytes = p.stride * (E_ - used_);
            // Ese tramo va a vivir en la reserva, así que no debe quedarse además en la caché
            // de disco: ocuparía dos veces en RAM y echaría de la caché lo que la siguiente
            // carga sí va a leer (medido: 7-10 GiB releídos del disco en cada carga).
            if (!(all.file_mapped && read_from_file(source, p.stride * used_, reserve, bytes))) {
                // Sin fichero que leer: del mapa, o de la copia anclada del cargador por su
                // dirección de anfitrión. A través del mapa el núcleo trae 128 KiB por fallo
                // de página: los 2,3 GiB del 35B tardan 30 s por USB, contra 7 leyendo.
                advise(all, p.stride * used_, bytes, MADV_SEQUENTIAL);
                const uint8_t* host = all.host_mapped && all.allocation_ptr
                    ? static_cast<const uint8_t*>(all.allocation_ptr) : p.all;
                std::memcpy(reserve, host + p.stride * used_, bytes);
                advise(all, p.stride * used_, bytes, MADV_PAGEOUT);
            }
            void* device = nullptr;
            check(cudaHostGetDevicePointer(&device, reserve, 0), "ver la reserva de expertos desde la tarjeta");
            p.reserve = static_cast<uint8_t*>(reserve);
            p.all = nullptr;
            engine_.tensors().replace_storage(source, device, reserve, bytes, true);
            reserve_s_ += seconds_since(t_reserve);
        }
        return p;
    }

    // Lee `bytes` de un tensor directamente del fichero, a partir de `offset`, y los saca de
    // la caché de disco. false si no se puede: quien llama tiene otro camino.
    bool read_from_file(const std::string& name, size_t offset, void* destination, size_t bytes) {
        if (!loader_ || loader_->file_path().empty()) return false;
        const TensorEntry* entry = nullptr;
        for (const TensorEntry& candidate : loader_->tensors()) {
            if (candidate.name == name) { entry = &candidate; break; }
        }
        if (!entry || offset + bytes > entry->size) return false;
        if (fd_ < 0) {
            fd_ = ::open(loader_->file_path().c_str(), O_RDONLY | O_CLOEXEC);
            if (fd_ < 0) return false;
            (void)posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
        }
        const off_t start = off_t(entry->offset + offset);
        size_t done = 0;
        while (done < bytes) {
            const ssize_t n = ::pread(fd_, static_cast<uint8_t*>(destination) + done,
                                      std::min(bytes - done, size_t(8) << 20), start + off_t(done));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            done += size_t(n);
        }
        (void)posix_fadvise(fd_, start, off_t(bytes), POSIX_FADV_DONTNEED);
        return true;
    }

    // Consejo al sistema sobre `bytes` de un tensor que vive en el fichero mapeado, a partir
    // de `offset`. Sin efecto si la copia está en RAM anclada.
    static int advise(const TensorInfo& tensor, size_t offset, size_t bytes, int advice) {
        if (!tensor.file_mapped) return 0;
        const uintptr_t page = uintptr_t(sysconf(_SC_PAGESIZE));
        const uintptr_t begin = (uintptr_t(tensor.ptr) + offset) & ~(page - 1);
        const uintptr_t end = (uintptr_t(tensor.ptr) + offset + bytes + page - 1) & ~(page - 1);
        return madvise(reinterpret_cast<void*>(begin), end - begin, advice);
    }

    void view(const std::string& name, int32_t* ptr, uint32_t count) {
        TensorInfo info;
        info.ptr = ptr;
        info.shape = {count};
        info.dtype = dtype::INT32();
        info.size_bytes = size_t(count) * sizeof(int32_t);
        info.owns_memory = false;
        engine_.tensors().register_tensor(name, info);
        names_.push_back(name);
    }

    void upload(const Part& p, const uint8_t* from, uint32_t slot) {
        check(cudaMemcpy(p.cache + size_t(slot) * p.stride, from, p.stride, cudaMemcpyDefault),
              "subir un experto a la caché");
    }

    void download(const Part& p, uint32_t slot, uint32_t place) {
        check(cudaMemcpy(p.reserve + size_t(place) * p.stride, p.cache + size_t(slot) * p.stride, p.stride,
                         cudaMemcpyDeviceToHost), "bajar un experto a la reserva");
    }

    void push_map(size_t i) {
        check(cudaMemcpy(maps_ + i * E_, layers_[i].slot_of.data(), E_ * sizeof(int32_t), cudaMemcpyHostToDevice),
              "actualizar el mapa de la caché");
    }

    Engine& engine_;
    const HnfLoader* loader_ = nullptr;
    int fd_ = -1;                    // el fichero del modelo, solo mientras se llena la reserva
    uint32_t E_ = 0, K_ = 0, slots_ = 0;
    uint32_t used_ = 0;              // huecos ocupados: slots_, o slots_ - 1 con reserva
    bool reserve_ = false;
    size_t reserve_bytes_ = 0;
    double fill_s_ = 0.0, reserve_s_ = 0.0;   // lo que tardó la carga, para el registro
    std::vector<Layer> layers_;
    std::vector<std::string> names_;
    int32_t* maps_ = nullptr;
    int32_t* records_ = nullptr;
    std::vector<int32_t> host_records_;
    uint64_t clock_ = 0, tokens_ = 0, misses_ = 0, uploads_ = 0;
};

}  // namespace helios
