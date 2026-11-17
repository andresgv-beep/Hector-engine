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
// Aquí todos los expertos viven en RAM, en la copia que el cargador deja mapeada para que
// la tarjeta pueda leerla (lento: va por PCIe). En VRAM, cada capa tiene `slots` huecos.
// Un mapa por capa dice en qué hueco está cada experto. Si un token elige uno que no está,
// el kernel lo lee de la RAM en ese mismo momento: el resultado es el correcto y el grafo
// de decode no se entera. Al acabar el token, after_token() mira qué faltó y lo sube al
// hueco del experto que lleve más tiempo sin usarse.
//
// El prefill lee de la caché lo que está y de la RAM lo que no, y no la mueve: en una
// tanda de 1.024 tokens salen elegidos casi todos los expertos de cada capa.
//
// Hay una por motor: con el modelo repartido, cada tarjeta lleva la caché de sus capas.
#pragma once

#include "engine.hpp"
#include "graph_builder.hpp"
#include "../kernels/kernels.hpp"

#include <cuda_runtime.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace helios {

class ExpertCache {
public:
    // Lee de la RAM los primeros `slots` expertos de cada capa de este motor. Con el
    // motor activo.
    ExpertCache(Engine& engine, const ArchDescriptor& arch, uint32_t slots) : engine_(engine) {
        const uint32_t E = arch.moe_experts;
        K_ = arch.moe_top_k;
        E_ = E;
        slots_ = std::min(slots, E);
        if (slots_ < K_) throw std::invalid_argument("la caché de expertos necesita al menos top-k huecos");
        for (uint32_t l = 0; l < arch.num_layers; ++l) {
            const std::string base = arch.prefix + ".layer" + std::to_string(l) + ".moe.experts.";
            if (!engine.tensors().exists(base + "gate_up.weight")) continue;
            Layer layer;
            layer.index = l;
            layer.gate_up = part(base + "gate_up.weight", GraphBuilder::expert_cache_name(arch, l, "gate_up"));
            layer.down = part(base + "down.weight", GraphBuilder::expert_cache_name(arch, l, "down"));
            layer.slot_of.assign(E, -1);
            layer.expert_in.resize(slots_);
            layer.last_used.assign(slots_, 0);
            for (uint32_t s = 0; s < slots_; ++s) { layer.slot_of[s] = int32_t(s); layer.expert_in[s] = int32_t(s); }
            layers_.push_back(std::move(layer));
        }
        if (layers_.empty()) return;
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
                uint32_t victim = 0;
                for (uint32_t s = 1; s < slots_; ++s) if (layer.last_used[s] < layer.last_used[victim]) victim = s;
                layer.slot_of[size_t(layer.expert_in[victim])] = -1;
                layer.slot_of[size_t(expert)] = int32_t(victim);
                layer.expert_in[victim] = expert;
                layer.last_used[victim] = clock_;
                upload(layer.gate_up, uint32_t(expert), victim);
                upload(layer.down, uint32_t(expert), victim);
                ++uploads_;
                changed = true;
            }
            if (changed) push_map(i);
        }
    }

private:
    struct Part {
        const uint8_t* all = nullptr;    // la copia completa, en RAM
        uint8_t* cache = nullptr;        // [slots] en VRAM
        size_t stride = 0;
    };
    struct Layer {
        uint32_t index = 0;
        Part gate_up, down;
        std::vector<int32_t> slot_of, expert_in;
        std::vector<uint64_t> last_used;
    };

    static void check(cudaError_t error, const char* what) {
        if (error != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }

    Part part(const std::string& source, const std::string& cached) {
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
        advise(all, p.stride * slots_, MADV_SEQUENTIAL);
        check(cudaMemcpy(p.cache, p.all, p.stride * slots_, cudaMemcpyDefault), "llenar la caché de expertos");
        advise(all, p.stride * slots_, MADV_RANDOM);
        TensorInfo info;
        info.ptr = p.cache;
        info.shape = {slots_, all.shape[1], all.shape[2]};
        info.dtype = all.dtype;
        info.size_bytes = p.stride * slots_;
        info.owns_memory = false;
        engine_.tensors().register_tensor(cached, info);
        names_.push_back(cached);
        return p;
    }

    // Consejo al sistema sobre los primeros `bytes` de un tensor que vive en el fichero
    // mapeado. Sin efecto si la copia está en RAM anclada.
    static void advise(const TensorInfo& tensor, size_t bytes, int advice) {
        if (!tensor.file_mapped) return;
        const uintptr_t page = uintptr_t(sysconf(_SC_PAGESIZE));
        const uintptr_t begin = uintptr_t(tensor.ptr) & ~(page - 1);
        const uintptr_t end = (uintptr_t(tensor.ptr) + bytes + page - 1) & ~(page - 1);
        madvise(reinterpret_cast<void*>(begin), end - begin, advice);
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

    void upload(const Part& p, uint32_t expert, uint32_t slot) {
        check(cudaMemcpy(p.cache + size_t(slot) * p.stride, p.all + size_t(expert) * p.stride, p.stride,
                         cudaMemcpyDefault), "subir un experto a la caché");
    }

    void push_map(size_t i) {
        check(cudaMemcpy(maps_ + i * E_, layers_[i].slot_of.data(), E_ * sizeof(int32_t), cudaMemcpyHostToDevice),
              "actualizar el mapa de la caché");
    }

    Engine& engine_;
    uint32_t E_ = 0, K_ = 0, slots_ = 0;
    std::vector<Layer> layers_;
    std::vector<std::string> names_;
    int32_t* maps_ = nullptr;
    int32_t* records_ = nullptr;
    std::vector<int32_t> host_records_;
    uint64_t clock_ = 0, tokens_ = 0, misses_ = 0, uploads_ = 0;
};

}  // namespace helios
