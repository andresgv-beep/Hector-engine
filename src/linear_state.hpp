// src/linear_state.hpp
// ============================================================================
// El estado de las capas de atención lineal (Gated DeltaNet) de una sesión
// ============================================================================
//
// Las capas lineales de la familia Qwen3.5 no guardan claves y valores: llevan una matriz
// por cabeza y las últimas muestras de la convolución, de tamaño fijo. Aquí vive ese estado
// para una conversación, en un solo bloque de la GPU registrado con los nombres que busca
// el grafo (GraphBuilder::linear_state_name y linear_conv_name).
//
// Una caché se rebobina moviendo la posición; un estado recurrente no se puede deshacer.
// Por eso se guardan copias en RAM. Rebobinar a una posición restaura la copia más cercana
// por debajo, y quien llama vuelve a procesar lo que falte desde ahí. Hay dos clases:
//
//   - anclas: los puntos a los que una sesión vuelve de verdad, el principio de un turno y
//     el final de su prompt. Se conservan las de los últimos turnos.
//   - de paso: cada tanda de prefill y cada kEvery tokens generados. Acotan cuánto hay que
//     volver a procesar si el turno siguiente se aparta a mitad de una respuesta. Solo se
//     conservan las más recientes, y nunca desalojan a un ancla.
#pragma once

#include "engine.hpp"
#include "graph_builder.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace helios {

class LinearState {
public:
    LinearState() = default;
    ~LinearState() { if (data_) cudaFree(data_); }
    LinearState(const LinearState&) = delete;
    LinearState& operator=(const LinearState&) = delete;

    bool active() const { return data_ != nullptr; }
    size_t bytes() const { return bytes_; }
    size_t snapshots() const { return snapshots_.size(); }

    // Reserva y registra el estado de todas las capas lineales, a cero.
    void allocate(Engine& engine, const std::string& prefix, const ArchDescriptor& arch) {
        const size_t state = size_t(arch.lin_value_heads) * arch.lin_key_dim * arch.lin_value_dim;
        const size_t conv = size_t(arch.lin_channels()) * (arch.lin_kernel - 1);
        size_t layers = 0;
        for (uint32_t l = 0; l < arch.num_layers; ++l) layers += arch.is_linear(l) ? 1 : 0;
        if (layers == 0) return;
        bytes_ = layers * (state + conv) * sizeof(float);
        if (cudaMalloc(&data_, bytes_) != cudaSuccess) {
            data_ = nullptr;
            throw std::runtime_error("no pude reservar el estado de las capas lineales");
        }
        float* at = data_;
        const auto add = [&](const std::string& name, std::vector<uint32_t> shape, size_t count) {
            TensorInfo info;
            info.ptr = at;
            info.shape = std::move(shape);
            info.dtype = dtype::FP32();
            info.size_bytes = count * sizeof(float);
            info.owns_memory = false;
            engine.tensors().register_tensor(name, info);
            names_.push_back(name);
            at += count;
        };
        for (uint32_t l = 0; l < arch.num_layers; ++l) {
            if (!arch.is_linear(l)) continue;
            add(GraphBuilder::linear_state_name(prefix, l),
                {arch.lin_value_heads, arch.lin_key_dim, arch.lin_value_dim}, state);
            add(GraphBuilder::linear_conv_name(prefix, l), {arch.lin_channels(), arch.lin_kernel - 1}, conv);
        }
        reset();
    }

    // Las vistas registradas no deben sobrevivir a la memoria.
    void release(Engine& engine) {
        for (const std::string& name : names_) engine.tensors().remove(name);
        names_.clear();
        snapshots_.clear();
        if (data_) cudaFree(data_);
        data_ = nullptr;
        bytes_ = 0;
    }

    // Sesión vacía: estado a cero y ninguna copia.
    void reset() {
        snapshots_.clear();
        if (data_ && cudaMemset(data_, 0, bytes_) != cudaSuccess) {
            throw std::runtime_error("no pude poner a cero el estado de las capas lineales");
        }
    }

    // Cada cuántos tokens generados se deja una copia de paso.
    static constexpr uint32_t kEvery = 256;

    // Guarda el estado tal como está, que corresponde a `position` tokens procesados. El
    // llamante ha sincronizado ya el flujo de cómputo.
    void save(uint32_t position, bool anchor) {
        if (!data_) return;
        while (!snapshots_.empty() && snapshots_.back().position >= position) {
            anchor = anchor || (snapshots_.back().position == position && snapshots_.back().anchor);
            snapshots_.pop_back();
        }
        // La más antigua de su clase deja sitio.
        size_t same = 0;
        for (const Snapshot& s : snapshots_) same += s.anchor == anchor ? 1 : 0;
        if (same >= (anchor ? kMaxAnchors : kMaxPassing)) {
            for (auto it = snapshots_.begin(); it != snapshots_.end(); ++it) {
                if (it->anchor == anchor) { snapshots_.erase(it); break; }
            }
        }
        Snapshot snapshot{position, anchor, std::vector<char>(bytes_)};
        if (cudaMemcpy(snapshot.bytes.data(), data_, bytes_, cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("no pude copiar el estado de las capas lineales");
        }
        snapshots_.push_back(std::move(snapshot));
    }

    // Restaura la copia más reciente con posición <= position y olvida las posteriores.
    // Devuelve la posición restaurada, o nada si no hay ninguna copia que sirva.
    std::optional<uint32_t> restore_at_or_before(uint32_t position) {
        while (!snapshots_.empty() && snapshots_.back().position > position) snapshots_.pop_back();
        if (!data_ || snapshots_.empty()) return std::nullopt;
        const Snapshot& snapshot = snapshots_.back();
        if (cudaMemcpy(data_, snapshot.bytes.data(), bytes_, cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("no pude restaurar el estado de las capas lineales");
        }
        return snapshot.position;
    }

private:
    struct Snapshot { uint32_t position; bool anchor; std::vector<char> bytes; };
    // 63 MB cada una en el 35B: 1 GB de RAM como mucho. Más atrás se vuelve a procesar.
    static constexpr size_t kMaxAnchors = 8;
    static constexpr size_t kMaxPassing = 8;

    float* data_ = nullptr;
    size_t bytes_ = 0;
    std::vector<std::string> names_;
    std::deque<Snapshot> snapshots_;
};

}  // namespace helios
