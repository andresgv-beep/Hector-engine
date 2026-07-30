// src/hexos_bridge.hpp
// ============================================================================
// HEXOS BRIDGE — Telemetría de inferencia hacia el monitor HEXOS
// ============================================================================
//
// Interfaz ESTRECHA por diseño: Héctor solo escribe métricas al blackboard
// de memoria compartida que HEXOS crea en /dev/shm/hexos_state. Si HEXOS no
// está corriendo, el puente queda desactivado y todo es no-op (coste cero).
//
// El struct replica el contrato de hexos_ipc.h (HEXOS v2.5). Se valida
// magic + version al conectar: si no cuadran, el puente se desactiva solo.
//

#pragma once

#include <cstdint>

namespace helios {

class HexosBridge {
public:
    HexosBridge() = default;
    ~HexosBridge() { disconnect(); }

    // Conecta al shm de HEXOS si existe. Devuelve false (sin error) si no está.
    bool connect();
    bool connected() const { return state_ != nullptr; }

    // Anuncia que el Cognitive Kernel (orquestador: presupuestos, rienda de
    // thinking, governor) está activo en este proceso
    void announce_cognitive();

    // Métricas de inferencia (throughput actual, tokens acumulados)
    void update_inference(float tokens_per_sec, uint64_t total_tokens, bool running);

    // Presupuestos de VRAM declarados por el motor (MB)
    void update_vram_budgets(uint32_t model_mb, uint32_t kv_mb, uint32_t workspace_mb);

    // Marca el motor como parado y suelta el shm
    void disconnect();

private:
    void* state_ = nullptr;   // hexos_shared_state_t* (opaco aquí)
    int fd_ = -1;
};

} // namespace helios
