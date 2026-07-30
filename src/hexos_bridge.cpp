// src/hexos_bridge.cpp
// HEXOS BRIDGE — implementación
// ============================================================================

#include "hexos_bridge.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctime>
#include <cstring>

namespace helios {

// ============================================================================
// Réplica del contrato hexos_ipc.h (HEXOS v2.5) — mantener sincronizado.
// Validado en runtime con magic + version; ante mismatch el puente se apaga.
// ============================================================================

namespace {

constexpr uint32_t HEXOS_SHM_MAGIC   = 0x4845584F;  // "HEXO"
constexpr uint32_t HEXOS_SHM_VERSION = 1;
constexpr int      HEXOS_MODULE_COGNITIVE = 2;      // hexos_module_id_t
constexpr int      HEXOS_MODULE_INFERENCE = 4;
constexpr const char* HEXOS_SHM_STATE = "/hexos_state";

struct hexos_shared_state_t {
    uint32_t magic;
    uint32_t version;
    uint64_t last_update_ns;

    uint8_t hexos_running;
    uint8_t hexos_state;
    uint8_t profile;
    uint8_t reserved1;

    uint32_t modules_connected;

    struct {
        uint8_t  running;
        uint8_t  state;
        uint8_t  reserved[2];
        float    tokens_per_sec;
        uint32_t queue_depth;
        uint32_t active_ctx_id;
        uint64_t total_tokens;
    } cognitive;

    struct {
        uint8_t  initialized;
        uint8_t  spillover_active;
        uint8_t  reserved[2];
        uint32_t episodic_count;
        uint32_t identity_hash;
        char     continuity_text[256];
    } memory;

    struct {
        uint8_t  vram_pressure;
        uint8_t  ram_pressure;
        uint8_t  thermal_state;
        uint8_t  reserved;
        uint32_t vram_budget_model_mb;
        uint32_t vram_budget_kv_mb;
        uint32_t vram_budget_workspace_mb;
    } resources;

    struct {
        uint8_t  armed;
        uint8_t  active;
        uint8_t  reserved[2];
        uint32_t predictions_queued;
        float    hit_rate;
    } predictive;
};

uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

} // namespace

bool HexosBridge::connect() {
    if (state_) return true;

    // Solo conectar — HEXOS es el dueño y creador del shm
    int fd = shm_open(HEXOS_SHM_STATE, O_RDWR, 0);
    if (fd < 0) return false;  // HEXOS no está corriendo: puente apagado

    void* mem = mmap(nullptr, sizeof(hexos_shared_state_t),
                     PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mem == MAP_FAILED) {
        close(fd);
        return false;
    }

    auto* st = static_cast<hexos_shared_state_t*>(mem);
    if (st->magic != HEXOS_SHM_MAGIC || st->version != HEXOS_SHM_VERSION) {
        // Contrato distinto: no tocar nada ajeno
        munmap(mem, sizeof(hexos_shared_state_t));
        close(fd);
        return false;
    }

    state_ = mem;
    fd_ = fd;

    // Anunciar presencia del motor (OR atómico: HEXOS también escribe el campo)
    __atomic_or_fetch(&st->modules_connected,
                      1u << HEXOS_MODULE_INFERENCE, __ATOMIC_RELAXED);
    return true;
}

void HexosBridge::announce_cognitive() {
    if (!state_) return;
    auto* st = static_cast<hexos_shared_state_t*>(state_);
    __atomic_or_fetch(&st->modules_connected,
                      1u << HEXOS_MODULE_COGNITIVE, __ATOMIC_RELAXED);
}

void HexosBridge::update_inference(float tokens_per_sec, uint64_t total_tokens,
                                   bool running) {
    if (!state_) return;
    auto* st = static_cast<hexos_shared_state_t*>(state_);
    st->cognitive.running = running ? 1 : 0;
    st->cognitive.state = running ? 1 : 0;
    st->cognitive.tokens_per_sec = tokens_per_sec;
    st->cognitive.total_tokens = total_tokens;
    st->last_update_ns = now_ns();
}

void HexosBridge::update_vram_budgets(uint32_t model_mb, uint32_t kv_mb,
                                      uint32_t workspace_mb) {
    if (!state_) return;
    auto* st = static_cast<hexos_shared_state_t*>(state_);
    st->resources.vram_budget_model_mb = model_mb;
    st->resources.vram_budget_kv_mb = kv_mb;
    st->resources.vram_budget_workspace_mb = workspace_mb;
    st->last_update_ns = now_ns();
}

void HexosBridge::disconnect() {
    if (!state_) return;
    auto* st = static_cast<hexos_shared_state_t*>(state_);
    st->cognitive.running = 0;
    st->cognitive.state = 0;
    st->cognitive.tokens_per_sec = 0.0f;
    __atomic_and_fetch(&st->modules_connected,
                       ~((1u << HEXOS_MODULE_INFERENCE) |
                         (1u << HEXOS_MODULE_COGNITIVE)), __ATOMIC_RELAXED);
    munmap(state_, sizeof(hexos_shared_state_t));
    close(fd_);
    state_ = nullptr;
    fd_ = -1;
}

} // namespace helios
