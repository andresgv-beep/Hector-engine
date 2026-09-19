// Link this source against each engine revision. Keep coalesced prefill enabled
// in both binaries; only HQ42/HQ52 dequantization stores differ.
#include "inference_session.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: benchmark model.hnf prefill(0|1)\n");
        return 2;
    }
    helios::Model::Config cfg;
    cfg.hnf_path = argv[1];
    cfg.max_seq_len = 16384;
    cfg.use_cuda_graphs = true;
    cfg.use_coalesced_prefill = std::atoi(argv[2]) != 0;
    std::string error, code;
    auto model = helios::Model::load(cfg, &error);
    if (!model) { std::fprintf(stderr, "%s\n", error.c_str()); return 1; }
    helios::InferenceSession session;
    if (!session.attach(model, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str()); return 1;
    }
    std::atomic<bool> cancel{false};
    for (int repeat : {0, 128, 384}) {
        if (const char* selected = std::getenv("ATTENTION_REPEAT")) {
            if (repeat != std::atoi(selected)) continue;
        }
        std::string prompt = "Datos de referencia:\n";
        for (int i = 0; i < repeat; ++i)
            prompt += "El servidor registra eventos de red, memoria, disco y conexiones activas.\n";
        prompt += "Escribe una explicación técnica extensa de al menos mil palabras sobre cómo funciona un sistema operativo. Empieza directamente y desarrolla todos los detalles.";
        // First trial warms this shape; report all trials, summarize only 1..5.
        const int trials = std::getenv("ATTENTION_TRIALS") ?
            std::atoi(std::getenv("ATTENTION_TRIALS")) : 6;
        for (int trial = 0; trial < trials; ++trial) {
            session.reset();
            helios::InferenceSession::GenConfig gen;
            gen.temperature = 0;
            gen.max_visible_tokens = 128;
            gen.max_thinking_tokens = 0;
            gen.close_turn = false;
            helios::InferenceSession::TurnStats stats;
            helios::InferenceSession::FinishReason reason;
            std::string answer;
            bool ok = session.run_turn({{"user", prompt}}, {}, gen,
                [&](const std::string& s) { answer += s; }, {}, {}, cancel,
                &stats, &reason, &code, &error);
            if (!ok) {
                std::fprintf(stderr, "%s: %s\n", code.c_str(), error.c_str());
                return 1;
            }
            size_t available = 0, total = 0;
            cudaMemGetInfo(&available, &total);
            uint64_t hash = 14695981039346656037ull;
            for (unsigned char ch : answer) { hash ^= ch; hash *= 1099511628211ull; }
            std::printf("BENCH prefill=%d repeat=%d trial=%d prompt=%u output=%u prefill_ms=%.3f decode_ms=%.3f tok_s=%.3f used_MiB=%.1f captures=%u replays=%u fallbacks=%u hash=%llu bytes=%zu\n",
                cfg.use_coalesced_prefill, repeat, trial, stats.prefill_tokens,
                stats.generated_tokens, stats.prefill_ms, stats.decode_ms,
                1000.0 * stats.generated_tokens / stats.decode_ms,
                (total - available) / 1048576.0, stats.decode_graph_captures,
                stats.decode_graph_replays, stats.decode_graph_fallbacks,
                static_cast<unsigned long long>(hash), answer.size());
            std::fflush(stdout);
            if (const char* directory = std::getenv("ATTENTION_OUTPUT_DIR")) {
                const auto path = std::string(directory) + "/prefill-" + argv[2] + "-" + std::to_string(repeat) + "-" + std::to_string(trial) + ".txt";
                FILE* f = std::fopen(path.c_str(), "wb");
                if (!f) return 1;
                std::fwrite(answer.data(), 1, answer.size(), f);
                std::fclose(f);
            }
            if (stats.generated_tokens != 128 || stats.decode_graph_fallbacks ||
                (cfg.use_cuda_graphs && (stats.decode_graph_captures != 1 ||
                                         stats.decode_graph_replays == 0))) return 1;
        }
    }
}
