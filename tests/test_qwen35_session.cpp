// Regresión con un modelo real de la familia Qwen3.5. Ejecutar las suites de GPU de una en una.
#include "inference_session.hpp"
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

using helios::InferenceSession;
using Clock = std::chrono::steady_clock;
static void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}
struct Reply {
    InferenceSession::TurnStats stats;
    InferenceSession::FinishReason reason;
    std::string text;
    double wall_ms;
};
static InferenceSession::GenConfig config() {
    InferenceSession::GenConfig c;
    c.temperature = 0; c.max_visible_tokens = 16; c.max_thinking_tokens = 0;
    c.preformatted = true; c.reuse_prefix = true; c.close_turn = false;
    return c;
}
static Reply run(InferenceSession& s, const std::string& prompt,
                 std::atomic<bool>& cancel, InferenceSession::GenConfig gen = config(),
                 const InferenceSession::PrefillProgressCallback& progress = {},
                 const InferenceSession::PrefillCallback& prefill = {},
                 const InferenceSession::TextCallback& text = {}) {
    Reply r;
    std::string code = "stale", error = "stale";
    const auto start = Clock::now();
    const bool ok = s.run_turn({{"user", prompt}}, {}, gen,
        [&](const std::string& v) { r.text += v; if (text) text(v); }, {}, prefill,
        cancel, &r.stats, &r.reason, &code, &error, progress);
    require(ok, "run: " + code + " " + error);
    r.wall_ms = std::chrono::duration<double, std::milli>(Clock::now()-start).count();
    require(code.empty() && error.empty(), "stale error survived successful call");
    require(r.stats.cache_position == s.cache_position(), "reported KV differs from session");
    std::cout << "result reason=" << InferenceSession::finish_reason_name(r.reason)
              << " prefill=" << r.stats.prefill_tokens << " cache=" << r.stats.cache_position
              << " before=" << r.stats.cache_position_before << " wall_ms=" << r.wall_ms
              << " queue_ms=" << r.stats.queue_ms << " ttft=" << r.stats.first_token_ms << std::endl;
    return r;
}
static void cancelled(const Reply& r) {
    require(r.reason == InferenceSession::FinishReason::Cancelled, "cancel ignored");
    require(r.text.empty() && r.stats.generated_tokens == 0, "cancelled prefill generated output");
    require(r.stats.first_token_ms == -1, "cancelled prefill reported a first token");
}

// Qwen3.5: cancelar no puede dejar a medias el estado recurrente de las capas lineales.
// Una caché se rebobina moviendo la posición; aquí hay que restaurar la copia del principio
// del turno. Cada caso cancela, repite el turno de antes y exige el mismo texto que la
// primera vez, que es lo que delataría un estado mal restaurado.
//
// No es un add_test: necesita un HNF de la familia. Con HELIOS_REP=1 HELIOS_FREQ=0, para que
// el texto dependa solo del modelo.
static std::string chatml(const std::string& user) {
    return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
}
int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "Usage: test_qwen35_session model.hnf\n"; return 2; }
    try {
        helios::Model::Config c; c.hnf_path = argv[1]; c.max_seq_len = 16384;
        std::string error;
        auto model = helios::Model::load(c, &error);
        require(bool(model), "load: " + error);
        InferenceSession main, small;
        require(main.attach(model, &error), "attach main: " + error);
        require(small.attach(model, &error, 128), "attach small: " + error);
        const auto short_prompt = chatml("Explain in detail how an operating system schedules processes.");
        std::string body;
        for (int i = 0; i < 400; ++i) body += "The server logs network, memory, disk and active connection events.\n";
        const auto long_prompt = chatml(body + "Explain the report.");

        // Antes de empezar: no se toca nada.
        std::atomic<bool> stop{true};
        auto early = run(main, long_prompt, stop);
        cancelled(early);
        require(early.stats.prefill_tokens == 0 && main.cache_position() == 0, "early cancel executed prefill");

        stop = false;
        const auto baseline = run(main, short_prompt, stop);
        require(!baseline.text.empty(), "missing baseline text");
        const auto before = main.cache_position();

        // Con la sesión poblada y cancelado de antemano: sigue igual.
        stop = true;
        cancelled(run(main, long_prompt, stop));
        require(main.cache_position() == before, "pre-cancel changed a populated session");

        // A mitad del prefill de un turno que continúa la conversación: la primera tanda ya
        // ha pasado por las capas lineales y ha movido su estado. Hay que volver al de antes.
        auto incremental = config(); incremental.preformatted = false;
        incremental.reuse_prefix = false; incremental.close_turn = true;
        stop = false;
        uint32_t first_chunk = 0;
        auto partial = run(main, body, stop, incremental,
            [&](uint32_t done, uint32_t total, double) {
                require(total > done, "the long turn fitted in one chunk: nothing to cancel");
                if (!first_chunk) first_chunk = done;
                stop = true;
            });
        cancelled(partial);
        require(first_chunk > 0 && partial.stats.prefill_tokens == first_chunk && main.cache_position() == before,
                "partial prefill was not rolled back to the start of the turn");
        stop = false;
        require(run(main, short_prompt, stop).text == baseline.text, "retry after a cancelled prefill changed text");

        // Lo mismo, y en vez de repetir se sigue la conversación: el turno siguiente tiene
        // que salir como en una sesión que nunca vio el turno cancelado.
        const auto follow = chatml("Explain in detail how an operating system schedules processes.") + baseline.text +
                            "<|im_end|>\n" + chatml("Now summarise that in one sentence.");
        stop = false;
        const auto expected_follow = run(main, follow, stop);
        main.reset();
        require(run(main, short_prompt, stop).text == baseline.text, "baseline is not reproducible after reset");
        stop = false; first_chunk = 0;
        cancelled(run(main, body, stop, incremental,
            [&](uint32_t done, uint32_t, double) { if (!first_chunk) first_chunk = done; stop = true; }));
        stop = false;
        require(run(main, follow, stop).text == expected_follow.text, "a cancelled turn leaked into the next one");

        // En el último aviso de progreso y en el aviso final de prefill.
        for (int phase = 0; phase < 2; ++phase) {
            main.reset(); stop = false;
            auto last = run(main, short_prompt, stop, config(),
                [&](uint32_t done, uint32_t total, double) { if (!phase && done == total) stop = true; },
                [&](uint32_t, double) { if (phase) stop = true; });
            cancelled(last);
            require(main.cache_position() == 0, "last-boundary cancellation retained input");
            stop = false;
            require(run(main, short_prompt, stop).text == baseline.text, "retry after last-boundary cancel changed text");
        }

        // Durante el decode: lo emitido se queda escrito y repetir da lo mismo.
        main.reset(); stop = false;
        int fragments = 0;
        auto decoded = run(main, short_prompt, stop, config(), {}, {},
            [&](const std::string&) { if (++fragments == 4) stop = true; });
        require(decoded.reason == InferenceSession::FinishReason::Cancelled &&
                decoded.stats.generated_tokens > 0 && !decoded.text.empty(), "decode cancellation failed");
        stop = false;
        require(run(main, short_prompt, stop).text == baseline.text, "decode cancel/retry changed text");

        // Contexto agotado en una sesión pequeña.
        auto limited = config(); limited.max_visible_tokens = 256;
        auto full = run(small, short_prompt, stop, limited);
        require(full.reason == InferenceSession::FinishReason::ContextFull, "context exhaustion was not distinguished from stop");
        std::cout << "PASS qwen35 session: before work, populated session, prefill rollback, next turn after a "
                     "cancelled one, last chunk, final callback, decode retry and context exhaustion\n";
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << std::endl; return 1; }
}
