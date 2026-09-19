// Real-model regression tests. Run GPU suites sequentially, with a fixed tune cache.
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
int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "Usage: test_inference_cancellation model.hnf\n"; return 2; }
    try {
        helios::Model::Config c; c.hnf_path = argv[1]; c.max_seq_len = 16384;
        std::string error;
        auto model = helios::Model::load(c, &error);
        require(bool(model), "load: " + error);
        InferenceSession main, waiter, small;
        require(main.attach(model, &error), "attach main: " + error);
        require(waiter.attach(model, &error, 1024), "attach waiter: " + error);
        require(small.attach(model, &error, 128), "attach small: " + error);
        const auto short_prompt = helios::format_gemma4_chat({{"user",
            "Escribe una explicación técnica extensa de al menos mil palabras sobre cómo funciona un sistema operativo. Empieza directamente y desarrolla todos los detalles."}});
        std::string body;
        for (int i = 0; i < 544; ++i)
            body += "El servidor registra eventos de red, memoria, disco y conexiones activas.\n";
        const auto long_prompt = helios::format_gemma4_chat({{"user", body + "Explica el informe."}});
        std::atomic<bool> stop{true};
        int callbacks = 0;
        auto early = run(main, long_prompt, stop, config(),
            [&](uint32_t, uint32_t, double) { ++callbacks; },
            [&](uint32_t, double) { ++callbacks; });
        cancelled(early);
        require(early.stats.prefill_tokens == 0 && main.cache_position() == 0 && callbacks == 0,
                "early cancel executed prefill");

        stop = false;
        const auto baseline = run(main, short_prompt, stop);
        require(!baseline.text.empty() && baseline.stats.first_token_ms >= 0, "missing baseline text/timing");
        const auto before = main.cache_position();
        stop = true;
        auto untouched = run(main, long_prompt, stop);
        cancelled(untouched);
        require(main.cache_position() == before && untouched.stats.prefill_tokens == 0,
                "pre-cancel changed a populated cache");

        // Incremental turn, so rollback must preserve the previous conversation.
        auto incremental = config(); incremental.preformatted = false;
        incremental.reuse_prefix = false; incremental.close_turn = true;
        stop = false;
        auto partial = run(main, body, stop, incremental,
            [&](uint32_t done, uint32_t total, double) {
                require(done == 512 && total > done, "wrong first chunk counters"); stop = true;
            });
        cancelled(partial);
        require(partial.stats.prefill_tokens == 512 && main.cache_position() == before,
                "resident rollback failed or close_turn ran on partial input");
        stop = false;
        require(run(main, short_prompt, stop).text == baseline.text, "retry after resident rollback changed text");

        // E4B's visual-capable ring also wraps by 8192; 12B wraps earlier.
        stop = false;
        uint32_t previous = 0;
        auto wrapped = run(main, body, stop, incremental,
            [&](uint32_t done, uint32_t total, double) {
                require(done > previous && done <= total, "nonmonotonic progress");
                previous = done; if (done >= 8192) stop = true;
            });
        cancelled(wrapped);
        require(wrapped.stats.prefill_tokens == 8192 && main.cache_position() == 0,
                "overwritten ring was retained after cancellation");
        stop = false;
        require(run(main, short_prompt, stop).text == baseline.text, "retry after ring reset changed text");

        // Cancel in the last progress callback and in the final prefill callback.
        for (int phase = 0; phase < 2; ++phase) {
            main.reset(); stop = false;
            auto last = run(main, short_prompt, stop, config(),
                [&](uint32_t done, uint32_t total, double) { if (!phase && done == total) stop = true; },
                [&](uint32_t, double) { if (phase) stop = true; });
            cancelled(last);
            require(main.cache_position() == 0, "last-boundary cancellation retained input");
        }

        // No arbitrary GPU-duration assumption: hold the owning session at a
        // callback, and cancel a waiting session before releasing the owner.
        main.reset(); stop = false;
        std::atomic<bool> waiting_cancel{false};
        std::promise<void> entered, release;
        auto entered_event = entered.get_future();
        auto release_event = release.get_future().share();
        auto owner = std::async(std::launch::async, [&] {
            return run(main, long_prompt, stop, config(),
                [&](uint32_t, uint32_t, double) {
                    entered.set_value(); release_event.wait_for(std::chrono::seconds(5)); stop = true;
                });
        });
        const bool owner_entered = entered_event.wait_for(std::chrono::seconds(20)) == std::future_status::ready;
        if (!owner_entered) { release.set_value(); owner.get(); throw std::runtime_error("owner never reached prefill"); }
        auto queued = std::async(std::launch::async, [&] { return run(waiter, short_prompt, waiting_cancel); });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        waiting_cancel = true;
        const bool prompt_cancel = queued.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready;
        release.set_value();
        const auto queued_reply = queued.get();
        cancelled(owner.get());
        cancelled(queued_reply);
        require(prompt_cancel && queued_reply.stats.prefill_tokens == 0 && waiter.cache_position() == 0,
                "cancelled waiter waited for the active session");

        // Decode cancellation still writes the emitted token, and retry is exact.
        main.reset(); stop = false;
        int fragments = 0;
        auto decoded = run(main, short_prompt, stop, config(), {}, {},
            [&](const std::string&) { if (++fragments == 4) stop = true; });
        require(decoded.reason == InferenceSession::FinishReason::Cancelled &&
                decoded.stats.generated_tokens > 0 && !decoded.text.empty(), "decode cancellation failed");
        stop = false;
        require(run(main, short_prompt, stop).text == baseline.text, "decode cancel/retry changed text");

        auto limited = config(); limited.max_visible_tokens = 256;
        auto full = run(small, short_prompt, stop, limited);
        require(full.reason == InferenceSession::FinishReason::ContextFull,
                "context exhaustion was not distinguished from stop");
        require(full.stats.cache_position + 4 == 128, "context boundary mismatch");
        std::cout << "PASS cancellation: before work, populated KV, chunk rollback, ring reset, last chunk, "
                     "final callback, waiting session, decode retry and context exhaustion\n";
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << std::endl; return 1; }
}
