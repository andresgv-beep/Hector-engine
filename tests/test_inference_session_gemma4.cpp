// Run against the SAME real HNF with graphs disabled/enabled. No assert():
// these checks must remain active in Release builds.
#include "inference_session.hpp"
#include <atomic>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

using helios::InferenceSession;

static void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

struct Reply {
    std::string text;
    InferenceSession::TurnStats stats;
    InferenceSession::FinishReason reason;
};

static Reply turn(InferenceSession& session, const std::string& prompt,
                  bool graphs, int budget = 48, int cancel_after = 0) {
    InferenceSession::GenConfig gen;
    gen.temperature = 0;
    gen.max_visible_tokens = budget;
    gen.max_thinking_tokens = 0;
    gen.preformatted = true;
    gen.reuse_prefix = true;
    gen.close_turn = false;
    Reply reply;
    std::string code, error;
    std::atomic<bool> cancel{false};
    int fragments = 0;
    const bool ok = session.run_turn({{"user", prompt}}, {}, gen,
        [&](const std::string& s) {
            reply.text += s;
            if (cancel_after && ++fragments >= cancel_after) cancel.store(true);
        }, {}, {}, cancel, &reply.stats, &reply.reason, &code, &error);
    require(ok, "turn failed: " + code + " " + error);
    require(!reply.text.empty(), "empty response");
    require(reply.stats.decode_graph_fallbacks == 0, "unexpected graph fallback");
    if (graphs && reply.stats.generated_tokens >= 3) {
        require(reply.stats.decode_graph_captures == 1, "missing capture for active session");
        require(reply.stats.decode_graph_replays > 0, "missing graph replay");
    }
    if (!graphs) require(reply.stats.decode_graph_captures == 0 &&
                         reply.stats.decode_graph_replays == 0, "graphs enabled in eager control");
    std::cout << "  tokens=" << reply.stats.generated_tokens
              << " prefill=" << reply.stats.prefill_tokens
              << " reused=" << reply.stats.prefill_reused
              << " captures=" << reply.stats.decode_graph_captures
              << " replays=" << reply.stats.decode_graph_replays << std::endl;
    return reply;
}

static std::string framed(const std::string& user) {
    return helios::format_gemma4_chat({{"user", user}});
}

static std::map<std::string, std::string> suite(const std::string& path, bool graphs,
                                              bool split = false, bool prefill = false,
                                              bool long_ring = false) {
    std::cout << "SUITE graphs=" << graphs << " split=" << split
              << " prefill=" << prefill << " long_ring=" << long_ring << std::endl;
    helios::Model::Config cfg;
    cfg.hnf_path = path;
    cfg.max_seq_len = long_ring ? 12288 : 4096;
    cfg.use_cuda_graphs = graphs;
    cfg.use_split_attention = split;
    cfg.use_coalesced_prefill = prefill;
    std::string error;
    auto model = helios::Model::load(cfg, &error);
    require(bool(model), "load: " + error);
    InferenceSession main, auxiliary, reference;
    require(main.attach(model, &error), "attach main: " + error);
    require(auxiliary.attach(model, &error, 1024), "attach auxiliary: " + error);
    require(reference.attach(model, &error), "attach reference: " + error);
    require(main.kv_namespace() != auxiliary.kv_namespace(), "shared KV namespace");
    std::map<std::string, std::string> outputs;
    const auto request = framed("Explica en un párrafo cómo funciona una base de datos relacional.");
    outputs["main"] = turn(main, request, graphs).text;
    const auto pos = main.cache_position();
    outputs["auxiliary"] = turn(auxiliary,
        framed("Explica en un párrafo cómo se forman las nubes."), graphs).text;
    require(main.cache_position() == pos, "auxiliary moved main cursor");
    outputs["main_repeat"] = turn(main, request, graphs).text;
    require(outputs["main_repeat"] == outputs["main"], "repeated prompt changed output");
    auxiliary.reset();
    require(main.cache_position() == pos, "resetting auxiliary changed main cursor");
    {
        InferenceSession temporary;
        require(temporary.attach(model, &error, 512), "temporary attach");
        outputs["temporary"] = turn(temporary, framed("Describe un bosque en primavera."), graphs).text;
    }
    outputs["after_destroy"] = turn(main, request, graphs).text;
    require(outputs["after_destroy"] == outputs["main"], "destroyed session affected main");

    // Cross the local ring capacity, then edit the old prefix and compare to
    // rebuilding the exact edited prompt in a fresh cache.
    main.reset();
    std::string body = "Informe inicial del sistema.\n";
    // Multimodal Gemma reserves room for a 6144-token image prefill even in
    // text chats. The longer fixture crosses that larger local ring too.
    const uint32_t ring_bound = long_ring ? 7168 : 1536;
    for (int i = 0; i < (long_ring ? 512 : 128); ++i)
        body += "El servidor registra eventos de red, memoria, disco y conexiones activas.\n";
    const std::string instruction = "\nExplica cómo mantener estos servidores en funcionamiento.";
    outputs["long"] = turn(main, framed(body + instruction), graphs).text;
    require(main.cache_position() > ring_bound, "fixture did not wrap the local ring");
    const std::string edited = framed("Informe corregido del sistema.\n" +
        body.substr(body.find('\n') + 1) + instruction);
    auto changed = turn(main, edited, graphs);
    require(changed.stats.prefill_reused == 0, "reused an overwritten prefix");
    outputs["old_edit"] = changed.text;
    reference.reset();
    require(changed.text == turn(reference, edited, graphs).text,
            "old-prefix rebuild disagrees with fresh session");
    auto repeated = turn(main, edited, graphs);
    require(repeated.stats.prefill_reused > 0 && repeated.stats.prefill_tokens == 1,
            "resident exact prefix was not reused");
    outputs["long_repeat"] = repeated.text;

    // Changing only the tail must keep the valid local window and must work
    // after prefill has changed all the shared scratch tensor shapes.
    const auto tail = framed(body + "\nExplica cómo monitorizar estos servidores.");
    main.reset();
    turn(main, framed(body + instruction), graphs);
    auto incremental = turn(main, tail, graphs);
    require(incremental.stats.prefill_reused > ring_bound, "lost a valid recent prefix");
    outputs["tail_edit"] = incremental.text;

    main.reset();
    auto partial = turn(main, request, graphs, 48, 8);
    require(partial.reason == InferenceSession::FinishReason::Cancelled, "cancel ignored");
    outputs["cancelled"] = partial.text;
    outputs["after_cancel"] = turn(main, request, graphs).text;
    require(outputs["after_cancel"] == outputs["main"], "cancel/retry changed response");
    main.reset();
    outputs["after_reset"] = turn(main, request, graphs).text;
    require(outputs["after_reset"] == outputs["main"], "reset changed response");
    return outputs;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--long-ring")) {
        std::cerr << "Usage: test_inference_session_gemma4 model.hnf [--long-ring]\n"; return 2;
    }
    const bool long_ring = argc == 3;
    try {
        const auto eager = suite(argv[1], false, false, false, long_ring);
        const auto graphed = suite(argv[1], true, false, false, long_ring);
        require(eager == graphed, "eager and captured sessions generated different text");
        const auto split = suite(argv[1], true, true, false, long_ring);
        require(graphed == split, "split attention changed session output");
        const auto prefill = suite(argv[1], true, true, true, long_ring);
        require(split == prefill, "coalesced prefill changed session output");
        std::cout << "PASS: " << eager.size()
                  << " exact text comparisons; session isolation, ring reuse, reset, cancellation and destruction\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
