"""Real-model protocol regression; sequential, no external Python dependencies.

Usage: python3 tests/test_runtime_cancellation.py build/helios_runtime model.hnf
"""
import json
import queue
import subprocess
import sys
import threading
import time


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    proc = subprocess.Popen(
        [sys.argv[1], "--model", sys.argv[2], "--ctx", "16384", "--temp", "0"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
    )
    events = queue.Queue()

    def read():
        for line in proc.stdout:
            try:
                events.put(json.loads(line))
            except Exception as exc:
                events.put(exc)
        events.put(EOFError("runtime closed stdout"))

    threading.Thread(target=read, daemon=True).start()

    def receive(timeout=60):
        event = events.get(timeout=timeout)
        if isinstance(event, Exception):
            raise event
        print(json.dumps(event, ensure_ascii=False), flush=True)
        return event

    def send(value):
        proc.stdin.write(json.dumps(value, ensure_ascii=False) + "\n")
        proc.stdin.flush()

    def until(kind, rid=None):
        observed = []
        deadline = time.monotonic() + 60
        while True:
            event = receive(max(.01, deadline-time.monotonic()))
            observed.append(event)
            if event["type"] == kind and (rid is None or event.get("request_id") == rid):
                return event, observed

    gen = {"temperature": 0, "max_visible_tokens": 16, "max_thinking_tokens": 0,
           "preformatted": True, "reuse_prefix": True, "close_turn": False,
           "stop_tokens": ["<turn|>"], "prefill_progress": True}

    def prompt(body):
        return "<bos><|turn>user\n" + body + "<turn|>\n<|turn>model\n"

    def turn(rid, body):
        send({"type": "turn", "request_id": rid,
              "messages": [{"role": "user", "content": prompt(body)}], "generation": gen})

    try:
        until("ready")
        until("session_opened")
        # Wrong targets must not cancel an unrelated active turn. After an
        # explicit rejection, another progress event proves work continued.
        body = "El servidor registra eventos de red, memoria, disco y conexiones activas.\n" * 384
        turn("long", body + "Explica el informe.")
        first, _ = until("prefill_progress", "long")
        send({"type": "cancel", "request_id": "wrong", "target": "different"})
        rejected, _ = until("error", "wrong")
        check(rejected["code"] == "not_active", "wrong cancellation not rejected")
        progress, _ = until("prefill_progress", "long")
        check(progress["processed_tokens"] > first["processed_tokens"], "wrong target stopped progress")
        sent_at = time.monotonic()
        send({"type": "cancel", "request_id": "cancel-long", "target": "long"})
        done, trace = until("completed", "long")
        check(done["finish_reason"] == "cancelled", "prefill cancellation failed")
        check(done["usage"]["prefill_tokens"] < progress["total_tokens"], "processed full cancelled input")
        check(done["usage"]["generated_tokens"] == 0 and done["visible_text"] == "", "cancel generated text")
        check(done["model_state"]["cache_position"] == 0, "cancel retained partial input")
        check(done["timings"]["first_token_ms"] == -1, "cancel reported first token")
        check(any(e.get("request_id") == "cancel-long" and e.get("ok") for e in trace), "cancel not acknowledged")
        print("CANCEL_ROUNDTRIP_MS", (time.monotonic()-sent_at)*1000, flush=True)

        short = "Explica en un párrafo cómo funciona una base de datos relacional."
        turn("first", short)
        first, trace = until("completed", "first")
        check(first["visible_text"], "missing visible text")
        kinds = [e["type"] for e in trace]
        check(kinds.index("prefill") < kinds.index("text_delta") < kinds.index("completed"), "event order")
        check(first["timings"]["first_token_ms"] >= 0, "missing first token timing")
        check("queue_ms" in first["timings"], "missing queue timing")
        check(first["model_state"]["cache_position"] ==
              first["usage"]["prefill_tokens"] + first["usage"]["generated_tokens"],
              "runtime ignored close_turn=false")
        turn("repeat", short)
        repeated, _ = until("completed", "repeat")
        check(repeated["visible_text"] == first["visible_text"], "repeat changed output")
        check(repeated["usage"]["prefill_tokens"] == 1 and repeated["usage"]["prefill_reused"] > 0,
              "runtime did not honor preformatted/reuse_prefix")
        gen["stop_tokens"] = ["<|not_a_real_stop_token_for_this_test|>"]
        turn("invalid-stop", short)
        invalid, _ = until("error", "invalid-stop")
        check(invalid["code"] == "invalid_stop_token", "runtime ignored stop_tokens")
        check(invalid["model_state"]["cache_position"] == repeated["model_state"]["cache_position"],
              "invalid stop token changed the cache")
        send({"type": "shutdown", "request_id": "end"})
        until("result", "end")
        check(proc.wait(timeout=10) == 0, "runtime failed on shutdown")
        print("PASS runtime: targeted cancel, chunk progress, streaming order, timings and prefix reuse")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()
