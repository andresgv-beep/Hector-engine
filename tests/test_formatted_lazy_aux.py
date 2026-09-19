"""Measure lazy auxiliary KV and verify legacy framing/main prefix preservation.

Usage: python3 tests/test_formatted_lazy_aux.py build/helios_formatted model.hnf expected_aux_MiB
Requires nvidia-smi. Run without other inference processes.
"""
import json
import os
import select
import subprocess
import sys
import time


def main():
    proc = subprocess.Popen([sys.argv[1], sys.argv[2], "16384", "<turn|>"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
    pending = bytearray()

    def fill(deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([proc.stdout], [], [], remaining)[0]:
            raise TimeoutError("formatted bridge timeout")
        chunk = os.read(proc.stdout.fileno(), 65536)
        if not chunk:
            raise EOFError("formatted bridge closed stdout")
        pending.extend(chunk)

    def line():
        deadline = time.monotonic() + 90
        while b"\n" not in pending:
            fill(deadline)
        end = pending.index(b"\n") + 1
        data = bytes(pending[:end])
        del pending[:end]
        return data

    def exact(n):
        deadline = time.monotonic() + 90
        while len(pending) < n:
            fill(deadline)
        data = bytes(pending[:n])
        del pending[:n]
        return data

    def memory():
        rows = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"
        ], text=True).splitlines()
        for row in rows:
            pid, mib = row.split(",")
            if int(pid.strip()) == proc.pid:
                return float(mib.strip())
        raise RuntimeError("process GPU allocation unavailable")

    def turn(aux=False):
        body = "Responde únicamente: SI." if aux else "Responde únicamente: OK."
        prompt = ("<bos><|turn>user\n" + body + "<turn|>\n<|turn>model\n").encode()
        proc.stdin.write(("*" if aux else "").encode() + str(len(prompt)).encode() + b"\n" + prompt)
        proc.stdin.flush()
        fields = line().decode().split()
        if len(fields) != 7:
            raise AssertionError(f"invalid response header: {fields}")
        data = exact(int(fields[0])).decode()
        if fields[4] == "error" or not data:
            raise AssertionError(f"response failed: {fields} {data}")
        print(json.dumps({"aux": aux, "header": fields, "text": data}, ensure_ascii=False), flush=True)
        return fields, data

    try:
        if line() != b"READY\n":
            raise AssertionError("missing READY")
        ready = memory()
        _, baseline = turn()
        warmed = memory()
        turn(True)
        attached = memory()
        turn(True)
        reused = memory()
        header, repeated = turn()
        if repeated != baseline or int(header[5]) != 1 or int(header[6]) <= 0:
            raise AssertionError("auxiliary turn affected main response/prefix")
        expected = float(sys.argv[3])
        if abs((attached-warmed)-expected) > 4 or abs(reused-attached) > 4:
            raise AssertionError(f"auxiliary allocation mismatch: {warmed}, {attached}, {reused}")
        print(json.dumps({"ready_MiB": ready, "main_warm_MiB": warmed,
                          "first_aux_MiB": attached, "second_aux_MiB": reused,
                          "lazy_aux_delta_MiB": attached-warmed}), flush=True)
        proc.stdin.close()
        if proc.wait(timeout=10) != 0:
            raise AssertionError("bridge failed on EOF")
        print("PASS formatted: lazy auxiliary KV, no second allocation, legacy framing, main prefix/output intact")
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
