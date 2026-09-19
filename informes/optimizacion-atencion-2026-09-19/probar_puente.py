"""Smoke test of the compiled Hexos bridge: main -> auxiliary -> main."""
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile

repo = Path(__file__).resolve().parents[2]
if len(sys.argv) != 2:
    raise SystemExit("Usage: probar_puente.py model.hnf")

def framed(text):
    return f"<bos><|turn>user\n{text}<turn|>\n<|turn>model\n".encode()

main = framed("El servidor registra eventos de red, memoria, disco y conexiones activas.\n" * 384 + "Explica en dos frases cómo funciona una base de datos relacional.")
aux = framed("Explica en dos frases cómo se forman las nubes.")
requests = [(False, main), (True, aux), (False, main)]
data = b"".join((b"*" if auxiliary else b"") + str(len(prompt)).encode() + b"\n" + prompt
                for auxiliary, prompt in requests)
with tempfile.TemporaryDirectory(prefix="hector-bridge-") as home:
    result = subprocess.run(
        [str(repo / "build/helios_formatted"), sys.argv[1], "8192", "<turn|>"],
        input=data, capture_output=True, timeout=90,
        env={**os.environ, "HELIOS_HOME": home}, check=True)
sys.stderr.write(result.stderr.decode())
wire = io.BytesIO(result.stdout)
assert wire.readline() == b"READY\n"
answers = []
for index in range(3):
    fields = wire.readline().decode().split()
    assert len(fields) == 7 and fields[4] != "error", fields
    answer = wire.read(int(fields[0])).decode()
    assert answer.strip() and not answer.startswith("ERROR:")
    answers.append(answer)
    print(f"Reply {index}: header={' '.join(fields)} text={answer!r}")
assert wire.read() == b""
assert answers[0] == answers[2], "Auxiliary request altered the main response"
assert int(fields[6]) > 6000, "Repeated main request did not reuse its prefix"
print("PASS: helios_formatted protocol, auxiliary isolation and main-prefix reuse")
