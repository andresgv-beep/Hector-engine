#!/usr/bin/env python3
"""Six sequential runs, paired AB/BA/AB; trial zero warms each prompt length."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

source_report = Path(__file__).resolve().parent
repo = source_report.parent.parent
report = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else source_report
report.mkdir(exist_ok=True, parents=True)
model = Path(sys.argv[1]).resolve()
results = []
gpu = subprocess.Popen([
    "nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw",
    "--format=csv", "-l", "1"], stdout=(report / "gpu-pareado.csv").open("w"))
try:
    for run, mode in enumerate([0, 1, 1, 0, 0, 1]):
        directory = report / f"run-{run}-{mode}"
        directory.mkdir(exist_ok=True)
        (directory / "tune.cache").write_bytes((report / "tune.cache").read_bytes())
        env = dict(os.environ, HELIOS_HOME=str(directory),
                   ATTENTION_OUTPUT_DIR=str(directory), ATTENTION_TRIALS="2")
        env.pop("ATTENTION_REPEAT", None)
        log = report / f"pareado-{run}-{mode}.log"
        with log.open("w") as output:
            subprocess.run([str(repo / "build/benchmark_prefill"), str(model), str(mode)],
                           env=env, stdout=output, stderr=subprocess.STDOUT,
                           check=True, timeout=240)
        for line in log.read_text().splitlines():
            if not line.startswith("BENCH "):
                continue
            row = dict(field.split("=", 1) for field in line.split()[1:])
            if row["trial"] == "0":
                continue
            row["run"] = run
            results.append(row)
            print(line, flush=True)
finally:
    gpu.terminate()
    gpu.wait()

summary = {}
for repeat in [0, 128, 384]:
    rows = [r for r in results if int(r["repeat"]) == repeat]
    assert len(rows) == 6
    texts = []
    for r in rows:
        texts.append((report / f'run-{r["run"]}-{r["prefill"]}' /
                      f'prefill-{r["prefill"]}-{repeat}-1.txt').read_bytes())
    assert all(text == texts[0] for text in texts), "greedy responses differ"
    groups = [[r for r in rows if int(r["prefill"]) == m] for m in [0, 1]]
    times = [statistics.median(float(r["prefill_ms"]) for r in group) for group in groups]
    reductions = []
    for pair in range(3):
        a, b = sorted([r for r in rows if r["run"] // 2 == pair], key=lambda r: r["prefill"])
        reductions.append(100 * (1 - float(b["prefill_ms"]) / float(a["prefill_ms"])))
    summary[str(repeat)] = dict(prompt=int(rows[0]["prompt"]),
        reference_ms=times[0], optimized_ms=times[1], reduction_pct=100*(1-times[1]/times[0]),
        paired_reduction_pct=reductions, identical_text=True,
        decode_tok_s=[statistics.median(float(r["tok_s"]) for r in group) for group in groups])
(report / "resumen.json").write_text(json.dumps(dict(summary=summary, samples=results), indent=2) + "\n")
print(json.dumps(summary, indent=2))
