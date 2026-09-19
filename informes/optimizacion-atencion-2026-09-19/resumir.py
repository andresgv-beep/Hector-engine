"""Validate output parity and summarize five warm samples per context."""
import json
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
records = {}
for mode in (0, 1):
    rows = []
    for line in (root / f"split-{mode}.log").read_text().splitlines():
        if line.startswith("BENCH "):
            rows.append(dict(field.split("=", 1) for field in line.split()[1:]))
    assert len(rows) == 18, f"Incomplete benchmark: split={mode}"
    records[mode] = rows

summary = []
for repeat in (0, 128, 384):
    group = {m: [r for r in records[m] if int(r["repeat"]) == repeat] for m in (0, 1)}
    all_rows = group[0] + group[1]
    assert len({(r["hash"], r["bytes"], r["output"]) for r in all_rows}) == 1
    assert all(int(r["output"]) == 128 and int(r["fallbacks"]) == 0 for r in all_rows)
    assert all(int(r["captures"]) == 1 and int(r["replays"]) == 126 for r in group[1])
    assert all(int(r["captures"]) == 1 and int(r["replays"]) == 126 for r in group[0])
    row = {"prompt_tokens": int(group[0][0]["prompt"]), "samples_per_mode": 5}
    for m, label in ((0, "reference"), (1, "split")):
        warm = [r for r in group[m] if int(r["trial"]) > 0]
        rates = [float(r["tok_s"]) for r in warm]
        row[label] = {
            "median_tok_s": median(rates), "min_tok_s": min(rates), "max_tok_s": max(rates),
            "median_prefill_ms": median(float(r["prefill_ms"]) for r in warm),
            "median_gpu_used_MiB": median(float(r["used_MiB"]) for r in warm),
        }
    row["gain_percent"] = (row["split"]["median_tok_s"] / row["reference"]["median_tok_s"] - 1) * 100
    row["same_output_hash"] = group[0][0]["hash"]
    summary.append(row)
print(json.dumps(summary, indent=2))
