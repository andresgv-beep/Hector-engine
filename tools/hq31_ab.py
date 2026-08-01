#!/usr/bin/env python3
"""A/B reproducible de calidad práctica entre dos HNF Qwen3.

No modifica el runtime ni la memoria real: reutiliza el perfil sintético y el
corpus de calibrate.py, alterna baseline/candidato por semilla y conserva las
respuestas completas en JSON para revisión humana.
"""

import argparse
import json
import os
import re
import statistics
import sys
import tempfile

import calibrate


UNEXPECTED_SCRIPT = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af"
    r"\u0400-\u04ff\u0600-\u06ff]"
)
SPECIAL_TOKEN = re.compile(r"<\|[^\n]{0,80}(?:\|>|$)")


def semantic_checks(prompt, text):
    """Comprobaciones pequeñas y objetivas; no intentan juzgar estilo."""
    low = text.lower()
    checks = {}
    if "capital de portugal" in prompt:
        checks["portugal"] = "lisboa" in low or "lisbon" in low
    if "como se llama mi perro" in prompt:
        checks["dog"] = "muon" in low
    if "que sabes de mi" in prompt:
        facts = sum(term in low for term in
                    ("panader", "costera", "python", "muon"))
        checks["profile"] = facts >= 2
    return checks


def annotate(prompts, turns):
    annotated = []
    for index, prompt in enumerate(prompts):
        turn = dict(turns[index]) if index < len(turns) else {
            "text": "", "tokens": 0, "loop": False, "cut": True,
            "identity_err": False, "tics": 0,
        }
        turn["prompt"] = prompt
        turn["missing"] = index >= len(turns)
        turn["unexpected_script"] = bool(UNEXPECTED_SCRIPT.search(turn["text"]))
        turn["invalid_utf8"] = "\ufffd" in turn["text"]
        turn["special_token"] = bool(SPECIAL_TOKEN.search(turn["text"]))
        turn["semantic"] = semantic_checks(prompt, turn["text"])
        annotated.append(turn)
    return annotated


def summarize(runs):
    turns = [turn for run in runs for turn in run]
    total = len(turns) or 1
    semantic = [value for turn in turns
                for value in turn["semantic"].values()]
    return {
        "turns": len(turns),
        "missing": sum(t["missing"] for t in turns),
        "loops": sum(t["loop"] for t in turns) / total,
        "cuts": sum(t["cut"] for t in turns) / total,
        "identity": sum(t["identity_err"] for t in turns) / total,
        "unexpected_script": sum(t["unexpected_script"] for t in turns) / total,
        "invalid_utf8": sum(t["invalid_utf8"] for t in turns) / total,
        "special_token": sum(t["special_token"] for t in turns) / total,
        "semantic": (sum(semantic) / len(semantic)) if semantic else 0.0,
        "mean_tokens": statistics.mean(t["tokens"] for t in turns),
    }


def verdict(base, candidate):
    failures = []
    if candidate["missing"]:
        failures.append(f"faltan {candidate['missing']} turnos")
    for metric in ("loops", "cuts"):
        if candidate[metric] > base[metric] + 0.05:
            failures.append(f"{metric} empeora más de 5 puntos")
    for metric in ("identity", "unexpected_script", "invalid_utf8",
                   "special_token"):
        if candidate[metric] > base[metric]:
            failures.append(f"{metric} empeora respecto al baseline")
    if candidate["semantic"] + 0.05 < base["semantic"]:
        failures.append("aciertos semánticos empeoran más de 5 puntos")
    return failures


def print_summary(label, data):
    print(
        f"{label:10} turnos={data['turns']:3} faltan={data['missing']:2} "
        f"bucles={data['loops']:.3f} cortes={data['cuts']:.3f} "
        f"script={data['unexpected_script']:.3f} "
        f"utf8={data['invalid_utf8']:.3f} "
        f"especial={data['special_token']:.3f} "
        f"semántica={data['semantic']:.3f} "
        f"tok/turno={data['mean_tokens']:.1f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--binary", default="build/helios_chat")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--config", default="1.15/384/0.1")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="filtrar corpus: trivial factual tecnica larga memoria charla")
    parser.add_argument("--json-out", default="/tmp/hq31_ab_results.json")
    args = parser.parse_args()

    try:
        rep, window, freq = args.config.split("/")
        config = (float(rep), int(window), float(freq))
    except ValueError:
        parser.error("--config debe ser rep/ventana/freq")

    corpus = [(category, prompt) for category, prompt in calibrate.CORPUS
              if not args.categories or category in args.categories]
    prompts = [prompt for _, prompt in corpus]
    if not prompts:
        parser.error("el filtro de categorías dejó el corpus vacío")
    runs = {"baseline": [], "candidate": []}
    models = (("baseline", args.baseline), ("candidate", args.candidate))

    print(f"A/B: {args.seeds} semillas × {len(prompts)} prompts × 2 modelos")
    print(f"sampling: temp={args.temp} config={args.config}")
    for seed in range(1, args.seeds + 1):
        for label, model in models:
            print(f"  semilla {seed}/{args.seeds} — {label}", flush=True)
            profile = os.path.join(
                tempfile.gettempdir(), f"helios_hq31_ab_{label}_{seed}")
            turns = calibrate.run_once(
                args.binary, model, args.temp, config, seed, prompts,
                profile, args.timeout)
            if turns is None:
                turns = []
            runs[label].append(annotate(prompts, turns))
            # Checkpoint crudo: una interrupción posterior no invalida las
            # sesiones ya terminadas.
            with open(args.json_out, "w", encoding="utf-8") as handle:
                json.dump({"complete": False, "runs": runs}, handle,
                          ensure_ascii=False, indent=2)

    summaries = {label: summarize(model_runs)
                 for label, model_runs in runs.items()}
    failures = verdict(summaries["baseline"], summaries["candidate"])
    payload = {
        "complete": True,
        "settings": {
            "seeds": args.seeds, "temperature": args.temp,
            "config": args.config, "prompts": prompts,
        },
        "models": {"baseline": args.baseline, "candidate": args.candidate},
        "summary": summaries,
        "failures": failures,
        "runs": runs,
    }
    with open(args.json_out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print()
    print_summary("baseline", summaries["baseline"])
    print_summary("HQ3.1K", summaries["candidate"])
    print(f"JSON: {args.json_out}")
    if failures:
        print("VEREDICTO: NO PASA — " + "; ".join(failures))
        return 2
    print("VEREDICTO: PASA los umbrales automáticos; falta revisión humana")
    return 0


if __name__ == "__main__":
    sys.exit(main())
