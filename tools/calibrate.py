#!/usr/bin/env python3
"""
calibrate.py — calibrador reproducible del muestreo de helios_chat.

Motivación: cambiar los defaults del sampler "por sensaciones" (una tirada,
un prompt) lleva a conclusiones falsas. Esto ejecuta un corpus fijo con N
semillas por configuración y saca métricas objetivas con media y desviación.

Aislamiento: cada tirada corre con un HELIOS_HOME temporal y una ficha de
usuario SINTÉTICA. No toca la memoria real de nadie y es reproducible en
cualquier máquina — el calibrador no sabe quién eres.

Uso:
  tools/calibrate.py --model out.hnf
  tools/calibrate.py --model out.hnf --configs 1.15/384/0.1 1.1/128/0 --seeds 5
  tools/calibrate.py --model out.hnf --categories memoria larga --verbose
"""

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# PERFIL SINTÉTICO — persona ficticia, ninguna dependencia del usuario real
# ---------------------------------------------------------------------------

FIXTURE_OWNER = "Sam"
FIXTURE_FACTS = """- Trabaja de panadera y vive en una ciudad costera.
- Su lenguaje de programación favorito es Python.
- Prefiere respuestas cortas y directas.
- Tiene un perro llamado Muon.
"""
FIXTURE_EPISODIC = """
## Sesión 2026-01-01 10:00
me contó que está montando una web para la panadería y decidimos usar Python.
"""

# ---------------------------------------------------------------------------
# CORPUS — genérico, sin identidad; cada entrada es (categoría, prompt)
# ---------------------------------------------------------------------------

CORPUS = [
    ("trivial",  "hola"),
    ("trivial",  "gracias, buenas noches"),
    ("factual",  "que es una GPU en dos frases"),
    ("factual",  "cual es la capital de Portugal"),
    ("tecnica",  "explica que es un compilador y para que sirve"),
    ("tecnica",  "diferencia entre memoria RAM y VRAM"),
    ("larga",    "escribe una guia para empezar a programar, con detalle"),
    ("larga",    "que lenguaje de programacion me recomiendas aprender y por que? explicalo bien"),
    ("memoria",  "que sabes de mi?"),
    ("memoria",  "como se llama mi perro?"),
    ("charla",   "que tal va el dia?"),
    ("charla",   "me aburro, cuentame algo"),
]

# Coletillas de asistente comercial (reflejo que queremos medir)
ASSISTANT_TICS = [
    "en qué puedo ayudarte", "en que puedo ayudarte",
    "no dudes en", "estoy aquí para", "estoy aqui para",
    "espero que esto te ayude", "si tienes alguna pregunta",
    "¡claro que sí! aquí tienes", "claro que si! aqui tienes",
]

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# helios> [(pensando...)] <texto> [<n> tok ...]
TURN = re.compile(r"helios>\s*(?:\(pensando[^)]*\))?\s*(.*?)\[(\d+) tok(.*?)\]",
                  re.DOTALL)


def make_profile(tmpdir):
    """Crea un HELIOS_HOME limpio con la ficha sintética."""
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
    os.makedirs(tmpdir)
    with open(os.path.join(tmpdir, "owner"), "w") as f:
        f.write(FIXTURE_OWNER + "\n")
    with open(os.path.join(tmpdir, "facts.md"), "w") as f:
        f.write(FIXTURE_FACTS)
    with open(os.path.join(tmpdir, "episodic.md"), "w") as f:
        f.write(FIXTURE_EPISODIC)


def repeated_ngram(text, n=8):
    """¿Hay un n-grama de palabras que aparece 2+ veces? Señal de bucle."""
    words = re.sub(r"\s+", " ", text.lower()).split()
    if len(words) < n * 2:
        return False
    seen = set()
    for i in range(len(words) - n + 1):
        g = " ".join(words[i:i + n])
        if g in seen:
            return True
        seen.add(g)
    return False


def run_once(binary, model, temp, cfg, seed, prompts, profile_dir, timeout,
             fast=True):
    """Una tirada: N prompts en una sesión. Devuelve lista de turnos medidos."""
    make_profile(profile_dir)
    rep, win, freq = cfg
    env = dict(os.environ)
    env.update({
        "HELIOS_HOME": profile_dir,
        "HELIOS_NO_DISTILL": "1",
        "HELIOS_SEED": str(seed),
        "HELIOS_REP": str(rep),
        "HELIOS_WINDOW": str(win),
        "HELIOS_FREQ": str(freq),
    })
    # El governor interactivo duerme hasta ajustar la salida a ritmo de lectura.
    # En calibración sólo añade minutos: /fast elimina la espera, no cambia el
    # muestreo, los tokens ni los presupuestos de respuesta.
    prefix = "/fast\n" if fast else ""
    stdin_data = prefix + "".join(p + "\n" for p in prompts) + "/salir\n"
    try:
        proc = subprocess.run(
            [binary, model, str(temp)],
            input=stdin_data, env=env, timeout=timeout,
            capture_output=True, text=True, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return None

    out = ANSI.sub("", proc.stdout)
    turns = []
    for m in TURN.finditer(out):
        text, tokens, tail = m.group(1).strip(), int(m.group(2)), m.group(3)
        low = text.lower()
        turns.append({
            "text": text,
            "tokens": tokens,
            "loop": ("se repetía" in out[m.start():m.end() + 40]
                     or repeated_ngram(text)),
            "cut": ("presupuesto" in tail or "…" in text[-6:]
                    or "se repetía" in tail),
            "identity_err": bool(re.search(
                r"\bsoy\s+" + FIXTURE_OWNER.lower() + r"\b", low)),
            "tics": sum(1 for t in ASSISTANT_TICS if t in low),
        })
    return turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ruta al .hnf")
    ap.add_argument("--binary", default="build/helios_chat")
    ap.add_argument("--configs", nargs="+", default=["1.15/384/0.1"],
                    help="rep/ventana/freq, p.ej. 1.15/384/0.1")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--categories", nargs="+", default=None,
                    help="filtrar corpus: trivial factual tecnica larga memoria charla")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    corpus = [(c, p) for c, p in CORPUS
              if not args.categories or c in args.categories]
    prompts = [p for _, p in corpus]
    cats = [c for c, _ in corpus]

    profile_dir = os.path.join(tempfile.gettempdir(), "helios_calib_profile")

    print(f"Corpus: {len(prompts)} prompts | semillas: {args.seeds} | "
          f"temp: {args.temp}")
    print(f"Perfil sintético: dueño='{FIXTURE_OWNER}' en {profile_dir}\n")

    results = {}
    for cfg_str in args.configs:
        try:
            rep, win, freq = cfg_str.split("/")
            cfg = (float(rep), int(win), float(freq))
        except ValueError:
            print(f"config inválida: {cfg_str} (formato rep/ventana/freq)")
            sys.exit(1)

        per_seed = []
        for seed in range(1, args.seeds + 1):
            print(f"  [{cfg_str}] semilla {seed}/{args.seeds}...", flush=True)
            turns = run_once(args.binary, args.model, args.temp, cfg, seed,
                             prompts, profile_dir, args.timeout)
            if turns is None:
                print("    TIMEOUT — tirada descartada")
                continue
            if len(turns) < len(prompts):
                print(f"    aviso: {len(turns)}/{len(prompts)} turnos parseados")
            n = len(turns) or 1
            per_seed.append({
                "loops": sum(t["loop"] for t in turns) / n,
                "tokens": statistics.mean([t["tokens"] for t in turns]) if turns else 0,
                "cuts": sum(t["cut"] for t in turns) / n,
                "identity": sum(t["identity_err"] for t in turns) / n,
                "tics": sum(t["tics"] for t in turns) / n,
            })
            if args.verbose and turns:
                for cat, t in zip(cats, turns):
                    flags = "".join(["L" if t["loop"] else ".",
                                     "C" if t["cut"] else ".",
                                     "I" if t["identity_err"] else ".",
                                     "T" if t["tics"] else "."])
                    print(f"      {cat:8} {flags} {t['tokens']:5} tok  "
                          f"{t['text'][:60]!r}")
        results[cfg_str] = per_seed

    # ---- tabla ----
    def ms(vals):
        if not vals:
            return "  n/a "
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{m:5.2f}±{s:.2f}"

    print("\n" + "=" * 78)
    print(f"{'config (rep/win/freq)':24} {'bucles':>12} {'cortes':>12} "
          f"{'identidad':>12} {'coletillas':>12}")
    print("-" * 78)
    for cfg_str, seeds in results.items():
        if not seeds:
            print(f"{cfg_str:24} {'sin datos':>12}")
            continue
        print(f"{cfg_str:24} "
              f"{ms([s['loops'] for s in seeds]):>12} "
              f"{ms([s['cuts'] for s in seeds]):>12} "
              f"{ms([s['identity'] for s in seeds]):>12} "
              f"{ms([s['tics'] for s in seeds]):>12}")
    print("-" * 78)
    print(f"{'':24} {'tokens/turno':>12}")
    for cfg_str, seeds in results.items():
        if seeds:
            print(f"{cfg_str:24} {ms([s['tokens'] for s in seeds]):>12}")
    print("=" * 78)
    print("bucles/cortes/identidad = fracción de turnos afectados (menos es "
          "mejor)\ncoletillas = frases de asistente comercial por turno")
    print(f"n = {args.seeds} semillas × {len(prompts)} prompts por configuración")


if __name__ == "__main__":
    main()
