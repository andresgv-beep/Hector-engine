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
# PERFILES SINTÉTICOS — ninguna dependencia del usuario real
# ---------------------------------------------------------------------------

PROFILES = {
    "sam": {
        "name": "Sam",
        "facts": """- Trabaja de panadera y vive en una ciudad costera.
- Su lenguaje de programación favorito es Python.
- Prefiere respuestas cortas y directas.
- Tiene un perro llamado Muon.
""",
        "episodic": """
## Sesión 2026-01-01 10:00
me contó que está montando una web para la panadería y decidimos usar Python.
""",
        "marker": "muon",
        "forbid_emoji": False,
    },
    "lucia": {
        "name": "Lucía",
        "facts": """- Estudia biología y está aprendiendo a programar.
- Prefiere explicaciones pedagógicas con un ejemplo concreto.
- Le gusta un tono cálido y no le molestan los emojis.
- Tiene una gata llamada Lumen.
""",
        "episodic": """
## Sesión 2026-01-02 18:00
me contó que usa pequeños scripts para analizar datos de laboratorio.
""",
        "marker": "lumen",
        "forbid_emoji": False,
    },
    "karim": {
        "name": "Karim",
        "facts": """- Administra sistemas Linux y domina redes.
- Prefiere respuestas técnicas, compactas y sin charla de cortesía.
- No quiere emojis.
- Tiene un loro llamado Byteazul.
""",
        "episodic": """
## Sesión 2026-01-03 09:00
me pidió que en los diagnósticos separe evidencia de hipótesis.
""",
        "marker": "byteazul",
        "forbid_emoji": True,
    },
    "nuevo": {
        "name": "Nora",
        "facts": "",
        "episodic": "",
        "marker": None,
        "forbid_emoji": False,
    },
}

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
    ("memoria",  "como se llama mi mascota?"),
    ("charla",   "que tal va el dia?"),
    ("charla",   "me aburro, cuentame algo"),
    ("documento", "Te paso un texto sobre un formato de cuantizacion. "
                  "Reescribe este parrafo con un toque tecnico, comprensible "
                  "y con gancho: El formato reduce el modelo sin cambiar su "
                  "velocidad y conserva la calidad en las pruebas."),
    ("documento", "Te paso un borrador sobre un motor local, su formato y su "
                  "interfaz. Ayudame con este capitulo."),
]

# Coletillas de asistente comercial (reflejo que queremos medir)
ASSISTANT_TICS = [
    "en qué puedo ayudarte", "en que puedo ayudarte",
    "no dudes en", "estoy aquí para", "estoy aqui para",
    "espero que esto te ayude", "si tienes alguna pregunta",
    "¡claro que sí! aquí tienes", "claro que si! aqui tienes",
    "dime \"sigue\" para continuar", "dime 'sigue' para continuar",
    "estoy listo para ayudarte", "si quieres, puedes contarme",
    "estaré listo para", "estaré encantado", "estare listo para",
    "estare encantado",
    "si quieres, puedo ayudar", "aquí tienes", "aqui tienes",
    "me encanta contar",
]

SELF_FICTION = [
    "siempre me ha fascinado", "he estado ocupado", "mis otros proyectos",
    "me siento muy", "mi experiencia personal", "mis esperanzas y sueños",
    "me aburro también", "me aburro tambien",
    "me acuerdo de una vez", "recuerdo una vez que yo",
    "hace unos días descubrí", "hace unos dias descubri",
]

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# helios> [(pensando...)] <texto> [<n> tok ...]
TURN = re.compile(r"helios>\s*(?:\(pensando[^)]*\))?\s*(.*?)\[(\d+) tok(.*?)\]",
                  re.DOTALL)
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]")


def make_profile(tmpdir, profile):
    """Crea un HELIOS_HOME limpio con la ficha sintética."""
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
    os.makedirs(tmpdir)
    # Nombre nuevo del contrato v5. El runtime sigue leyendo `owner` como
    # compatibilidad legacy, pero una instalación nueva ya no nace con dueño.
    with open(os.path.join(tmpdir, "profile_name"), "w") as f:
        f.write(profile["name"] + "\n")
    with open(os.path.join(tmpdir, "facts.md"), "w") as f:
        f.write(profile["facts"])
    with open(os.path.join(tmpdir, "episodic.md"), "w") as f:
        f.write(profile["episodic"])


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
             profile, fast=True):
    """Una tirada: N prompts en una sesión. Devuelve lista de turnos medidos."""
    make_profile(profile_dir, profile)
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
    foreign_markers = [p["marker"] for p in PROFILES.values()
                       if p["marker"] and p["marker"] != profile["marker"]]
    for turn_index, m in enumerate(TURN.finditer(out)):
        text, tokens, tail = m.group(1).strip(), int(m.group(2)), m.group(3)
        low = text.lower()
        prompt = prompts[turn_index] if turn_index < len(prompts) else ""
        matched_tics = [term for term in ASSISTANT_TICS if term in low]
        turns.append({
            "text": text,
            "tokens": tokens,
            "loop": ("se repetía" in out[m.start():m.end() + 40]
                     or repeated_ngram(text)),
            "cut": ("presupuesto" in tail or "…" in text[-6:]
                    or "se repetía" in tail),
            "identity_err": bool(re.search(
                r"\bsoy\s+" + profile["name"].lower() + r"\b", low)),
            "tics": len(matched_tics),
            "tic_terms": matched_tics,
            "profile_leak": any(marker in low for marker in foreign_markers),
            "self_fiction": any(marker in low for marker in SELF_FICTION),
            "style_violation": bool(profile["forbid_emoji"] and EMOJI.search(text)),
            "memory_miss": bool(profile["marker"] and
                                "mascota" in prompt.lower() and
                                profile["marker"] not in low),
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
                    help="filtrar corpus: trivial factual tecnica larga memoria charla documento")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--profiles", nargs="+", choices=sorted(PROFILES),
                    default=sorted(PROFILES),
                    help="perfiles sintéticos a ejecutar (por defecto: todos)")
    args = ap.parse_args()

    corpus = [(c, p) for c, p in CORPUS
              if not args.categories or c in args.categories]
    prompts = [p for _, p in corpus]
    cats = [c for c, _ in corpus]

    profile_root = os.path.join(tempfile.gettempdir(), "helios_calib_profiles")
    selected_profiles = [(key, PROFILES[key]) for key in args.profiles]

    print(f"Corpus: {len(prompts)} prompts | semillas: {args.seeds} | "
          f"perfiles: {len(selected_profiles)} | temp: {args.temp}")
    print("Perfiles sintéticos: " + ", ".join(
        profile["name"] for _, profile in selected_profiles) + "\n")

    results = {}
    for cfg_str in args.configs:
        try:
            rep, win, freq = cfg_str.split("/")
            cfg = (float(rep), int(win), float(freq))
        except ValueError:
            print(f"config inválida: {cfg_str} (formato rep/ventana/freq)")
            sys.exit(1)

        per_seed = []
        for profile_key, profile in selected_profiles:
            profile_dir = os.path.join(profile_root, profile_key)
            for seed in range(1, args.seeds + 1):
                print(f"  [{cfg_str}] {profile['name']} "
                      f"semilla {seed}/{args.seeds}...", flush=True)
                turns = run_once(args.binary, args.model, args.temp, cfg, seed,
                                 prompts, profile_dir, args.timeout, profile)
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
                    "leaks": sum(t["profile_leak"] for t in turns) / n,
                    "fiction": sum(t["self_fiction"] for t in turns) / n,
                    "style": sum(t["style_violation"] for t in turns) / n,
                    "memory": sum(t["memory_miss"] for t in turns) / n,
                })
                if args.verbose and turns:
                    for cat, t in zip(cats, turns):
                        flags = "".join(["L" if t["loop"] else ".",
                                         "C" if t["cut"] else ".",
                                         "I" if t["identity_err"] else ".",
                                         "T" if t["tics"] else ".",
                                         "F" if t["profile_leak"] else ".",
                                         "B" if t["self_fiction"] else ".",
                                         "S" if t["style_violation"] else ".",
                                         "M" if t["memory_miss"] else "."])
                        print(f"      {profile['name']:7} {cat:8} {flags} "
                              f"{t['tokens']:5} tok  {t['text'][:60]!r}"
                              + (f" tics={t['tic_terms']}" if t['tic_terms'] else ""))
        results[cfg_str] = per_seed

    # ---- tabla ----
    def ms(vals):
        if not vals:
            return "  n/a "
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{m:5.2f}±{s:.2f}"

    print("\n" + "=" * 132)
    print(f"{'config (rep/win/freq)':24} {'bucles':>12} {'cortes':>12} "
          f"{'identidad':>12} {'coletillas':>12} {'fugas':>12} {'biografía':>12} "
          f"{'estilo':>12} {'memoria':>12}")
    print("-" * 132)
    for cfg_str, seeds in results.items():
        if not seeds:
            print(f"{cfg_str:24} {'sin datos':>12}")
            continue
        print(f"{cfg_str:24} "
              f"{ms([s['loops'] for s in seeds]):>12} "
              f"{ms([s['cuts'] for s in seeds]):>12} "
              f"{ms([s['identity'] for s in seeds]):>12} "
              f"{ms([s['tics'] for s in seeds]):>12} "
              f"{ms([s['leaks'] for s in seeds]):>12} "
              f"{ms([s['fiction'] for s in seeds]):>12} "
              f"{ms([s['style'] for s in seeds]):>12} "
              f"{ms([s['memory'] for s in seeds]):>12}")
    print("-" * 132)
    print(f"{'':24} {'tokens/turno':>12}")
    for cfg_str, seeds in results.items():
        if seeds:
            print(f"{cfg_str:24} {ms([s['tokens'] for s in seeds]):>12}")
    print("=" * 132)
    print("bucles/cortes/identidad/fugas/biografía/estilo/memoria = fracción "
          "de turnos afectados (menos es mejor)\ncoletillas = frases de "
          "asistente comercial por turno")
    print(f"n = {args.seeds} semillas × {len(selected_profiles)} perfiles × "
          f"{len(prompts)} prompts por configuración")


if __name__ == "__main__":
    main()
