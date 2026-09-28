#!/usr/bin/env python3
# Copia de lab/runtime_bench.py de Hector-hqs-vx3 (GPT), apuntando a esta rama.
"""Isolated NDJSON benchmark: immutable model, per-run cache, fixed synthetic prompts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import threading

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/home/andres/Documentos/GitHub/helios_convert_v9.1/output/gemma4_12b_unified.hnf')


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--binary', type=Path, default=ROOT/'build/helios_runtime')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--model',type=Path,default=MODEL)
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--repeats', type=int, nargs='+', default=[0, 384])
    ap.add_argument('--temperature', type=float, default=0)
    ap.add_argument('--env', action='append', default=[])
    args = ap.parse_args()
    binary = args.binary.resolve()
    if not binary.is_relative_to(ROOT) or not binary.is_file():
        ap.error('El ejecutable debe pertenecer al repositorio experimental.')
    if Path(args.tag).name != args.tag or args.tag in ('.', '..'):
        ap.error('tag debe ser un nombre, no una ruta')
    compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=process_name', '--format=csv,noheader'], text=True)
    if any('helios_runtime' in line for line in compute.splitlines()):
        ap.error('Ya hay un motor cargado en GPU. Dejarlo descargado antes del benchmark; no se detiene automáticamente.')
    out = ROOT/'lab/results'/args.tag
    out.mkdir(parents=True, exist_ok=False)
    cache = out/'cache'
    cache.mkdir()
    seed = ROOT/'lab/cache/tune.cache'
    shutil.copy2(seed, cache/'tune.cache') if seed.exists() else (cache/'tune.cache').touch()
    env = os.environ.copy()
    env.update(HELIOS_HOME=str(cache), CUDA_CACHE_PATH=str(ROOT/'lab/data/cuda-cache'), HELIOS_SEED='42')
    for pair in args.env:
        key, value = pair.split('=', 1)
        if not key.startswith('HELIOS_EXPERIMENT_'):
            ap.error('Solo se admiten variables HELIOS_EXPERIMENT_')
        env[key] = value
    meta = dict(binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                model=str(args.model), model_size=args.model.stat().st_size,
                cache_sha256=hashlib.sha256((cache/'tune.cache').read_bytes()).hexdigest(),
                options=vars(args) | {'binary': str(args.binary),'model':str(args.model)})
    (out/'metadata.json').write_text(json.dumps(meta, indent=2))
    results = []
    with (out/'stderr.log').open('w') as log, (out/'gpu.csv').open('w') as gpulog:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used,clocks_event_reasons.active', '--format=csv', '-l', '1'], stdout=gpulog)
        proc = subprocess.Popen([str(binary), '--model', str(args.model), '--ctx', '16384'],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=env)
        timer = threading.Timer(600, proc.kill)
        timer.start()

        def send(e):
            proc.stdin.write(json.dumps(e)+'\n')
            proc.stdin.flush()

        def until(kind, rid=None):
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError(f'Runtime cerrado: {proc.poll()}, ver {out}/stderr.log')
                event = json.loads(line)
                if event.get('type') == 'error':
                    raise RuntimeError(str(event))
                if event.get('type') == kind and (rid is None or event.get('request_id') == rid):
                    return event

        try:
            until('ready')
            for repeat in args.repeats:
                prompt = 'Datos de referencia:\n' + 'El servidor registra eventos de red, memoria, disco y conexiones activas.\n'*repeat + 'Escribe una explicación técnica extensa de al menos mil palabras sobre cómo funciona un sistema operativo. Empieza directamente y desarrolla todos los detalles.'
                for trial in range(args.trials+1):
                    rid = f'{repeat}-{trial}'
                    send(dict(type='reset', request_id='reset-'+rid))
                    until('result', 'reset-'+rid)
                    send(dict(type='turn', request_id=rid, messages=[dict(role='user', content=prompt)],
                              generation=dict(temperature=args.temperature, max_visible_tokens=args.tokens, max_thinking_tokens=0, close_turn=False)))
                    result = until('completed', rid)
                    text = result.pop('visible_text', '')
                    (out/(rid+'.txt')).write_text(text)
                    result.update(repeat=repeat, trial=trial, warmup=(trial==0), sha256=hashlib.sha256(text.encode()).hexdigest())
                    results.append(result)
                    (out/'results.json').write_text(json.dumps(results, indent=2))
                    print(json.dumps(result), flush=True)
                    if result['usage']['generated_tokens'] != args.tokens:
                        raise RuntimeError('Salida corta: no es un benchmark comparable')
            send(dict(type='shutdown', request_id='end'))
            proc.wait(timeout=20)
            if proc.returncode:
                raise RuntimeError(f'Runtime exit={proc.returncode}')
        finally:
            timer.cancel()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            monitor.terminate()
            monitor.wait(timeout=10)
    summary = {str(r): statistics.median(x['timings']['tokens_per_second'] for x in results if x['repeat']==r and not x['warmup']) for r in args.repeats}
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print('MEDIAN_TOK_S', summary, flush=True)


if __name__ == '__main__':
    main()
