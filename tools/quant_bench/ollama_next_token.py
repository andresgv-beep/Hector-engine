#!/usr/bin/env python3
"""Pregunta a Ollama el siguiente token en posiciones concretas del texto.

Alineacion verificada, no supuesta: se comprueba que prompt_eval_count coincida
con la longitud del prefijo. Si no coincide, la posicion se descarta en vez de
contaminar la medida.
"""
import json, os, subprocess, sys, urllib.request
HNF = sys.argv[1]; TOKS = [int(x) for x in open(sys.argv[2]).read().strip().split(',')]
OUT = sys.argv[3]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 150
# OLLAMA_MODEL elige el modelo; TOK_BIN el codec (por defecto ./dec junto al script)
MODEL = os.environ.get('OLLAMA_MODEL', 'qwen3:4b')
# Gemma exige <bos> y Ollama lo antepone incluso en raw, asi que su cuenta sale
# en +1. BOS_EXTRA=1 lo espera de forma explicita en vez de relajar la
# comprobacion; el lado de Hector debe recibir el mismo <bos> por delante.
BOS_EXTRA = int(os.environ.get('BOS_EXTRA', '0'))
DEC = os.environ.get('TOK_BIN') or (sys.argv[0].rsplit('/',1)[0] + '/dec')

def decode(ids):
    return subprocess.run([DEC, HNF, 'dec', ','.join(map(str,ids))],
                          capture_output=True).stdout.decode('utf-8','replace')
def encode(txt):
    r = subprocess.run([DEC, HNF, 'enc'], input=txt.encode(), capture_output=True)
    s = r.stdout.decode().strip()
    return [int(x) for x in s.split(',')] if s else []

paso = max(1, len(TOKS)//N)
posiciones = list(range(20, len(TOKS)-1, paso))       # saltar el arranque
res, desalineadas = {}, 0
for n, p in enumerate(posiciones):
    pref = TOKS[:p+1]
    txt = decode(pref)
    body = json.dumps({"model":MODEL,"prompt":txt,"raw":True,"stream":False,
                       "options":{"temperature":0,"num_predict":1,"num_ctx":4096}}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(
                "http://localhost:11434/api/generate", body,
                {"Content-Type":"application/json"}), timeout=180) as r:
            d = json.load(r)
    except Exception as e:
        print(f'\r  pos {p}: fallo {e}', file=sys.stderr); continue
    if d.get('prompt_eval_count') != len(pref) + BOS_EXTRA:   # el prefijo no re-tokeniza igual
        desalineadas += 1; continue
    ids = encode(d.get('response',''))
    if ids: res[p] = ids[0]
    if n % 10 == 0: print(f'\r  {n}/{len(posiciones)}', end='', file=sys.stderr)
print(f'\r  {len(res)} posiciones validas, {desalineadas} descartadas por desalineacion',
      file=sys.stderr)
json.dump(res, open(OUT,'w'))
