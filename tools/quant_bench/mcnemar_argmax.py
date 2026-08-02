#!/usr/bin/env python3
"""Compara dos cuantizaciones contra una referencia comun, posicion a posicion.

Entrada:
  --ref     JSON {posicion: token} de la referencia (aqui, el bf16 sin cuantizar)
  --a/--b   los dos candidatos. Cada uno es un JSON como el de --ref, o un .bin
            de int32 de Hector (un argmax por posicion) con --a-bin/--b-bin.
  --shift   desplazamiento a aplicar a los indices del .bin de Hector. Con el
            corpus precedido de <bos>, la posicion p de Ollama es el indice p+1
            de Hector, asi que --shift 1.

Salida: aciertos de cada candidato sobre las posiciones comunes y McNemar
exacto (binomial de dos colas) sobre los pares discordantes. Ver el README:
la regla de la casa es exigir p < 0,01 y confirmar en un segundo corpus.
"""
import argparse, json, math, struct, sys


def cargar(path, es_bin, shift):
    if not es_bin:
        return {int(k): v for k, v in json.load(open(path)).items()}
    datos = open(path, 'rb').read()
    ids = struct.unpack(f'<{len(datos)//4}i', datos)
    # el indice i del volcado predice el token que sigue al prefijo [0..i]
    return {i - shift: t for i, t in enumerate(ids) if i - shift >= 0}


def mcnemar_exacto(b, c):
    """Binomial exacta de dos colas sobre los pares discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cola = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * cola)


ap = argparse.ArgumentParser()
ap.add_argument('--ref', required=True)
ap.add_argument('--a', required=True); ap.add_argument('--a-bin', action='store_true')
ap.add_argument('--b', required=True); ap.add_argument('--b-bin', action='store_true')
ap.add_argument('--shift', type=int, default=0)
ap.add_argument('--nombre-a', default='A'); ap.add_argument('--nombre-b', default='B')
args = ap.parse_args()

ref = cargar(args.ref, False, 0)
A = cargar(args.a, args.a_bin, args.shift if args.a_bin else 0)
B = cargar(args.b, args.b_bin, args.shift if args.b_bin else 0)

comunes = sorted(set(ref) & set(A) & set(B))
if not comunes:
    sys.exit('sin posiciones comunes')

ok_a = sum(A[p] == ref[p] for p in comunes)
ok_b = sum(B[p] == ref[p] for p in comunes)
# pares discordantes: uno acierta y el otro no
solo_a = sum(A[p] == ref[p] and B[p] != ref[p] for p in comunes)
solo_b = sum(B[p] == ref[p] and A[p] != ref[p] for p in comunes)
p = mcnemar_exacto(solo_a, solo_b)

n = len(comunes)
print(f'  posiciones comunes: {n}')
print(f'  {args.nombre_a:22} {ok_a:4}/{n}  {100*ok_a/n:5.1f}%')
print(f'  {args.nombre_b:22} {ok_b:4}/{n}  {100*ok_b/n:5.1f}%')
print(f'  discordantes: solo {args.nombre_a} acierta {solo_a}, solo {args.nombre_b} {solo_b}')
print(f'  McNemar exacto p = {p:.4g}' + ('  (significativo)' if p < 0.01 else '  (NO concluyente)'))
