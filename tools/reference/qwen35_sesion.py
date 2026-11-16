#!/usr/bin/env python3
"""Comprueba una sesión real del motor contra un volcado de la referencia (qwen35_ref.py --dump).

    qwen35_sesion.py <modelo.hnf> <carpeta del checkpoint> <volcado de la referencia> [k ...]

qwen35_dump compara capa a capa pero se salta la sesión. Esto pasa por ella de verdad, con
helios_formatted: prefill por tandas, decode con grafo, reaprovechamiento del prefijo y
rebobinado. Para cada k manda los k primeros tokens del volcado como prompt y mira que el primer
token generado sea el más probable según la referencia en esa posición. Los k van en el orden
dado y sin reiniciar: uno menor que el anterior obliga a rebobinar, que en esta familia es
restaurar una copia del estado de las capas lineales y volver a procesar desde ahí.

Con un modelo de pesos al azar (qwen35_sintetico.py) conviene elegir posiciones donde el
primero gane con margen: sin k, se eligen solas.
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

MOTOR = Path(__file__).resolve().parents[2] / 'build' / 'helios_formatted'


def bytes_de_pieza():
    """La tabla de GPT-2: cada byte se escribe en vocab.json como un carácter imprimible."""
    visibles = list(range(ord('!'), ord('~') + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    tabla, extra = {}, 0
    for b in range(256):
        if b in visibles:
            tabla[chr(b)] = b
        else:
            tabla[chr(256 + extra)] = b
            extra += 1
    return tabla


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    hnf, checkpoint, volcado = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    tabla = bytes_de_pieza()
    vocab = json.loads((checkpoint / 'vocab.json').read_text())
    por_id = {i: bytes(tabla[c] for c in pieza) for pieza, i in vocab.items()}
    ids = [int(i) for i in np.load(volcado / 'ids.npy')]
    logits = np.load(volcado / 'logits.npy')
    if len(sys.argv) > 4:
        orden = [int(k) for k in sys.argv[4:]]
    else:
        ordenados = np.sort(logits, axis=1)
        margen = ordenados[:, -1] - ordenados[:, -2]
        claros = [k for k in range(16, len(ids)) if margen[k - 1] > np.median(margen)]
        orden = random.Random(3).sample(claros, min(12, len(claros)))

    # Sin penalizaciones: la referencia elige el más probable y nada más.
    entorno = dict(os.environ, HELIOS_REP='1', HELIOS_FREQ='0')
    motor = subprocess.Popen([str(MOTOR), hnf, str(len(ids) + 64), '<|im_end|>'], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=entorno)
    if motor.stdout.readline().strip() != b'READY':
        raise SystemExit('el motor no ha arrancado con ' + hnf)
    bien = 0
    for k in orden:
        prompt = b''.join(por_id[i] for i in ids[:k])
        motor.stdin.write(b'%d\n' % len(prompt) + prompt)
        motor.stdin.flush()
        cabecera = motor.stdout.readline().split()
        salida = motor.stdout.read(int(cabecera[0]))
        prefill, reusados = int(cabecera[5]), int(cabecera[6])
        if prefill + reusados != k:
            raise SystemExit('k=%d: el motor ha contado %d tokens; el tokenizador no da los mismos ids'
                             % (k, prefill + reusados))
        esperado = por_id[int(logits[k - 1].argmax())]
        acierta = salida.startswith(esperado)
        bien += acierta
        print('k=%4d  procesados %4d  reaprovechados %4d  %s  %r' % (
            k, prefill, reusados, 'igual   ' if acierta else 'DISTINTO', esperado.decode(errors='replace')))
    motor.stdin.close()
    print('%d de %d' % (bien, len(orden)))
    return 0 if bien == len(orden) else 1


if __name__ == '__main__':
    sys.exit(main())
