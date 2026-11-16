#!/usr/bin/env python3
"""Compara dos volcados por capas: el de la referencia (qwen35_ref.py) y el del motor (qwen35_dump).

    comparar_volcados.py <carpeta de la referencia> <carpeta del motor> [umbral] [--resumen]

Por cada tensor da el error relativo (norma de la diferencia entre norma de la referencia), el
peor token y el coseno. Al final, si el motor elige el mismo token siguiente en cada posición.
Sale con 1 si algún tensor se aparta más del umbral. El umbral por defecto es para FP16 contra
FP32; con pesos cuantizados hay que pasar otro.

Con secuencias largas da además el error de la salida final por cuartos de la secuencia: en esta
familia el estado de las capas lineales es recurrente y un error de redondeo no se queda en su
token, así que hay que mirar si crece con el contexto. `--resumen` calla la tabla por tensor.
"""
import sys
from pathlib import Path

import numpy as np

UMBRAL = 0.02          # FP16 en el motor contra FP32 en la referencia: del orden de 1e-3


def main():
    resumen = '--resumen' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--resumen']
    if len(args) not in (2, 3):
        raise SystemExit(__doc__)
    ref, motor = Path(args[0]), Path(args[1])
    umbral = float(args[2]) if len(args) == 3 else UMBRAL
    decir = (lambda *a: None) if resumen else print
    nombres = sorted(p.stem for p in ref.glob('*.npy') if p.stem != 'ids')
    orden = lambda n: (0, n) if n == 'emb' else (2, n) if n in ('final', 'logits') else (1, n[-2:] + n[:4])
    peor, fallos = 0.0, []
    decir('%-8s %10s %10s %10s' % ('tensor', 'error', 'peor tok.', 'coseno'))
    for nombre in sorted(nombres, key=orden):
        a = np.load(ref / (nombre + '.npy')).astype(np.float64)
        ruta = motor / (nombre + '.npy')
        if not ruta.exists():
            print('%-8s %s' % (nombre, 'FALTA en el motor')); fallos.append(nombre); continue
        b = np.load(ruta).astype(np.float64)
        if a.shape != b.shape:
            print('%-8s formas distintas: %s y %s' % (nombre, a.shape, b.shape)); fallos.append(nombre); continue
        if not np.isfinite(b).all():
            print('%-8s valores no finitos en el motor' % nombre); fallos.append(nombre); continue
        por_token = np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(a, axis=1), 1e-12)
        total = np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)
        coseno = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        decir('%-8s %10.5f %10.5f %10.6f%s' % (nombre, total, por_token.max(), coseno, '   <--' if total > umbral else ''))
        peor = max(peor, total)
        if total > umbral:
            fallos.append(nombre)
    a, b = np.load(ref / 'logits.npy'), np.load(motor / 'logits.npy')
    iguales = int((a.argmax(axis=1) == b.argmax(axis=1)).sum())
    if len(a) >= 16:
        fa, fb = np.load(ref / 'final.npy').astype(np.float64), np.load(motor / 'final.npy').astype(np.float64)
        cuartos = np.array_split(np.arange(len(a)), 4)
        errores = [np.linalg.norm(fa[c] - fb[c]) / np.linalg.norm(fa[c]) for c in cuartos]
        aciertos = [int((a[c].argmax(axis=1) == b[c].argmax(axis=1)).sum()) for c in cuartos]
        print('salida final por cuartos de la secuencia: ' + '  '.join('%.4f' % e for e in errores))
        print('token siguiente por cuartos:              ' + '  '.join('%d/%d' % (n, len(c)) for n, c in zip(aciertos, cuartos)))
    print('token siguiente: %d de %d iguales' % (iguales, len(a)))
    print('peor error relativo: %.5f (umbral %.2f)' % (peor, umbral))
    if fallos:
        print('se apartan: ' + ', '.join(fallos[:12]))
    return 1 if fallos or iguales != len(a) else 0


if __name__ == '__main__':
    sys.exit(main())
