#!/usr/bin/env python3
"""Aísla el error de cada bloque: el motor contra el original, los dos con la misma entrada.

    qwen35_aislar.py <carpeta del modelo original> <volcado de la referencia> <volcado del motor> [--moe]

`comparar_volcados.py` da el error de cada tensor con todo lo que arrastra de las capas
anteriores. Aquí cada bloque del original recibe la entrada que tuvo el motor (su volcado de la
capa anterior), así que lo que queda es solo lo que ese bloque hace distinto: sus pesos
cuantizados o un fallo de cálculo. Por capa:

  entrada   error con el que le llega el residual al motor
  propio    bloque del motor contra el del original, con la entrada del motor
  arrastre  bloque del original con la entrada del motor contra el de la referencia: cuánto
            amplifica ese bloque el error que le llega

Con `--moe` hace lo mismo con la MLP de expertos (lento: lee los expertos del original) y da
además cuántos de los expertos elegidos coinciden con los del motor.
"""
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen35_ref import Modelo, rms_centrada


def rel(a, b):
    return float(np.linalg.norm(a.astype(np.float64) - b) / max(np.linalg.norm(b.astype(np.float64)), 1e-12))


def main():
    moe = '--moe' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--moe']
    if len(args) != 3:
        raise SystemExit(__doc__)
    modelo = Modelo(args[0])
    ref, motor = Path(args[1]), Path(args[2])
    r = lambda n: np.load(ref / (n + '.npy'))
    m = lambda n: np.load(motor / (n + '.npy'))
    print('%-5s %-8s %9s %9s %9s' % ('capa', 'tipo', 'entrada', 'propio', 'arrastre')
          + ('   %9s %9s %9s' % ('moe prop.', 'moe arr.', 'expertos') if moe else ''), flush=True)
    for i, capa in enumerate(modelo.capas):
        entra_m = m('emb') if i == 0 else m('capa%02d' % (i - 1))
        entra_r = r('emb') if i == 0 else r('capa%02d' % (i - 1))
        attn_m, attn_r = m('attn%02d' % i), r('attn%02d' % i)
        modelo.reiniciar()
        estado = modelo.estado[i]
        lineal = capa['tipo'] == 'linear_attention'
        mezcla = np.empty_like(attn_m)
        for t in range(len(entra_m)):
            modelo.pos3 = (t, t, t)
            x = rms_centrada(entra_m[t], capa['ln1'], modelo.eps)
            mezcla[t] = modelo._deltanet(capa, estado, x) if lineal else modelo._atencion(capa, estado, x)
        linea = '%-5d %-8s %9.4f %9.4f %9.4f' % (i, 'lineal' if lineal else 'atención', rel(entra_m, entra_r),
                                                 rel(attn_m, mezcla), rel(mezcla, attn_r))
        if moe:
            medio_m, medio_r = entra_m + attn_m, entra_r + attn_r
            mlp_m = m('capa%02d' % i) - medio_m
            mlp_r = r('capa%02d' % i) - medio_r
            volcar = collections.defaultdict(list)
            mlp = np.empty_like(mlp_m)
            for t in range(len(medio_m)):
                mlp[t] = modelo._mlp(capa, rms_centrada(medio_m[t], capa['ln2'], modelo.eps), volcar, i)
            rutas = np.stack(volcar['ruta%02d' % i]).astype(np.int64)
            del_motor = m('ruta%02d' % i).astype(np.int64)
            comunes = np.mean([len(set(a) & set(b)) for a, b in zip(rutas, del_motor)])
            linea += '   %9.4f %9.4f %6.2f/%d' % (rel(mlp_m, mlp), rel(mlp, mlp_r), comunes, rutas.shape[1])
        print(linea, flush=True)


if __name__ == '__main__':
    main()
