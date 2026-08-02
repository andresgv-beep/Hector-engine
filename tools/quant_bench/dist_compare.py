#!/usr/bin/env python3
"""Por que Gemma 4 se cuantiza peor que Qwen3 con el mismo formato.

Descompone el error de reconstruccion de HQ4.1K/HQ5.1K en sus dos fuentes:

  1. cuantizar los PESOS a q_max niveles dentro de cada grupo de 8
  2. cuantizar la ESCALA y el MINIMO de cada grupo a 4 bits dentro del
     superbloque de 256

La segunda no existe en Q4_0 de llama.cpp, que guarda una escala fp16 entera
por cada 32 pesos. Si Gemma nos castiga a nosotros y a ellos no, el sospechoso
natural es (2): grupos con escalas muy dispares dentro de un superbloque.

Uso:  dist_compare.py <dir_gemma4> <dir_qwen3>
"""
import glob, json, sys
import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from hqs_sim import quant_dequant

SUPER, GRUPO = 256, 8


def carga_bf16(directorio, nombre):
    """Lee un tensor bf16 de un safetensors shardeado, como float32."""
    for ruta in sorted(glob.glob(f'{directorio}/*.safetensors')):
        with open(ruta, 'rb') as f:
            n = int.from_bytes(f.read(8), 'little')
            cab = json.loads(f.read(n))
            if nombre not in cab:
                continue
            info = cab[nombre]
            ini, fin = info['data_offsets']
            f.seek(8 + n + ini)
            crudo = np.frombuffer(f.read(fin - ini), dtype=np.uint16)
            # bf16 -> f32: son los 16 bits altos de un float32
            f32 = (crudo.astype(np.uint32) << 16).view(np.float32)
            return f32.reshape(info['shape'])
    return None


def nrmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()) / (a.std() + 1e-12))


def analiza(w, q_max):
    """Error real del formato y error con escalas exactas."""
    real = nrmse(w, quant_dequant(w, q_max, group_size=GRUPO, scale_bits=4))
    # scale_bits alto = las escalas dejan de ser el cuello de botella
    exacto = nrmse(w, quant_dequant(w, q_max, group_size=GRUPO, scale_bits=14))

    # dispersion de las escalas de grupo dentro de cada superbloque: es lo que
    # los 4 bits de escala tienen que cubrir
    plano = w.astype(np.float32).ravel()
    plano = plano[:plano.size - plano.size % SUPER]
    g = plano.reshape(-1, SUPER // GRUPO, GRUPO)
    esc = g.max(axis=2) - g.min(axis=2)
    disp = float(np.median(esc.max(axis=1) / np.maximum(esc.mean(axis=1), 1e-12)))
    return real, exacto, disp


TENSORES = {
    'gemma4': [('mlp.gate', 'model.language_model.layers.{L}.mlp.gate_proj.weight', 15),
               ('mlp.down', 'model.language_model.layers.{L}.mlp.down_proj.weight', 15),
               ('attn.q',   'model.language_model.layers.{L}.self_attn.q_proj.weight', 31),
               ('attn.o',   'model.language_model.layers.{L}.self_attn.o_proj.weight', 31)],
    'qwen3':  [('mlp.gate', 'model.layers.{L}.mlp.gate_proj.weight', 15),
               ('mlp.down', 'model.layers.{L}.mlp.down_proj.weight', 15),
               ('attn.q',   'model.layers.{L}.self_attn.q_proj.weight', 15),
               ('attn.o',   'model.layers.{L}.self_attn.o_proj.weight', 15)],
}

dirs = {'gemma4': sys.argv[1], 'qwen3': sys.argv[2]}
CAPAS = [2, 12, 24]

print(f"  {'modelo':8} {'tensor':10} {'capa':>4} {'NRMSE real':>11} {'esc.exactas':>12} "
      f"{'peaje escala':>13} {'dispersion':>11}")
resumen = {}
for modelo, plantillas in TENSORES.items():
    acum = []
    for etiqueta, plantilla, q_max in plantillas:
        for L in CAPAS:
            w = carga_bf16(dirs[modelo], plantilla.format(L=L))
            if w is None:
                print(f'  {modelo:8} {etiqueta:10} {L:4}  no encontrado')
                continue
            real, exacto, disp = analiza(w, q_max)
            peaje = 100 * (real - exacto) / real
            acum.append((real, peaje, disp))
            print(f'  {modelo:8} {etiqueta:10} {L:4} {real:11.4f} {exacto:12.4f} '
                  f'{peaje:12.1f}% {disp:11.2f}')
    if acum:
        a = np.array(acum)
        resumen[modelo] = a.mean(axis=0)

print()
for modelo, m in resumen.items():
    print(f'  MEDIA {modelo:8} NRMSE {m[0]:.4f}   peaje de escala {m[1]:.1f}%   '
          f'dispersion {m[2]:.2f}')
if len(resumen) == 2:
    g, q = resumen['gemma4'], resumen['qwen3']
    print(f'\n  Gemma tiene {g[0]/q[0]:.2f}x el error de Qwen3 con el mismo formato.')
