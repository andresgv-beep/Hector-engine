#!/usr/bin/env python3
"""Referencia con transformers de verdad, para arbitrar entre implementaciones.

Nuestra referencia numpy (`ref_gemma4.py`) y Hector son codigos distintos, pero
los dos salen de LEER `modeling_gemma4.py`. Si el error estuviera en la lectura,
seria un error compartido y ninguna de las dos lo veria. Esto ejecuta la
implementacion oficial y zanja la discusion.

Vuelca un argmax int32 por posicion, en el mismo formato que
`argmax_all_positions_gemma4.cu` y que el modo REF_ALL_POSITIONS de
`ref_gemma4.py`.

Uso:  ref_transformers.py MODEL_DIR IDS_CSV SALIDA.bin
"""
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM

modelo_dir, ids_csv, salida = sys.argv[1], sys.argv[2], sys.argv[3]
ids = [int(x) for x in open(ids_csv).read().strip().split(',')]
print(f'  {len(ids)} tokens', file=sys.stderr)

# fp32: es la referencia, no queremos que el redondeo de bf16 entre en la ecuacion
modelo = AutoModelForCausalLM.from_pretrained(
    modelo_dir, dtype=torch.float32, attn_implementation='eager')
modelo.eval()
print(f'  clase: {type(modelo).__name__}', file=sys.stderr)

with torch.no_grad():
    out = modelo(input_ids=torch.tensor([ids], dtype=torch.long))
    lg = out.logits[0]                      # [S, vocab]
    arg = lg.argmax(dim=-1).to(torch.int32).numpy()

arg.astype(np.int32).tofile(salida)
print(f'  argmax de {arg.size} posiciones -> {salida}', file=sys.stderr)
print(f'  ultima posicion: top1={int(arg[-1])}', file=sys.stderr)
