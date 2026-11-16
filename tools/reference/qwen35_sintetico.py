#!/usr/bin/env python3
"""Fabrica un Qwen3.5 MoE pequeño con pesos al azar, para comparar el motor con la referencia.

    qwen35_sintetico.py <carpeta de salida> [--tokenizer <carpeta de un Qwen3.5 real>]

No existe un hermano pequeño del Qwen3.6-35B-A3B: los Qwen3.5 pequeños son densos. Este modelo
tiene su misma forma (capas DeltaNet y de atención con puerta, MoE con experto compartido,
cabeza sin compartir) en pequeño. No sabe nada; sirve para ver que el motor calcula lo mismo
que tools/reference/qwen35_ref.py, enrutado incluido. Los pesos se guardan en FP16 para que
los dos lean exactamente los mismos números.

El conversor pide tokenizer: con --tokenizer se copian los ficheros de un modelo real, y el
vocabulario se pone a su tamaño.
"""
import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np

CAPAS = ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'] * 2
H = 512
ATENCION = dict(num_attention_heads=4, num_key_value_heads=2, head_dim=128)
LINEAL = dict(linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=128, linear_value_head_dim=128,
              linear_conv_kernel_dim=4)
MOE = dict(num_experts=32, num_experts_per_tok=4, moe_intermediate_size=256, shared_expert_intermediate_size=256)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('salida')
    ap.add_argument('--tokenizer')
    ap.add_argument('--vocab', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    vocab = args.vocab
    if args.tokenizer:
        origen = Path(args.tokenizer)
        for fichero in origen.iterdir():
            if fichero.suffix != '.safetensors' and fichero.name != 'config.json' and fichero.is_file():
                shutil.copy(fichero, salida / fichero.name)
        real = json.loads((origen / 'config.json').read_text())
        vocab = real.get('text_config', real)['vocab_size']

    rng = np.random.default_rng(args.seed)
    tensores = {}

    def lineal(nombre, filas, columnas, ganancia=1.0):
        tensores[nombre] = (rng.standard_normal((filas, columnas)) * ganancia / np.sqrt(columnas)).astype(np.float16)

    def vector(nombre, n, escala=0.1, centro=0.0):
        tensores[nombre] = (centro + rng.standard_normal(n) * escala).astype(np.float16)

    raiz = 'model.language_model.'
    tensores[raiz + 'embed_tokens.weight'] = rng.standard_normal((vocab, H)).astype(np.float16)
    lineal('lm_head.weight', vocab, H)
    vector(raiz + 'norm.weight', H)
    nh, nkv, hd = ATENCION['num_attention_heads'], ATENCION['num_key_value_heads'], ATENCION['head_dim']
    nk, nv = LINEAL['linear_num_key_heads'], LINEAL['linear_num_value_heads']
    dk, dv, K = LINEAL['linear_key_head_dim'], LINEAL['linear_value_head_dim'], LINEAL['linear_conv_kernel_dim']
    E, I, IC = MOE['num_experts'], MOE['moe_intermediate_size'], MOE['shared_expert_intermediate_size']
    for i, tipo in enumerate(CAPAS):
        p = '%slayers.%d.' % (raiz, i)
        vector(p + 'input_layernorm.weight', H)
        vector(p + 'post_attention_layernorm.weight', H)
        if tipo == 'linear_attention':
            a = p + 'linear_attn.'
            canales = 2 * nk * dk + nv * dv
            lineal(a + 'in_proj_qkv.weight', canales, H)
            lineal(a + 'in_proj_z.weight', nv * dv, H)
            lineal(a + 'in_proj_a.weight', nv, H)
            lineal(a + 'in_proj_b.weight', nv, H)
            tensores[a + 'conv1d.weight'] = (rng.standard_normal((canales, 1, K)) * 0.5).astype(np.float16)
            vector(a + 'A_log', nv, 0.5)
            vector(a + 'dt_bias', nv, 0.5)
            vector(a + 'norm.weight', dv, 0.1, 1.0)
            lineal(a + 'out_proj.weight', H, nv * dv)
        else:
            a = p + 'self_attn.'
            lineal(a + 'q_proj.weight', 2 * nh * hd, H)
            lineal(a + 'k_proj.weight', nkv * hd, H)
            lineal(a + 'v_proj.weight', nkv * hd, H)
            lineal(a + 'o_proj.weight', H, nh * hd)
            vector(a + 'q_norm.weight', hd)
            vector(a + 'k_norm.weight', hd)
        m = p + 'mlp.'
        # Enrutador con ganancia: que las probabilidades de los elegidos no queden empatadas.
        lineal(m + 'gate.weight', E, H, 3.0)
        tensores[m + 'experts.gate_up_proj'] = (rng.standard_normal((E, 2 * I, H)) / np.sqrt(H)).astype(np.float16)
        tensores[m + 'experts.down_proj'] = (rng.standard_normal((E, H, I)) / np.sqrt(I)).astype(np.float16)
        lineal(m + 'shared_expert.gate_proj.weight', IC, H)
        lineal(m + 'shared_expert.up_proj.weight', IC, H)
        lineal(m + 'shared_expert.down_proj.weight', H, IC)
        lineal(m + 'shared_expert_gate.weight', 1, H)

    cabecera, lugar = {}, 0
    for nombre, valores in tensores.items():
        cabecera[nombre] = {'dtype': 'F16', 'shape': list(valores.shape), 'data_offsets': [lugar, lugar + valores.nbytes]}
        lugar += valores.nbytes
    texto = json.dumps(cabecera, separators=(',', ':')).encode()
    texto += b' ' * (-len(texto) % 8)
    with open(salida / 'model.safetensors', 'wb') as f:
        f.write(struct.pack('<Q', len(texto)))
        f.write(texto)
        for valores in tensores.values():
            f.write(valores.tobytes())

    config = dict(model_type='qwen3_5_moe', text_config=dict(
        model_type='qwen3_5_moe_text', hidden_size=H, num_hidden_layers=len(CAPAS), layer_types=CAPAS,
        vocab_size=vocab, rms_norm_eps=1e-6, hidden_act='silu', attn_output_gate=True, tie_word_embeddings=False,
        max_position_embeddings=4096,
        rope_parameters=dict(rope_theta=10000000, partial_rotary_factor=0.25, rope_type='default'),
        **ATENCION, **LINEAL, **MOE))
    (salida / 'config.json').write_text(json.dumps(config, indent=2))
    print('%d tensores, %.1f MB, en %s' % (len(tensores), lugar / 1e6, salida))


if __name__ == '__main__':
    main()
