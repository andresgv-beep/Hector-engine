#!/usr/bin/env python3
"""Fixture de una capa completa de Gemma 4 MoE, ejecutada por Transformers.

Corre Gemma4TextModel de verdad y captura, con ganchos, lo que entra en la cola
de la capa (el residual tras la atención) y lo que sale. Uso: salida.bin
"""
import struct
import sys

import torch
from transformers.models.gemma4 import modeling_gemma4 as m
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

L, E, K, H, I, M = 6, 16, 4, 32, 8, 48
torch.manual_seed(2816)
config = Gemma4TextConfig(hidden_size=H, intermediate_size=M, num_experts=E, top_k_experts=K, moe_intermediate_size=I,
                          enable_moe_block=True, hidden_activation='gelu_pytorch_tanh', num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=2, num_global_key_value_heads=2, head_dim=8,
                          global_head_dim=8, vocab_size=64, hidden_size_per_layer_input=0,
                          layer_types=['sliding_attention'], sliding_window=16, attn_implementation='eager')
modelo = m.Gemma4TextModel(config).float().eval()
capa = modelo.layers[0]
for nombre, parametro in capa.named_parameters():
    if 'self_attn' not in nombre:
        torch.nn.init.normal_(parametro, mean=1.0 if 'norm' in nombre or 'scale' in nombre else 0.0, std=0.3)
capa.layer_scalar.fill_(0.9)

captura = {}
capa.pre_feedforward_layernorm.register_forward_pre_hook(lambda _m, entrada: captura.update(residual=entrada[0].clone()))
capa.register_forward_hook(lambda _m, _e, salida: captura.update(salida=(salida[0] if isinstance(salida, tuple) else salida).clone()))
with torch.no_grad():
    modelo(input_ids=torch.arange(1, L + 1)[None], use_cache=False)

router, expertos = capa.router, capa.experts
tensores = [
    ('ln_mlp_in.weight', capa.pre_feedforward_layernorm.weight), ('ln_mlp_branch_post.weight', capa.post_feedforward_layernorm_1.weight),
    ('ln_moe_in.weight', capa.pre_feedforward_layernorm_2.weight), ('ln_moe_branch_post.weight', capa.post_feedforward_layernorm_2.weight),
    ('ln_mlp_post.weight', capa.post_feedforward_layernorm.weight), ('mlp.gate.weight', capa.mlp.gate_proj.weight),
    ('mlp.up.weight', capa.mlp.up_proj.weight), ('mlp.down.weight', capa.mlp.down_proj.weight),
    ('moe.router.weight', router.proj.weight), ('moe.router.input_scale', router.scale),
    ('moe.router.expert_scale', router.per_expert_scale), ('moe.experts.gate_up.weight', expertos.gate_up_proj),
    ('moe.experts.down.weight', expertos.down_proj), ('layer_scalar', capa.layer_scalar)]
with open(sys.argv[1], 'wb') as f:
    f.write(struct.pack('<8i', 0x4D4F4532, L, E, K, H, I, M, len(tensores)))
    for nombre, tensor in tensores:
        crudo = nombre.encode()
        f.write(struct.pack('<ii', len(crudo), tensor.dim()) + crudo + struct.pack('<%di' % tensor.dim(), *tensor.shape))
        f.write(tensor.detach().float().numpy().tobytes())
    f.write(captura['residual'][0].float().numpy().tobytes())
    f.write(captura['salida'][0].float().numpy().tobytes())
print('residual', float(captura['residual'].abs().mean()), 'salida', float(captura['salida'].abs().mean()))
