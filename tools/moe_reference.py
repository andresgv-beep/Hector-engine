#!/usr/bin/env python3
"""Fixture de la rama MoE de Gemma 4, sacada del código oficial de Transformers.

No reimplementa nada: llama a Gemma4TextRouter y Gemma4TextExperts con pesos
aleatorios pequeños. El motor se compara contra esto, no contra otra lectura
nuestra del modelo. Uso: moe_reference.py salida.bin
"""
import struct
import sys

import torch
from transformers.models.gemma4 import modeling_gemma4 as m
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

L, E, K, H, I = 7, 16, 4, 32, 8
torch.manual_seed(26)
config = Gemma4TextConfig(hidden_size=H, num_experts=E, top_k_experts=K, moe_intermediate_size=I,
                          enable_moe_block=True, hidden_activation='gelu_pytorch_tanh')
router = m.Gemma4TextRouter(config).float()
experts = m.Gemma4TextExperts(config).float()
for parametro in [*router.parameters(), *experts.parameters()]:
    torch.nn.init.normal_(parametro, std=0.5)
residual = torch.randn(L, H)
entrada = torch.randn(L, H)
with torch.no_grad():
    normalizado = router.norm(residual)
    escalado = normalizado * router.scale * router.scalar_root_size
    logits = router.proj(escalado)
    _, pesos, indices = router(residual)
    salida = experts(entrada, indices, pesos)
tensores = [residual, router.scale, router.proj.weight, router.per_expert_scale, entrada, experts.gate_up_proj,
            experts.down_proj, logits, pesos, salida]
with open(sys.argv[1], 'wb') as f:
    f.write(struct.pack('<6i', 0x4D4F4531, L, E, K, H, I))
    f.write(indices.to(torch.int32).numpy().tobytes())
    for tensor in tensores:
        f.write(tensor.detach().float().numpy().tobytes())
print('índices', indices[0].tolist(), 'salida', float(salida.abs().mean()))
