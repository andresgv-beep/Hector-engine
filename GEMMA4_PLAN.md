# Gemma 4 E2B en Héctor — reconocimiento y plan

> Reconocimiento hecho el 2026-07-30 sobre
> `~/Documentos/GitHub/helios_convert_v9.1/models/gemma-4-E2B`
> (10.2 GB, `Gemma4ForConditionalGeneration`, texto + visión + audio).
> Motivación: Andrés lo probó fuera de HELIOS — la calidad de respuesta es
> claramente superior a Qwen3-4B, que es el techo que nos frena hoy.

## Datos del modelo (text_config)

| Parámetro | Valor |
|---|---|
| Capas | 35 |
| hidden_size | 1536 |
| intermediate_size | 6144 |
| heads / kv_heads | 8 / **1 (MQA)** |
| head_dim | 256 (global_head_dim: 512) |
| vocab | 262.144 |
| activación | `gelu_pytorch_tanh` (GeGLU) |
| sliding_window | 512, alternando 4 sliding + 1 full |
| num_kv_shared_layers | **20** |
| final_logit_softcapping | 30.0 |
| tie_word_embeddings | sí (no hay lm_head) |
| hidden_size_per_layer_input | 256 |

Tensores por capa (language_model): `input_layernorm`,
`post_attention_layernorm`, `pre_feedforward_layernorm`,
`post_feedforward_layernorm`, `mlp.{gate,up,down}_proj`,
`self_attn.{q,k,v,o}_proj`, `self_attn.{q,k}_norm`, **`per_layer_input_gate`
[256,1536]**, **`per_layer_projection` [1536,256]**,
**`post_per_layer_input_norm`**, **`layer_scalar` [1]**.

Globales: `embed_tokens [262144,1536]`, **`embed_tokens_per_layer
[262144, 8960]`** (= 35 × 256), `per_layer_model_projection [8960,1536]`,
`per_layer_projection_norm [256]`, `norm`.

## Lo que Héctor ya cubre (gratis)

- MQA (kv_heads=1) → caso extremo del GQA que ya soporta
- q_norm / k_norm por-head → hecho para Qwen3 (`use_qk_norm`)
- GeGLU → kernels de gelu ya existentes
- Embeddings atados sin lm_head → resuelto (el conversor emite lm_head
  cuantizado desde el embedding)
- Prefill batch, compactación, memoria → agnósticos del modelo

## Lo que hay que construir

1. **Per-Layer Embeddings (PLE)** — el grueso. Una SEGUNDA tabla de
   embeddings (2.350M params, más grande que el resto del modelo) que
   inyecta un vector propio a cada capa: lookup → `per_layer_model_projection`
   → norm → por capa: `per_layer_input_gate` + `per_layer_projection` +
   `post_per_layer_input_norm`. Camino de datos nuevo en el graph builder.
   Es el truco que da calidad de 8B con coste de 2B.
2. **Sliding window attention alternada** (4×512 + 1 global). Máscara por
   capa; el kernel de decode necesita límite inferior de posición.
3. **KV compartido entre 20 capas**: las últimas 20 reutilizan KV de capas
   previas. Hoy `KVCache` asume una entrada por capa.
4. **Doble head_dim** (256 sliding / 512 global) — verificar en los pesos.
5. **Detalles pequeños y letales** (silencian el modelo sin petar):
   - RMSNorm de Gemma: `(1 + weight) * normalized`, ¡el peso va desplazado!
   - Embedding escalado por `sqrt(hidden_size)`
   - Softcapping de logits: `tanh(logits/30)*30`
   - Cuatro normas por capa (sandwich) + `layer_scalar`

## Plan sugerido (2-4 días)

1. Mapper `gemma4` en el conversor + hints (medio día)
2. Detalles pequeños del punto 5 en el motor (medio día) — hacerlos PRIMERO
   y verificar con un forward de 1 capa contra referencia
3. PLE en graph builder + kernels (1-1.5 días)
4. Sliding window + KV compartido (1 día)
5. Verificación: perplejidad o texto coherente en español/inglés

**Riesgo principal**: bugs numéricos silenciosos (el modelo genera, pero
peor). Mitigación: comparar activaciones capa a capa contra HF/transformers
en las primeras 2 capas antes de seguir.

## Alternativa barata (si urge calidad)

Qwen3-8B en HQ4.1K: ~5 GB, 45-55 tok/s estimados, **cero código** (misma
arquitectura que el 4B ya soportado). Es el atajo si lo que se quiere es
mejor conversación esta semana; Gemma 4 es la apuesta arquitectónica.

## Regalo colateral

El archivo trae `vision_tower` y `audio_tower` completos → los bloques
0x1 (visión) y 0x2 (audio) del HNF dejarían de estar vacíos. Multimodal
real en el mismo formato, cuando toque.
