# Gemma 4 — verificación de la Fase 6 contra referencia externa

Criterio de salida de la F6 en `GEMMA4_PLAN.md`: *"comparar logits de prefill y
varios pasos de decode con la referencia usando prompts cortos y otros mayores
que 512 tokens"*. **Cumplido el 2026-07-31.**

## Qué es la referencia

`tools/quant_bench/ref_gemma4.py` — implementación fp32 de Gemma 4 en numpy que lee el
safetensors original de HuggingFace. **No comparte una línea de código con
Héctor.** Las fórmulas están portadas de `modeling_gemma4.py` y
`modeling_rope_utils.py` de transformers, que son la autoridad.

Uso:

    tools/quant_bench/ref_gemma4.py <dir_modelo_hf> <tok1,tok2,...> <salida.bin>
    tools/quant_bench/compare_logits.py <referencia.bin> <hector.bin>

La secuencia de 600 tokens era pseudoaleatoria con semilla 7. Para repetirlo
hoy, usar los corpus reales de `tools/quant_bench/` — el banco completo y las
reglas de medida están documentados en su README.

## Resultado

Contra el HNF **fp16**, que aísla el motor de la cuantización:

| | 3 tokens | 600 tokens (>512) |
|---|---:|---:|
| correlación | 0,999975 | 0,999879 |
| error rms | 0,0907 | 0,0383 |
| error abs máx | 0,1984 | 0,2098 |
| KL(ref‖héctor) | 0,000254 nats | 0,000691 nats |
| argmax | coincide | coincide |
| solapamiento top-10 | 9/10 | 10/10 |
| solapamiento top-100 | 100/100 | 97/100 |

El residuo es el esperado de acumular en fp16 frente a fp32. Queda validada la
ventana deslizante en el límite de 512, el KV compartido de las capas 15-34, la
RoPE proporcional de las globales y la ruta PLE completa.

## Hallazgo aparte: la cuantización sí cuesta

El mismo prompt de 3 tokens contra el HNF **compacto** (HQ4.1K/HQ5.1K):

| | fp16 | compacto |
|---|---:|---:|
| correlación | 0,999975 | 0,990170 |
| rms | 0,0907 | 0,5750 |
| KL | 0,000254 | 0,123965 |
| masa en el top-1 | 0,0603 (ref: 0,0599) | **0,1161** |

El orden se mantiene en lo grueso (argmax igual, top-50 43/50), pero Héctor
concentra **el doble de probabilidad en el token más probable**. Eso no es
ruido: es un afilado sistemático de la distribución, y es exactamente la forma
que tiene el modelo de volverse repetitivo y monótono al muestrear.

No es un fallo del motor —el fp16 lo demuestra— sino del reparto de bits del
conversor para esta arquitectura. Merece una vuelta antes de dar Gemma 4 por
bueno en producción.

## Detalles que la referencia confirmó del código de GPT

- `v_norm` sin pesos (`with_scale=False`) — el `add_rmsnorm_no_weight` sobre V
  es correcto, aunque no exista tensor `v_norm` en el checkpoint.
- Fuentes del KV compartido: última local (13) y última global (14).
- Las capas compartidas no llaman a `update()` del caché.
- Factor `1/√2` al combinar identidad y contexto en el PLE.
- RoPE proporcional: rota 64 de 256 ángulos y **pone a cero los 192 restantes**,
  dividiendo el exponente por `head_dim` completo.
