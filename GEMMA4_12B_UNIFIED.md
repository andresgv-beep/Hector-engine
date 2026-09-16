# Gemma 4 12B «unified» — motor

> Abierto el 2026-09-16. **El texto funciona y genera lenguaje correcto.** La
> visión carga desde el HNF pero el motor todavía no la ejecuta (ver «Dónde
> quedó»).

## Principio que rige todos los cambios

Cada rama nueva está condicionada a un flag del GM4X o a un campo por capa, y
**flag apagado o campo a cero ⇒ el comportamiento de siempre**. Los HNF de
E2B/E4B traen 0 en los campos nuevos, así que recorren exactamente el mismo
código que antes. Verificado tras cada paso ejecutando el E4B.

## Lo que «unified» rompe respecto a E2B/E4B

1. **KV heads por capa.** 8 en las 40 deslizantes, **1** en las 8 globales. Antes
   el número era único para todo el modelo.
2. **`attention_k_eq_v`.** Las capas globales no tienen `v_proj`: V se proyecta
   con el peso de K. Confirmado en la referencia
   (`modeling_gemma4_unified.py`, `Gemma4UnifiedTextAttention.forward`):

   ```python
   value_states = self.v_proj(h) if self.v_proj is not None else key_states
   key_states   = self.k_norm(key_states);  key_states = apply_rotary_pos_emb(...)
   value_states = self.v_norm(value_states)          # sin RoPE
   ```

   V sale de `k_proj` **antes** de `k_norm` y de RoPE, y lleva sólo `v_norm`
   (sin escala).
3. **Sin PLE.** `hidden_size_per_layer_input: 0`.
4. **`layer_scalar` sigue existiendo.** La referencia cierra cada capa con
   `hidden_states *= self.layer_scalar`. En Héctor ese producto vivía DENTRO de
   la inyección de PLE, así que al saltarse el PLE se perdía la escala de las 48
   capas — y el modelo emitía `<pad>` indefinidamente. Fue el último fallo en
   caer y el más caro de encontrar.

## Qué se tocó

| Fichero | Cambio |
|---|---|
| `src/hnf_loader.hpp` | `GEMMA4_LAYER_FLAG_K_EQ_V`; el `reserved` del registro pasa a ser `num_kv_heads`; helpers `k_eq_v()` y `kv_heads_or(global)`. |
| `src/hnf_loader.cpp` | Se parsea `num_kv_heads` por capa. |
| `src/gemma4_kv_cache.hpp` | `kv_heads_` **por capa** (antes uno global), usado en el stride, en las shapes registradas y en la comprobación de las capas compartidas. |
| `src/graph_builder.cpp` | KV heads por capa en las dos rutas Gemma 4; V proyectada con `k_proj` si `k_eq_v`; vistas del scratch con los KV heads de la capa; PLE opcional en scratch, entrada de texto y entrada multimodal; **`layer_scalar` aplicado cuando no hay PLE**. |
| `src/graph_builder.hpp` | `validate_weights` acepta un `Gemma4Config*` opcional (default `nullptr`: los 9 llamadores siguen igual). |
| `src/gemma4_validator.cpp` | Anchura KV por capa; no se exige `v_proj` en capas con `k_eq_v`. |
| `kernels/gemma4_vision.cu` | **Nuevo kernel** `gemma4_unified_pos_add_fp16`: posiciones factorizadas X+Y, con `-1` = relleno que aporta cero. |
| `kernels/kernels.hpp`, `kernels/register_kernels.cpp`, `src/optype.{hpp,cpp}` | Op `g4u_pos_add` declarada y registrada. |
| `test_smoke.cpp` | Pasa el `Gemma4Config` al validador cuando el HNF lo trae. |

## Verificado

| | |
|---|---|
| **12B texto** | carga 666 tensores (9114 MB VRAM), 48 capas, `lm_head: tied`, grafo de 1061 comandos, CUDA Graph capturado |
| **Generación** | «La capital de Francia es **París**.» · definición correcta de fotosíntesis |
| **Velocidad** | 42,6 tok/s decode, prefill 178 ms (RTX 4070 Ti, CUDA 13.1, sm_89) |
| **E4B (regresión)** | intacto: 85,3 tok/s, misma salida que antes de tocar nada |

## Dónde quedó

### Visión: falta la mitad

✅ Bloque 0x1 escrito y validado en el HNF (95,2 MB, 10 tensores).
✅ Kernel de posiciones factorizadas, compilando y registrado.
❌ **Config binario.** El `GM4V` actual pide `num_key_value_heads`, `head_dim`,
   `rope_theta`, `attention_scale` — campos de torre que aquí no existen — y le
   faltan `mm_posemb_size`, `num_soft_tokens` y `model_patch_size`.
❌ **Preprocesado.** Parches de **48×48×3 = 6912** (no 16), `do_normalize: false`
   (sólo rescale 1/255, sin media ni desviación) y coordenadas `(x, y)` por
   parche con `-1` en el relleno.
❌ **Grafo.** Falta encadenar `LN₁ → Dense+bias → LN₂ → +pos → LN₃ → RMSNorm sin
   escala → Linear` y meter los **280 soft tokens** en el decoder.

El resto del pipeline NO necesita kernels nuevos: `op::LAYERNORM()` ya acepta
bias en `ctx.in(2)`, y `MATMUL`, `ADD_BIAS` y la RMSNorm sin escala ya existen.

### Fallo preexistente, ajeno a estos cambios

`test_smoke` fase 4 («Prefill con KV cache») falla en cualquier modelo: en
`test_smoke.cpp:608` hay un `cudaMemcpy` sin comprobar el retorno tras
`free_scratch()` + `allocate_scratch()`. El error queda pendiente y CUDA lo
reporta en la siguiente llamada que sí lo mira — el `embedding` —, lo que hace
parecer que el fallo está en el kernel de embeddings. No se ha tocado.

La fase 3 de `test_smoke` usa la ruta genérica `build_attention_block`, que
exige `v_proj` en todas las capas; por eso un 12B falla ahí aunque funcione en
`helios_chat`. Tampoco se ha tocado: esa ruta la comparten todas las
arquitecturas.
