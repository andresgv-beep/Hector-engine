# Gemma 4 E2B Vision — puente multimodal V5

> Cerrado el 2026-08-02. Esta fase conecta la salida visual al decoder sin
> introducir decodificación de imágenes ni una interfaz de producto.

## Contrato implementado

Para una entrada con un único placeholder `IMAGE`, Héctor construye:

```text
texto + BOI + IMAGE × N + EOI + texto
```

donde `N` es el número dinámico de soft tokens producido por la torre visual.
Los IDs especiales proceden de `GM4V`; no están duplicados en el grafo de
producto. V5 admite una imagen por secuencia y rechaza cero/múltiples
placeholders, conteos superiores a `max_soft_tokens` e IDs fuera del vocabulario.

Upstream tiene dos caminos que no se pueden confundir:

1. el embedding principal busca PAD=0 en las posiciones IMAGE, aplica la escala
   `sqrt(hidden)` al texto y después sustituye esas filas por la proyección
   visual FP16;
2. la identidad PLE busca también PAD=0, pero su proyección contextual consume
   los embeddings principales después de la sustitución visual.

El nuevo `scatter_rows` CUDA hace esa sustitución en el mismo stream y es
capturable por CUDA Graph. `Gemma4VisionRunner::projected_states_device()`
expone la frontera `[N,1536]` sin copia host. El prefill multimodal usa el KV
existente; los pasos posteriores vuelven a la ruta textual normal y no ejecutan
otra vez la torre visual.

## Regresión de texto

Antes de editar se congelaron los logits de los IDs
`2,105,200,108,106,105,107`. Después de implementar V5 se repitió el forward
con el mismo HNF y `HELIOS_EMBED_MMAP=1`:

| salida | bytes | SHA256 |
|---|---:|---|
| antes de V5 | 1.048.576 | `e8490bf94b415dc125ca5c58bff2b83f45f7de220283b57dfde95a1b0384a17f` |
| después de V5 | 1.048.576 | `e8490bf94b415dc125ca5c58bff2b83f45f7de220283b57dfde95a1b0384a17f` |

`cmp` no encontró ninguna diferencia. La ruta histórica de texto no fue
redirigida al nuevo builder multimodal.

## Comparación externa FP16

La entrada de control contiene 280 posiciones visuales y 285 tokens totales.
La proyección procede del oráculo upstream V0 y se redondea a FP16 en ambos
lados para aislar exactamente la frontera del puente. El decoder de referencia
lee los pesos BF16 originales y aplica las fórmulas fijadas del commit
Transformers `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`.

El HNF de control conserva todos los pesos de texto en FP16. Como no cabe por
su `lm_head` separado, la verificación activa explícitamente
`HELIOS_VERIFY_LM_HEAD_MMAP=1` junto a `HELIOS_EMBED_MMAP=1`. Esta bandera es
solo de laboratorio: lee el head completo por HMM y no se activa ni se
recomienda para inferencia normal.

| frontera | correlación | RMS | KL | argmax | top-10 |
|---|---:|---:|---:|---:|---:|
| prefill | 0,999966328 | 0,112854 | 0,00015864 | 1 / 1 | 10/10 |
| decode 1 | 0,999950488 | 0,054630 | 0,00008736 | 106 / 106 | 10/10 |
| decode 2 | 0,999889355 | 0,111794 | 0,00005756 | 107 / 107 | 10/10 |
| decode 3 | 0,999901402 | 0,141502 | 0,00008735 | 1 / 1 | 10/10 |

La secuencia greedy coincide `1, 106, 107`; el cuarto argmax vuelve a ser `1`.
Esto verifica prefill, continuidad del KV y decode sin rerun visual.

## Control HQS

El HNF compacto de producción acierta los dos primeros tokens, pero en el
tercero elige `106` donde FP16/BF16 eligen `107`. En prefill conserva correlación
0,985950, KL 0,05915 y 9/10 del top-10. El mismo control solo texto de 285 tokens
es peor frente a BF16 (correlación 0,915701 y 6/10 del top-10), y el HNF FP16
multimodal recupera toda la paridad. Por tanto, la divergencia no la introduce
el puente V5: pertenece al perfil HQS ya abierto y el compacto no sirve para
certificar correctitud multimodal.

## Pruebas y salida

- 18/18 pruebas CTest.
- Expansión dinámica y copia texto puro exacta.
- Rechazo de contratos ambiguos o fuera de límites.
- Sustitución numérica FP16 de filas.
- Orden del grafo `embedding → escala → scatter visual → identidad/contexto PLE`.
- Forward textual anterior idéntico byte a byte.
- Prefill y tres pasos greedy contra referencia externa FP16.

Los logits y volcados grandes permanecen en `/tmp/gemma4_v5`; no se versionan.

**V5 cerrada.** V6 puede añadir el camino de producto RGB → preprocesado →
torre visual → registro de la salida → prefill, además de medir VRAM y tiempos.
