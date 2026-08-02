# Gemma 4 E2B Vision — V4 encoder FP16

> Cerrada el 2026-08-02. Esta fase ejecuta la torre visual completa y termina
> en los 280 embeddings de ancho 1536. Todavía no modifica el prompt ni el
> decoder de texto.

## Ruta independiente

`Gemma4VisionRunner` vive fuera del `GraphBuilder` textual y consume el bloque
visual ya cargado por `HnfLoader`. Conserva buffers reutilizables y exige el
orden explícito de las fronteras:

1. proyección de patches y posición XY aprendida;
2. capas visuales 0 a 15;
3. pooler espacial 3 × 3 en FP32;
4. RMSNorm sin peso y proyector 768→1536.

La entrada se transforma a `[-1,1]` antes de convertirla a FP16. El padding
`(-1,-1)` no recibe posición aprendida, no participa como clave de atención y
se elimina al agrupar los soft tokens. La geometría válida debe formar el
rectángulo XY canónico producido por V3.

Cada capa aplica los cuatro RMSNorm, los clamps aprendidos antes y después de
sus siete lineales, Q/K norm con peso, V norm sin peso, RoPE 2D, atención
bidireccional con escala 1,0, GeGLU y ambos residuales. Q/K/V se transponen a
layout por cabeza; los dos GEMM de atención usan cuBLAS, y el softmax almacena
FP16 pero reduce en FP32, igual que la ruta FP16 de upstream.

El pooler promedia en FP32, redondea ese promedio a FP16 y aplica
`sqrt(768)` en FP32. Los buffers no persistentes `std_bias=0` y `std_scale=1`
del checkpoint son identidad. La proyección final normaliza sin peso y usa el
tensor FP16 `vision.projector.weight`.

## Comparación contra V0

HNF combinado canónico (el test carga únicamente su bloque visual):

`/home/andres/Documentos/GitHub/helios_convert_v9.1/output/gemma4-e2b-it-multimodal.hnf`

Oráculo FP32:

`/home/andres/.cache/helios/gemma4_vision_oracle/fp32-b3a36037-run1`

| Frontera | NRMSE | Límite | Correlación | Mínimo |
|---|---:|---:|---:|---:|
| patch embedder | 0,000255030 | 0,005 | 0,999999968 | 0,99999 |
| capa 0 | 0,000445495 | 0,015 | 0,999999905 | 0,9998 |
| capa 15 | 0,005997680 | 0,030 | 0,999982024 | 0,9990 |
| pooler | 0,003254457 | 0,030 | 0,999994747 | 0,9990 |
| proyección | 0,004182350 | 0,030 | 0,999991278 | 0,9990 |

Todas las formas coinciden y todos los valores son finitos. El mismo test
pasó originalmente con el HNF solo visual y con el combinado de V1. La ruta
vigente anterior sustituye ambos artefactos operativos sin cambiar los pesos.

La tabla fue recertificada después de `d635d2b`, que sustituyó la acumulación
FP16 de `cublasHgemm` por acumulación FP32. Los valores anteriores permanecían
dentro de las barreras, pero los actuales se acercan más al oráculo.

## Memoria

La medición con `cudaMemGetInfo`, después de sincronizar la salida final, dio:

```text
pesos visuales cargados             337.641.472 bytes
runner + workspace cuBLAS           318.767.104 bytes
cota conservadora calculada en V2   368.040.960 bytes
```

El contexto cuBLAS es compartido con los matmul existentes de Héctor. Así no
se crea un segundo workspace al entrar en visión. La matriz de atención es
FP16 y se reutiliza en las 16 capas; no se reservan 16 copias.

## Pruebas

CTest incluye una prueba autocontenida de escala, posición XY y padding. La
prueba numérica pesada, que además carga y descarga el bloque real, se ejecuta
con:

```bash
GEMMA4_HNF=/home/andres/Documentos/GitHub/helios_convert_v9.1/output/gemma4-e2b-it-multimodal.hnf
./build/test_gemma4_vision_runner \
  "$GEMMA4_HNF" \
  /home/andres/.cache/helios/gemma4_vision_oracle/fp32-b3a36037-run1
```

Suite completa tras V4: **17/17**. El conversor no se modificó.

## Límite y siguiente fase

V4 certifica la torre hasta `[soft_tokens,1536]`, pero esos vectores aún no
entran en el decoder. V5 debe construir el prompt con el número dinámico de
`<image_soft_token>`, sustituir sus embeddings y respetar la doble entrada de
PLE: IDs con imagen reemplazada por PAD para identidad, embeddings visuales
para la proyección contextual. Solo entonces se compararán logits y pasos
greedy contra upstream.
