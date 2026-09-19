# Atención de decode distribuida: Gemma 4 12B

Trabajo posterior a [KV y grafos CUDA](OPTIMIZACION_KV_GRAFOS_2026-09-19.md),
sobre el mismo HNF `gemma4_12b_hq42_hq52.hnf`, sin reconversión.
GPU de validación: RTX 4070 Ti, 12 GB, 60 SM; compilación Release con CUDA 13.1.

**Estado:** integrada en InferenceSession y activada por defecto únicamente para
las geometrías y longitudes descritas abajo. En tres pares A/B con contexto de
6.185 tokens, la ganancia fue 0,70–3,72 %, mediana 2,99 %. Es una mejora modesta;
no elimina la caída con contexto ni acelera el prefill.

## Problema observado

El kernel de decode asignaba un bloque a cada cabeza de query. Las 16 cabezas
del 12B solo proporcionan 16 bloques para 60 SM. Cada bloque contiene **16 warps**
en el código actual; los comentarios históricos que hablaban de cuatro estaban
desactualizados. El coste de atención aumenta con el contexto, especialmente en
las ocho capas globales HD512, mientras las capas locales usan una ventana de
1.024 tokens.

## Implementación

Se conservan las mismas 16 particiones del intervalo KV, el cálculo half2,
los acumuladores FP32, el softmax online y el orden de combinación. En la nueva
variante cada warp es un bloque independiente: 256 bloques en la primera fase
para el 12B. Un segundo kernel combina los estados en el orden original.
Esto permite distribuir el trabajo entre más SM sin cambiar el resultado
aritmético de la reducción. Un avance circular del índice KV evita calcular
el módulo en cada posición.

El workspace temporal contiene máximos, sumas y acumuladores: **526.336 bytes**
para 16 cabezas, aproximadamente 514 KiB. Pertenece al scratch del modelo,
se comparte entre capas y turnos serializados y se libera con él. No contiene
historial, no duplica pesos y no se reserva por token. Su dirección permanece
estable durante la captura y replay de CUDA Graphs.

La selección se hace al construir los comandos de decode, a partir de 2.048
tokens de contexto, para estas geometrías Gemma: H16/KVH8/HD256/ventana1024 y
H16/KVH1/HD512/global. Se mantienen el kernel anterior para contexto corto y
otras geometrías, y el prefill existente. Si un turno comienza corto y cruza
el umbral, conserva su kernel hasta reconstruir los comandos en otro turno:
no se lee la longitud desde GPU ni se recaptura por token.

`Model::Config::use_split_attention` (por defecto `true`) permite comparar o desactivar la variante.
`EngineConfig::use_split_attention` permite seleccionarla en consumidores de
GraphBuilder; Engine conserva el valor desactivado por defecto. La integración
de InferenceSession transmite la opción de Model antes de crear Engine.

Archivos principales:

- `kernels/attention.cu`: variante de particiones y combinación FP32.
- `kernels/kernels.hpp`: declaración del launcher y tamaño del workspace.
- `kernels/register_kernels.cpp`: ejecución y validación del cuarto tensor.
- `src/graph_builder.cpp`: reserva, selección por geometría/longitud y liberación.
- `src/engine.hpp`, `src/inference_session.hpp/.cpp`: opción A/B.
- `tests/test_attention_split.cu`: referencia CUDA, referencia CPU FP64, anillo
  y replay con longitud variable.
- `tests/test_gemma4_kv.cu`: contrato de selección y vida del workspace.
- `tests/test_inference_session_gemma4.cpp`: comparación de sesiones con atención
  anterior/distribuida, además de ejecución normal/grafos.

## Resultados y validación

La tanda secuencial completa conservó **los 36 textos byte a byte** entre modos
y repeticiones. Sin embargo, mostró variación de rendimiento que impide usar
las medianas de contexto corto/medio como efecto causal del cambio:

| Prompt | Referencia (tok/s) | Distribuida (tok/s) | Observación |
|---:|---:|---:|---|
| 41 | 47,110 | 43,019 | Ambas usan el kernel anterior; control de variabilidad |
| 2.089 | 41,674 | 37,793 | El prefill también pasó de 1,97 a 2,26 s, sin cambiar su kernel |
| 6.185 | 34,321 | 34,593 | +0,79 % en esta tanda; ver contraste pareado abajo |

Se guardan todos los resultados, incluidos los desfavorables, en
[`resumen.json`](optimizacion-atencion-2026-09-19/resumen.json). Los logs de GPU
registran la variación de frecuencia, temperatura, potencia y uso. No se
interpreta el -8,68 % del control corto como una regresión causada por la
atención distribuida: ni siquiera se selecciona en ese caso. Tampoco se
presentan los tiempos aislados de atención como mejora equivalente del modelo.

El contraste adicional alterna el orden AB/BA/AB con contexto de 6.185 tokens,
una muestra de calentamiento y otra medida por proceso, con las mismas decisiones
GEMV persistidas en `tune.cache` y grafos CUDA activos en ambos modos:

| Pareja | Referencia (tok/s) | Distribuida (tok/s) | Ganancia |
|---:|---:|---:|---:|
| A/B | 33,605 | 34,611 | +2,99 % |
| B/A | 33,626 | 34,877 | +3,72 % |
| A/B | 34,343 | 34,585 | +0,70 % |

La mediana de las ganancias pareadas es **2,99 %**. La dispersión y el número
limitado de parejas impiden presentarla como una garantía. El prefill continúa
en torno a 8,4 s: no se ha modificado su atención. Las huellas de salida coinciden
en cada pareja. Datos completos en
[`pareado.json`](optimizacion-atencion-2026-09-19/pareado.json); frecuencias de
núcleo/memoria y estado de energía en `gpu-pareado.csv`.

- 24/24 pruebas CTest superadas.
- 72 casos sintéticos: comparación bit a bit con el kernel anterior y referencia
  independiente en CPU FP64, con error absoluto inferior a 0,001. Incluyen
  HD64/128/256/512, MHA/GQA/MQA, dos batches, scores concentrados, longitudes
  1–12.000, límites de ventana/anillo y un mismo grafo con longitud variable.
- 6.144 comparaciones de atención con activaciones reales del 12B (48 capas ×
  128 pasos tras el prompt de 6.185 tokens), todas idénticas bit a bit.
- 12 textos de regresión idénticos entre ejecución normal, grafos con atención
  anterior y grafos con atención distribuida. Incluye sesiones intercaladas,
  reset, destrucción, cancelación y edición del historial tras envolver el anillo.
- Contrato de selección comprobado a ambos lados del umbral de 2.048 tokens;
  workspace FP32 válido y liberado al destruir el scratch; prefill sin cambios.
- Build final y 24/24 pruebas repetidas tras activar el valor por defecto.
- Puente `helios_formatted` final: principal con 6.167 tokens → auxiliar corto →
  principal repetido. Misma respuesta principal byte a byte; el segundo turno
  principal reutiliza 6.166 tokens y procesa uno. El proceso termina sin errores.

Se descartó una variante que especializaba HD256/HD512 en compilación: aunque
superaba las pruebas sintéticas, con activaciones reales produjo un valor de
atención distinto en 0,000244141 y cambió una frase de la generación greedy.
La variante final mantiene la dimensión dinámica y conserva la salida bit a
bit en la prueba de 48 capas × 128 pasos sobre el mismo KV real. Los logs
`exploratorio-especializado-*` documentan el ensayo descartado y **no** son las
cifras finales de rendimiento ni evidencia de paridad.

## Reproducción

Desde la raíz del repositorio, con el build actualizado y sin otra inferencia
ocupando la GPU:

```bash
ctest --test-dir build --output-on-failure
build/test_attention_split --bench
build/test_attention_split_model /ruta/modelo.hnf
build/test_inference_session_gemma4 /ruta/modelo.hnf
python3 informes/optimizacion-atencion-2026-09-19/probar_puente.py /ruta/modelo.hnf
bash informes/optimizacion-atencion-2026-09-19/reproducir.sh /ruta/modelo.hnf 0
bash informes/optimizacion-atencion-2026-09-19/reproducir.sh /ruta/modelo.hnf 1
```

El benchmark activa CUDA Graphs en ambos modos, usa el mismo binario, el mismo
HNF y prompts de 41, 2.089 y 6.185 tokens. Genera 128 tokens greedy, descarta
una muestra de calentamiento y conserva cinco por longitud y modo. El tiempo
de decode incluye el primer paso normal y la captura; el prefill se mide aparte.
Los textos completos pueden guardarse con `ATTENTION_OUTPUT_DIR`.
`ATTENTION_REPEAT=384 ATTENTION_TRIALS=2` selecciona el caso largo con una
muestra de calentamiento y una medida para el contraste alternado. Usar el
mismo `tune.cache` en ambos modos evita cambiar además los kernels GEMV.

## Alcance

La optimización afecta a generación de texto en la ruta InferenceSession usada
por `helios_formatted`. No cambia los prompts de Hexos, el formato HNF, el
muestreo ni la atención de prefill. La prueba sintética con otras geometrías
no certifica otros modelos reales. No se han validado visión ni ejecución
concurrente entre modelos. El siguiente frente independiente es acelerar el
prefill mediante reutilización de bloques de queries/KV; la presente mejora
no elimina su coste.
