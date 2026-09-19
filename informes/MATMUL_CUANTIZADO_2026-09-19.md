# Multiplicación cuantizada: descompresión HQ4.2/HQ5.2 vectorizada

2026-09-19. Base `93afc47`, que ya incorpora atención local HD256 compacta.
RTX 4070 Ti 12 GB, sm_89, CUDA 13.1, Release y `--use_fast_math`.
Modelo real `gemma4_12b_hq42_hq52.hnf`; mismos pesos, sin reconversión.

## Problema y cambio

El perfil de arquitectura previo atribuía un 19,1 % de la suma de tiempos de
kernels de prefill a descuantización y un 17,3 % a GEMM. Son cifras históricas,
anteriores a las optimizaciones de atención, no porcentajes recalculados aquí.

Para M >= 9, HQ4.2/HQ5.2 expande los pesos a un buffer FP16 y llama a cuBLAS.
Cada hilo descodifica ocho valores contiguos, pero los escribía mediante ocho
instrucciones escalares: en cada instrucción, las lanes escriben con stride de
16 bytes. Se conservan los cálculos y redondeos a FP16 y se empaquetan los ocho
resultados en una escritura de 128 bits. El ensamblador sm_89 confirma ocho
`STG.E.U16` frente a una `STG.E.128`.

Solo se selecciona esta variante cuando K es múltiplo de ocho: todos los grupos
están completos y cada destino está alineado a 16 bytes. El resto conserva las
escrituras escalares con sus comprobaciones de límites. El buffer temporal ya
existía: no aumenta su reserva ni se guardan nuevas copias permanentes de pesos.

No se modifican GEMV, el umbral M=9, cuBLAS ni la cuantización. HQ4.1/HQ5.1
conservan sus kernels. El paso 2 del plan queda **parcialmente avanzado**:
seguimos materializando la matriz FP16 completa, todavía no hay GEMM fusionada
con descuantización por tiles.

## Resultado con el 12B completo

Medianas de tres pares AB/BA/AB, un calentamiento y una muestra por longitud y
proceso. Mismo autotune, grafos y atención optimizada en ambos binarios; salida
greedy de 128 tokens. Comparación contra la base inmediatamente anterior, no
contra un motor sin las mejoras de atención.

| Tokens de entrada | Prefill anterior (ms) | Prefill nuevo (ms) | Reducción |
|---|---:|---:|---:|
| 41 | 180.16 | 127.42 | 29.27 % |
| 2089 | 1825.84 | 1584.32 | 13.23 % |
| 6185 | 6861.19 | 6229.65 | 9.20 % |

Las seis respuestas medidas por longitud coinciden byte a byte. El benchmark
no incluye carga del modelo, HTTP, herramientas del agente ni renderizado.
La GPU comparte escritorio y no se fijaron relojes. Las cifras de decode se
conservan en el JSON, pero no se atribuyen mejoras de generación a este cambio:
M=1 sigue por la misma ruta GEMV.

En el microbenchmark, K=15360/N=4096 reduce aproximadamente un 29 % en HQ4.2 y
un 31 % en HQ5.2 el tiempo del kernel de descompresión. Matrices que caben en
cache muestran ganancias mayores; no extrapolarlas a todo el modelo. Los
casos de pocos microsegundos están especialmente afectados por el lanzamiento.

## Corrección y alcance

- Compilación completa y CTest 25/25.
- 20 formas sintéticas de descompresión HQ4.2/HQ5.2 comparadas bit a bit entre
  variante escalar y vectorizada; 3 pares de tiempos por forma.
- `test_hqs_symmetric`: expansión CPU seguida del mismo GEMM como referencia,
  11 anchos por tipo y dos modos (directo/CUDA Graph): 44 comparaciones exactas.
  Incluye K=7/8, 255/256, 2040/2048/2056, 3840 y 15360, N=7, M=9;
  además conserva las comprobaciones previas de GEMV y embeddings.
- Sesiones reales 12B: 12 comparaciones de texto exactas, aislamiento, anillo,
  reutilización, reset, cancelación y destrucción.
- Prueba NDJSON de cancelación/progreso y reutilización pasada.
- HNF sin cambios y agente habitual restaurado con el runtime recompilado.

No se recertifica E4B/E2B/Qwen con pesos en esta campaña. El beneficio depende
del tipo HQ4.2/HQ5.2 de cada matriz, no únicamente del nombre del modelo.

## Evidencias y continuación

[Reproducción](matmul-cuantizado-2026-09-19/README.md),
[resultados completos](matmul-cuantizado-2026-09-19/resumen.json),
[CTest](matmul-cuantizado-2026-09-19/ctest.log),
[microbenchmark](matmul-cuantizado-2026-09-19/microbench.log) y
[escrituras en SASS](matmul-cuantizado-2026-09-19/stores.json).

Pendiente: evaluar la descuantización integrada con GEMM por tiles y el umbral
por forma. Para acelerar la emisión de tokens, perfilar GEMV cuantizado por
forma; esta optimización del prefill no resuelve ese frente.
