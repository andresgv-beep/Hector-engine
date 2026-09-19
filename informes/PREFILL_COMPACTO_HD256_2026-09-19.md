# Prefill local HD256: acumuladores compactos

2026-09-19. Base: `333b7e5`. RTX 4070 Ti de 12 GB, CUDA 13.1, sm_89,
Release y `--use_fast_math`. Mismos HNF 12B HQ42/HQ52 y E4B HQ62K de las
campañas anteriores; no se reconvirtieron pesos.

## Por qué este cambio

El perfil anterior situó atención en el 61,1 % de la suma de tiempos de kernels
para el prefill largo del 12B. El kernel coalescido ya mejorado seguía reservando
16 valores por lane (hasta HD512), también cuando solo necesitaba ocho (HD256).
Esto dejaba acumuladores, reescalados y escrituras compartidas innecesarios.

Se parametriza la cantidad de valores por hilo y se usa una entrada compacta
solo en H16/KVH8/HD256/ventana1024 (12B) y H8/KVH2/HD256/ventana512 (E4B).
Se conservan 16 particiones softmax, causalidad, anillo y orden de combinación.
Se mantiene head_dim dinámico y se hace explícito el redondeo de la multiplicación
de corrección antes de añadir V. No es atención por tiles ni usa nuevos Tensor Cores.

La memoria compartida dinámica baja de **32,125 a 16,125 KiB por bloque** y los
registros de **64 a 56 por hilo**, sin memoria local/stack en esta compilación.
No cambia la reserva persistente de pesos/KV ni se promete más capacidad de contexto.

Se descartó un prototipo que hacía head_dim completamente constante: aceleraba
un caso corto pero no pasó igualdad bit a bit. No se incorporó. El prototipo
conservador sí pasó las 53 formas originales contra el kernel coalescido previo;
la suite de producción ampliada pasa 63 formas contra la referencia original.

## Recorrido completo, medianas de tres pares

Ambos binarios tienen grafos CUDA, atención de decode optimizada y prefill
coalescido habilitados. **La referencia no desactiva la optimización anterior**.
Una ejecución de calentamiento y una medida por longitud/proceso; orden AB/BA/AB,
mismo cache de autotune. Entrada 41, 2089 y 6185 tokens, salida greedy 128.

| Modelo | Entrada (tokens) | Antes (ms) | Después (ms) | Reducción de prefill |
|---|---:|---:|---:|---:|
| 12b | 41 | 182.43 | 182.87 | -0.24 % |
| 12b | 2089 | 2148.67 | 1830.86 | 14.79 % |
| 12b | 6185 | 8025.03 | 6899.41 | 14.03 % |
| e4b | 41 | 60.75 | 60.74 | 0.01 % |
| e4b | 2089 | 735.77 | 653.60 | 11.17 % |
| e4b | 6185 | 2771.89 | 2501.92 | 9.74 % |

Las respuestas medidas son idénticas byte a byte en las seis ejecuciones por
longitud/modelo. Los casos cortos se interpretan como ruido, no como una mejora
acreditable. Decode no se optimiza en esta entrega; sus pequeñas diferencias
figuran en el JSON, sin atribuirlas al cambio. No se mide carga de pesos, HTTP,
latencia del agente ni renderizado. GPU compartida con escritorio, relojes sin
bloquear: no extrapolar porcentajes a otro equipo o modelo.

En el microbenchmark del prototipo, la atención local de tandas de 512 queries
ya llenas redujo aproximadamente 42 % en el 12B y 47–49 % en E4B. No son ganancias
del prefill completo. La variante HD512 del prototipo no se incorporó: la atención
global mantiene el kernel anterior.

## Validación

- Compilación completa y CTest **25/25**.
- `test_attention_prefill`: **63 casos**, igualdad GPU bit a bit, referencia CPU
  FP64 con error absoluto < 0,001, captura/replay y lote de 6144 queries.
  Incluye fronteras 1/16/17/511, anillo, logits concentrados y fallbacks HD256.
- Activaciones reales: **624 comparaciones 12B y 546 E4B**, todas exactas, capa
  por capa y tanda por tanda; las mismas entradas Q/K/V para ambas rutas.
- Sesiones: **12 escenarios por modelo**, comparando eager, grafos, decode
  distribuido y prefill optimizado. Aislamiento, edición/reutilización, reset,
  cancelación y destrucción, con el mismo texto.
- Regresión NDJSON de cancelación con 12B pasada.
- SASS de los kernels genérico y coalescido anterior: huellas idénticas antes y
  después. El fallback de HD512 y geometrías fuera de la selección no se reescribe.

La primera invocación de sesiones E4B usó por error la fixture corta: falló en
la suite eager, **con prefill optimizado desactivado**, porque 2072 tokens no
sobrescribían el anillo reservado para visión. Se repitió con `--long-ring`,
que usa más de 8k tokens y sí prueba esa condición. Se conserva el fallo inicial
como evidencia de la invocación incorrecta; no se modificó el test para ocultarlo.

No se recertifican E2B/Qwen con pesos en esta entrega: sus geometrías fuera de
la selección conservan la ruta anterior. Tampoco se declara una nueva certificación
de respuestas visuales reales a partir del ensayo sintético de 6144 queries.

## Integración y reproducción

`Model::Config::use_coalesced_prefill=true` selecciona la nueva variante para las
dos formas validadas. Ponerlo a false recupera la referencia anterior a toda la
optimización coalescida, no solo anterior a este cambio. El agente habitual se
restaura al terminar las pruebas con el runtime recompilado y el mismo workspace.

[Procedimiento y scripts](prefill-compact256-2026-09-19/README.md),
[datos completos](prefill-compact256-2026-09-19/resumen.json),
[CTest](prefill-compact256-2026-09-19/ctest.log),
[recursos](prefill-compact256-2026-09-19/resources.log) y
[huellas SASS de las referencias](prefill-compact256-2026-09-19/reference-sass.json).

## Lo que sigue pendiente

Atención global de prefill con reutilización K/V entre queries y costes de
GEMV/descuantización según el perfil. Este cambio reduce trabajo inútil en
atención local; no resuelve esos otros frentes ni el coste del prompt de Hexos.
