# Atención durante el prefill: Gemma 4 12B

Ampliación posterior: [validación y activación para E4B](OPTIMIZACION_PREFILL_E4B_2026-09-19.md).

Mejora posterior adicional: [acumuladores compactos HD256](PREFILL_COMPACTO_HD256_2026-09-19.md),
medida contra el kernel coalescido de esta campaña. Las cifras siguientes
corresponden a la primera optimización y se conservan como evidencia histórica.

Continuación de [la optimización de decode](OPTIMIZACION_ATENCION_2026-09-19.md).
Base: `2e4f2f581bc5ae318dfeb68be5e38135b8e6ab35`. GPU: RTX 4070 Ti de 12 GB,
60 SM; Release, CUDA 13.1, `--use_fast_math`, arquitectura nativa sm_89.
Modelo: el mismo `gemma4_12b_hq42_hq52.hnf`, sin reconversión.

**Resultado:** prefill completo un 6,73 % más corto con 2.089 tokens y un
7,89 % con 6.185 tokens, comparando medianas. Activado por defecto en
InferenceSession para las dos geometrías validadas; sin reserva adicional de VRAM.

## Cambio

El kernel anterior dedica un bloque de 512 hilos a cada query/cabeza. Divide
el intervalo KV en 16 segmentos, acumula en FP32 y combina en memoria compartida.
Esta variante conserva esos segmentos, la causalidad, el anillo y el orden
aritmético, pero cambia tres detalles de ejecución:

1. Transpone los acumuladores compartidos para que las lanes vecinas accedan
   a palabras consecutivas. El acceso anterior tenía un stride de 16 floats.
2. Avanza circularmente por KV con incremento y comparación, en lugar de
   calcular el módulo en cada posición.
3. Solicita al compilador dos bloques residentes por SM mediante
   `__launch_bounds__(512, 2)`. En esta compilación pasa de 80 a 64 registros
   por hilo, sin stack ni memoria local. Esto elimina el límite de un bloque
   por SM impuesto por los registros en la variante anterior.

No se reservan buffers adicionales ni cambia el HNF, la cuantización o el KV.
Cada bloque sigue usando 32.125 KiB de memoria compartida, como la referencia.
El kernel original conserva su punto de entrada y su código máquina SASS:
se comprobó contra el fichero del commit base con el mismo compilador y flags
([huellas](optimizacion-prefill-2026-09-19/referencia-sass.json)).

Para el 12B, la selección automática comprende H16/KVH8/HD256/ventana1024 y
H16/KVH1/HD512/global. La ampliación E4B añade las dos geometrías documentadas
en su informe; las demás usan el kernel anterior.
`Model::Config::use_coalesced_prefill` (por defecto `true`) controla la opción en InferenceSession;
`EngineConfig` permite el opt-in explícito para consumidores de GraphBuilder.
Poner la opción a `false` recupera la ruta anterior. El valor por defecto de
EngineConfig se mantiene en `false`.

La primera variante, sin limitar registros, mejoraba la atención local pero
empeoraba ligeramente la global. No es la que se ha seleccionado: la variante
final también mejora la global en las mediciones aisladas.

## Medición del recorrido completo

| Tokens de entrada | Referencia, mediana | Variante, mediana | Reducción de tiempo | Reducciones en cada par |
|---:|---:|---:|---:|---|
| 41 | 173,91 ms | 174,51 ms | −0,35 % | −0,45 / +0,31 / −1,41 % |
| 2.089 | 2.235,49 ms | 2.085,09 ms | 6,73 % | 7,40 / 6,13 / 6,84 % |
| 6.185 | 8.412,12 ms | 7.748,70 ms | 7,89 % | 8,50 / 7,86 / 8,05 % |

Con entrada corta no hay una mejora apreciable. Con 6.185 tokens el ahorro
entre medianas es de **663 ms**. No se interpreta la variación de decode
como una mejora: el kernel de generación no cambia en este trabajo.
Las medianas de decode son 34,65 y 34,75 tokens/s en el caso largo.
Las respuestas de 128 tokens son iguales byte a byte en las seis ejecuciones
de cada longitud. Datos completos: [resumen.json](optimizacion-prefill-2026-09-19/resumen.json).

En las micropruebas, con tanda de 512 queries y contexto ya lleno, la
atención local pasa de unos 8,01–8,12 ms a 5,65–5,71 ms: alrededor de un
29–30 % menos de tiempo de kernel. La atención global con 6.144 tokens
previos pasa de 47,51 a 44,55 ms, un 6,23 % menos.
Estas cifras aisladas no son los porcentajes de mejora de todo el prefill.

El harness llama a InferenceSession, con grafos CUDA y atención de decode
distribuida habilitados en ambos modos. Solo cambia `use_coalesced_prefill`.
Contexto máximo 16K; greedy; 128 tokens de salida; reinicio de sesión por ensayo.
Las entradas contienen 41, 2.089 y 6.185 tokens. Cada forma se calienta una vez
y después se mide una vez por proceso. Se ejecutan tres pares AB/BA/AB, con
las mismas decisiones de autotune GEMV. Los logs conservan también el calentamiento.

`prefill_ms` mide el procesamiento de la entrada dentro del motor; no incluye
arranque del proceso, carga del modelo ni la latencia de Python, red o interfaz.
El escritorio sigue usando la GPU y los relojes no están bloqueados. Por tanto,
los porcentajes describen estas pruebas y este equipo, no una garantía universal.

## Pruebas

**Validación final completada:** compilación completa correcta, CTest **25/25**,
comparación real **624/624** exacta y **12 casos de sesión** con texto idéntico
entre las cuatro configuraciones. Logs:
[CTest](optimizacion-prefill-2026-09-19/ctest.log),
[capas reales](optimizacion-prefill-2026-09-19/modelo-paridad-final.log),
[sesiones](optimizacion-prefill-2026-09-19/sesiones.log).
Los binarios, incluido `build/helios_formatted`, están recompilados con la opción
activada por defecto. [Huellas del HNF y artefactos](optimizacion-prefill-2026-09-19/artefactos.sha256).

- `test_attention_prefill`: 27 combinaciones originales de MHA/GQA/MQA, dimensiones
  64/96/128/256/512, entradas de 1 a 512 tokens, atención local/global, anillo
  antes y después de dar la vuelta, distribuciones suaves y muy concentradas.
  Comparación bit a bit GPU y referencia CPU FP64 en queries inicial/intermedia/final.
  Todos los elementos se comprueban finitos; error absoluto CPU inferior a 0,001.
  Los lanzamientos se prueban dentro de captura/replay CUDA Graph.
  La ampliación posterior para E4B eleva la suite a 53 casos.
- `test_attention_prefill_model`: ambos kernels reciben **las mismas activaciones**
  Q/K/V reales, capa por capa. Se comparan todos los elementos de las 48 capas
  y 13 tandas: **624 comparaciones bit a bit**, incluyendo el anillo local.
  El test exige que se haya comparado cada capa y cada tanda de prefill.
- `test_inference_session_gemma4` añade una cuarta suite con el nuevo prefill,
  para compararlo con la misma ruta de decode: sesiones auxiliares, reutilización
  de prefijo, edición antigua y reciente, cancelación, reset y destrucción.

La igualdad numérica y de respuestas se refiere a estos casos comprobados.
No se ha cambiado el orden de las operaciones para buscar una ganancia de velocidad.

## Reproducción y archivos

Tras configurar CMake, `cmake --build build -j 6` compila motor y pruebas.
`ctest --test-dir build --output-on-failure` ejecuta las pruebas sin pesos reales.
Los tests `test_attention_prefill_model` y `test_inference_session_gemma4`
reciben la ruta del HNF como único argumento y deben ejecutarse secuencialmente.
Para comparar rendimiento:

```bash
bash informes/optimizacion-prefill-2026-09-19/reproducir.sh /ruta/modelo.hnf
```

El script escribe logs y respuestas en su directorio. `tune.cache` corresponde
a esta GPU; otra GPU necesita un perfil compartido nuevo para ambos controles.

Cambios principales: `kernels/attention.cu`, declaraciones en `kernels.hpp`,
selección en `register_kernels.cpp`, opción en `engine.hpp` y
`inference_session.hpp/.cpp`, pruebas y registros en
[`optimizacion-prefill-2026-09-19`](optimizacion-prefill-2026-09-19/).

## Qué queda

Esto reduce el coste de preparar la entrada. La generación token a token sigue
usando la optimización anterior. La atención global aún recorre todo el historial.
Un salto mayor requeriría reutilizar bloques KV entre queries y evaluar una
implementación por tiles/Tensor Cores, con validación numérica propia.
La cantidad de prompt que introduce Hexos es otro coste, pendiente de su auditoría.
