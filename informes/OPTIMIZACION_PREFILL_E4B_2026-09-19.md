# Prefill de Gemma 4 E4B: validación y activación

Extensión del [kernel validado para el 12B](OPTIMIZACION_PREFILL_2026-09-19.md).
Modelo real: `gemma4_e4b_hq62k.hnf`, sin reconversión. RTX 4070 Ti de 12 GB,
CUDA 13.1, Release, sm_89, `--use_fast_math`.

**Resultado:** procesar 2.089 tokens tarda un **5,13 % menos**, y procesar
6.185 tokens un **5,90 % menos**, comparando medianas. La variante queda
activada por defecto en InferenceSession también para las geometrías del E4B.
No requiere reservas adicionales de VRAM.

## Qué se habilita

El HNF inspeccionado tiene 42 capas, 8 cabezas de query y 2 de KV. Las últimas
18 capas comparten KV con capas anteriores. Se añaden estas dos geometrías a
la selección de `ATTENTION_PREFILL_CACHED`:

- H8/KVH2/HD256, ventana local de 512 tokens.
- H8/KVH2/HD512, atención global.

El kernel CUDA es exactamente el mismo que se validó para el 12B: merge de
memoria compartida contiguo, avance circular del índice KV y 64 registros por
hilo en esta compilación. Solo se amplía la condición de selección.
Las geometrías del 12B siguen admitidas. E2B (H8/KVH1) y Qwen3 8B (H32)
permanecen en la implementación anterior.

`Model::Config::use_coalesced_prefill = false` permite volver a la referencia.
`EngineConfig` mantiene el opt-in desactivado por defecto para consumidores
directos de GraphBuilder. No se activa la atención de decode distribuida
para E4B: esa selección sigue limitada al 12B.

## Rendimiento del motor completo

| Tokens de entrada | Referencia, mediana | Variante, mediana | Reducción de tiempo | Reducciones en los tres pares |
|---:|---:|---:|---:|---|
| 41 | 51,509 ms | 51,198 ms | 0,60 % | 0,23 / 1,23 / 0,60 % |
| 2.089 | 660,273 ms | 626,409 ms | 5,13 % | 4,93 / 5,13 / 4,95 % |
| 6.185 | 2.506,035 ms | 2.358,136 ms | 5,90 % | 5,79 / 5,90 / 5,77 % |

Con 6.185 tokens se ahorran unos **148 ms** de prefill. La generación posterior
se mantiene prácticamente igual: medianas de **72,83 frente a 72,92 tokens/s**.
No se atribuye esa pequeña variación de decode al parche. La diferencia con
entrada corta también es pequeña y no justifica prometer una mejora perceptible.

Se usa el mismo harness InferenceSession que para el 12B: contexto máximo 16K,
128 tokens de salida greedy, grafos activos en ambos controles, tres pares
AB/BA/AB, un calentamiento y una muestra por longitud/proceso. Las decisiones
de autotune son idénticas en todos los controles. Las respuestas medidas
coinciden byte a byte en las seis ejecuciones de cada longitud.
Memoria global ocupada registrada por el benchmark: aproximadamente 8.445 MiB
en ambos modos, incluyendo el resto del sistema.

El escritorio comparte GPU y no se han bloqueado relojes. Son mediciones de
este equipo y este HNF, no un porcentaje universal. Se mide el prefill dentro
del motor, no la latencia de toda la interfaz ni la carga del modelo.
Datos: [resumen](optimizacion-prefill-e4b-2026-09-19/resumen.json),
[serie A/B](optimizacion-prefill-e4b-2026-09-19/pareado.log).

## Precisión y regresión

**Validación final completada:** compilación completa correcta, CTest **25/25**,
**546/546** comparaciones reales E4B y **624/624** del 12B idénticas bit a bit.
Los **12 casos de sesión** dan el mismo texto en las cuatro configuraciones,
incluyendo los prompts de **8.216–8.217 tokens**, reutilización de prefijo y
cancelación. `build/helios_formatted` está recompilado con la selección ampliada.
Evidencias: [CTest](optimizacion-prefill-e4b-2026-09-19/ctest.log),
[capas E4B](optimizacion-prefill-e4b-2026-09-19/modelo-paridad.log),
[regresión 12B](optimizacion-prefill-e4b-2026-09-19/regresion-12b.log),
[sesiones](optimizacion-prefill-e4b-2026-09-19/sesiones.log),
[HNF y binarios](optimizacion-prefill-e4b-2026-09-19/artefactos.sha256).

- **546 comparaciones bit a bit** con las mismas activaciones Q/K/V reales:
  42 capas × 13 tandas, hasta una entrada de 6.185 tokens. Se incluyen capas
  que comparten KV y lecturas con el anillo local ya envuelto.
- **53 casos sintéticos** en la prueba ampliada: los 27 previos, 24 con las
  geometrías E4B y dos tandas completas de 6.144 queries. Estas últimas cubren
  el tamaño que puede usar un llamador multimodal, también con caché previa.
  Igualdad GPU bit a bit, resultados finitos y comparación independiente con
  CPU FP64; error absoluto inferior a 0,001 en las queries verificadas.
- Regresión del 12B con el test generalizado: 624 salidas de capa/tanda
  comparadas contra la referencia.
- El test de sesiones admite `--long-ring`: contexto máximo 12K y prompt de
  unos 8K tokens para sobrepasar también el anillo reservado por modelos con
  metadatos visuales. Compara ejecución sin grafos, grafos, decode distribuido
  cuando corresponde, y prefill optimizado. Incluye sesiones auxiliares,
  destrucción, edición de prefijos antiguos y recientes, cancelación y reset.

Las pruebas reales de esta ampliación son de texto; no constituyen una nueva
certificación completa de inferencia con imágenes.

## Reproducir

Compilar con `cmake --build build -j 6` y ejecutar las pruebas secuencialmente.
La suite general está registrada en CTest. Pruebas con pesos reales:

```bash
./build/test_attention_prefill_model /ruta/gemma4_e4b_hq62k.hnf
./build/test_inference_session_gemma4 /ruta/gemma4_e4b_hq62k.hnf --long-ring
bash informes/optimizacion-prefill-2026-09-19/reproducir.sh \
  /ruta/gemma4_e4b_hq62k.hnf informes/optimizacion-prefill-e4b-2026-09-19
```

El último comando sobrescribe los logs de ese directorio. Usa el perfil
`tune.cache` guardado para esta GPU; otra GPU necesita un control compartido
nuevo para ambos modos. Los logs originales de 12B quedan en su directorio.
