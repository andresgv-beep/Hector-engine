# Héctor: KV válido y grafos CUDA en InferenceSession

Implementación local del 19 de septiembre de 2026, sobre `fdc99e5`, posterior a
la [auditoría de arquitectura](ARQUITECTURA_RENDIMIENTO_2026-09-19.md).
GPU: RTX 4070 Ti de 12 GB. Modelo real: `gemma4_12b_hq42_hq52.hnf`, el mismo
HNF HQ4.2K/HQ5.2K de la auditoría, sin reconversión ni cambios de pesos.

Continuación: [optimización de atención de decode](OPTIMIZACION_ATENCION_2026-09-19.md).
El benchmark de este hito desactiva explícitamente esa optimización posterior
para seguir aislando la ganancia de los grafos.

## Problema y comportamiento resultante

`InferenceSession` intentaba capturar CUDA Graphs sobre el stream por defecto,
que no admite esa captura. Reintentaba el fallo por token y lanzaba los kernels
por separado. Además, el grafo pertenece al Engine compartido: habilitarlo sin
gestionar su propietario habría permitido reutilizar punteros del KV de otro
chat. Las vistas de scratch también cambian de forma entre prefill y decode.

Ahora Model tiene un stream CUDA propio, que vive más que Engine y sus recursos.
Cada turno invalida el grafo compartido y reconstruye los comandos/vistas de
decode. El primer paso se ejecuta normalmente para calentar las rutas; el
siguiente captura y el resto reutiliza el grafo. Si no se puede capturar, se
ejecuta normalmente y no se reintenta hasta el siguiente turno. Destruir o
reiniciar una sesión invalida el grafo; destruirla elimina sus vistas KV del
registro. El mutex del modelo sigue serializando los turnos.

La reutilización de prefijos antes solo comprobaba coincidencias de tokens.
Un historial editado podía retroceder a posiciones cuyo KV local ya había sido
sobrescrito por el anillo. Ahora la caché registra hasta dónde se ha escrito,
incluyendo forwards fallidos, y comprueba que todas las ventanas necesarias
siguen residentes antes de retroceder. Si no están, se reconstruye el prompt
completo. Si un error impide restaurar la ventana inicial, se vacía el KV y el
llamante debe reenviar el historial; la posición resultante figura en stats.
Los errores de ejecución visual también vacían el KV por precaución ante
escrituras parciales.

## Archivos afectados

| Archivo | Cambio |
|---|---|
| `src/inference_session.cpp` | Stream propietario, captura por turno, reconstrucción de vistas, limpieza y comprobación de prefijos |
| `src/inference_session.hpp` | `Model::Config::use_cuda_graphs` para control A/B, contadores por turno y contrato de recuperación |
| `src/gemma4_kv_cache.hpp` | Límite de escritura física y validación de retrocesos del anillo |
| `src/engine.cpp` | Diagnóstico del fallo inicial de captura |
| `tests/test_gemma4_kv.cu` | Casos límite del anillo y comparación de logits reales con/sin grafo |
| `tests/test_inference_session_gemma4.cpp` | Regresión con pesos reales para sesiones intercaladas, prefijos y cancelación |
| `CMakeLists.txt` | Ejecutable de regresión con modelo explícito |

No se han cambiado kernels de atención, cuantización, HNF ni Hexos. La ruta
`helios_formatted`, que usa InferenceSession, recibe la mejora al ejecutar el
binario recompilado. Los contadores añadidos son internos a `TurnStats`; este
parche no modifica el protocolo del puente ni el endpoint de telemetría.

## Validación

- Compilación completa en Release.
- 23 pruebas de CTest.
- Prueba real de KV: comparación prefill/decode existente y cinco pasos con
  captura/replay frente a ejecución normal, comparando todos los logits FP16
  bit a bit en cada paso.
- Prueba real de InferenceSession: 12 textos idénticos entre modos con y sin
  grafos; sesiones de 4K, 1K y 512 sobre los mismos pesos; destrucción, reset,
  cancelación y reintento; edición del prefijo antiguo tras superar el anillo;
  edición de cola que conserva el prefijo residente.
- En la edición antigua se reutilizan cero tokens y el resultado coincide con
  una sesión nueva. En la edición de cola se reutilizan 2.061 tokens. Repetir
  el prompt largo residente requiere solo un token de prefill.
- Sin fallos de captura en las pruebas reales. Las ejecuciones con grafos
  verifican que hubo captura y replay, no solo que devolvieron una respuesta.
- Prueba del binario `helios_formatted` recompilado: principal → auxiliar →
  principal, tres respuestas correctas a nivel de protocolo. La respuesta
  principal se conserva byte a byte; su repetición procesa un token y reutiliza
  22. El proceso termina correctamente al cerrar la entrada.

Los logs y el benchmark reproducible están en
[`optimizacion-kv-grafos-2026-09-19/`](optimizacion-kv-grafos-2026-09-19/).

## Medición

Mismo ejecutable y HNF, `use_cuda_graphs=false/true`, contexto reservado de
16.384 y 128 tokens de salida greedy. Una muestra de calentamiento y cinco
muestras medidas por modo y longitud; cada muestra reinicia la sesión.
La tabla muestra medianas; incluye captura y primer decode normal en el tiempo
de generación, y excluye el prefill. Las pruebas se ejecutaron en serie.

| Tokens de prompt | Sin grafos (tok/s) | Con grafos (tok/s) | Mejora |
|---:|---:|---:|---:|
| 41 | 42,308 | 43,368 | +2,51 % |
| 2.089 | 36,539 | 37,676 | +3,11 % |
| 6.185 | 33,386 | 34,019 | +1,90 % |

Las 36 generaciones terminaron en 128 tokens, con la misma huella y longitud
de salida en cada contexto entre modos y repeticiones. En cada generación con
grafos hubo una captura, 126 replays y cero fallbacks. Las 12 comparaciones
de la prueba de sesiones contrastan además los textos completos directamente.

El prefill prácticamente no cambia: aproximadamente 176 ms, 2,24 s y 8,43 s.
El consumo comunicado por `cudaMemGetInfo` es global de la GPU, incluido el
escritorio: no debe interpretarse como memoria exclusiva del proceso.
`gpu.csv` registra frecuencia, temperatura, potencia y uso durante la prueba.
En 2.089 tokens hubo más variación con grafos (37,276–38,069 tok/s), por lo que
la mediana no debe presentarse como una garantía para cualquier sesión.

Esta comparación aísla el efecto de activar grafos sobre el código corregido;
no es una comparación binaria completa contra el commit anterior. La estimación
exploratoria del 5–9 % de la auditoría, con solo dos muestras y otro control,
no se adopta como cifra del parche. La medición repetida sostiene aquí un
beneficio del **1,9–3,1 %**. La protección del anillo es un arreglo de
corrección independiente de esa ganancia.

[`resumen.json`](optimizacion-kv-grafos-2026-09-19/resumen.json) incluye rangos,
tiempos y huellas. `resumir.py` comprueba que los logs estén completos, que
coincidan las salidas y que los contadores acrediten captura/replay antes de
calcular el resumen.

## Reproducir y continuar

Desde la raíz del repositorio, después de compilar:

```bash
ctest --test-dir build --output-on-failure
build/test_gemma4_kv /ruta/modelo.hnf
build/test_inference_session_gemma4 /ruta/modelo.hnf
python3 informes/optimizacion-kv-grafos-2026-09-19/probar_puente.py /ruta/modelo.hnf
bash informes/optimizacion-kv-grafos-2026-09-19/reproducir.sh /ruta/modelo.hnf 0
bash informes/optimizacion-kv-grafos-2026-09-19/reproducir.sh /ruta/modelo.hnf 1
```

Ejecutar las mediciones GPU en serie. La prueba con modelo no forma parte de
CTest: requiere un HNF real explícito y no se omite silenciosamente. Para
desactivar grafos en un consumidor de la API, usar `use_cuda_graphs=false`.

La siguiente optimización a medir es la atención: sigue creciendo con el
contexto y domina el prefill largo según la auditoría. Este parche reduce el
coste de despachar el decode y corrige la validez del KV; no elimina ese coste
matemático. Una futura atención optimizada debe compararse contra esta versión
con las mismas entradas y logits de referencia.

Límites: validación con este 12B, esta GPU y texto greedy. No es una evaluación
de calidad general ni certifica otras arquitecturas, visión, muestreo aleatorio
o concurrencia entre modelos distintos. Los chats sobre un mismo modelo siguen
ejecutándose en serie. Los prefijos sobrescritos se reprocesan deliberadamente:
ahorrar ese prefill leyendo un anillo inválido no conserva la conversación.
