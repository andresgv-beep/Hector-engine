# Cancelación, progreso y sesión auxiliar bajo demanda

2026-09-19. Cambios sobre `ceadd0b`, siguiendo la [revisión de arquitectura](REVISION_ARQUITECTURA_ACTUAL_2026-09-19.md). No se han cambiado kernels, pesos HNF, cuantización ni plantillas.

## Qué cambia

### Sesión de inferencia

`run_turn` puede terminar por cancelación antes de tokenizar, mientras espera el cerrojo del modelo, entre tandas de prefill de texto y antes de entrar en decode. La espera usa un mutex temporizado y consulta la bandera cada cinco milisegundos aproximadamente; no es una garantía de tiempo real del sistema operativo. Los buffers compartidos siguen protegidos: cancelar otra sesión no permite ejecutarlas simultáneamente.

La posición publicada para consultar una sesión es atómica. Una petición cancelada mientras espera devuelve esa posición sin leer el estado mutable del KV de otra ejecución. No invalida el grafo del turno activo.

Semántica del KV:

- Cancelación antes de empezar: cero tokens procesados; conversación intacta.
- Cancelación durante prefill: volver al inicio efectivo de ese prefill. Si el anillo sobrescribió la ventana necesaria, vaciar el KV y devolver posición cero para que el cliente reenvíe el historial. Con prefijo reutilizado, ese inicio puede ser posterior al origen de la conversación y anterior al final del turno previo.
- Entrada incompleta: no añadir tokens de cierre, aunque `close_turn=true`.
- Cancelación durante decode: conservar los tokens generados y la política anterior de `close_turn`.
- Visión: comprobación antes/después del adaptador, no dentro de la operación visual. No se promete cancelación entre tiles visuales.

El callback opcional `PrefillProgressCallback(processed, total, ms)` informa después de cada tanda de texto completada. El total excluye tokens reutilizados. El callback final `on_prefill` mantiene su función y precede al texto; si se cancela sin iniciar trabajo, no se emite un prefill ficticio. Los callbacks se ejecutan con el cerrojo adquirido: pueden señalar cancelación, pero no llamar de forma reentrante a sesiones del mismo modelo.

`FinishReason::ContextFull` distingue agotamiento de contexto durante generación de una parada normal. La entrada que no cabe sigue devolviendo el error `context_full`. `TurnStats` añade `queue_ms` y `first_token_ms` (desde la entrada a `run_turn`, incluyendo espera; `-1` si no hubo fragmento visible). No se cambia el mecanismo de muestreo ni se promete reproducibilidad estocástica tras cancelar: las comparaciones exactas de esta entrega son greedy.

Código: [inference_session.hpp](../src/inference_session.hpp), [inference_session.cpp](../src/inference_session.cpp).

### Runtime NDJSON

Una cancelación debe apuntar al `request_id` activo. Un objetivo inexistente/inactivo recibe `not_active` y no cancela otro turno. La publicación del turno y el reinicio de la bandera se hacen bajo el mismo cerrojo que la recepción de cancelaciones, evitando perder una petición por una carrera.

Nuevos campos opcionales dentro de `generation`:

```json
{
  "preformatted": true,
  "reuse_prefix": true,
  "close_turn": false,
  "stop_tokens": ["<|tool_response>"],
  "prefill_progress": true
}
```

Estos campos permiten trasladar las opciones del puente preformateado al runtime que ya transmite `text_delta`. `prefill_progress` está desactivado por defecto para no introducir un evento nuevo en clientes que no lo pidan. Ejemplo de evento:

```json
{"type":"prefill_progress","request_id":"r1","processed_tokens":512,"total_tokens":6158,"ms":349.4}
```

`completed.usage` añade `prefill_reused`; `completed.timings` añade `queue_ms` y `first_token_ms`. `finish_reason` puede ser `context_full`. Una cancelación sin generación permite reintentar los adjuntos sin volver a subir los píxeles. Cancelar peticiones aún no activas en la cola no está implementado: se rechaza explícitamente.

Código: [helios_runtime.cpp](../tools/helios_runtime.cpp).

### Puente preformateado: memoria auxiliar

`helios_formatted` adjunta la sesión auxiliar al primer pedido `*`, en vez de hacerlo antes de `READY`. Conserva su KV separado y lo reutiliza después. Si no consigue adjuntarla, devuelve una respuesta de error con la cabecera de siete campos del protocolo existente; mantiene la sesión principal.

Medición por proceso con `nvidia-smi`, tras calentar primero la sesión principal:

| Modelo, contexto principal 16.384 | Principal caliente | Tras primera auxiliar | Tras segunda auxiliar | Asignación aplazada |
|---|---:|---:|---:|---:|
| 12B HQ4.2/HQ5.2 | 8.550 MiB | 8.886 MiB | 8.886 MiB | **336 MiB** |
| E4B HQ6.2K | 7.760 MiB | 7.816 MiB | 7.816 MiB | **56 MiB** |

El ahorro existe mientras no se utilice la auxiliar. Después permanece reservada; no se afirma una reducción permanente del pico. El prefill del chat principal después de las llamadas auxiliares reutiliza su prefijo y procesa un solo token; la salida es exactamente la misma.

Código: [helios_formatted.cpp](../tools/helios_formatted.cpp). La cabecera y el cuerpo del protocolo antiguo se conservan. **Este puente todavía acumula la respuesta completa y no recibe una orden de cancelación.** La cancelación comprobada por protocolo es la de `helios_runtime`.

### Conexión posterior a Hexos, completada el mismo día

El adaptador nativo de Hexos ya consume `helios_runtime` y conecta sus eventos con
la interfaz React mediante instantáneas de `/api/status` cada 450 ms. El botón
Detener envía una cancelación dirigida al turno. Se probaron texto progresivo,
cancelación, conservación del parcial, siguiente turno y `note.list` con el 12B
real; también el botón desde el navegador. El arranque habitual de este equipo
usa ahora esa ruta. Detalles en `hexos-core/docs/PROGRESSIVE_NATIVE_RUNTIME.md`.

Para esta conexión se añadieron tres piezas al protocolo:

- `turn_started`, solo cuando se solicita progreso, permite volver a enviar una
  cancelación que llegó antes de activar el turno.
- `session_open.max_seq_len` fija una capacidad por sesión (entero positivo,
  limitado por el contexto del modelo). Reabrir con otra capacidad se rechaza;
  los estados posteriores publican la capacidad de esa sesión.
- `attachment_drop` libera una subida por identificador, también después de una
  cancelación sin generación. Es idempotente.

El E4B real reconoció una imagen azul, continuó con un turno textual y extrajo un
nombre en una sesión auxiliar de 1024 tokens. Se volvió a pasar la regresión
NDJSON con el 12B después de estas extensiones. No se modificaron kernels ni
pesos para conectar la interfaz.

## Validación y medidas

GPU RTX 4070 Ti 12 GB, CUDA 13.1, Release, mismas rutas HNF que en la auditoría. Pruebas GPU secuenciales, sin otra inferencia simultánea. No es un benchmark de aumento de tokens/s.

| Caso | 12B | E4B |
|---|---|---|
| Cancelación previa, sesión vacía | 0 tokens; 0,0032 ms de llamada | 0 tokens; 0,0027 ms |
| Cancelación previa, sesión poblada | Posición 52 intacta; 0 tokens | Posición 52 intacta; 0 tokens |
| Cancelar tras primera tanda incremental | 512 procesados; vuelve a posición 52 | 512 procesados; vuelve a posición 52 |
| Cancelar tras sobrescribir anillo | 8.192 procesados; KV vacío | 8.192 procesados; KV vacío |
| Última tanda / callback final | Sin salida ni tokens de cierre | Sin salida ni tokens de cierre |
| Espera detrás de otra sesión | Sale cancelada antes de liberar la activa | Igual |
| Retry tras cancelación | Texto greedy idéntico al control | Texto greedy idéntico al control |
| Contexto pequeño, generación larga | `context_full`, posición 124/128 | Igual |

Los tiempos de cancelación previa son observaciones de llamadas individuales, no percentiles ni una promesa de latencia. La mejora esencial es que no ejecutan prefill. Antes, el [probe de auditoría](revision-arquitectura-2026-09-19/cancel-12b.log) procesaba los 6.185 tokens incluso con el flag ya activado.

En la prueba del protocolo NDJSON con 12B, se rechazó un `cancel` con objetivo incorrecto y continuó el progreso. Después se canceló el turno correcto: **536 ms** desde la orden hasta el resultado, terminando la tanda ya en curso. Procesó 1.536 de 6.158 tokens y devolvió KV vacío, sin texto generado. En E4B la orden llegó entre tandas: terminó con 1.024 procesados. Esto no significa cancelación instantánea de kernels CUDA.

Pruebas añadidas:

- [test_inference_cancellation.cpp](../tests/test_inference_cancellation.cpp): los casos de la tabla, con KV real, callbacks y dos hilos para el caso de espera.
- [test_runtime_cancellation.py](../tests/test_runtime_cancellation.py): objetivo de cancelación, progreso, confirmación, orden prefill/texto/completed, métricas, preformateado y reutilización de prefijo.
- [test_formatted_lazy_aux.py](../tests/test_formatted_lazy_aux.py): memoria por PID, formato legado, asignación única y aislamiento del chat principal.

Compilación completa correcta. CTest: **25/25**, ejecución secuencial. La batería existente de sesiones también pasa: **12 escenarios por modelo** con igualdad exacta de texto en cuatro configuraciones (eager, CUDA Graphs, atención dividida y prefill optimizado); E4B usa la variante larga con entradas de 8.216–8.217 tokens. El protocolo NDJSON y la memoria auxiliar pasan con ambos modelos, incluida la validación de `close_turn=false` y rechazo de tokens de parada inválidos sin alterar el KV.

Logs y resultados de las pruebas con modelos reales: [directorio de evidencia](cancelacion-recursos-2026-09-19/). No hay validación con pesos reales de E2B/Qwen en esta entrega, ni una prueba de cancelación de imagen real. Las pruebas generales de multimodalidad de CTest sí se ejecutaron.

Reproducción desde la raíz del repositorio (sustituir rutas de modelos):

```bash
cmake --build build -j 6
ctest --test-dir build --output-on-failure -j 1
export HELIOS_HOME="$PWD/informes/revision-arquitectura-2026-09-19"
./build/test_inference_cancellation /ruta/modelo.hnf
python3 tests/test_runtime_cancellation.py build/helios_runtime /ruta/modelo.hnf
python3 tests/test_formatted_lazy_aux.py build/helios_formatted /ruta/modelo12b.hnf 336
python3 tests/test_formatted_lazy_aux.py build/helios_formatted /ruta/modeloE4B.hnf 56
./build/test_inference_session_gemma4 /ruta/modelo12b.hnf
./build/test_inference_session_gemma4 /ruta/modeloE4B.hnf --long-ring
```

## Punto de continuidad

Los binarios se han recompilado y probado. No se han reiniciado servicios ni cambiado Hexos. Su cliente `hexos/agent/native.py` sigue hablando el protocolo de longitud explícita: cambiar el binario por NDJSON sin adaptar el cliente rompería ese contrato. La migración debe conservar parsing incremental de canales/herramientas, adjuntos, separación de sesiones, límites y reutilización de prefijos.

Quedan para las siguientes entregas la reserva visual bajo demanda (el ahorro estimado de 1.034 MiB del E4B **no está aplicado**), la propagación completa de errores CUDA/cuBLAS y el experimento de atención por tiles. Esta entrega no se atribuye mejoras numéricas ni de compresión del modelo.
