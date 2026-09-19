# Contrato operativo de helios_runtime (NDJSON)

Revisado el 2026-09-19 contra `tools/helios_runtime.cpp` e
`src/inference_session.cpp` en `b822e1b`. `ready.protocol` sigue siendo **2**;
los eventos/campos opt-in posteriores amplían el contrato inicial.

## Transporte y sesiones

Una petición JSON por línea en stdin, un evento JSON por línea en stdout;
stderr contiene diagnóstico. Cada petición del cliente lleva un `request_id`
único. El proceso no ofrece HTTP. Un `attachment` añade bytes RGB8 después de
su cabecera y un salto de línea; el escritor debe enviar el frame sin intercalar
otras peticiones.

`ready` anuncia arquitectura, `max_seq_len`, `multimodal` y `vision_adapter`.
La sesión predeterminada es `"0"`. Se pueden abrir otras sesiones sobre los
mismos pesos; solo una genera a la vez. Hay eventos de sesión sin `request_id`
(`session_opened`, `session_closed`, `session_paused`), distintos de los terminales.

| Petición | Campos adicionales | Terminal |
|---|---|---|
| `turn` | `session`, `messages`, `attachment_ids`, `generation` | `completed` o `error` |
| `session_open` | `session`, `max_seq_len` opcional | `result` o `error` |
| `session_close` | `session` (no se elimina `0`) | `result` o `error` |
| `reset`, `status` | `session` opcional | `result` o `error` |
| `attachment` | `kind: rgb8`, `width`, `height`, `stride`, `bytes` + payload | `result` o `error` |
| `attachment_drop` | `attachment_id` | `result`; eliminación idempotente |
| `cancel` | `target`: ID de turno activo | `result` o `error: not_active` |
| `shutdown` | — | `result` y cierre |

La capacidad al abrir una sesión debe ser un entero positivo, no mayor que la
del modelo; cambiarla en una sesión existente devuelve `session_exists`.
Los estados de sesión informan su capacidad real, no siempre la global.

## Turnos y prefijos

```json
{"type":"turn","request_id":"r1","session":"0","messages":[{"role":"user","content":"Hola"}],"generation":{"temperature":0,"max_visible_tokens":128,"max_thinking_tokens":0,"prefill_progress":true}}
```

Normalmente `messages` contiene mensajes nuevos y el motor aplica la plantilla.
Con `generation.preformatted=true` se exige un único mensaje cuyo contenido ya
es el prompt formateado. El cliente puede reenviar su prompt completo con
`reuse_prefix=true`; se reutiliza solo la parte válida que coincide con el KV.
`close_turn=false` evita añadir automáticamente el cierre. `stop_tokens` es una
lista de tokens especiales existentes; un token desconocido se rechaza.
Estas opciones permiten que Hexos aplique la plantilla oficial y su catálogo.

## Eventos y campos

Cuando se solicita `prefill_progress`, se emite `turn_started` después de activar
el ID y `prefill_progress` entre tandas, con `processed_tokens`, `total_tokens`
y `ms`. El total excluye prefijo reutilizado. `prefill` final precede al texto
cuando hubo procesamiento; una cancelación previa no inventa ese evento.

`text_delta.text` contiene fragmentos, no necesariamente un token cada uno.
**En modo preformateado puede contener marcadores y contenido de canales del
modelo.** El adaptador nativo de Hexos filtra pensamiento y sintaxis de herramientas
antes de publicar la vista provisional. No prometer que todo `text_delta` es
texto listo para mostrar por el mero nombre del evento.

`completed` incluye:

- `visible_text`: concatenación de los fragmentos emitidos por el runtime.
- `finish_reason`: `eos`, `max_tokens`, `stop`, `cancelled` o `context_full`.
- `usage`: `prefill_tokens`, `prefill_reused`, `generated_tokens`, `thinking_tokens`.
- `timings`: `prefill_ms`, `queue_ms`, `first_token_ms`, `decode_ms`, `tokens_per_second`.
  `first_token_ms=-1` si no hubo fragmento; incluye espera desde la llamada.
- `model_state`: `cache_position_before`, `cache_position`, `max_seq_len`.

Un error posterior a texto incluye también el parcial y métricas. Errores de
lector, como `not_active`, pueden usar estado cero porque no inspeccionan el KV.
Los clientes no deben inferir el contexto a partir del número de caracteres.

## Cancelación y KV

`cancel` apunta al turno activo, no a cualquier petición en cola. Si llega antes
se rechaza; un cliente puede volver a enviarlo al recibir `turn_started`.
La confirmación indica si se aplicó, y el turno termina por separado.

Se comprueba cancelación antes de tokenizar, al esperar el modelo, entre tandas
de prefill y antes/durante decode. No se interrumpe un kernel ya lanzado ni el
interior del adaptador visual. Durante prefill se retrocede al inicio efectivo;
si el anillo perdió esa ventana, se vacía el KV. Durante decode se conserva lo
generado. La reutilización de un prefijo también puede reducir posición.
Por eso es incorrecto exigir que el KV solo disminuya por `reset` o por un evento
`context_compacted`: **este runtime no emite `context_compacted`**.

## Adjuntos

RGB8 requiere dimensiones positivas, `stride >= width*3`, `bytes == stride*height`
y tamaño máximo de 300 MiB en esta frontera. Los límites HTTP/Python pueden ser
menores. El ID del adjunto es el `request_id` de su subida. Tras uso correcto se
consume; una cancelación sin generación permite reintentar. `attachment_drop`
permite descartarlo explícitamente. La capacidad visual depende del HNF y su
adaptador, no del nombre de archivo.

## Evidencia

[Cancelación y recursos](../informes/CANCELACION_RECURSOS_2026-09-19.md),
[test de protocolo real](../tests/test_runtime_cancellation.py) y
[test de sesiones](../tests/test_inference_session_gemma4.cpp).
Las fixtures iniciales de `hexos-core/tools/protocolo_test.py` describen un
esquema anterior; no certifican por sí solas estas extensiones.
