# Estado actual e índice documental de Héctor

Actualizado: **2026-09-19**, con prefill compacto validado sobre `333b7e5`. Este archivo resume el
estado operativo. Los planes e informes fechados conservan sus resultados de
entonces; sus «falta», «siguiente paso» y cifras no son un backlog actual.

## Capacidades contrastadas

| Área | Estado y evidencia |
|---|---|
| Gemma 4 E2B/E4B texto | Implementado. [Campaña E2B](GEMMA4_PLAN.md); regresión E4B en [prefill](informes/OPTIMIZACION_PREFILL_E4B_2026-09-19.md) y [cancelación](informes/CANCELACION_RECURSOS_2026-09-19.md). |
| Gemma 4 12B unified texto | Implementado, KV heads por capa, K=V y PLE opcional. [Integración inicial](GEMMA4_12B_UNIFIED.md) y campañas de septiembre. |
| Visión E2B/E4B con torre | Runner y adaptador implementados; [campaña visual](GEMMA4_VISION_PLAN.md). E4B probado desde el adaptador Python con imagen y seguimiento textual el 19 de septiembre. |
| Visión 12B unified | Sigue pendiente: no hay runner encoder-free ni configuración/preprocesado completos. Un bloque visual convertido o el kernel `g4u_pos_add` no completan ese recorrido. |
| Qwen3 | Ruta genérica Qwen/Qwen2 con Q/K norm; evidencia histórica de Qwen3-4B y sesiones Qwen8B. No se repitió esa certificación en la última campaña 12B/E4B. |
| Phi / DeepSeek | Referencias y pruebas históricas; no se declaran recertificados por la batería reciente Gemma. |
| Falcon / LLaMA y otras familias | No elevar detección/configuración a soporte end-to-end certificado sin prueba específica con pesos. |
| Sesiones | `Model` comparte pesos; `InferenceSession` tiene KV y sampler propios. Ejecución serializada, no batching concurrente. |
| Prefill local HD256 | Acumuladores compactos integrados para las dos geometrías Gemma 12B/E4B validadas. Prefill largo 14–15 % / 10–11 % más corto frente al coalescido previo; decode sin cambios. [Pruebas y medidas](informes/PREFILL_COMPACTO_HD256_2026-09-19.md). |
| Prefijo y CUDA Graphs | Implementados con límites del anillo y reconstrucción cuando corresponde. [KV/grafos](informes/OPTIMIZACION_KV_GRAFOS_2026-09-19.md). |
| Progreso/cancelación | Implementados en NDJSON y conectados al agente/UI de Hexos. [Contrato](tools/RUNTIME_PROTOCOL.md). |
| HTTP | Pertenece a Hexos, no al ejecutable de inferencia. Héctor sí tiene CLI/runtime utilizables, no solo tests. |
| Audio/vídeo | Detectar bloques no equivale a ejecutarlos: adaptadores no disponibles. |

Para un HNF concreto, usar `build/helios_model_probe --json modelo.hnf`.
`runtime_ready` indica un adaptador registrado, no una nueva validación numérica
ni que cualquier variante con nombre parecido sea compatible.

## Cuantización: bytes por bloque de 256 valores

Fuente: [src/dtype.cpp](src/dtype.cpp).

| Tipo | Bytes/bloque | Bits efectivos/valor |
|---|---:|---:|
| HQ31K | 136 | 4,25 |
| HQ41K | 168 | 5,25 |
| HQ51K | 200 | 6,25 |
| HQ42K | 152 | 4,75 |
| HQ52K | 184 | 5,75 |
| HQ62K | 264 | 8,25 |

Incluyen metadatos del bloque; no incluyen otros tensores ni el contenedor.
HQ62K es lookup de embeddings, no matmul general. Un archivo con ese sufijo
puede mezclar tipos: no comparar su tamaño como si todos los pesos usaran HQ62K.

## Qué queda después del trabajo reciente

- Atención de prefill con reutilización de K/V entre consultas: los kernels
  actuales ya están optimizados, pero eso no completa una atención por tiles.
- GEMV cuantizado y costes de descuantización/GEMM, según el perfil real.
- Visión encoder-free del 12B y audio/vídeo: trabajo distinto del streaming.
- Multi-GPU y continuous batching: no implementados.
- Comprobar nuevas familias con pesos reales antes de generalizar resultados.

Cancelación temprana, progreso, sesión auxiliar diferida y conexión de la UI
**ya no son pendientes**. La revisión base `ceadd0b` precede a su implementación.
La cancelación conserva límites: espera a que termine el trabajo CUDA en curso.

## Cómo leer el resto de documentos

- **Contratos vigentes:** [runtime](tools/RUNTIME_PROTOCOL.md),
  [capacidades](tools/MODEL_CAPABILITIES.md), [adaptadores](tools/MULTIMODAL_ADAPTER.md),
  [pesos mapeados](tools/MAPPED_WEIGHT_PIPELINE.md).
- **Implementación reciente:** [prefill compacto HD256](informes/PREFILL_COMPACTO_HD256_2026-09-19.md), [cancelación y recursos](informes/CANCELACION_RECURSOS_2026-09-19.md),
  los informes `OPTIMIZACION_*_2026-09-19.md` enlazados en el README.
- **Diagnósticos de una base anterior:** [arquitectura inicial](informes/ARQUITECTURA_RENDIMIENTO_2026-09-19.md),
  [revisión posterior](informes/REVISION_ARQUITECTURA_ACTUAL_2026-09-19.md),
  [prueba local del 18](PRUEBA_12B_2026-09-18.md), [notas de julio](OPTIMIZATION_NOTES.md).
- **Campañas históricas:** `GEMMA4_PLAN.md`, `GEMMA4_VISION_PLAN.md`,
  `tools/GEMMA4_F*.md`, `tools/GEMMA4_VISION_V*.md`,
  `tools/HQS_DECODE_OPTIMIZATION_PLAN.md`, `tools/DECODE_FUSION_PLAN.md`,
  `tools/HQ31K_IMPLEMENTACION.md`, `tools/GEMV_BATCHED_KILLTEST.md`,
  `tools/quant_bench/README.md` e `informes/E4_SESIONES.md`.
  Se conservan errores observados y medidas de cada fase, sin tratarlos como
  defectos vigentes ni promesas generales de calidad.

## Desfases corregidos en esta revisión documental

El README anterior omitía Gemma y Qwen3, decía «sin streaming»/«solo tests»,
listaba arquitecturas CUDA predeterminadas distintas del CMake y describía
incorrectamente HQ41/HQ51 como bloques de 1K. Esos puntos se han contrastado
con código y corregido. Los planes antiguos llevan un aviso y este enlace;
se conservan mediciones originales. El contrato NDJSON documenta las extensiones
ya implementadas, sin presentar las fixtures iniciales como cobertura completa.

La revisión documental inicial (`333b7e5`) solo cambió Markdown. La ampliación
posterior de prefill HD256 sí recompila y mide el motor: CTest 25/25, 63 casos
sintéticos y regresiones con pesos reales 12B/E4B; véase su informe enlazado.

## Mantenimiento

Al cambiar una capacidad, actualizar este estado, su contrato y el README si
cambia el arranque. Un informe nuevo debe identificar revisión, modelo,
configuración y límites. No borrar evidencia antigua ni marcar como hecho lo
que solo existe en el conversor o en una propuesta.
