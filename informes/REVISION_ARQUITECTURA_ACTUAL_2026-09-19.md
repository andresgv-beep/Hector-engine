# Héctor: revisión de arquitectura después de las optimizaciones

Fecha: 2026-09-19. Código revisado: `ceadd0bee0fd0400e28e0793e143671268999902`.

Seguimiento posterior: [cancelación, progreso y KV auxiliar bajo demanda](CANCELACION_RECURSOS_2026-09-19.md). Los hallazgos y medidas de este documento describen la revisión base; consultar ese seguimiento para distinguir lo ya corregido de lo pendiente.

Esta revisión añade instrumentación y documentación; **no cambia el motor de producción**. Actualiza las prioridades del [primer análisis](ARQUITECTURA_RENDIMIENTO_2026-09-19.md), que describía una revisión anterior. Las optimizaciones de KV, CUDA Graphs y atención ya incorporadas se documentan por separado en el README.

## Conclusión

El motor tiene una base aprovechable: pesos compartidos entre sesiones, KV separado, atención heterogénea, reutilización de prefijos y decode con CUDA Graphs. No hace falta sustituirlo para seguir mejorando. Hay que separar tres objetivos: velocidad de cálculo, memoria disponible y capacidad de responder/cancelar durante un turno.

El mayor frente numérico pendiente es la atención de **prefill**: ocupa el 61,1 % de los tiempos de kernels del 12B en la prueba de entrada larga. Durante **decode**, las multiplicaciones cuantizadas ocupan el 71,4 % y la atención el 20,9 %. Son problemas diferentes; acelerar prefill no equivale a aumentar en la misma proporción los tokens/s de generación.

Antes de otra optimización numérica, corregiría la cancelación tardía y el contrato de entrega del puente, y reservaría algunos recursos solo cuando se necesiten. Son mejoras de uso y capacidad que no exigen cambiar la cuantización ni el resultado matemático del modelo.

## Arquitectura actual

```text
HNF → HnfLoader: configuración, tokenizer y pesos
    → Model: Engine + GraphBuilder + scratch + stream CUDA + adaptador multimodal
        → InferenceSession principal: KV + prefijo + sampler + comandos de decode
        → InferenceSession auxiliar: KV propio, mismos pesos

Entrada → plantilla/tokenización → comparación de prefijo
        → prefill por tandas de 512 → decode con CUDA Graph → callback de texto
```

- `Model` comparte el trabajo temporal y el grafo CUDA. `run_turn` bloquea un mutex durante todo el turno: dos sesiones del mismo modelo se ejecutan en serie. Es una protección de coherencia, no continuous batching.
- Las sesiones conservan KV y tokens del prefijo independientes. La invalidación del grafo al cambiar de turno/scratch evita reutilizar punteros de otra sesión. No debe eliminarse para ahorrar una recaptura sin resolver antes la propiedad de esos recursos.
- Para HQ4.2/HQ5.2, el prefill con `M >= 9` descomprime pesos a FP16 y llama a cuBLAS. Con `M=2…8` repite GEMV. El umbral es fijo, aunque el punto de equilibrio depende de la matriz.
- La atención de prefill actual mejora accesos y ocupación, pero sigue sin reutilizar un tile K/V entre varias consultas mediante un kernel de atención por bloques.
- `helios_formatted` y `helios_runtime` son dos contratos de acceso distintos al mismo núcleo. El primero acepta prompts preformateados y reutilización de prefijos; el segundo ya transmite `text_delta` y recibe cancelaciones mientras genera. Conviene unificar capacidades antes de crear otro backend.

Referencias: [sesión y modelo](../src/inference_session.cpp), [grafo](../src/graph_builder.cpp), [GEMV/GEMM cuantizado](../kernels/matmul_hqs_compact.cu), [puente preformateado](../tools/helios_formatted.cpp), [runtime NDJSON](../tools/helios_runtime.cpp).

## Evidencia nueva

### 1. Perfil de GPU del 12B

RTX 4070 Ti de 12 GB, 60 SM; biblioteca Release con CUDA 13.1, Nsight Systems 2024.5.1. Modelo `gemma4_12b_hq42_hq52.hnf`, SHA-256 `429785268c93f544174e9e2deb90a93a8c9df6c63a7c006ffdf5076d2a0ce765`. Contexto configurado a 16.384, entrada efectiva de 6.185 tokens, salida de 128, temperatura cero, una ejecución de calentamiento antes de la capturada. Las dos fases se capturaron en procesos separados.

| Grupo | Prefill: suma de kernels | Porcentaje | Decode: suma de kernels | Porcentaje |
|---|---:|---:|---:|---:|
| Atención, incluida reducción de particiones | 4.683,13 ms | 61,06 % | 765,01 ms | 20,95 % |
| Descuantización separada | 1.467,09 ms | 19,13 % | — | — |
| GEMM | 1.323,12 ms | 17,25 % | — | — |
| GEMV cuantizado | 22,86 ms | 0,30 % | 2.607,15 ms | 71,38 % |
| RMSNorm | 69,45 ms | 0,91 % | 209,69 ms | 5,74 % |
| Resto | 103,70 ms | 1,35 % | 70,54 ms | 1,93 % |
| Total | 7.669,34 ms | 100 % | 3.652,39 ms | 100 % |

En decode son aproximadamente 20,37 ms de GEMV y 5,98 ms de atención por paso. Esto justifica investigar atención para contexto largo, pero no presentarla como el único límite de generación.

**Límites de interpretación:** son sumas de duraciones de kernels bajo instrumentación, no latencias de usuario ni un benchmark A/B. GPU compartida con escritorio, frecuencias sin fijar. Las esperas de API se solapan con ejecución de GPU: no deben sumarse a la tabla. `cudaMemcpyAsync` acumula mucha espera en decode; eso no significa que copiar cuatro bytes tarde por ancho de banda varios segundos. Tampoco deben usarse los tiempos de pared de los logs con `cudaProfilerStart/Stop` como TTFT normal.

Datos: [prefill CSV](revision-arquitectura-2026-09-19/profile-prefill.csv), [decode CSV](revision-arquitectura-2026-09-19/profile-decode.csv), [resumen calculado](revision-arquitectura-2026-09-19/profile-summary.json).

### 2. Cancelación: se procesa la entrada aunque ya esté cancelada

En `src/inference_session.cpp`, `forward_batch` (línea 346) recorre todas las tandas sin comprobar cancelación. `run_turn` adquiere el mutex en la línea 608 y comprueba `cancel_flag` dentro del bucle de decode, línea 789.

La prueba pone el flag a `true` **antes de llamar a `run_turn`**. Aun así, el 12B procesa 6.185 tokens, llena ese KV y devuelve `cancelled` después de 7.740 y 7.767 ms en dos ejecuciones. No genera tokens de salida. Es una reproducción de cancelación tardía, no una conjetura sobre la interfaz.

Propuesta: comprobar antes del prefill y entre tandas; emitir progreso por tanda; permitir abandonar una espera de turno. Una tanda CUDA ya lanzada puede tener que terminar. Al cancelar, el prefijo registrado y el KV válido deben corresponder exactamente a lo procesado; no basta con insertar un `return` en el bucle. Validar también reutilización posterior, imagen y política de cierre del turno.

Datos: [cancel-12b.log](revision-arquitectura-2026-09-19/cancel-12b.log).

### 3. Entrega de texto: el núcleo transmite, el puente acumula

`tools/helios_formatted.cpp` acumula todos los callbacks en `output` y solo después escribe cabecera y cuerpo completos. Su flag `stop` permanece a `false`. En esa ruta, el cliente no puede recibir los fragmentos ni cancelar mediante el protocolo actual.

En una prueba sin profiler, E4B produce el primer callback a **2.703 ms** y termina 128 tokens a **4.701 ms**. Una entrega acumulada retendría casi dos segundos ese primer texto disponible. Es una medida del callback nativo y una consecuencia del código del puente; no se ha medido aquí la ruta completa de Hexos/navegador.

`helios_runtime` ya dispone de lector de comandos y eventos `text_delta`/cancelación. Le faltan las opciones equivalentes de preformateado/reutilización y la paridad de condiciones de parada del puente. La siguiente auditoría de Hexos debe comprobar qué binario/protocolo usa realmente y adaptar ambos extremos juntos.

Añadir al contrato: fase actual, tiempo en cola, primer token, tokens procesados/reutilizados y motivo inequívoco de fin. Hoy el agotamiento de contexto durante generación se devuelve como `Stop`, mientras que el desbordamiento de la entrada devuelve `context_full`; esa diferencia dificulta explicar por qué se corta una respuesta.

Datos: [latencia-e4b.log](revision-arquitectura-2026-09-19/latencia-e4b.log). No comparar sus tiempos con otro día como prueba de regresión.

### 4. Memoria que puede reservarse bajo demanda

**E4B con visión, usado solo para texto.** `Model::load` elige scratch para 6.144 tokens por la presencia de metadatos de visión, y `attach` dimensiona los anillos locales con ese mismo máximo. No depende de si la petición lleva imagen.

| Recurso E4B, contexto 16.384 | Capacidad actual para 6.144 | Capacidad de texto para 512 | Diferencia |
|---|---:|---:|---:|
| Scratch proporcional al batch | 888 MiB | 74 MiB | 814 MiB |
| KV de la sesión principal | 516 MiB | 296 MiB | 220 MiB |
| Total de estas dos partidas | 1.404 MiB | 370 MiB | **1.034 MiB** |

Son **cálculos de layout**, contrastados con la configuración cargada, no una reducción implementada. Excluyen pesos de visión y asignaciones fijas. El scratch sigue la fórmula de `GraphBuilder`; KV suma por capa propietaria `slots × kv_heads × head_dim × 4 bytes`. E4B tiene 20 capas locales propietarias y cuatro globales; las restantes comparten KV.

Propuesta: recursos visuales ampliables al primer uso. Requiere invalidar grafos y migrar/reiniciar KV según un contrato explícito. Bajar el tamaño sin adaptar el camino visual rompería imágenes. El HNF 12B usado en esta auditoría no contiene esos metadatos visuales y **ya reserva scratch de texto**: este ahorro de 1.034 MiB no se le aplica.

**Sesión auxiliar.** El puente la adjunta siempre al arrancar con contexto 1.024. El aumento observado de VRAM al adjuntarla es **336 MiB en 12B** y **56 MiB en E4B**. Adjuntarla al primer trabajo auxiliar evita ese coste mientras no se use; una vez creada habrá que decidir si conservarla. No se debe compartir su KV con el chat principal.

### 5. Muestreo: salto de coste al superar top_k=64

`kernels/sampling.cu:316–323` usa un bloque paralelo hasta 64 candidatos y un kernel de **un solo hilo** por encima. Prueba sintética con 262.144 logits FP16 uniformes, temperatura 0,7, top_p 0,95; tres calentamientos y mediana de 11 muestras:

| Modo | Tiempo por selección |
|---|---:|
| Greedy, temperatura cero | 0,131 ms |
| top_k=16 | 0,223 ms |
| top_k=64 | 0,969 ms |
| top_k=65 | 13,451 ms |
| top_k=128 | 16,266 ms |

Es un microbenchmark, dependiente de distribución y frecuencias, no una medida de generación de un modelo real. **El puente preformateado usa greedy**, así que no explica su lentitud actual. Para muestreo aleatorio conviene una selección paralela por bloques y combinación, conservando desempates y distribución.

También hay una incoherencia de contrato: `SamplingConfig` documenta `top_k=0` como desactivado, pero el sampler estocástico lo sustituye por 50. Resolver la semántica antes de optimizar. No bajar top_k silenciosamente para aparentar más velocidad.

Datos: [sampling.log](revision-arquitectura-2026-09-19/sampling.log).

## Experimentos numéricos siguientes, por valor esperado

1. **Atención de prefill por tiles de consultas y K/V.** Reutilizar K/V entre consultas y estudiar operaciones matriciales con softmax estable. Empezar por una forma concreta del 12B, con kernel actual como referencia y fallback. Debe respetar ventana local, índice de anillo, atención global, GQA/MQA, head_dim heterogéneo y softcap. Un nuevo orden de reducción puede cambiar resultados: si no es bitexacto, necesita tolerancias y evaluación de calidad explícitas. Como orientación matemática, reducir a la mitad el 61 % de atención reduciría el tiempo total de kernels un 30,5 % si todo lo demás fuera constante; **no es una predicción de rendimiento**.
2. **Prefill cuantizado sin materializar la matriz FP16 completa.** La descuantización separada ocupa el 19,1 %. Investigar descuantización por tiles integrada con GEMM y calibrar el umbral por forma. No expandir todos los pesos permanentemente: consumiría precisamente la VRAM que necesitamos. Evaluar también entradas pequeñas, donde M=2…8 repite GEMV.
3. **Separar avance de KV, logits y selección.** Para 6.185 tokens el perfil muestra 13 lm_head y 13 argmax: 12 selecciones intermedias se descartan. Evitarlas ahorra trabajo y sincronizaciones, aunque lm_head+argmax son solo unos 24,5 ms en este prefill. También se selecciona un próximo token que se descarta al terminar ciertos recorridos de decode/cierre. Mantener la escritura del último token al KV. En muestreo aleatorio, eliminar sorteos cambia la secuencia RNG: definir compatibilidad de semilla.
4. **Decode GEMV y fusiones selectivas.** Es el mayor coste de generación. Usar perfil por forma y ancho de banda efectivo antes de tocar la cuantización. Revisar los experimentos previos de `tools/DECODE_FUSION_PLAN.md`: algunas fusiones cambiaron reducciones o no mejoraron el rendimiento. En las ocho capas globales 12B con K=V se repite la proyección inicial; podría calcularse una vez y copiar antes de normalización/RoPE divergentes. GELU+MUL requiere conservar el redondeo FP16 intermedio si se exige igualdad exacta.

Un plan de ejecución con referencias tipadas y recursos estables puede reducir búsquedas y facilitar grafos por sesión, pero no aparece como el principal coste de una generación larga. No priorizaría paged KV, continuous batching ni especulación antes de arreglar estos puntos y medir el caso de un único chat.

## Propiedad de recursos y errores

`kernels/matmul_cublas.cu` conserva handle cuBLAS y buffer de descuantización globales al proceso. El mutex por modelo no protege dos modelos diferentes ejecutados simultáneamente: **riesgo identificado en código**, no corrupción reproducida en esta auditoría. Conviene que un contexto de ejecución por modelo/dispositivo posea esos recursos antes de ampliar concurrencia.

`cleanup_cublas` no tiene llamadas en producción. En el proceso aislado del probe, después de destruir sesiones/modelo, llamarlo reduce memoria ocupada de 1.116,375 a 992,375 MiB: **124 MiB**. Es almacenamiento global que sobrevive al modelo, no evidencia de una fuga creciente por token ni de que los pesos no se liberen.

Además, el crecimiento del buffer libera el anterior antes de `cudaMalloc` y actualiza su capacidad sin comprobar el resultado. Hay estados CUDA/cuBLAS no propagados en ejecución/sincronización. Priorizar asignación transaccional y errores estructurados con fase; `cudaGetLastError` no sustituye comprobar el retorno de cuBLAS. No atribuir a esto los bloqueos históricos de la interfaz sin una traza que los conecte.

## Orden propuesto y criterio de aceptación

| Paso | Cambio | Comprobación necesaria |
|---|---|---|
| 1 | Cancelación entre tandas y estados de fin/errores explícitos | Cancelación previa sin prefill; cancelación intermedia; KV y siguiente turno correctos; contexto agotado distinguible |
| 2 | Auxiliar bajo demanda; después scratch/KV visual bajo demanda | Memoria antes/después; primer uso visual y vuelta a texto; recaptura de grafo segura |
| 3 | Auditar Hexos y completar un protocolo progresivo | Primer fragmento visible antes del fin; cancelación de extremo a extremo; prompt real y prefijo reutilizado medidos |
| 4 | Atención de prefill por tiles; después GEMM cuantizado | A/B alternado, mismos pesos/tune/entrada; casos cortos y largos; exactitud o calidad documentadas; fallback |
| 5 | Contrato top_k y selección paralela | Empates, distribución, semillas, k=0/64/65/128 y varios vocabularios |

Los selectores actuales de prefill acelerado cubren 12B y E4B. E2B y Qwen3-8B conservan su ruta de referencia. **No se han ejecutado pesos reales de E2B/Qwen en esta revisión**, ni pruebas de rendimiento de visión. Las mejoras de contrato y propiedad son generales; una mejora de kernel debe validarse por arquitectura.

La revisión previa documenta CTest 25/25 y comparaciones bitexactas de los cambios ya incorporados. No se ha repetido esa batería aquí porque el código de producción no se ha modificado; los probes compilados y ejecutados son la validación de este informe.

## Reproducción

[probe.cpp](revision-arquitectura-2026-09-19/probe.cpp) enlaza la biblioteca existente sin modificarla. [run.sh](revision-arquitectura-2026-09-19/run.sh) compila y permite ejecutar una prueba a la vez. Requiere `build/libhelios-engine.a` de la revisión indicada, CUDA compatible y el modelo local. Conserva `tune.cache` para estabilizar las elecciones GEMV. El modo `cancel` es deliberadamente una reproducción del comportamiento defectuoso.

```bash
bash informes/revision-arquitectura-2026-09-19/run.sh sampling
bash informes/revision-arquitectura-2026-09-19/run.sh metadata /ruta/modelo.hnf
bash informes/revision-arquitectura-2026-09-19/run.sh cancel /ruta/modelo12b.hnf
bash informes/revision-arquitectura-2026-09-19/run.sh latency /ruta/modeloE4B.hnf
bash informes/revision-arquitectura-2026-09-19/run.sh profile-prefill /ruta/modelo12b.hnf
bash informes/revision-arquitectura-2026-09-19/run.sh profile-decode /ruta/modelo12b.hnf
```

No ejecutar los experimentos GPU en paralelo ni simultáneamente con el agente. El script guarda resultados nuevos en un directorio temporal y copia allí el tuning; no sobrescribe los logs de esta auditoría. `AUDIT_REPEAT` cambia la longitud de la entrada; 384 es el valor usado. Los informes binarios grandes de Nsight permanecen fuera del repositorio.
