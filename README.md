# HELIOS Engine — Héctor

Motor de inferencia local C++/CUDA para modelos HNF. El agente, las herramientas,
el historial y la interfaz React viven en el repositorio `hexos-core`.

**Estado revisado el 2026-09-19; prefill compacto validado sobre la base `333b7e5`.**
Consulta [estado actual e índice documental](ESTADO_ACTUAL.md) para distinguir
capacidades implementadas, límites y resultados de campañas históricas.

## Qué funciona

- Carga HNF v9, tokenizer HTF, detección de configuración y construcción del grafo.
  Hay rutas específicas por arquitectura; no se admite cualquier transformer
  únicamente por reconocer nombres de tensores.
- Texto Gemma 4 E2B/E4B y 12B unified. Qwen3 utiliza la ruta Qwen con Q/K norm.
  La evidencia de modelos y sus límites se detalla en el [estado actual](ESTADO_ACTUAL.md).
- Visión Gemma 4 con torre (E2B/E4B), una imagen por turno, mediante adaptador
  persistente. La visión encoder-free del **12B unified sigue sin estar integrada**.
- Pesos compartidos entre sesiones, KV separado, anillo local Gemma 4,
  reutilización de prefijo textual y decode con CUDA Graphs.
- Runtime NDJSON con fragmentos `text_delta`, progreso opcional de prefill,
  cancelación dirigida, métricas y capacidad de contexto por sesión.
- Puente `helios_formatted` compatible con clientes antiguos, sin entrega
  progresiva ni orden de cancelación. La ruta nativa actual de Hexos usa NDJSON.

Atención local HD256 con acumuladores compactos: prefill largo del 12B un
14–15 % más corto y del E4B un 10–11 % en la campaña local, frente al kernel
coalescido previo. [Mediciones y límites](informes/PREFILL_COMPACTO_HD256_2026-09-19.md).

## Compilación

Linux, CMake >= 3.18, compilador C++17 y CUDA Toolkit compatible con la GPU.
La campaña local del 19 de septiembre usa CUDA 13.1 y RTX 4070 Ti.

Desde la raíz de este repositorio:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 6
ctest --test-dir build --output-on-failure -j 1
```

`CMakeLists.txt` fija actualmente `CMAKE_CUDA_ARCHITECTURES` a `native`, no a
una lista 75/86/89. Para un binario portable hay que ajustar esa asignación y
usar una versión de CMake/CUDA que soporte la selección deseada. Los binarios
con pesos reales requieren además un HNF; no todos forman parte de CTest.

## Entradas de uso

```bash
# Capacidades del HNF sin cargar pesos ni crear contexto CUDA
build/helios_model_probe --json /ruta/modelo.hnf

# Chat de terminal (ruta propia, no el agente Python)
build/helios_chat /ruta/modelo.hnf

# Servicio de inferencia por stdin/stdout NDJSON
build/helios_runtime --model /ruta/modelo.hnf --ctx 16384 --temp 0
```

El motor no sirve HTTP por sí mismo. Hexos ofrece servidores HTTP; su agente
Python conecta este runtime con la interfaz. No confundir `helios_chat`,
`helios_formatted`, `helios_runtime` y los dos backends HTTP de Hexos.

El [contrato NDJSON](tools/RUNTIME_PROTOCOL.md) describe peticiones, eventos,
KV y cancelación. El lanzador del agente está en `hexos-core/runtime/run-helios-native`.

## Cuantización y memoria

Los tipos compactos registrados incluyen HQ31K, HQ41K, HQ51K, HQ42K, HQ52K y
HQ62K. Las formas y bytes reales están en [dtype.cpp](src/dtype.cpp); el nombre
no indica el tamaño completo del HNF ni todos sus metadatos. HQ62K tiene lookup
de embeddings, **no un matmul general implementado**. HQ42K/HQ52K se usan en el
12B probado. El conversor produce los pesos; Héctor los consume.

El prefill cuantizado puede descuantizar a FP16 y usar cuBLAS; no todas las
rutas mantienen cada operación fusionada. El KV y el scratch se reservan según
arquitectura/capacidad; las sesiones auxiliares se crean bajo demanda en los
clientes documentados. Varias sesiones comparten pesos y recursos de trabajo,
pero ejecutan inferencia en serie.

## Evidencia reciente

- [Cancelación, progreso y conexión con Hexos](informes/CANCELACION_RECURSOS_2026-09-19.md).
- [KV y CUDA Graphs](informes/OPTIMIZACION_KV_GRAFOS_2026-09-19.md).
- [Atención de decode](informes/OPTIMIZACION_ATENCION_2026-09-19.md).
- [Atención de prefill](informes/OPTIMIZACION_PREFILL_2026-09-19.md) y
  [regresión E4B](informes/OPTIMIZACION_PREFILL_E4B_2026-09-19.md).
- [Perfil y siguientes prioridades](informes/REVISION_ARQUITECTURA_ACTUAL_2026-09-19.md).

Los tiempos pertenecen al modelo, GPU, contexto y revisión de cada informe.
No se extrapolan como rendimiento garantizado ni como certificación de todas
las familias.

## Límites vigentes

Sin multi-GPU/tensor parallelism ni continuous batching. Sin importación directa
de GGUF/safetensors: se requiere conversión a HNF. Audio y vídeo no tienen un
adaptador de inferencia operativo documentado. No hay soporte Windows certificado.
Cancelar no interrumpe un kernel CUDA a mitad de ejecución ni el interior del
preprocesado/encoder visual. Quedan optimizaciones numéricas de prefill y decode;
mostrar texto progresivo no aumenta por sí mismo los tokens/s.

## Código principal

- `src/inference_session.*`: modelo compartido, sesiones, KV y generación.
- `src/graph_builder.*`, `src/gemma4_kv_cache.hpp`: grafos y caché por arquitectura.
- `src/hnf_loader.*`, `src/htf_tokenizer.*`: pesos, metadatos y tokenizer.
- `src/model_capabilities.*`, `src/multimodal_adapter.*`: capacidades y adaptación visual.
- `kernels/`: operaciones CUDA, atención y multiplicaciones cuantizadas.
- `tools/helios_runtime.cpp`: frontera NDJSON.
- `tests/`, `informes/`: pruebas y evidencia fechada.

## Licencia

MIT.
