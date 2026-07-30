# Notas de optimización — Héctor v1

> Diagnóstico hecho el 2026-07-29. Plan: **medir primero, tocar después.**

## ✅ ESTADO (actualizado 2026-07-29 tarde)

- **Fase 0 COMPLETADA.** CUDA 13.1 instalado, motor compilado para sm_120,
  Qwen3-4B convertido y **generando texto coherente por primera vez**.
- **Baseline: 57.9-59.2 tok/s** (decode, HQ4K, RTX 5070 Laptop).
  Prefill: 123 ms / 6 tokens. Loop actual: rebuild+execute+sync por token
  (SIN graph replay — el techo de la Fase 1 sigue pendiente).
- **Soporte Qwen3 añadido** (era el bloqueo de hace un año):
  - Conversor: `head_dim` explícito del config (128, no hidden/heads=80),
    mapeo de `q_norm`/`k_norm` (antes: 72 tensores skipped), hint
    `use_qk_norm`. Archivo: `src/mapping/qwen2.rs` del conversor.
  - Motor: `ArchDescriptor.use_qk_norm` autodetectado por presencia de
    `attn.q_norm.weight`; RMSNorm por-head en Q/K antes de RoPE
    (`graph_builder.cpp`); kernel RMSNORM acepta `dim` explícito para
    normalizar sub-filas (`register_kernels.cpp`).
- Parche local a CUDA 13.1 por glibc 2.43 (rsqrt/rsqrtf noexcept):
  `/usr/local/cuda/.../crt/math_functions.h` (backup en `.bak`).
- El validador del conversor (`validate`) da falsos FATAL con hints válidos —
  está desactualizado, pendiente de arreglar.
- Siguiente paso: **Fase 1** (graph replay en el loop de decode).

## ✅ FASE 1 COMPLETADA (2026-07-29 noche) — y reevaluación del diagnóstico

Implementado el decode con CUDA Graph replay real:
- Kernels `_dp` por fin registrados (dispatch con flag `device_pos` por comando).
- GraphBuilder marca los comandos de decode (rope/kv_update/attention_cached).
- `test_generate_kv`: stream dedicado (el legacy no es capturable), build UNA vez,
  token 1 por `execute()` (calienta auto-tune GEMV, que sincroniza), captura en
  token 2, replay puro después. 615 comandos capturados OK.
- Correctitud verificada: secuencia greedy idéntica token a token vs pre-Fase 1.

**Resultado: 52-59 tok/s (antes 52-58). La aguja apenas se movió — otra vez.**

### Diagnóstico corregido

El overhead CPU existía pero estaba mayormente solapado con la GPU. La verdad
medida: con Qwen3-4B HQ4K (4.32 GB de pesos leídos por token):
- 59 tok/s ≈ 16.9 ms/token ≈ **260 GB/s efectivos**
- Techo teórico RTX 5070 Laptop (128-bit GDDR7): ~384 GB/s
- **Estamos al ~65-68% del ancho de banda. El decode es memory-bound de verdad.**
- Ojo: varianza entre runs 52↔59 tok/s = térmica del portátil. Benchmarkear
  enchufado y con el mismo estado térmico.

## ✅ FASE 2 PARCIAL (2026-07-29 noche): 57 → 70.4 tok/s (+23%)

Perfilado con `nsys --cuda-graph-trace=node` (régimen estacionario, no el
primer token con auto-tune). Hallazgos y cambios:

1. **lm_head cuantizado para modelos tied** (conversor `builder.rs`): con
   `tie_word_embeddings` el motor multiplicaba contra el embedding FP16 —
   778 MB/token. Ahora el conversor emite copia HQ5K (437 MB) como
   `lm_head.weight` y el motor la usa automáticamente. lm_head: 367 GB/s.
2. **gemv_hq5k era COMPUTE-bound, no memory-bound**: extraía pesos con shifts
   de 64 bits (multi-instrucción en GPU) y bounds-check por elemento.
   Arreglado con `__funnelshift_r` (1 instr) + camino rápido sin checks +
   staging del payload en shared. HQ5K: 167 → **273 GB/s**. (+13 tok/s)
3. gemv_hq4k: bounds-check hoisted + lecturas half2. Sin cambio medible —
   ya estaba a 345 GB/s, cerca del techo.
4. Los kernels pequeños (rmsnorm/rope/attention decode/etc.) son ~1.1 ms/token
   (~8%) — fusionarlos es fruta menor, no prioridad.

Estado: **70.4 tok/s estables**, ~293 GB/s agregados (~76% del pico).
Correctitud verificada en cada paso (secuencia greedy idéntica).

## ✅ FASE 3 COMPLETADA (2026-07-29 madrugada): formatos compactos → 91.7 tok/s

Implementado el encoder Rust de HQ4.1K/HQ5.1K (header 40B) que faltaba:
- `src/hqs/compact.rs` del conversor: meta-cuantiza los GroupParams del grid
  search MSE a 4 bits contra rangos f16 del superbloque, recuantizando los
  elementos contra los parámetros EFECTIVOS (los mismos números que decodifica
  el kernel). Tests unitarios con réplica bit a bit del decode CUDA.
- CLI: `--compact` (HQ4K→HQ4.1K, HQ5K→HQ5.1K, lm_head→HQ5.1K).
- Archivo: 4.74 → **3.42 GB** (tráfico/token 4.16 → ~2.8 GB).
- Kernels compactos del engine reescritos (estaban sin optimizar, 63 tok/s):
  staging del bloque ENTERO en shared (carga cooperativa coalescada, header
  incluido), decode con `__funnelshift_r`, camino rápido sin bounds-check,
  half2. → **82-92 tok/s** (varianza térmica).
- Calidad verificada: explicaciones coherentes y correctas en español.

**Historial del día: 55 (basura) → 57 (correcto) → 70.4 (kernels) → 91.7 (compacto).**

### Cara a cara con Ollama (2026-07-29, runs alternadas, mismo prompt, 100 tokens)

| Motor | Formato | GB/token | tok/s (estable) | BW efectivo |
|---|---|---|---|---|
| Ollama 0.32.5 (qwen3:4b) | Q4_K_M (~4.5 bpw) | ~2.5 | **~117** | ~292 GB/s |
| Héctor (compact) | HQ4.1K/HQ5.1K (~5.7 bpw) | ~2.8 | **~85** | ~238 GB/s |

El gap era 2.1× al empezar el día (55 vs 117); ahora es **1.38×**, y se
descompone en: ~1.12× de bytes (su atención va a 4 bits, la nuestra HQ5.1K
a 6.25) y ~1.24× de eficiencia de kernels (fusión, flash decode, menos
launches). Ambos motores saturan de forma similar el ancho de banda cuando
se mide por bytes — la diferencia restante es dieta y overhead por token.

## ✅ FASE 4 — ÚLTIMA RONDA (2026-07-29 noche): 95 tok/s, codo a codo con Ollama

1. **`--attn4` en el conversor**: atención en HQ4.1K (lm_head se protege en
   HQ5.1K). Archivo 3.31 GB, ~2.7 GB/token — dieta nivel Q4_K_M. Calidad
   verificada: misma explicación coherente que con HQ5.1K.
2. **Kernel fusionado `qk_norm_rope`**: rmsnorm(q)+rmsnorm(k)+rope(q)+rope(k)
   → 1 kernel por capa (144 → 36 lanzamientos/token). Un bloque por head:
   RMSNorm en shared → RoPE → writeback. Con variante `_dp` para graph replay.
   El graph builder elige fusionado si `use_qk_norm && rope`, clásico si no.

**Resultado (runs alternadas con Ollama, mismo estado térmico):**
- Héctor: **95.4 / 95.2 / 95.2 / 85.2** tok/s (el primer run de cada proceso
  paga el auto-tune: ~60)
- Ollama qwen3:4b (Q4_K_M): **92-120** tok/s (misma varianza térmica)

En rondas templadas se intercambian golpes (95.4 vs 93.6 ganó Héctor).
En pico frío Ollama conserva ~1.26× (120 vs 95). El gap era 2.1× por la mañana.

Historial completo del día: **55 (basura) → 57 → 70 → 92 → 95 tok/s.**

**HALLAZGO (2026-07-30):** el sampling con temperatura cuesta ~10 ms/token —
greedy corre a ~84 tok/s pero temp 0.7 cae a ~46. Sospechoso: `launch_top_k`
sobre 151k de vocab + 2 memcpy D2H + softmax en CPU por token
(`sampler.cpp:sample_with_temperature`). Optimizar: top-k en GPU con una sola
pasada (o muestrear en GPU del todo — los kernels ya existen:
`launch_top_p_cutoff`, `launch_categorical_sample`, sin usar).

Pendiente para la próxima:
1. Benchmark térmicamente controlado (enchufado, GPU fría, ventilador fijo).
2. Fusión de kernels pequeños (~1 ms/token: rope q+k, qk_norm+rope).
3. Opción atención HQ4.1K (−0.2 GB/token más, probar calidad).
4. Prefill attention (la redundancia ~384× de `attention_naive_kernel`).
5. Arreglar el validador del conversor (falsos FATAL con hints válidos).
6. Convertir modelo HQ5.1K completo para comparar calidad vs velocidad.

## 🎯 EL SIGUIENTE GRAN SALTO: formatos compactos HQ4.1K/HQ5.1K

Comparando con llama.cpp quedó claro el gap real con Ollama: **el header de
HQ4K/HQ5K es enorme**. 32 grupos × 4 B (min+scale fp16) = 128 B de header por
256 pesos:
- HQ4K: 128B header + 128B payload = **8.0 bits/peso** (Q4_K de llama.cpp: 4.5)
- HQ5K: 128B header + 160B payload = 9.0 bits/peso (Q5_K: 5.5)

Los formatos compactos **ya existen a medias**: `hqs_common.cuh` define el
header de 40 B (HQ41K: 168 B/bloque = 5.25 bits/peso) y hay kernels en
`matmul_hqs_compact.cu` + dispatch HQ41K/HQ51K en el engine. **Falta que el
conversor sepa escribirlos** (su enum QuantFormat solo tiene FP16/HQ3K/HQ4K/HQ5K).

Proyección con HQ41K (MLP) + HQ51K (atención y lm_head): tráfico por token
4.16 → ~2.95 GB → **~100 tok/s** al BW actual. La liga de Ollama, con tu
formato y tu motor.

### Dónde está el ~35% restante (Fase 2 real)

Por token hay ~615 kernels: ~250 GEMVs (dominan los bytes) + ~350 kernels
pequeños (rmsnorm ×73, qk_norm ×72, rope ×72, add, etc. — µs cada uno pero
suman ~2-3 ms/token):
1. **Fusionar ops pequeñas**: rope(q)+rope(k) en un launch; qk_norm dentro del
   rope o del epílogo del GEMV; menos nodos = menos overhead fijo.
2. **Eficiencia GEMV**: perfilar con nsys el % de BW real de gemv_hq4k/hq5k;
   probar __ldg, doble superbloque en vuelo, half2.
3. Objetivo realista: **75-85 tok/s** (85-90% del BW).

## TL;DR — Por qué 5 intentos de optimizar kernels no movieron la aguja

El cuello de botella del decode **no está en los kernels CUDA**: está en la
orquestación desde CPU. El loop de generación de todos los tests reconstruye
y relanza el forward completo desde CPU en cada token, y **nunca usa la
infraestructura de CUDA Graphs que ya está implementada en el engine**.

Mientras la CPU domine el tiempo por token, acelerar cualquier kernel es
invisible en la métrica de tokens/s. La aguja no se movía porque se estaba
optimizando la parte que no era el cuello de botella.

## Evidencia

Loop de `test_generate_kv.cpp:127-141` (idéntico patrón en
`test_generate_full.cpp:174-241` y `test_generate_modal.cpp:239,354`):

```cpp
for (int step = 0; step < max_tokens; step++) {
    cudaMemcpy(...);                         // copia síncrona del token
    auto cb = gb.build_forward_cached(...);  // (1) reconstruye TODO el command buffer
    engine.execute(cb);                      // (2) lanza op por op, resolviendo strings
    engine.sync();                           // (3) stall completo por token
    // sampler...
}
```

Costes fijos por token:

1. **`build_forward_cached` por token** — construye cientos de objetos
   `Command` con `std::string`s y mapas de parámetros (~30 capas × ~12 ops).
   Puro CPU, se repite en cada token porque `position` va horneada en el buffer.
2. **`engine.execute()` en vez de `execute_graph_replay()`** — cada op hace
   lookups de `unordered_map<string>` (`resolve_tensors`, `engine.cpp:316`) y un
   lanzamiento individual de kernel (~5-10 µs overhead × ~300 ops/token).
3. **`engine.sync()` por token** — la GPU queda ociosa mientras la CPU
   reconstruye el siguiente buffer y muestrea.

**Ningún test del repo llama a `execute_graph_replay()` ni `execute_graphed()`**
(verificado con grep). Toda la infraestructura de captura-una-vez ya existe
pero está muerta:

- `Engine::execute_graph_replay()` — `engine.cpp:151`
- `Engine::update_device_cache_pos()` / `d_cache_pos_` / `d_total_seq_` — `engine.cpp:138`
- Kernels `_dp` que leen posición/seq_len desde device (compatibles con
  captura): `attention_cached_v2_kernel_dp`, `rope_kernel_dp`,
  `kv_cache_update_kernel_dp` — `kernels/attention.cu`

## Plan (en orden)

### Fase 0 — Baseline (ANTES de tocar nada)

1. Instalar CUDA Toolkit **≥ 12.8** (requisito para Blackwell sm_120, ver
   "Entorno" abajo). No hay nvcc instalado ahora mismo.
2. Conseguir/convertir un modelo `.hnf` (no hay ninguno en el equipo;
   el conversor Rust está en otro repo). Sugerencia: Qwen2 1.5B en HQ4K.
3. Añadir `120` a `CMAKE_CUDA_ARCHITECTURES` en `CMakeLists.txt` (ahora es
   `"75;86;89"` — sin SASS sm_120 ni PTX embebido, los kernels pueden no
   cargar en la RTX 5070).
4. Compilar y correr `test_generate_kv` → anotar **tokens/s baseline**.
5. Perfilar de verdad: `nsys profile ./test_generate_kv ...` o activar
   `config.enable_profiling` del engine. Confirmar cuánto tiempo por token es
   GPU real vs. hueco (gaps = CPU). Esto valida (o refuta) el diagnóstico.

### Fase 1 — El fix principal: decode con graph replay

Cambiar el loop de decode de los tests para:

1. Construir el command buffer **una sola vez** antes del loop, usando las
   variantes `_dp` (posición leída de `d_cache_pos_` en device, no horneada).
2. Por token: `update_device_cache_pos()` (memcpy async de 4 bytes) +
   `execute_graph_replay()` → un solo `cudaGraphLaunch` por token.
3. Único sync por token: el del sampler para leer el token elegido
   (`sampler.cpp:136`).

Ganancia esperada: **2-5× en tokens/s** con modelos pequeños/medianos.
Además, a partir de aquí las optimizaciones de kernels SÍ se notarán.

### Fase 2 — Kernels (solo después de re-medir con Fase 1)

Por orden de impacto esperado:

1. **Prefill attention** (`attention_naive_kernel`, `kernels/attention.cu:29`):
   un thread por elemento de salida, cada thread recalcula TODOS los dot
   products Q·K, y lo hace en 3 pasadas (max, suma, salida). Los 128 threads
   del mismo (posición, head) repiten el mismo trabajo → redundancia ~3×head_dim
   (~384× con head_dim=128). Reescribir: un bloque por (query, head), scores
   en shared memory, una pasada con online softmax. Afecta al tiempo de prefill
   (prompts largos), no al decode.
2. **Decode attention** (`attention_cached_v2_kernel`): cargas de K/V escalares
   (half, 2 bytes) → vectorizar a `half2`/`float4`. Y los 4 warps por head son
   fijos: con contextos largos conviene split-K entre bloques.
3. **RoPE** (`rope_kernel`): `powf/cosf/sinf` por elemento por token → tabla
   de frecuencias precalculada. Coste pequeño, prioridad baja.
4. **GEMV HQ4K/HQ5K** (`matmul_hqs.cu`): ya está bien (coalescado, shared
   input, warp reduce, auto-tune). Es memory-bandwidth-bound: cerca del techo
   físico. NO invertir más aquí salvo que el profile diga lo contrario.

## Entorno (estado a 2026-07-29)

- GPU: **RTX 5070 Laptop, 8 GB** — Blackwell, `sm_120`, requiere CUDA 12.8+.
- Driver NVIDIA: instalado (`nvidia-smi` OK). **CUDA Toolkit: NO instalado**
  (sin `nvcc` en PATH, nada en `/usr/local/cuda*` ni `/opt/cuda`).
- `CMakeLists.txt` compila para `75;86;89` → **falta `120`** (o al menos
  `89-virtual` para que el JIT de PTX funcione en sm_120).
- Modelos `.hnf`: ninguno encontrado en el equipo.

## Regla de oro para la próxima ronda

**No optimizar nada sin un profile que muestre dónde va el tiempo.** Si el
profile de Fase 0 muestra la GPU ociosa entre kernels, el diagnóstico de este
documento es correcto y la Fase 1 es lo único que importa.
