# Gemma 4 E2B en Héctor — plan de integración por fases

> Actualizado el 2026-07-31. Este documento es la fuente de verdad para la
> integración de Gemma 4 E2B. El trabajo debe avanzar en orden: no se comienza
> una fase hasta cumplir el criterio de salida de la anterior.

## Objetivo y alcance

Ejecutar en Héctor el bloque de texto de Gemma 4 E2B convertido al formato HNF
v9 propio de HELIOS. La primera entrada será el modelo compacto ya generado:

- HNF: `gemma4-e2b-text-compact.hnf`
- Tamaño: 4.250.889.356 bytes (3,959 GiB)
- Tensores: 601
- Cuantización: 283 FP16, 213 HQ51K y 105 HQ41K
- Validador estricto del convertidor: 0 errores, 0 avisos

Fuera de alcance hasta terminar texto: vision tower, audio tower, interfaz de
chat, servidor/API y optimizaciones de rendimiento no necesarias para lograr
correctitud.

## Reglas para no desviarnos

1. Solo puede haber **una fase activa**.
2. Cada turno empieza leyendo este documento y comprobando `git status`.
3. Antes de editar se identifica el fallo concreto y la prueba que lo demuestra.
4. Cada fase termina con pruebas reproducibles y se anota aquí el resultado.
5. No se optimiza código que todavía no produce resultados numéricamente correctos.
6. No se modifica el convertidor para ocultar una limitación del motor. Si el
   contrato HNF cambia, debe documentarse en ambos repositorios.
7. No se hacen refactors generales durante esta integración.
8. Ante un bloqueo, se registra en `Registro de trabajo`; no se salta de fase.

## Contrato del modelo ya convertido

### Arquitectura de texto

| Parámetro | Valor |
|---|---:|
| Arquitectura HNF | `gemma4` (`ARCH_GEMMA4 = 15`) |
| Capas | 35 |
| `hidden_size` | 1536 |
| `intermediate_size` base | 6144 |
| Heads / KV heads | 8 / 1 (MQA) |
| `head_dim` local / global | 256 / 512 |
| Vocabulario | 262.144 |
| Activación | `gelu_pytorch_tanh` (GeGLU) |
| Ventana local | 512 |
| Patrón | 4 capas locales + 1 global |
| Capas con KV compartido | 20 |
| Softcap final | 30,0 |
| PLE por capa | 256 |
| RoPE por capa | local: default, θ=10.000, rotary=1,0; global: proporcional (`8`), θ=1.000.000, rotary=0,25 |
| Escala de atención | 1,0 (no `1/sqrt(head_dim)`) |

Las capas no son uniformes. El HNF incluye una extensión binaria `GM4X` con
35 registros por capa. Esos registros son la autoridad para tipo de atención,
`head_dim`, dimensión intermedia, RoPE y demás variaciones por capa.

### Per-Layer Embeddings (PLE)

- `embed_tokens_per_layer`: `[262144, 8960]`, donde `8960 = 35 × 256`.
- Contiene aproximadamente **2,35 mil millones** de parámetros, no millones.
- En el HNF compacto está cuantizado como HQ51K y ocupa 1.835.008.000 bytes.
- También participan `per_layer_model_projection`, las proyecciones y gates por
  capa, `post_per_layer_input_norm` y `layer_scalar`.

Pipeline PLE confirmado contra la implementación oficial:

1. Lookup PLE y escala por `sqrt(256) = 16`.
2. Proyección del embedding principal a 8960, escala por `1/sqrt(1536)`,
   reshape a `[35, 256]` y RMSNorm por cada segmento de 256.
3. Combinación `(proyección_contextual + embedding_PLE) / sqrt(2)`.
4. En cada capa: gate desde hidden → GELU tanh → multiplicación por el segmento
   PLE → proyección a hidden → RMSNorm → residual.

### Nombres canónicos relevantes por capa

- `ln_attn_in`
- `ln_attn_post`
- `ln_mlp_in`
- `ln_mlp_post`
- `ln_ple_post`
- `attn.{q,k,v,o}` y normas Q/K
- `mlp.{gate,up,down}`
- proyección y gate PLE
- `layer_scalar`

No asumir `ln_attn_out`: ese nombre no pertenece al mapeo Gemma 4 actual.

## Estado inicial confirmado

Héctor ya dispone de HNF v9, HTF3, HQ41K/HQ51K, MQA/GQA, normas Q/K y
matmul cuantizado. Sin embargo, todavía existen estos bloqueos:

- `ExecArch` termina en 14 y el valor 15 puede degradarse a SigLIP.
- `ExecRoPEType` termina en 7.
- El loader ignora los datos `GM4X` guardados en `reserved[20]` del hint.
- El parser JSON omite objetos y arrays anidados; no puede sustituir a `GM4X`.
- El detector de normas espera nombres que no coinciden con el mapper actual.
- Graph builder y scratch presuponen dimensiones uniformes en todas las capas.
- El lookup de embeddings solo admite FP16; el PLE compacto es HQ51K.
- La atención genérica escala por `1/sqrt(head_dim)`, pero Gemma 4 usa 1,0.
- V necesita RMSNorm sin peso aprendido; solo Q/K tienen pesos de norma.
- Las capas 15-34 no calculan K/V propios: las locales reutilizan la capa 13 y
  las globales la capa 14.

## Hoja de ruta

### Fase 0 — Baseline y protección del trabajo existente

**Objetivo:** disponer de una referencia reproducible antes de cambiar Héctor.

- Confirmar árbol Git y build actual.
- Ejecutar las pruebas existentes que sean viables sin Gemma 4.
- Registrar GPU, versión CUDA y comando de compilación.
- No arreglar fallos ajenos a Gemma 4; solo documentarlos.

**Criterio de salida:** baseline registrado y distinción clara entre fallos
previos y regresiones nuevas.

### Fase 1 — Reconocimiento del contrato HNF/GM4X

**Objetivo:** leer correctamente todos los metadatos sin cargar pesos ni usar CUDA.

- Añadir `ARCH_GEMMA4 = 15` y `ROPE_PROPORTIONAL = 8`.
- Definir estructuras versionadas para cabecera y registros `GM4X`.
- Leer offset y tamaño desde el área reservada del hint.
- Validar límites, magic, versión, tamaño de registro y número de capas.
- Mantener una configuración Gemma 4 dedicada con sus 35 registros; no
  aplastarla dentro de una configuración uniforme.
- Añadir una utilidad o prueba de inspección exclusivamente de metadatos.

**Prueba obligatoria:** abrir el HNF compacto real y comprobar arquitectura,
35 capas, alternancia local/global, dimensiones variables, RoPE y PLE sin
reservar VRAM.

**Criterio de salida:** el loader informa `gemma4`, interpreta íntegramente
`GM4X` y rechaza limpiamente una extensión corrupta o incompatible.

### Fase 2 — Inventario y formas de tensores

**Objetivo:** demostrar que Héctor entiende todos los pesos antes de construir
el forward.

- Definir el descriptor de nombres Gemma 4 conforme al mapeo canónico.
- Validar los 601 tensores, sus dtypes, rangos y formas.
- Validar por capa Q/K/V/O, MLP variable, cinco normas, gates, proyecciones PLE
  y `layer_scalar`.
- Detectar explícitamente tensores ausentes, duplicados o con forma incorrecta.
- Calcular y mostrar el presupuesto de pesos, scratch y KV antes de reservarlos.

**Criterio de salida:** validación completa 0/0 contra el HNF compacto, sin
construir todavía el grafo de inferencia.

### Fase 3 — Primitivas numéricas Gemma 4

**Objetivo:** implementar y probar aisladas las operaciones pequeñas que pueden
producir errores silenciosos.

- RMSNorm Gemma 4: `weight * normalized` directamente. No aplicar `1 + weight`.
- RMSNorm de V sin peso aprendido.
- Escala del embedding por `sqrt(hidden_size)`.
- `gelu_pytorch_tanh` / GeGLU exacto.
- Sandwich de normas y aplicación de `layer_scalar`.
- Softcap de logits: `tanh(logits / 30) * 30`.
- RoPE proporcional y rotary parcial según el registro de cada capa.
- Escala de atención 1,0.

**Prueba obligatoria:** comparar cada primitiva con una referencia de alta
precisión usando tolerancias documentadas. La comparación de bloque se realiza
en Fase 5, cuando exista el primer grafo Gemma 4 y no antes.

**Criterio de salida:** todas las primitivas pasan pruebas numéricas; todavía
no se exige generación completa.

### Fase 4 — Lookup cuantizado de PLE

**Objetivo:** obtener una fila del enorme PLE HQ51K sin descomprimir la tabla.

- Implementar lookup/dequant de fila HQ51K con límites estrictos.
- Evaluar ruta GPU y, solo si aporta valor medido, memoria host mapeada.
- Conectar proyección global, selección del segmento de 256 valores por capa,
  gate, proyección de capa y norma posterior.
- Evitar una copia o dequant completa de los 1,835 GB.

**Prueba obligatoria:** comparar filas y salida de la ruta PLE para varios token
IDs y capas contra la referencia.

**Criterio de salida:** PLE correcto y dentro del presupuesto de memoria.

### Fase 5 — Grafo Gemma 4 por capa

**Objetivo:** construir el forward respetando la heterogeneidad de las 35 capas.

- Scratch y shapes por capa, no globales.
- Usar `head_dim` 256 o 512 según `GM4X`.
- Usar dimensión MLP específica de cada capa.
- Integrar las cuatro normas principales, PLE y residual en el orden exacto.
- Normalizar V sin peso y usar escala de atención 1,0.
- Mantener inicialmente una ruta Gemma 4 explícita y legible; generalizar solo
  cuando la equivalencia esté demostrada.

**Prueba obligatoria:** comparar activaciones de embedding, PLE y al menos las
dos primeras capas; incluir una capa global y una capa MLP ancha.

**Criterio de salida:** activaciones dentro de tolerancia antes de implementar
la semántica avanzada del KV.

### Fase 6 — Atención alternada y KV compartido

**Objetivo:** reproducir exactamente la atención temporal de Gemma 4.

- Máscara local de 512 tokens en las capas indicadas.
- Atención global en su patrón real.
- KV con dimensiones distintas para capas locales y globales.
- Implementar y validar el uso compartido de KV de las últimas 20 capas.
- Reutilizar capa 13 para atención local y capa 14 para global en las capas
  compartidas 15-34; no reservar ni calcular K/V propios para ellas.
- Verificar prefill y decode por separado, incluidos límites de ventana.

**Prueba obligatoria:** comparar logits de prefill y varios pasos de decode con
la referencia usando prompts cortos y otros mayores que 512 tokens.

**Criterio de salida:** logits y evolución del KV dentro de tolerancia.

### Fase 7 — Generación y tokenizer

**Objetivo:** producir texto coherente mediante HTF3 sin compensar errores del
modelo desde el frontend.

- Validar vocabulario 262.144, 514.906 merges, tokens especiales y plantilla.
- Probar greedy primero; después temperatura/top-k/top-p.
- Probar español e inglés, EOS y secuencias largas.
- Guardar prompts y resultados reproducibles.

**Criterio de salida:** salida coherente y estable, sin NaN, accesos inválidos ni
crecimiento inesperado de memoria.

### Fase 8 — Rendimiento y robustez

**Objetivo:** optimizar únicamente después de conseguir correctitud.

- Medir prefill, decode, VRAM, ancho de banda y coste específico de PLE.
- Recuperar CUDA Graph replay si la ruta nueva lo impide.
- Perfilar antes de fusionar o reescribir kernels.
- Añadir casos negativos del loader y pruebas de regresión para otros modelos.

**Criterio de salida:** benchmark reproducible, sin regresiones conocidas en
Qwen/Phi/DeepSeek y con objetivos de rendimiento anotados a partir de medidas.

### Fase 9 — Multimodal, aplazada

Vision y audio se planificarán en un documento separado cuando el texto haya
superado todas las fases anteriores. Su presencia en el modelo original no
autoriza cambios multimodales durante este plan.

## Matriz mínima de verificación

| Nivel | Qué se compara | Momento |
|---|---|---|
| Contenedor | hints, `GM4X`, tensors, formas y dtypes | Fases 1-2 |
| Operación | normas, GELU, RoPE, softcap, lookup PLE | Fases 3-4 |
| Activación | embedding, PLE, capas local/global/ancha | Fase 5 |
| Estado | KV local/global/compartido | Fase 6 |
| Modelo | logits de prefill y decode | Fases 6-7 |
| Producto | texto, estabilidad, VRAM y rendimiento | Fases 7-8 |

Usar FP32 de referencia cuando sea posible. Las tolerancias deben fijarse antes
de mirar el resultado, separando error de FP16 del error de cuantización HQS.

## Estado actual

- **Fase activa:** Fase 6 — atención alternada y KV compartido.
- **Última fase completada:** Fase 5 — grafo Gemma 4 por capa.
- **Siguiente acción exacta:** capturar logits de una implementación de
  referencia para un prompt corto y otro mayor de 512 tokens y compararlos con
  el forward cached real ya operativo; no cerrar Fase 6 solo con consistencia
  interna prefill/decode.
- **Cambios de código realizados:** loader/validador GM4X, pruebas metadata-only,
  primitivas, ruta PLE, grafo por capa, KV heterogéneo compartido y forward
  cached completo.

## Registro de trabajo

| Fecha | Fase | Resultado | Evidencia / siguiente paso |
|---|---|---|---|
| 2026-07-31 | Planificación | Plan reordenado desde contrato hasta optimización | Comenzar Fase 0 en el próximo turno |
| 2026-07-31 | Fase 0 | Build correcto; 4/6 tests legacy pasan | Fallos previos: registro duplicado `silu_mul` y crash final de `test_hnf_loader` |
| 2026-07-31 | Fase 1 | GM4X v1 leído y validado en compacto y FP16 | 35 capas; corrupción estructural rechazada; sin reservar VRAM |
| 2026-07-31 | Fase 2 | 601/601 tensores válidos en ambos HNF | Compacto: 283 FP16, 105 HQ41K, 213 HQ51K; 0 errores |
| 2026-07-31 | Fase 3 | Primitivas numéricas verificadas en CUDA | RMSNorm directo/sin peso, GELU tanh, escalas, softcap y RoPE proporcional |
| 2026-07-31 | Fase 4 | Lookup HQ51K terminado; ruta PLE aún incompleta | Fila real 42: 8960 valores desde solo 7000 bytes, igualdad con decoder CPU |
| 2026-07-31 | Fase 4 | Ruta PLE completa verificada y cerrada | Filas 0/42/262143; preparación de 35 segmentos max abs 0,015625; inyección en capas 0/4/15/34 max abs 0,00390625; workspace 17,5 MiB para 1x512 |
| 2026-07-31 | Fase 5 | Grafo heterogéneo por capa verificado y cerrado | PLE multi-token; capas reales 0-4 encadenadas contra CPU, global HD512 max abs 0,0234375; cola MLP/PLE de capa 15 a 12288 max abs 0,0273438 |
| 2026-07-31 | Baseline | Suite legacy reparada: 11/11 tests pasan | Ops fusionadas son builtins reutilizables; mock HNF corregido; loader rechaza size/dtype inválidos y libera correctamente RAM mapeada |
| 2026-07-31 | Fase 6 (parcial) | Layout KV y forward cached completo operativos | 15 slots físicos; capas 15-34 aliasan local 13/global 14; máscaras prefill/decode incluido límite >512; HD512; HNF real prefill vs decode max/mean 0 y mismo argmax 236772 |
