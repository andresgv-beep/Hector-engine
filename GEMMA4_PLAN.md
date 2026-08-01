# Gemma 4 E2B en Héctor — plan de integración por fases

> Actualizado el 2026-07-31. Este documento es la fuente de verdad para la
> integración de Gemma 4 E2B. El trabajo debe avanzar en orden: no se comienza
> una fase hasta cumplir el criterio de salida de la anterior.

## Objetivo y alcance

Ejecutar en Héctor el bloque de texto de Gemma 4 E2B convertido al formato HNF
v9 propio de HELIOS. El artefacto de chat certificado es el checkpoint instruct:

- Checkpoint: `google/gemma-4-E2B-it`, SHA256
  `2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`
- HNF: `gemma4-e2b-it-text-compact.hnf`
- Tamaño: 4.250.889.374 bytes (3,959 GiB)
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

### Línea cerrada — ajuste ponderado del conversor

La solución de producción para el embedding sigue siendo el HNF actual con
`HELIOS_EMBED_IN_RAM=1`: conserva calidad y velocidad y desplaza 741,87 MiB de
VRAM a RAM. Esos bytes continúan presentes en el archivo y en la memoria host;
ese coste se acepta hasta encontrar un perfil que pase la barrera de producto.

La tabla compartida HQ5.1K queda únicamente como experimento opt-in. Su ahorro
de archivo y RAM es real, pero la batería conversacional encontró escritura
china, pérdida parcial de recuerdos y un bucle. Tampoco queda programada una
tabla compartida HQ5K no compacta: podría recuperar calidad, pero arriesga el
coste del `lm_head` y volvería a intercambiar calidad, tamaño y velocidad sin
haber aislado antes qué matrices toleran compresión.

El siguiente trabajo de calidad pertenece al **conversor**, no a Héctor:

1. Proteger embedding y tensores identificados como sensibles.
2. Proteger `down_proj`, que escribe de vuelta al flujo residual.
3. Comprimir solo matrices cuya tolerancia esté demostrada por familia.
4. Conservar las fusiones ya verificadas en Héctor y medir alrededor de los
   106,57 tok/s sin atribuir su ganancia al nuevo HNF.
5. No combinar un cambio de cuantización con otro cambio de kernels en el mismo
   artefacto o comparación.

Todo candidato a producción debe superar simultáneamente:

- **calidad conversacional:** cero fallos nuevos;
- **dos corpus:** sin regresión estadísticamente demostrable;
- **velocidad:** pérdida menor o igual al 1 %;
- **VRAM/tamaño:** ahorro real medido por separado en archivo, RAM y VRAM.

Si falla una sola barrera, el perfil permanece experimental aunque mejore las
otras tres.

El candidato que cumple ese reparto ya existía:
`gate/up=HQ3.1K`, `down=HQ5.1K`, atención HQ4.1K y embedding FP16. Tras retirar
los formatos legacy se repitió el A/B con el binario actual. Perdió 1,148 % en
potencia alta y 6,770 % en potencia baja, por lo que incumple la barrera de
velocidad y no se promueve. La línea queda cerrada con el HNF actual y
`HELIOS_EMBED_IN_RAM=1` como producción.

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

- **Fase activa:** Fase 9 — multimodal, desarrollada por separado en
  `GEMMA4_VISION_PLAN.md`.
- **Última fase completada:** Fase 8 — rendimiento y robustez.
- **Siguiente acción exacta:** iniciar V0 de visión con el HNF actual y
  `HELIOS_EMBED_IN_RAM=1` congelados como baseline de texto. La tabla compartida
  HQ5.1K, HQ3.1K y el perfil ponderado permanecen experimentales; no se cambia
  cuantización ni kernels durante el oráculo visual.
- **Cambios de código realizados:** loader/validador GM4X, pruebas metadata-only,
  primitivas, ruta PLE, grafo por capa, KV heterogéneo compartido y forward
  cached completo.
- **Hallazgo pendiente del conversor:** el HNF compacto afila la distribución
  (`KL=0,123965`, masa top-1 `0,1161` frente a `0,0599` de referencia), mientras
  el HNF FP16 reproduce la referencia. La calibración IT de 1.157 predicciones
  localiza el mismo riesgo en las posiciones abiertas: cuando top-1 FP16 < 0,80,
  el compacto sube la masa media de 0,4641 a 0,5525 y baja la entropía de 3,0674
  a 2,6868. No es un defecto de Héctor ni de la ventana de 512; aislar la familia
  de tensores antes de certificar calidad de sampling.

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
| 2026-07-31 | Fase 6 | Verificación externa FP32 completada y fase cerrada | HNF FP16: 3 tokens corr 0,999975, RMS 0,0907, KL 0,000254; 600 tokens corr 0,999879, RMS 0,0383, KL 0,000691; argmax coincide y top-10 9/10 y 10/10. Evidencia: `tools/GEMMA4_F6_VERIFICACION.md` |
| 2026-07-31 | Fase 7 (parcial) | HTF3 y plantilla de texto canónica verificados | 262.144 vocab, 514.906 merges y 24 added tokens; vectores idénticos al JSON en ES/EN/Unicode/byte fallback/control tokens; suite 13/13 |
| 2026-07-31 | Fase 7 (parcial) | Greedy completo estable; el supuesto chat no obedece | 23 tokens de prefill, ~124 tok/s; inglés responde París pero copia el prompt y español continúa el texto. Los experimentos posteriores demostraron que el checkpoint era base, no instruct |
| 2026-07-31 | Fase 7 (experimento) | Elevar los 105 MLP a HQ5.1K no mejora calidad | Greedy idéntico; en tokens 42,43,44 la masa top-1 empeora de 0,1260 a 0,2115 y KL de 0,6667 a 0,7064. Se descarta el perfil y se prueba `lm_head` FP16 |
| 2026-07-31 | Fase 7 (experimento) | `lm_head` FP16 mejora poco y no cambia greedy | En tokens 42,43,44 KL 0,6667→0,6469 y top-1 0,1260→0,1215, pero ES/EN generan exactamente lo mismo por 468 MiB extra. Siguiente aislamiento: tabla PLE FP16 en RAM |
| 2026-07-31 | Fase 7 (experimento) | PLE FP16 tampoco cambia greedy | KL 0,6667→0,6313 pero ES/EN idénticos por +2,86 GiB; incluso el HNF totalmente FP16 repite igual. El greedy valida estabilidad, pero Gemma 4 exige evaluar su sampling oficial |
| 2026-07-31 | Fase 7 (diagnóstico) | El checkpoint local es base, no instruct | SHA256 local `76dc84a5…` coincide exactamente con `google/gemma-4-E2B`; el oficial `E2B-it` es `2db5482b…`. Config local 4.914 bytes también coincide con base; instruct añade EOS 106. Explica las continuaciones/copias incluso en FP16 |
| 2026-07-31 | Fase 7 (base) | Completion crudo validado en FP16 y compacto | Greedy idéntico: `The capital of France is Paris…`; sampling ES coherente; compacto 112-124 tok/s y FP16 60-64 tok/s. Sin NaN, crash ni crecimiento inesperado. Chat queda pendiente exclusivamente de pesos `E2B-it` |
| 2026-07-31 | Fase 7 (IT) | Checkpoint instruct auténtico convertido y validado | SHA256 oficial `2db5482b…`; HNF de 4.250.889.374 bytes, 601 tensores; validador estricto 0 errores/0 avisos; contrato metadata 283/105/213 correcto |
| 2026-07-31 | Fase 7 | Tokenizer, plantilla y chat real verificados; fase cerrada | HTF3 y plantilla canónica pasan; greedy ES/EN responde París y cierra turno; sampling oficial ES coherente e instrucción system respetada en EN; contexto de 2.945 tokens recupera `ORIÓN-27`; 104-109 tok/s en prompts cortos; Héctor 13/13 y convertidor 34/34 |
| 2026-07-31 | Fase 8 (parcial) | Compacto IT comparado con HNF IT FP16 | Tres posiciones de chat: argmax coincide; top-10 9/10, 9/10 y 10/10; KL 0,000010, 0,000005 y 0,002833. En la posición menos forzada el compacto no afila el top-1: 0,9942 frente a 0,9988 FP16. Sampling oficial coherente en ambos; falta ampliar la calibración |
| 2026-07-31 | Fase 8 (parcial) | Rendimiento y memoria medidos | Compacto: pico 1.906 MiB, 106,4-107,9 tok/s; FP16: 4.626 MiB, 58,8-59,5 tok/s. Contexto compacto de 2.945 tokens: prefill 1.411,95 ms y respuesta correcta. Evidencia y comandos en `tools/GEMMA4_F8_BENCHMARK.md` |
| 2026-07-31 | Fase 8 (regresión) | Qwen3-4B y Qwen3-8B reales siguen operativos | Ambos cargan 399 tensores y generan París: 4B a 99,17 tok/s, 8B a 40,40 tok/s. `tests/qwen3_hq4k_v3.hnf` se excluye: está truncado (3.337.977.856 bytes reales frente a 4.533.663.705 declarados) y el validador encuentra 5 errores fatales |
| 2026-07-31 | Fase 8 (producto) | `helios_chat` ejecuta Gemma 4 con su contrato nativo | Selección por arquitectura: plantilla canónica, KV heterogéneo/compartido, forward Gemma y sampling oficial. Smoke greedy responde París; sesión de dos turnos a 0,7 recuerda el número 17; CUDA Graph replay activo. Regresión `helios_chat` Qwen3-4B correcta a 101 tok/s |
| 2026-07-31 | Fase 8 (calibración) | 1.158 posiciones IT separan motor, ventana y cuantización | FP16 coincide 98,27 % con FP32; compacto/FP16 79,53 %, estable antes/después de 512. En 293 posiciones con top-1 FP16 < 0,80, HQS sube la masa 0,4641→0,5525 y baja entropía 3,0674→2,6868. `gemma4_calibrate` deja NLL/entropía por posición; falta aislar familias de tensores |
| 2026-07-31 | Fase 8 (rendimiento) | Campaña HQS antes/después cerrada | Qwen3-4B queda limitado por memoria: HQ4.1K ocupa 74,9 % del tiempo GPU y las matrices grandes ya alcanzan 329-365 GB/s. HQ4 `1x16` solo aporta +0,65-0,85 % combinado y se descarta; HQ5 directo mejora `lm_head` 4B un 3,4 %, simplifica el kernel y conserva salidas Qwen/Gemma idénticas. Suite 13/13. Evidencia: `tools/HQS_DECODE_OPTIMIZATION_PLAN.md` |
| 2026-08-01 | Fase 8 (HQ3.1K) | Dtype compacto de 136 bytes integrado sin sustituir formatos existentes | Segundo corpus supera barrera estadística; HNF real Qwen3-4B ahorra 320,6 MiB (9,69%), queda a 0,271 puntos del simulador, mantiene el HNF anterior operativo y pasa convertidor 37/37 + Héctor 14/14. En régimen alto mide 107,61 frente a 100,99 tok/s. Evidencia: `tools/HQ31K_IMPLEMENTACION.md` |
| 2026-08-01 | Fase 8 (HQ3.1K producto) | Formato correcto, perfiles Qwen3-4B rechazados por conversación | MLP completo corrompe idioma/memoria; `gate_proj` completo deja 1/50 intrusión china; capas 0-17 pasan 50 turnos núcleo pero empeoran bucles largos 2/4 frente a 1/4. HQ3.1K queda experimental; siguiente palanca separada: embedding HQ4.1K. Evidencia: `tools/HQ31K_IMPLEMENTACION.md` |
| 2026-08-01 | Fase 8 (plan decode) | Prioridad corregida: fusiones antes de HQ3.1K | Se abre `tools/DECODE_FUSION_PLAN.md`: perfil por nodo, residual MLP + RMSNorm siguiente como primer candidato, embedding compartido HQ5.1K después y perfil mixto HQ3.1K al final. El dato de 1,67 ms se considera hipótesis hasta reproducirlo |
| 2026-08-01 | Fase 8 (perfil decode) | La bolsa pequeña existe, pero las fusiones probadas no llegan al corte | Qwen3-4B: 15,013 ms/token a 55 W; 1,434 ms de kernels no-HQS y 0,379 ms de huecos. RMSNorm optimizado aporta ~0,8%, una warp empeora y ADD+RMS siguiente tiene techo <0,08 ms. Perfil mixto real gate/up3+down5 ahorra 106,87 MiB pero da 15,364 ms/token; no es vía de +5 tok/s. Evidencia: `tools/DECODE_FUSION_PLAN.md` |
| 2026-08-01 | Fase 8 (embedding compartido) | Contrato y ahorro correctos; candidato HQ5.1K rechazado por producto | HNF 741,87 MiB menor, VRAM sin offload 4.368→3.626 MiB, carga 1,62→1,11 s y decode ~103,5 tok/s sin cambio; 1.755 posiciones no muestran regresión significativa (`p=0,246`), pero el A/B de 12 turnos introduce escritura china, olvida parte del perfil y entra en un bucle. Se conserva opt-in, no se promueve. Evidencia: `tools/DECODE_FUSION_PLAN.md` |
| 2026-08-01 | Fase 8 (decisión de producción) | Offload actual conservado; siguiente palanca es cuantización ponderada | `HELIOS_EMBED_IN_RAM=1` mantiene calidad/velocidad y saca 741,87 MiB de VRAM; el duplicado sigue en archivo/RAM. Se descarta por ahora HQ5K compartido y se fija barrera conjunta: cero fallos conversacionales, dos corpus sin regresión, velocidad dentro del 1 % y ahorro de memoria/tamaño medido |
| 2026-08-01 | Fase 8 (cierre) | Perfil ponderado rechazado por velocidad; baseline congelado | Tras retirar HQS legacy: conversor 39/39 y Héctor 14/14. A/B de 256 tokens: producción 104,3195 frente a 103,122 tok/s en régimen alto (−1,148 %) y 71,9505 frente a 67,0791 en bajo (−6,770 %). Se conserva `qwen3_4b_final.hnf` + offload y se abre V0 de visión |
