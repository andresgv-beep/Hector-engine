# Gemma 4 E2B Vision en Héctor — plan de integración

> Creado el 2026-08-01. Este documento separa la integración visual de la
> campaña de texto/HQS. Visión avanza por pruebas de correctitud y no modifica
> la cuantización del decoder hasta disponer de paridad FP16.

## Objetivo y límites

Portar la torre visual incluida en el checkpoint local `gemma-4-E2B-it` y
conectarla al decoder Gemma 4 ya funcional en Héctor.

Alcance inicial:

- una imagen por prompt;
- `max_soft_tokens=280`;
- encoder visual y proyector en FP16/BF16;
- prefill multimodal y decode textual normal;
- comparación numérica contra la implementación oficial.

Fuera de alcance hasta lograr paridad: audio, vídeo, lotes de varias imágenes,
cuantización de visión, optimizaciones y decodificación de todos los formatos
de imagen dentro del motor.

## Reglas de trabajo

1. Solo una fase activa y una prueba de salida por fase.
2. La ruta de texto puro debe quedar idéntica y pasar sus regresiones.
3. El oráculo promueve los pesos BF16 exactos a FP32; el primer HNF visual usa
   FP16 y se compara contra esa referencia. No se cuantiza visión para ocultar errores.
4. No se reutiliza el mapper CLIP: Gemma 4 tiene otro contrato.
5. Un campo o una operación no soportados deben producir error explícito.
6. Los artefactos grandes y los volcados de activaciones no se versionan.
7. Los experimentos HQS de texto quedan fuera de los archivos de esta campaña.
8. El oráculo queda fijado a una revisión concreta de la implementación oficial;
   no se compara contra `main` móvil durante la campaña.

## Referencia fijada

- Checkpoint local: `google/gemma-4-E2B-it`, el mismo certificado para texto.
- Implementación de referencia: Hugging Face Transformers commit
  `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`.
- Fuentes autoritativas: `modeling_gemma4.py`, `image_processing_gemma4.py` y
  `processing_gemma4.py` de esa revisión.
- El oráculo cargará solo `model.vision_tower.*` y
  `model.embed_vision.embedding_projection.weight`; nunca instanciará los
  aproximadamente 10 GB del modelo completo para obtener una referencia de
  321,47 MiB.
- El Python del sistema es 3.14 y no tiene PyTorch/Transformers. V0 usará un
  entorno aislado reproducible fuera del repositorio, con versiones registradas
  en el informe del oráculo; no se instalarán dependencias en HEXOS ni Héctor.

## Contrato confirmado del checkpoint local

### Inventario

| Bloque | Tensores |
|---|---:|
| `model.vision_tower.encoder` | 656 |
| `model.vision_tower.patch_embedder` | 2 |
| `model.embed_vision` | 1 |
| **Total visión** | **659** |

La torre tiene 168.544.704 valores y ocupa 321,47 MiB en BF16:

| Grupo | Parámetros | BF16 |
|---|---:|---:|
| MLP | 113.246.208 | 216,00 MiB |
| Q/K/V/O | 37.748.736 | 72,00 MiB |
| tabla posicional 2D | 15.728.640 | 30,00 MiB |
| proyector a texto | 1.179.648 | 2,25 MiB |
| proyección de patches | 589.824 | 1,12 MiB |
| normas | 51.200 | 0,10 MiB |
| límites de clipping | 448 escalares | <0,01 MiB |

Los 448 límites de clipping son finitos y aprendidos; en este checkpoint van
de -91 a 90. No se pueden descartar como metadatos decorativos.

### Arquitectura visual

| Parámetro | Valor |
|---|---:|
| Capas | 16 |
| Hidden / MLP | 768 / 3072 |
| Heads / KV heads | 12 / 12 |
| Head dim | 64 |
| Patch | 16 × 16 RGB = 768 valores |
| Pooling espacial | 3 × 3 |
| Soft tokens por defecto | 280 |
| Activación | `gelu_pytorch_tanh` / GeGLU |
| Norma | RMSNorm, epsilon 1e-6 |
| RoPE | 2D, theta 100 |
| Atención | bidireccional, escala 1,0 |
| Posición aprendida | `[2, 10240, 768]`, X + Y |
| Proyector visual→texto | `[1536, 768]` |
| Estandarización final | desactivada |
| Clipped linears | activados |

Cada capa ejecuta:

1. RMSNorm de entrada.
2. Q/K/V/O con clamp antes y después de cada proyección.
3. RMSNorm Q y K con peso; RMSNorm V sin peso.
4. RoPE 2D independiente para X e Y.
5. Atención completa no causal con escala 1,0.
6. RMSNorm posterior y residual.
7. RMSNorm pre-MLP, GeGLU, RMSNorm posterior y residual.

Después de las 16 capas se promedian grupos espaciales 3 × 3, se multiplica
por `sqrt(768)`, se aplica RMSNorm sin peso y se proyecta de 768 a 1536.

### Preprocesado y puente con texto

- La imagen se convierte a RGB, se redimensiona conservando relación de aspecto
  mediante bicúbico con antialias y cada lado queda alineado a
  `3 × 16 = 48` píxeles.
- El presupuesto por defecto es 2520 patches (`280 × 3²`).
- Los soft tokens reales son dinámicos y pueden ser menos de 280 según la
  relación de aspecto; el prompt debe usar exactamente el número producido.
- Los píxeles se reescalan a `[0,1]` y el modelo aplica `2*x-1`.
- Cada patch se aplana en orden HWC y recibe coordenadas `(x,y)`; el padding usa
  `(-1,-1)`.
- El prompt contiene BOI, un placeholder por soft token y EOI. Para este
  checkpoint: BOI=255999, IMAGE=258880, EOI=258882 y PAD=0.
- El embedding principal de cada placeholder se sustituye por el vector visual.
- PLE debe calcular esas posiciones usando PAD=0, no el ID de imagen.
- El checkpoint local no activa `use_bidirectional_attention` en el decoder;
  por tanto, el prefill textual conserva sus máscaras actuales. Si otro Gemma 4
  solicita esa opción, Héctor debe rechazarlo hasta implementarla.

## Estado real del código antes de empezar

### Conversor

- `Gemma4Mapper` es deliberadamente solo texto y omite los 659 tensores visuales.
- `--vision <mismo-checkpoint>` no selecciona una modalidad: vuelve a crear
  `Gemma4Mapper`, mapea los 600 tensores de lenguaje y los escribe con prefijo
  `vision.`. Usarlo hoy generaría un bloque incorrecto que parece válido.
- El mapper CLIP no sirve: presupone dos normas, MLP `fc1/fc2`, posición 1D y
  no conoce clamps, GeGLU, RoPE 2D ni pooling Gemma 4.
- `VisionModelConfigBin` solo describe CLIP/SigLIP/ViT/EVA y carece de campos
  esenciales de Gemma 4.
- `build_execution_hints_binary()` declara siempre un modelo de texto y escribe
  `TextModelConfigBin`, incluso en una conversión solo visual. V1 debe corregir
  el contrato por modalidades, no limitarse a añadir expresiones regulares.
- El diccionario canónico de visión actual solo acepta nombres CLIP (`ln1`,
  `fc1/fc2`, posición 1D); también debe ampliarse y validarse para Gemma 4.

### Héctor

Héctor puede cargar un bloque HNF `vision`, pero todavía no tiene un forward
visual. `test_multimodal.cpp` prueba decoders de distintos dominios, no imágenes.

Primitivas reutilizables:

- matmul FP16 y HQS;
- RMSNorm con y sin peso;
- GELU/GeGLU;
- suma y escalado;
- atención prefill con `causal=false`.

Huecos concretos:

- clamp por escalares aprendidos;
- lookup y suma de posiciones X/Y;
- RoPE 2D;
- escala de atención configurable a 1,0 (el helper fija `1/sqrt(head_dim)`);
- pooling espacial por coordenadas;
- runner/grafo visual independiente;
- entrada desde embeddings ya calculados al decoder;
- sustitución de placeholders conservando PLE con PAD;
- preprocesado exacto de imagen.

## Hoja de ruta

### V0 — Oráculo y contrato reproducible — CERRADA

Crear una herramienta de referencia que cargue únicamente torre visual y
proyector desde el checkpoint. La fixture principal será un RGB sintético
determinista de 960 × 672: ya está alineado a 48, ocupa los 2520 patches y
produce exactamente 280 soft tokens. Debe volcar:

- patches y coordenadas;
- salida del patch embedder;
- salida de las capas 0 y 15;
- salida del pooler;
- 280 embeddings proyectados a 1536.

El informe guardará revisión upstream, versiones Python/PyTorch/Transformers,
formas, dtype, mínimos/máximos, media, RMS, SHA256 de cada volcado y tolerancias.
Los binarios de activaciones vivirán fuera de Git; solo se versionan fixture,
script e informe pequeño.

La ejecución dorada será FP32 en CPU: los valores BF16 del checkpoint se
representan exactamente al promoverlos y el resultado no depende de kernels de
una GPU concreta. El futuro runner FP16 se aceptará por error numérico frente a
este oráculo, nunca por igualdad bit a bit.

**Salida:** artefacto reproducible con formas, hashes y tolerancias fijadas antes
de implementar CUDA.

Resultado y hashes: `tools/GEMMA4_VISION_V0.md`.

### V1 — Mapper visual y HNF — CERRADA

- Añadir selección de mapper por arquitectura **y modalidad**.
- Implementar `Gemma4VisionMapper` para los 659 tensores.
- Mantener pesos y clamps en FP16 para el primer HNF de referencia.
- Extender los hints de visión de forma versionada, sin reinterpretar campos
  CLIP existentes.
- Hacer que `--text DIR --vision DIR` produzca bloques disjuntos.

Contrato mínimo de nombres:

- `vision.patch_embed.{input_proj,position_embedding}.weight`;
- `vision.layerN.{ln_attn_in,ln_attn_post,ln_mlp_in,ln_mlp_post}.weight`;
- `vision.layerN.attn.{q,k,v,o}_proj.{weight,input_min,input_max,output_min,output_max}`;
- `vision.layerN.attn.{q,k}_norm.weight`;
- `vision.layerN.mlp.{gate,up,down}.{weight,input_min,input_max,output_min,output_max}`;
- `vision.projector.weight`.

El binario conserva `VisionModelConfigBin` de 64 bytes y sus valores legacy.
Gemma 4 recibe un nuevo `encoder_type` sin renumerar CLIP/SigLIP/ViT/EVA y una
extensión `GM4V` apuntada desde los ocho bytes aún libres de
`ExecutionHintsBin.reserved`. La extensión debe transportar al menos head/KV
dim, posición 2D, RoPE, pooling, salida dinámica, activación, RMS epsilon,
clipping, estandarización, proyección y contrato de preprocesado. Un lector
antiguo podrá ignorarla; un lector Gemma 4 deberá exigirla.

**Pruebas:** mapper de texto 600/600 y 0 visuales; mapper visual 659/659 y 0
textuales/audio; nombres únicos y aceptados por el diccionario; hints binarios
correctos tanto para texto+visión como para visión sola; rechazo del contrato
incompleto. Ejecutar además las 39 pruebas actuales del conversor sin cambios.

**Salida:** HNF texto+visión inspeccionable, todavía sin ejecutar visión.

Resultado: mapper 659/659, HNF visual FP16 reproducible y HNF combinado
texto+visión válidos. Evidencia en `../helios_convert_v9.1/GEMMA4_VISION_V1.md`.

### V2 — Loader y validador de visión — CERRADA

- Leer la extensión visual Gemma 4.
- Validar arquitectura, formas, dtype y los 448 clamps.
- Calcular presupuesto de pesos/scratch antes de reservar VRAM.
- Cargar y descargar el bloque de visión independientemente del texto.

**Salida:** 659/659 tensores validados y bloque visual cargable sin construir
el forward.

Resultado: `GM4V` se valida de forma estricta, los 448 clamps escalares son
finitos, los 337.089.408 bytes de pesos cargan y se liberan tanto en el HNF
visual como en el combinado. La cota previa de scratch es 368.040.960 bytes.
Una sola reserva contigua deja el consumo medido en 337.641.472 bytes y evita
136.314.880 bytes de overhead frente a reservar los 659 tensores por separado.
Evidencia en `tools/GEMMA4_VISION_V2.md`.

### V3 — Preprocesado exacto sin dependencia de UI — CERRADA

- Implementar primero sobre un buffer RGB ya decodificado.
- Resize bicúbico con relación de aspecto y alineación a 48 píxeles.
- Rescale, patchify HWC, padding y coordenadas `(x,y)`.
- Devolver también el número real de soft tokens para construir el prompt; no
  rellenar siempre 280 placeholders.
- Dejar la decodificación PNG/JPEG detrás de una interfaz; no bloquear la
  correctitud del motor eligiendo todavía una librería de imagen.

**Pruebas:** coincidencia exacta de tamaños/coordenadas y tolerancia numérica
del resize/patchify contra el oráculo en varias relaciones de aspecto.

Resultado: cinco relaciones de aspecto, incluido el caso extremo, coinciden
byte a byte con `aten._upsample_bicubic2d_aa`; patches y posiciones del caso
alineado reproducen además los hashes raw de V0. La salida conserva 2.520 slots
pero devuelve 280/256/252/266/280 soft tokens reales según la imagen. Evidencia
en `tools/GEMMA4_VISION_V3.md`.

### V4 — Encoder visual FP16 — CERRADA

Implementar una ruta `Gemma4VisionRunner` separada del `GraphBuilder` de texto:

- patch projection y posición aprendida 2D;
- clamp de entrada/salida para las siete lineales por capa;
- cuatro normas, Q/K norm, V norm sin peso;
- RoPE 2D y atención no causal con escala 1,0;
- GeGLU y residuales;
- pooling 3 × 3, escala en FP32, RMSNorm sin peso y proyector 768→1536.

**Pruebas:** comparar cada frontera de V0. La salida final debe conservar
argmax/correlación y cumplir tolerancias FP16 acordadas, sin NaN.

Resultado: `Gemma4VisionRunner` independiente ejecuta patch embedder, las 16
capas, pooler y proyector. Las cinco fronteras cumplen las barreras de V0; la
peor desviación es NRMSE 0,00807 en capa 15 frente al límite 0,030. Pesos y
scratch/workspace medidos ocupan 337.641.472 y 318.767.104 bytes. Evidencia en
`tools/GEMMA4_VISION_V4.md`.

### V5 — Puente multimodal al decoder — CERRADA

- Expandir `<image_soft_token>` al número real de salidas visuales.
- Sustituir embeddings principales en las posiciones de imagen.
- Calcular la parte de identidad de PLE con una copia de IDs donde imagen se
  reemplaza por PAD=0, pero calcular la proyección contextual de PLE desde los
  embeddings ya sustituidos por visión; son dos entradas distintas en upstream.
- Hacer prefill con el KV actual y continuar decode textual sin volver a
  ejecutar la torre visual.

**Pruebas:** prompt solo texto idéntico bit a bit; prompt con imagen compara
logits prefill y varios pasos greedy contra la referencia.

### V6 — Producto mínimo y robustez — CERRADA

- Una imagen desde CLI con error claro de formato/tamaño.
- Probar imágenes cuadradas, verticales y panorámicas.
- Medir VRAM máxima, tiempo de visión, prefill y decode.
- Verificar carga repetida, dos turnos y liberación de memoria.

**Salida:** descripción coherente de imágenes y cero regresiones de texto.

Resultado: `gemma4_vision_chat` conecta PNG→RGB, V3, V4, V5 y decode textual.
Las pruebas cuadrada, vertical y panorámica producen 256/252/266 soft tokens;
la captura real lee correctamente cinco nombres de archivo. La torre tarda
aproximadamente 80–85 ms y su pico es `+626 MiB`; después se descarga antes de
cargar texto. Un segundo turno reutiliza el KV sin repetir visión. Héctor pasa
18/18 y el baseline textual conserva exactamente su SHA de logits. Evidencia
en `tools/GEMMA4_VISION_V6.md`.

### V7 — Residencia estable y staging por fases — CERRADA

Mantener texto y KV residentes mientras la torre visual permanece respaldada
por el HNF/RAM. La lectura HMM directa se mide pero no se adopta: cuBLAS tarda
~1,18 s porque relee tiles por PCIe. La solución usa una ventana CUDA de 32 MiB
y una transferencia contigua por patch/layer/proyector.

**Salida:** precisión idéntica a V4, KV de 4096 residente, pico visual de
`+336 MiB` frente a `+626 MiB`, torre caliente de ~102 ms frente a ~69 ms y
decode dentro del 1 %. Evidencia en `tools/GEMMA4_VISION_V7.md`.

### V8 — Integración de producto y optimización opcional

- mover la selección por capacidades y el ciclo de vida a HexOS — política
  implementada y pendiente de commit en `hexos-core`;
- perfil de doble buffer cerrado: 28,95 ms calientes son copias, cada capa
  ofrece 4,25 ms de cómputo para ocultar 1,62 ms de transporte y la GPU solapa
  de verdad memoria pageable. El pipeline genérico recupera 24,49 ms y cuesta
  20 MiB adicionales, superando el objetivo mínimo de 20 ms; evidencia en
  `tools/GEMMA4_VISION_V8_PROFILE.md` y contrato reutilizable en
  `tools/MAPPED_WEIGHT_PIPELINE.md`;
- cuantización visual solo contra el HNF FP16 y con A/B conversacional;
- varias imágenes, vídeo y audio como contratos separados.

## Primera acción segura

La primera edición será el oráculo V0, aislado de los dos productos. La primera
edición de producto será V1: hacer que el conversor distinga modalidad y añadir
un mapper visual que solo define nombres canónicos y hints. No se tocarán
kernels ni el forward de texto. Antes de convertir pesos se exigirá una prueba
de inventario 659/659; así el fallo actual de `--vision` queda bloqueado sin
arriesgar el HNF de producción.

## Estado actual

- **Fase cerrada:** V0 — dos ejecuciones CPU/FP32 produjeron ocho artefactos
  idénticos byte a byte. El oráculo cargó 659/659 tensores y terminó con
  `[280,1536]` sin valores no finitos. Evidencia en
  `tools/GEMMA4_VISION_V0.md`.
- **Fase cerrada:** V1 — mapper 659/659, modalidad separada y extensión binaria
  `GM4V`. El HNF solo visual de 337.212.903 bytes se reprodujo byte a byte
  (`5dde9dbc…4edb057`); el combinado texto+visión contiene 1260 tensores y
  pasa el validador estricto. Evidencia en
  `../helios_convert_v9.1/GEMMA4_VISION_V1.md`.
- **Fase cerrada:** V2 — loader estricto `GM4V`, inventario 659/659, clamps
  448/448 y carga/descarga independiente verificados sobre los HNF visual y
  combinado. Evidencia en `tools/GEMMA4_VISION_V2.md`.
- **Fase cerrada:** V3 — resize bicúbico AA RGB8, patchify HWC, posiciones XY,
  padding y soft tokens dinámicos verificados contra ATen 2.9.1 y V0. Evidencia
  en `tools/GEMMA4_VISION_V3.md`.
- **Fase cerrada:** V4 — runner FP16 visual independiente verificado contra
  las cinco fronteras del oráculo V0, tanto con HNF visual como combinado.
  Evidencia en `tools/GEMMA4_VISION_V4.md`.
- **Fase cerrada:** V5 — expansión dinámica BOI/IMAGE/EOI, sustitución de
  embeddings, separación correcta de identidad/contexto PLE y continuidad KV.
  Texto puro permanece idéntico byte a byte; el control FP16 coincide en los
  cuatro argmax y 10/10 del top-10 contra la referencia externa. Evidencia en
  `tools/GEMMA4_VISION_V5.md`.
- **Fase cerrada:** V6 — CLI PNG a conversación con torre descargable, tres
  relaciones de aspecto, segundo turno y métricas de VRAM/latencia. La captura
  real identificó y leyó correctamente los archivos mostrados. Evidencia en
  `tools/GEMMA4_VISION_V6.md`.
- **Fase cerrada:** V7 — texto y KV permanecen residentes; visión vive en el
  HNF/RAM y usa una ventana temporal de 32 MiB. Ahorra 290 MiB de pico frente
  a V6, añade ~33 ms por imagen caliente y conserva precisión/decode. Evidencia
  en `tools/GEMMA4_VISION_V7.md`.
- **Fase activa:** V8 — política HexOS implementada y pipeline genérico de
  doble buffer certificado con Gemma 4; falta decidir el default de producto y
  conectar adjuntos al proceso persistente.
- **Baseline de texto congelado:** `qwen3_4b_final.hnf` para regresión general y
  `gemma4-e2b-it-text-compact.hnf` para integración, ambos con
  `HELIOS_EMBED_MMAP=1`. `HELIOS_EMBED_IN_RAM=1` queda únicamente como ruta
  histórica de comparación. La campaña de decode cerró el perfil ponderado por
  perder 1,148 % en potencia alta y 6,770 % en potencia baja. Los dos HNF se
  recertificaron con el conversor determinista `37610b3`: Qwen SHA256
  `5aec0eba…269105` y Gemma SHA256 `b67df381…b79660`.
- **Última comprobación:** V4 coincide con las mismas métricas usando pesos en
  VRAM o staging desde HNF; carga/descarga mmap recupera toda la memoria. Con
  contexto 4096, prefill y decode permanecen en 85,8 ms y 136,4 tok/s. Tras V7,
  Héctor pasa 18/18 y texto conserva sus barreras independientes.
- **Siguiente acción exacta:** hacer que HexOS active
  `HELIOS_MMAP_DOUBLE_BUFFER=1` junto al mmap visual compatible, conservando el
  override a cero. Después, diseñar el contrato de adjuntos del chat persistente
  sobre detección de modalidad HNF, no sobre nombres de modelos.
