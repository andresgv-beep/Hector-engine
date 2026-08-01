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
3. La referencia inicial es FP16/BF16; no se cuantiza visión para ocultar errores.
4. No se reutiliza el mapper CLIP: Gemma 4 tiene otro contrato.
5. Un campo o una operación no soportados deben producir error explícito.
6. Los artefactos grandes y los volcados de activaciones no se versionan.
7. Los experimentos HQS de texto quedan fuera de los archivos de esta campaña.

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
  y cada lado queda alineado a `3 × 16 = 48` píxeles.
- El presupuesto por defecto es 2520 patches (`280 × 3²`).
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

### V0 — Oráculo y contrato reproducible

Crear una herramienta de referencia que cargue únicamente torre visual y
proyector desde el checkpoint. Con una imagen sintética fija debe volcar:

- patches y coordenadas;
- salida del patch embedder;
- salida de las capas 0 y 15;
- salida del pooler;
- 280 embeddings proyectados a 1536.

**Salida:** artefacto reproducible con formas, hashes y tolerancias fijadas antes
de implementar CUDA.

### V1 — Mapper visual y HNF

- Añadir selección de mapper por arquitectura **y modalidad**.
- Implementar `Gemma4VisionMapper` para los 659 tensores.
- Mantener pesos y clamps en FP16 para el primer HNF de referencia.
- Extender los hints de visión de forma versionada, sin reinterpretar campos
  CLIP existentes.
- Hacer que `--text DIR --vision DIR` produzca bloques disjuntos.

**Pruebas:** 600 tensores de texto y 659 de visión, cero nombres duplicados,
cero cruces entre bloques y rechazo del contrato incompleto.

**Salida:** HNF texto+visión inspeccionable, todavía sin ejecutar visión.

### V2 — Loader y validador de visión

- Leer la extensión visual Gemma 4.
- Validar arquitectura, formas, dtype y los 448 clamps.
- Calcular presupuesto de pesos/scratch antes de reservar VRAM.
- Cargar y descargar el bloque de visión independientemente del texto.

**Salida:** 659/659 tensores validados y bloque visual cargable sin construir
el forward.

### V3 — Preprocesado exacto sin dependencia de UI

- Implementar primero sobre un buffer RGB ya decodificado.
- Resize bicúbico con relación de aspecto y alineación a 48 píxeles.
- Rescale, patchify HWC, padding y coordenadas `(x,y)`.
- Dejar la decodificación PNG/JPEG detrás de una interfaz; no bloquear la
  correctitud del motor eligiendo todavía una librería de imagen.

**Pruebas:** coincidencia exacta de tamaños/coordenadas y tolerancia numérica
del resize/patchify contra el oráculo en varias relaciones de aspecto.

### V4 — Encoder visual FP16

Implementar una ruta `Gemma4VisionRunner` separada del `GraphBuilder` de texto:

- patch projection y posición aprendida 2D;
- clamp de entrada/salida para las siete lineales por capa;
- cuatro normas, Q/K norm, V norm sin peso;
- RoPE 2D y atención no causal con escala 1,0;
- GeGLU y residuales;
- pooling 3 × 3, escala y proyector 768→1536.

**Pruebas:** comparar cada frontera de V0. La salida final debe conservar
argmax/correlación y cumplir tolerancias FP16 acordadas, sin NaN.

### V5 — Puente multimodal al decoder

- Expandir `<image_soft_token>` al número real de salidas visuales.
- Sustituir embeddings principales en las posiciones de imagen.
- Alimentar PLE con una copia de IDs donde imagen se reemplaza por PAD=0.
- Hacer prefill con el KV actual y continuar decode textual sin volver a
  ejecutar la torre visual.

**Pruebas:** prompt solo texto idéntico bit a bit; prompt con imagen compara
logits prefill y varios pasos greedy contra la referencia.

### V6 — Producto mínimo y robustez

- Una imagen desde CLI con error claro de formato/tamaño.
- Probar imágenes cuadradas, verticales y panorámicas.
- Medir VRAM máxima, tiempo de visión, prefill y decode.
- Verificar carga repetida, dos turnos y liberación de memoria.

**Salida:** descripción coherente de imágenes y cero regresiones de texto.

### V7 — Optimización posterior

Solo después de V6:

- carga bajo demanda o residencia permanente según medida;
- cuantización selectiva comparada contra el HNF visual FP16;
- kernels/fusiones y más tamaños de soft tokens;
- varias imágenes y vídeo reutilizando la torre.

## Primera acción segura

La primera edición de código será V1: hacer que el conversor distinga modalidad
y añadir un mapper visual que solo inventaría nombres canónicos y hints. No se
tocarán kernels ni el forward de texto. Antes de convertir pesos se exigirá una
prueba de inventario 659/659; así el fallo actual de `--vision` queda bloqueado
sin arriesgar el HNF de producción.

## Estado actual

- **Estado:** pausado antes de V0. No se ha iniciado código de visión.
- **Motivo:** primero se cierra la línea de decode/calidad y se congela un HNF
  de texto de producción. Así cualquier desviación multimodal se compara contra
  un decoder estable y no contra una cuantización que todavía cambia.
- **Última comprobación:** checkpoint local con 659 tensores visuales y 321,47
  MiB BF16; todos los límites de clipping son finitos.
- **Condición para reanudar:** perfil ponderado del conversor aceptado o
  descartado mediante la barrera de producción; baseline, HNF y velocidad de
  texto registrados.
- **Primera acción al reanudar:** crear el oráculo mínimo y fijar las salidas de
  una imagen sintética antes de implementar `Gemma4VisionMapper`.
