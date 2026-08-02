# Gemma 4 E2B Vision — V6 producto mínimo y robustez

> Cerrada el 2026-08-02. Esta fase conecta el preprocesado V3, la torre V4 y
> el puente multimodal V5 en una conversación real. No modifica la
> cuantización ni los kernels visuales certificados.

## Resultado

`gemma4_vision_chat` ejecuta ya el recorrido completo:

1. decodifica una imagen PNG a RGB8;
2. aplica el resize, patchify y posiciones de V3;
3. carga la torre visual, ejecuta sus 16 capas, pooler y proyector;
4. conserva únicamente los embeddings visuales proyectados;
5. libera pesos y scratch visuales antes de cargar el decoder;
6. hace el prefill multimodal de V5 y continúa con decode textual normal;
7. opcionalmente añade un segundo turno textual reutilizando el KV, sin volver
   a ejecutar la imagen ni la torre.

La dependencia de `libpng` pertenece solo a esta herramienta. Héctor conserva
su frontera de producto sobre buffers RGB8 sin dependencias, de modo que HexOS
o una UI pueden decodificar otros formatos y entregar los píxeles directamente.

## Compilación y uso

Si CMake encuentra `libpng`, construye el ejecutable automáticamente:

```bash
cmake --build build -j"$(nproc)"
GEMMA4_HNF=/home/andres/Documentos/GitHub/helios_convert_v9.1/output/gemma4-e2b-it-multimodal.hnf
build/gemma4_vision_chat \
  "$GEMMA4_HNF" \
  imagen.png \
  "Describe la imagen" \
  64 0
```

El quinto argumento es `max_tokens` y el sexto la temperatura. Para comprobar
continuidad conversacional:

```bash
HELIOS_VISION_FOLLOWUP="¿Qué colores predominan?" \
  build/gemma4_vision_chat modelo.hnf imagen.png "Describe la imagen" 24 0
```

La herramienta activa `HELIOS_EMBED_MMAP=1` si el usuario no ha fijado un
valor explícito.

## Geometrías y rendimiento medidos

Las medidas se tomaron con el HNF combinado FP16 visual de V1. El pico visual
incluye pesos, scratch y workspace; la columna de decode corresponde al primer
turno greedy.

> Estas latencias son el registro histórico de V6 con `cublasHgemm`. Tras
> `d635d2b`, la acumulación FP32 deja la primera torre en ~163 ms por la
> inicialización perezosa de GemmEx y las siguientes en ~69 ms. V7 mide y
> explica ambas condiciones sin cambiar los resultados funcionales.

| entrada | resize | patches válidos | soft tokens | torre | prefill | decode |
|---|---:|---:|---:|---:|---:|---:|
| 960×672 dorada | 960×672 | 2520/2520 | 280 | ~80 ms | ~79 ms | >119 tok/s |
| 512×512 cuadrada | 768×768 | 2304/2520 | 256 | 80,13 ms | 75,56 ms | 129,96 tok/s |
| 333×1000 vertical | 432×1344 | 2268/2520 | 252 | 84,76 ms | 97,60 ms | 114,48 tok/s |
| 1600×300 panorámica | 1824×336 | 2394/2520 | 266 | 81,04 ms | 77,65 ms | 114,10 tok/s |
| captura real 640×473 | 912×672 | 2394/2520 | 266 | 80,79 ms | 102,11 ms | 126,65 tok/s |

En las cuatro geometrías sintéticas el pico de visión fue `+626 MiB` respecto
al inicio. Tras liberar la torre quedaron aproximadamente `+70 MiB`, que
corresponden al contexto persistente de cuBLAS y a la proyección visual que el
decoder todavía necesita. Con texto cargado el proceso quedó en torno a
`+1788 MiB`, y el prefill alcanzó aproximadamente `+1824 MiB`.

En la prueba de dos turnos, la torre se ejecutó una sola vez:

| operación | medida |
|---|---:|
| torre visual | 79,24 ms |
| prefill inicial | 79,89 ms |
| primer decode | 24 tokens · 131,8 tok/s |
| prefill del seguimiento | 62,14 ms |
| segundo decode | 24 tokens · 116,48 tok/s |

El segundo turno respondió sobre los colores de la imagen anterior, lo que
verifica continuidad del KV y conservación de los tokens visuales sin repetir
el encoder.

## Prueba cualitativa real

Se usó la captura PNG:

`/home/andres/.local/state/codex-desktop/tmp/codex-clipboard-5eeae416-28f1-46ee-8d7e-e0c451859038.png`

SHA256:
`0b72e92a66d8ca1774bc7eb0ef6880e45964b4db47343a7318e4558eadc98b77`.

Ante la petición de describir la captura y leer los nombres, el modelo
identificó una lista de archivos de código y transcribió correctamente
`common.rs`, `compact.rs`, `grid_search.rs`, `hq4k.rs` y `hq5k.rs` antes del
límite de 80 tokens. Esto es una prueba de humo de producto; la correctitud
numérica de torre y decoder sigue respaldada por V4 y V5.

## Robustez

- Un fichero que no es PNG termina con código 1 y el error
  `V6 acepta PNG; formato no reconocido`.
- Un HNF solo textual termina con código 1 y exige explícitamente un combinado
  con `GM4X` y `GM4V`.
- Se rechazan imágenes vacías o superiores a 100 megapíxeles, prompts vacíos,
  parámetros fuera de rango y contratos tokenizer/GM4V incompatibles.
- Las ejecuciones repetidas sobre las cuatro geometrías cargaron, descargaron
  y finalizaron sin error ni crecimiento persistente entre procesos.

Durante la integración apareció una incompatibilidad independiente de visión:
CUDA Graph no puede capturar directamente el lookup de embeddings respaldados
por `mmap` pageable. `Engine::execute_graph_replay()` ejecuta ahora esos
lookups en un prefijo no capturable sobre el mismo stream y captura el resto
del decode. Así se conservan simultáneamente `HELIOS_EMBED_MMAP=1` y CUDA Graph.

## Regresiones

- Héctor: **18/18** pruebas CTest.
- V4 contra el oráculo: se mantienen las cinco fronteras dentro de tolerancia;
  tras acumular en FP32, la capa 15 queda en NRMSE `0,0059977` y la proyección
  en `0,0041824`.
- Texto Gemma 4: logits idénticos al baseline congelado, SHA256
  `e8490bf94b415dc125ca5c58bff2b83f45f7de220283b57dfde95a1b0384a17f`.
- `git diff --check`: limpio.

## Límites deliberados

- Una imagen por primer turno.
- El CLI decodifica PNG; JPEG y otros formatos quedan para la capa de producto.
- Audio, vídeo, lotes y cuantización visual pertenecen a fases posteriores.
- La prueba conversacional del HNF compacto de texto es una barrera separada;
  V6 no mezcla cambios de calidad HQS con la integración de visión.

## Criterio de salida

V6 queda cerrada: existe un camino local de imagen a conversación, maneja las
tres relaciones de aspecto exigidas, conserva la imagen durante un segundo
turno, libera el bloque visual antes del decoder, mide VRAM y latencias, rechaza
entradas incompatibles y no altera las regresiones de texto.
