# Gemma 4 E2B Vision — V2 loader y validador

> Cerrada el 2026-08-02. Esta fase no construye el forward visual ni añade
> kernels; solo establece el contrato de carga seguro de los pesos de V1.

## Resultado

Héctor reconoce `VisionModelConfigBin.encoder_type=4` como Gemma 4 y exige su
extensión `GM4V` de 96 bytes. Un registro ausente, truncado, solapado con
`GM4X`, con versión desconocida o con campos/flags no soportados se rechaza
antes de tocar la VRAM.

El validador independiente comprueba el bloque `vision` completo:

- 659 tensores canónicos exactos, sin ausencias, duplicados ni extras;
- 448 límites de clipping FP16 escalares, legibles, finitos y con rangos
  mínimo/máximo no invertidos;
- formas derivadas de la geometría `GM4V`, dtype FP16 y tamaño físico;
- rangos dentro del bloque y ausencia de solapamientos entre tensores;
- 337.089.408 bytes de pesos;
- máximo de 2.520 patches y 280 soft tokens;
- cota conservadora de scratch de 368.040.960 bytes para el futuro forward.

La carga usa una única reserva contigua y registra los 659 tensores como
vistas. La primera implementación directa —un `cudaMalloc` por tensor— consumía
473.956.352 bytes reales por el coste de 659 asignaciones, pese a contener solo
337.089.408 bytes de datos. La carga contigua mide **337.641.472 bytes** y
recupera todo al descargar: 136.314.880 bytes de VRAM evitados sin modificar un
solo peso.

La cota incluye patches, cuatro streams hidden, Q/K/V, una matriz de atención
FP32 con softmax in-place, gate/up y salida pooled/proyectada. V2 no reserva ese
scratch: lo calcula antes de cargar y lo conserva en `BlockState` para que V4
pueda decidir la estrategia real sin una cifra oculta.

## Fallo real descubierto

Los clamps del checkpoint son escalares safetensors con forma `[]`. El loader
histórico rechazaba toda forma vacía y por ello el HNF visual de V1 nunca habría
podido cargarse aunque sus hints y sus pesos fueran correctos. Una forma `[]`
ahora representa correctamente un escalar de un elemento; el parser sigue
exigiendo que la clave `shape` exista, para no confundir metadatos ausentes con
escalares válidos.

## Evidencia

Prueba sintética incluida en CTest:

```text
PASS: synthetic GM4V contract and rejection
```

Artefacto solo visual de V1:

```text
PASS: real Gemma 4 vision metadata
  tensors=659 clamps=448 weights=337089408 scratch_upper=368040960
PASS: real Gemma 4 vision load/unload
  observed_vram=337641472 unrecovered_after_unload=0
```

El mismo test pasa sobre el HNF combinado texto+visión: `GM4X` y `GM4V`
coexisten, y `load_block(BLOCK_VISION)` carga únicamente los 659 tensores
visuales. `unload_block` devuelve el registro al estado vacío.

Comandos de reproducción:

```bash
cmake --build build -j2 --target test_gemma4_vision_metadata
GEMMA4_HNF=/home/andres/Documentos/GitHub/helios_convert_v9.1/output/gemma4-e2b-it-multimodal.hnf
./build/test_gemma4_vision_metadata \
  "$GEMMA4_HNF" --load
ctest --test-dir build --output-on-failure
```

El HNF solo visual usado al cerrar V2 era un artefacto de certificación. La
reproducción vigente usa el bloque `vision` del combinado canónico anterior.

Regresión completa al cierre: **15/15**. La prueba nueva se ejecuta sin modelo
ni GPU mediante el HNF sintético; el argumento `--load` activa explícitamente
la prueba pesada con el artefacto real.

## Límite de la fase

Que el bloque cargue no significa que la torre visual ejecute. No se ha tocado
el `GraphBuilder` de texto, el preprocesado, el puente de embeddings ni CUDA
visual. El siguiente paso es V3: reproducir el preprocesado oficial desde un
buffer RGB y compararlo contra el oráculo V0.
