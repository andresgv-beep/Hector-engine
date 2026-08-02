# Pipeline genérico de pesos mapeados

## Propósito

`MappedWeightPipeline` separa la política de residencia de la arquitectura del
modelo. Un runner multimodal no administra CUDA streams, eventos ni ventanas de
VRAM: únicamente declara qué prefijo de tensores usa ahora y cuál usará a
continuación.

El mismo contrato sirve para una torre visual, un encoder de audio, expertos
MoE respaldados por RAM u otros bloques HNF procesados por fases.

## Contrato del adaptador

```cpp
MappedWeightPipeline weights(loader, engine, BLOCK_VISION,
                             engine.config().stream);

{
    MappedWeightPhase phase(weights, "vision.layer0.", &error);
    if (!phase) return false;

    // Encolar aquí todos los kernels que consumen vision.layer0.*

    if (!phase.prefetch("vision.layer1.", &error)) return false;
    if (!phase.close(&error)) return false;
}
```

La guarda RAII restaura los punteros originales aunque el runner salga antes
por error. `prefetch()` debe llamarse después de encolar los kernels actuales:
el pipeline registra en ese punto cuándo queda libre la ventana activa.

No hay nombres de Gemma, visión ni número de capas dentro del pipeline. El
adaptador puede construir su lista de fases desde metadatos HNF o desde su
contrato de arquitectura.

## Modos

Sin variables, o con `HELIOS_MMAP_DOUBLE_BUFFER=0`, se conserva el backend V7:
una transferencia y una ventana reutilizable en el stream de cómputo.

Con:

```bash
HELIOS_MMAP_DOUBLE_BUFFER=1
```

el pipeline crea bajo demanda:

- dos ventanas dimensionadas al máximo realmente visto;
- un stream CUDA no bloqueante para copias;
- eventos `ready` y `released` por ventana.

El stream de cómputo espera `ready[N]` antes de usar una fase. La copia de
N+1 espera `released[N-1]` antes de reutilizar su ventana. Los punteros del
`TensorRegistry` solo se sustituyen al activar una fase y nunca durante su
prefetch.

En un bloque residente en VRAM ambas rutas son no-op: el mismo runner funciona
sin bifurcar código.

## Requisitos del HNF

Para que una fase pueda copiarse con una sola operación:

1. sus tensores deben estar registrados como `file_mapped`;
2. deben compartir bloque HNF;
3. sus nombres deben tener un prefijo canónico común;
4. el layout determinista debe mantener el rango de la fase contiguo.

Los huecos internos se copian como parte del rango, por lo que no afectan la
corrección pero sí deben evitarse en el mapper para no gastar ancho de banda.
Una mezcla de tensores residentes y mapeados bajo el mismo prefijo se rechaza.

## Cómo añadir otra arquitectura

1. El conversor emite nombres canónicos y agrupa cada fase físicamente.
2. El loader de la modalidad registra el bloque como file-backed.
3. El runner crea un `MappedWeightPipeline` para su `BlockID`.
4. Cada fase usa `MappedWeightPhase` y anuncia la siguiente.
5. Se certifica primero con una ventana y después con doble buffer.

No hay que duplicar asignaciones, streams, eventos ni lógica de fallback.

## Estado certificado

Gemma 4 Vision es el primer consumidor. En la RTX 5070 Laptop:

- una ventana: 97,74 ms calientes y +334 MiB de pico;
- doble buffer: 73,25 ms y +356 MiB;
- pesos residentes: 68,31 ms y +626 MiB.

El doble buffer recupera 24,49 ms manteniendo las cinco fronteras numéricas V4,
el hash de texto y 18/18 tests.

La infraestructura es genérica, pero hoy solo el bloque visual tiene una ruta
de carga file-backed. Cada modalidad futura deberá añadir esa decisión de
residencia a su loader antes de consumir el pipeline.
