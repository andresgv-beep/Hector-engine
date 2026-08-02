# Gemma 4 Vision V8 — perfil para decidir doble buffer

Fecha: 2026-08-02

## Veredicto

El doble buffer **sí merece un prototipo opt-in**. La torre V7 dedica unos
`28,95 ms` calientes a copiar fases visuales y la GPU puede solapar esas copias
con el cómputo. La estimación conservadora es recuperar `24–25 ms` por imagen,
dejando la torre en aproximadamente `74–76 ms` frente a `99,34 ms` actuales y
`68,31 ms` con todos los pesos residentes.

El coste previsto es otra ventana CUDA de 32 MiB: el pico visual pasaría de
`+336 MiB` a aproximadamente `+368 MiB`, todavía `258 MiB` por debajo de los
`+626 MiB` del modo VRAM.

## Protocolo

Mismo HNF combinado, imagen de 1234×1104, contexto 4096 y tres ejecuciones de
la torre dentro del mismo proceso. La primera se descarta por inicialización
fría de cuBLAS y page cache.

```bash
HELIOS_VISION_MMAP=1 HELIOS_EMBED_MMAP=1 HELIOS_CTX=4096 \
HELIOS_VISION_REPEAT=3 \
/usr/local/cuda/bin/nsys profile --trace=cuda --sample=none \
  --cpuctxsw=none --output=/tmp/gemma4_v8_staging \
  build/gemma4_vision_chat modelo-combinado.hnf imagen.png \
  "Describe brevemente la imagen." 1 0
```

Control sin staging:

```bash
HELIOS_VISION_MMAP=0 HELIOS_EMBED_MMAP=1 HELIOS_CTX=4096 \
HELIOS_VISION_REPEAT=3 \
  build/gemma4_vision_chat modelo-combinado.hnf imagen.png \
  "Describe brevemente la imagen." 1 0
```

## Resultados calientes

| modo | repetición 2 | repetición 3 | media |
|---|---:|---:|---:|
| pesos visuales en VRAM | 68,388 ms | 68,228 ms | **68,308 ms** |
| staging V7 | 99,259 ms | 99,429 ms | **99,344 ms** |
| diferencia | | | **31,036 ms** |

Nsight separó las 18 copias de pesos de cada torre caliente:

| fase | cantidad | tiempo total medio |
|---|---:|---:|
| patch embedder, 32,637 MB | 1 | 2,849 ms |
| capas, 18,881 MB cada una | 16 | 25,926 ms |
| proyector, 2,359 MB | 1 | 0,172 ms |
| **total staging** | **18** | **28,947 ms** |

Una capa copia en torno a `1,62 ms`. Entre el final de una copia de capa y el
inicio de la siguiente hay `4,251 ms` de cómputo de media (`4,242–4,262 ms`).
Por tanto, desde la capa 1 existe más del doble de ventana de cómputo que de
transporte. La copia inicial del patch no se puede ocultar por completo y fija
el suelo por encima del modo con todos los pesos residentes.

## Prueba de capacidad de solapamiento

Se ejecutó además una sonda CUDA temporal con una copia pageable host→device
de 18,881 MB y trabajo GPU independiente en dos streams. Usó las mismas
advises host-preferred/accessed-by que el mmap visual. Resultado:

```text
pageable=1 concurrent=1 async_engines=2
copy=1,652 ms  compute=2,201 ms
sequential=3,832 ms  overlap=2,165 ms  saved=1,667 ms
```

La copia quedó completamente escondida incluso detrás de solo 2,2 ms de
cómputo. Las capas reales ofrecen unos 4,25 ms. La sonda se eliminó después de
medir; no forma parte del motor.

## Diseño mínimo del prototipo

1. Dos ventanas reutilizables de 32 MiB, no una reserva por capa.
2. Un stream de copia y eventos `ready/released` por ventana.
3. La capa N espera `ready[N]` en el stream de cómputo.
4. La copia de N+1 espera que su ventana haya sido liberada por N−1.
5. Los punteros del registro solo cambian al activar una fase ya copiada.
6. Sin `HELIOS_VISION_DOUBLE_BUFFER=1`, la ruta V7 queda byte por byte igual.

No se adoptará por defecto hasta superar:

- fronteras V4 idénticas;
- tres repeticiones calientes sin carrera ni corrupción;
- decode dentro del 1 %;
- pico real compatible con los 64 MiB totales de staging;
- 18/18 tests y carga/descarga sin memoria no recuperada.

## Decisión

El perfil autoriza implementar el prototipo. El techo no son los 31 ms
completos: patch inicial, eventos y preparación permanecen en la ruta crítica.
El objetivo de aceptación es ahorrar al menos `20 ms` calientes. Por debajo de
eso se retira el doble buffer y se conserva la V7 simple.
