# Gemma 4 E2B Vision — V7 residencia y staging por fases

> Cerrada el 2026-08-02. V7 sustituye el intercambio completo de modalidades
> por un único régimen: texto y KV permanecen en VRAM, la torre visual vive en
> el HNF/RAM y solo la fase que se está ejecutando ocupa una ventana temporal
> de VRAM. La ruta V6 íntegramente residente se conserva como control.

## Decisión arquitectónica

Con `HELIOS_VISION_MMAP=1`, el loader:

1. mapea los 321 MiB visuales directamente desde el bloque HNF;
2. aplica `MADV_SEQUENTIAL`/`MADV_WILLNEED` para la caché de páginas;
3. fija residencia preferida en host y acceso desde la GPU mediante HMM;
4. mantiene los 659 tensores registrados sin reservar sus pesos completos en
   VRAM;
5. copia una sola región contigua por fase a una ventana CUDA reutilizable:
   patch embedder, cada una de las 16 capas y el proyector;
6. restaura los punteros mapeados al terminar cada fase y libera la ventana al
   destruir el runner.

La mayor fase es la tabla posicional del patch embedder. La ventana máxima es
de 32 MiB; las capas rondan 18 MiB. El orden determinista del HNF agrupa todos
los tensores de una fase, incluidos los clamps escalares, por lo que cada fase
requiere una copia PCIe contigua y no cientos de transferencias pequeñas.

Texto y KV no se descargan, no se recrean y no cambian de puntero. La torre
visual tampoco entra en el CUDA Graph del decode.

## Alternativas eliminadas

### Lectura HMM directa desde cuBLAS

Correcta numéricamente, pero demasiado lenta:

| modo | torre caliente |
|---|---:|
| pesos en VRAM | ~69 ms |
| GEMM leyendo el HNF/RAM directamente | ~1.180 ms |

La cuenta ideal de leer cada peso una vez no describe el kernel real: cuBLAS
vuelve a consumir tiles de pesos para distintas regiones de `M`, y ese tráfico
repetido cae sobre PCIe.

### Prefetch de todo el `mmap`

`cudaMemPrefetchAsync` sobre el mapeo de fichero no creó residencia observable
en VRAM en esta plataforma: mantuvo `+0 MiB` y la torre siguió en ~1.180 ms.
Esta ruta experimental se retiró; no queda bandera ni código muerto.

## A/B controlado

Misma RTX 5070 Laptop, misma captura 640×473, mismo HNF combinado, acumulación
FP32 de cuBLAS y `HELIOS_CTX=4096`:

| medida | V6: torre VRAM | V7: HNF + ventana | diferencia |
|---|---:|---:|---:|
| carga/registro visual | 72,39 ms | 1,70 ms | −70,69 ms |
| primera torre, cuBLAS frío | 162,80 ms | 194,06 ms | +31,26 ms |
| torre caliente, mediana | 69,15 ms | 102,41 ms | +33,26 ms |
| pesos visuales persistentes | 322 MiB | 0 MiB | −322 MiB |
| pico visual sobre texto+KV | 626 MiB | 336 MiB | −290 MiB |
| prefill | 85,85 ms | 85,79 ms | −0,06 ms |
| decode de 64 tokens | 137,01 tok/s | 136,42 tok/s | −0,43 % |
| ejecución completa, una imagen | 1.586,74 ms | 1.489,79 ms | −96,95 ms |

El coste sostenido de transporte es aproximadamente 33 ms por imagen. La ruta
por ventanas es más rápida de extremo a extremo cuando la alternativa exige
cargar la torre completa, y solo pierde frente a dejar permanentemente 322 MiB
de pesos visuales en VRAM.

Con KV heterogéneo de 4096 posiciones ya reservado antes de visión:

```text
texto cargado                 +1662 MiB
pico de la torre              +336 MiB sobre texto+KV
tras terminar la torre        +1802 MiB total
decoder listo                 +1850 MiB total
pico de prefill               +1886 MiB total
```

Así queda demostrado el caso que motivó V7: una imagen puede procesarse sin
sacar texto ni KV de VRAM.

## Acumulación FP32 y arranque frío

El commit `d635d2b` sustituyó `cublasHgemm` por `cublasGemmEx` con acumulación
FP32. Un A/B temporal, retirado tras medir, separó coste de inicialización y
coste sostenido:

| acumulación | primer uso | caliente | NRMSE proyectado |
|---|---:|---:|---:|
| FP16 anterior | 87,2 ms | 64,3 ms | 0,0055103 |
| FP32 actual | 150–163 ms | 70,2 ms | 0,0041824 |

Los ~80 ms que parecían una regresión permanente son inicialización perezosa
de GemmEx. En HexOS, proceso persistente, se pagan una vez. FP32 se mantiene:
mejora la referencia y solo cuesta unos 5,9 ms en caliente.

## Correctitud

La ruta VRAM y la ruta mapeada producen las mismas métricas contra V0:

| frontera | NRMSE | correlación |
|---|---:|---:|
| patch embedder | 0,0002550303 | 0,9999999676 |
| capa 0 | 0,0004454949 | 0,9999999052 |
| capa 15 | 0,0059976799 | 0,9999820240 |
| pooler | 0,0032544570 | 0,9999947472 |
| proyección | 0,0041823502 | 0,9999912780 |

El test de metadatos verifica además ambos ciclos:

- VRAM: 659 tensores, 337.089.408 bytes lógicos y cero bytes sin recuperar;
- mmap: 659/659 tensores file-backed, `block_vram_usage=0` y cero bytes sin
  recuperar al desmontar el bloque.

La conversación de dos turnos con `HELIOS_CTX=4096` conserva la imagen, ejecuta
la torre una sola vez y mantiene CUDA Graph:

```text
torre fría              194,47 ms
prefill inicial          85,99 ms
decode inicial           24 tokens · 133,16 tok/s
segundo prefill           57,57 ms
segundo decode            24 tokens · 116,40 tok/s
```

## Uso experimental

```bash
GEMMA4_HNF=/home/andres/Documentos/GitHub/helios_convert_v9.1/output/gemma4-e2b-it-multimodal.hnf
HELIOS_VISION_MMAP=1 HELIOS_CTX=4096 \
  build/gemma4_vision_chat "$GEMMA4_HNF" imagen.png \
  "Describe la imagen" 64 0
```

`HELIOS_VISION_REPEAT=N` repite la torre dentro del mismo proceso para separar
el primer uso de cuBLAS de su latencia caliente. La ruta sin
`HELIOS_VISION_MMAP` permanece idéntica y sirve de fallback/control.

Si la GPU no expone acceso pageable y managed concurrente, el modo mmap falla
con un error explícito en vez de fingir que los pesos están fuera de VRAM.

## Límites y siguiente fase

- La copia de la fase siguiente todavía no se solapa con el cómputo actual.
  El techo medido de esa optimización son unos 33 ms por imagen; no justifica
  añadir doble buffering antes de integrarlo en HexOS.
- La selección automática por capacidades y el ciclo de vida del proceso
  pertenecen a HexOS; la variable de entorno es el mando de certificación.
- Cuantización visual, múltiples imágenes, vídeo y audio siguen separados de
  esta decisión de residencia.

V7 queda cerrada con el diseño simple buscado: un decoder residente, un KV
residente y una torre file-backed procesada con una ventana temporal pequeña.
