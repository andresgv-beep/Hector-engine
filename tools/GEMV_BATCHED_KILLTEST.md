# GEMV batcheado — test de muerte superado (2026-08-01)

Prototipo en `tools/gemv_batched_prototype.cu`. **No es código de producción**:
es la prueba de que la idea funciona antes de integrarla.

## De dónde viene

Fable propuso decodificación especulativa. Su test de muerte —¿cuánto cuesta
verificar M tokens de una pasada frente a M decodes?— dio 3,7× y mató la idea
con los kernels de entonces, porque **ninguno de los dos caminos amortiza la
lectura de pesos con M pequeño**:

- el bucle de GEMVs relee la matriz comprimida M veces;
- el camino cuBLAS la lee una vez pero luego escribe y relee 50 MB de fp16.

Falta el kernel que la física pide: decodificar cada tile **una vez** y
aplicarlo a M vectores, con M acumuladores por hilo y los vectores en shared.
Es lo que hace `mmvq` de llama.cpp para M pequeño.

## Resultado (K=2560, N=9728 — mlp gate de Qwen3-4B)

| M | tiempo | vs M=1 | por token |
|---:|---:|---:|---:|
| 1 | 56,7 µs | 1,00× | 56,7 |
| **2** | **62,2** | **1,10×** | **31,1** |
| 3 | 75,4 | 1,33× | 25,1 |
| 4 | 91,7 | 1,62× | 22,9 |
| 8 | 173,7 | 3,07× | 21,7 |

**Barrera superada**: M=2 en 1,10×. El coste por token cae de 56,7 a ~22 µs.

Contra lo que hay hoy, tomando el GEMV de producción (48,8 µs) como referencia:

    M=4 batcheado    91,7 us
    M=4 hoy         146,4 us   -> 1,6x mejor
    4 decodes       195,2 us   -> 2,1x mejor

Aviso: el M=1 del prototipo (56,7) es un 16% más lento que el de producción
(48,8), porque es genérico en M. Al integrar hay que **conservar el kernel M=1
especializado** y usar el batcheado solo para M≥2.

## Límite físico que hay que resolver al integrar

El vector de entrada va entero en shared: `K * M * sizeof(half)`. El máximo de
shared dinámica por bloque son ~100 KB.

| K | M=1 | M=2 | M=4 | M=8 |
|---|---:|---:|---:|---:|
| 2560 (mlp gate 4B) | 5 KB | 10 | 20 | 40 |
| 12288 (mlp down 8B) | 24,6 KB | 49 | **98** | **no cabe** |

Con K grande hay que trocear en K por tiles en vez de meter el vector entero.
Eso es bastante más kernel: la "tarde de trabajo" son dos o tres días si se
quiere cubrir el 8B.

## Dos clientes independientes

1. **Turnos cortos de chat** (M=5-15): ganan aunque la especulativa no exista.
2. **Verificación especulativa**: con verify(5) ≈ 110 µs frente a 244 de cinco
   decodes, la idea vuelve a la mesa — aunque sigue por encima del criterio de
   1,5× que fijó Fable, así que el margen real depende mucho de la tasa de
   aceptación.

## Bug encontrado de camino

El kernel de producción hacía `__shfl_down_sync(0xFFFFFFFF, ...)` dentro de
`if (lane_id < WARPS_PER_ROW)`: la máscara nombra 32 lanes y solo entran
`WARPS_PER_ROW`. Es UB en Volta+ y funcionaba de milagro porque justo después
el kernel termina. Al añadir un `__syncthreads()` detrás **se cuelga en seco**
— así apareció. Corregido con máscara `(1u << WARPS_PER_ROW) - 1`.
