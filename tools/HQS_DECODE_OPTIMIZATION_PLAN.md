# Campaña de optimización HQS decode — medición antes/después

Campaña ejecutada el 2026-07-31. El objetivo fue acelerar el `decode` real de
Héctor sin mezclar cambios de calidad, formato ni memoria. El perfil obligó a
medir HQ4.1K y el `lm_head` HQ5.1K como un mismo recorrido.

## Alcance y reglas

- Motor base: commit `600f4200e3f353d59eb1d694f7eb0eaac436cdcc`.
- Modelo principal: `qwen3_4b_final.hnf`, 3.470.618.923 bytes, SHA256
  `c82c9fb86f4513a07f6f567181cff1579019be95bc93b2fc2c83deb4123f8969`.
- GPU: NVIDIA GeForce RTX 5070 Laptop, 8.151 MiB.
- Compilador CUDA: `/usr/local/cuda/bin/nvcc`.
- Durante esta campaña no se introducen HQ3.1K, embeddings HQ4, cambios de
  sampling ni refactors ajenos. Así, cualquier diferencia de velocidad se
  puede atribuir al cambio medido.
- Cada candidato se prueba por separado. Solo se integra el ganador.

## Protocolo reproducible

### Medición end-to-end

Comando base:

```bash
HELIOS_EMBED_IN_RAM=1 ./build/test_generate_kv \
  /home/andres/Documentos/GitHub/helios_convert_v9.1/output/qwen3_4b_final.hnf \
  "raw:La ingeniería de software requiere medir cuidadosamente cada cambio porque" \
  256 0
```

Condiciones fijas:

1. equipo conectado a corriente y mismo perfil de potencia;
2. registrar temperatura, potencia y reloj SM antes y después de cada pasada;
3. ejecutar una pasada de calentamiento y excluirla;
4. medir al menos tres pasadas calientes consecutivas;
5. publicar mediana y rango, no la mejor cifra;
6. exigir 256 tokens greedy y la misma secuencia de tokens antes y después;
7. mantener `HELIOS_EMBED_IN_RAM=1`, contexto, prompt y CUDA Graph sin cambios.

### Línea base end-to-end

La primera pasada arrancó con la GPU fría y limitada de reloj, por lo que no
entra en la mediana estable:

| Pasada | Decode | Reloj al terminar | Temperatura | Uso |
|---|---:|---:|---:|---|
| Calentamiento | 67,7249 tok/s | 1.305 MHz | 59 °C | excluida |
| Caliente 1 | 103,184 tok/s | 2.467 MHz | 72 °C | válida |
| Caliente 2 | 103,101 tok/s | 2.475 MHz | 74 °C | válida |
| Caliente 3 | 103,018 tok/s | 2.437 MHz | 75 °C | válida |
| Caliente 4 | 102,894 tok/s | 2.460 MHz | 76 °C | válida |

Línea base de potencia alta: mediana **103,0595 tok/s**, rango
**102,894–103,184 tok/s**.

El portátil alterna además un régimen sostenido de 48–56 W y 1.207–1.275 MHz.
Seis muestras válidas dieron una mediana de **67,126 tok/s**, con rango
**64,794–67,779 tok/s**. No se mezclan ambos regímenes.

Un cambio debía superar **108,21 tok/s** para acreditar +5 % end-to-end en
potencia alta. Las diferencias por debajo del 2 % se consideran ruido salvo un
A/B emparejado o evidencia de microkernel consistente.

## Línea base de los kernels

`tools/bench_gemv.cu` se compiló sin modificar con:

```bash
/usr/local/cuda/bin/nvcc -O3 -arch=native \
  -o /tmp/hector_bench_gemv_baseline tools/bench_gemv.cu
```

Ancho de banda efectivo observado, en GB/s:

| Forma | A 2x8 | A 4x4 | B 2x8 | B 4x4 | B 1x16 | B 2x16 |
|---|---:|---:|---:|---:|---:|---:|
| 4B gate/up 2560→9728 | 329 | 289 | 353 | 325 | 355 | 317 |
| 4B down 9728→2560 | 327 | 316 | 344 | 339 | 342 | 313 |
| 4B attention Q 2560→4096 | 299 | 266 | 329 | 299 | 331 | 277 |
| 8B gate/up 4096→12288 | 352 | 351 | 365 | 364 | 363 | 321 |
| 8B down 12288→4096 | 348 | 345 | 358 | 361 | 358 | 320 |
| 4B attention K/V 2560→1024 | 205 | 215 | 242 | 223 | 273 | 240 |
| 8B attention K/V 4096→1024 | 228 | 251 | 269 | 283 | 300 | 277 |
| 4B attention O 4096→2560 | 307 | 304 | 330 | 330 | 321 | 294 |

Los GEMV grandes ya alcanzan 329–365 GB/s, cerca del techo práctico medido de
la GPU. El margen más visible está en K/V estrechos y en pérdidas entre kernels
encadenados: en el banco actual varían aproximadamente entre 4 % y 28 % según
la forma.

Importante: las etiquetas históricas de `bench_gemv.cu` llaman «producción» a
A 2x8, pero el kernel actual ya usa lecturas globales directas y autoajuste de
varias configuraciones. Antes de decidir nada, el banco debe ejecutar la ruta y
las configuraciones exactas que hoy selecciona el motor.

### Banco corregido y perfil real

El banco se corrigió para incluir las tres configuraciones reales de producción
(`4x4`, `2x8`, `4x1`), microsegundos por llamada y candidatos separados. Nsight
Systems con `--cuda-graph-trace=node` confirmó que HQ4.1K supone **74,9 %** del
tiempo GPU del recorrido perfilado y que producción selecciona principalmente
`2x8`. Los GEMV grandes quedan a 329–365 GB/s: apenas existe margen sin reducir
los bytes leídos.

Para HQ5.1K se añadió `bench_gemv_hq51.cu`. Resultado emparejado dentro del
mismo proceso:

| Forma | Staging 2x8 | Directo 2x8 | Diferencia |
|---|---:|---:|---:|
| Qwen3-4B lm_head | 851–852 μs | 823–824 μs | **−3,4 %** |
| Qwen3-8B lm_head | 1.318–1.320 μs | 1.313–1.316 μs | **−0,4 %** |

La lectura directa HQ5.1K elimina staging, memoria compartida y dos
sincronizaciones por superbloque. Además de ser más rápida, deja un kernel más
sencillo.

## Resultado antes/después

Se probó primero el paquete de dos candidatos medidos:

- HQ4.1K `1x16` solo para K/V 2560/4096→1024;
- HQ5.1K con lectura global directa.

En potencia alta, cinco muestras dieron mediana **103,938 tok/s** frente a
**103,0595 tok/s**: **+0,85 %**. El A/B más cercano bajo el mismo estado fue
102,935→103,602 tok/s: **+0,65 %**. En el régimen de 55 W el cambio no se
separó del ruido del control dinámico de potencia.

El `1x16` mejora K/V de 7,09 a 6,22 μs (−12,3 %), pero esas proyecciones pesan
tan poco en el token completo que añade una ruta especial por menos de un punto
porcentual. Se descartó.

Se conserva únicamente HQ5.1K directo y la retirada del espacio de staging que
HQ4.1K ya no utilizaba. No se atribuye una mejora end-to-end certificada al
cambio retenido: su efecto esperado es inferior al 1 %, aunque el micro A/B de
HQ5.1K es estable y el código resultante es más simple.

Validación del cambio retenido:

- suite completa: **13/13**;
- Qwen3-4B: bloque greedy de 64 tokens idéntico, SHA256
  `7a7a67b5c6458cf21c29487d94842c8b84843346ac97e8280cac0ac1fa5acb99`;
- Gemma 4 E2B compacto: respuesta greedy idéntica entre commit base y cambio,
  SHA256 `eef962bf0f956e640fd715c20d6b2cdb637abf022b48ba94e7927b2735ced825`;
- HNF Gemma real: prefill/decode `max=0`, `mean=0`, argmax 42;
- CUDA Graph continúa activo con 507 nodos en Qwen y 942 en Gemma.

## Plan por fases

### 0. Congelar el antes

- [x] Fijar commit, modelo, tamaño y hash.
- [x] Fijar prompt, longitud, greedy y residencia de embeddings.
- [x] Medir el banco histórico sin modificarlo.
- [x] Obtener dos pasadas calientes estables de Qwen3-4B.
- [x] Añadir muestras calientes y guardar mediana y rango definitivos.

### 1. Hacer fiel el banco de HQ4.1K

- [x] Añadir al banco las configuraciones exactas del kernel de producción.
- [x] Informar también microsegundos por llamada, no solo GB/s.
- [x] Comprobar contra una referencia numérica y detectar lecturas fuera de
  rango antes de comparar velocidad.
- [x] Repetir las formas candidatas y comprobar estabilidad.

### 2. Perfilar un decode real y estable

- [x] Capturar un tramo de CUDA Graph replay de Qwen3-4B.
- [x] Ordenar kernels por tiempo acumulado y número de invocaciones.
- [x] Separar coste de GEMV HQ4.1K, HQ5.1K, atención y normalización.
- [x] Ponderar las formas del microbenchmark por su coste real por token.

### 3. Probar candidatos aislados

Orden inicial, sujeto al perfil:

1. configuración especializada para salidas estrechas K/V (`1x16` es el
   primer candidato, no una conclusión);
2. ajuste de warps/bloques por forma y eliminación de configuraciones que el
   autoajuste elija por ruido;
3. reducción de pérdidas por dependencia o lanzamiento entre GEMV encadenados;
4. fusión de operaciones solo si el perfil demuestra que compensa la mayor
   complejidad.

Cada candidato vive primero en el banco. No se acumulan varios cambios antes de
medirlos.

### 4. Integrar un ganador

- [x] Exigir mejora repetible en las formas que pesan de verdad en el perfil.
- [x] Mantener equivalencia numérica y límites de memoria.
- [x] Descartar HQ4 `1x16` e integrar únicamente HQ5 directo.
- [x] Recompilar los objetivos afectados.

### 5. Medir el después y comprobar regresiones

- [x] Repetir el protocolo Qwen3-4B y publicar mediana/rango por potencia.
- [x] Comparar una secuencia greedy de Qwen entre dos binarios independientes.
- [x] Ejecutar la suite de tests.
- [x] Repetir la regresión real de Qwen y una generación Gemma 4 E2B.
- [x] Verificar memoria, errores CUDA y la regresión disponible >512; el cambio
  reduce memoria compartida y el HNF real pasa sin errores.
- [x] Confirmar identidad de logits/tokens representativos, ya que no cambia la
  aritmética: Qwen y Gemma producen exactamente las mismas salidas.

## Criterio de salida

El cambio se conserva si cumple todos estos puntos:

- al menos **+5 % de mediana end-to-end** en el recorrido fijado, o una mejora
  menor pero demostrablemente útil en varios modelos sin añadir complejidad;
- secuencia greedy y comprobaciones numéricas sin regresión;
- sin aumento material de VRAM, fallos CUDA ni pérdida en contexto largo;
- resultado reproducible en pasadas calientes, no solo en un pico aislado.

Si el HQ4.1K ya está demasiado cerca del límite de memoria y el resultado total
no llega al 5 %, se documenta la evidencia y la siguiente campaña pasa a
embeddings HQ4 o HQ3.1K. Esos cambios tendrán su propia línea base de calidad y
tamaño, separada de esta optimización de inferencia.
