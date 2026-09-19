# Descompresión HQ4.2/HQ5.2 con escrituras vectorizadas

Base: `93afc47`; RTX 4070 Ti, CUDA 13.1, Release, sm_89, `--use_fast_math`.
La base ya incluye la mejora de atención HD256. No desactivar esa mejora para
medir este cambio: ambos benchmarks reciben `1` como segundo argumento.

## Comparación completa

1. Compilar `93afc47` en un checkout independiente y la variante en el árbol actual.
2. Enlazar `benchmark.cpp` estáticamente con cada biblioteca, usando el mismo
   compilador y Toolkit (sustituir rutas):

```bash
c++ -O3 -std=c++17 -I src -I kernels -I /ruta/cuda/include \
  /ruta/a/benchmark.cpp build/libhelios-engine.a -L /ruta/cuda/lib64 \
  -Wl,-rpath,/ruta/cuda/lib64 -lcudart -lcublas -lcublasLt -lpthread -ldl \
  -o /ruta/benchmark-before
```

3. Reservar la GPU para una inferencia a la vez. Los scripts no paran servicios.
   Sembrar `/ruta/resultados/home/tune.cache` con el cache conservado aquí solo
   si corresponde al mismo hardware; otro equipo debe usar su propio autotune.

```bash
python3 comparar.py /ruta/12b.hnf /ruta/benchmark-before /ruta/benchmark-after /ruta/resultados 12b
python3 resumir.py /ruta/resultados 12b > /ruta/resultados/resumen.json
```

Tres pares AB/BA/AB, calentamiento y muestra por longitud: 41, 2089, 6185 tokens.
El resumen exige 128 tokens generados, cero fallbacks de grafos y respuestas
idénticas byte a byte. Las medidas son del motor; no incluyen HTTP ni UI.

## Kernel aislado y corrección

El microbenchmark incluye el archivo de implementación para comparar directamente
las dos especializaciones, escalar y vectorizada, de la misma versión. No enlazar
además `libhelios-engine.a`. Desde la raíz del repositorio:

```bash
/ruta/cuda/bin/nvcc -O3 --use_fast_math -std=c++17 -arch=native -I kernels \
  informes/matmul-cuantizado-2026-09-19/microbenchmark.cu -lcublas \
  -Xlinker -rpath -Xlinker /ruta/cuda/lib64 -o /tmp/microbenchmark-hqs
/tmp/microbenchmark-hqs
ctest --test-dir build --output-on-failure -j 1
build/test_inference_session_gemma4 /ruta/12b.hnf
python3 tests/test_runtime_cancellation.py build/helios_runtime /ruta/12b.hnf
```

`test_hqs_symmetric` compara también GEMM contra expansión CPU, tanto directamente
como mediante CUDA Graph, para K divisible y no divisible entre ocho. Se conserva
la ruta escalar para el segundo caso. El microbenchmark usa pesos sintéticos y
comprueba los valores FP16 bit a bit; no sustituye la prueba con el modelo real.

`stores.json` registra el ensamblador sm_89: ocho `STG.E.U16` en la variante
escalar y una `STG.E.128` en la vectorizada. No es una promesa para otros Toolkit.
