# Reproducción: acumuladores compactos HD256

Base anterior: `333b7e5`. CUDA 13.1, sm_89, mismas opciones `-O3 --use_fast_math`.
El benchmark se enlaza estáticamente con cada versión de `libhelios-engine.a`.
En **ambos** modos se usa `use_coalesced_prefill=true`; comparar contra `false`
mediría también la optimización anterior y exageraría esta mejora.

1. Compilar la base en un checkout separado y la variante en el checkout actual.
   No cambiar de revisión sobre un árbol con modificaciones locales.
2. Enlazar `benchmark.cpp` dos veces, con las respectivas bibliotecas. Ejemplo
   desde cada raíz del motor (ajustar el Toolkit y nombre de salida):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 6
c++ -O3 -std=c++17 -I src -I kernels -I /ruta/cuda/include \
  /ruta/a/benchmark.cpp build/libhelios-engine.a -L /ruta/cuda/lib64 \
  -Wl,-rpath,/ruta/cuda/lib64 -lcudart -lcublas -lcublasLt -lpthread -ldl \
  -o /ruta/benchmark-before
```

3. Con la GPU libre de otras inferencias, ejecutar secuencialmente:

```bash
python3 comparar.py /ruta/12b.hnf /ruta/benchmark-before /ruta/benchmark-after /ruta/resultados 12b
python3 comparar.py /ruta/e4b.hnf /ruta/benchmark-before /ruta/benchmark-after /ruta/resultados e4b
python3 resumir.py /ruta/resultados > /ruta/resultados/resumen.json
```

El script no detiene servicios. Cada forma tiene calentamiento (trial 0) y una
muestra (trial 1) en cada uno de tres pares AB/BA/AB. Se usa el mismo `home/tune.cache`
para todos; en la campaña registrada se sembró con el cache de la prueba de prefill
anterior, conservado aquí como `tune.cache`. Otro hardware necesita su propio cache.

El resumen exige 128 tokens, cero fallbacks de CUDA Graph y respuestas idénticas
byte a byte en las seis ejecuciones medidas por longitud y modelo. No mide carga
de pesos, HTTP, Python, renderizado ni latencia total desde el clic del usuario.

Para corrección del motor:

```bash
ctest --test-dir build --output-on-failure -j 1
HELIOS_EMBED_MMAP=1 HELIOS_VISION_MMAP=1 build/test_attention_prefill_model /ruta/modelo.hnf
HELIOS_EMBED_MMAP=1 HELIOS_VISION_MMAP=1 build/test_inference_session_gemma4 /ruta/modelo.hnf
python3 tests/test_runtime_cancellation.py build/helios_runtime /ruta/12b.hnf
```

Repetir las dos pruebas con pesos con 12B y E4B. Para sesiones del E4B
multimodal, añadir `--long-ring`: la entrada corta no sobrescribe su anillo
reservado para lotes visuales. La prueba sintética de prefill
incluye el lote visual de 6144 queries; no equivale a una nueva certificación de
calidad de respuestas visuales. Las mediciones GPU no deben ejecutarse en paralelo.
