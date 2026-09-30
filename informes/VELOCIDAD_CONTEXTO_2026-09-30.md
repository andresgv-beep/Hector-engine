# Velocidad por longitud de contexto: Héctor (HQS v4) frente a llama.cpp (Q4_K_M)

Referencia para afinar la atención de contexto largo **sin volver a medir desde cero**.
Medido el 30/09/2026 en la RTX 4070 Ti de 12 GB (60 SM, límite 285 W).

## Protocolo

- **Script:** `hqs_v4/lab/engine/velocidad_contexto.py`. **Resultados** (JSON, monitor de GPU y cachés de
  autotune usadas): `hqs_v4/results/velocidad-contexto-2026-09-30/`, commit `6bcbc51` de `hqs_v4`.
- Mismos prompts renderizados con la plantilla de Gemma 4 para los dos motores: 45, 6.189, 12.205 y 20.045
  tokens de entrada. 128 tokens de salida, temperatura 0, ventana de 24.576 (la de producción).
- Un calentamiento y mediana de tres repeticiones por contexto. Orden alternado: prod, llama, fresca, lab52,
  lab52, fresca, llama, prod (cuatro tandas por motor). Nada más en la GPU (Helios parado).
- **Héctor:** `build/helios_runtime` de `main`, con `gemma4_12b_hqs_v4.hnf` (7,45 GB), compilado con CUDA 12.6.
- **llama.cpp:** `Hector-q4km-reference/build/bin/llama-server`, `-ngl 99 -fa on -b 512 -ub 256`, con
  `gemma-4-12b-it-Q4_K_M.gguf` (7,12 GB), compilado con CUDA 13.1 (necesita su `LD_LIBRARY_PATH`).
- Velocidad = la de decode que informa cada motor (sin prefill).
- GPU durante la prueba: mediana de 75 °C, 2.760 MHz en los núcleos, 10.251 MHz en la memoria y 265 W;
  la mitad del tiempo, en su límite de potencia. Afecta igual a los dos motores.

## Resultados (tok/s)

| Contexto | Héctor v4 | llama.cpp Q4_K_M | Diferencia |
|---:|---:|---:|---:|
| 45 | 52,39 | 57,75 | −9,3 % |
| 6.189 | 47,51 | 54,47 | −12,8 % |
| 12.205 | 46,06 | 53,73 | −14,3 % |
| 20.045 | 44,32 | 53,00 | −16,4 % |

Mismo resultado que las rondas del 27/09 y el 28/09, así que es estable. Las cifras más bajas medidas por
GPT (Héctor 44,5 y llama.cpp 48,3 en corto) bajaban en los dos motores por igual: era el entorno.

## En milisegundos por token

| | Héctor | llama.cpp |
|---|---:|---:|
| Contexto corto (45) | 19,09 ms | 17,32 ms |
| A 20.045 | 22,56 ms | 18,87 ms |
| **Lo que cuesta el contexto (45 → 20k)** | **+3,48 ms** | **+1,55 ms** |

- **Parte fija, 1,77 ms.** Unos 0,8 ms salen del tamaño: v4 lee un 4,6 % más de bytes por token que el
  Q4_K_M. El resto es eficiencia de los GEMV y del resto del paso de decode.
- **Parte que crece con el contexto.** Héctor paga **2,2 veces** lo que paga llama.cpp. Ahí está el margen:
  si igualara a llama.cpp, a 20k pasaría de ≈44 a ≈50 tok/s.

## Autotune de GEMV: descartado como causa

Tres cachés: la de producción (`hexos-prueba-12b-hq42/.helios/tune.cache`), la de la ronda de 52 tok/s
(`hqs_v4/results/speed/speed-v4-1-v2`) y una vacía recalculada en frío. Eligen variantes distintas para
varias formas (vocabulario A o C, o_proj B o C, down A o B), pero la velocidad queda **a menos del 0,2 %**
en todos los contextos. Las variantes empatan y el ruido del cronómetro decide, sin efecto práctico.

Queda una incoherencia menor: las formas `hq53k` (atención y vocabulario) se cronometran con la L2 caliente,
y las `hq44k` en frío. No cambia la velocidad medida; si algún día se toca, que sea para que el resultado
salga igual en cada arranque, no para ganar velocidad.

## Por dónde atacar el contexto largo

Antecedente: [atención de decode distribuida](OPTIMIZACION_ATENCION_2026-09-19.md) (19/09), con 256 bloques
por encima de 2.048 tokens. Solo ganó ≈3 % de mediana a 6k. Repartir más no es la palanca principal.

Orden de magnitud: a 20k, la lectura mínima del KV por token es del orden de 0,7 ms en esta GPU. Sale de
las 8 capas globales (1 cabeza KV × 512, K = V) y de las 40 locales (ventana de 1.024). llama.cpp paga
1,55 ms y Héctor 3,48 ms: los dos están por encima de ese mínimo, y Héctor bastante más.

Hipótesis, por comprobar con perfilado (Nsight Systems/Compute, a 6k, 12k y 20k):

1. **`attention_k_eq_v` en las capas globales.** ¿Se lee el mismo vector dos veces (como K y como V), o se
   guarda duplicado en el KV? Si es así, se puede leer una sola vez.
2. **Accesos al KV sin coalescer, o con reducciones por warp demasiado pequeñas** para HD512.
3. **La fase de combinación** del kernel distribuido: coste fijo por capa que crece con los bloques.
4. **Kernels pequeños entre capas** (normas, RoPE, copias al KV) cuyo coste aparece solo con contexto.

Criterio de éxito: bajar los **+3,48 ms** del contexto hacia los **+1,55 ms** de llama.cpp, repitiendo este
mismo protocolo con el mismo script, sin tocar la salida a temperatura 0 salvo diferencias de redondeo
justificadas.
