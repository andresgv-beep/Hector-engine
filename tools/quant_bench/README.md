# Banco de pruebas de cuantización

Aparato para responder con números a **"¿cuánta calidad pierde nuestro formato,
y cómo estamos frente a los demás?"**. Montado el 2026-07-31; los resultados de
esa sesión están al final.

## La métrica

**% de posiciones donde el modelo elige el mismo token que una referencia fp32.**

Se descartaron dos métricas que engañan:

- **Correlación de logits.** Sobre 262.144 logits la dominan los cientos de
  miles de tokens irrelevantes en los que todo el mundo coincide. Se llegó a
  ver una correlación de 0,98 conviviendo con un argmax equivocado, y otra de
  0,980 con un KL de 0,00001 (o sea, perfecto). No sirve.
- **MSE sobre los pesos.** Se bajó un 32% en MLP y un 59% en atención y el
  modelo solo mejoró +0,6 puntos: el error que se quitaba vivía en pesos que
  apenas influyen en la salida.

KL y solapamiento top-k sí valen, pero el % de aciertos es el que se entiende
sin traducción.

## Reglas de uso (aprendidas a base de equivocarse)

1. **Dos corpus como mínimo, siempre.** Un efecto de +2,2 puntos con p=0,018 en
   un texto resultó ser +0,1 (p=0,95) en otro. Hay dos corpus incluidos, de
   registro distinto a propósito: `corpus_tecnico.txt` y `corpus_narrativa.txt`.
   La misma cuantización da 79,5% en uno y 63,7% en el otro — el número
   **depende del texto tanto como del modelo**.
2. **McNemar sobre pares discordantes**, no comparación de porcentajes. Son
   medidas pareadas sobre las mismas posiciones.
3. **Exigir p < 0,01.** Con p entre 0,01 y 0,05 lo honesto es decir "no
   concluyente" y buscar más muestra.
4. **Anotar `clocks.sm` en cualquier medida de velocidad.** La térmica invalidó
   dos conclusiones antes de que HEXOS mostrara el reloj en el panel.
5. **Aislar el motor de la cuantización** midiendo también el HNF fp16. Si el
   fp16 reproduce la referencia y el comprimido no, el motor está limpio y el
   problema es del formato. Ese control resolvió el susto de esta sesión.

## Piezas

| Fichero | Qué hace |
|---|---|
| `ref_qwen3.py` | Referencia fp32 de Qwen3 en numpy, leyendo el safetensors de HF. No comparte código con el motor. |
| `ref_gemma4.py` | Lo mismo para Gemma 4 (PLE, ventana deslizante, KV compartido, RoPE proporcional). Fórmulas portadas de `modeling_gemma4.py` de transformers. |
| `hqs_sim.py` | Simulador de HQ4.1K/HQ5.1K portado de `src/hqs/{compact,common}.rs`. Permite probar variantes del formato sin escribir Rust ni CUDA. |
| `argmax_all_positions.cu` | Argmax de Héctor en cada posición, ruta genérica (Qwen y demás). |
| `argmax_all_positions_gemma4.cu` | Ídem por la ruta explícita de Gemma 4. |
| `ollama_next_token.py` | Pregunta a Ollama el siguiente token. **Verifica la alineación** comparando `prompt_eval_count` con la longitud del prefijo; descarta la posición si no cuadra en vez de contaminar la medida. |
| `compare_logits.py` | Compara dos volcados de logits: error, correlación, KL, solapamiento top-k. |
| `tok_codec.cpp` | Codificar/decodificar con el tokenizer del propio HNF. |

**El simulador está calibrado**: predijo 87,5% para el perfil de 4 bits y el
HNF real dio 88,0%. Medio punto. Por eso sus proyecciones son creíbles.

## Cómo se corre

```bash
# 1. tokenizar el corpus con el tokenizer del modelo
g++ -std=c++17 -I. -o /tmp/tok tools/quant_bench/tok_codec.cpp \
    build/libhelios-engine.a -L/usr/local/cuda/lib64 -lcudart -lcublas
/tmp/tok modelo.hnf enc < tools/quant_bench/corpus_narrativa.txt > toks.txt

# 2. referencia fp32 (numpy; necesita numpy en el PATH de Python)
python3 tools/quant_bench/ref_qwen3.py <dir_modelo_hf> "$(cat toks.txt)" ref.bin

# 3. argmax de Hector en cada posicion
nvcc -O2 -std=c++17 -I. -Isrc -Ikernels -o /tmp/allpos \
    tools/quant_bench/argmax_all_positions.cu build/libhelios-engine.a -lcudart -lcublas
/tmp/allpos modelo.hnf "$(cat toks.txt)" hector.bin

# 4. Ollama, si se quiere comparar
python3 tools/quant_bench/ollama_next_token.py modelo.hnf toks.txt ollama.json 500

# 5. simular una variante del formato antes de implementarla
HQS_QMLP=7 HQS_QATT=15 HQS_EMB=15 python3 tools/quant_bench/ref_qwen3.py ...
```

Variables del simulador: `HQS_QMAX` (7=3 bits, 15=4, 31=5) para todas las
capas, o `HQS_QMLP`/`HQS_QATT` por separado, más `HQS_EMB` para la tabla de
embeddings. En `ref_gemma4.py`: `HQS_SIM=1` y `HQS_GROUP`/`HQS_SCALE_BITS`
para variar el layout.

## Resultados del 2026-07-31

**Qwen3-4B, 542 posiciones, referencia fp32 propia:**

| Perfil | Aciertos | Tamaño |
|---|---:|---:|
| Ollama Q4_K_M | 74,2% | 2560 MB |
| HELIOS todo 3 bits (simulado) | 81,2% | 2873 MB |
| **HELIOS MLP 3b + atención 4b + embed 4b** | **84,3%** | **2487 MB** |
| HELIOS todo 4 bits (HNF real) | 88,0% | 2807 MB |
| HELIOS todo 5 bits (simulado) | 92,3% | 6,25 bpw |

McNemar contra Ollama: 4 bits gana 100-25 (p=3,6e-11), 3 bits gana 86-48
(p=1,4e-03). **HQS supera a Q4_K_M a igualdad de bits en las capas** (5,01 vs
4,97 bpw), y el perfil mixto sería más pequeño que Ollama con diez puntos más.

Motivo probable: HQS agrupa los pesos **de 8 en 8** y Q4_K de 32 en 32. Cuatro
veces más resolución en las escalas locales, pagada con cabecera más gorda.

**Gemma 4 E2B-it, 3.082 posiciones de los dos corpus:** motor fp16 al 98,3% y
98,1% (el motor está limpio); compacto HQ4/HQ5 al 69,7% combinado.

**Contexto que evita sustos:** perder entre un 12% y un 30% de las decisiones
de token es lo normal a 4-5 bits. Ollama pierde el 25,8% en la misma prueba.

## Lo que ya se probó y NO funciona

Cuatro variantes del layout, todas peores o irrelevantes:

1. Grupos de 16 con escalas de 6 bits (5,00 bpw en vez de 5,25): **peor**
   (58,5% vs 63,9%). La granularidad del grupo pesa más que la precisión de la
   escala.
2. Encoger `d_scale` para sacrificar el grupo atípico: **peor de forma
   monótona**. El `max` actual es correcto.
3. Búsqueda conjunta con ventana ±1 sobre el redondeo: MSE idéntico, nada.
4. Búsqueda conjunta completa sobre valores representables: −32%/−59% de MSE en
   los pesos pero solo **+0,6 puntos** end-to-end.

Y la rejilla MSE de `optimize_group` quedó apagada por defecto en el conversor:
no mejora la calidad de forma medible (p=0,16 sobre 3.082 posiciones) y cuesta
minutos frente a ~20 s.

**No rediseñar el layout sin medir primero con este banco.** Está en un óptimo
local mejor de lo que parece.

## Siguiente jugada

Detallada en `helios_convert_v9.1/HQ3_PROPUESTA.md`. Resumen del orden:

1. **Cuantizar la tabla de embeddings** — gratis, sin código nuevo: +0,3 puntos
   (p=0,55, sin diferencia medible) y −499 MB.
2. **Confirmar la curva con el segundo corpus** antes de escribir el encoder.
3. **HQ3.1K**: encoder en Rust (3 bits, 96 B de payload → 4,25 bpw) y kernel de
   dequant en CUDA. Cabecera y superbloque sin cambios; el desempaquetado de 5
   bits de `matmul_hqs_compact.cu` ya resuelve el cruce de frontera de byte.
4. Repetir el banco en Qwen3-8B antes de afirmar nada en público.
