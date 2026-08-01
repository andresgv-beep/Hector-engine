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

## LO QUE ESTE BANCO NO VE — leer antes de decidir nada

**El % de aciertos es un filtro, no un veredicto.** Mide fidelidad media en
posiciones con *teacher forcing*: en cada paso se le devuelve el token correcto
al modelo. Eso hace ciego el banco a tres cosas que sí matan un perfil:

1. **Acumulación de error en generación libre.** Aquí cada posición parte de la
   verdad; generando de verdad, los errores se arrastran.
2. **Catástrofes de cola.** Un carácter chino cada 50 turnos es invisible en una
   media del 83%, y sin embargo es inaceptable.
3. **Comportamiento a largo plazo.** Bucles, pérdida de memoria contextual y
   deriva de idioma aparecen a los cientos de tokens, no en una posición suelta.

**Caso real (HQ3.1K, 2026-07-31):** el banco dio 83,0% y el simulador lo había
predicho con 0,27 puntos de error — la medida era correcta. Pero la batería de
conversaciones reales encontró mezcla de idiomas, UTF-8 inválido, tokens
especiales visibles y los controles de memoria cayendo del 80% al 30%. El
formato se rechazó como perfil de producción **después** de pasar este banco.

El orden correcto es: **este banco criba, el A/B con sampling real decide.**
Sirve para descartar barato (así murieron cuatro hipótesis de layout en una
tarde), no para promover. Para lo segundo está `tools/hq31_ab.py`, que corre
perfiles de usuario sintéticos con varias semillas y cuenta artefactos.

**Umbral práctico:** por debajo de ~85% de aciertos, no molestarse en probar el
A/B. Por encima, el A/B manda.

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

**El simulador está calibrado**, con dos puntos de contraste contra HNF reales:

| perfil | predicho | real | desvío |
|---|---:|---:|---:|
| todo 4 bits | 87,5% | 88,0% | +0,5 |
| MLP 3b + attn 4b (HQ3.1K) | 83,29% | 83,02% | −0,27 |

Sus proyecciones son creíbles **para esta métrica**. Que la métrica prediga el
comportamiento en producción es otra cosa: ver la sección de arriba.

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
embeddings. Para aislar el MLP, `HQS_QGATE`, `HQS_QUP` y `HQS_QDOWN`
sobrescriben la familia correspondiente; `HQS_LAYER_START`/`HQS_LAYER_END`
limitan esos overrides a un rango inclusivo de capas. En `ref_gemma4.py`:
`HQS_SIM=1` y `HQS_GROUP`/`HQS_SCALE_BITS` para variar el layout.

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

**HQ3.1K ya está implementado y ya está descartado como perfil de producción
para Qwen3-4B** (detalle en `helios_convert_v9.1/HQ3_PROPUESTA.md`). El encoder,
el layout y el kernel son correctos y el ahorro es real, pero ninguna asignación
probada sobrevive al A/B de conversación. No se tira: puede servir en otro
modelo o en tensores concretos.

### El reparto que sí funciona (medido 2026-07-31 noche)

Un GGUF Q3_K_M que trajo Andrés dio la pista: llama.cpp pone `ffn_gate` y
`ffn_up` en Q3_K pero **`ffn_down` en Q5_K**. Ese punto intermedio no se había
probado — se fue de "todo el MLP a 3 bits" (se rompe) a "solo gate" (limpio
pero 53 MiB). Medido sobre 1755 posiciones de los dos corpus:

| perfil | aciertos | tamaño | vs el actual |
|---|---:|---:|---:|
| todo 4 bits (producción hoy) | 87,7% | 3306 MB | — |
| gate3 + up3 + **down4** + embed5 | 86,4% | 2640 MB | p=0,10 |
| **gate3 + up3 + down5 + embed5** | **88,1%** | **2747 MB** | **p=0,73** |

**559 MB menos con calidad estadísticamente idéntica.** Y confirmado en los dos
corpus por separado (87,7%/88,7% frente a 87,5%/88,1%), no es efecto de uno.

Bajar también `down` a 4 bits ahorra 107 MB más pero pierde 1,3 puntos y cae en
la zona ambigua (p=0,10). No compensa: `down_proj` escribe directo al flujo
residual y ahí el error no se lava, mientras que `gate` y `up` alimentan la no
linealidad y parte del error se pierde por el camino.

**Esto rescata HQ3.1K.** El encoder, el layout y el kernel de GPT son correctos;
lo que estaba mal era a qué matrices se aplicaban. El cambio es de mapper.

Desglose honesto del ahorro, porque no todo es mérito del 3 bits:

| | MB |
|---|---:|
| embeddings fp16 → HQ5.1K | −452 |
| gate + up a HQ3.1K | −224 |
| subir `down` a HQ5.1K (peaje) | +112 |

**Falta el A/B.** 88,1% es métrica de banco, y el banco ya dejó pasar HQ3.1K
una vez. La diferencia es que aquí el perfil es indistinguible de uno que *ya
funciona en producción*, lo que da mucha mejor base de partida que el 83% del
intento anterior. Pero decide `tools/hq31_ab.py`.

### Embeddings duplicados en modelos con pesos atados (2026-08-01)

Qwen3-4B y Gemma 4 tienen `tie_word_embeddings: True`: la tabla de embeddings y
el `lm_head` **son el mismo tensor en origen**. El conversor lo guarda dos veces:

    text.token_embedding.weight    742 MB   fp16    (solo lookup)
    text.lm_head.weight            290 MB   hq51k   (solo matmul final)

llama.cpp guarda uno y lo usa para ambas cosas — está comentado en
`src/llama-quant.cpp`: *"for arches that share the same tensor between the token
embeddings and the output, we quantize the token embeddings with the
quantization of the output tensor"*.

**Ahorro: los 742 MB enteros**, apuntando el lookup al tensor `hq51k` que ya
existe. `launch_embedding_hq51k` ya está en producción (Gemma 4 carga así su
tabla PLE). No hace falta cuantizar nada nuevo ni escribir kernel.

El banco teacher-forced indicaba que la tabla a 5 bits no tenía un coste
medible: era el `embed5` del perfil de arriba. La implementación real confirma
esa parte en 1.755 posiciones (88,09% actual frente a 87,35% compartido,
McNemar `p=0,246`), pero **no supera el A/B conversacional**: en una semilla de
12 turnos apareció texto chino, pérdida parcial de recuerdos y un bucle. Es
otra demostración de que el banco permite decidir qué merece probarse, no qué
entra en producción.

El HNF real baja 741,87 MiB y el decode queda igual, pero
`--shared-embedding-hq51` continúa experimental hasta encontrar una precisión
de entrada que pase producto. No se combina con HQ3.1K mientras falle esa
barrera.

### Ancho de banda: por qué llama.cpp va más rápido

No es el kernel, son los bytes.

| | bpw | metadatos por 256 pesos |
|---|---:|---:|
| Q4_K | 4,50 | 16 B (grupos de 32, escalas de 6 bits) |
| HQ4.1K | 5,25 | 40 B (grupos de 8, escalas de 4 bits) |

Medido en esta máquina: Ollama 116,8 tok/s leyendo ~2309 MB por token → **270
GB/s**. HELIOS ~101 tok/s leyendo ~2564 MB → **259 GB/s**. Ambos rondan el
**70% del techo de 371 GB/s**: la eficiencia de banda es prácticamente igual y
la diferencia de velocidad es casi toda bytes leídos.

Ese 14% extra de bytes es el precio de agrupar de 8 en 8 — exactamente lo que
compra la ventaja de calidad. Es un intercambio deliberado, no un defecto.

### Ajuste ponderado de llama.cpp: SÍ aporta, +2,3 puntos gratis

`make_qkx2_quants` no coge los extremos: para cada asignación tentativa de
índices resuelve por mínimos cuadrados **ponderados por magnitud** la escala y
el mínimo óptimos. Los pesos existen **aunque no haya matriz de importancia**:

    weights[l] = av_x + fabsf(x[l]);                    // sin imatrix
    weights[l] = qw[l] * sqrtf(sigma2 + x[l]*x[l]);     // con imatrix

Portado al simulador (`HQS_WLSQ=1` en `hqs_sim.py`) y medido sobre el perfil de
4 bits:

| corpus | min/max (nuestro) | ponderado |
|---|---:|---:|
| narrativa (1107 pos.) | 87,5% | 88,5% |
| técnico (648 pos.) | 88,1% | 92,4% |
| **combinado (1755 pos.)** | **87,7%** | **90,0%** |

McNemar 115 vs 76, **p=0,006**. Positivo en ambos corpus. **+2,3 puntos a coste
cero**: mismo formato, mismos bits, mismo kernel, misma velocidad. Solo cambia
`compute_group_params` en el conversor.

**Ojo con el MSE, otra vez:** el error sin ponderar apenas baja (−0,6% en MLP,
−3,3% en atención) y aun así el modelo mejora 2,3 puntos. Es la tercera vez en
la sesión que el MSE sobre los pesos no predice nada.

**Predicción fallida documentada a propósito:** se predijo que no aportaría,
razonando que con grupos de 8 los extremos ya describen casi todo el grupo. El
error de razonamiento fue creer que ponderar sirve para *describir* mejor; lo
que hace es *decidir a quién sacrificar*. Con `w = av_x + |x|` los pesos grandes
mandan en la escala y los pequeños aceptan más error, y eso funciona igual con
8 elementos que con 32.

**Falta el A/B** antes de producción, como todo lo demás.

### Heurísticas suyas que no hemos probado

- `use_more_bits(i_layer, n)`: más bits al primer octavo de capas, al último
  octavo y a una de cada tres del medio.
- `ffn_down` de las primeras capas con protección extra, con cicatriz incluida:
  *"Guard against craziness in the first few ffn_down layers"*.

Ambas son reparto, no formato — la misma familia que dio los 559 MB.

### Orden de trabajo

Lo que queda, por orden de relación beneficio/riesgo:

1. **Tabla de embeddings a HQ5.1K.** Hoy va en fp16 (742 MB, 22% del fichero).
   Pasarla a HQ5.1K ahorra **452 MB** y **el kernel ya existe y está en
   producción**: Gemma 4 carga su tabla PLE en `hq51k` con
   `launch_embedding_hq51k`. Es una decisión de mapper en el conversor, sin
   tocar el motor.

   HQ4.1K ahorraría 498 MB, solo 46 MB más, pero exige escribir
   `launch_embedding_hq41k` desde cero. **Empezar por HQ5.1K**: el 91% del
   ahorro, cero código nuevo y más bits de precisión.

   Aviso: la medida de +0,3 puntos (p=0,55) se hizo simulando **HQ4.1K**. HQ5.1K
   solo puede salir igual o mejor, pero conviene confirmarlo con el banco antes
   de darlo por bueno, y pasar el A/B después.

2. **Repetir el banco en Qwen3-8B** antes de afirmar nada en público. Todo lo
   medido hasta ahora es un modelo de 4B.
