# Estudio: meter Qwen3.6-35B-A3B en Héctor

2026-11-16. Solo lectura: no se ha descargado ningún peso ni se ha tocado el motor. Leído:
`config.json`, el índice y la cabecera de los safetensors publicados, `chat_template.jinja`, y
`modeling_qwen3_5_moe.py` de transformers (rama main). Motor en `be971d1`.

Acompaña al estudio del Coder (`../estudio-qwen3-coder-2026-11-16`). Este modelo es mejor
agente según las tablas de Qwen (SWE-bench Verified 73,4 contra 51,6 del Coder y 17,4 del
Gemma 4 26B que corre hoy; cifras suyas) y, sobre todo, su caché cuesta la quinta parte.

## El modelo

40 capas de ancho 2.048, en diez grupos de «tres capas Gated DeltaNet y una de atención».
Todas llevan MoE como MLP.

| | |
|---|---|
| Capas DeltaNet (30) | 16 cabezas de clave y consulta, 32 de valor, dimensión 128; convolución causal de 4 |
| Capas de atención (10) | 16 cabezas de consulta, 2 de clave/valor, dimensión 256, con puerta de salida |
| Posiciones | rotatorias sobre un cuarto de la dimensión (64 de 256) |
| MoE | 256 expertos, 8 por token, intermedio 512, más un experto compartido con su puerta |
| Vocabulario | 248.320, embeddings y cabeza sin compartir |
| Visión | ViT de 27 capas, ancho 1.152 |
| Parámetros | 35 mil millones; unos 3 activos por token |
| Pesos publicados | 26 ficheros BF16, 71,9 GB, 1.045 tensores |

De los 35 mil millones, 32,2 son expertos.

## La capa nueva, tal como la calcula la referencia

Por token, en una capa DeltaNet:

1. Cuatro proyecciones del residual normalizado: `in_proj_qkv` (a 8.192), `in_proj_z` (a
   4.096), `in_proj_a` e `in_proj_b` (a 32, una por cabeza de valor).
2. Convolución causal de 4 muestras, canal a canal, sobre las 8.192 de `qkv`, y SiLU. Necesita
   recordar las 3 muestras anteriores.
3. Se parte en consulta (2.048), clave (2.048) y valor (4.096). Consulta y clave se normalizan
   a norma 1 por cabeza, y cada cabeza de clave sirve a dos de valor.
4. `beta = sigmoide(b)`; `g = −exp(A_log) · softplus(a + dt_bias)`.
5. Regla delta, por cabeza de valor, sobre un estado `S` de 128×128:

   ```
   S ← S · exp(g)
   S ← S + k ⊗ ((v − Sᵀk) · beta)
   salida = Sᵀ(q / √128)
   ```

6. Norma RMS por cabeza, multiplicada por `silu(z)`, y `out_proj`.

Es exacta y secuencial: dos productos matriz-vector de 128×128 y una actualización de rango 1
por cabeza. La referencia tiene además una forma por bloques para secuencias largas; da lo
mismo y solo es más rápida.

**El estado es de tamaño fijo**: 32 × 128 × 128 en FP32 son 2,1 MB por capa, 63 MB por sesión
para las 30 capas, sea cual sea el contexto. Más 8.192 × 3 de la convolución.

## Lo que encaja con el motor

- **Los expertos ya vienen empaquetados** como los de Gemma: `experts.gate_up_proj
  [256, 1024, 2048]` y `experts.down_proj [256, 2048, 512]`. El conversor no tiene que apilar
  nada (con el Coder eran 18.432 tensores sueltos).
- **El enrutado es el de `moe_route_kernel`**: softmax, los 8 mayores, renormalizar.
- **512, 1.024, 2.048 y 8.192 son múltiplos de 256**, lo que exige HQ44.
- **La rotación parcial ya existe** (`partial_rotary` en `qk_norm_rope`, puesta para Phi). Con
  solo texto, las tres secciones de posición llevan el mismo índice y queda una rotación normal.
- **`q_norm` y `k_norm`** por cabeza, como Qwen3.
- **El formato de llamadas es el del Coder** (`<function=…>`, `<parameter=…>`): el analizador
  que entró hoy en hexos-core (`a3d3e9e`) sirve tal cual.
- **Hay precedente de estado que se guarda y se restaura**: los anillos de Gemma
  (`save_rings` / `restore_rings`).

## Lo que hay que escribir

### Motor

1. **La capa DeltaNet en decode.** Convolución con su estado, normalización de consulta y
   clave, puertas, el kernel de la regla delta y la norma con puerta. Las cuatro proyecciones
   y `out_proj` son `matmul` de los que ya hay.
2. **Estado por sesión.** Un tensor por capa DeltaNet en lugar de caché. Lo delicado es
   **rebobinar**: hoy una sesión vuelve a una posición anterior (`rewind_to`) para reaprovechar
   el prefijo; de un estado recurrente no se puede deshacer. Hacen falta copias del estado en
   los puntos a los que se vuelve (fin de cada turno): 63 MB cada una.
3. **Prefill.** Primera versión: el mismo kernel recorriendo los tokens en orden, un bloque
   por cabeza. Para 4.096 tokens son 4.096 pasos pequeños por capa; lo estimo en torno a un
   segundo para las 30 capas, sin medir. La forma por bloques, si hace falta, después.
4. **Atención con puerta.** `q_proj` saca el doble: media es la consulta y media una puerta
   cuya sigmoide multiplica la salida antes de `o_proj`.
5. **Experto compartido.** Una MLP densa de 512 sumada a la salida del MoE, por
   `sigmoide(shared_expert_gate · x)`.
6. **Capas de dos tipos en el camino general**, según `layer_types`.
7. **MoE en el camino general** y **reparto entre dos tarjetas fuera de Gemma**: lo mismo que
   pide el Coder (ver su estudio). El reparto sigue siendo lo más gordo.

### Conversor

8. **Mapeador `qwen3_5_moe`.** Tensores nuevos de la capa DeltaNet; los pequeños (`A_log`,
   `dt_bias`, la convolución, las normas) sin cuantizar.
9. **Las normas van centradas en cero**: la referencia multiplica por `1 + peso`. Hay que
   sumarles 1 al convertir, o el modelo sale mudo. Es la trampa clásica de esta familia.

### HexOS

10. **El pensamiento.** La plantilla trae `<think>` y una opción nueva, `preserve_thinking`,
    para conservar el razonamiento de turnos anteriores. HexOS ya separa `<think>`; falta
    decidir qué se hace con lo nuevo.

Fuera de este alcance: la visión (otro adaptador) y el módulo MTP, que solo sirve para
predecir varios tokens y se puede ignorar al generar.

## Cuentas de memoria

Sobre la ficha, sin medir. La caché solo la llevan las 10 capas de atención: 20,5 KB por token.

| Pesos | Ocupan | Contexto 40.000 | 64.000 | 100.000 | 128.000 |
|---|---|---|---|---|---|
| ~4,5 bits | 19,5 GB | 20,3 | 20,8 | 21,6 | 22,1 |
| ~4,0 bits | 17,5 GB | 18,3 | 18,8 | 19,6 | 20,1 |

Con unos 22 GB aprovechables entre las dos tarjetas: **a 4,5 bits, 64.000 holgado y 100.000
justo; a 4 bits, 128.000 con margen.** Pesa más que el Coder (35 contra 30,5 mil millones),
así que la ventaja no es que sobre sitio, sino que cada token de contexto cuesta cinco veces
menos. Con la tabla de embeddings en RAM, como ya se hace con Gemma, se recupera algo más.

Velocidad, a ojo: unos 3 mil millones de parámetros activos por token, como el Coder. La
regla delta en sí es despreciable.

Disco: 71,9 GB de descarga y un HNF de unos 20. En el portátil hay 119 GB libres.

## La buena noticia: hay hermanos pequeños

La familia Qwen3.5 tiene modelos con **la misma capa DeltaNet y la misma atención con
puerta**, densos y pequeños:

| Modelo | Capas | Ancho |
|---|---|---|
| Qwen3.5-0.8B | 24 | 1.024 |
| Qwen3.5-2B | 24 | 2.048 |
| Qwen3.5-4B | 32 | 2.560 |

Caben en una sola tarjeta y, enteros, en la RAM. Eso permite:

- cargar el original en transformers y **comparar los logits capa a capa**, cosa que con los
  72 GB del grande no se puede;
- desarrollar la capa nueva, el estado y el prefill **sin MoE y sin reparto entre tarjetas**.

## Orden propuesto

1. **Qwen3.5-0.8B o 2B de punta a punta**: mapeador, capa DeltaNet, atención con puerta,
   estado y rebobinado, prefill. Comprobado contra la referencia.
2. **MoE en el camino general**, con el experto compartido.
3. **Reparto entre tarjetas** fuera de Gemma.
4. **Convertir el 3.6-35B** y medir cuánto aguanta la cuantización.
5. **Banco de cadenas de HexOS** contra el 26B.

Tras el paso 1 ya hay un modelo de la familia corriendo en Héctor, y el denso de 27B queda a
falta solo del reparto.

## Lo que este estudio no sabe

- Cuánto degrada la cuantización a las proyecciones de la capa DeltaNet, ni si el estado
  recurrente acumula el error a lo largo de un contexto largo. Con caché, un error de
  redondeo se queda en su token; aquí se arrastra.
- Cuánto tarda de verdad el prefill secuencial.
- Si el modelo se porta en HexOS como dicen las tablas.
- El coste del reparto entre tarjetas: dimensionado por el tamaño del código de Gemma.
