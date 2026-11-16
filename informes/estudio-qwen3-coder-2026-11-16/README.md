# Estudio: meter Qwen3-Coder-30B-A3B en Héctor

2026-11-16. Solo lectura de código y de la ficha del modelo: no se ha descargado ningún peso
ni se ha tocado el motor. Revisiones leídas: Hector-engine `4d812fb`, helios_convert_v10
`29eb089`, hexos-core `3143438`, y `modeling_qwen3_moe.py` de transformers (rama main).

## Por qué este modelo

El Agente de Helios corre hoy sobre Gemma 4 26B-A4B. Razona bien, pero como agente sobre un
repositorio flojea: repite llamadas rechazadas, da por hecho lo que no ha hecho. Las tablas
publicadas por Qwen dicen lo mismo con números (cifras de los fabricantes, andamiajes
distintos, no comparables al punto):

| SWE-bench Verified | Gemma 4 26B-A4B | Qwen3-Coder-30B-A3B | Qwen3.6-35B-A3B |
|---|---|---|---|
| | 17,4 | 51,6 | 73,4 |

El Coder es el paso corto: su atención ya corre en Héctor. El 3.6-35B es mejor, pero añade
capas Gated DeltaNet, que el motor no tiene. Este estudio cubre el Coder y deja apuntado qué
se reaprovecha para el 3.6.

## El modelo

| | |
|---|---|
| Capas | 48, todas con atención completa |
| Ancho | 2.048 |
| Atención | 32 cabezas de consulta, 4 de clave/valor, dimensión 128, con `q_norm` y `k_norm` |
| MoE | 128 expertos por capa, 8 por token, intermedio 768, activación SiLU |
| Vocabulario | 151.936 |
| Parámetros | 30,5 mil millones; unos 3 mil millones activos por token |
| Pesos publicados | 16 ficheros safetensors BF16, 61,1 GB |

Reparto de los parámetros: los expertos son 29,0 mil millones (el 95 %), la atención 0,9 y
los embeddings 0,6.

## Lo que encaja ya

**La atención es la de Qwen3.** Los tensores de la primera capa son `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `q_norm` y `k_norm`, los mismos que el motor detecta hoy
(`GraphBuilder::detect_architecture`, `use_qk_norm`) y fusiona en `qk_norm_rope`.

**El MoE de referencia es casi el de Gemma que ya hay en `kernels/moe.cu`.** Comparando
`Qwen3MoeSparseMoeBlock` con `append_gemma4_moe_branch`:

| Pieza | Gemma 4 (hoy) | Qwen3 MoE | Cambio |
|---|---|---|---|
| Entrada del enrutador | `rms(x) · escala · factor`, en FP32 | la salida de la norma, tal cual | pasar de FP16 a FP32 sin escalar |
| Logits | lineal sin sesgo, en FP32 | lineal sin sesgo | ninguno |
| Elección | softmax, los K mayores, renormalizar, por `expert_scale` | softmax, los K mayores, renormalizar (`norm_topk_prob`) | `expert_scale` opcional (o unos) |
| Experto | `gelu(gate) · up`, luego `down` | `silu(gate) · up`, luego `down` | activación elegible |
| Disposición | `gate_up [E, 2I, H]`, `down [E, H, I]` | la misma en la referencia actual | ninguno en el motor |
| Combinación | rama MoE **sumada** a una MLP densa, cada una con su norma | el MoE **es** la MLP de la capa | rama propia en `build_mlp_block` |

La renormalización de Qwen (softmax sobre todos y dividir entre la suma de los K) es
exactamente lo que hace `moe_route_kernel`. Las dimensiones cumplen lo que exige HQ44
(768 y 2.048 son múltiplos de 256).

**HexOS ya habla Qwen**: `qwen_protocol.py` y las ramas `self.qwen` de `native.py`
(plantilla `<|im_start|>`, respuestas de herramienta, pensamiento).

## Lo que hay que escribir

### Motor

1. **MoE en el camino general.** Activación elegible en `moe_gate_up_mul_kernel` y en
   `run_moe_experts_reference`; entrada del enrutador sin escalar; `expert_scale` opcional.
   En `GraphBuilder`: detectar el MoE por sus tensores, rama en `build_mlp_block`,
   reserva de trabajo y `validate_weights`. Las dimensiones del MoE viajan hoy en la
   extensión GM4X de Gemma: hacen falta en un sitio que no sea de Gemma.
   *Corto: las piezas existen.*

2. **Reparto entre dos tarjetas fuera de Gemma. Es lo más gordo.** El modelo no cabe en una:
   pasa de 17 GB. Hoy el reparto es solo de Gemma (`gemma4_split.cpp`, 335 líneas, y 46
   referencias en `inference_session.cpp`), y el camino general construye un único grafo con
   todas las capas (`build_forward`) sobre una sola `KVCache`. Hay que poder construir el
   grafo por tramos de capas, pasar el residual de una tarjeta a otra y partir la caché.
   *Medio-largo, y con riesgo: toca el camino por el que corren Qwen3 4B y 8B.*

3. **Prefill.** El MoE por lotes ya existe (agrupa por experto en CPU, fuera del grafo). El
   prefill «flash» de Gemma no aplica. Vale para empezar; la velocidad de prefill se mide
   después.

### Conversor

4. **Mapeador de Qwen3 MoE.** Hoy `factory.rs` manda todo lo que se llama «qwen» a
   `Qwen2Mapper`, que es denso. Los pesos publicados traen cada experto por separado
   (`mlp.experts.N.gate_proj`, `up_proj`, `down_proj`: 18.432 tensores) y el enrutador en
   `mlp.gate.weight`. Hay que apilarlos en la disposición empaquetada que ya se usa con
   Gemma (`gemma4_packed_experts_v1`) y emitir las pistas del MoE.
   *Corto-medio: el empaquetado de Gemma sirve de modelo.*

### HexOS

5. **Formato de llamadas del Coder.** No es el JSON de Qwen que `qwen_protocol.calls`
   entiende. Es de etiquetas:

   ```
   <tool_call>
   <function=nombre>
   <parameter=ruta>
   valor
   </parameter>
   </function>
   </tool_call>
   ```

   Hace falta su analizador y su forma de declarar herramientas. Tiene una ventaja para
   `project_write`: el valor va en crudo, sin escapar como JSON, así que un fichero de código
   entero viaja sin comillas ni `\n`.
   *Corto.*

### Comprobación

6. **Una referencia con la que comparar.** Los 61 GB en BF16 no caben ni en las tarjetas ni
   en los 32 GB de RAM, así que no se puede cargar el modelo entero en transformers. Opciones:
   un recorrido capa a capa con los safetensors mapeados (lento, pero exacto para unos pocos
   tokens), o el GGUF con llama.cpp como referencia floja. Sin esto no se sabe si un fallo es
   del motor o de la cuantización.

## Cuentas de memoria

A partir de la ficha, sin medir. La caché de claves y valores, con las 48 capas en atención
completa, son 98 KB por token.

| Formato | Pesos | Caché a 24.000 | a 32.000 | a 40.000 |
|---|---|---|---|---|
| ~4,5 bits (como HQ44) | 17,2 GB | 2,4 GB → 19,6 | 3,2 GB → 20,4 | 4,0 GB → 21,2 |
| ~5,4 bits (lo que pesa hoy el 26B) | 20,6 GB | 23,0 | 23,8 | 24,6 |

Las dos tarjetas suman 23,5 GB y el escritorio ocupa cerca de 1. **Con el formato del 26B
actual no cabe.** Con uno de 4,5 bits entra a 32.000 con un margen parecido al de hoy. Gemma
tiene capas de ventana deslizante que abaratan su caché; este no.

Velocidad, a ojo: unos 3 mil millones de parámetros activos por token contra los 4 del 26B.
Debería quedar en su orden de magnitud (90 tok/s) o algo por encima, pero son 48 capas con
ocho GEMV pequeños cada una, y ahí manda el número de lanzamientos, no los bytes.

Disco: la descarga son 61 GB y el HNF saldrá en 17-20. En el sobremesa quedan 80 GB libres:
justo. En el portátil hay 119.

## Orden propuesto

1. Analizador de llamadas del Coder en HexOS, con pruebas. No necesita el modelo.
2. Mapeador en el conversor; descargar y convertir a un formato de ~4,5 bits.
3. MoE en el camino general del motor, comprobado capa a capa contra la referencia.
4. Reparto entre tarjetas para el camino general.
5. Banco de cadenas de HexOS con el Coder y con el 26B, mismo historial: ahí se decide si
   la mejora es la de las tablas.

Los pasos 3 y 4 se pueden probar antes con un modelo pequeño que quepa en una tarjeta, si
existe uno de la misma familia; no lo he buscado.

## Lo que sirve después para Qwen3.6-35B-A3B

- El enrutado y los expertos (8 de 256, más uno compartido que es un experto fijo sumado).
- El reparto entre tarjetas fuera de Gemma.
- El analizador de llamadas, si mantiene el formato.

Lo que no: sus 30 capas Gated DeltaNet, con estado por sesión en lugar de caché, y su prefill.

## Lo que este estudio no sabe

- Cuánto degrada un formato de 4,5 bits a un modelo cuyo 95 % son expertos pequeños. Con el
  26B los expertos van en HQ44 y la parte densa en otro formato; aquí casi no hay parte densa.
- Si el modelo se porta en HexOS como dicen las tablas. Ninguna cifra de aquí es nuestra.
- El coste real del reparto entre tarjetas: lo he dimensionado por el tamaño del código de
  Gemma, no leyendo cada camino.
