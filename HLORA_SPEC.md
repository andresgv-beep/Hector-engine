# HLORA — formato de adaptadores LoRA de Helios

| | |
|---|---|
| Versión | 1.1 (borrador) |
| Estado | Diseño. Nada implementado todavía. |
| Extensión | `.hlora` |
| Compatibilidad | HNF v9.1, Gemma 4 12B (texto) |
| Anterior | v1.0, enero de 2026 (`HLORA_SPEC.txt`, en el NAS) |

## Historia

- **v6 (HNF):** el LoRA iba dentro del propio `.hnf`, en otro bloque.
- **v1.0 (enero de 2026):** se sacó a un archivo aparte, `.hlora`. Decisión correcta, que se mantiene.
- **v1.1 (30/09/2026):** puesta al día para HQS v4 y Gemma 4. Cambios respecto a v1.0:
  1. El LoRA **no se suma a los pesos**: se aplica aparte en cada capa (§7). Con pesos HQS v4 de 4 bits, sumar y restar obliga a recuantizar y va acumulando error.
  2. Nombres de tensores del HNF 9.1 (`text.layerN.…`), no `layerN.…` (§5).
  3. `alpha` entero → `scale` en float32, ya calculada (§3).
  4. El flag `IS_QLORA` desaparece: QLoRA cuantiza el modelo base al *entrenar*, pero el adaptador sale en FP16/BF16 (§8).
  5. El tipo de dato de cada tensor va en la tabla binaria, no solo en el JSON (§4).
  6. La compatibilidad se comprueba con `base_id`, la huella del modelo **original**: vale para cualquier cuantización de ese modelo y rechaza cualquier otro, aunque tenga las mismas formas (§6.1). v1.0 usaba el SHA del `.hnf` entero: 7,5 GB de lectura y ligado a una sola cuantización.
  7. Particularidades de Gemma 4: capas de atención global sin `v_proj` (§5.1).

## 1. Filosofía

Los adaptadores viven **fuera** del HNF principal porque son:

- **Modulares:** varios LoRA para un mismo modelo base.
- **Intercambiables:** se montan y desmontan en marcha, sin recargar el modelo.
- **Pequeños:** de decenas a cientos de MB (§5.2).
- **Opcionales:** el modelo base funciona igual sin ellos.
- **Exactos:** el modelo base (HQS v4) no se toca. El LoRA se aplica con su precisión completa.

```
gemma4_12b_hqs_v4.hnf     ← modelo base, 7,45 GB, nunca se modifica
├── tono.hlora            ← «compañero, no mayordomo»
├── herramientas.hlora    ← reglas de uso que hoy van en el prompt
└── nocturno.hlora        ← HLORA nocturno (ver CK_V4_NORTE.md en hexos-core)
```

## 2. Estructura del archivo

```
[0x00] CABECERA              64 bytes
[0x40] TABLA DE TENSORES     N × 32 bytes
       DATOS DE TENSORES     cada tensor alineado a 256 bytes
       MANIFIESTO JSON       al final
```

Todos los enteros son little-endian.

## 3. Cabecera (64 bytes)

| Offset | Bytes | Campo | Tipo | Descripción |
|---|---|---|---|---|
| 0x00 | 8 | magic | char[8] | `b"HLORA\0\0\0"` |
| 0x08 | 2 | version_major | uint16 | 1 |
| 0x0A | 2 | version_minor | uint16 | 1 |
| 0x0C | 4 | flags | uint32 | §3.1 |
| 0x10 | 4 | tensor_count | uint32 | Número de entradas de la tabla (A y B cuentan por separado) |
| 0x14 | 4 | rank | uint32 | Rango del LoRA (1–512) |
| 0x18 | 4 | scale | float32 | Factor final que multiplica B·A. Con LoRA clásico, alpha/rank; con rsLoRA, alpha/√rank. El conversor lo calcula desde `adapter_config.json` |
| 0x1C | 4 | header_size | uint32 | Siempre 64 |
| 0x20 | 8 | tensor_table_off | uint64 | Siempre 64 en v1.1 |
| 0x28 | 8 | manifest_offset | uint64 | |
| 0x30 | 8 | manifest_size | uint64 | |
| 0x38 | 8 | file_size | uint64 | |

v1.0 guardaba `alpha` como uint32 en 0x18. Los entrenamientos actuales usan alpha decimal y distintas fórmulas de escala, así que se guarda el resultado.

### 3.1 Flags

| Bit | Nombre | Descripción |
|---|---|---|
| 0 | RESERVED | Era `IS_QLORA` en v1.0. Debe valer 0 (§8) |
| 1 | HAS_BIAS | Incluye bias LoRA |
| 2 | ATTN_ONLY | Solo proyecciones de atención (informativo) |
| 3 | MLP_ONLY | Solo MLP (informativo) |
| 4 | ALL_LAYERS | Todas las capas (informativo) |
| 5–31 | RESERVED | 0 |

Los bits 2–4 son informativos: lo que manda es la tabla de tensores.

## 4. Tabla de tensores (32 bytes por entrada)

| Offset | Bytes | Campo | Tipo | Descripción |
|---|---|---|---|---|
| 0x00 | 4 | name_hash | uint32 | FNV-1a de 32 bits del nombre completo (§5) |
| 0x04 | 2 | kind | uint16 | 0x01 = A, 0x02 = B, 0x04 = bias A, 0x08 = bias B |
| 0x06 | 2 | dtype | uint16 | 1 = FP16, 2 = BF16, 3 = FP32 |
| 0x08 | 8 | offset | uint64 | Offset absoluto de los datos, múltiplo de 256 |
| 0x10 | 8 | size | uint64 | Bytes |
| 0x18 | 4 | rows | uint32 | |
| 0x1C | 4 | cols | uint32 | |

v1.0 tenía `flags` de 4 bytes y el tipo solo en el JSON. Ahora el motor carga sin leer el manifiesto; el JSON queda para personas y herramientas.

- A tiene forma `[rank, in_features]`; B tiene forma `[out_features, rank]`. Filas contiguas (row-major).
- El conversor comprueba que no hay dos nombres con el mismo hash. Si ocurriera, falla; no se resuelve en el motor.

## 5. Nombres de tensores

Nombre base = el del tensor en el HNF 9.1 sin `.weight`. Se añade `.lora_A` o `.lora_B`:

| Tensor HNF | LoRA A | LoRA B |
|---|---|---|
| `text.layer{N}.attn.q_proj.weight` | `text.layer{N}.attn.q_proj.lora_A` | `text.layer{N}.attn.q_proj.lora_B` |
| `text.layer{N}.attn.k_proj.weight` | `…k_proj.lora_A` | `…k_proj.lora_B` |
| `text.layer{N}.attn.v_proj.weight` | `…v_proj.lora_A` | `…v_proj.lora_B` |
| `text.layer{N}.attn.o_proj.weight` | `…o_proj.lora_A` | `…o_proj.lora_B` |
| `text.layer{N}.mlp.gate.weight` | `…mlp.gate.lora_A` | `…mlp.gate.lora_B` |
| `text.layer{N}.mlp.up.weight` | `…mlp.up.lora_A` | `…mlp.up.lora_B` |
| `text.layer{N}.mlp.down.weight` | `…mlp.down.lora_A` | `…mlp.down.lora_B` |

En v1.1 solo se adapta la torre de texto. Embeddings, PLE, normas, visión y audio no.

### 5.1 Gemma 4 12B

Datos del `config.json`:

- 48 capas: 40 de atención local (*sliding*) y 8 globales (*full*, una de cada seis).
- `hidden_size` 3840 e `intermediate_size` 15360.
- Capas locales: 16 cabezas de consulta y 8 de clave/valor, de dimensión 256.
- Capas globales: 16 cabezas de consulta y 1 de clave/valor, de dimensión 512.

| Proyección | Capa local [out, in] | Capa global [out, in] |
|---|---|---|
| q_proj | [4096, 3840] | [8192, 3840] |
| k_proj | [2048, 3840] | [512, 3840] |
| v_proj | [2048, 3840] | **no existe** |
| o_proj | [3840, 4096] | [3840, 8192] |
| mlp.gate / mlp.up | [15360, 3840] | [15360, 3840] |
| mlp.down | [3840, 15360] | [3840, 15360] |

**`attention_k_eq_v`:** las capas globales no tienen `v_proj`; V sale de la misma proyección que K (`graph_builder.cpp`). Un LoRA sobre `k_proj` en esas capas cambia K y V a la vez, igual que en el entrenamiento. El motor debe aplicar la corrección a la salida de `k_proj` **antes** de reutilizarla como V. Un `v_proj.lora_*` en una capa global es un error de conversión.

### 5.2 Tamaño esperado (FP16)

| Rango | Destino | Parámetros | Tamaño |
|---|---|---|---|
| 16 | q, k, v, o | 21,3 M | ≈ 43 MB |
| 16 | las 7 proyecciones | 65,6 M | ≈ 131 MB |
| 64 | las 7 proyecciones | 262 M | ≈ 525 MB |

Son cuentas a partir de las formas de §5.1, no archivos medidos. Rango 16 en todo son unos 65 M parámetros, frente a unos 12.000 M del modelo: menos del 1 % de trabajo extra por token.

## 6. Manifiesto JSON

```json
{
  "format": "HLORA v1.1",
  "adapter": {
    "name": "tono",
    "description": "Habla como un compañero técnico, no como un mayordomo",
    "author": "Andrés",
    "version": "1.0.0"
  },
  "config": {
    "rank": 16,
    "alpha": 32.0,
    "scale_rule": "alpha/rank",
    "scale": 2.0,
    "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down"],
    "bias": "none"
  },
  "base_model": {
    "family": "gemma4",
    "name": "gemma-4-12b-it",
    "base_id": "hbase1:3f9c…",
    "source": "google/gemma-4-12b-it",
    "text_layers": 48,
    "hidden_size": 3840
  },
  "training": {
    "tool": "unsloth",
    "method": "qlora",
    "epochs": 3,
    "learning_rate": 2e-4,
    "dataset": "helios-conversaciones-corregidas",
    "examples": 800
  },
  "tensors": [
    { "name": "text.layer0.attn.q_proj.lora_A", "shape": [16, 3840], "dtype": "fp16", "offset": 4096, "size": 122880 }
  ]
}
```

- `base_id` empareja el LoRA con su modelo (§6.1). `source` es solo informativo: el nombre o la ruta que traía `adapter_config.json`.
- El formato del modelo base (HQS v4, estable…) da igual: el LoRA se entrena contra el modelo original en BF16 y se aplica sobre cualquier cuantización de ese mismo modelo. Si un LoRA se ajustó sobre una cuantización concreta, se indica en `training`.

### 6.1 `base_id`: la huella del modelo original

Un LoRA solo vale para los pesos exactos con los que se entrenó. Otro modelo con las mismas formas (un reentrenamiento como el 12B de Fable) cargaría sin error y daría basura, así que la huella tiene que depender del **contenido** de los pesos originales, no de las formas ni del nombre.

```
base_id = "hbase1:" + SHA-256 de, en este orden:
  1. "hbase1\n"
  2. por cada tensor del checkpoint original, ordenados por nombre original:
       nombre \n  dtype \n  forma separada por comas \n  SHA-256 de sus bytes crudos
  3. SHA-256 del tokenizer.json
```

- **Se calcula al convertir.** El conversor ya lee todos los tensores del safetensors; hashearlos por el camino cuesta unos segundos sobre los ~9 minutos de conversión.
- **Va en el manifiesto del `.hnf`** (`"base_id"`). Todas las cuantizaciones del mismo modelo (estable, HQS v4…) llevan el mismo, porque sale de los pesos de origen y no de los cuantizados.
- **No depende de cómo se repartieron los shards** (`model-0000x-of-0000y`): se hashea tensor a tensor, no archivo a archivo.
- **`helios-convert-lora` lo copia** del `.hnf` que se le indica con `--base`. Si además recibe `--source` (la carpeta del modelo original), lo recalcula y comprueba que coincide antes de escribir el `.hlora`.
- **Héctor compara** el `base_id` del `.hlora` con el del `.hnf` cargado. Si no coinciden, o alguno no lo tiene, se niega a cargarlo.
- **Los `.hnf` actuales no lo llevan.** Se añade en la próxima conversión, o con una herramienta que lo calcule desde el modelo original y lo escriba en el manifiesto.

## 7. Aplicación en el motor

### 7.1 Carga

1. Cargar el `.hnf` base, igual que ahora.
2. Leer la cabecera y la tabla del `.hlora` y validarlo (§9).
3. Comprobar la compatibilidad: `base_model.base_id` (§6.1), y que las formas de A y B casan con los tensores base.
4. Subir A y B a la GPU tal cual (FP16/BF16). Son pocos MB, así que no hace falta mapearlos desde disco como los pesos grandes.

### 7.2 Cálculo: sin fusionar (cambio respecto a v1.0)

Para cada proyección con LoRA:

```
y = W·x + scale · B·(A·x)
```

- `W·x` es el kernel de siempre (HQS v4, GEMV al generar, dequant+cuBLAS en el prefill).
- `A·x` produce un vector de tamaño `rank` (16–64); `B·(…)` lo lleva a `out_features`. En el prefill son dos GEMM finos; al generar, dos GEMV pequeños.
- La suma puede ir en el epílogo del kernel base o en un kernel aparte. Empezar aparte, más simple de validar, y fusionar solo si el perfilado lo pide.
- Prueba de referencia: la misma capa con el LoRA fusionado en FP32 sobre pesos descuantizados, con tolerancia fijada antes de medir (como `test_hqs_v4.cu`).

v1.0 hacía `W_merged = W + (alpha/rank)·B·A` sobre los pesos. Con HQS v4 eso exige descuantizar, sumar y recuantizar: se pierde parte del LoRA en el redondeo y hacen falta minutos y 7,5 GB por cambio.

### 7.3 Cambio en caliente

Cambiar de adaptador es cambiar punteros: el modelo base no se ha tocado, así que no hay nada que «desaplicar». v1.0 restaba el LoRA anterior de los pesos, lo que con pesos cuantizados acumula error en cada cambio.

- **Apagar el LoRA** (para comparar con el modelo tal cual) = no sumar el término.
- **Tras cambiar de LoRA, el KV de la conversación ya no vale:** se calculó con otro adaptador. Hay que resetear la sesión, o se mezclan dos modelos.

### 7.4 Varios adaptadores a la vez (futuro)

Matemáticamente es sumar sus términos, `Σ scaleᵢ · Bᵢ·(Aᵢ·x)`. En v1.1 se admite **uno activo** a la vez. Combinar necesita pruebas de calidad, no solo de cálculo.

### 7.5 Fusión fuera de línea (opcional)

`helios-merge-lora` produce un `.hnf` nuevo con el LoRA dentro. Descuantiza los pesos afectados, suma B·A y los recuantiza con el conversor (HQS v4). Solo para distribuir un modelo cerrado. Se pierde algo de calidad por la recuantización, así que hay que medir KL, como con HQS v4.

## 8. QLoRA

v1.0 definía QLoRA como «adaptador cuantizado a HQ4K». No es eso: QLoRA es **entrenar** con el modelo base cuantizado para que quepa en la GPU. El adaptador resultante sale en FP16/BF16 y pesa poco, así que en v1.1 no se cuantizan A ni B: `dtype` es siempre FP16, BF16 o FP32. El bit 0 de flags queda reservado.

## 9. Validación

Un `.hlora` es válido si:

- [ ] magic es `b"HLORA\0\0\0"`, `version_major` es 1 y `header_size` es 64;
- [ ] `0 < rank ≤ 512` y `scale` es finito y distinto de 0;
- [ ] `tensor_count > 0`, y cada `lora_A` tiene su `lora_B` (y viceversa);
- [ ] A = `[rank, in]` y B = `[out, rank]`, con `in` y `out` iguales a los del tensor base;
- [ ] ningún `v_proj.lora_*` apunta a una capa con `attention_k_eq_v`;
- [ ] los offsets son múltiplos de 256, no se solapan y caben en el archivo;
- [ ] `manifest_offset + manifest_size == file_size`;
- [ ] los hashes de nombre son únicos y coinciden con `tensors[].name` del manifiesto;
- [ ] `base_id` coincide con el del `.hnf` cargado, y ambos existen. Si no, el motor se niega a cargarlo; no lo intenta «a ver qué sale».

## 10. Herramientas

**Convertir un LoRA de HuggingFace/PEFT** (`adapter_model.safetensors` + `adapter_config.json`):

```bash
helios-convert-lora --input ruta/al/lora --base ruta/al/modelo.hnf --output tono.hlora
```

- Traduce los nombres de PEFT, del estilo `…layers.{N}.self_attn.q_proj.lora_A.weight`, a los de §5. El prefijo exacto se fija al escribir el conversor, contra un adaptador real de Gemma 4.
- Calcula `scale` según `adapter_config.json` (`lora_alpha`, `r`, `use_rslora`).
- Lee la forma de cada tensor base del `.hnf` y rechaza lo que no case.

**Fusionar** (opcional, §7.5):

```bash
helios-merge-lora --model base.hnf --lora tono.hlora --output fusionado.hnf
```

## 11. Qué falta para usarlo

0. `base_id` en `helios_convert_v9.1`: calcularlo al convertir y escribirlo en el manifiesto del `.hnf` (§6.1).
1. Conversor PEFT → `.hlora` (Python o Rust, junto a `helios_convert_v9.1`).
2. Cargador en Héctor: cabecera, tabla, validación y subida a GPU.
3. Kernel del término `scale·B·(A·x)` para prefill y generación, con prueba de referencia.
4. Opción en `helios_runtime` para montar o quitar un `.hlora`, y reset de sesión al cambiarlo.
5. Entrenar un primer LoRA pequeño (rango 16) sobre conversaciones de Helios corregidas y medirlo con `hexos-core/tools/banco_temperatura.py`: herramientas 19/21 como mínimo y muletillas serviles por respuesta.
