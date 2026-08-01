# Gemma 4 E2B Vision — oráculo V0

> Cerrado el 2026-08-01. Este informe fija la referencia numérica antes de
> convertir pesos o escribir kernels de visión en Héctor.

## Referencia

- Checkpoint: `google/gemma-4-E2B-it` local.
- `model.safetensors`: 10.246.621.918 bytes, SHA256
  `2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`.
- `config.json`: SHA256
  `1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330`.
- Implementación oficial: Transformers
  `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`.
- Script: `tools/gemma4_vision_oracle.py`.

El oráculo carga únicamente `model.vision_tower.*` y
`model.embed_vision.*`: 658 + 1 = **659 tensores**. No instancia el decoder,
la torre de audio ni el modelo completo.

Fuentes fijadas:

- [`modeling_gemma4.py`](https://github.com/huggingface/transformers/blob/b3a36037d3feb22e3f0174b3dd4248fcc0f0f722/src/transformers/models/gemma4/modeling_gemma4.py)
- [`image_processing_gemma4.py`](https://github.com/huggingface/transformers/blob/b3a36037d3feb22e3f0174b3dd4248fcc0f0f722/src/transformers/models/gemma4/image_processing_gemma4.py)
- [`processing_gemma4.py`](https://github.com/huggingface/transformers/blob/b3a36037d3feb22e3f0174b3dd4248fcc0f0f722/src/transformers/models/gemma4/processing_gemma4.py)

## Entorno dorado

La referencia se ejecutó dos veces en CPU/FP32 con:

- Python 3.14.4;
- PyTorch 2.9.1+debian, CPU, 16 hilos;
- Transformers 5.15.0.dev0 construido desde el commit fijado;
- NumPy 2.3.5 y safetensors 0.8.0;
- atención `eager`, algoritmos deterministas y TF32 desactivado.

Los pesos BF16 del checkpoint se promueven exactamente a FP32. El entorno vive
en `~/.cache/helios`; no añade dependencias a Héctor ni modifica Python del
sistema. El comando lógico es:

```bash
python tools/gemma4_vision_oracle.py \
  /ruta/a/gemma-4-E2B-it \
  --device cpu --dtype fp32 --output-dir /ruta/vacia/oracle
```

La herramienta rechaza un directorio de salida que ya contenga archivos para
no mezclar ejecuciones.

## Fixture y resultados

La fixture RGB se genera por fórmula entera dentro del script. Mide 960 × 672,
ya está alineada a 48, ocupa 2520 patches y produce 280 soft tokens. Esto aísla
el forward visual del resize, que se validará con varias relaciones de aspecto
en V3.

| Frontera | Forma | RMS FP32 | SHA256 `.npy` |
|---|---:|---:|---|
| fixture RGB | `[672,960,3]` | 147,378202 | `a3750585882437650f050b8f5b8293920c49248e268fd9ce711581cc7b5c44bb` |
| patches | `[1,2520,768]` | 0,577954 | `da633a7b2a2b0626800d811470dc32a5f0899f68d58f53aba752fe0dd39e7eb9` |
| posiciones XY | `[1,2520,2]` | 29,473152 | `d621b20dd425eb21a1a1ea3ee147701bc7655bce193c73f6d2c15907837b7e5a` |
| patch embedder | `[1,2520,768]` | 0,764737 | `45ff71d589f26c7dbc55ae6ca946bc8ebbda998a1c429843a140d78bdea0c20f` |
| capa 0 | `[1,2520,768]` | 3,279222 | `3f2361cf11c4a0448357d1fe91d2552957aa20c2d1d560e6ac60904914bc4442` |
| capa 15 | `[1,2520,768]` | 41,519270 | `ae8429cc072dc659ccddd8e095df98aef1fb6e4fb1d7e1fe91ba128621ec4dec` |
| pooler 3 × 3 | `[1,280,768]` | 926,852668 | `3b498690f8ab734c203d485c5ec663501e6bc746263d287fcf506fb9183c39e5` |
| proyección a texto | `[280,1536]` | 0,959750 | `2882fa64fbcb83523e23d629fad1949e26192116adce314c7e8860d8ca281515` |

Todos los valores son finitos. El forward tardó 191,21 y 190,75 segundos. Los
ocho `.npy` y el manifiesto estable —excluyendo únicamente tiempos y ruta
absoluta del checkpoint— fueron idénticos byte a byte entre ambas ejecuciones.
Los volcados ocupan 34 MiB por ejecución y permanecen fuera de Git en
`~/.cache/helios/gemma4_vision_oracle/`.

## Barrera numérica para V3/V4

Las formas deben coincidir exactamente y toda salida debe ser finita. Para una
referencia `r` y una implementación `x`, se aplana el tensor y se calcula
`NRMSE = RMS(x-r) / RMS(r)` junto a la correlación de Pearson.

| Frontera | NRMSE máximo | Correlación mínima |
|---|---:|---:|
| patches FP32 | `1e-7` de error absoluto máximo | — |
| posiciones XY | igualdad exacta | — |
| patch embedder FP16 | 0,005 | 0,99999 |
| capa 0 FP16 | 0,015 | 0,9998 |
| capa 15 FP16 | 0,030 | 0,9990 |
| pooler FP16 | 0,030 | 0,9990 |
| proyección FP16 | 0,030 | 0,9990 |

Estas tolerancias no certifican todavía la conversación multimodal. V5 deberá
comparar además logits de prefill y varios pasos greedy contra upstream.

## Veredicto

**V0 cerrada.** El contrato oficial es ejecutable, selectivo y reproducible.
La siguiente fase puede limitarse al mapper/HNF visual: todavía no necesita
tocar kernels, el grafo de texto ni los HNF certificados.
