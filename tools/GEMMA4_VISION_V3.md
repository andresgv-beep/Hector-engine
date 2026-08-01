# Gemma 4 E2B Vision — V3 preprocesado RGB

> Cerrada el 2026-08-02. Esta fase trabaja exclusivamente sobre RGB8 ya
> decodificado. No añade PNG/JPEG, CUDA visual ni conexión con el decoder.

## Implementación

`gemma4_vision_preprocess_rgb()` reproduce el contrato fijado por el procesador
oficial:

1. calcula el mayor tamaño dentro del presupuesto de 2.520 patches;
2. conserva la relación de aspecto y redondea ambos lados hacia abajo a
   múltiplos de `3 × 16 = 48`;
3. ejecuta bicúbico Keys (`a=-0.5`) con antialias sobre RGB8;
4. redondea cada pasada separable a `uint8`, igual que ATen;
5. reescala a FP32 con `1/255`;
6. aplana patches en HWC, genera posiciones `(x,y)` y rellena hasta 2.520 con
   patches cero y coordenadas `(-1,-1)`;
7. devuelve el número dinámico real de soft tokens.

La API admite stride de fila, rechaza punteros, tamaños, strides o geometrías
inválidos y comprueba overflow antes de reservar. La decodificación de imagen
queda fuera mediante `Gemma4RgbView`; una futura UI o CLI podrá usar cualquier
decoder sin cambiar el contrato numérico del motor.

## Oráculo de resize

`tools/gemma4_vision_preprocess_oracle.py` fija:

- Transformers `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722` para la geometría;
- PyTorch `2.9.1+debian`;
- operador `aten._upsample_bicubic2d_aa`;
- entrada y salida del resize en `uint8`.

Los binarios RGB viven fuera de Git en
`~/.cache/helios/gemma4_vision_preprocess_v3/`.

| Fixture | Entrada | Resize | Patches | Soft tokens | SHA256 RGB |
|---|---:|---:|---:|---:|---|
| alineada | 960×672 | 960×672 | 2520 | 280 | `9fda5922…04b69` |
| cuadrada | 512×512 | 768×768 | 2304 | 256 | `3d6da83c…32338` |
| vertical | 333×1000 | 432×1344 | 2268 | 252 | `2fb1a480…37ee` |
| panorámica | 1600×300 | 1824×336 | 2394 | 266 | `f0705a5f…b2fc` |
| extrema | 5000×1 | 13440×48 | 2520 | 280 | `e705a400…709fe` |

Los cinco RGB producidos por C++ coinciden **byte a byte** con ATen: cero
canales distintos, incluido el caso extremo que activa la rama donde una
dimensión redondea inicialmente a cero.

## Continuidad con V0

El caso alineado se comparó también contra los arrays dorados de V0, no solo
contra el nuevo generador:

```text
patches raw SHA256
5c6723b33d7302fd519210b25467948a2cc3d551505bfc28abb46fcf72e30a6c

positions raw SHA256
b82136b2ee1dc891b4e1a9b75362b1fa3375322a18e81e7d93052fa90749dad8
```

Ambos hashes son idénticos entre V0 y V3. Por tanto, el nuevo preprocesador
conserva exactamente el input que produjo las activaciones visuales doradas.

## Pruebas

CTest ejecuta sin archivos externos:

- las cinco geometrías y sus soft tokens dinámicos;
- orden HWC dentro del patch;
- posiciones XY y padding;
- stride de fila no compacto;
- rechazo de entrada inválida.

La comparación pesada opcional es:

```bash
./build/test_gemma4_vision_preprocess \
  /home/andres/.cache/helios/gemma4_vision_preprocess_v3
```

Salida esperada: cinco líneas `resize byte-identical to ATen`.

## Límite y siguiente fase

V3 termina en dos vectores CPU: `[2520,768]` FP32 y `[2520,2]` int32, más el
número de soft tokens. Todavía no se cargan en GPU ni se ejecuta la torre.

V4 construirá `Gemma4VisionRunner` por fronteras: patch projection/posición,
capa 0, capa 15, pooler y proyección. Cada frontera se comparará con V0 antes
de conectar el decoder.
