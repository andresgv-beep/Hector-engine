# Gemma 4 E2B-IT — evidencia parcial de la Fase 8

Mediciones realizadas el 2026-07-31 con `google/gemma-4-E2B-it`, cuyo
`model.safetensors` tiene SHA256
`2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`.
Los dos HNF pasaron el validador estricto con 601 tensores y 0 errores/avisos.

## Artefactos usados en la medición histórica

| Perfil | Archivo | Tamaño | SHA256 | Reparto |
|---|---|---:|---|---|
| Compacto (layout legacy) | `gemma4-e2b-it-text-compact-legacy-layout.hnf` | 4.250.889.374 bytes | `c1e9a31fc1eb59ac5794a69f38f63f046ad2eb0015b55d423ea7d5eb193dca5e` | 283 FP16, 213 HQ51K, 105 HQ41K |
| Referencia (layout legacy) | `gemma4-e2b-it-text-fp16.hnf` | 10.109.254.667 bytes | `f60d335c75ff35775f267f673e4f1afe18dbd6f3e7371108ccac4481d9a70295` | 601 FP16 |

Esos SHA siguen identificando exactamente los ficheros usados en las medidas,
pero su layout nació antes del commit determinista `37610b3` y no se reproduce
con el conversor actual. Los pesos y resultados numéricos no estaban afectados.

## Artefacto compacto recertificado

Dos conversiones independientes con `37610b3` y el mismo checkpoint resultan
idénticas byte a byte:

| Archivo canónico | Tamaño | SHA256 | Validación |
|---|---:|---|---|
| `gemma4-e2b-it-text-compact.hnf` | 4.250.889.522 bytes | `b67df38138b3aac46c7fef0a1fce5b9b74b96ce95aff2bf1c2b6839920b79660` | 601 tensores; 0 errores; 0 avisos |

Héctor lo carga con `HELIOS_EMBED_IN_RAM=1` y el smoke greedy responde
`La capital de Francia es París.`. La referencia FP16 conserva por ahora su
SHA histórico; no se usa como baseline de la fase visual.

Ambos se ejecutaron con `HELIOS_EMBED_IN_RAM=1`. El compacto mantiene fuera de
VRAM 768 MiB del embedding principal y 1.750 MiB del PLE; el FP16 mantiene
fuera 768 MiB y 4.480 MiB respectivamente.

## Calidad compacto frente a FP16

Se volcaron los 262.144 logits del último token con
`gemma4_dump_logits` sobre tres prefijos reales de una conversación española.

| Posición comparada | Correlación | RMS | KL(FP16‖compacto) | Argmax | Top-10 | Masa top-1 FP16 / compacto |
|---|---:|---:|---:|---|---:|---:|
| Tras el turno de usuario | 0,980441 | 0,8977 | 0,000010 | coincide | 9/10 | 1,0000 / 1,0000 |
| Tras `¡Hola!` | 0,980681 | 0,4508 | 0,000005 | coincide | 9/10 | 1,0000 / 1,0000 |
| Tras la primera frase | 0,978818 | 0,6525 | 0,002833 | coincide | 10/10 | 0,9988 / 0,9942 |

Estas posiciones son muy deterministas, por lo que no bastan para certificar
todo el perfil HQS. Sí descartan que el afilado observado en el checkpoint base
se reproduzca automáticamente en este recorrido IT. Con temperatura 1, top-k
64, top-p 0,95 y semilla 7, tanto el compacto como el FP16 produjeron respuestas
españolas coherentes, sin bucles ni monotonía.

### Calibración amplia por posición

`gemma4_calibrate` recorre el HNF token a token con KV real y calcula, sin
guardar cientos de GiB de logits, NLL, entropía, masa top-1, margen y acierto
del token siguiente. El corpus inicial contiene 1.158 tokens de prosa técnica
española (455 palabras) y produce 1.157 predicciones comparables.

Como control del recorrido, el argmax del HNF FP16 coincide con la referencia
NumPy FP32 independiente en 1.138/1.158 posiciones (98,27 %). No existe un
desfase: con offsets de una a tres posiciones el acuerdo cae al 7,52-25,69 %.
El compacto coincide con FP32 en 923/1.158 (79,71 %) y con Héctor FP16 en
921/1.158 (79,53 %). El acuerdo compacto/FP16 es 79,49 % hasta la posición 512
y 79,57 % después; por tanto, la diferencia no nace en el límite de la ventana
deslizante.

| Perfil | NLL media | Perplejidad | Entropía media | Masa top-1 media | Acierto del token siguiente |
|---|---:|---:|---:|---:|---:|
| FP16 | 7,932371 | 2.786,03 | 0,870244 | 0,849569 | 37,165 % |
| Compacto | 7,222117 | 1.369,39 | 0,989420 | 0,826611 | 38,634 % |

El agregado no basta para aprobar HQS: el corpus contiene muchos tokens de
espacio muy previsibles y unas pocas mejoras grandes favorecen la NLL del
compacto. En las posiciones con entropía útil aparece el efecto contrario:

| Selección según FP16 | Posiciones | NLL FP16 / compacto | Entropía FP16 / compacto | Masa top-1 FP16 / compacto |
|---|---:|---:|---:|---:|
| top-1 < 0,80 | 293 | 8,454731 / 8,469759 | 3,067435 / 2,686792 | 0,464078 / 0,552542 |
| top-1 < 0,50 | 149 | 8,529797 / 8,785573 | 4,543247 / 3,734827 | 0,283869 / 0,425968 |

Es decir: globalmente el compacto parece algo menos concentrado, pero justo en
las decisiones abiertas aumenta la masa top-1 en 8,85 puntos para el corte 0,80
y en 14,21 para el corte 0,50. Esta es la señal relevante para monotonía al
muestrear. El perfil sigue siendo funcional, pero aún no queda certificado como
perfil de producción. El siguiente aislamiento debe separar cuantización del
`lm_head`, MLP y ruta PLE/atención antes de promover tensores a ciegas.

Ejemplo reproducible (con la lista de IDs ya tokenizada):

```bash
HELIOS_EMBED_IN_RAM=1 ./build/gemma4_calibrate modelo.hnf \
  "$(<tokens.txt)" posiciones.tsv
```

## Rendimiento y memoria

Con el mismo prompt sobre el color del cielo:

| Perfil | Prompt | Salida | Prefill | Decode |
|---|---:|---:|---:|---:|
| Compacto | 27 tokens | 71 tokens | 102,525 ms | 107,881 tok/s |
| FP16 | 27 tokens | 51 tokens | 38,992 ms | 59,511 tok/s |

La respuesta diverge por el muestreo, pero la carga por token de decode es
comparable. En una generación compacta de 256 tokens, el pico observado por PID
con `nvidia-smi` fue 1.906 MiB y el decode 106,415 tok/s. La referencia FP16
alcanzó 4.626 MiB y 58,786 tok/s en una generación de 128 tokens.

El recorrido largo compacto usó 2.945 tokens de prompt, recuperó correctamente
`ORIÓN-27`, tardó 1.411,95 ms en prefill y generó a 61,018 tok/s. No hubo NaN,
errores CUDA, accesos inválidos ni crecimiento inesperado de memoria.

## Estado

El perfil compacto actual es funcional y entra con margen en una GPU de 8 GB.
La calibración amplia confirma estabilidad a ambos lados de la ventana de 512,
pero detecta afilado en las posiciones de entropía útil. La decisión final del
perfil HQS queda pendiente de aislar la familia de tensores responsable.

## Regresiones disponibles

Después de los cambios del tokenizer, loader y pool de memoria se ejecutaron
dos HNF Qwen3 reales con el forward cached genérico:

| Modelo | Tensores | Resultado greedy | Rendimiento |
|---|---:|---|---:|
| Qwen3-4B | 399 | `The capital of France is Paris...` | 99,174 tok/s |
| Qwen3-8B | 399 | `The capital of France is Paris...` | 40,404 tok/s |

El archivo histórico `tests/qwen3_hq4k_v3.hnf` no es una regresión ejecutable:
está truncado a 3.337.977.856 bytes aunque su cabecera declara 4.533.663.705.
El validador estricto lo rechaza con cinco errores fatales antes de cargar pesos.
No hay artefactos Phi o DeepSeek disponibles en el árbol actual para ejecutar
esas dos regresiones reales.

## Integración en `helios_chat`

El binario interactivo selecciona ahora el contrato por arquitectura. Qwen
conserva ChatML, KV uniforme y sus parámetros de sampling anteriores. Gemma 4
usa la plantilla canónica `<|turn>...<turn|>`, el `Gemma4KVCache` heterogéneo de
15 slots físicos, `build_gemma4_forward_cached` y los defaults oficiales
top-k 64/top-p 0,95 sin penalización adicional.

Pruebas reales con contexto 4096 y embeddings en RAM:

- greedy: `¿Cuál es la capital de Francia?` → `París, claro.`;
- dos turnos, temperatura 0,7: guarda el número favorito 17 y responde después
  `El 17.`;
- CUDA Graph replay queda activo en decode con 942 comandos;
- la misma ruta interactiva con Qwen3-4B conserva ChatML y genera a 101 tok/s.
