# HQ3.1K — contrato e integración verificada

Fecha: 2026-08-01

## Alcance seguro

HQ3.1K se añadió como dtype nuevo `hq31k`. No sustituye el hueco histórico
`HQ3K`, no cambia el layout de HQ4.1K/HQ5.1K y sólo entra en el dispatch cuando
el manifiesto HNF declara explícitamente `hq31k`.

Contrato por superbloque de 256 pesos:

- header compacto común: 40 bytes;
- payload: 96 bytes (8 índices de 3 bits por cada grupo de 8 pesos);
- total: 136 bytes, 4,25 bits por peso;
- empaquetado: LSB-first, tres bytes exactos por grupo.

El conversor usa `--quant HQ31K --compact --attn4` para el perfil aislado:
MLP HQ3.1K, atención HQ4.1K, embedding FP16 y `lm_head` HQ5.1K.

Estado de producto: **formato y motor verificados; ningún perfil HQ3.1K probado
supera la barrera completa de conversación**. No usar como perfil de producción
de Qwen3-4B. Se conserva para investigación y para medir otros modelos.

## Evidencia

### Calidad antes de implementar

Segundo corpus narrativo, 1.107 posiciones:

| Perfil simulado | Coincidencia argmax con FP32 |
|---|---:|
| Todo 3 bits | 80,578% (892/1107) |
| MLP3 + atención4 + embedding4 | 83,288% (922/1107) |

El perfil mixto ganó 59 posiciones contra 29 (McNemar exacto p=0,001824).

### Encoder y kernel reales

El HNF `qwen3_4b_mlp-hq31_attn-hq41.hnf` contiene 399 tensores: 108 HQ3.1K,
144 HQ4.1K, 1 HQ5.1K y 146 FP16. El validador estricto informa 0 errores y 0
avisos.

| Medida | Resultado |
|---|---:|
| Tamaño anterior | 3.470.618.923 bytes |
| Tamaño HQ3.1K | 3.134.420.176 bytes |
| Ahorro | 336.198.747 bytes (320,6 MiB; 9,69%) |
| HNF real vs FP32, 1.107 posiciones | 83,017% |
| Simulador vs FP32 | 83,288% |
| HNF real vs simulador | 95,845% |

El desvío real respecto a la predicción es -0,271 puntos, por lo que encoder y
decoder CUDA reproducen el contrato simulado.

### Correctitud y regresión

- Convertidor: 37/37 tests.
- Héctor: prueba sintética CUDA contra decoder CPU para `M=1` y `M=2`.
- Héctor: 14/14 tests CTest.
- Generación real Qwen3-4B HQ3.1K correcta, con CUDA Graph replay.
- El HNF Qwen3-4B anterior carga y genera tras añadir el dtype.

En una muestra greedy de 512 tokens, los dos HNF derivaron a repetición y el
HQ3.1K terminó alternando a chino. Esto no es un fallo de lectura —la ruta real
coincide con el simulador en 95,845% de las posiciones—, pero la calidad de
producto debe juzgarse con una batería de chat y el sampling previsto, no con
una sola trayectoria greedy.

### Rendimiento

La GPU alterna automáticamente entre dos límites de potencia, por lo que las
muestras se separan por reloj/potencia y no se promedian entre sí.

| Régimen | HNF anterior | HQ3.1K |
|---|---:|---:|
| Alto, ~100-112 W / ~2,4 GHz | 100,99 tok/s | 107,61 tok/s |
| Bajo, ~54-55 W / ~1,28 GHz | ~67,1 tok/s | 65,83-67,35 tok/s |

Resultado: no aparece una regresión de inferencia; con ancho de banda alto el
menor tráfico de pesos aporta alrededor de 6,6%.

## Barrera de producto con sampling real

Arnés: `tools/hq31_ab.py`, perfil sintético aislado, temperatura 0,7, top-k 50,
top-p 0,9, repetición 1,15, ventana 384, frecuencia 0,1 y semillas 1-5. El
corpus núcleo contiene 10 prompts por semilla. Los umbrales se fijaron antes de
ver las respuestas: cero fallos/turnos perdidos, no empeorar escritura, UTF-8,
tokens especiales ni identidad, y como máximo +5 puntos en bucles/cortes y -5
puntos en controles semánticos.

### Todo el MLP a HQ3.1K — rechazado

| Métrica núcleo, 50 turnos | HQ4.1K | MLP HQ3.1K |
|---|---:|---:|
| Bucles | 6% | 0% |
| Escritura no latina inesperada | 0% | 16% |
| UTF-8 inválido | 0% | 2% |
| Token especial visible | 0% | 2% |
| Controles de memoria | 80% | 30% |

Con semilla 5 hubo mezcla de chino, cirílico y árabe, un byte inválido y pérdida
de la memoria sintética. El ahorro de 320,6 MiB no compensa esta degradación.

### Aislamiento por familia

Simulación sobre 1.107 posiciones narrativas, dejando el resto a 4 bits:

| Familia HQ3.1K | Coincidencia con FP32 |
|---|---:|
| `gate_proj` | 87,173% |
| `up_proj` | 86,269% |
| `down_proj` | 86,089% |

`gate_proj` fue la mejor. En técnico obtuvo 87,654%. El HNF real con los 36
`gate_proj` dio 87,082% narrativa y 87,191% técnico, pero el A/B núcleo aún
produjo un carácter chino visible en 1/50 turnos; no pasó el umbral estricto.

### `gate_proj` sólo en capas 0-17 — núcleo aprobado, largo rechazado

El selector conservador se expone como:

```text
--quant HQ31K --compact --attn4 --hq31-gate-only \
--hq31-layer-start 0 --hq31-layer-end 17
```

El artefacto tiene 18 HQ3.1K, 234 HQ4.1K, 1 HQ5.1K y 146 FP16. Ocupa
3.414.586.610 bytes frente a 3.470.618.923: ahorra 56.032.313 bytes (53,44 MiB;
1,61%). Validador estricto 0 errores/0 avisos.

| Barrido real | Real vs FP32 | Simulación vs FP32 |
|---|---:|---:|
| Narrativa, 1.107 posiciones | 87,534% | 88,076% |
| Técnico, 648 posiciones | 88,426% | 88,426% |

El núcleo de 50 turnos pasó: 0% escritura/UTF-8/tokens especiales, memoria 90%
frente a 80% y bucles 2% frente a 6% del baseline. La batería larga lo rechazó:
2/4 respuestas con bucle frente a 1/4, y 937 frente a 776 tokens por turno.

Conclusión: reducir HQ3.1K hasta 53,44 MiB elimina la corrupción multilingüe,
pero no la regresión de repetición larga. No seguir recortando capas para
perseguir un ahorro menor; la siguiente palanca es el embedding HQ4.1K medido.

## Límites y siguiente paso

La primera versión usa GEMV correcto en bucle para `M > 1` y limita
`K <= 16384` por memoria compartida. No se añadirá GEMM sin un perfil de
prefill largo que demuestre que hace falta.

Los embeddings continúan en FP16. Pasarlos a HQ4.1K es otro cambio de contrato:
requiere ruta explícita en el conversor y lookup HQ4.1K en Héctor. Debe hacerse
en una fase y un artefacto separados para conservar una comparación atribuible.

## Adenda 2026-08-01 — perfil mixto real y rendimiento

El conversor incorpora el selector aditivo `--hq31-mixed-mlp`:

```text
--quant HQ31K --compact --attn4 --hq31-mixed-mlp
```

Asigna `gate/up` a HQ3.1K, protege `down` con HQ5.1K y deja atención en
HQ4.1K. No cambia los perfiles anteriores y es incompatible por CLI con
`--hq31-gate-only`.

HNF real generado:

- `qwen3_4b_gate-up-hq31_down-hq51_attn-hq41.hnf`;
- 72 HQ3.1K, 37 HQ5.1K, 144 HQ4.1K y 146 FP16;
- 3.358.553.312 bytes: ahorro de 112.065.611 bytes (106,87 MiB; 3,23%);
- validador estricto: 0 errores y 0 avisos;
- SHA256 histórico: `b2f8bd1614670500ba6ae84658295628cc63f909df244d49f32d7d95000dd17d`.
  Este artefacto experimental usa el layout anterior a `37610b3`; el hash
  identifica el fichero existente, pero no es una certificación reproducible
  con el conversor actual.

La asignación reproduce el perfil que obtuvo 88,1% en los dos corpus, pero no
superó la barrera de rendimiento. En un perfil bajo comparable, producción dio
15,013 ms/token y el mixto 15,364 ms/token (−2,3%). En potencia alta oscila
alrededor del empate y no se acerca a +5 tok/s.

Se probaron dos lecturas alternativas del payload HQ3.1K. La alineada cambió
la relación temporal HQ3/HQ4 solo 0,33%; la cooperativa con `shuffle` empeoró
4-5%. Ambas se retiraron y el kernel quedó restaurado. El perfil mixto sigue
siendo candidato de memoria sujeto al A/B largo, no optimización de velocidad.
