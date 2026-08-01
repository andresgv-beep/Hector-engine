# Campaña de rendimiento decode — fusiones, embedding y HQ3.1K

Fecha de apertura: 2026-08-01

## Objetivo

Mejorar el `decode` real de Héctor sin confundir tres palancas distintas:

1. reducir lanzamientos y tráfico intermedio de kernels pequeños;
2. eliminar la tabla FP16 de embedding duplicada;
3. reducir bytes de pesos con el perfil mixto HQ3.1K.

La campaña empieza con Qwen3-4B, porque la línea base y el perfil de 74,9 % de
tiempo HQ4.1K pertenecen a ese modelo. Gemma 4 se usa como regresión después de
integrar cada ganador; sus 942 nodos y su grafo heterogéneo no se mezclarán con
las conclusiones de Qwen.

No se promete de antemano 125-135 tok/s. El primer objetivo acreditable es
superar **108,21 tok/s** en el régimen alto ya documentado. Un resultado de
110-112 tok/s sería una ganancia importante. El techo se recalculará con el
perfil posterior a cada fase.

## Punto de partida demostrado

- Qwen3-4B HQ4.1K: mediana alta **103,0595 tok/s**, rango
  **102,894-103,184 tok/s**.
- La GPU también cae a un régimen de 48-56 W y unos 67 tok/s. Esas muestras no
  se promedian con las de potencia alta.
- HQ4.1K ocupa **74,9 %** del tiempo GPU observado.
- Los GEMV grandes alcanzan 329-365 GB/s; no justifican otra campaña de ajuste
  fino del mismo kernel.
- El grafo Qwen ya fusiona `QK_NORM_ROPE`, `SILU_MUL` y el residual de atención
  con el RMSNorm del MLP mediante `ADD_RMSNORM`.
- El intento histórico de fusionar RMSNorm dentro del GEMV fue más lento por
  serializar la reducción. No se reabre sin un diseño distinto y una medida que
  lo justifique.
- El perfil HQ3.1K de todo el MLP dio +6,6 % en régimen alto, pero se rechazó
  por calidad de conversación. El perfil defendible `gate/up=HQ3.1K`,
  `down=HQ5.1K`, atención HQ4.1K ahorra aproximadamente **4,7 %** de los bytes
  de matrices, no 19 %.

El reparto comunicado de 9,9 ms/token, con 1,67 ms en normas, RoPE,
elementwise y residuales, es una **hipótesis que hay que reproducir por nodo**.
Podría mezclar modelos, huecos entre kernels o categorías ya fusionadas.

## Reglas de la campaña

1. Un candidato por vez y siempre contra el mismo binario base.
2. Prompt, contexto, sampler, semilla, HNF y `HELIOS_EMBED_IN_RAM` fijos.
3. Una pasada de calentamiento; después al menos cinco muestras calientes.
4. Registrar reloj, temperatura y potencia, y separar los regímenes de la GPU.
5. Excluir carga, captura inicial del CUDA Graph y prefill al medir decode.
6. Comparar 256 tokens greedy y su SHA256 antes/después.
7. Un cambio menor del 2 % solo se acepta con A/B emparejado y reducción
   equivalente visible en el perfil GPU.
8. Si una fase no alcanza su barrera, se revierte solo esa fase y no se apilan
   cambios para esconder el resultado.
9. HQ3.1K, embedding y fusiones usan artefactos/ramas de medida separados.

## Fase 0 — congelar la línea base

### Trabajo

- Anotar commit, hash del binario y SHA256 del HNF.
- Repetir el comando certificado de 256 tokens greedy.
- Obtener cinco muestras en potencia alta y, si aparece, documentar aparte el
  régimen bajo.
- Guardar tokens, tok/s, tiempo por token, reloj, temperatura y potencia.

### Salida

- Mediana dentro de +/-2 % de 103,0595 tok/s, o nueva línea base explicada.
- Secuencia greedy reproducible.
- Ninguna mezcla de regímenes térmicos.

## Fase 1 — contabilidad por nodo del CUDA Graph

Nsight Systems está disponible en:

```text
/usr/local/cuda-13.1/bin/nsys
```

Se perfilará una ejecución caliente con `--cuda-graph-trace=node`, agrupando
por nombre de kernel:

| Categoría | Qué incluye |
|---|---|
| HQ4.1K | Q/K/V/O, gate/up/down |
| HQ5.1K | `lm_head` |
| Atención/KV | actualización de caché y atención |
| Normas | RMSNorm y ADD_RMSNORM |
| Posición | RoPE y QK_NORM_ROPE |
| Elementwise | add, scale, activación/mul y copias |
| Huecos | dependencias y separación entre nodos GPU |

Se registrarán por token el número de llamadas, tiempo total, mediana y p95. La
suma debe reconciliar al menos el 98 % del tiempo GPU observado. También se
comparará el tiempo GPU con el tiempo end-to-end para no llamar «kernel» a una
espera de CPU.

### Salida

- Tabla reproducible de microsegundos/token y llamadas/token.
- Confirmar o refutar los 1,67 ms de kernels pequeños.
- Elegir la siguiente fusión por ahorro máximo medido, no por intuición.

## Fase 2 — primer candidato: residual MLP + RMSNorm siguiente

El grafo genérico termina cada capa con:

```text
down_proj -> ADD(hidden, residual, mlp_out)
            -> RMSNorm de atención de la capa siguiente
```

Héctor ya tiene un `ADD_RMSNORM` que produce a la vez el residual FP16 y la
salida normalizada. El candidato consiste en reorganizar el constructor del
grafo Qwen para usarlo entre capas:

- la entrada de la primera capa conserva su RMSNorm normal;
- entre capas, un `ADD_RMSNORM` sustituye la pareja `ADD + RMSNorm`;
- en la última capa se estudia la misma unión con el RMSNorm final;
- no se modifica todavía la ruta Gemma 4.

### Verificación antes del benchmark

- Prueba CUDA de `ADD + RMSNorm` separado contra `ADD_RMSNORM` para las formas
  reales de Qwen.
- Residual idéntico; norma dentro de la tolerancia FP16 fijada antes de correr.
- Logits finales: mismo argmax y top-10; 256 tokens greedy idénticos.
- Suite completa de Héctor.
- El CUDA Graph sigue activo y reduce realmente su número de nodos.

### Barrera

Se conserva si aporta **>=2 % end-to-end** en A/B emparejado o si ahorra al
menos 0,20 ms/token medidos y el ruido térmico impide separar el total. Si no,
se documenta y se descarta.

## Fase 3 — segundo candidato solo después de reperfilar

No se decide aún cuál será. Se escogerá el mayor coste restante entre:

- lanzamientos Q/K/V separados;
- copia de K/V después de normalización/RoPE;
- residuales o normas que no cubra la fase 2;
- fusiones específicas de Gemma 4 (`GELU + MUL`, post-norm + residual y
  Q/K/V norm + RoPE), únicamente si el perfil Gemma demuestra el coste.

No se reabre el GEMV HQ4.1K grande. Tampoco se fusionan operaciones a través de
una frontera si obliga a releer más pesos, duplicar una reducción o perder la
configuración rápida que el autoajuste ya elige.

### Barrera

Mismos criterios numéricos y de producto que en la fase 2. Máximo un segundo
candidato durante esta campaña; después se mide el techo nuevo.

## Fase 4 — eliminar la tabla FP16 de embedding duplicada

Este trabajo busca memoria y tamaño, no se contabiliza como ganancia de tok/s.
El primer diseño a probar será una **única tabla compartida HQ5.1K** para
embedding y `lm_head` en modelos con pesos atados:

- añadir lookup de una fila HQ5.1K para el embedding normal;
- hacer que el `lm_head` reutilice exactamente el mismo tensor;
- omitir del HNF la copia FP16 y el tensor duplicado cuando el contrato declare
  pesos atados.

Es la ruta conservadora: el `lm_head` ya usa HQ5.1K. Frente al HNF actual puede
eliminar aproximadamente **742 MB** de copia FP16. Con
`HELIOS_EMBED_IN_RAM=1` esa copia ya no ocupa VRAM, de modo que allí la mejora
principal será archivo, RAM y tiempo de carga; sin la variable también libera
VRAM.

HQ4.1K compartido se evaluará después y solo si merece arriesgar la precisión
del `lm_head`. No se mezclará con la primera prueba.

### Barrera

- Lookup de filas contra decoder CPU en filas 0, 42 y última.
- Validador HNF sin tensor huérfano ni alias ambiguo.
- Dos corpus de logits y batería de chat sin regresión de producto.
- Medidas separadas de archivo, RAM, VRAM, carga y tok/s.

### Resultado 2026-08-01 — mecanismo válido, perfil no promovido

Se añadió al conversor el selector opt-in `--shared-embedding-hq51`. Solo se
acepta con `tie_word_embeddings=true`, rechaza checkpoints con un `lm_head`
independiente, escribe `token_embedding.weight` en HQ5.1K y omite
`lm_head.weight`. El grafo reutiliza la tabla automáticamente. El loader
mantiene esa tabla en VRAM aunque se solicite `HELIOS_EMBED_IN_RAM=1`; PLE y
los HNF antiguos con cabeza separada conservan el offload anterior.

Qwen3-4B real, mismo perfil HQ5K compacto con atención HQ4.1K:

| Medida | Producción | Compartido HQ5.1K |
|---|---:|---:|
| tensores | 399 | 398 |
| tamaño HNF | 3.470.618.923 B | 2.692.707.354 B |
| ahorro | — | 777.911.569 B / 741,87 MiB |
| VRAM sin offload | 4.368 MiB | 3.626 MiB |
| VRAM con offload | 3.626 MiB | 3.626 MiB |
| pico RSS mediano | 1.890.044 KiB | 667.704 KiB |
| carga mediana | 1,62 s | 1,11 s |
| decode caliente | 103,52 / 103,35 tok/s | 103,56 / 103,43 tok/s |

El HNF pasa el validador estricto con 0 errores y 0 avisos; Héctor pasa 14/14
y el conversor 43/43. El banco contra FP32 no demuestra regresión: 1.755
posiciones dan 88,09 % para producción y 87,35 % para el compartido; McNemar
60 frente a 47, `p=0,246`. Los dos corpus por separado también quedan sin
diferencia significativa (`p=0,556` y `p=0,311`).

La barrera de producto, sin embargo, **no pasa**. En el A/B de una semilla por
12 turnos, el candidato introduce una frase china, pierde parte del perfil de
memoria y termina una respuesta en bucle. El detector automático marca script
inesperado 0,083 frente a 0 y acierto semántico 0,50 frente a 1,00. Por tanto,
el mecanismo y el ahorro quedan disponibles para investigación, pero el modo
no sustituye al perfil de producción. No se programa ahora una variante HQ5K
no compacta: puede proteger mejor el embedding, pero también encarecer el
`lm_head` y desplazar el problema a velocidad/tamaño. El baseline permanece en
el HNF actual con `HELIOS_EMBED_IN_RAM=1`.

La siguiente variante que merece un artefacto es un perfil ponderado del
conversor: embedding y `down_proj` protegidos y compresión solo en familias que
pasen aislamiento previo. Se compara sin cambios simultáneos de kernels y solo
se promueve con cero fallos conversacionales, dos corpus sin regresión, pérdida
de velocidad no superior al 1 % y ahorro real separado en archivo, RAM y VRAM.

## Fase 5 — HQ3.1K mixto, como última palanca

Solo se construye tras confirmar el segundo corpus comunicado para:

```text
gate_proj = HQ3.1K
up_proj   = HQ3.1K
down_proj = HQ5.1K
atención  = HQ4.1K
```

El encoder, layout de 136 bytes y kernel ya existen. El trabajo esperado es el
selector del mapper, un HNF aislado y validación. No se presenta como +20 %:
por el ahorro de bytes observado, la expectativa razonable es del orden de
**3-5 tok/s**, pendiente de medida real.

### Barrera

- Dos corpus por encima del umbral fijado antes de convertir.
- `tools/hq31_ab.py`: núcleo y conversación larga sin escritura extraña,
  pérdida de memoria ni aumento de repetición.
- Mediana alta, rango y A/B térmico contra el ganador de la fase 3.
- Si falla conversación, HQ3.1K continúa experimental aunque el banco de
  logits pase.

## Fase 6 — cierre y decisión

Publicar una única tabla antes/después:

| Medida | Base | Fusiones | + embedding | + HQ3 mixto |
|---|---:|---:|---:|---:|
| tok/s régimen alto | | | | |
| ms/token | | | | |
| nodos CUDA Graph | 507 | | | |
| tamaño HNF | | | | |
| RAM pico | | | | |
| VRAM pico | | | | |
| SHA256 greedy | | | | |
| chat A/B | | | | |

Regresión final mínima:

- Qwen3-4B de producción;
- Qwen3-8B legado;
- Gemma 4 E2B instruct corto y contexto mayor de 512;
- suite completa del motor y validador de los HNF nuevos.

## Orden exacto para el siguiente turno

1. Congelar como baseline el HNF actual con `HELIOS_EMBED_IN_RAM=1`, incluida
   su salida conversacional y los 106,57 tok/s de referencia.
2. Aislar en el conversor las familias de matrices con los dos corpus ya
   definidos; embedding y `down_proj` permanecen protegidos.
3. Emitir un único HNF ponderado, sin cambios simultáneos de kernels, y medir
   calidad conversacional, corpus, velocidad, archivo, RAM y VRAM.
4. Promoverlo solo si supera juntas las cuatro barreras. Si falla, conservar el
   baseline actual y cerrar la campaña sin apilar otra variante.
5. Reanudar visión únicamente cuando el baseline de texto quede congelado.

No se reabre `ADD + RMSNorm siguiente`: el perfil demuestra que no alcanza la
barrera. La tabla compartida HQ5.1K y el perfil mixto HQ3.1K permanecen
experimentales y no son el siguiente candidato de producción.

## Resultado de la primera campaña — 2026-08-01

### Perfil reproducido

Nsight Systems separó 77 ciclos estables de Qwen3-4B en el régimen de 55 W:

| Categoría | ms/token | llamadas/token |
|---|---:|---:|
| HQ4.1K | 12,0157 | 252 |
| HQ5.1K (`lm_head`) | 1,1847 | 1 |
| RMSNorm | 0,3562 | 37 |
| ADD_RMSNORM | 0,3838 | 36 |
| Atención | 0,2953 | 36 |
| KV update | 0,1242 | 72 |
| QK norm + RoPE | 0,1015 | 36 |
| ADD | 0,0639 | 36 |
| SiLU * up | 0,0498 | 36 |

Total: **15,0132 ms/token**, de los que 14,6345 ms son trabajo GPU y
0,3787 ms son huecos. El dato comunicado de unos 1,67 ms fuera de los GEMV era
razonable, pero incluye trabajo necesario que no puede eliminarse entero.

El mensaje de captura habla de **507 comandos**. El perfil observa 548
operaciones GPU por ciclo contando los dos updates K/V, copias y operaciones
externas al `CommandBuffer`; no deben llamarse 507 nodos GPU.

### Candidatos descartados

- `ADD` final + RMSNorm siguiente: el ahorro calculado con los kernels reales
  es aproximadamente 0,03-0,08 ms/token, por debajo de la barrera de 0,20 ms.
- Reducción RMSNorm warp→bloque: 14/14 tests, texto idéntico y A/B alto
  10,121→10,040 ms/token, alrededor de +0,8 %. Se retiró por no alcanzar 2 %.
- RMSNorm de una sola warp: texto idéntico, pero 95,6 frente a 104,7 tok/s en
  el mismo régimen alto. Se retiró.
- Lecturas HQ3.1K alineadas o cooperativas: la primera alteró la relación
  HQ3/HQ4 solo un 0,33 % y la segunda empeoró 4-5 %. Se restauró el kernel.

### Perfil mixto HQ3.1K real

Se generó y validó el HNF `qwen3_4b_gate-up-hq31_down-hq51_attn-hq41.hnf`:

- 72 tensores HQ3.1K (`gate/up`);
- 37 HQ5.1K (36 `down` + `lm_head`);
- 144 HQ4.1K de atención;
- 146 FP16;
- 3.358.553.312 bytes, 106,87 MiB menos que producción (3,23 %).

En el régimen bajo comparable pasó de **15,013 a 15,364 ms/token**: −2,3 %.
`gate/up` HQ3.1K tardó 6,089 ms frente a unos 5,94 ms en HQ4.1K y el peaje de
`down` HQ5.1K añadió aproximadamente 0,23 ms. En potencia alta las pasadas
oscilaron alrededor del empate, nunca cerca de +5 tok/s. El perfil puede tener
valor por memoria/calidad, pero **no es una optimización de velocidad** en el
kernel actual.
