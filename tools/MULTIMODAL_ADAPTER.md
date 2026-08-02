# Adaptadores multimodales persistentes

## Frontera del motor

`MultimodalAdapter` separa la sesión conversacional de cada arquitectura. El
chat conserva tokenizer, sampler, KV y decode; el adaptador recibe un turno ya
encuadrado y uno o más adjuntos prestados, ejecuta el encoder de modalidad y
prefillea el decoder en la posición actual del KV.

La sesión solo avanza su contador de KV cuando `prefill()` termina con éxito.
El resultado informa cuántos tokens de secuencia se escribieron y cuántos
tokens proceden del adjunto.

La entrada común incluye:

- clase de adjunto (`ImageRgb8`, reservado `AudioPcmF32`);
- puntero y tamaño de un payload prestado;
- tipo de medio;
- geometría de imagen o audio;
- tokens ya encuadrados por la plantilla de chat.

PNG, JPEG o WebP no pertenecen al motor. La frontera visual recibe RGB8
decodificado, por lo que añadir un frontend, cambiar la librería de imágenes o
usar el decoder nativo del navegador no modifica Héctor.

## Primer adaptador

`helios.gemma4.vision.v1` admite una imagen RGB8 por turno. Internamente:

1. valida tamaño, stride y capacidad del prefill antes de tocar CUDA;
2. preprocesa la imagen con el contrato certificado de Gemma 4;
3. ejecuta la torre visual mediante el pipeline genérico de pesos mapeados;
4. expande el placeholder y sustituye sus filas visuales;
5. ejecuta el prefill textual en `KVCacheParams.cache_position`;
6. deja logits listos para que la sesión use su sampler habitual.

Los buffers de intercambio son estables y reutilizables. Con
`HELIOS_VISION_MMAP=1`, el bloque visual queda mapeado y cada imagen reutiliza
el doble buffer. Texto, KV y grafo de decode permanecen residentes.

La prueba real ejecuta dos imágenes en posiciones diferentes del mismo KV. La
prueba de producto envía después un turno solo textual y confirma que conserva
la observación visual anterior.

## Protocolo de proceso v1

`helios_chat` acepta un frame interno, pensado para HexOS:

```text
/adjunto-rgb8 WIDTH HEIGHT STRIDE BYTES\n
<BYTES bytes RGB8>\n
```

El frame no genera un turno. Deja una imagen pendiente que consume el siguiente
mensaje normal. `/adjunto-limpiar` la descarta. El límite de transporte es 300
MiB y cualquier cabecera imposible cierra el proceso: seguir leyendo tras una
longitud corrupta desincronizaría texto y binario silenciosamente.

Este protocolo es únicamente la tubería local HexOS→Héctor. La API HTTP puede
usar el encuadre que convenga a la UI y convertirlo en este frame tras validar
longitud y geometría.

## Añadir otra arquitectura

Un nuevo adaptador implementa `id()`, `limits()` y `prefill()`, se incorpora a
`create_multimodal_adapter` y registra su terna en `model_capabilities`. No
cambia el bucle de chat, el protocolo de proceso, HexOS ni la UI.

Audio reutilizará el mismo contrato con `AudioPcmF32`; antes de anunciarlo como
disponible necesita config binaria HNF, preprocesador, runner y regla de
adaptador, en ese orden.
