# Contrato genérico de capacidades HNF

## Objetivo

La orquestación no debe reconocer modelos por su nombre de fichero ni conocer
los structs privados de cada arquitectura. `model_capabilities` convierte los
metadatos HNF en un descriptor estable y separa dos preguntas:

1. qué modalidades contiene el artefacto;
2. qué adaptador de Héctor sabe ejecutarlas junto a su decoder de texto.

El contrato actual es `helios.model-capabilities.v1`.

## Sonda sin cargar pesos

`helios_model_probe` solo llama a `HnfLoader::load_metadata`. No crea un
contexto CUDA ni lleva tensores a VRAM.

```bash
build/helios_model_probe modelo.hnf
build/helios_model_probe --json modelo.hnf
build/helios_model_probe --kv modelo.hnf
```

El formato `--kv` es la frontera de procesos que consume HexOS. JSON se ofrece
para herramientas y UI; la salida humana es únicamente diagnóstica.

Cada modalidad expone:

- `declared`: aparece en flags o hints;
- `present`: existe su bloque HNF;
- `configured`: tiene metadatos de arquitectura suficientes;
- `architecture`: encoder o arquitectura declarada por el HNF;
- `adapter`: identificador estable del adaptador resuelto;
- `status`: `absent`, `declared_only`, `metadata_ready` o `runtime_ready`.

`runtime_ready` significa que existe un adaptador compatible registrado. La
validación completa de tensores sigue perteneciendo al adaptador al cargar el
bloque; la sonda no ejecuta kernels ni sustituye esa validación.

## Registro de adaptadores

Una regla se resuelve por esta terna:

```text
(arquitectura de texto, modalidad, arquitectura del encoder) -> adapter_id
```

Las primeras reglas cubren el chat de texto Qwen/Gemma 4 y la visión Gemma 4.
La regla multimodal es:

```text
(gemma4, vision, gemma4) -> helios.gemma4.vision.v1
```

No hay una condición Gemma en HexOS. Para portar otra arquitectura:

1. el conversor emite su configuración genérica en `ExecutionHintsBin`;
2. `HnfLoader` conserva esa arquitectura en la configuración del bloque;
3. se implementa y valida el runner específico;
4. se añade una regla a `kAdapterRules`;
5. se prueba que un HNF compatible resulte `runtime_ready` y uno incompatible
   permanezca `metadata_ready`.

El protocolo de HexOS, `/api/status` y la UI no cambian al hacer esos cinco
pasos.

## Estado de las modalidades

Visión Gemma 4 tiene adaptador persistente listo mediante
`MultimodalAdapter`; su frontera y framing se documentan en
`MULTIMODAL_ADAPTER.md`. CLIP, SigLIP, ViT y EVA ya caben en el
vocabulario binario, pero permanecen `metadata_ready` hasta tener runner y
regla. Audio y vídeo pueden detectarse por bloque/flags; sus configs binarias
y adaptadores todavía no existen y por ello aparecen `declared_only`.
