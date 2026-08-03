# E4 · subhito — modelo compartido y estado por sesión

Héctor ya sabe tener varias sesiones sobre unos mismos pesos. **19 pruebas · 0
fallos**, y la batería se ha visto fallar.

Héctor **no sabe qué es la reflexión**. Aquí no hay jobs, ni colas, ni
consolidación: hay `Model` (pesos) y `InferenceSession` (una conversación).
Para qué sirve cada sesión lo decide quien llama.

## El corte

| Compartido — `Model` | Por sesión — `InferenceSession` |
|---|---|
| `Engine` y kernels CUDA | `KVCache` / `Gemma4KVCache` |
| pesos del HNF (`HnfLoader`) | `Sampler` y su `SamplingConfig` |
| tokenizer y tokens de turno | grafo de decode (`CommandBuffer`) |
| `GraphBuilder` y sus buffers de trabajo | prefijo propio en el registro de tensores |
| adaptador visual | posición del caché |
| parámetros mecánicos del muestreo | |

Cargar los pesos cuesta segundos y gigabytes de VRAM; una conversación cuesta
un KV. Sin este corte, "otra sesión" significaba **otro proceso con el modelo
entero otra vez**.

## Lo que de verdad separa las sesiones

El KV se registra en el motor **por nombre**: `_kv.layer0.k`, `_kv.layer0.v`…
Dos sesiones con el mismo prefijo no serían dos conversaciones — serían una con
dos voces escribiendo en el mismo caché. Por eso cada sesión tiene su prefijo
(`_kv`, `_kv_s1`, …) y el grafo de decode se construye contra el suyo.

El muestreador también es propio: las penalizaciones de repetición son historia
de **esa** conversación. Compartirlo haría que lo dicho en una frenara las
palabras de la otra.

## La API de antes no se ha movido

`InferenceSession::load(Config)` sigue existiendo y hace lo mismo — ahora por
dentro es `Model::load` + `attach`. `Config` y `ModelInfo` siguen accesibles con
sus nombres de siempre. La primera sesión conserva el prefijo `_kv` exacto, así
que la ruta ya certificada registra los mismos nombres que antes.

## En serie, y dicho en voz alta

Las sesiones comparten los buffers de trabajo del grafo, así que dos turnos a
la vez se pisarían las activaciones. `run_turn` los serializa con un cerrojo:
si dos hilos entran, uno espera. Es una espera, no una corrupción — y cuando el
scratch sea por sesión, esa línea se cae sola. Fingir concurrencia daría
resultados corruptos en vez de lentos.

## Aislamiento exacto

Todo greedy: si algo cambia entre dos ejecuciones, es contaminación, no
muestreo.

| Prueba | |
|---|---|
| lo dicho en A no existe en B | ✅ |
| usar B no altera A, ni le pega nada | ✅ |
| `reset()` de A deja el caché de B intacto hasta el token | ✅ |
| destruir una sesión no toca a las demás ni al modelo | ✅ |

Y la fina: una sesión limpia responde a una sonda; se usa **otra** sesión a
fondo (dos párrafos largos); una tercera sesión limpia responde a la misma
sonda. **Byte a byte idéntico.** Con greedy no hay otra explicación posible: si
difiriera, algo del estado de la segunda habría sobrevivido al cambio.

## La batería se ha visto fallar

`HELIOS_KV_PREFIJO_FIJO=1` es un grifo de sabotaje —no una opción— que obliga a
todas las sesiones a compartir prefijo. Con él, **5 de las 19 pruebas fallan**.

Y sale algo que no esperaba: el motor **rechaza el registro duplicado**
(`Tensor already registered: _kv.layer0.k`), así que la colisión no puede
ocurrir en silencio ni aunque alguien se salte el prefijo. Es defensa en
profundidad que ya estaba, y ahora consta.

## No se ha roto nada

| | |
|---|---|
| Ruta tipada con memoria, extremo a extremo | 29 · 0 |
| Oráculo de la legacy contra la base de E1 | exit 0, cero diferencias |

## Lo que sigue sin existir

Trabajador, encolado desde la ruta, autoridad por origen y checkpoint. La cola
durable sigue esperando, vacía y a propósito.
