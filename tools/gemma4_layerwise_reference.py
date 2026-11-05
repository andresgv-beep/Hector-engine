"""Trayectoria completa de Gemma 4 26B en CPU, una capa cada vez, en FP32.

uso: gemma4_layerwise_reference.py forward variante pesos casos/ modelo-original/ salida/
     gemma4_layerwise_reference.py compare casos/ modelo-original/ hnf-A hnf-B runs-motor/ prompts.jsonl salida/
     gemma4_layerwise_reference.py pieces casos/ modelo-original/ hnf salida/ [pieza...]
     gemma4_layerwise_reference.py engine casos/ modelo-original/ runs-motor/ salida/ run...
     gemma4_layerwise_reference.py imatrix casos-calibración/ modelo-original/ captura.npz
     gemma4_layerwise_reference.py export-imatrix modelo-original/ captura.npz carpeta/

imatrix suma x² por columna de la entrada de cada experto en el original. Con IMATRIX
(captura), GPTQ_DIR (bloques HQ44 por capa) y GGUF, pieces mide expertos HQ44 sin
calibrar, con imatrix, con GPTQ y de Unsloth, con el resto del modelo original, y guarda
la KL de cada documento.

engine compara los logits reales del motor de cada run con la trayectoria original.

pieces parte del original y cuantiza una sola pieza cada vez, para ver de dónde sale el error.

variante: original, A o B. pesos: el HNF, o - con original. Cada capa carga solo sus
pesos y pasa todos los casos de una vez, así que el modelo nunca está entero en RAM.
Misma aritmética en las tres variantes: «pesos BF16 originales, aritmética FP32 CPU»
en original, no ejecución nativa BF16 ni réplica bit a bit de CUDA.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import struct
import subprocess
import tempfile
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma4_reference_common import Weights, gelu, rms, route
from hnf10 import decode, decode_hq44

LAYERS, E, H, I, K, SHARED, HEADS = 30, 128, 2816, 704, 8, 2112, 16
GLOBAL_LAYERS = {5, 11, 17, 23, 29}
NAMES_HNF = os.environ.get('NAMES_HNF', str(Path(__file__).resolve().parents[2]/'helios_convert_v10/output/gemma4-26b-moe-dense-hq8-gptq.v10.hnf'))
FIRST, MAX_FORCING, CHUNK = 16, 512, 64


class Reader(Weights):
    """Un HNFv10 y el checkpoint original, por el mismo nombre canónico."""
    def __init__(self, model, hnf=None):
        # Sin HNF, el de NAMES_HNF solo da nombres y procedencia: todos los HNF del 26B los comparten.
        super().__init__(hnf or NAMES_HNF, model, verify=False)
        self.quantized = hnf is not None

    def get(self, name, expert=None):
        return self.packed(name, expert) if self.quantized else self.original(name, expert)

    def embedding(self, rows=None):
        name = 'text.token_embedding.weight'
        if not self.quantized:
            full = self.original(name)
            return full if rows is None else full[rows]
        t, raw = self.tensors[name], self.hnf10.raw(name)
        wanted = np.arange(t.shape[0]) if rows is None else rows
        out = np.empty((len(wanted), H), np.float32)
        for start in range(0, len(wanted), 4096):
            part = wanted[start:start+4096]
            out[start:start+len(part)] = decode(t.dtype, np.asarray(raw[part])).reshape(len(part), H)
        return out


PIECES = {
    'atencion': r'\.attn\.[qkvo]_proj',
    'mlp_compartida': r'\.mlp\.',
    'expertos_gate_up': r'moe\.experts\.gate_up',
    'expertos_down': r'moe\.experts\.down',
    'embedding_cabeza': r'token_embedding',
    'normas_router_fp16': r'\.ln_|_norm\.|moe\.router|layer_scalar|final_norm',
    # Combinaciones: cuánto queda si las partes densas no se cuantizan.
    'expertos_y_embedding': r'moe\.experts|token_embedding',
    'todo_menos_atencion': r'\.mlp\.|moe\.experts|token_embedding',
}


class OnePiece(Reader):
    """El original con una sola pieza leída del HNF."""
    def __init__(self, model, hnf, pattern):
        super().__init__(model)
        self.pattern = re.compile(pattern)
        self.quantized = Reader(model, hnf)

    def get(self, name, expert=None):
        return self.quantized.get(name, expert) if self.pattern.search(name) else super().get(name, expert)

    def embedding(self, rows=None):
        return self.quantized.embedding(rows) if self.pattern.search('token_embedding') else super().embedding(rows)


# Cuantizaciones estándar de referencia (bloques de 32 por fila, redondeo al más
# cercano, estilo Q8_0/Q4_0): dicen si el error es del modelo o del formato HQ.
BASELINES = {'q8_0_estandar': 8, 'q4_0_estandar': 4}
# Expertos y embedding como en el HNF; atención y MLP compartida desde el original
# con otro formato: cuánto se puede comprimir lo denso sin perder lo ganado en FP16.
DENSE = {'densa_hq52': 'hq52', 'densa_q8_0': 8, 'densa_hq62': 'hq62k'}


def rust_roundtrip(weights, form):
    """Codificador real del conversor (HQ_ROUNDTRIP apunta al binario de ida y vuelta)."""
    values = np.ascontiguousarray(weights, np.float32)
    with tempfile.TemporaryDirectory() as folder:
        source, target = Path(folder)/'in.f32', Path(folder)/'out.f32'
        values.tofile(source)
        subprocess.run([os.environ['HQ_ROUNDTRIP'], form, source, target, '1'], check=True)
        return np.fromfile(target, np.float32).reshape(values.shape)
DENSE_PATTERN = r'\.attn\.[qkvo]_proj|\.mlp\.'


def fake_hq52(weights):
    """Mismo redondeo que quantize_hq52k: bloques de 256, pasos por grupo de 8 en 5 bits."""
    blocks = np.asarray(weights, np.float32).reshape(-1, 32, 8)
    ideal = np.maximum(blocks.max(2).clip(0)/15, -blocks.min(2).clip(None, 0)/16)
    d = ideal.max(1, keepdims=True).astype(np.float16).astype(np.float32)
    # Rust redondea los empates alejándose de cero; np.round los lleva al par.
    away = lambda x: np.sign(x)*np.floor(np.abs(x)+np.float32(.5))
    q = np.where(d > 1e-10, np.clip(away(ideal/np.where(d > 1e-10, d, 1)*np.float32(31)), 1, 31), 0)
    step = (d*q/np.float32(31))[..., None]
    codes = np.clip(away(blocks/np.where(step > 1e-10, step, 1)), -16, 15)*(step > 1e-10)
    return (codes*step).reshape(np.shape(weights)).astype(np.float32)


def hq52_rows(weights):
    """HQ52 por filas, con la cola de columnas rellenada a 256 como en GM4X v2."""
    values = np.asarray(weights, np.float32)
    columns = values.shape[-1]
    padded = np.zeros(values.shape[:-1]+(-(-columns//256)*256,), np.float32)
    padded[..., :columns] = values
    return fake_hq52(padded)[..., :columns]


class Recipe(Reader):
    """Parte densa original (FP16 la guarda exacta); expertos y embedding a elegir."""
    def __init__(self, model, experts, embedding, dense_bits=None):
        super().__init__(model)
        self.experts, self.embedding_source = experts, embedding
        self.dense_bits, self.dense = dense_bits, re.compile(DENSE_PATTERN)

    def get(self, name, expert=None):
        if 'moe.experts' not in name:
            value = super().get(name, expert)
            return fake_quant(value, self.dense_bits) if self.dense_bits and self.dense.search(name) else value
        if self.experts == 'hq52':
            return hq52_rows(super().get(name, expert))
        if self.experts == 'down52':
            return hq52_rows(super().get(name, expert)) if 'down' in name else self.unit.get(name, expert)
        return self.experts.get(name, expert)

    def embedding(self, rows=None):
        if self.embedding_source == 'original':
            return super().embedding(rows)
        return self.embedding_source.embedding(rows)


GGUF_NAMES = {'attn.q_proj': 'attn_q', 'attn.k_proj': 'attn_k', 'attn.v_proj': 'attn_v', 'attn.o_proj': 'attn_output',
              'mlp.gate': 'ffn_gate', 'mlp.up': 'ffn_up', 'mlp.down': 'ffn_down',
              'moe.experts.gate_up': 'ffn_gate_up_exps', 'moe.experts.down': 'ffn_down_exps'}


class Gguf(Reader):
    """Matrices y embedding de un GGUF de llama.cpp; normas, router y escalas del original."""
    def __init__(self, model, path):
        super().__init__(model)
        import gguf
        self.dequantize = gguf.dequantize
        self.tensors_gguf = {t.name: t for t in gguf.GGUFReader(path).tensors}

    def matrix(self, gguf_name, expert=None):
        t = self.tensors_gguf[gguf_name]
        raw = t.data if expert is None else t.data[expert]
        return np.asarray(self.dequantize(raw, t.tensor_type), np.float32)

    def get(self, name, expert=None):
        m = re.fullmatch(r'text\.layer(\d+)\.(.+)\.weight', name)
        if m and m.group(2) in GGUF_NAMES:
            return self.matrix(f'blk.{m.group(1)}.{GGUF_NAMES[m.group(2)]}.weight', expert)
        return super().get(name, expert)

    def embedding(self, rows=None):
        t = self.tensors_gguf['token_embd.weight']
        raw = t.data if rows is None else t.data[rows]
        return np.asarray(self.dequantize(raw, t.tensor_type), np.float32)


def hq44_roundtrip(weights, importance=None):
    """HQ44 del conversor por filas rellenadas a 256, con importancia por columna opcional."""
    values = np.ascontiguousarray(weights, np.float32)
    columns = values.shape[-1]
    physical = -(-columns//256)*256
    padded = np.zeros(values.shape[:-1]+(physical,), np.float32)
    padded[..., :columns] = values
    with tempfile.TemporaryDirectory() as folder:
        source, target, weight = Path(folder)/'in.f32', Path(folder)/'out.f32', Path(folder)/'imp.f32'
        padded.tofile(source)
        command = [os.environ['HQ_ROUNDTRIP'], 'hq44', source, target, str(physical)]
        if importance is not None:
            np.concatenate([importance, np.zeros(physical-columns)]).astype(np.float32).tofile(weight)
            command.append(weight)
        subprocess.run(command, check=True)
        return np.fromfile(target, np.float32).reshape(padded.shape)[..., :columns]


class Hq44Importance(Reader):
    """Expertos del original cuantizados en HQ44 con la importancia de una captura imatrix.

    Expertos con pocos tokens: se mezclan con la media de su capa como si esta aportara
    `smoothing` tokens. Sin captura, importancia unitaria, igual que el HNF H."""
    def __init__(self, model, capture=None, smoothing=64):
        super().__init__(model)
        self.importance = None
        if capture:
            stats = np.load(capture)
            n = stats['n'][..., None].astype(np.float64)
            self.importance = {}
            for kind in ('gate_up', 'down'):
                total = stats[kind]
                layer_mean = total.sum(1, keepdims=True)/np.maximum(n.sum(1, keepdims=True), 1)
                self.importance[kind] = (total+smoothing*layer_mean)/(n+smoothing)

    def get(self, name, expert=None):
        value = super().get(name, expert)
        if 'moe.experts' not in name:
            return value
        if self.importance is None:
            return hq44_roundtrip(value)
        layer = int(re.search(r'layer(\d+)\.', name).group(1))
        kind = 'gate_up' if 'gate_up' in name else 'down'
        return hq44_roundtrip(value, self.importance[kind][layer, expert])


class Hq44Blocks(Reader):
    """Expertos HQ44 de ficheros de bloques por capa (layerN.gate_up.hq44, layerN.down.hq44),
    con el orden del HNF: experto, fila, superbloque."""
    def __init__(self, model, folder):
        super().__init__(model)
        self.folder = Path(folder)

    def get(self, name, expert=None):
        if 'moe.experts' not in name:
            return super().get(name, expert)
        layer = int(re.search(r'layer(\d+)\.', name).group(1))
        kind = 'gate_up' if 'gate_up' in name else 'down'
        rows, logical = (2*I, H) if kind == 'gate_up' else (H, I)
        physical = -(-logical//256)*256
        stride = rows*physical//256*152
        raw = np.memmap(self.folder/f'layer{layer}.{kind}.hq44', mode='r', dtype=np.uint8, offset=expert*stride, shape=(rows, physical//256*152))
        decoded = decode_hq44(np.asarray(raw)).reshape(rows, physical)
        assert np.isfinite(decoded).all() and np.all(decoded[:, logical:] == 0), 'bad HQ44 values or padding'
        return decoded[:, :logical].copy()


def recipes(model, dense_hnf):
    """Candidatas al HNF final: todas con atención y MLP compartida en FP16."""
    unit = Reader(model, dense_hnf)
    if os.environ.get('IMATRIX'):
        # Expertos calibrados frente a los mismos expertos sin calibrar, el resto original.
        found = {'receta_hq44_unidad_emb16': lambda: Recipe(model, Hq44Importance(model), 'original'),
                 'receta_hq44_imatrix_emb16': lambda: Recipe(model, Hq44Importance(model, os.environ['IMATRIX']), 'original')}
        if os.environ.get('GPTQ_DIR'):
            found['receta_hq44_gptq_emb16'] = lambda: Recipe(model, Hq44Blocks(model, os.environ['GPTQ_DIR']), 'original')
        if os.environ.get('GGUF'):
            def gguf_experts():
                recipe, source = Recipe(model, None, 'original'), Gguf(model, os.environ['GGUF'])
                recipe.experts = source
                return recipe
            # Solo los expertos de Unsloth; embedding original, como las demás de este grupo.
            found['gguf_expertos_emb16'] = gguf_experts
        return found
    if os.environ.get('GGUF'):
        def gguf_experts():
            recipe, source = Recipe(model, None, 'original'), Gguf(model, os.environ['GGUF'])
            recipe.experts, recipe.embedding_source = source, source
            return recipe

        def down52():
            recipe = Recipe(model, 'down52', unit)
            recipe.unit = unit
            return recipe
        return {'gguf_q4_k_m': lambda: Gguf(model, os.environ['GGUF']),
                # Expertos y embedding del GGUF con la parte densa sin cuantizar.
                'gguf_expertos_densa16': gguf_experts,
                # Como reparte Unsloth: más bits en down que en gate/up.
                'receta_down52': down52}
    sn = Reader(model, os.environ['SN_HNF'])
    return {
        'receta_d_emb16': lambda: Recipe(model, unit, 'original'),
        'receta_sn': lambda: Recipe(model, sn, sn),
        'receta_sn_emb16': lambda: Recipe(model, sn, 'original'),
        'receta_exp52': lambda: Recipe(model, 'hq52', unit),
        'receta_exp52_emb16': lambda: Recipe(model, 'hq52', 'original'),
        'receta_exp52_emb16_q8': lambda: Recipe(model, 'hq52', 'original', 8),
        'receta_exp52_q8': lambda: Recipe(model, 'hq52', unit, 8),
    }


def fake_quant(weights, bits):
    blocks = np.asarray(weights, np.float32).reshape(-1, 32)
    if bits == 8:
        scale = np.abs(blocks).max(1, keepdims=True)/127
        codes = np.round(blocks/np.where(scale == 0, 1, scale))
    else:
        # Q4_0: escala con el signo del máximo absoluto y códigos 0..15 centrados en 8.
        peak = np.take_along_axis(blocks, np.abs(blocks).argmax(1)[:, None], 1)
        scale = peak/-8
        codes = np.clip(np.round(blocks/np.where(scale == 0, 1, scale)), -8, 7)
    return (codes*scale).reshape(np.shape(weights)).astype(np.float32)


class FakeQuant(Reader):
    """El original con todas las matrices cuantizadas por un formato estándar."""
    def __init__(self, model, bits):
        super().__init__(model)
        self.bits = bits

    def get(self, name, expert=None):
        value = super().get(name, expert)
        return fake_quant(value, self.bits) if np.ndim(value) >= 2 else value

    def embedding(self, rows=None):
        return fake_quant(super().embedding(rows), self.bits)


class DenseFormat(OnePiece):
    def __init__(self, model, hnf, form):
        super().__init__(model, hnf, r'moe\.experts|token_embedding')
        self.form, self.dense = form, re.compile(DENSE_PATTERN)

    def get(self, name, expert=None):
        value = super().get(name, expert)
        if not self.dense.search(name):
            return value
        if self.form == 'hq52':
            return fake_hq52(value)
        return rust_roundtrip(value, self.form) if isinstance(self.form, str) else fake_quant(value, self.form)


def load_cases(cases):
    names = sorted(p.name[:-len('.forcing.ids')] for p in Path(cases).glob('*.forcing.ids'))
    return names, [np.fromfile(Path(cases)/f'{n}.forcing.ids', dtype='<i4')[:MAX_FORCING] for n in names]


def norm(x, weight, dim):
    """rmsnorm por cabeza sobre [n, cabezas, dim]."""
    y = x/np.sqrt(np.mean(x*x, axis=-1, keepdims=True, dtype=np.float32)+np.float32(1e-6))
    return y if weight is None else y*weight


def rope(x, n, head_dim, is_global):
    if is_global:
        pairs, theta, base, second = int(.25*head_dim/2), 1e6, head_dim, head_dim//2
    else:
        pairs, theta, base, second = head_dim//2, 1e4, head_dim, head_dim//2
    freq = 1/np.power(np.float32(theta), 2*np.arange(pairs, dtype=np.float32)/base)
    angle = np.arange(n, dtype=np.float32)[:, None]*freq
    cos, sin = np.cos(angle)[:, None], np.sin(angle)[:, None]
    a, b = x[..., :pairs].copy(), x[..., second:second+pairs].copy()
    x[..., :pairs] = a*cos-b*sin
    x[..., second:second+pairs] = a*sin+b*cos
    return x


def attention(hidden, get, layer):
    is_global = layer in GLOBAL_LAYERS
    head_dim, kv_heads = (512, 2) if is_global else (256, 8)
    p = f'text.layer{layer}.'
    n = len(hidden)
    x = rms(hidden, get(p+'ln_attn_in.weight'))
    k_proj = get(p+'attn.k_proj.weight')
    v_proj = get(p+'attn.v_proj.weight') if p+'attn.v_proj.weight' in reader.tensors else k_proj
    q = (x@get(p+'attn.q_proj.weight').T).reshape(n, HEADS, head_dim)
    k = (x@k_proj.T).reshape(n, kv_heads, head_dim)
    v = (x@v_proj.T).reshape(n, kv_heads, head_dim)
    q = rope(norm(q, get(p+'attn.q_norm.weight'), head_dim), n, head_dim, is_global)
    k = rope(norm(k, get(p+'attn.k_norm.weight'), head_dim), n, head_dim, is_global)
    v = norm(v, None, head_dim)
    out = np.empty((n, HEADS, head_dim), np.float32)
    mask = np.triu(np.ones((n, n), bool), 1)
    for h in range(HEADS):
        kv = h//(HEADS//kv_heads)
        scores = q[:, h]@k[:, kv].T
        scores[mask] = -np.inf
        scores -= scores.max(1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(1, keepdims=True)
        out[:, h] = weights@v[:, kv]
    return out.reshape(n, HEADS*head_dim)@get(p+'attn.o_proj.weight').T


# Con el modo imatrix, suma de x² por columna de la entrada de cada experto (estilo imatrix).
STATS = None


def tail(hidden, get, layer):
    """MLP compartida, router y expertos sobre todos los tokens a la vez."""
    p = f'text.layer{layer}.'
    x = rms(hidden, get(p+'ln_mlp_in.weight'))
    activation = gelu(x@get(p+'mlp.gate.weight').T)*(x@get(p+'mlp.up.weight').T)
    shared = activation@get(p+'mlp.down.weight')[:, :SHARED].T
    shared = rms(shared, get(p+'ln_mlp_branch_post.weight'))
    router = rms(hidden)*get(p+'moe.router.input_scale')*np.float32(H**-.5)
    ids, weights = route(router@get(p+'moe.router.weight').T, get(p+'moe.router.expert_scale'))
    x = rms(hidden, get(p+'ln_moe_in.weight'))
    total = np.zeros_like(hidden)
    used = [int(e) for e in np.unique(ids)]
    # Descodificar HQ44 ocupa casi todo el tiempo y libera el GIL.
    with ThreadPoolExecutor(14) as pool:
        matrices = list(pool.map(lambda e: (get(p+'moe.experts.gate_up.weight', e), get(p+'moe.experts.down.weight', e)), used))
    for expert, (gate_up_weight, down) in zip(used, matrices):
        tokens, slots = np.where(ids == expert)
        gate_up = x[tokens]@gate_up_weight.T
        activation = gelu(gate_up[:, :I])*gate_up[:, I:]
        rows = activation@down[:, :I].T
        np.add.at(total, tokens, rows*weights[tokens, slots, None])
        if STATS is not None:
            STATS['gate_up'][layer, expert] += (x[tokens].astype(np.float64)**2).sum(0)
            STATS['down'][layer, expert] += (activation.astype(np.float64)**2).sum(0)
            STATS['n'][layer, expert] += len(tokens)
    mlp = rms(shared+rms(total, get(p+'ln_moe_branch_post.weight')), get(p+'ln_mlp_post.weight'))
    return (hidden+mlp)*get(p+'layer_scalar'), ids.astype(np.uint8)


def forward(variant, weights, cases, model, output, keep=range(LAYERS)):
    global reader
    reader = weights if isinstance(weights, Reader) else Reader(model, None if weights == '-' else weights)
    output = Path(output)/variant
    output.mkdir(parents=True, exist_ok=True)
    names, ids = load_cases(cases)
    offsets = np.cumsum([0]+[len(a) for a in ids])
    tokens = np.concatenate(ids)
    unique, inverse = np.unique(tokens, return_inverse=True)
    hidden = (reader.embedding(unique)[inverse]*np.float32(H**.5)).astype(np.float32)
    start = time.time()
    for layer in range(LAYERS):
        cache = {}

        def get(name, expert=None):
            key = (name, expert)
            if key not in cache:
                value = np.asarray(reader.get(name, expert), np.float32)
                if expert is None:
                    cache[key] = value
                return value
            return cache[key]
        for i in range(len(names)):
            part = hidden[offsets[i]:offsets[i+1]]
            part += rms(attention(part, get, layer), get(f'text.layer{layer}.ln_attn_post.weight'))
        hidden, routed = tail(hidden, get, layer)
        if layer in keep:
            np.save(output/f'hidden-{layer:02}.npy', hidden)
            np.save(output/f'ids-{layer:02}.npy', routed)
        print(variant, 'capa', layer, f'{time.time()-start:.0f} s', flush=True)


def softcap(x):
    return np.tanh(x/np.float32(30))*np.float32(30)


def head(hidden, final, embedding):
    return softcap(rms(np.asarray(hidden), final)@embedding.T).astype(np.float64)


def log_probs(logits):
    shifted = logits-logits.max(1, keepdims=True)
    return shifted-np.log(np.exp(shifted).sum(1, keepdims=True))


def compare(cases, model, hnf_a, hnf_b, engine, prompts, output):
    global reader
    names, ids = load_cases(cases)
    output = Path(output)
    offsets = np.cumsum([0]+[len(a) for a in ids])
    domains = {f"{i:02}-{json.loads(l)['id']}": json.loads(l)['domain'] for i, l in enumerate(Path(prompts).read_text().splitlines())}
    readers = {'original': Reader(model), 'A': Reader(model, hnf_a), 'B': Reader(model, hnf_b)}
    embeddings, finals = {}, {}
    for variant, r in readers.items():
        embeddings[variant] = np.asarray(r.embedding(), np.float32)
        reader = r
        finals[variant] = np.asarray(r.get('text.final_norm.weight'), np.float32)
        print('cabeza', variant, flush=True)
    layers = {}
    reference = {}
    for layer in range(LAYERS):
        o = np.load(output/'original'/f'hidden-{layer:02}.npy', mmap_mode='r').astype(np.float64)
        ids_o = np.load(output/'original'/f'ids-{layer:02}.npy')
        row = {}
        for variant in ('A', 'B'):
            x = np.load(output/variant/f'hidden-{layer:02}.npy', mmap_mode='r').astype(np.float64)
            ids_x = np.load(output/variant/f'ids-{layer:02}.npy')
            token = np.sqrt(((x-o)**2).sum(-1)/(o**2).sum(-1))
            overlap = np.mean([len(set(a) & set(b))/K for a, b in zip(ids_o, ids_x)])
            row[variant] = {'relative_l2': float(np.sqrt(((x-o)**2).sum()/(o**2).sum())), 'token_p50': float(np.median(token)),
                            'token_p95': float(np.percentile(token, 95)), 'router_overlap': float(overlap)}
        layers[layer] = row
        print('capa', layer, {v: round(r['relative_l2'], 5) for v, r in row.items()}, flush=True)
    documents = {}
    for i, name in enumerate(names):
        n = len(ids[i])
        if n-1-FIRST <= 0:
            continue
        positions = offsets[i]+FIRST+np.arange(n-1-FIRST)
        targets = ids[i][FIRST+1:]
        sums = {v: dict(kl=0., agree=0., top8=0., nll=0., engine_diff=0., engine_energy=0., engine_agree=0.) for v in ('A', 'B')}
        original_nll = 0.
        hidden = {v: np.load(output/v/'hidden-29.npy', mmap_mode='r') for v in readers}
        for start in range(0, len(positions), CHUNK):
            part = positions[start:start+CHUNK]
            target = targets[start:start+CHUNK]
            rows_index = np.arange(len(part))
            logp_o = log_probs(head(hidden['original'][part], finals['original'], embeddings['original']))
            p_o = np.exp(logp_o)
            original_nll -= logp_o[rows_index, target].sum()
            top_o = np.argpartition(-logp_o, 8, axis=1)[:, :8]
            for variant, run in (('A', 'A1'), ('B', 'B1')):
                cpu = head(hidden[variant][part], finals[variant], embeddings[variant])
                motor = np.memmap(Path(engine)/run/f'{name}.logits.f16', dtype='<f2', mode='r').reshape(len(targets), -1)[start:start+CHUNK].astype(np.float64)
                logp = log_probs(cpu)
                top_x = np.argpartition(-logp, 8, axis=1)[:, :8]
                total = sums[variant]
                total['kl'] += (p_o*(logp_o-logp)).sum()
                total['agree'] += (logp_o.argmax(1) == logp.argmax(1)).sum()
                total['top8'] += sum(len(set(a) & set(b))/8 for a, b in zip(top_o, top_x))
                total['nll'] -= logp[rows_index, target].sum()
                total['engine_diff'] += ((cpu-motor)**2).sum()
                total['engine_energy'] += (motor**2).sum()
                total['engine_agree'] += (cpu.argmax(1) == motor.argmax(1)).sum()
        count = len(positions)
        rows = {'original_nll': original_nll/count}
        for variant, total in sums.items():
            rows[variant] = {'kl_original_to_variant': total['kl']/count, 'argmax_agreement': float(total['agree'])/count,
                             'top8_overlap': total['top8']/count, 'nll': total['nll']/count,
                             'cpu_vs_engine_l2': float(np.sqrt(total['engine_diff']/total['engine_energy'])),
                             'cpu_vs_engine_argmax': float(total['engine_agree'])/count}
        documents[name] = rows
        print(name, f"KL A {rows['A']['kl_original_to_variant']:.5f} B {rows['B']['kl_original_to_variant']:.5f}", flush=True)
    keys = list(documents)
    delta = np.array([documents[n]['B']['kl_original_to_variant']-documents[n]['A']['kl_original_to_variant'] for n in keys])
    rng = np.random.default_rng(26)
    groups = {'total': np.ones(len(keys), bool)}
    kinds = np.array([domains[n] for n in keys])
    groups.update({str(d): kinds == d for d in sorted(set(kinds))})
    summary = {}
    for group, mask in groups.items():
        means = [delta[mask][rng.integers(mask.sum(), size=mask.sum())].mean() for _ in range(2000)]
        chosen = [documents[n] for n, m in zip(keys, mask) if m]
        summary[group] = {'documents': int(mask.sum()), 'mean_delta_kl_b_minus_a': float(delta[mask].mean()), 'ci95': [float(x) for x in np.percentile(means, [2.5, 97.5])],
                          'kl_a': float(np.mean([d['A']['kl_original_to_variant'] for d in chosen])), 'kl_b': float(np.mean([d['B']['kl_original_to_variant'] for d in chosen])),
                          'argmax_a': float(np.mean([d['A']['argmax_agreement'] for d in chosen])), 'argmax_b': float(np.mean([d['B']['argmax_agreement'] for d in chosen])),
                          'nll_original': float(np.mean([d['original_nll'] for d in chosen])), 'nll_a': float(np.mean([d['A']['nll'] for d in chosen])), 'nll_b': float(np.mean([d['B']['nll'] for d in chosen])),
                          'better_b': int((delta[mask] < 0).sum()), 'worse_b': int((delta[mask] > 0).sum())}
        print(group, summary[group], flush=True)
    (output/'fidelity.json').write_text(json.dumps({'kind': 'gemma4_layerwise_fidelity', 'label': 'pesos BF16 originales, aritmética FP32 CPU', 'summary': summary, 'layers': layers, 'documents': documents}, indent=1))


def pieces(cases, model, hnf, output, *chosen):
    output = Path(output)
    names, ids = load_cases(cases)
    offsets = np.cumsum([0]+[len(a) for a in ids])
    positions = np.concatenate([offsets[i]+FIRST+np.arange(len(ids[i])-1-FIRST) for i in range(len(names))])
    counts = np.array([max(len(ids[i])-1-FIRST, 0) for i in range(len(names))])
    document_of = np.repeat(np.arange(len(names)), counts)
    base = Reader(model)
    global reader
    reader = base
    final, embedding = np.asarray(base.get('text.final_norm.weight'), np.float32), np.asarray(base.embedding(), np.float32)
    hidden_o = np.load(output/'original'/'hidden-29.npy', mmap_mode='r')
    kept = (0, 5, 10, 15, 20, 25, 29)
    report = {}
    built = recipes(model, hnf) if any(c.startswith(('receta_', 'gguf_')) for c in chosen) else {}
    for piece, pattern in {**PIECES, **BASELINES, **DENSE, **built}.items():
        if chosen and piece not in chosen:
            continue
        variant = 'solo-'+piece
        one = (FakeQuant(model, pattern) if piece in BASELINES else DenseFormat(model, hnf, pattern) if piece in DENSE
               else pattern() if piece in built else OnePiece(model, hnf, pattern))
        forward(variant, one, cases, model, output, kept)
        reader = one
        final_x = np.asarray(one.get('text.final_norm.weight'), np.float32)
        own_head = piece in BASELINES or piece in DENSE or piece in built or re.search(pattern if piece in PIECES else '', 'token_embedding')
        embedding_x = np.asarray(one.embedding(), np.float32) if own_head else embedding
        hidden_x = np.load(output/variant/'hidden-29.npy', mmap_mode='r')
        kl, agree = 0., 0.
        per_document = np.zeros(len(names))
        for start in range(0, len(positions), CHUNK):
            part = positions[start:start+CHUNK]
            logp_o = log_probs(head(hidden_o[part], final, embedding))
            logp = log_probs(head(hidden_x[part], final_x, embedding_x))
            value = (np.exp(logp_o)*(logp_o-logp)).sum(1)
            kl += value.sum()
            np.add.at(per_document, document_of[start:start+CHUNK], value)
            agree += (logp_o.argmax(1) == logp.argmax(1)).sum()
        row = {'kl_token': kl/len(positions), 'argmax_agreement': float(agree)/len(positions),
               'kl_documents': dict(zip(names, (per_document/counts).tolist())), 'layers': {}}
        for layer in kept:
            o = np.load(output/'original'/f'hidden-{layer:02}.npy', mmap_mode='r').astype(np.float64)
            x = np.load(output/variant/f'hidden-{layer:02}.npy', mmap_mode='r').astype(np.float64)
            ids_o, ids_x = np.load(output/'original'/f'ids-{layer:02}.npy'), np.load(output/variant/f'ids-{layer:02}.npy')
            row['layers'][layer] = {'relative_l2': float(np.sqrt(((x-o)**2).sum()/(o**2).sum())),
                                    'router_overlap': float(np.mean([len(set(a) & set(b))/K for a, b in zip(ids_o, ids_x)]))}
        report[piece] = row
        print(piece, round(row['kl_token'], 4), round(row['argmax_agreement'], 3), {k: round(v['relative_l2']*100, 2) for k, v in row['layers'].items()}, flush=True)
    path = output/'pieces.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**previous, **report}, indent=1))


def engine_fidelity(cases, model, engine, output, *runs):
    output = Path(output)
    names, ids = load_cases(cases)
    offsets = np.cumsum([0]+[len(a) for a in ids])
    global reader
    reader = Reader(model)
    final, embedding = np.asarray(reader.get('text.final_norm.weight'), np.float32), np.asarray(reader.embedding(), np.float32)
    hidden_o = np.load(output/'original'/'hidden-29.npy', mmap_mode='r')
    report = {}
    for run in runs:
        kl, agree, count, per_document = 0., 0., 0, {}
        for i, name in enumerate(names):
            positions = offsets[i]+FIRST+np.arange(len(ids[i])-1-FIRST)
            logits = np.memmap(Path(engine)/run/f'{name}.logits.f16', dtype='<f2', mode='r').reshape(len(positions), -1)
            document = 0.
            for start in range(0, len(positions), CHUNK):
                logp_o = log_probs(head(hidden_o[positions[start:start+CHUNK]], final, embedding))
                logp = log_probs(logits[start:start+CHUNK].astype(np.float64))
                value = (np.exp(logp_o)*(logp_o-logp)).sum()
                kl += value
                document += value
                agree += (logp_o.argmax(1) == logp.argmax(1)).sum()
            count += len(positions)
            per_document[name] = document/len(positions)
        report[run] = {'kl_token': kl/count, 'kl_document_mean': float(np.mean(list(per_document.values()))),
                       'argmax_agreement': float(agree)/count, 'documents': per_document}
        print(run, round(report[run]['kl_token'], 4), round(report[run]['kl_document_mean'], 4), round(report[run]['argmax_agreement'], 3), flush=True)
    path = output/'engine-fidelity.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**previous, **report}, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'forward':
        forward(*sys.argv[2:7])
    elif sys.argv[1] == 'pieces':
        pieces(*sys.argv[2:])
    elif sys.argv[1] == 'engine':
        engine_fidelity(*sys.argv[2:])
    elif sys.argv[1] == 'imatrix':
        # imatrix casos-calibración/ modelo-original/ salida.npz
        STATS = {'gate_up': np.zeros((LAYERS, E, H)), 'down': np.zeros((LAYERS, E, I)), 'n': np.zeros((LAYERS, E), np.int64)}
        with tempfile.TemporaryDirectory() as folder:
            forward('calibracion', '-', sys.argv[2], sys.argv[3], folder, ())
        np.savez(sys.argv[4], **STATS)
    elif sys.argv[1] == 'export-imatrix':
        # export-imatrix modelo-original/ captura.npz carpeta/: la misma importancia
        # que mide Hq44Importance, en f32 [expertos, columnas físicas] para --importance.
        importance, folder = Hq44Importance(sys.argv[2], sys.argv[3]).importance, Path(sys.argv[4])
        folder.mkdir()
        for layer in range(LAYERS):
            for kind, values in importance.items():
                padded = np.zeros((E, -(-values.shape[-1]//256)*256), np.float32)
                padded[:, :values.shape[-1]] = values[layer]
                padded.tofile(folder/f'layer{layer}.{kind}.f32')
    else:
        compare(*sys.argv[2:])
