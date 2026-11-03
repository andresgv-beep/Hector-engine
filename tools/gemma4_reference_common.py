"""Pesos y primitivas numéricas de las referencias CPU de Gemma 4 26B MoE.

Weights da cada tensor por su nombre canónico desde un HNFv10 (`packed`) o desde el checkpoint
original BF16 (`original`). El HNF guarda de cada tensor su nombre en el checkpoint y el
xxh3-64 de sus bytes, así que los originales se comprueban la primera vez que se leen.
"""
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from hnf10 import Hnf10, xxh3_64

LAYERS, E, H, I, P, K, SH = 30, 128, 2816, 704, 768, 8, 2112


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def h16(x):
    result = np.asarray(x, np.float32).astype(np.float16).astype(np.float32)
    require(np.isfinite(result).all(), 'FP16 overflow')
    return result


def rms(x, w=None):
    y = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True, dtype=np.float32) + np.float32(1e-6))
    return y if w is None else y*w


def gelu(x):
    return np.float32(.5)*x*(1 + np.tanh(np.float32(.79788456)*(x + np.float32(.044715)*x*x*x)))


def route(logits, scale, k=K):
    """Top-k de la softmax con desempate por ID ascendente, renormalizado y por per_expert_scale."""
    probs = np.exp(logits-logits.max(-1, keepdims=True), dtype=np.float32)
    probs /= probs.sum(-1, keepdims=True, dtype=np.float32)
    ids = np.argsort(-probs, axis=-1, kind='stable')[:, :k]
    weights = np.take_along_axis(probs, ids, axis=-1)
    weights = weights/weights.sum(-1, keepdims=True)*scale[ids]
    return ids, weights


def safetensors_header(path):
    """Cabecera JSON de un shard y el offset donde empiezan sus datos."""
    with Path(path).open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        require(0 < size <= min(100_000_000, Path(path).stat().st_size - 8), f'cabecera no válida: {path}')
        info = json.loads(f.read(size))
    info.pop('__metadata__', None)
    return info, 8 + size


class Weights:
    """HNFv10 y checkpoint original por el mismo nombre canónico.

    expected_sha: SHA-256 del HNF con el que se hizo una captura, si la herramienta la compara.
    verify: comprobar los bytes originales contra la procedencia del HNF (cuesta leer el
    tensor entero una vez)."""
    def __init__(self, hnf, model, expected_sha=None, verify=True):
        self.hnf, self.model = Path(hnf), Path(model)
        if expected_sha is not None:
            require(sha(self.hnf) == expected_sha, 'el SHA-256 del HNF no es el de la captura')
        self.hnf10 = Hnf10(self.hnf)
        self.tensors = self.hnf10.tensors
        self.index = json.loads((self.model/'model.safetensors.index.json').read_text())['weight_map']
        self.headers, self.verify, self.verified = {}, verify, set()

    def packed(self, name, expert=None):
        return self.hnf10.tensor(name, expert)

    def original(self, name, expert=None):
        t = self.tensors[name]
        shard = self.model/self.index[t.source]
        if shard not in self.headers:
            self.headers[shard] = safetensors_header(shard)
        info, base = self.headers[shard]
        s = info[t.source]
        require(s['dtype'] == 'BF16', f'{t.source}: se espera BF16 en el original')
        raw = np.memmap(shard, mode='r', dtype='<u2', offset=base+s['data_offsets'][0], shape=tuple(s['shape']))
        if self.verify and name not in self.verified:
            require(xxh3_64(raw) == t.source_hash, f'los bytes originales cambiaron: {name}')
            self.verified.add(name)
        if expert is not None:
            raw = raw[expert]
        return (raw.astype(np.uint32) << 16).view(np.float32)
