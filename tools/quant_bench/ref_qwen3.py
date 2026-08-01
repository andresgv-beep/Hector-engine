#!/usr/bin/env python3
"""Referencia fp32 independiente de Qwen3, en numpy sobre el safetensors de HF.
Emite el argmax de CADA posicion para comparar contra Hector y contra Ollama."""
import json, struct, sys, math, os, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hqs_sim import quant_dequant

# Simula el formato sobre los pesos que de verdad se cuantizan. Todo lo demas
# (normas, embeddings) va en fp16 igual que en el conversor real.
_Q = float(os.environ.get('HQS_QMAX', '0'))       # todas las capas
_QMLP = float(os.environ.get('HQS_QMLP', '0'))    # solo MLP (perfil mixto)
_QATT = float(os.environ.get('HQS_QATT', '0'))    # solo atencion
_QGATE = float(os.environ.get('HQS_QGATE', '0'))  # override gate_proj
_QUP = float(os.environ.get('HQS_QUP', '0'))       # override up_proj
_QDOWN = float(os.environ.get('HQS_QDOWN', '0'))   # override down_proj
_QLAYER_START = int(os.environ.get('HQS_LAYER_START', '0'))
_QLAYER_END = int(os.environ.get('HQS_LAYER_END', str(1 << 30)))
_MLP = ('gate_proj','up_proj','down_proj')
_ATT = ('q_proj','k_proj','v_proj','o_proj')
def _scoped_override(name, value):
    if value <= 0:
        return 0.0
    match = re.search(r'model\.layers\.(\d+)\.', name)
    if not match:
        return value
    layer = int(match.group(1))
    return value if _QLAYER_START <= layer <= _QLAYER_END else 0.0

def _q(name, w):
    if 'gate_proj' in name:
        q = _scoped_override(name, _QGATE) or _QMLP or _Q
    elif 'up_proj' in name:
        q = _scoped_override(name, _QUP) or _QMLP or _Q
    elif 'down_proj' in name:
        q = _scoped_override(name, _QDOWN) or _QMLP or _Q
    elif any(k in name for k in _MLP):
        q = _QMLP or _Q
    elif any(k in name for k in _ATT):
        q = _QATT or _Q
    else:
        return w
    return quant_dequant(w, q) if q > 0 else w

class Shards:
    """Safetensors repartido en varios ficheros, leido por mmap."""
    def __init__(self, d):
        self.h, self.mm, self.base = {}, {}, {}
        for p in sorted(glob.glob(f'{d}/*.safetensors')):
            with open(p,'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                hdr = json.loads(f.read(n))
            hdr.pop('__metadata__', None)
            self.mm[p] = np.memmap(p, dtype=np.uint8, mode='r')
            self.base[p] = 8 + n
            for k, v in hdr.items(): self.h[k] = (p, v)
    def f32(self, name):
        p, e = self.h[name]; a, b = e['data_offsets']
        raw = self.mm[p][self.base[p]+a : self.base[p]+b].tobytes()
        arr = ((np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
               if e['dtype']=='BF16' else np.frombuffer(raw, np.float16).astype(np.float32))
        return _q(name, arr.reshape(e['shape']))
    def rows(self, name, idx):
        p, e = self.h[name]; cols = e['shape'][1]; a = e['data_offsets'][0]
        w = 2; out = np.empty((len(idx), cols), np.float32)
        for j, i in enumerate(idx):
            off = self.base[p] + a + i*cols*w
            raw = self.mm[p][off:off+cols*w].tobytes()
            out[j] = ((np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
                      if e['dtype']=='BF16' else np.frombuffer(raw, np.float16).astype(np.float32))
        # hidden=2560 es multiplo de 256, asi que las filas caen en frontera de
        # superbloque: cuantizar solo las pedidas da el mismo resultado.
        _E = float(os.environ.get('HQS_EMB', '0'))
        return quant_dequant(out, _E) if _E > 0 else out

def rmsnorm(x, w, eps):
    v = x.astype(np.float32)
    return v * np.power(np.mean(v*v, -1, keepdims=True) + eps, -0.5) * w

def rope(x, pos, ivf):
    ang = pos[:,None] * ivf[None,:]
    cos, sin = np.cos(ang)[:,None,:], np.sin(ang)[:,None,:]
    x1, x2 = np.split(x, 2, -1)
    return np.concatenate([x1*cos - x2*sin, x2*cos + x1*sin], -1)

def main(d, toks, out):
    c = json.load(open(f'{d}/config.json'))
    st = Shards(d)
    L, D, eps = c['num_hidden_layers'], c['hidden_size'], c['rms_norm_eps']
    H, KVH = c['num_attention_heads'], c['num_key_value_heads']
    hd = c.get('head_dim') or D//H                       # Qwen3: 128 explicito, NO D//H
    ids = np.asarray(toks, np.int64); S = len(ids)
    pos = np.arange(S, dtype=np.float32)
    ivf = (1.0/(c['rope_theta'] ** (np.arange(0, hd, 2, dtype=np.float64)/hd))).astype(np.float32)
    scale = 1.0/math.sqrt(hd)
    mask = np.tril(np.ones((S,S), bool))
    print(f'[qwen] {S} tokens, {L} capas, hd={hd}, GQA {H}/{KVH}', file=sys.stderr)

    h = st.rows('model.embed_tokens.weight', ids)
    for i in range(L):
        w = lambda s: st.f32(f'model.layers.{i}.{s}')
        res = h
        x = rmsnorm(h, w('input_layernorm.weight'), eps)
        q = rmsnorm((x @ w('self_attn.q_proj.weight').T).reshape(S,H,hd), w('self_attn.q_norm.weight'), eps)
        k = rmsnorm((x @ w('self_attn.k_proj.weight').T).reshape(S,KVH,hd), w('self_attn.k_norm.weight'), eps)
        v = (x @ w('self_attn.v_proj.weight').T).reshape(S,KVH,hd)
        q, k = rope(q,pos,ivf), rope(k,pos,ivf)
        rep = H//KVH
        k = np.repeat(k, rep, 1); v = np.repeat(v, rep, 1)     # GQA
        sc = np.einsum('shd,thd->hst', q, k) * scale
        sc = np.where(mask[None], sc, -np.inf)
        sc -= sc.max(-1, keepdims=True)
        p = np.exp(sc); p /= p.sum(-1, keepdims=True)
        a = np.einsum('hst,thd->shd', p, v).reshape(S, H*hd)
        h = res + a @ w('self_attn.o_proj.weight').T
        res = h
        x = rmsnorm(h, w('post_attention_layernorm.weight'), eps)
        g = x @ w('mlp.gate_proj.weight').T
        x = (g/(1.0+np.exp(-g))) * (x @ w('mlp.up_proj.weight').T)   # SwiGLU
        h = res + x @ w('mlp.down_proj.weight').T
        print(f'\r[qwen] capa {i+1}/{L}', end='', file=sys.stderr)
    print(file=sys.stderr)
    h = rmsnorm(h, st.f32('model.norm.weight'), eps)
    emb = st.f32('model.embed_tokens.weight')             # tie_word_embeddings
    am = np.empty(S, np.int32)
    for i in range(0, S, 64):
        am[i:i+64] = np.argmax(h[i:i+64] @ emb.T, 1)
    am.tofile(out)
    print(f'[qwen] {S} argmax -> {out}', file=sys.stderr)

if __name__ == '__main__':
    main(sys.argv[1], [int(x) for x in sys.argv[2].split(',')], sys.argv[3])
