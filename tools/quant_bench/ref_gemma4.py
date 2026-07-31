#!/usr/bin/env python3
"""Referencia fp32 independiente de Gemma 4, en numpy sobre el safetensors de HF.

No comparte una linea de codigo con Hector. Las formulas salen de
transformers/models/gemma4/modeling_gemma4.py y modeling_rope_utils.py.
Emite los logits de la ultima posicion para comparar contra el motor.
"""
import json, struct, sys, math
import numpy as np

class Safetensors:
    """Lector mmap. Solo materializa en fp32 lo que se pide."""
    def __init__(self, path):
        with open(path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            self.hdr = json.loads(f.read(n))
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode='r')
        self.hdr.pop('__metadata__', None)

    def raw(self, name):
        e = self.hdr[name]
        assert e['dtype'] == 'BF16', e['dtype']
        a, b = e['data_offsets']
        buf = self.mm[self.base + a: self.base + b]
        return np.frombuffer(buf.tobytes(), dtype=np.uint16).reshape(e['shape'])

    def f32(self, name):
        u = self.raw(name).astype(np.uint32) << 16
        return u.view(np.float32)

    def rows(self, name, idx):
        """Gather de filas sin materializar la tabla entera (tablas de 4.7 GB)."""
        e = self.hdr[name]
        rows_, cols = e['shape']
        a = e['data_offsets'][0]
        out = np.empty((len(idx), cols), dtype=np.float32)
        for j, i in enumerate(idx):
            off = self.base + a + i * cols * 2
            u16 = np.frombuffer(self.mm[off: off + cols * 2].tobytes(), dtype=np.uint16)
            out[j] = (u16.astype(np.uint32) << 16).view(np.float32)
        return out

def rmsnorm(x, w, eps):
    v = x.astype(np.float32)
    ms = np.mean(v * v, axis=-1, keepdims=True)
    n = v * np.power(ms + eps, -0.5)
    return n * w if w is not None else n

def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

def inv_freq(head_dim, theta, rope_type, partial):
    """default: rota todo. proportional: rota int(partial*hd//2) angulos y el
       resto queda a CERO (dimensiones sin posicion). El exponente divide por
       head_dim completo en ambos casos."""
    half = head_dim // 2
    if rope_type == 'proportional':
        k = int(partial * head_dim // 2)
        f = 1.0 / (theta ** (np.arange(0, 2 * k, 2, dtype=np.float64) / head_dim))
        return np.concatenate([f, np.zeros(half - k)]).astype(np.float32)
    f = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    return f.astype(np.float32)

def rope(x, pos, ivf):
    """x: [s, heads, hd]. Rotacion por mitades (convencion HF)."""
    ang = pos[:, None] * ivf[None, :]                  # [s, hd/2]
    cos = np.cos(ang)[:, None, :]
    sin = np.sin(ang)[:, None, :]
    x1, x2 = np.split(x, 2, axis=-1)
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

def main(model_dir, tokens, out_path):
    cfg = json.load(open(f'{model_dir}/config.json'))['text_config']
    st = Safetensors(f'{model_dir}/model.safetensors')
    P = 'model.language_model.'
    L   = cfg['num_hidden_layers']; D = cfg['hidden_size']
    PLE = cfg['hidden_size_per_layer_input']; eps = cfg['rms_norm_eps']
    H   = cfg['num_attention_heads']; W = cfg['sliding_window']
    types = cfg['layer_types']; ropep = cfg['rope_parameters']
    ids = np.asarray(tokens, dtype=np.int64); S = len(ids)
    pos = np.arange(S, dtype=np.float32)
    print(f'[ref] {S} tokens, {L} capas', file=sys.stderr)

    h = st.rows(P + 'embed_tokens.weight', ids) * math.sqrt(D)

    # --- PLE: identidad del token + contexto proyectado, combinados a 1/sqrt(2)
    ident = st.rows(P + 'embed_tokens_per_layer.weight', ids) * math.sqrt(PLE)
    ident = ident.reshape(S, L, PLE)
    ctx = (h @ st.f32(P + 'per_layer_model_projection.weight').T) / math.sqrt(D)
    ctx = rmsnorm(ctx.reshape(S, L, PLE), st.f32(P + 'per_layer_projection_norm.weight'), eps)
    ple = (ctx + ident) * (1.0 / math.sqrt(2.0))

    shared = {}                       # KV de la ultima capa fisica de cada tipo
    first_shared = L - cfg['num_kv_shared_layers']
    for i in range(L):
        w = lambda s: st.f32(f'{P}layers.{i}.{s}')
        t = types[i]; glob = (t == 'full_attention')
        hd = cfg['global_head_dim'] if glob else cfg['head_dim']
        rp = ropep[t]
        res = h
        x = rmsnorm(h, w('input_layernorm.weight'), eps)

        q = (x @ w('self_attn.q_proj.weight').T).reshape(S, H, hd)
        q = rmsnorm(q, w('self_attn.q_norm.weight'), eps)
        q = rope(q, pos, inv_freq(hd, rp['rope_theta'], rp['rope_type'], rp.get('partial_rotary_factor', 1.0)))

        if i < first_shared:
            k = (x @ w('self_attn.k_proj.weight').T).reshape(S, 1, hd)
            k = rmsnorm(k, w('self_attn.k_norm.weight'), eps)
            k = rope(k, pos, inv_freq(hd, rp['rope_theta'], rp['rope_type'], rp.get('partial_rotary_factor', 1.0)))
            v = (x @ w('self_attn.v_proj.weight').T).reshape(S, 1, hd)
            v = rmsnorm(v, None, eps)          # v_norm lleva with_scale=False
            shared[t] = (k, v)                 # la ultima de cada tipo gana
        else:
            k, v = shared[t]                   # capas 15-34 releen 13 (local) / 14 (global)

        sc = q.transpose(1, 0, 2) @ k[:, 0, :].T * 1.0          # scaling = 1.0
        mask = np.tril(np.ones((S, S), dtype=bool))
        if not glob:
            qi = np.arange(S)[:, None]; ki = np.arange(S)[None, :]
            mask &= (qi - ki) < W
        sc = np.where(mask[None], sc, -np.inf)
        sc -= sc.max(-1, keepdims=True)
        p = np.exp(sc); p /= p.sum(-1, keepdims=True)
        a = (p @ v[:, 0, :]).transpose(1, 0, 2).reshape(S, H * hd)
        a = a @ w('self_attn.o_proj.weight').T

        h = res + rmsnorm(a, w('post_attention_layernorm.weight'), eps)
        res = h
        x = rmsnorm(h, w('pre_feedforward_layernorm.weight'), eps)
        x = gelu_tanh(x @ w('mlp.gate_proj.weight').T) * (x @ w('mlp.up_proj.weight').T)
        x = x @ w('mlp.down_proj.weight').T
        h = res + rmsnorm(x, w('post_feedforward_layernorm.weight'), eps)

        res = h                                 # inyeccion del PLE de esta capa
        g = gelu_tanh(h @ w('per_layer_input_gate.weight').T) * ple[:, i, :]
        g = g @ w('per_layer_projection.weight').T
        h = res + rmsnorm(g, w('post_per_layer_input_norm.weight'), eps)
        h = h * w('layer_scalar')
        print(f'\r[ref] capa {i+1}/{L}', end='', file=sys.stderr)

    print(file=sys.stderr)
    h = rmsnorm(h, st.f32(P + 'norm.weight'), eps)
    emb = st.f32(P + 'embed_tokens.weight')     # tie_word_embeddings
    lg = h[-1] @ emb.T
    cap = cfg['final_logit_softcapping']
    lg = cap * np.tanh(lg / cap)
    lg.astype(np.float32).tofile(out_path)
    top = np.argsort(-lg)[:5]
    print(f'[ref] logits -> {out_path}  min {lg.min():.3f} max {lg.max():.3f} '
          f'media {lg.mean():.3f}\n[ref] top-5: ' +
          ' '.join(f'{t}({lg[t]:.2f})' for t in top), file=sys.stderr)

if __name__ == '__main__':
    toks = [int(x) for x in sys.argv[2].split(',')]
    main(sys.argv[1], toks, sys.argv[3])
