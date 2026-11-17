#!/usr/bin/env python3
"""Referencia en numpy de la familia Qwen3.5 (Gated DeltaNet + atención con puerta).

Sigue `modeling_qwen3_5.py` de transformers operación por operación, en FP32, token a token
y con el estado que llevaría una sesión: el recurrente y el de la convolución en las capas
DeltaNet, y la caché de claves y valores en las de atención. No necesita PyTorch.

Sirve para dos cosas: comprobar que el modelo se entiende (que continúa un texto con sentido)
y dejar en disco la salida de cada capa, que es contra lo que se compara el motor.

    qwen35_ref.py <carpeta del modelo> --ids 760 6511 ...        # ids sueltos
    qwen35_ref.py <carpeta del modelo> --text "The capital of France is"
    qwen35_ref.py ... --generate 8 --dump salida/                 # capas a .npy

`--text` parte por palabras del vocabulario (sirve para
frases sencillas en inglés); para algo más fino, pasar los ids.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def _fp32(crudo, dtype):
    if dtype == 'BF16':
        # BF16 son los 16 bits altos de un FP32.
        return (crudo.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dtype == 'F32':
        return crudo.view(np.float32)
    if dtype == 'F16':
        return crudo.view(np.float16).astype(np.float32)
    raise ValueError('dtype sin soporte: ' + dtype)


class Perezoso:
    """Un tensor [N, ...] del que solo se convierte a FP32 la rebanada que se pide.

    Los expertos de un MoE grande no caben convertidos (en el 35B son más de 100 GB en
    FP32) y por token solo se usan 8 de 256: se lee del fichero mapeado el que toca.
    """
    def __init__(self, crudo, info):
        self.crudo, self.dtype, self.shape = crudo, info['dtype'], tuple(info['shape'])
        self.paso = len(crudo) // self.shape[0]

    def __getitem__(self, i):
        i = int(i)
        trozo = self.crudo[i * self.paso:(i + 1) * self.paso]
        if self.dtype == 'BF16':
            # Una sola pasada: los 16 bits van a la mitad alta de cada FP32 ya a cero.
            salida = np.zeros((len(trozo) // 2, 2), dtype=np.uint16)
            salida[:, 1] = trozo.view(np.uint16)
            return salida.view(np.float32).reshape(self.shape[1:])
        return np.array(_fp32(trozo, self.dtype), dtype=np.float32).reshape(self.shape[1:])


def cargar_safetensors(rutas):
    """Devuelve nombre -> array FP32 y nombre -> Perezoso, leyendo de los ficheros mapeados."""
    donde = {}
    for ruta in rutas:
        with open(ruta, 'rb') as f:
            largo = struct.unpack('<Q', f.read(8))[0]
            cabecera = json.loads(f.read(largo))
        datos = np.memmap(ruta, dtype=np.uint8, mode='r', offset=8 + largo)
        cabecera.pop('__metadata__', None)
        for nombre, info in cabecera.items():
            donde[nombre] = (datos, info)

    def crudo(nombre):
        datos, info = donde[nombre]
        a, b = info['data_offsets']
        return datos[a:b], info

    def tensor(nombre):
        trozo, info = crudo(nombre)
        return np.array(_fp32(trozo, info['dtype']), dtype=np.float32).reshape(info['shape'])

    def perezoso(nombre):
        return Perezoso(*crudo(nombre))
    return tensor, perezoso, donde


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.logaddexp(0.0, x)


def rms_centrada(x, peso, eps):
    """La norma de la familia: el peso guardado está centrado en cero, se usa 1 + peso."""
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * (1.0 + peso)


class Modelo:
    def __init__(self, carpeta):
        carpeta = Path(carpeta)
        self.config = json.loads((carpeta / 'config.json').read_text())
        t = self.t = self.config.get('text_config', self.config)
        self.expertos = t.get('num_experts', 0)
        self.elegidos = t.get('num_experts_per_tok', 0)
        pesos = sorted(carpeta.glob('*.safetensors'))
        if not pesos:
            raise SystemExit('No hay ficheros .safetensors en ' + str(carpeta))
        self.w, self.perezoso, self.nombres = cargar_safetensors(pesos)
        self.raiz = 'model.language_model.' if any(n.startswith('model.language_model.') for n in self.nombres) else 'model.'
        self.eps = t['rms_norm_eps']
        self.H = t['hidden_size']
        self.tipos = t['layer_types']
        # Atención
        self.nh, self.nkv, self.hd = t['num_attention_heads'], t['num_key_value_heads'], t['head_dim']
        rot = int(self.hd * t['rope_parameters'].get('partial_rotary_factor', 1.0))
        self.rot = rot
        self.inv_freq = 1.0 / (t['rope_parameters']['rope_theta'] ** (np.arange(0, rot, 2, dtype=np.float64) / rot))
        # DeltaNet
        self.nk, self.nv = t['linear_num_key_heads'], t['linear_num_value_heads']
        self.dk, self.dv = t['linear_key_head_dim'], t['linear_value_head_dim']
        self.key_dim, self.value_dim = self.nk * self.dk, self.nv * self.dv
        self.kernel = t['linear_conv_kernel_dim']
        self.emb = self.w(self.raiz + 'embed_tokens.weight')
        self.cabeza = self.emb if t.get('tie_word_embeddings', False) or 'lm_head.weight' not in self.nombres \
            else self.w('lm_head.weight')
        from concurrent.futures import ThreadPoolExecutor
        self.hilos = ThreadPoolExecutor(max(1, self.elegidos))
        self.capas = [self._capa(i) for i in range(t['num_hidden_layers'])]
        self.norma_final = self.w(self.raiz + 'norm.weight')
        self.reiniciar()

    def _capa(self, i):
        p = '%slayers.%d.' % (self.raiz, i)
        w = self.w
        capa = dict(tipo=self.tipos[i], ln1=w(p + 'input_layernorm.weight'), ln2=w(p + 'post_attention_layernorm.weight'))
        if self.expertos:
            # El MoE es la MLP de la capa: enrutador, expertos empaquetados y un experto
            # compartido, que es una MLP densa con su puerta.
            m = p + 'mlp.'
            capa.update(enrutador=w(m + 'gate.weight'), exp_gate_up=self.perezoso(m + 'experts.gate_up_proj'),
                        exp_down=self.perezoso(m + 'experts.down_proj'), puerta_comun=w(m + 'shared_expert_gate.weight')[0],
                        gate=w(m + 'shared_expert.gate_proj.weight'), up=w(m + 'shared_expert.up_proj.weight'),
                        down=w(m + 'shared_expert.down_proj.weight'))
        else:
            capa.update(gate=w(p + 'mlp.gate_proj.weight'), up=w(p + 'mlp.up_proj.weight'), down=w(p + 'mlp.down_proj.weight'))
        if capa['tipo'] == 'linear_attention':
            a = p + 'linear_attn.'
            capa.update(qkv=w(a + 'in_proj_qkv.weight'), z=w(a + 'in_proj_z.weight'), a=w(a + 'in_proj_a.weight'),
                        b=w(a + 'in_proj_b.weight'), conv=w(a + 'conv1d.weight')[:, 0, :], A_log=w(a + 'A_log'),
                        dt_bias=w(a + 'dt_bias'), norm=w(a + 'norm.weight'), out=w(a + 'out_proj.weight'))
        else:
            a = p + 'self_attn.'
            capa.update(q=w(a + 'q_proj.weight'), k=w(a + 'k_proj.weight'), v=w(a + 'v_proj.weight'),
                        o=w(a + 'o_proj.weight'), q_norm=w(a + 'q_norm.weight'), k_norm=w(a + 'k_norm.weight'))
        return capa

    def reiniciar(self):
        """Estado de una sesión vacía."""
        self.pos = 0
        self.estado = []
        for capa in self.capas:
            if capa['tipo'] == 'linear_attention':
                self.estado.append(dict(S=np.zeros((self.nv, self.dk, self.dv), np.float32),
                                        conv=np.zeros((self.key_dim * 2 + self.value_dim, self.kernel - 1), np.float32)))
            else:
                self.estado.append(dict(K=[], V=[]))

    # --- capas -------------------------------------------------------------------------

    def _deltanet(self, capa, estado, x):
        mezcla = capa['qkv'] @ x                                    # [2·key + value]
        z = (capa['z'] @ x).reshape(self.nv, self.dv)
        b = capa['b'] @ x
        a = capa['a'] @ x
        # Convolución causal canal a canal: las kernel-1 muestras anteriores y la actual.
        ventana = np.concatenate([estado['conv'], mezcla[:, None]], axis=1)
        estado['conv'] = ventana[:, 1:].copy()
        mezcla = silu(np.sum(ventana * capa['conv'], axis=1))
        q = mezcla[:self.key_dim].reshape(self.nk, self.dk)
        k = mezcla[self.key_dim:2 * self.key_dim].reshape(self.nk, self.dk)
        v = mezcla[2 * self.key_dim:].reshape(self.nv, self.dv)
        # Cada cabeza de clave sirve a nv/nk cabezas de valor consecutivas.
        if self.nk != self.nv:
            q = np.repeat(q, self.nv // self.nk, axis=0)
            k = np.repeat(k, self.nv // self.nk, axis=0)
        q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        q = q / np.sqrt(self.dk)
        beta = sigmoide(b)
        g = -np.exp(capa['A_log']) * softplus(a + capa['dt_bias'])
        S = estado['S'] * np.exp(g)[:, None, None]
        recordado = np.einsum('hkv,hk->hv', S, k)
        delta = (v - recordado) * beta[:, None]
        S = S + k[:, :, None] * delta[:, None, :]
        estado['S'] = S.astype(np.float32)
        salida = np.einsum('hkv,hk->hv', S, q)
        # Norma por cabeza (peso tal cual, no centrado) y puerta.
        salida = salida / np.sqrt(np.mean(salida * salida, axis=-1, keepdims=True) + self.eps) * capa['norm'] * silu(z)
        return capa['out'] @ salida.reshape(-1)

    def _rotar(self, x):
        # Tres ejes (tiempo, fila, columna), una frecuencia para cada uno por turnos. En texto
        # los tres valen lo mismo y queda la rotación de siempre; en una imagen, no.
        t, fila, columna = self.pos3
        eje = np.arange(len(self.inv_freq)) % 3
        angulos = np.where(eje == 0, t, np.where(eje == 1, fila, columna)) * self.inv_freq
        cos, sin = np.cos(angulos), np.sin(angulos)
        cos = np.concatenate([cos, cos]).astype(np.float32)
        sin = np.concatenate([sin, sin]).astype(np.float32)
        giro, resto = x[:, :self.rot], x[:, self.rot:]
        mitad = self.rot // 2
        girado = np.concatenate([-giro[:, mitad:], giro[:, :mitad]], axis=1)
        return np.concatenate([giro * cos + girado * sin, resto], axis=1)

    def _atencion(self, capa, estado, x):
        doble = (capa['q'] @ x).reshape(self.nh, 2 * self.hd)
        q, puerta = doble[:, :self.hd], doble[:, self.hd:]
        q = self._rotar(rms_centrada(q, capa['q_norm'], self.eps))
        k = self._rotar(rms_centrada((capa['k'] @ x).reshape(self.nkv, self.hd), capa['k_norm'], self.eps))
        v = (capa['v'] @ x).reshape(self.nkv, self.hd)
        estado['K'].append(k)
        estado['V'].append(v)
        K = np.stack(estado['K'], axis=1)                           # [nkv, T, hd]
        V = np.stack(estado['V'], axis=1)
        grupos = self.nh // self.nkv
        salida = np.empty((self.nh, self.hd), np.float32)
        for h in range(self.nh):
            kv = h // grupos
            puntos = (K[kv] @ q[h]) * (self.hd ** -0.5)
            puntos = np.exp(puntos - puntos.max())
            salida[h] = (puntos / puntos.sum()) @ V[kv]
        return capa['o'] @ (salida * sigmoide(puerta)).reshape(-1)

    def _mlp(self, capa, x, volcar=None, i=0):
        densa = capa['down'] @ (silu(capa['gate'] @ x) * (capa['up'] @ x))
        if not self.expertos:
            return densa
        # Softmax sobre todos, los K mayores y renormalizar entre ellos.
        logits = (capa['enrutador'] @ x).astype(np.float32)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        elegidos = np.argsort(-probs, kind='stable')[:self.elegidos]
        pesos = probs[elegidos] / probs[elegidos].sum()
        if volcar is not None:
            volcar['ruta%02d' % i].append(np.sort(elegidos).astype(np.float32))
        intermedio = capa['exp_down'].shape[2]
        salida = np.zeros_like(x)
        def experto(e):
            gate_up = capa['exp_gate_up'][e] @ x
            return capa['exp_down'][e] @ (silu(gate_up[:intermedio]) * gate_up[intermedio:])
        # En hilos: convertir cada experto desde el fichero es lo que más tarda en un MoE grande.
        for peso, aporte in zip(pesos, self.hilos.map(experto, elegidos)):
            salida += peso * aporte
        return salida + sigmoide(capa['puerta_comun'] @ x) * densa

    # --- un token ----------------------------------------------------------------------

    def imagen(self, vectores, filas, columnas, volcar=None):
        """Mete una imagen ya pasada por la torre visual: un vector por bloque, fila a fila.

        Todos comparten el mismo «tiempo» y cada uno lleva su fila y su columna, contadas
        desde la posición donde empieza la imagen. Después el texto sigue a partir de esa
        posición más el lado mayor de la rejilla: la imagen ocupa filas×columnas puestos en
        la caché, pero solo avanza la posición lo que mide su lado.
        """
        inicio, logits = self.pos, None
        for f in range(filas):
            for c in range(columnas):
                logits = self.paso(None, volcar, vector=vectores[f * columnas + c],
                                   pos3=(inicio, inicio + f, inicio + c))
        self.pos = inicio + max(filas, columnas)
        return logits

    def paso(self, token, volcar=None, vector=None, pos3=None):
        """Avanza la sesión un token. Devuelve los logits del siguiente."""
        h = (self.emb[token] if vector is None else vector).astype(np.float32).copy()
        self.pos3 = pos3 if pos3 is not None else (self.pos, self.pos, self.pos)
        if volcar is not None:
            volcar['emb'].append(h.copy())
        for i, (capa, estado) in enumerate(zip(self.capas, self.estado)):
            x = rms_centrada(h, capa['ln1'], self.eps)
            mezcla = self._deltanet(capa, estado, x) if capa['tipo'] == 'linear_attention' else self._atencion(capa, estado, x)
            h = h + mezcla
            if volcar is not None:
                volcar['attn%02d' % i].append(mezcla.copy())
            x = rms_centrada(h, capa['ln2'], self.eps)
            h = h + self._mlp(capa, x, volcar, i)
            if volcar is not None:
                volcar['capa%02d' % i].append(h.copy())
        if pos3 is None:
            self.pos += 1
        final = rms_centrada(h, self.norma_final, self.eps)
        logits = self.cabeza @ final
        if volcar is not None:
            volcar['final'].append(final.copy())
            volcar['logits'].append(logits.copy())
        return logits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('modelo')
    ap.add_argument('--ids', type=int, nargs='+')
    ap.add_argument('--text')
    ap.add_argument('--generate', type=int, default=0, help='tokens a continuar, eligiendo siempre el más probable')
    ap.add_argument('--dump', help='carpeta donde dejar la salida de cada capa')
    ap.add_argument('--image', help='imagen que sustituye a los tokens <|image_pad|> de la entrada')
    ap.add_argument('--max-pixels', type=int, default=1024 * 1024)
    args = ap.parse_args()

    # Un modelo sintético no trae vocabulario: se le habla con --ids.
    fichero_vocab = Path(args.modelo) / 'vocab.json'
    vocab = json.loads(fichero_vocab.read_text()) if fichero_vocab.exists() else {}
    por_id = {v: k for k, v in vocab.items()}
    if args.ids:
        ids = list(args.ids)
    elif args.text:
        palabras = args.text.split(' ')
        ids = []
        for n, palabra in enumerate(palabras):
            clave = palabra if n == 0 else 'Ġ' + palabra
            if clave not in vocab:
                raise SystemExit('«%s» no es una sola pieza del vocabulario: pasa los ids con --ids' % palabra)
            ids.append(vocab[clave])
    else:
        raise SystemExit('Hace falta --ids o --text')

    modelo = Modelo(args.modelo)
    import collections
    volcar = collections.defaultdict(list) if args.dump else None
    texto = lambda i: por_id.get(i, '<%d>' % i).replace('Ġ', ' ').replace('Ċ', '\\n')
    print('entrada:', ids, repr(''.join(texto(i) for i in ids)))
    logits = None
    vectores = rejilla = None
    if args.image:
        # Un solo <|image_pad|> en la entrada marca dónde va la imagen entera.
        from PIL import Image
        from qwen35_vision_ref import Torre
        marcador = modelo.config['image_token_id']
        if ids.count(marcador) != 1:
            raise SystemExit('Con --image la entrada lleva un único token %d (<|image_pad|>)' % marcador)
        vectores, rejilla = Torre(args.modelo).imagen(Image.open(args.image), args.max_pixels)
        print('imagen: rejilla %d×%d, %d tokens' % (rejilla[0], rejilla[1], len(vectores)))
    import time
    inicio = time.time()
    for n, token in enumerate(ids):
        if n and n % 32 == 0:
            print('  %d de %d tokens, %.1f s por token' % (n, len(ids), (time.time() - inicio) / n), file=sys.stderr, flush=True)
        if vectores is not None and token == modelo.config['image_token_id']:
            logits = modelo.imagen(vectores, rejilla[0], rejilla[1], volcar)
        else:
            logits = modelo.paso(token, volcar)
    generados = []
    for _ in range(args.generate):
        siguiente = int(np.argmax(logits))
        generados.append(siguiente)
        logits = modelo.paso(siguiente, volcar)
    if generados:
        print('continúa:', generados, repr(''.join(texto(i) for i in generados)))
    mejores = np.argsort(logits)[::-1][:5]
    print('siguientes más probables:', [(int(i), texto(int(i)), round(float(logits[i]), 3)) for i in mejores])
    if args.dump:
        salida = Path(args.dump)
        salida.mkdir(parents=True, exist_ok=True)
        np.save(salida / 'ids.npy', np.array(ids + generados, dtype=np.int32))
        if vectores is not None:
            np.save(salida / 'imagen.npy', vectores)
            np.save(salida / 'rejilla.npy', np.array(rejilla, dtype=np.int32))
        for nombre, filas in volcar.items():
            np.save(salida / (nombre + '.npy'), np.stack(filas).astype(np.float32))
        print('volcado en', salida, '(%d tensores, %d tokens)' % (len(volcar), len(ids) + len(generados)))


if __name__ == '__main__':
    sys.exit(main())
