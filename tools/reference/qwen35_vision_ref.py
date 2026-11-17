#!/usr/bin/env python3
"""La torre visual de la familia Qwen3.5, en numpy y FP32.

    qwen35_vision_ref.py <carpeta del checkpoint> <imagen> [--max-pixels N] [--dump carpeta]

Hace lo mismo que `Qwen3_5VisionModel` de transformers, leído de su código (no hay PyTorch
en estos equipos para compararlo): sirve de vara de medir para el motor.

1. Preprocesado, como el procesador de imágenes de Qwen2-VL: la imagen se redimensiona a
   múltiplos de 32 (parche 16 × fusión 2) dentro de un tope de píxeles, se normaliza a
   (x/255 − 0,5)/0,5 y se repite dos veces en el tiempo (un «vídeo» de dos fotogramas iguales).
2. Parches de 3×2×16×16 = 1.536 valores, en orden de bloques de fusión: los cuatro parches que
   luego se funden en uno van seguidos. Una proyección lineal (la convolución 3D del original).
3. Se suma la posición aprendida, una rejilla de 48×48 interpolada (bilineal, con las esquinas
   alineadas) al tamaño de la imagen.
4. Bloques con norma previa: atención completa entre todos los parches, con rotación en dos
   ejes (fila y columna) sobre toda la dimensión de la cabeza, y MLP con GELU aproximada.
5. Fusión: norma, los cuatro parches de cada bloque concatenados, y dos lineales con GELU
   exacta en medio. Sale un vector del ancho del modelo de texto por cada bloque de 2×2.

Devuelve esos vectores y la rejilla (filas, columnas) que forman, que es lo que el modelo de
texto necesita para darles posición.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen35_ref import cargar_safetensors  # noqa: E402


def tamano(alto, ancho, factor, min_pixels, max_pixels):
    """smart_resize de Qwen2-VL: múltiplos de `factor`, área entre los dos topes."""
    if max(alto, ancho) / min(alto, ancho) > 200:
        raise ValueError('imagen demasiado alargada')
    h = max(factor, round(alto / factor) * factor)
    w = max(factor, round(ancho / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt(alto * ancho / max_pixels)
        h = max(factor, math.floor(alto / beta / factor) * factor)
        w = max(factor, math.floor(ancho / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (alto * ancho))
        h = math.ceil(alto * beta / factor) * factor
        w = math.ceil(ancho * beta / factor) * factor
    return h, w


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu(x):
    # erf sin scipy: la de Abramowitz y Stegun no basta; se usa la de math, elemento a elemento.
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def norma_capa(x, peso, sesgo, eps=1e-6):
    media = x.mean(axis=-1, keepdims=True)
    var = ((x - media) ** 2).mean(axis=-1, keepdims=True)
    return (x - media) / np.sqrt(var + eps) * peso + sesgo


class Torre:
    def __init__(self, carpeta):
        carpeta = Path(carpeta)
        config = json.loads((carpeta / 'config.json').read_text())
        self.c = c = config['vision_config']
        pesos = sorted(carpeta.glob('*.safetensors'))
        if len(pesos) != 1:
            raise SystemExit('Se espera un único fichero .safetensors en ' + str(carpeta))
        w, _ = cargar_safetensors(pesos[0])
        self.w = lambda nombre: w('model.visual.' + nombre)
        self.parche, self.fusion, self.tiempo = c['patch_size'], c['spatial_merge_size'], c['temporal_patch_size']
        self.ancho, self.cabezas = c['hidden_size'], c['num_heads']
        self.lado = int(c['num_position_embeddings'] ** 0.5)
        self.proyeccion = self.w('patch_embed.proj.weight').reshape(self.ancho, -1)
        self.posiciones = self.w('pos_embed.weight')

    def preprocesar(self, imagen, min_pixels=65536, max_pixels=1024 * 1024):
        """Imagen PIL -> parches [N, 3·2·16·16] en orden de bloques de fusión, y la rejilla."""
        from PIL import Image
        imagen = imagen.convert('RGB')
        factor = self.parche * self.fusion
        alto, ancho = tamano(imagen.height, imagen.width, factor, min_pixels, max_pixels)
        imagen = imagen.resize((ancho, alto), Image.BICUBIC)
        x = (np.asarray(imagen, dtype=np.float32) / 255.0 - 0.5) / 0.5            # [alto, ancho, 3]
        x = x.transpose(2, 0, 1)                                                  # [3, alto, ancho]
        x = np.stack([x] * self.tiempo, axis=1)                                   # [3, T, alto, ancho]
        gh, gw, m, p = alto // self.parche, ancho // self.parche, self.fusion, self.parche
        # [C, T, gh/m, m, p, gw/m, m, p] -> [gh/m, gw/m, m, m, C, T, p, p]
        x = x.reshape(3, self.tiempo, gh // m, m, p, gw // m, m, p).transpose(2, 5, 3, 6, 0, 1, 4, 7)
        return x.reshape(gh * gw, 3 * self.tiempo * p * p).astype(np.float32), (gh, gw)

    def _filas_columnas(self, gh, gw):
        """Fila y columna de cada parche en el orden de bloques de fusión."""
        m = self.fusion
        filas, columnas = np.meshgrid(np.arange(gh), np.arange(gw), indexing='ij')
        orden = lambda a: a.reshape(gh // m, m, gw // m, m).transpose(0, 2, 1, 3).reshape(-1)
        return orden(filas), orden(columnas)

    def _posicion_aprendida(self, filas, columnas, gh, gw):
        def eje(indice, tamano_eje):
            origen = indice * (self.lado - 1) / max(tamano_eje - 1, 1)
            suelo = np.floor(origen)
            tomas = np.stack([suelo, suelo + 1], axis=1)
            pesos = np.clip(1.0 - np.abs(origen[:, None] - tomas), 0.0, None)
            return np.clip(tomas, 0, self.lado - 1).astype(np.int64), pesos
        tf, pf = eje(filas.astype(np.float32), gh)
        tc, pc = eje(columnas.astype(np.float32), gw)
        suma = np.zeros((len(filas), self.ancho), np.float32)
        for a in range(2):
            for b in range(2):
                suma += self.posiciones[tf[:, a] * self.lado + tc[:, b]] * (pf[:, a] * pc[:, b])[:, None]
        return suma

    def _rotacion(self, filas, columnas):
        dim = self.ancho // self.cabezas
        espacial = dim // 2
        inv = 1.0 / (10000.0 ** (np.arange(0, espacial, 2, dtype=np.float32) / espacial))
        angulos = np.concatenate([filas[:, None] * inv, columnas[:, None] * inv], axis=1)   # [N, dim/2]
        angulos = np.concatenate([angulos, angulos], axis=1)                                 # [N, dim]
        return np.cos(angulos).astype(np.float32), np.sin(angulos).astype(np.float32)

    def adelante(self, parches, rejilla, volcar=None):
        gh, gw = rejilla
        filas, columnas = self._filas_columnas(gh, gw)
        h = parches @ self.proyeccion.T + self.w('patch_embed.proj.bias')
        h = h + self._posicion_aprendida(filas, columnas, gh, gw)
        if volcar is not None:
            volcar['entrada'] = h.copy()
        cos, sin = self._rotacion(filas, columnas)
        n, cabezas, dim = len(h), self.cabezas, self.ancho // self.cabezas
        mitad = dim // 2
        girar = lambda t: t * cos[:, None, :] + np.concatenate([-t[..., mitad:], t[..., :mitad]], axis=-1) * sin[:, None, :]
        for i in range(self.c['depth']):
            b = 'blocks.%d.' % i
            x = norma_capa(h, self.w(b + 'norm1.weight'), self.w(b + 'norm1.bias'))
            qkv = (x @ self.w(b + 'attn.qkv.weight').T + self.w(b + 'attn.qkv.bias')).reshape(n, 3, cabezas, dim)
            q, k, v = girar(qkv[:, 0]), girar(qkv[:, 1]), qkv[:, 2]
            puntos = np.einsum('nhd,mhd->hnm', q, k) * dim ** -0.5
            puntos = np.exp(puntos - puntos.max(axis=-1, keepdims=True))
            puntos /= puntos.sum(axis=-1, keepdims=True)
            atendido = np.einsum('hnm,mhd->nhd', puntos, v).reshape(n, self.ancho)
            h = h + atendido @ self.w(b + 'attn.proj.weight').T + self.w(b + 'attn.proj.bias')
            x = norma_capa(h, self.w(b + 'norm2.weight'), self.w(b + 'norm2.bias'))
            x = gelu_tanh(x @ self.w(b + 'mlp.linear_fc1.weight').T + self.w(b + 'mlp.linear_fc1.bias'))
            h = h + x @ self.w(b + 'mlp.linear_fc2.weight').T + self.w(b + 'mlp.linear_fc2.bias')
            if volcar is not None:
                volcar['bloque%02d' % i] = h.copy()
        x = norma_capa(h, self.w('merger.norm.weight'), self.w('merger.norm.bias'))
        x = x.reshape(n // self.fusion ** 2, -1)
        x = gelu(x @ self.w('merger.linear_fc1.weight').T + self.w('merger.linear_fc1.bias'))
        salida = (x @ self.w('merger.linear_fc2.weight').T + self.w('merger.linear_fc2.bias')).astype(np.float32)
        if volcar is not None:
            volcar['salida'] = salida.copy()
        return salida, (gh // self.fusion, gw // self.fusion)

    def imagen(self, imagen, max_pixels=1024 * 1024, volcar=None):
        parches, rejilla = self.preprocesar(imagen, max_pixels=max_pixels)
        if volcar is not None:
            volcar['parches'] = parches.copy()
        return self.adelante(parches, rejilla, volcar)


def main():
    from PIL import Image
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('modelo')
    ap.add_argument('imagen')
    ap.add_argument('--max-pixels', type=int, default=1024 * 1024)
    ap.add_argument('--dump')
    args = ap.parse_args()
    volcar = {} if args.dump else None
    salida, rejilla = Torre(args.modelo).imagen(Image.open(args.imagen), args.max_pixels, volcar)
    print('rejilla %d×%d: %d tokens de ancho %d; norma media %.3f' % (
        rejilla[0], rejilla[1], len(salida), salida.shape[1], np.linalg.norm(salida, axis=1).mean()))
    if args.dump:
        carpeta = Path(args.dump)
        carpeta.mkdir(parents=True, exist_ok=True)
        for nombre, valores in volcar.items():
            np.save(carpeta / (nombre + '.npy'), valores.astype(np.float32))
        np.save(carpeta / 'rejilla.npy', np.array(rejilla, dtype=np.int32))
        print('volcado en', carpeta)


if __name__ == '__main__':
    main()
