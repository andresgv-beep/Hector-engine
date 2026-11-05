"""Lector HNFv10 para las herramientas Python del motor (helios_convert_v10/docs/HNFV10.md).

Comprueba lo mismo que src/hnf10.cpp en lo barato: CRC de las dos cabeceras, xxh3-128 de los
metadatos y estado completo. No lee pesos: `tensor()` los abre con memmap y `decode()` los
pasa a FP32 (fp16, bf16, fp32, hq44k_g16, hq52k, hq8k).

uso como programa: hnf10.py modelo.hnf  (lista componentes y tensores)
"""
import ctypes
import ctypes.util
from dataclasses import dataclass
from pathlib import Path
import struct
import sys
import zlib
import numpy as np

MAGIC, HEADER, PAYLOAD_START = b'HNFv10\0\0', 512, 4096
TEXT, VISION, AUDIO, VIDEO, TOKENIZER, CONFIG, TEMPLATE = 1, 2, 3, 4, 16, 17, 18
BLOCK = {'hq44k_g16': 152, 'hq52k': 184, 'hq62k': 264, 'hq8k': 272}


class Hash128(ctypes.Structure):
    _fields_ = [('low64', ctypes.c_uint64), ('high64', ctypes.c_uint64)]


_library = ctypes.util.find_library('xxhash')
if _library is None:
    raise ImportError('hace falta libxxhash para comprobar los metadatos')
_xxhash = ctypes.CDLL(_library)
_xxhash.XXH3_128bits.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_xxhash.XXH3_128bits.restype = Hash128
_xxhash.XXH3_64bits.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_xxhash.XXH3_64bits.restype = ctypes.c_uint64


def _pointer(data):
    data = np.ascontiguousarray(data)
    return data, data.ctypes.data, data.nbytes


def xxh3_128(data):
    """Mismo orden de bytes que el conversor: low64 y luego high64, en little endian."""
    data, pointer, size = _pointer(np.frombuffer(data, np.uint8) if isinstance(data, bytes) else data)
    h = _xxhash.XXH3_128bits(pointer, size)
    return struct.pack('<QQ', h.low64, h.high64)


def xxh3_64(data):
    data, pointer, size = _pointer(data)
    return _xxhash.XXH3_64bits(pointer, size)


@dataclass
class Component:
    id: int
    kind: int
    required: bool
    name: str
    offset: int
    size: int
    first_tensor: int
    tensor_count: int


@dataclass
class Tensor:
    name: str
    component: int
    layer: int
    dtype: str
    shape: tuple
    logical_last: int   # columnas lógicas de la última dimensión (relleno GM4X)
    offset: int
    size: int
    experts: int
    hash: bytes
    source: str         # nombre en el checkpoint
    source_hash: int    # xxh3-64 de sus bytes en el checkpoint


class _In:
    def __init__(self, body, strings):
        self.body, self.strings, self.at = body, strings, 0

    def take(self, n):
        if self.at + n > len(self.body):
            raise ValueError('sección truncada')
        part = self.body[self.at:self.at+n]
        self.at += n
        return part

    def u8(self): return self.take(1)[0]
    def u32(self): return struct.unpack('<I', self.take(4))[0]
    def u64(self): return struct.unpack('<Q', self.take(8))[0]
    def f64(self): return struct.unpack('<d', self.take(8))[0]

    def s(self):
        offset, length = self.u32(), self.u32()
        if offset + length > len(self.strings):
            raise ValueError('cadena fuera de STRINGS')
        return self.strings[offset:offset+length].decode()

    def value(self):
        kind = self.u8()
        if kind == 1: return bool(self.u8())
        if kind == 2: return struct.unpack('<q', self.take(8))[0]
        if kind == 3: return self.f64()
        if kind == 4: return self.s()
        if kind == 5: return bytes(self.take(self.u32()))
        raise ValueError(f'valor de tipo {kind}')


def _header(raw, file_size):
    if raw[:8] != MAGIC or struct.unpack_from('<H', raw, 8)[0] != 10 or struct.unpack_from('<I', raw, 12)[0] != HEADER:
        return None
    if struct.unpack_from('<I', raw, 508)[0] != zlib.crc32(raw[:508]):
        return None
    sequence, state, _, offset, size = struct.unpack_from('<QIIQQ', raw, 16)
    declared = struct.unpack_from('<Q', raw, 64)[0]
    if declared > file_size or offset < PAYLOAD_START or offset + size > declared:
        return None
    return {'sequence': sequence, 'state': state, 'offset': offset, 'size': size,
            'hash': raw[48:64], 'file_size': declared, 'identity': raw[72:88]}


class Hnf10:
    def __init__(self, path):
        self.path = Path(path)
        file_size = self.path.stat().st_size
        best = None
        with self.path.open('rb') as f:
            slots = f.read(2*HEADER)
            for i in range(2):
                h = _header(slots[i*HEADER:(i+1)*HEADER], file_size)
                if h is None:
                    continue
                f.seek(h['offset'])
                meta = f.read(h['size'])
                if len(meta) == h['size'] and xxh3_128(meta) == h['hash'] and (best is None or h['sequence'] > best[0]['sequence']):
                    best = (h, meta)
        if best is None:
            raise ValueError(f'{self.path}: ninguna cabecera válida, no es un HNFv10 o está dañado')
        header, meta = best
        if header['state'] != 1:
            raise ValueError(f'{self.path}: la última escritura de pesos quedó en curso (hnf10 repair)')
        self.identity, self.sequence = header['identity'].hex(), header['sequence']
        self._parse(meta)

    def _parse(self, meta):
        if meta[:4] != b'HNFM' or struct.unpack_from('<I', meta, 4)[0] != 1:
            raise ValueError('metadatos HNFM v1 esperados')
        sections = {}
        for i in range(struct.unpack_from('<I', meta, 8)[0]):
            kind, _, offset, size = struct.unpack_from('<IIQQ', meta, 16+24*i)
            sections[kind & 0x7fffffff] = meta[offset:offset+size]
        strings = sections[1]
        r = _In(sections[2], strings)
        self.components = []
        for _ in range(r.u32()):
            id_, kind, flags, name = r.u32(), r.u32(), r.u32(), r.s()
            offset, size = r.u64(), r.u64()
            r.take(4+4+8+4)   # alineación, reservado, bytes de pesos, fórmula de memoria
            first, count = r.u32(), r.u32()
            r.take(4+16+16)   # dependencia, hash compuesto, hash de bytes
            self.components.append(Component(id_, kind, bool(flags & 1), name, offset, size, first, count))
        r = _In(sections[3], strings)
        self.tensors = {}
        for _ in range(r.u32()):
            name, component, layer, dtype = r.s(), r.u32(), struct.unpack('<i', r.take(4))[0], r.s()
            rank = r.u32(); r.u32()
            shape = tuple(r.u64() for _ in range(4))[:rank]
            logical_last, offset, size, experts = r.u64(), r.u64(), r.u64(), r.u32(); r.u32()
            t = Tensor(name, component, layer, dtype, shape, logical_last, offset, size, experts,
                       bytes(r.take(16)), r.s(), r.u64())
            if name in self.tensors:
                raise ValueError(f'tensor repetido: {name}')
            self.tensors[name] = t
        r = _In(sections[6], strings)
        self.properties = {}
        for _ in range(r.u32()):
            component, key = r.u32(), r.s()
            self.properties[(component, key)] = r.value()

    def component(self, kind):
        return next((c for c in self.components if c.kind == kind), None)

    def raw(self, name, expert=None):
        """Bytes de un tensor (o de un experto) con memmap, filas por superbloques o elementos."""
        t = self.tensors[name]
        shape = t.shape if expert is None else t.shape[1:]
        if expert is not None and not t.experts:
            raise ValueError(f'{name}: no tiene eje de expertos')
        elements = int(np.prod(shape))
        size = t.size if expert is None else t.size//t.experts
        offset = t.offset + (expert or 0)*size
        if t.dtype in BLOCK:
            return np.memmap(self.path, mode='r', dtype=np.uint8, offset=offset, shape=(elements//shape[-1], shape[-1]//256*BLOCK[t.dtype]))
        dtype = {'fp16': '<f2', 'bf16': '<u2', 'fp32': '<f4'}[t.dtype]
        return np.memmap(self.path, mode='r', dtype=dtype, offset=offset, shape=shape)

    def tensor(self, name, expert=None):
        """FP32 con la forma lógica: el relleno de la última dimensión se comprueba y se quita."""
        t = self.tensors[name]
        raw = self.raw(name, expert)
        if t.dtype in BLOCK:
            value = decode(t.dtype, raw).reshape(raw.shape[0], -1)
            value = value.reshape((t.shape if expert is None else t.shape[1:]))
        else:
            value = decode(t.dtype, raw)
        if t.logical_last < value.shape[-1]:
            if not np.all(value[..., t.logical_last:] == 0):
                raise ValueError(f'{name}: relleno distinto de cero')
            value = value[..., :t.logical_last].copy()
        if not np.isfinite(value).all():
            raise ValueError(f'{name}: valores no finitos')
        return value


def _fields5(packed, count):
    """`count` campos de 5 bits seguidos, LSB primero."""
    packed = np.pad(packed, ((0, 0), (0, 1)))
    out = []
    for bit in range(0, 5*count, 5):
        word = packed[:, bit//8].astype(np.uint16) | (packed[:, bit//8+1].astype(np.uint16) << 8)
        out.append(((word >> (bit % 8)) & 31).astype(np.float32))
    return np.stack(out, axis=1)


def decode_hq44(raw):
    """HQ44: d, dm en f16; 16 escalas y 16 mínimos de 5 bits; códigos de 4 bits."""
    blocks = np.ascontiguousarray(raw).reshape(-1, 152)
    d = blocks[:, :2].copy().view('<f2').reshape(-1).astype(np.float32)
    dm = blocks[:, 2:4].copy().view('<f2').reshape(-1).astype(np.float32)
    fields = _fields5(blocks[:, 4:24], 32)
    codes = np.empty((len(blocks), 256), dtype=np.float32)
    codes[:, 0::2] = blocks[:, 24:] & 15
    codes[:, 1::2] = blocks[:, 24:] >> 4
    return ((d[:, None]*fields[:, :16]).repeat(16, axis=1)*codes - (dm[:, None]*fields[:, 16:]).repeat(16, axis=1)).reshape(-1)


def decode_hq52(raw):
    """HQ52: d en f16; 32 pasos de 5 bits (d·q/31); códigos de 5 bits en [-16, 15]."""
    b = np.ascontiguousarray(raw).reshape(-1, 184)
    d = b[:, :2].copy().view('<f2').ravel().astype(np.float32)
    steps = (d[:, None]*_fields5(b[:, 4:24], 32))/np.float32(31.)
    payload = b[:, 24:].reshape(-1, 32, 5).astype('u8')
    words = np.zeros(payload.shape[:2], dtype='u8')
    for j in range(5):
        words |= payload[:, :, j] << (j*8)
    codes = np.stack([((words >> (j*5)) & 31).astype('i2')-16 for j in range(8)], axis=-1)
    return (codes.astype('f4')*steps[:, :, None]).reshape(-1)


def decode_hq8k(raw):
    """HQ8K: d en f16; 8 pasos de grupo (d·q/255); 256 códigos i8."""
    b = np.ascontiguousarray(raw).reshape(-1, 272)
    d = b[:, :2].copy().view('<f2').ravel().astype(np.float32)
    steps = d[:, None]*b[:, 2:10].astype(np.float32)/np.float32(255.)
    codes = b[:, 16:].view(np.int8).astype(np.float32).reshape(-1, 8, 32)
    return (codes*steps[:, :, None]).reshape(-1)


def decode(dtype, raw):
    if dtype == 'fp16':
        return np.asarray(raw).astype(np.float32)
    if dtype == 'bf16':
        return (np.asarray(raw).astype(np.uint32) << 16).view(np.float32)
    if dtype == 'fp32':
        return np.array(raw, np.float32)
    decoder = {'hq44k_g16': decode_hq44, 'hq52k': decode_hq52, 'hq8k': decode_hq8k}.get(dtype)
    if decoder is None:
        raise ValueError(f'sin decodificador para {dtype}')
    return decoder(raw)


if __name__ == '__main__':
    hnf = Hnf10(sys.argv[1])
    print(f'HNFv10 secuencia {hnf.sequence}, identidad {hnf.identity}')
    for c in hnf.components:
        print(f'  componente {c.id} «{c.name}» tipo {c.kind}: {c.size} bytes en {c.offset}, {c.tensor_count} tensores')
    for (component, key), value in hnf.properties.items():
        print(f'  propiedad de {component}: {key} = {value}')
    for t in hnf.tensors.values():
        print(f'  {t.name} {t.dtype} {list(t.shape)} lógico {t.logical_last} ← {t.source}')
