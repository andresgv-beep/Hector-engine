#!/usr/bin/env python3
"""Compara logits de Hector contra la referencia fp32. Metricas pensadas para
   cuantizacion: lo que importa es el orden y la distribucion, no el bit."""
import sys
import numpy as np
a = np.fromfile(sys.argv[1], dtype=np.float32)   # referencia
b = np.fromfile(sys.argv[2], dtype=np.float32)   # Hector
assert a.shape == b.shape, (a.shape, b.shape)
d = b - a
print(f"vocab {a.size}")
print(f"  error abs   max {np.abs(d).max():.4f}   medio {np.abs(d).mean():.4f}   rms {np.sqrt((d**2).mean()):.4f}")
print(f"  correlacion {np.corrcoef(a, b)[0,1]:.6f}")
ra, rb = np.argsort(-a), np.argsort(-b)
print(f"  argmax      ref {ra[0]}  hector {rb[0]}   {'COINCIDE' if ra[0]==rb[0] else 'DIFIERE'}")
for k in (1, 5, 10, 50, 100):
    ov = len(set(ra[:k].tolist()) & set(rb[:k].tolist()))
    print(f"  top-{k:<4} solapamiento {ov}/{k}  ({100*ov/k:.0f}%)")
# probabilidades: lo que de verdad ve el muestreador
def sm(x):
    e = np.exp(x - x.max()); return e / e.sum()
pa, pb = sm(a), sm(b)
kl = float(np.sum(pa * np.log(np.clip(pa, 1e-30, None) / np.clip(pb, 1e-30, None))))
print(f"  KL(ref||hector) {kl:.6f} nats   masa top-1 ref {pa.max():.4f} hector {pb.max():.4f}")
