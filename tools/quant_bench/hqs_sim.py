"""Simulador fiel de HQ4.1K / HQ5.1K, portado de src/hqs/{compact,common}.rs.

Permite probar variantes del formato sobre los pesos reales sin escribir una
linea de Rust ni de CUDA. Todo se hace en numpy y se mide con la misma metrica
de aciertos que ya usamos contra el motor.
"""
import os
import numpy as np

SUPER = 256
EPS = 1e-7

def f16(x):
    return np.float16(x).astype(np.float32)

def quant_dequant(w, q_max, group_size=8, scale_bits=4, super_size=SUPER):
    """Devuelve w reconstruido tras pasar por el formato.

    group_size / scale_bits parametrizan la variante: el formato actual es
    (8, 4); la hipotesis a batir es (16, 6), que ademas ocupa menos.
    """
    shape = w.shape
    flat = w.astype(np.float32).ravel()
    pad = (-flat.size) % super_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.float32)])
    blocks = flat.reshape(-1, super_size)
    ngroups = super_size // group_size
    g = blocks.reshape(-1, ngroups, group_size)

    if os.environ.get('HQS_WLSQ') == '1':
        # Ajuste por minimos cuadrados PONDERADO POR MAGNITUD, portado de
        # make_qkx2_quants de llama.cpp (ggml/src/ggml-quants.c).
        #
        # Nuestro compute_group_params coge los extremos: scale = max - min.
        # Eso le da el mismo voto a un peso enorme que a uno insignificante.
        # llama.cpp pondera cada elemento por su magnitud —w = av_x + |x|,
        # y eso YA SIN matriz de importancia— y para cada asignacion tentativa
        # de indices resuelve en forma cerrada la (escala, minimo) optimos.
        av = np.sqrt(2.0 * (g * g).sum(axis=2) / g.shape[2])[:, :, None]
        w = av + np.abs(g)
        gmax = g.max(axis=2, keepdims=True)
        gmn = np.minimum(g.min(axis=2, keepdims=True), 0.0)   # if (min > 0) min = 0
        rng = np.maximum(gmax - gmn, EPS)
        sw = w.sum(axis=2, keepdims=True)
        sx = (w * g).sum(axis=2, keepdims=True)
        best_s = rng / q_max
        best_m = gmn
        best_e = np.full(gmn.shape, np.inf)
        RMIN, RDELTA, NSTEP = -0.5, 0.1, 15
        for it in range(NSTEP + 2):
            isc = (q_max if it == 0 else (RMIN + RDELTA * (it - 1) + q_max)) / rng
            L = np.clip(np.round(isc * (g - gmn)), 0, q_max)
            sl = (w * L).sum(axis=2, keepdims=True)
            sl2 = (w * L * L).sum(axis=2, keepdims=True)
            sxl = (w * L * g).sum(axis=2, keepdims=True)
            D = sw * sl2 - sl * sl
            ok = D > 0
            ts = np.where(ok, (sw * sxl - sx * sl) / np.where(ok, D, 1.0), rng / q_max)
            tm = np.where(ok, (sl2 * sx - sl * sxl) / np.where(ok, D, 1.0), gmn)
            tm = np.minimum(tm, 0.0)
            ts = np.maximum(ts, EPS)
            err = (w * (ts * L + tm - g) ** 2).sum(axis=2, keepdims=True)
            better = err < best_e
            best_e = np.where(better, err, best_e)
            best_s = np.where(better, ts, best_s)
            best_m = np.where(better, tm, best_m)
        # el resto del pipeline espera (min, rango) en fp16
        gmin = f16(best_m[:, :, 0])
        gscale = np.maximum(f16(np.maximum(best_s[:, :, 0] * q_max, EPS)), EPS)
    else:
        # 1) parametros por grupo, redondeados a fp16 como hace common.rs
        gmin = f16(g.min(axis=2))
        gscale = np.maximum(f16(np.maximum(g.max(axis=2) - g.min(axis=2), EPS)), EPS)

    # 2) parametros del superbloque, tambien en fp16 antes de derivar nada
    #
    # d_scale sale del MAXIMO de los 32 grupos. Con solo 16 escalas posibles,
    # un unico grupo atipico estira el rango y deja a los otros 31 sin
    # resolucion. Con shrink<1 se sacrifica ese grupo (satura a 15) a cambio
    # de que el resto cuantice mejor. Coste en bits: CERO.
    shrink = float(os.environ.get('HQS_SHRINK', '1.0'))
    smax = f16(gscale.max(axis=1, keepdims=True) * shrink)
    mbase = f16(gmin.min(axis=1, keepdims=True))
    dmin = f16(gmin.max(axis=1, keepdims=True) - mbase)

    qmaxs = float((1 << scale_bits) - 1)
    if os.environ.get('HQS_JOINT') == '1':
        # Elegir q_s/q_m entre los valores REPRESENTABLES, minimizando el error
        # real de reconstruccion de los 8 pesos del grupo.
        #
        # El conversor actual optimiza (min, scale) en el continuo y DESPUES
        # los machaca a 4 bits: se optimiza contra un objetivo que el paso
        # siguiente destruye. Aqui se busca directamente sobre lo que el
        # decodificador puede representar. Mismos bits, misma cabecera, mismo
        # kernel — solo se elige mejor.
        # Ventana estrecha: el optimo casi siempre esta pegado al redondeo.
        # HQS_JOINT=2 prueba solo {-1,0,+1} -> 6 evaluaciones en vez de 64.
        WIN = os.environ.get('HQS_JOINT') == '2'
        cand = np.arange(qmaxs + 1, dtype=np.float32)          # 0..15
        qs = np.round(np.where(smax > EPS, gscale / np.maximum(smax, EPS), 0) * qmaxs)
        qs = np.clip(np.where((gscale > EPS) & (qs == 0), 1, qs), 0, qmaxs)
        qm = np.clip(np.round(np.where(dmin > EPS, (gmin - mbase) / np.maximum(dmin, EPS), 0) * qmaxs), 0, qmaxs)
        gw = g[:, :, None, :]                                   # [B, G, 1, gs]
        for _ in range(1 if WIN else 2):                        # alternar q_s / q_m
            for who in ('s', 'm'):
                ref = qs if who == 's' else qm
                cc = (np.clip(ref[:, :, None] + np.array([-1., 0., 1.]), 0, qmaxs)
                      if WIN else np.broadcast_to(cand, (*ref.shape, cand.size)))
                if who == 's':
                    es = smax[:, :, None] * cc / qmaxs
                    em = np.broadcast_to((mbase + dmin * qm / qmaxs)[:, :, None], cc.shape)
                else:
                    es = np.broadcast_to((smax * qs / qmaxs)[:, :, None], cc.shape)
                    em = mbase[:, :, None] + dmin[:, :, None] * cc / qmaxs
                esb = es[:, :, :, None]; emb = em[:, :, :, None]
                qq = np.where(esb > EPS, np.clip(np.round((gw - emb) / np.maximum(esb, EPS) * q_max), 0, q_max), 0)
                err = ((emb + (qq / q_max) * esb - gw) ** 2).sum(axis=3)   # [B, G, 16]
                bi = np.argmin(err, axis=2)
                best = np.take_along_axis(cc, bi[:, :, None], axis=2)[:, :, 0].astype(np.float32)
                if who == 's':
                    qs = np.where(gscale > EPS, np.maximum(best, 1), 0)
                else:
                    qm = best
        eff_scale = smax * qs / qmaxs
        eff_min = mbase + dmin * qm / qmaxs
        es = eff_scale[:, :, None]; em = eff_min[:, :, None]
        q = np.where(es > EPS, np.clip(np.round((g - em) / np.maximum(es, EPS) * q_max), 0, q_max), 0)
        out = em + (q / q_max) * es
        out = out.reshape(-1)[:w.size] if pad else out.reshape(-1)
        return out.reshape(shape).astype(np.float32)
    # 3) escalas y minimos cuantizados a scale_bits
    qs = np.clip(np.round(np.where(smax > EPS, gscale / np.maximum(smax, EPS), 0) * qmaxs), 0, qmaxs)
    qs = np.where((gscale > EPS) & (qs == 0), 1, qs)      # no colapsar un grupo con rango
    qm = np.clip(np.round(np.where(dmin > EPS, (gmin - mbase) / np.maximum(dmin, EPS), 0) * qmaxs), 0, qmaxs)

    # 4) parametros efectivos: los que vera el kernel
    eff_scale = smax * qs / qmaxs
    eff_min = mbase + dmin * qm / qmaxs

    # 5) pesos contra los parametros efectivos
    es = eff_scale[:, :, None]
    em = eff_min[:, :, None]
    q = np.where(es > EPS, np.clip(np.round((g - em) / np.maximum(es, EPS) * q_max), 0, q_max), 0)
    out = em + (q / q_max) * es

    out = out.reshape(-1)[:w.size] if pad else out.reshape(-1)
    return out.reshape(shape).astype(np.float32)

def bits_per_weight(group_size=8, scale_bits=4, weight_bits=4, super_size=SUPER):
    ngroups = super_size // group_size
    header_bits = ngroups * 2 * scale_bits + 3 * 16 + 16   # escalas+minimos, 3 fp16, pad
    return (header_bits + super_size * weight_bits) / super_size

if __name__ == '__main__':
    for gs, sb in ((8,4), (16,6), (16,5), (32,6)):
        print(f"grupo {gs:2d}, escala {sb} bits -> "
              f"HQ4 {bits_per_weight(gs,sb,4):.3f} bpw   HQ5 {bits_per_weight(gs,sb,5):.3f} bpw")
