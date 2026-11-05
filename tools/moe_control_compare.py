"""Compara los volcados de gemma4_moe_control_replay con referencias en CPU.

uso: moe_control_compare.py modelo.hnf modelo-original/ captura/ pilot-down/ replay-gpu/ informe.json [pilot-shared-down/]

Kernel: mismos pesos en ambos lados. Calidad: pesos down originales con la misma
activación capturada, para aislar down; shared_branch_post queda fijo en la GPU.
Con pilot-shared-down/ compara además down compartido con la rama experta S/n fija,
sobre la entrada compartida guardada por cada GPU.
"""
import json
import sys
from pathlib import Path
import numpy as np

hnf, model, capture, payloads, replay, report_path = sys.argv[1:7]
shared_payloads = Path(sys.argv[7]) if len(sys.argv) > 7 else None
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma4_reference_common import Weights, h16
from hnf10 import decode_hq44
capture, payloads, replay = Path(capture), Path(payloads), Path(replay)

LAYERS, VARIANTS = (0, 5, 15, 29), ('unit', 'second_moment', 'routing_weighted')
K, H, I, P, SHARED = 8, 2816, 704, 768, 2112


def rel(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.sqrt(((got-want)**2).sum()/(want**2).sum()))


def per_token(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return np.sqrt(((got-want)**2).sum(-1)/np.maximum((want**2).sum(-1), 1e-300))


def rms(x, w):
    x = np.asarray(x, np.float64)
    return x/np.sqrt((x*x).mean(-1, keepdims=True)+1e-6)*w


def read(path, shape):
    return np.fromfile(path, dtype='<f4').reshape(shape)


def shared_down(prefix, layer, directory, r, documents):
    T = len(r)
    a = read(directory/'shared_down_input.f32', (T, SHARED)).astype(np.float64)
    unit = weights.packed(prefix+'mlp.down.weight').astype(np.float64)
    original = weights.original(prefix+'mlp.down.weight').astype(np.float64)
    raw = np.fromfile(shared_payloads/f'layer-{layer:02}-shared-down-second_moment.hq44', dtype=np.uint8).reshape(H, 1368)
    candidate = decode_hq44(raw).reshape(H, 2304)[:, :SHARED].astype(np.float64)
    scale = weights.packed(prefix+'ln_mlp_branch_post.weight').astype(np.float64)
    original_scale = weights.original(prefix+'ln_mlp_branch_post.weight').astype(np.float64)
    common = weights.packed(prefix+'ln_mlp_post.weight').astype(np.float64)
    scalar = weights.packed(prefix+'layer_scalar').astype(np.float64)
    expert = read(directory/'second_moment_expert_branch_post.f32', (T, H))
    assert np.array_equal(expert, read(directory/'second_moment_shared_expert_branch_post.f32', (T, H)))
    target_rows = a@original.T
    target_post = rms(target_rows, original_scale)
    target_output = (r+rms(target_post+expert.astype(np.float64), common))*scalar
    result, tokens = {}, {}
    for name, dump, matrix in (('unit', 'second_moment', unit), ('second_moment', 'second_moment_shared', candidate)):
        rows = a@matrix.T
        exact_post = rms(rows, scale)
        got_rows = read(directory/f'{dump}_shared_down_rows.f32', (T, H))
        got_post = read(directory/f'{dump}_shared_branch_post.f32', (T, H))
        got_combined = read(directory/f'{dump}_combined.f32', (T, H))
        got_output = read(directory/f'{dump}_output.f32', (T, H))
        assert np.array_equal(got_combined, h16(got_post+expert))
        control_rows = h16(rows).astype(np.float64)
        control_post = h16(rms(control_rows, scale)).astype(np.float64)
        tokens[name] = per_token(got_post, target_post)
        result[name] = {
            'kernel_same_weights': {
                'down_rows': {'gpu': rel(got_rows, rows), 'control': rel(control_rows, rows)},
                'branch_post': {'gpu': rel(got_post, exact_post), 'control': rel(control_post, exact_post)},
            },
            'quality_vs_original_shared_down': {
                'down_rows': rel(got_rows, target_rows),
                'branch_post': rel(got_post, target_post),
                'output': rel(got_output, target_output),
                'branch_post_token_p50': float(np.median(tokens[name])),
                'branch_post_token_p95': float(np.percentile(tokens[name], 95)),
            },
        }
        for item in result[name]['kernel_same_weights'].values():
            item['ratio'] = item['gpu']/item['control']
    result['paired_vs_unit'] = paired(tokens['unit'], tokens['second_moment'], documents)
    return result


def paired(base, cand, documents):
    gain = 1-cand/base
    names = sorted(set(documents))
    groups = [gain[documents == name] for name in names]
    rng = np.random.default_rng(26)
    medians = [np.median(np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]))
               for _ in range(2000)]
    domains = np.array([name.split('-')[0] for name in documents])
    return {
        'median_token_improvement': float(np.median(gain)),
        'median_95_by_document_bootstrap': [float(x) for x in np.percentile(medians, [2.5, 97.5])],
        'by_domain': {d: float(np.median(gain[domains == d])) for d in sorted(set(domains))},
        'tokens_improved': int((cand < base).sum()),
        'tokens_worse': int((cand > base).sum()),
        'tokens_over_5_percent_worse': int((cand > 1.05*base).sum()),
    }


manifest = json.loads((capture/'manifest.json').read_text())
documents = np.array([line.split('\t')[0] for line in (capture/'control-tokens.tsv').read_text().splitlines()[1:]])
weights = Weights(hnf, model, manifest['model']['hnf_sha256'])
report = {'kind': 'gpu_control_replay', 'weights_scope': 'down HQ44 unitario y candidatos; gate_up sin recalibrar; down compartido solo en el bloque shared_down', 'gpus': {}}
for gpu in sorted(p.name for p in replay.iterdir() if p.is_dir()):
    report['gpus'][gpu] = {}
    for layer in LAYERS:
        prefix = f'text.layer{layer}.'
        control = capture/f'control-layer-{layer}'
        r = read(control/'residual.f32', (-1, H))
        T = len(r)
        ids = np.fromfile(control/'expert_ids.i32', dtype='<i4').reshape(T, K)
        w = read(control/'expert_weights.f32', (T, K)).astype(np.float64)
        a = read(control/'down_input.f32', (T, K, I)).astype(np.float64)
        scale_post = weights.packed(prefix+'ln_moe_branch_post.weight').astype(np.float64)
        scale_common = weights.packed(prefix+'ln_mlp_post.weight').astype(np.float64)
        scalar = weights.packed(prefix+'layer_scalar').astype(np.float64)
        original_post = weights.original(prefix+'ln_moe_branch_post.weight').astype(np.float64)
        directory = replay/gpu/f'layer-{layer:02}'
        original_rows = np.empty((T, K, H))
        down_unit, down_original = {}, {}
        for expert in np.unique(ids):
            expert = int(expert)
            down_unit[expert] = weights.packed(prefix+'moe.experts.down.weight', expert).astype(np.float64)
            down_original[expert] = weights.original(prefix+'moe.experts.down.weight', expert).astype(np.float64)
            ts, slots = np.where(ids == expert)
            original_rows[ts, slots] = a[ts, slots]@down_original[expert].T
        target_sum = (original_rows*w[..., None]).sum(1)
        target_post = rms(target_sum, original_post)
        shared = read(directory/'unit_shared_branch_post.f32', (T, H)).astype(np.float64)
        target_output = (r+rms(shared+target_post, scale_common))*scalar

        result, tokens = {}, {}
        for variant in VARIANTS:
            rows = np.empty((T, K, H))
            for expert in down_unit:
                file = payloads/f'layer-{layer:02}-expert-{expert:03}-{variant}.hq44'
                matrix = down_unit[expert]
                if variant != 'unit' and file.exists():
                    raw = np.fromfile(file, dtype=np.uint8).reshape(H, P//256*152)
                    matrix = decode_hq44(raw).reshape(H, P)[:, :I].astype(np.float64)
                ts, slots = np.where(ids == expert)
                rows[ts, slots] = a[ts, slots]@matrix.T
            exact_sum = (rows*w[..., None]).sum(1)
            exact_post = rms(exact_sum, scale_post)
            got_rows = read(directory/f'{variant}_expert_rows.f32', (T, K, H))
            got_sum = read(directory/f'{variant}_expert_sum.f32', (T, H))
            got_post = read(directory/f'{variant}_expert_branch_post.f32', (T, H))
            got_output = read(directory/f'{variant}_output.f32', (T, H))
            control_rows = h16(rows).astype(np.float64)
            control_sum = h16((control_rows*w[..., None]).sum(1)).astype(np.float64)
            control_post = h16(rms(control_sum, scale_post)).astype(np.float64)
            result[variant] = {
                'kernel_same_weights': {
                    'down_rows': {'gpu': rel(got_rows, rows), 'control': rel(control_rows, rows)},
                    'expert_sum': {'gpu': rel(got_sum, exact_sum), 'control': rel(control_sum, exact_sum)},
                    'branch_post': {'gpu': rel(got_post, exact_post), 'control': rel(control_post, exact_post)},
                },
                'quality_vs_original_down': {
                    'expert_sum': rel(got_sum, target_sum),
                    'branch_post': rel(got_post, target_post),
                    'output': rel(got_output, target_output),
                    'branch_post_token_p50': float(np.median(per_token(got_post, target_post))),
                    'branch_post_token_p95': float(np.percentile(per_token(got_post, target_post), 95)),
                },
            }
            for name in ('down_rows', 'expert_sum', 'branch_post'):
                item = result[variant]['kernel_same_weights'][name]
                item['ratio'] = item['gpu']/item['control']
            tokens[variant] = per_token(got_post, target_post)
            if variant != 'unit':
                result[variant]['paired_vs_unit'] = paired(tokens['unit'], tokens[variant], documents)
        if shared_payloads:
            result['shared_down'] = shared_down(prefix, layer, directory, r, documents)
        report['gpus'][gpu][str(layer)] = result
Path(report_path).write_text(json.dumps(report, indent=1))
print(f"{'GPU':8}{'capa':>5}{'variante':>18}{'filas':>8}{'suma':>7}{'norma':>7}{'suma%':>8}{'rama%':>8}{'p50%':>7}{'p95%':>7}{'salida%':>9}")
for gpu, layers in report['gpus'].items():
    for layer, result in layers.items():
        for variant, item in result.items():
            if variant == 'shared_down':
                print(f"{gpu:8}{layer:>5}{'compartido':>18} {item['paired_vs_unit']}")
                continue
            k, q = item['kernel_same_weights'], item['quality_vs_original_down']
            print(f"{gpu:8}{layer:>5}{variant:>18}{k['down_rows']['ratio']:8.3f}{k['expert_sum']['ratio']:7.3f}{k['branch_post']['ratio']:7.3f}"
                  f"{q['expert_sum']*100:8.3f}{q['branch_post']*100:8.3f}{q['branch_post_token_p50']*100:7.2f}{q['branch_post_token_p95']*100:7.2f}{q['output']*100:9.4f}")
            if 'paired_vs_unit' in item:
                print(f"{'':31}pareada {item['paired_vs_unit']}")
