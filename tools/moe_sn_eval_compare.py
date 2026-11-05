"""Compara ejecuciones de gemma4_moe_generate --eval según el protocolo gemma4-sn-eval-v1.

uso: moe_sn_eval_compare.py casos/ runs/ prompts.jsonl salida/

Los logits se comparan en FP64 sobre el vocabulario completo. Unidad: documento.
La hoja ciega no dice qué variante es cada respuesta; la clave va en otro archivo.
"""
import json
import re
import sys
from pathlib import Path
import numpy as np

cases, runs, prompts, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
PAIRS = (('A1', 'A2'), ('A1', 'B1'), ('A-ca', 'B-ca'))
FIRST, MAX_FORCING, MARGIN = 16, 512, 0.01
output.mkdir(parents=True, exist_ok=True)
records = [json.loads(line) for line in prompts.read_text().splitlines()]
names = [f"{i:02}-{r['id']}" for i, r in enumerate(records)]
domains = np.array([r['domain'] for r in records])
vocab = None


def visible(run, name):
    """Lo que ve la persona: el chat retira el bloque de pensamiento vacío con el que abre."""
    text = (runs/run/f'{name}.output.txt').read_text(encoding='utf8', errors='replace')
    return re.sub(r'^\s*<\|channel>thought\n\s*<channel\|>', '', text)


def log_softmax(logits):
    shifted = logits-logits.max()
    return shifted-np.log(np.exp(shifted).sum())


def summary(values):
    values = np.asarray(values, np.float64)
    return {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)), 'max': float(values.max())}


def compare_case(name, a, b):
    ids = np.fromfile(cases/f'{name}.forcing.ids', dtype='<i4')[:MAX_FORCING]
    positions = len(ids)-1-FIRST
    x = np.memmap(runs/a/f'{name}.logits.f16', dtype='<f2', mode='r').reshape(positions, -1)
    y = np.memmap(runs/b/f'{name}.logits.f16', dtype='<f2', mode='r').reshape(positions, -1)
    rows = {key: np.empty(positions) for key in ('nll_a', 'nll_b', 'agree', 'top8', 'kl_ab', 'kl_ba', 'js', 'l2', 'max_abs')}
    for p in range(positions):
        u, v = x[p].astype(np.float64), y[p].astype(np.float64)
        assert np.isfinite(u).all() and np.isfinite(v).all(), name
        lu, lv = log_softmax(u), log_softmax(v)
        pu, pv = np.exp(lu), np.exp(lv)
        mix = np.log((pu+pv)/2)
        target = ids[p+FIRST+1]
        rows['nll_a'][p], rows['nll_b'][p] = -lu[target], -lv[target]
        rows['agree'][p] = u.argmax() == v.argmax()
        rows['top8'][p] = len(set(np.argpartition(-u, 8)[:8]) & set(np.argpartition(-v, 8)[:8]))/8
        rows['kl_ab'][p] = (pu*(lu-lv)).sum()
        rows['kl_ba'][p] = (pv*(lv-lu)).sum()
        rows['js'][p] = ((pu*(lu-mix)).sum()+(pv*(lv-mix)).sum())/2
        difference = (u-u.mean())-(v-v.mean())
        rows['l2'][p] = np.sqrt((difference**2).sum())
        rows['max_abs'][p] = np.abs(u-v).max()
    return rows


def bootstrap(deltas):
    rng = np.random.default_rng(26)
    means = [deltas[rng.integers(len(deltas), size=len(deltas))].mean() for _ in range(2000)]
    low, high = np.percentile(means, [2.5, 97.5])
    verdict = 'por_debajo_del_margen' if high < MARGIN else ('regresion' if low > MARGIN else 'inconcluso')
    return {'documents': len(deltas), 'mean_delta_nll': float(deltas.mean()), 'ci95': [float(low), float(high)], 'margin_verdict': verdict}


def compare(a, b):
    documents = {}
    for name in names:
        rows = compare_case(name, a, b)
        documents[name] = {
            'positions': len(rows['nll_a']),
            'nll_a': float(rows['nll_a'].mean()), 'nll_b': float(rows['nll_b'].mean()),
            'delta_nll': float((rows['nll_b']-rows['nll_a']).mean()),
            'argmax_agreement': float(rows['agree'].mean()), 'top8_overlap': float(rows['top8'].mean()),
            **{key: summary(rows[key]) for key in ('kl_ab', 'kl_ba', 'js', 'l2', 'max_abs')},
        }
        print(a, b, name, f"dNLL {documents[name]['delta_nll']:+.4f}", flush=True)
    deltas = np.array([documents[name]['delta_nll'] for name in names])
    groups = {'total': np.ones(len(names), bool)}
    groups.update({str(d): domains == d for d in sorted(set(domains))})
    result = {'documents': documents, 'groups': {}}
    for group, mask in groups.items():
        chosen = [documents[n] for n, m in zip(names, mask) if m]
        result['groups'][group] = {
            **bootstrap(deltas[mask]),
            'mean_nll_a': float(np.mean([d['nll_a'] for d in chosen])), 'mean_nll_b': float(np.mean([d['nll_b'] for d in chosen])),
            'documents_better_b': int((deltas[mask] < 0).sum()), 'documents_worse_b': int((deltas[mask] > 0).sum()),
            'mean_argmax_agreement': float(np.mean([d['argmax_agreement'] for d in chosen])),
            'mean_top8_overlap': float(np.mean([d['top8_overlap'] for d in chosen])),
            'median_js': float(np.median([d['js']['median'] for d in chosen])),
        }
    return result


def json_test(text, expected):
    """Estricto: el valor JSON de toda la respuesta. Aparte, si solo lo envuelve una valla."""
    def parsed(candidate):
        try:
            return json.loads(candidate), True
        except ValueError:
            return None, False
    value, valid = parsed(text.strip())
    fenced = re.fullmatch(r'\s*```(?:json)?\s*([\s\S]*?)\s*```\s*', text)
    loose, loose_valid = parsed(fenced.group(1)) if fenced else (value, valid)
    return {'strict': valid and value == expected, 'fenced': loose_valid and loose == expected}


def responses(run):
    rows = {}
    table = {line.split('\t')[0]: line.split('\t') for line in (runs/run/'runs.tsv').read_text().splitlines()[1:]}
    for name, record in zip(names, records):
        text = visible(run, name)
        rows[name] = {'tokens': int(table[name][4]), 'stop': table[name][5]}
        if record.get('expected_json') is not None:
            rows[name]['json'] = json_test(text, record['expected_json'])
    return rows


def blind_sheet(a, b):
    rng = np.random.default_rng(26)
    key, sheet = {}, ['# Hoja ciega gemma4-sn-eval-v1', '',
                      'Puntuar 0–2 en corrección, instrucciones y claridad. La clave está en otro archivo.', '']
    for name, record in zip(names, records):
        first, second = (a, b) if rng.integers(2) == 0 else (b, a)
        key[name] = {'X': first, 'Y': second}
        task = record['messages'][0]['content'] if record.get('messages') else '(continuación de conversación sintética)'
        sheet += [f'## {name}', '', f"Rúbrica: {record.get('rubric', '')}", '', '### Petición', '', task, '']
        for label, run in (('X', first), ('Y', second)):
            sheet += [f'### Respuesta {label}', '', visible(run, name), '']
    (output/'hoja-ciega.md').write_text('\n'.join(sheet), encoding='utf8')
    (output/'clave-ciega.json').write_text(json.dumps(key, indent=1))


report = {'kind': 'gemma4_sn_eval_v1', 'protocol_margin_nats': MARGIN, 'pairs': {}, 'responses': {}}
for run in ('A1', 'A2', 'B1', 'A-ca', 'B-ca'):
    report['responses'][run] = responses(run)
for a, b in PAIRS:
    report['pairs'][f'{a}/{b}'] = compare(a, b)
for a, b in PAIRS:
    same = [n for n in names if (runs/a/f'{n}.output.ids').read_bytes() == (runs/b/f'{n}.output.ids').read_bytes()]
    report['pairs'][f'{a}/{b}']['identical_responses'] = len(same)
(output/'comparison.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
blind_sheet('A1', 'B1')
print(f"{'par':12}{'grupo':18}{'docs':>5}{'NLL A':>9}{'NLL B':>9}{'delta':>9}{'IC 95 %':>22}{'argmax':>8}{'top8':>7}  veredicto")
for pair, result in report['pairs'].items():
    for group, g in result['groups'].items():
        print(f"{pair:12}{group:18}{g['documents']:5}{g['mean_nll_a']:9.4f}{g['mean_nll_b']:9.4f}{g['mean_delta_nll']:+9.4f}"
              f"  [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]{g['mean_argmax_agreement']:8.3f}{g['mean_top8_overlap']:7.3f}  {g['margin_verdict']}")
    print(pair, 'respuestas idénticas', result['identical_responses'], 'de', len(names))
for run, rows in report['responses'].items():
    tests = [r['json'] for r in rows.values() if 'json' in r]
    print(run, 'JSON estricto', sum(t['strict'] for t in tests), 'con valla', sum(t['fenced'] for t in tests), 'de', len(tests),
          'paradas', {s: sum(r['stop'] == s for r in rows.values()) for s in {r['stop'] for r in rows.values()}})
