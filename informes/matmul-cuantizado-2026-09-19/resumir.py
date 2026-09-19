"""Summarize paired before/after runs and require byte-identical answers."""
import json
from pathlib import Path
import re
import statistics
import sys

root = Path(sys.argv[1])
summary = {}
for model in (sys.argv[2:] or ('12b', 'e4b')):
    cases = {}
    for repeat in (0, 128, 384):
        groups = {'before': [], 'after': []}
        texts = []
        for pair in range(3):
            for mode in groups:
                directory = root / f'{model}-{pair}-{mode}'
                lines = (directory / 'run.log').read_text().splitlines()
                records = [dict(re.findall(r'(\w+)=([^ ]+)', line))
                           for line in lines if line.startswith('BENCH ')]
                measured = [r for r in records if int(r['repeat']) == repeat and r['trial'] == '1']
                assert len(measured) == 1, (directory, repeat)
                record = measured[0]
                assert record['output'] == '128' and record['fallbacks'] == '0', record
                groups[mode].append(record)
                texts.append((directory / f'prefill-1-{repeat}-1.txt').read_bytes())
        assert all(t == texts[0] for t in texts), (model, repeat, 'answers differ')
        a = statistics.median(float(r['prefill_ms']) for r in groups['before'])
        b = statistics.median(float(r['prefill_ms']) for r in groups['after'])
        cases[repeat] = dict(prompt_tokens=int(groups['before'][0]['prompt']),
            before_ms=a, after_ms=b, reduction_percent=100*(1-b/a),
            paired_reductions=[100*(1-float(y['prefill_ms'])/float(x['prefill_ms']))
                               for x,y in zip(groups['before'], groups['after'])],
            before_decode_tps=statistics.median(float(r['tok_s']) for r in groups['before']),
            after_decode_tps=statistics.median(float(r['tok_s']) for r in groups['after']),
            answers_identical=True, samples=groups)
    summary[model] = cases
print(json.dumps(summary, indent=2))
