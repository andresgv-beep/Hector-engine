"""Run two prebuilt statically-linked benchmarks sequentially, with AB/BA/AB order.
Usage: comparar.py model.hnf before_binary after_binary result_directory label
Requires a free GPU; does not stop or restart user processes.
"""
import os
from pathlib import Path
import subprocess
import sys

model, before, after, dest, label = sys.argv[1:]
model, before, after = map(lambda s: str(Path(s).resolve()), (model, before, after))
root = Path(dest).resolve()
root.mkdir(parents=True, exist_ok=True)
home = root / 'home'
home.mkdir(exist_ok=True)
env = dict(os.environ, HELIOS_HOME=str(home), HELIOS_EMBED_MMAP='1',
           HELIOS_VISION_MMAP='1', ATTENTION_TRIALS='2')
# Use the same home/tune.cache for all modes; optionally seed it before running.
for pair in range(3):
    for mode in (('before', 'after') if pair % 2 == 0 else ('after', 'before')):
        run = root / f'{label}-{pair}-{mode}'
        run.mkdir(exist_ok=True)
        print(label, pair, mode, flush=True)
        with (run / 'run.log').open('w') as log:
            subprocess.run([before if mode == 'before' else after, model, '1'],
                           env=dict(env, ATTENTION_OUTPUT_DIR=str(run)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
