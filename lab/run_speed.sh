#!/usr/bin/env bash
# Velocidad estable / vx3 / v4 con el mismo binario, orden A B C C B A (protocolo de GPT).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
G=/home/andres/Documentos/GitHub
stable=$G/helios_convert_v9.1/output/gemma4_12b_unified.hnf
vx3=$G/helios-convert-hqs-vx3/lab/output/gemma4_12b_vx3_weighted_only_experimental.hnf
v4=$G/Hector-hqs-vx3/lab/results/claude-hqs-v4/gemma4_12b_hqs_v4_mlp_experimental.hnf
run() { python3 lab/runtime_bench.py --model "$2" --tag "$1" --tokens 128 --trials 3 | grep MEDIAN_TOK_S | sed "s/^/$1 /"; }
run speed-stable-1 "$stable"
run speed-vx3-1 "$vx3"
run speed-v4-1 "$v4"
grep hq44k lab/results/speed-v4-1/cache/tune.cache >> lab/cache/tune.cache
run speed-v4-2 "$v4"
run speed-vx3-2 "$vx3"
run speed-stable-2 "$stable"
