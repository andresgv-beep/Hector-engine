#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || ! $2 =~ ^[01]$ ]]; then
  echo 'Uso: bash reproducir.sh /ruta/modelo.hnf 0|1 (grafos)' >&2
  exit 2
fi
report_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$report_dir/../.." && pwd)
cuda_dir=${AUDIT_CUDA_ROOT:-/home/andres/opt/cuda13/usr/local/cuda-13.1}
bench_work=$(mktemp -d /tmp/hector-kv-graphs-repro.XXXXXX)
c++ -O3 -std=c++17 -I "$repo_dir/src" -I "$repo_dir/kernels" \
  -I "$cuda_dir/include" "$report_dir/benchmark.cpp" \
  "$repo_dir/build/libhelios-engine.a" -L "$cuda_dir/lib64" \
  "-Wl,-rpath,$cuda_dir/lib64" -lcudart -lcublas -lcublasLt -lpthread -ldl \
  -o "$bench_work/benchmark"
HELIOS_HOME="$bench_work" "$bench_work/benchmark" "$1" "$2"
