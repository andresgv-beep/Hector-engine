#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 && $# != 2 ]]; then
  echo 'Uso: bash reproducir.sh /ruta/modelo.hnf [directorio_resultados_con_tune_cache]' >&2
  exit 2
fi
report_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$report_dir/../.." && pwd)
cuda_dir=${AUDIT_CUDA_ROOT:-/home/andres/opt/cuda13/usr/local/cuda-13.1}
cmake --build "$repo_dir/build" -j 6 --target helios-engine
c++ -O3 -std=c++17 -I "$repo_dir/src" -I "$repo_dir/kernels" \
  -I "$cuda_dir/include" "$report_dir/benchmark.cpp" \
  "$repo_dir/build/libhelios-engine.a" -L "$cuda_dir/lib64" \
  "-Wl,-rpath,$cuda_dir/lib64" -lcudart -lcublas -lcublasLt -lpthread -ldl \
  -o "$repo_dir/build/benchmark_prefill"
python3 "$report_dir/medir.py" "$1" "${2:-$report_dir}"
