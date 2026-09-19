#!/usr/bin/env bash
# Instrumentación aislada. No sustituye ni recompila los ejecutables de producción.
set -euo pipefail
audit_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$audit_dir/../.." && pwd)
cuda_dir=${AUDIT_CUDA_ROOT:-/home/andres/opt/cuda13/usr/local/cuda-13.1}
audit_work=$(mktemp -d /tmp/hector-audit-repro.XXXXXX)
mode=${1:-probes}
model=${2:-}
flags=(-O3 -std=c++17 -I "$repo_dir/src" -I "$repo_dir/kernels" -I "$cuda_dir/include")
libs=("$repo_dir/build/libhelios-engine.a" -L "$cuda_dir/lib64" "-Wl,-rpath,$cuda_dir/lib64" -lcudart -lcublas -lcublasLt -lpthread -ldl)
export HELIOS_HOME="$audit_work"
case "$mode" in
  probes|inventory)
    c++ "${flags[@]}" "$audit_dir/$mode.cpp" "${libs[@]}" -o "$audit_work/probe"
    "$audit_work/probe" "$model"
    ;;
  session|graph)
    objects=()
    if [[ "$mode" == graph ]]; then
      cp "$repo_dir/src/inference_session.cpp" "$audit_work/inference_session.cpp"
      patch --batch --forward "$audit_work/inference_session.cpp" "$audit_dir/experimento-grafos.patch"
      c++ "${flags[@]}" -c "$audit_work/inference_session.cpp" -o "$audit_work/session.o"
      objects+=("$audit_work/session.o")
    fi
    c++ "${flags[@]}" "$audit_dir/session_bench.cpp" "${objects[@]}" "${libs[@]}" -o "$audit_work/session_bench"
    "$audit_work/session_bench" "$model"
    ;;
  *) echo 'Uso: reproducir.sh probes|inventory|session|graph [modelo.hnf]' >&2; exit 2 ;;
esac
echo "Artefactos temporales: $audit_work"
