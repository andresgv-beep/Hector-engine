#!/usr/bin/env bash
set -euo pipefail

audit_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$audit_dir/../.." && pwd)
audit_mode=${1:-}
audit_model=${2:-}
case "$audit_mode" in
    sampling) ;;
    metadata|cancel|latency|profile-prefill|profile-decode)
        if [[ ! -f "$audit_model" ]]; then
            echo 'Indica la ruta de un modelo HNF existente como segundo argumento.' >&2
            exit 2
        fi
        audit_model=$(realpath -- "$audit_model")
        ;;
    *) echo 'Uso: bash run.sh sampling|metadata|cancel|latency|profile-prefill|profile-decode [modelo.hnf]' >&2; exit 2 ;;
esac

audit_cuda=${AUDIT_CUDA_ROOT:-/home/andres/opt/cuda13/usr/local/cuda-13.1}
audit_nsys=${AUDIT_NSYS:-/usr/local/cuda-12.6/bin/nsys}
if [[ ! -f "$repo_dir/build/libhelios-engine.a" ]]; then
    echo 'Falta build/libhelios-engine.a; compila primero la revisión que quieras medir.' >&2
    exit 2
fi
audit_out=$(mktemp -d /tmp/hector-architecture-audit.XXXXXX)
cp -- "$audit_dir/tune.cache" "$audit_out/tune.cache"
export HELIOS_HOME="$audit_out"
echo "Resultados: $audit_out"
git -C "$repo_dir" rev-parse HEAD > "$audit_out/revision.txt"
sha256sum "$repo_dir/build/libhelios-engine.a" "$audit_dir/probe.cpp" > "$audit_out/inputs.sha256"
if [[ -n "$audit_model" ]]; then
    sha256sum -- "$audit_model" >> "$audit_out/inputs.sha256"
fi
"${CXX:-c++}" -O3 -std=c++17 -I "$repo_dir/src" -I "$repo_dir/kernels" \
    -I "$audit_cuda/include" "$audit_dir/probe.cpp" \
    "$repo_dir/build/libhelios-engine.a" -L "$audit_cuda/lib64" \
    -Wl,-rpath,"$audit_cuda/lib64" -lcudart -lcublas -lcublasLt -lpthread -ldl \
    -o "$audit_out/probe"

if [[ "$audit_mode" == sampling ]]; then
    "$audit_out/probe" sampling 2>&1 | tee "$audit_out/sampling.log"
elif [[ "$audit_mode" == profile-* ]]; then
    "$audit_nsys" profile --trace=cuda --sample=none --cpuctxsw=none \
        --cuda-graph-trace=node --capture-range=cudaProfilerApi \
        --capture-range-end=stop --force-overwrite=true \
        -o "$audit_out/$audit_mode" "$audit_out/probe" "$audit_model" "$audit_mode" \
        2>&1 | tee "$audit_out/$audit_mode.log"
    "$audit_nsys" stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
        --force-export=true "$audit_out/$audit_mode.nsys-rep" > "$audit_out/$audit_mode.csv"
else
    "$audit_out/probe" "$audit_model" "$audit_mode" 2>&1 | tee "$audit_out/$audit_mode.log"
fi
