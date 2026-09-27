#!/usr/bin/env bash
# K1 measurement: accuracy against float64 and stock GGML, then decode-attention bandwidth of
# stock vs K1 on the v0.8 model geometries. The desktop is quieted for the timed part and the
# 4 worker threads are pinned to the 4 physical cores (unpinned runs were bimodal: two threads
# sharing a core halve its throughput).
# Usage: run.sh <llama.cpp dir with build-cuda> <output dir> [repeats]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; repo="$(git -C "$here" rev-parse --show-toplevel)"
llama="$(realpath "$1")"; out="$(realpath -m "$2")"; repeats="${3:-3}"; bin="$llama/build-cuda/bin"
mkdir -p "$out"; tools="$(mktemp -d)"; trap 'rm -rf "$tools"' EXIT
flags=(-std=c++17 -O3 -march=native -I"$llama/ggml/include" -I"$repo/argus_cache/csrc"
       "$repo/argus_cache/csrc/ggml_decode_attention.cpp" -L"$bin" -lggml -lggml-base -lggml-cpu "-Wl,-rpath,$bin")
g++ "$here/k1_bw.cpp" "${flags[@]}" -o "$tools/k1_bw"
g++ "$repo/tests/cpp/test_ggml_decode_attention.cpp" "${flags[@]}" -o "$tools/test_k1"

{
  date -Iseconds; uname -r
  echo "git $(git -C "$here" rev-parse HEAD) dirty=$(git -C "$here" status --porcelain | wc -l)"
  lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|MHz'
  echo "turbo disabled: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo unknown)"
  echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
  free -m; uptime
} > "$out/machine.txt"

"$tools/test_k1" > "$out/accuracy.txt"
for i in $(seq 1 "$repeats"); do
  "$repo/benchmarks/quiet.sh" taskset -c 0-3 "$tools/k1_bw" 4 10 > "$out/k1_bw_4t_run$i.json" 2>> "$out/machine.txt"
done
uptime >> "$out/machine.txt"
