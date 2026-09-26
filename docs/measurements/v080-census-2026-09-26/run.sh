#!/usr/bin/env bash
# Bandwidth census for v0.8. Builds the three probes, pauses background re-indexers for the
# duration (they spoiled earlier runs), and records the machine state next to the results.
# Usage: run.sh <llama.cpp dir with build-cuda> <output dir>
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
llama="$(realpath "$1")"; out="$(realpath "$2")"; bin="$llama/build-cuda/bin"
mkdir -p "$out"; tools="$(mktemp -d)"  # build output stays out of the results
nvcc -O3 -arch=sm_86 "$here/pcie.cu" -o "$tools/pcie"
g++ -O3 -march=native -pthread "$here/ram_bw.cpp" -o "$tools/ram_bw"
g++ -O3 -march=native -std=c++17 "$here/attn_bw.cpp" -I"$llama/ggml/include" -L"$bin" \
    -lggml -lggml-base -lggml-cpu -lggml-cuda "-Wl,-rpath,$bin" -o "$tools/attn_bw"

pkill -STOP -f '^/home/zwannfrederick/.local/bin/codebase-memory-mcp' || true
trap "pkill -CONT -f '^/home/zwannfrederick/.local/bin/codebase-memory-mcp' || true; rm -rf '$tools'" EXIT

{
  date -Iseconds; uname -r
  echo "git $(git -C "$here" rev-parse HEAD) dirty=$(git -C "$here" status --porcelain | wc -l)"
  lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core'
  free -m; uptime
  nvidia-smi --query-gpu=name,driver_version,memory.used,pcie.link.gen.max,pcie.link.width.max --format=csv
  # Process names only: full command lines can carry per-user identifiers.
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | awk -F', ' '{n=$2; sub(/ .*/, "", n); print $1", "n", "$3}'
} > "$out/machine.txt"

"$tools/pcie" > "$out/pcie.json" &
pid=$!; sleep 1
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv >> "$out/machine.txt"
wait $pid
"$tools/ram_bw" > "$out/ram_bw.json"
"$tools/attn_bw" cpu 4 > "$out/attn_cpu_4t.json"
"$tools/attn_bw" cpu 8 > "$out/attn_cpu_8t.json"
"$tools/attn_bw" cuda > "$out/attn_cuda.json"
uptime >> "$out/machine.txt"
