#!/usr/bin/env bash
# usage: ab.sh <dir> <mode> <pairs> <libA> <libB> [extra bench args]
set -u
D="$1"; MODE="$2"; N="$3"; A="$4"; B="$5"; shift 5
S=/tmp/claude-1000/-home-zwannfrederick-Masa-st--Sektor-Coding-mamba-fix/ef61ce56-069e-44aa-bf30-52a50e4d5020/scratchpad
cd "/home/zwannfrederick/Masaüstü/Sektor/Coding/mamba fix"
mkdir -p "$D"
MCP='^/home/zwannfrederick/.local/bin/codebase-memory-mcp'
trap 'pkill -CONT -f "$MCP"' EXIT
pkill -STOP -f "$MCP"; sleep 2
echo "start load $(cut -d' ' -f1 /proc/loadavg) io $(head -1 /proc/pressure/io | cut -d' ' -f2)" > "$D/load.txt"
for i in $(seq 1 "$N"); do
  if [ $((i % 2)) = 1 ]; then order="$A $B"; else order="$B $A"; fi
  for arm in $order; do
    LD_LIBRARY_PATH="$S/lib-$arm" .venv/bin/python benchmarks/bench_llama_paged_context.py \
      --server scratch/llama.cpp-v050/build-cuda/bin/llama-server --model scratch/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
      --kv-dir scratch/kv --contexts 4096 --resident-bytes 4194304 --gpu-bytes 67108864 --pinned-bytes 67108864 \
      --request-timeout 900 --ubatch 64 --kv-type f16 --predict 16 --warmups 1 --repeats 1 --modes "$MODE" \
      --output "$D/$i-$arm.json" "$@" > "$D/$i-$arm.log" 2>&1 || echo "fail $i $arm" >> "$D/load.txt"
  done
done
echo "end load $(cut -d' ' -f1 /proc/loadavg) io $(head -1 /proc/pressure/io | cut -d' ' -f2)" >> "$D/load.txt"
