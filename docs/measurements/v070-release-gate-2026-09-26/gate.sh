#!/usr/bin/env bash
set -u
S="$1"
cd "/home/zwannfrederick/Masaüstü/Sektor/Coding/mamba fix"
MCP='^/home/zwannfrederick/.local/bin/codebase-memory-mcp'
resume() { pkill -CONT -f "$MCP"; echo "resumed: $(pgrep -f "$MCP" | wc -l) mcp processes" >> "$S/load.txt"; }
trap resume EXIT
pkill -STOP -f "$MCP"
echo "stopped: $(pgrep -f "$MCP" | tr '\n' ' ')" > "$S/load.txt"
sleep 5
note() { echo "$1 load $(cut -d' ' -f1-3 /proc/loadavg) io $(head -1 /proc/pressure/io | cut -d' ' -f2-3)" >> "$S/load.txt"; }
B=(.venv/bin/python benchmarks/bench_llama_paged_context.py --server scratch/llama.cpp-v050/build-cuda/bin/llama-server
   --model scratch/models/qwen2.5-0.5b-instruct-q4_k_m.gguf --kv-dir scratch/kv --contexts 4096 --resident-bytes 4194304
   --gpu-bytes 67108864 --pinned-bytes 67108864 --request-timeout 900 --ubatch 64 --kv-type f16 --predict 16 --warmups 1)
note start
"${B[@]}" --modes stock-host-kv argus-cuda-control --repeats 5 --output "$S/final-baseline.json" > "$S/baseline.log" 2>&1
note mid
"${B[@]}" --modes argus-cuda-on --repeats 3 --output "$S/policy-on.json" > "$S/policy.log" 2>&1
note end
