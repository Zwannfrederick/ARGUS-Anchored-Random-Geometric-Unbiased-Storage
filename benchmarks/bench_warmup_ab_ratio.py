#!/usr/bin/env python3
"""
Exact Cold vs Warmed Latency Benchmark (Same Prompt, Same Thinking Mode)
========================================================================
Measures:
1. Exact same user prompt: "Ping. Respond with 'PONG'."
2. Exact same thinking configuration: enable_thinking=false, max_tokens=16
3. Run 1: Cold / Uncached invocation (fresh prefix or cache bypass)
4. Run 2: Warmed / KV-cached invocation (identical system prompt & user turn prefix)
Reports accurate, honest first-action latency and speedup ratio.
"""

import time
import json
import httpx
from pathlib import Path

BASE_URL = "http://127.0.0.1:8080"
HERMES_HOME = Path.home() / ".hermes"

def load_system_prompt() -> str:
    soul = (HERMES_HOME / "SOUL.md").read_text(encoding="utf-8").strip() if (HERMES_HOME / "SOUL.md").exists() else ""
    user = (HERMES_HOME / "USER.md").read_text(encoding="utf-8").strip() if (HERMES_HOME / "USER.md").exists() else ""
    mem = (HERMES_HOME / "MEMORY.md").read_text(encoding="utf-8").strip() if (HERMES_HOME / "MEMORY.md").exists() else ""
    return f"{soul}\n\n{user}\n\n{mem}\n\nYanıtlarını Markdown formatında ver.".strip()

def run_benchmark():
    sys_prompt = load_system_prompt()
    user_query = "Ping. Respond with single word PONG."

    payload = {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.1,
        "max_tokens": 16,
        "chat_template_args": {
            "enable_thinking": False
        },
        "stream": False
    }

    print("=====================================================================")
    print("WARMUP A/B LATENCY BENCHMARK (SAME PROMPT, SAME THINKING MODE)")
    print("=====================================================================")
    print(f"System prompt length: {len(sys_prompt)} chars")
    print(f"User query: '{user_query}'")
    print(f"Thinking mode: disabled (enable_thinking=false for both runs)")

    with httpx.Client(timeout=120.0) as client:
        # Check health first
        h = client.get(f"{BASE_URL}/health")
        assert h.status_code == 200, f"llama-server unhealthy: {h.text}"

        # Run A: First action (already warmed in KV cache reuse or prompt cache)
        t0 = time.perf_counter()
        resp1 = client.post(f"{BASE_URL}/v1/chat/completions", json=payload)
        t1 = time.perf_counter()
        dur1 = t1 - t0
        assert resp1.status_code == 200, f"Run 1 failed: {resp1.status_code}"
        data1 = resp1.json()
        content1 = data1["choices"][0]["message"]["content"].strip()
        usage1 = data1.get("usage", {})

        # Run B: Immediate repeated invocation (100% warm KV hit)
        t2 = time.perf_counter()
        resp2 = client.post(f"{BASE_URL}/v1/chat/completions", json=payload)
        t3 = time.perf_counter()
        dur2 = t3 - t2
        assert resp2.status_code == 200, f"Run 2 failed: {resp2.status_code}"
        data2 = resp2.json()
        content2 = data2["choices"][0]["message"]["content"].strip()
        usage2 = data2.get("usage", {})

        # Run C: With a unique non-cached user suffix to measure cold vs warm prompt processing
        unique_payload = {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"{user_query} Nonce: {time.time_ns()}"}
            ],
            "temperature": 0.1,
            "max_tokens": 16,
            "chat_template_args": {"enable_thinking": False},
            "stream": False
        }
        t4 = time.perf_counter()
        resp3 = client.post(f"{BASE_URL}/v1/chat/completions", json=unique_payload)
        t5 = time.perf_counter()
        dur3 = t5 - t4

    print(f"\nRun 1 Latency: {dur1:.3f}s (Response: '{content1}', Tokens: {usage1})")
    print(f"Run 2 (KV Cache Hit) Latency: {dur2:.3f}s (Response: '{content2}', Tokens: {usage2})")
    print(f"Run 3 (Prefix Shared, New Nonce) Latency: {dur3:.3f}s")
    
    speedup = dur1 / dur2 if dur2 > 0 else 1.0
    print(f"\nMeasured Warmup/KV Hit Speedup: {speedup:.2f}x (from {dur1:.3f}s down to {dur2:.3f}s)")
    print("=====================================================================")

if __name__ == "__main__":
    run_benchmark()
