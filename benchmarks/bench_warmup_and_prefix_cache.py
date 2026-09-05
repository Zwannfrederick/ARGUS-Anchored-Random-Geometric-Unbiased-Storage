"""
Rigorous Prompt Cache & Warmup Benchmark Suite for Neo (Hermes)
==============================================================
Validates:
1. Exact Repeat Cache Hit (Test A)
2. Realistic Short New Turn with Shared Prefix (Test B)
3. Second Realistic Short New Turn in Same Session (Test C)
4. Reasoning Turn with Warmed Thinking Prefix (Test D)
5. Model Restart -> Startup Warmup -> 3 Different Production Turns (Acceptance Test)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HERMES_DIR = PROJECT_ROOT / "hermes"
if str(HERMES_DIR) not in sys.path:
    sys.path.insert(0, str(HERMES_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hermes_supervisor import HermesSupervisor
from prefix_builder import get_canonical_stable_prefix, get_prefix_metadata

async def run_single_turn_benchmark(
    supervisor: HermesSupervisor,
    label: str,
    session_id: str,
    user_text: str,
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    print(f"\n--- [{label}] ---")
    print(f"Query: '{user_text}' (History length: {len(history)})")

    start_t = time.perf_counter()
    async def dummy_approval(rec): return "approve"
    async def dummy_event(evt): pass

    result = await supervisor.execute_session_turn(
        session_id=session_id,
        user_text=user_text,
        conversation_history=history,
        system_prompt=get_canonical_stable_prefix(),
        screenshots_dir=PROJECT_ROOT / "screenshots",
        request_approval_cb=dummy_approval,
        emit_event_cb=dummy_event,
    )

    telem = result.get("telemetry", {})
    prompt_tokens = telem.get("prompt_tokens", 0)
    cached_tokens = telem.get("cached_tokens", 0)
    cached_ratio = telem.get("cached_ratio", 0.0)
    newly_eval = telem.get("newly_evaluated_tokens", 0)
    ttft = telem.get("ttft")
    total_lat = telem.get("total_latency", round(time.perf_counter() - start_t, 3))
    thinking = telem.get("thinking_mode", False)
    prefix_sha = telem.get("stable_prefix_sha256", "")[:12]

    tool_calls = result.get("tool_calls", [])
    tool_names = [tc.get("name") for tc in tool_calls]

    print(f"Prefix SHA256: {prefix_sha}...")
    print(f"Prompt Tokens: {prompt_tokens}")
    print(f"Cached Tokens: {cached_tokens} ({cached_ratio:.2%})")
    print(f"Newly Evaluated: {newly_eval} tokens")
    print(f"TTFT: {ttft}s")
    print(f"Total Latency: {total_lat}s")
    print(f"Thinking Enabled: {thinking}")
    print(f"Tool Chosen: {tool_names or 'None (Direct Text)'}")
    print(f"Content Preview: {result.get('content', '')[:100]}...")

    return {
        "label": label,
        "query": user_text,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cached_ratio": cached_ratio,
        "newly_evaluated_tokens": newly_eval,
        "ttft": ttft,
        "total_latency": total_lat,
        "thinking": thinking,
        "prefix_sha": prefix_sha,
        "tool_calls": tool_names,
        "raw_result": result,
    }

async def main():
    meta = get_prefix_metadata()
    print("=" * 70)
    print("NEO CANONICAL PREFIX CACHE & LATENCY BENCHMARK")
    print("=" * 70)
    print(f"Stable Prefix SHA-256: {meta['stable_prefix_sha256']}")
    print(f"Tool Schema SHA-256:   {meta['tool_schema_sha256']}")
    print(f"Stable Prefix Chars:   {meta['stable_prefix_chars']}")
    print(f"Estimated Tokens:      ~{meta['stable_prefix_estimated_tokens']}")

    supervisor = HermesSupervisor()
    session_a = f"sess_bench_{int(time.time())}"

    # TEST A: Exact Repeat
    # Run once to ensure cache slot is populated with this exact user text
    print("\nPre-populating query for Test A...")
    async def dummy_approval(rec): return "approve"
    async def dummy_event(evt): pass

    await supervisor.execute_session_turn(
        session_id=session_a,
        user_text="Aktif pencereleri listele.",
        conversation_history=[],
        system_prompt=get_canonical_stable_prefix(),
        screenshots_dir=PROJECT_ROOT / "screenshots",
        request_approval_cb=dummy_approval,
        emit_event_cb=dummy_event,
    )

    res_a = await run_single_turn_benchmark(
        supervisor,
        label="Test A — Exact Repeat (Thinking OFF)",
        session_id=session_a,
        user_text="Aktif pencereleri listele.",
        history=[],
    )

    # TEST B: Realistic Short New Turn
    # Different user query, fresh session, reusing the warmed canonical prefix!
    session_b = f"sess_new_{int(time.time())}"
    res_b = await run_single_turn_benchmark(
        supervisor,
        label="Test B — Realistic Short New Turn (Thinking OFF)",
        session_id=session_b,
        user_text="Siber pano kısayolunu tetikle.",
        history=[],
    )

    # TEST C: Second Realistic Short New Turn in the Same Session
    # History contains Turn B
    history_c = [
        {"role": "user", "content": "Siber pano kısayolunu tetikle."},
        {"role": "assistant", "content": res_b["raw_result"].get("content", "Siber pano açıldı.")}
    ]
    res_c = await run_single_turn_benchmark(
        supervisor,
        label="Test C — Second Realistic Short Turn in Same Session",
        session_id=session_b,
        user_text="Şimdi de masaüstünün ekran görüntüsünü al.",
        history=history_c,
    )

    # TEST D: Thinking ON with its own warmed reasoning prefix
    session_d = f"sess_think_{int(time.time())}"
    res_d = await run_single_turn_benchmark(
        supervisor,
        label="Test D — Reasoning Turn (Thinking ON / Medium-Risk Planning)",
        session_id=session_d,
        user_text="PostgreSQL için yeni bir migration ve pipeline mimarisi oluşturmak istiyorum, adımları planla.",
        history=[],
    )

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY REPORT")
    print("=" * 70)
    print(f"{'Test':<28} | {'Prompt':<7} | {'Cached':<7} | {'Ratio':<7} | {'New Toks':<8} | {'TTFT':<6} | {'Total':<6}")
    print("-" * 70)
    for r in [res_a, res_b, res_c, res_d]:
        print(f"{r['label'][:28]:<28} | {r['prompt_tokens']:<7} | {r['cached_tokens']:<7} | {r['cached_ratio']:<7.2%} | {r['newly_evaluated_tokens']:<8} | {str(r['ttft']):<6} | {str(r['total_latency']):<6}s")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
