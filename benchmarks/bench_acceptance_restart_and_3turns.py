"""
Acceptance Test Script:
1. Restart neo-model.service and neo-gateway.service
2. Wait for startup warmup to complete (fast path + reasoning path)
3. Send three DIFFERENT short production-style Neo requests through the normal prompt builder
4. Prove stable prefix SHA-256 stays identical
5. Record cached_tokens / prompt_tokens and TTFT for each request
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HERMES_DIR = PROJECT_ROOT / "hermes"
if str(HERMES_DIR) not in sys.path:
    sys.path.insert(0, str(HERMES_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hermes_supervisor import HermesSupervisor
from prefix_builder import get_canonical_stable_prefix, get_prefix_metadata

GATEWAY_URL = "http://127.0.0.1:8765"
MODEL_URL = "http://127.0.0.1:8080"


async def wait_for_model_and_gateway(timeout_s: float = 600.0):
    print("\n[STEP 1] Waiting for llama-server (/health) to become ready...")
    t0 = time.time()
    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.time() - t0 < timeout_s:
            try:
                r = await client.get(f"{MODEL_URL}/health")
                if r.status_code == 200:
                    print(f"llama-server is healthy ({r.json().get('status', 'ok')}).")
                    break
            except Exception:
                pass
            await asyncio.sleep(2.0)
        else:
            raise TimeoutError("llama-server failed to become ready within timeout.")

    print("\n[STEP 2] Waiting for neo-gateway warmup to complete (both Reasoning and Fast paths)...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.time() - t0 < timeout_s:
            try:
                r = await client.get(f"{GATEWAY_URL}/health")
                if r.status_code == 200:
                    data = r.json()
                    warmed = data.get("model_warmed", False)
                    ready = data.get("ready", False)
                    warmup_info = data.get("warmup", {})
                    if ready and warmed and not warmup_info.get("in_progress", False):
                        print(f"Gateway warmup complete! cold={warmup_info.get('cold_action_latency_s')}s, warmed={warmup_info.get('warmed_action_latency_s')}s")
                        return data
                    print(f"Waiting for warmup... (warmed={warmed}, in_progress={warmup_info.get('in_progress')})")
            except Exception as e:
                print(f"Gateway query error: {e}")
            await asyncio.sleep(5.0)
        raise TimeoutError("Gateway warmup did not complete within timeout.")


async def run_production_turn(
    supervisor: HermesSupervisor,
    turn_num: int,
    user_text: str,
    session_id: str,
) -> Dict[str, Any]:
    print(f"\n" + "=" * 60)
    print(f"PRODUCTION TURN {turn_num}: \"{user_text}\"")
    print("=" * 60)

    start_t = time.perf_counter()
    async def dummy_approval(rec): return "approve"
    async def dummy_event(evt): pass

    result = await supervisor.execute_session_turn(
        session_id=session_id,
        user_text=user_text,
        conversation_history=[],
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
    prefix_sha = telem.get("stable_prefix_sha256", "")
    thinking = telem.get("thinking_mode", False)

    tool_calls = result.get("tool_calls", [])
    tool_names = [tc.get("name") for tc in tool_calls]

    print(f"Stable Prefix SHA-256: {prefix_sha}")
    print(f"Prompt Tokens:        {prompt_tokens}")
    print(f"Cached Tokens:        {cached_tokens} ({cached_ratio:.2%})")
    print(f"Newly Evaluated:      {newly_eval} tokens")
    print(f"TTFT:                 {ttft}s")
    print(f"Total Latency:        {total_lat}s")
    print(f"Thinking Mode:        {thinking}")
    print(f"Tool Dispatched:      {tool_names or 'Direct Response'}")
    print(f"Response Preview:     {result.get('content', '')[:120]}...")

    return {
        "turn": turn_num,
        "query": user_text,
        "prefix_sha": prefix_sha,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cached_ratio": cached_ratio,
        "newly_eval": newly_eval,
        "ttft": ttft,
        "total_latency": total_lat,
        "thinking": thinking,
        "tool_names": tool_names,
        "content": result.get("content", ""),
    }


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Neo Acceptance Test")
    parser.add_argument("--skip-restart", action="store_true", help="Skip restart if services are already freshly restarted and warmed")
    args = parser.parse_args()

    meta = get_prefix_metadata()
    print("=" * 70)
    print("NEO ACCEPTANCE TEST: MODEL RESTART -> WARMUP -> 3 PRODUCTION TURNS")
    print("=" * 70)
    print(f"Expected Canonical SHA-256: {meta['stable_prefix_sha256']}")
    print(f"Canonical Tool Schema Hash: {meta['tool_schema_sha256']}")
    print(f"Canonical Prefix Chars:     {meta['stable_prefix_chars']}")

    if not args.skip_restart:
        # Step 0: Restart services
        print("\n[STEP 0] Restarting neo-model.service and neo-gateway.service...")
        subprocess.run(["systemctl", "--user", "restart", "neo-model.service"], check=True)
        time.sleep(2.0)
        subprocess.run(["systemctl", "--user", "restart", "neo-gateway.service"], check=True)

    # Step 1 & 2: Wait for warmup
    health_data = await wait_for_model_and_gateway(timeout_s=600.0)

    # Step 3: Run 3 DIFFERENT short production-style Neo requests
    supervisor = HermesSupervisor()
    session_id = f"sess_acceptance_{int(time.time())}"

    test_queries = [
        "Aktif pencereleri listele.",
        "Siber pano kısayolunu tetikle.",
        "Masaüstünün ekran görüntüsünü al.",
    ]

    results = []
    for i, query in enumerate(test_queries, 1):
        res = await run_production_turn(supervisor, i, query, session_id)
        results.append(res)

    # Step 4: Verification and Summary Report
    print("\n" + "=" * 75)
    print("ACCEPTANCE TEST SUMMARY: 3 DIFFERENT SHORT PRODUCTION TURNS")
    print("=" * 75)
    print(f"{'Turn':<6} | {'Prompt':<7} | {'Cached':<7} | {'Ratio':<8} | {'New Toks':<8} | {'TTFT':<7} | {'Total':<7} | {'SHA-256 Match'}")
    print("-" * 75)

    all_sha_match = True
    all_cached_high = True
    for r in results:
        sha_match = (r["prefix_sha"] == meta["stable_prefix_sha256"])
        if not sha_match:
            all_sha_match = False
        if r["cached_ratio"] < 0.90:
            all_cached_high = False

        print(
            f"Turn {r['turn']:<1} | {r['prompt_tokens']:<7} | {r['cached_tokens']:<7} | "
            f"{r['cached_ratio']:<8.2%} | {r['newly_eval']:<8} | {str(r['ttft']):<7} | "
            f"{str(r['total_latency']):<7}s | {'PASS' if sha_match else 'FAIL'}"
        )
    print("=" * 75)

    print(f"\nFinal Acceptance Verdict:")
    print(f"- Stable Prefix SHA Identical Across All Turns: {'PASS' if all_sha_match else 'FAIL'}")
    print(f"- >= 90% Prompt Tokens Reused From Cache:      {'PASS' if all_cached_high else 'FAIL'}")


if __name__ == "__main__":
    asyncio.run(main())
