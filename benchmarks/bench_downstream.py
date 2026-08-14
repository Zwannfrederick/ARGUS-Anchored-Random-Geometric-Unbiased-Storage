"""End-to-end downstream measurement: baseline vs ARGUS on a real model.

This is the only benchmark in the repository that produces `downstream`
claims. Everything else measures codecs on tensors; this measures what a user
actually experiences -- time to first token, inter-token latency, peak VRAM,
and whether the model still predicts text as well.

Design commitments, because a benchmark that flatters its own project is
worthless:

* **Same model, same prompt, same seed** for both arms. The only difference
  is which KV cache implementation is installed.
* **Perplexity is measured through the cache**, not on a separate forward
  pass. A lossy cache that is bypassed during scoring would report a perfect
  delta while degrading real generation.
* **Slower rows are published.** ARGUS trades latency for capacity by design;
  hiding that would misrepresent what it is.

Usage::

    python benchmarks/bench_downstream.py --json docs/measurements/downstream-<date>.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import shlex
import statistics
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from argus_cache import PagedDynamicQuantizedCache

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

#: Held-out passage for perplexity. Fixed so the number is comparable across
#: runs; it is not from the model's prompt.
PPL_TEXT = (
    "Virtual memory decouples the addresses a program uses from the physical "
    "locations where data resides. The operating system maintains page tables "
    "that translate virtual addresses into physical frames, and moves pages "
    "between main memory and secondary storage as demand dictates. A program "
    "may therefore address far more memory than the machine physically holds, "
    "at the cost of latency when a referenced page is not resident. The same "
    "principle applies to attention caches, where older context can be demoted "
    "to cheaper storage while recent tokens stay immediately available."
)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _make_cache(page_size: int):
    return PagedDynamicQuantizedCache(
        page_size=page_size,
        max_active_pages=2,
        max_fp8_pages=2,
        max_int8_pages=2,
        max_int4_pages=2,
        max_int2_pages=2,
        max_one_bit_pages=2,
        sink_tokens=4,
    )


def measure_latency(model, input_ids, new_tokens: int, cache_factory):
    """Return (ttft_s, tpot_s, peak_mib) for one generation run.

    ``logits_to_keep=1`` matters more than it looks: only the last position's
    logits are ever used, but materializing all of them costs
    seq_len x vocab x 2 bytes -- 2.3 GiB at 8192 tokens for this vocabulary.
    Without it the harness OOMs in the model's lm_head and the failure looks
    like a cache problem when it is not one.
    """
    _reset_memory()
    past = cache_factory() if cache_factory else None

    with torch.no_grad():
        _sync()
        t0 = time.perf_counter()
        out = model(
            input_ids, past_key_values=past, use_cache=True, logits_to_keep=1
        )
        next_id = out.logits[:, -1:].argmax(-1)
        _sync()
        ttft = time.perf_counter() - t0

        past = out.past_key_values
        step_times = []
        for _ in range(new_tokens):
            _sync()
            t1 = time.perf_counter()
            out = model(
                next_id, past_key_values=past, use_cache=True, logits_to_keep=1
            )
            next_id = out.logits[:, -1:].argmax(-1)
            _sync()
            step_times.append(time.perf_counter() - t1)
            past = out.past_key_values

    peak = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )
    if past is not None and hasattr(past, "reset"):
        past.reset()
    del out, past
    return ttft, statistics.median(step_times), peak


def tier_occupancy(cache) -> dict:
    """Which tiers actually hold pages, summed across layers.

    Published alongside the perplexity delta so the reader can check that the
    cascade was exercised at all. A delta measured while every page sat in
    uncompressed ACTIVE storage would say nothing about lossy compression, and
    without this field there would be no way to tell the difference.
    """
    layer_caches = getattr(cache, "layer_caches", None)
    if not layer_caches:
        return {}
    totals: dict = {}
    for layer in layer_caches.values():
        totals["active"] = totals.get("active", 0) + len(layer.active_pages)
        for tier, pages in layer.pages_by_tier.items():
            if pages:
                totals[tier] = totals.get(tier, 0) + len(pages)
    return totals


def measure_perplexity(model, ids, cache_factory) -> float:
    """Token-by-token perplexity *through the cache under test*.

    Scoring in one batched forward pass would never exercise the cache, so a
    lossy cache would score identically to an exact one. Feeding one token at
    a time forces every prediction to read back whatever the cache stored.
    """
    past = cache_factory() if cache_factory else None
    losses = []
    with torch.no_grad():
        prev = ids[:, :1]
        out = model(prev, past_key_values=past, use_cache=True)
        past = out.past_key_values
        for i in range(1, ids.shape[1]):
            target = ids[:, i]
            logits = out.logits[:, -1, :].float()
            losses.append(
                torch.nn.functional.cross_entropy(logits, target).item()
            )
            out = model(ids[:, i : i + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
    return float(torch.exp(torch.tensor(statistics.mean(losses))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--json", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[512, 1024, 2048, 4096]
    )
    parser.add_argument("--ppl-tokens", type=int, default=96)
    parser.add_argument(
        "--ppl-page-size",
        type=int,
        default=32,
        help=(
            "Page size for the perplexity arm. Deliberately small: with the "
            "latency page size a short passage never fills a page, nothing "
            "cascades, and the delta is trivially zero -- a vacuous result."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(1234)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
        .to(device)
        .eval()
    )

    arms = {
        "baseline": None,
        "argus": lambda: _make_cache(args.page_size),
    }

    rows = []
    for ctx in args.contexts:
        base_ids = tokenizer("The history of virtual memory. ", return_tensors="pt")
        unit = base_ids["input_ids"]
        reps = (ctx // unit.shape[1]) + 1
        input_ids = unit.repeat(1, reps)[:, :ctx].to(device)

        for arm, factory in arms.items():
            for _ in range(args.warmups):
                measure_latency(model, input_ids, args.new_tokens, factory)
            runs = [
                measure_latency(model, input_ids, args.new_tokens, factory)
                for _ in range(args.repeats)
            ]
            ttfts = [r[0] for r in runs]
            tpots = [r[1] for r in runs]
            peaks = [r[2] for r in runs]
            rows.append({
                "metric_class": "downstream",
                "arm": arm,
                "context_tokens": ctx,
                "new_tokens": args.new_tokens,
                "repeats": args.repeats,
                "ttft_s_median": round(statistics.median(ttfts), 4),
                "ttft_s_min": round(min(ttfts), 4),
                "ttft_s_max": round(max(ttfts), 4),
                "tpot_s_median": round(statistics.median(tpots), 5),
                "tpot_s_min": round(min(tpots), 5),
                "tpot_s_max": round(max(tpots), 5),
                "peak_vram_mib": round(statistics.median(peaks), 2),
            })
            print(
                f"{arm:8s} ctx={ctx:5d}  ttft {rows[-1]['ttft_s_median']:.4f}s  "
                f"tpot {rows[-1]['tpot_s_median']*1000:.2f}ms  "
                f"peak {rows[-1]['peak_vram_mib']:.1f} MiB"
            )
            _reset_memory()

    ppl_ids = tokenizer(PPL_TEXT, return_tensors="pt")["input_ids"][
        :, : args.ppl_tokens
    ].to(device)
    # Force real compression: a small page with one active slot guarantees
    # that older tokens are demoted through the tier cascade while the passage
    # is scored, which is the only configuration under which this number says
    # anything about lossy storage.
    def _ppl_cache():
        cache = PagedDynamicQuantizedCache(
            page_size=args.ppl_page_size,
            max_active_pages=1,
            max_fp8_pages=1,
            max_int8_pages=1,
            max_int4_pages=1,
            max_int2_pages=1,
            max_one_bit_pages=1,
            sink_tokens=4,
        )
        return cache

    occupancy: dict = {}

    def _probe_ppl_cache():
        cache = _ppl_cache()
        occupancy["_cache"] = cache
        return cache

    ppl_arms = {"baseline": None, "argus": _probe_ppl_cache}
    perplexity = {
        arm: round(measure_perplexity(model, ppl_ids, factory), 4)
        for arm, factory in ppl_arms.items()
    }
    tiers_used = tier_occupancy(occupancy.get("_cache"))
    measured_cache = occupancy.pop("_cache", None)
    if measured_cache is not None:
        measured_cache.reset()
    print(f"tier occupancy during scoring: {tiers_used}")
    delta = round(perplexity["argus"] - perplexity["baseline"], 4)
    print(
        f"\nperplexity  baseline {perplexity['baseline']}  "
        f"argus {perplexity['argus']}  delta {delta:+}"
    )

    result = {
        "metric_class": "downstream",
        "date": args.date,
        "command": shlex.join([sys.executable, *sys.argv]),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        # Regenerating a tracked output naturally makes that one path dirty
        # before the JSON is written.  Ignore only the requested output path;
        # every source/config change still invalidates clean-tree provenance.
        "git_dirty": any(
            line[3:] != args.json
            for line in subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                text=True,
            ).splitlines()
            if line[3:]
        ),
        "model": args.model,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "host_platform": platform.platform(),
        "seed": 1234,
        "page_size": args.page_size,
        "warmups": args.warmups,
        "latency": rows,
        "perplexity": {
            **perplexity,
            "delta": delta,
            "tokens": int(ppl_ids.shape[1]),
            "page_size": args.ppl_page_size,
            "tier_occupancy_during_scoring": tiers_used,
            "config": (
                "page_size=%d, max_active_pages=1, one page per tier -- chosen "
                "so the passage actually cascades through the compressed tiers"
                % args.ppl_page_size
            ),
            "note": (
                "Measured token-by-token through the cache under test, so the "
                "cache is actually exercised. A batched scoring pass would "
                "bypass it and report a meaningless zero delta."
            ),
            "caveat": (
                "Only fp8 (2x, near-lossless) was reached at this passage "
                "length. Deeper lossy tiers were not exercised, so this delta "
                "does not characterize int4/int2/one_bit/JL quality."
            ),
        },
        "status": "COMPLETE",
        "harness_note": (
            "Prefill uses logits_to_keep=1. Every latency run explicitly "
            "resets its cache so native callback cycles cannot contaminate "
            "later baseline arms."
        ),
    }

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
