"""Measure the JL archival tier on REAL KV activations, not random tensors.

Random-tensor fidelity is meaningless for this tier. ARGUS's JL operator is
not a generic rank reducer: it is a *smoothness*-regularized least-squares
inverse (a 1-D Laplacian prior along the sequence axis). So its accuracy
depends on whether real KV activations vary smoothly from token to token, and
a poor score on white noise reflects the input, not the codec.

Note that the prior is smoothness and **not** low rank. Reconstructing a
rank-4 random signal through this operator is no better than reconstructing
white noise -- measured, not assumed:

    low-rank rel. error 1.106   vs   white-noise rel. error 1.111

Effective rank is still reported below, because it is informative about the
activations, but it is not the quantity the operator exploits.

The comparison that decides the tier's fate is against an **equal storage
budget**: int2 is also 4x smaller than fp16. A compressor only earns its place
if it beats the simpler thing that costs the same.

Claim class: `reconstruction`. This is synthetic fidelity on real tensors --
it is NOT downstream model quality. A tier can win here and still hurt
perplexity.

Usage::

    python benchmarks/bench_jl_fidelity.py --json docs/measurements/jl-<date>.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from argus_cache.backends.quantization import INT2Backend
from argus_cache.core.jl_operators import JLOperatorCache

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def effective_rank(x: torch.Tensor, energy: float = 0.99) -> int:
    """Singular values needed to retain `energy` of the spectrum."""
    sv = torch.linalg.svdvals(x.float())
    cumulative = torch.cumsum(sv**2, 0) / (sv**2).sum()
    return int((cumulative < energy).sum().item()) + 1


def roughness(x: torch.Tensor) -> float:
    """Relative energy of the token-to-token difference along the sequence axis.

    This is the quantity the Laplacian prior actually acts on. Near 0 means a
    smooth sequence the operator can reconstruct; values around sqrt(2) mean
    successive tokens are uncorrelated, which is the white-noise case.
    """
    diff = x[1:] - x[:-1]
    return (diff.norm() / x[:-1].norm()).item()


def capture_kv(model, tokenizer, prompt: str, device: str):
    """Run one forward pass and return per-layer key tensors."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    cache = out.past_key_values
    # transformers >=4.4x returns a Cache object; older returns tuples.
    if hasattr(cache, "layers"):
        return [layer.keys for layer in cache.layers]
    return [layer[0] for layer in cache]


def int2_naive_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Weak equal-budget control: 2-bit affine with one scale for the tensor."""
    vmin, vmax = x.min(), x.max()
    scale = (vmax - vmin) / 3.0 if vmax > vmin else torch.tensor(1.0)
    q = ((x - vmin) / scale).round().clamp(0, 3)
    return q * scale + vmin


def int2_shipped_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Strong equal-budget control: the int2 backend ARGUS actually ships.

    Comparing JL only against a single-scale quantizer would be a strawman --
    the shipped backend uses per-group scales and zero-points and is a much
    harder target. If JL cannot beat *this*, it is not earning its slot.
    """
    backend = INT2Backend()
    packed = backend.compress(x, seq_dim=-2)
    return backend.decompress(packed, seq_dim=-2).float()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--json", default=None)
    parser.add_argument("--date", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
        .to(device)
        .eval()
    )

    prompt = "The history of virtual memory management. " * (args.tokens // 8)
    keys = capture_kv(model, tokenizer, prompt, device)

    # Measure per PAGE, not per full sequence: in production the JL tier
    # compresses one page at a time, so a whole-sequence projection would be
    # measuring an operator the cache never actually applies.
    page = args.page_size
    jl_cache = JLOperatorCache(page_size=page, ratio=args.ratio)
    w = jl_cache.projection(torch.device("cpu"), torch.float32, seq_len=page)
    recon_op = jl_cache.reconstruction(torch.device("cpu"), torch.float32, seq_len=page)

    rows = []
    for idx, k in enumerate(keys):
        # [batch, heads, seq, head_dim] -> project along the sequence axis
        mat = k[0, 0].float().cpu()
        num_pages = mat.shape[0] // page
        if num_pages == 0:
            continue

        jl_errs, jl_coss, int2_errs, int2s_errs, roughs, ranks = [], [], [], [], [], []
        for p in range(num_pages):
            chunk = mat[p * page : (p + 1) * page]
            restored = recon_op @ (w @ chunk)

            jl_errs.append(((restored - chunk).norm() / chunk.norm()).item())
            jl_coss.append(
                torch.nn.functional.cosine_similarity(
                    restored.flatten(), chunk.flatten(), dim=0
                ).item()
            )
            int2_errs.append(
                ((int2_naive_roundtrip(chunk) - chunk).norm() / chunk.norm()).item()
            )
            int2s_errs.append(
                ((int2_shipped_roundtrip(chunk) - chunk).norm() / chunk.norm()).item()
            )
            roughs.append(roughness(chunk))
            ranks.append(effective_rank(chunk))

        jl_err = statistics.mean(jl_errs)
        int2_err = statistics.mean(int2_errs)
        int2s_err = statistics.mean(int2s_errs)
        rows.append({
            "layer": idx,
            "pages": num_pages,
            "page_size": page,
            "effective_rank_99": round(statistics.mean(ranks), 1),
            "full_rank": min(page, mat.shape[1]),
            "roughness": round(statistics.mean(roughs), 4),
            "jl_relative_error": round(jl_err, 4),
            "jl_cosine": round(statistics.mean(jl_coss), 4),
            "int2_naive_relative_error": round(int2_err, 4),
            "int2_shipped_relative_error": round(int2s_err, 4),
            "jl_beats_int2_naive": bool(jl_err < int2_err),
            "jl_beats_int2_shipped": bool(jl_err < int2s_err),
        })

    if not rows:
        print("Prompt produced fewer than one full page; raise --tokens.")
        return 1

    wins = sum(r["jl_beats_int2_naive"] for r in rows)
    wins_shipped = sum(r["jl_beats_int2_shipped"] for r in rows)
    result = {
        "metric_class": "reconstruction",
        "note": (
            "Synthetic reconstruction on real activations. NOT model quality. "
            "The JL operator's prior is smoothness along the sequence axis, "
            "not low rank; 'roughness' is the quantity it exploits."
        ),
        "date": args.date,
        "command": (
            f"python benchmarks/bench_jl_fidelity.py --model {args.model} "
            f"--tokens {args.tokens} --ratio {args.ratio} --page-size {args.page_size}"
        ),
        "model": args.model,
        "ratio": args.ratio,
        "page_size": args.page_size,
        "controls": {
            "int2_naive": "single-scale 2-bit affine round trip (equal 4x budget)",
            "int2_shipped": "argus_cache INT2Backend, per-group scales (equal 4x budget)",
        },
        "device": device,
        "torch": torch.__version__,
        "host_platform": platform.platform(),
        "layers": rows,
        "median_jl_relative_error": round(
            statistics.median(r["jl_relative_error"] for r in rows), 4
        ),
        "median_int2_naive_relative_error": round(
            statistics.median(r["int2_naive_relative_error"] for r in rows), 4
        ),
        "median_int2_shipped_relative_error": round(
            statistics.median(r["int2_shipped_relative_error"] for r in rows), 4
        ),
        "median_roughness": round(statistics.median(r["roughness"] for r in rows), 4),
        "jl_wins_vs_int2_naive": f"{wins}/{len(rows)}",
        "jl_wins_vs_int2_shipped": f"{wins_shipped}/{len(rows)}",
    }

    header = (
        f"{'layer':>5} {'eff.rank':>9} {'full':>5} {'rough':>7} "
        f"{'JL err':>8} {'JL cos':>8} {'int2 nv':>8} {'int2 shp':>9}"
    )
    print(header)
    for r in rows:
        print(
            f"{r['layer']:5d} {r['effective_rank_99']:9.1f} {r['full_rank']:5d} "
            f"{r['roughness']:7.4f} {r['jl_relative_error']:8.4f} "
            f"{r['jl_cosine']:8.4f} {r['int2_naive_relative_error']:8.4f} "
            f"{r['int2_shipped_relative_error']:9.4f}"
        )
    print(f"\nJL beats naive equal-budget int2 on {wins}/{len(rows)} layers.")
    print(f"JL beats SHIPPED equal-budget int2 on {wins_shipped}/{len(rows)} layers.")
    print(
        f"median JL err {result['median_jl_relative_error']} vs "
        f"int2 naive {result['median_int2_naive_relative_error']} / "
        f"shipped {result['median_int2_shipped_relative_error']}, "
        f"median roughness {result['median_roughness']}"
    )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
