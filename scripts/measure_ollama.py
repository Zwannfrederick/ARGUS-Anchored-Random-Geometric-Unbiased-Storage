"""Record a live Ollama baseline.

These are **external baseline** numbers. Ollama runs as a separate OS process
and owns its own KV cache; ARGUS cannot manage it, and the adapter's telemetry
says so explicitly. The measurement exists so the Ollama adapter's claims can
be checked against a real server, not so ARGUS can take credit for them.

Usage::

    python scripts/measure_ollama.py --output docs/measurements/ollama-<date>.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess

from argus_cache.adapters.ollama import OllamaAdapter

PROMPT = "Explain virtual memory in one sentence."


def _pkg() -> str | None:
    """The distro package providing the server, if it came from pacman."""
    for name in ("ollama-cuda", "ollama"):
        try:
            return subprocess.check_output(["pacman", "-Q", name], text=True).strip()
        except Exception:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--date", required=True, help="ISO date of this run")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--num-ctx", type=int, default=2048)
    args = parser.parse_args()

    with OllamaAdapter(model=args.model, num_ctx=args.num_ctx) as ollama:
        runs = [
            ollama.generate(PROMPT, max_tokens=args.max_tokens)
            for _ in range(args.runs)
        ]
        payload = {
            "runtime": "ollama",
            "claim_class": "runtime",
            "note": (
                "External baseline. ARGUS does NOT manage this KV cache: Ollama "
                "runs in its own process and owns its own memory. These numbers "
                "describe the unmodified server and must never be presented as "
                "an ARGUS result or an ARGUS speedup."
            ),
            "date": args.date,
            "command": (
                f"python scripts/measure_ollama.py --output {args.output} "
                f"--model {args.model} --date {args.date}"
            ),
            "server_version": ollama.server_version,
            "model": ollama.model,
            "num_ctx": args.num_ctx,
            "max_tokens": args.max_tokens,
            "prompt": PROMPT,
            "ollama_pkg": _pkg(),
            "host_platform": platform.platform(),
            "python": platform.python_version(),
            "runs": [
                {
                    "wall_s": round(r.wall_seconds, 4),
                    "tps": round(r.tokens_per_second, 3) if r.tokens_per_second else None,
                    "prompt_eval_s": r.prompt_eval_seconds,
                    "eval_s": r.eval_seconds,
                    "eval_tokens": r.eval_tokens,
                }
                for r in runs
            ],
            "telemetry": ollama.telemetry(),
        }

    # Written to a file rather than stdout: the ARGUS logger also writes to
    # stdout, and a redirect would splice log lines into the JSON.
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
