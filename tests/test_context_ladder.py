"""Small checks for benchmark input identity and failure-aware summaries."""
from unittest.mock import patch

from benchmarks.bench_llama_paged_context import QUESTION, NEEDLE, build_prompt, summarize


def test_token_budget_preserves_needle_question_and_reproducibility():
    def tokenize(_url, payload):
        return {"tokens": ([1] if payload.get("add_special") else []) + list(payload["content"].encode())}

    with patch("benchmarks.bench_llama_paged_context.http_json", side_effect=tokenize):
        prompt = build_prompt("unused", 512, 123456)
        assert prompt == build_prompt("unused", 512, 123456)
        assert len(prompt) == 512 and prompt[0] == 1
        assert bytes(prompt[1:]).startswith(NEEDLE.format(code=123456).encode())
        assert bytes(prompt).endswith(QUESTION.encode())
        try:
            build_prompt("unused", 2, 123456)
        except ValueError:
            pass
        else:
            raise AssertionError("undersized prompt budget was accepted")


def test_summary_excludes_failed_runs_instead_of_imputing_zero():
    metrics = ("wall_seconds", "prefill_seconds", "decode_tokens_per_second", "tpot_seconds")
    runs = [{"mode": "argus-cuda-on", "context": 1024, **dict.fromkeys(metrics, value)} for value in (1, 3, 8)]
    runs.append({"mode": "argus-cuda-on", "context": 1024, "error": "timeout"})
    result = summarize(runs)[0]
    assert result["completed"] == 3 and result["failed"] == 1
    assert result["wall_seconds"] == {"min": 1, "median": 3, "max": 8}
    assert "wall_seconds" not in summarize(runs[-1:])[0]
