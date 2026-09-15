"""Native llama.cpp paged-KV checks, driven from pytest.

Needs a llama.cpp checkout with integrations/llama.cpp/host-kv.patch built (see
integrations/llama.cpp/README.md) and the stories15M GGUF:

  ARGUS_LLAMA_CPP_DIR=/path/to/llama.cpp ARGUS_TEST_GGUF=/path/to/stories15M.gguf pytest tests/test_native_llama_paged.py
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
LLAMA = Path(os.environ.get("ARGUS_LLAMA_CPP_DIR", "/nonexistent"))
MODEL = Path(os.environ.get("ARGUS_TEST_GGUF", "/nonexistent"))
BUILD = LLAMA / os.environ.get("ARGUS_LLAMA_BUILD", "build-cpu") / "bin"

pytestmark = pytest.mark.skipif(
    not (BUILD / "libllama.so").exists() or not MODEL.exists(),
    reason="set ARGUS_LLAMA_CPP_DIR (patched, built) and ARGUS_TEST_GGUF to run native checks",
)


def _compile(tmp_path, source, extra, libraries):
    binary = tmp_path / Path(source).stem
    subprocess.run(
        ["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", f"-I{LLAMA}/include", f"-I{LLAMA}/ggml/include",
         "-isystem", f"{LLAMA}/ggml/src", f"-I{ROOT}/argus_cache/csrc", str(ROOT / source), *extra,
         f"-L{BUILD}", f"-Wl,-rpath,{BUILD}", *libraries, "-o", str(binary)],
        check=True,
    )
    return binary


def _storage():
    # Paging needs a filesystem whose page cache can be evicted; tmpfs cannot.
    directory = Path(os.environ.get("ARGUS_TEST_KV_DIR", ROOT / "scratch" / "kv"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def test_paged_kernel_matches_double_precision_reference(tmp_path):
    csrc = ["argus_cache/csrc/ggml_host_buffer.cpp", "argus_cache/csrc/ggml_paged_attention.cpp",
            "argus_cache/csrc/ggml_disk_buffer.cpp"]
    binary = _compile(tmp_path, "tests/cpp/test_ggml_paged_attention.cpp", [str(ROOT / c) for c in csrc],
                      ["-lggml-base", "-lggml-cpu", "-lggml"])
    result = subprocess.run([binary, _storage()], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.count("max_error=") == 7


def test_paged_workers_release_coordination_between_invocations(tmp_path):
    binary = _compile(tmp_path, "tests/cpp/test_ggml_paged_coordination.cpp",
                      [str(ROOT / "argus_cache/csrc/ggml_host_buffer.cpp"),
                       str(ROOT / "argus_cache/csrc/ggml_disk_buffer.cpp"), "-pthread"],
                      ["-lggml-base", "-lggml-cpu", "-lggml"])
    subprocess.run([binary], check=True, timeout=60)


def test_direct_disk_pages_preserve_data_and_enforce_budgets(tmp_path):
    binary = _compile(tmp_path, "tests/cpp/test_ggml_disk_buffer.cpp", ["-Wl,--wrap=pwrite"],
                      ["-lggml-base", "-lggml-cpu", "-lggml"])
    subprocess.run([binary, _storage()], check=True, timeout=60)


# stories15M has head_dim=48; GGML quantized KV needs a multiple of 32.
# Q8/Q4 are exercised by the stored-value kernel checks above.
@pytest.mark.parametrize("kv_type", ["f32", "f16"])
@pytest.mark.parametrize("direct", [False, True])
def test_llama_paged_attention_parity_residency_and_lifecycle(tmp_path, kv_type, direct):
    binary = _compile(tmp_path, "tests/cpp/test_llama_paged_attention.cpp", [], ["-lllama", "-lggml-base"])
    env = os.environ.copy()
    env.pop("ARGUS_KV_STAGING_BYTES", None)
    if direct:
        env["ARGUS_KV_STAGING_BYTES"] = "4194304"
    result = subprocess.run([binary, MODEL, _storage(), kv_type], env=env, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads(next(line for line in result.stdout.splitlines() if line.startswith("{\"kv_type\"")))
    assert report["greedy_mismatches"] == 0
    assert report["argus_kv_rss_bytes"] <= report["resident_budget_bytes"] + 64 * 1024
    if direct:
        assert report["stats"]["mode"] == "direct"
        assert report["stats"]["peak_staging_bytes"] <= 4194304
        assert report["stats"]["peak_resident_bytes"] <= report["resident_budget_bytes"]
        assert report["stats"]["committed_pages"] > 0
    else:
        assert report["stats"]["paged_out_bytes"] > 0


@pytest.mark.parametrize("kv_type", ["q8_0", "q4_0"])
def test_quantized_model_direct_lifecycle(tmp_path, kv_type):
    model = Path(os.environ.get("ARGUS_TEST_QUANT_GGUF", "/nonexistent"))
    if not model.exists():
        pytest.skip("set ARGUS_TEST_QUANT_GGUF to a model with quantized-KV-compatible head dimensions")
    binary = _compile(tmp_path, "tests/cpp/test_llama_paged_attention.cpp", [], ["-lllama", "-lggml-base"])
    env = {**os.environ, "ARGUS_KV_STAGING_BYTES": "4194304"}
    env.pop("ARGUS_TEST_REPORT_LOGITS_ONLY", None)
    result = subprocess.run([binary, model, _storage(), kv_type], env=env, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"kv_type"')))
    assert report["greedy_mismatches"] == 0
    assert report["stats"]["peak_staging_bytes"] <= 4194304
    assert report["stats"]["peak_resident_bytes"] <= report["resident_budget_bytes"]


def test_llama_server_stock_host_and_paged_outputs_match(tmp_path):
    server = BUILD / "llama-server"
    if not server.exists():
        pytest.skip("llama-server was not built")
    result = subprocess.run(
        [sys.executable, ROOT / "tests/cpp/check_llama_server_ownership.py", server, MODEL, _storage(),
         os.environ.get("ARGUS_TEST_GPU_LAYERS", "0")],
        capture_output=True, text=True, timeout=1800,
    )
    assert result.returncode == 0, result.stderr[-3000:]
