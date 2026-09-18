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


@pytest.mark.skipif(os.environ.get("ARGUS_TEST_CUDA") != "1", reason="set ARGUS_TEST_CUDA=1 for real CUDA mechanism checks")
def test_cuda_tier_migration_and_attention(tmp_path):
    import shutil

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("CUDA toolkit required")
    cuda = Path(nvcc).resolve().parents[1]
    obj = tmp_path / "cuda_attention.o"
    subprocess.run(
        [nvcc, "-std=c++17", "-arch=sm_86", "-Xcompiler=-fPIC",
         f"-I{LLAMA}/ggml/include", f"-I{LLAMA}/ggml/src", f"-I{ROOT}/argus_cache/csrc",
         "-c", str(ROOT / "argus_cache/csrc/ggml_cuda_attention.cu"), "-o", str(obj)], check=True,
    )
    binary = _compile(tmp_path, "tests/cpp/test_ggml_cuda_mechanism.cpp",
                      ["-DARGUS_CUDA", f"-I{cuda}/include", str(ROOT / "argus_cache/csrc/ggml_disk_buffer.cpp"),
                       str(ROOT / "argus_cache/csrc/ggml_kv_policy.cpp"),
                       str(obj), f"-L{cuda}/lib64", f"-Wl,-rpath,{cuda}/lib64",
                       "-Wl,--wrap=pwrite", "-Wl,--wrap=pread", "-pthread"],
                      ["-lggml-base", "-lggml-cpu", "-lggml", "-lcudart"])
    for dimension in (48, 64, 256):
        subprocess.run([binary, _storage(), str(dimension)], check=True, timeout=180,
                       env={**os.environ, "ARGUS_KV_PROFILE": "1"})
    subprocess.run([binary, _storage(), "64"], check=True, timeout=180,
                   env={**os.environ, "ARGUS_KV_PROFILE": "cpu"})


@pytest.mark.skipif(os.environ.get("ARGUS_TEST_CUDA") != "1", reason="set ARGUS_TEST_CUDA=1 with CUDA build")
def test_cuda_attention_native_lifecycle(tmp_path):
    binary = _compile(tmp_path, "tests/cpp/test_llama_paged_attention.cpp", ["-DARGUS_TEST_CUDA_TIER"], ["-lllama", "-lggml-base"])
    env = {**os.environ, "ARGUS_KV_STAGING_BYTES": "4194304", "ARGUS_KV_GPU_BYTES": "4194304",
           "ARGUS_KV_PINNED_BYTES": "4194304", "ARGUS_TEST_PROMPT_TOKENS": "64"}
    env.pop("ARGUS_TEST_REPORT_LOGITS_ONLY", None)
    result = subprocess.run([binary, MODEL, _storage(), "f16"], env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-4000:]
    report = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"kv_type"')))
    assert report["compared_steps"] == 19 and report["greedy_mismatches"] == 0
    assert report["migrated_pages"] == 2
    assert report["stats"]["cuda_attention_calls"] > 0
    for key in ("peak_staging_bytes", "peak_gpu_bytes", "peak_pinned_bytes"):
        assert report["stats"][key] <= 4194304


@pytest.mark.skipif(os.environ.get("ARGUS_TEST_CUDA") != "1", reason="set ARGUS_TEST_CUDA=1 with CUDA build")
@pytest.mark.parametrize("gpu_budget", [4194304, 262144])
def test_cuda_policy_off_on_preserves_outputs_and_budgets(tmp_path, gpu_budget):
    binary = _compile(tmp_path, "tests/cpp/test_llama_paged_attention.cpp", ["-DARGUS_TEST_CUDA_TIER"],
                      ["-lllama", "-lggml-base"])
    env = {**os.environ, "ARGUS_KV_STAGING_BYTES": "4194304", "ARGUS_KV_GPU_BYTES": str(gpu_budget),
           "ARGUS_KV_PINNED_BYTES": "4194304", "ARGUS_TEST_PROMPT_TOKENS": "64",
           "ARGUS_TEST_AUTO_POLICY": "1"}
    env.pop("ARGUS_TEST_REPORT_LOGITS_ONLY", None)
    env.pop("ARGUS_KV_RAM_BYTES", None)
    reports = {}
    for mode in ("off", "on"):
        result = subprocess.run([binary, MODEL, _storage(), "f16"], env={**env, "ARGUS_KV_POLICY": mode},
                                capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stderr[-4000:]
        report = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"kv_type"')))
        assert report["compared_steps"] == 19 and report["greedy_mismatches"] == 0
        assert report["migrated_pages"] == 0  # Every migration must come from the runtime policy.
        for key in ("peak_staging_bytes", "peak_gpu_bytes", "peak_pinned_bytes"):
            assert report["stats"][key] <= (gpu_budget if key == "peak_gpu_bytes" else 4194304)
        assert report["stats"]["policy_rejected"] == 0
        reports[mode] = report
    (tmp_path / "policy-off-on.json").write_text(json.dumps(reports, indent=2) + "\n")
    off, on = reports["off"]["stats"], reports["on"]["stats"]
    assert off["policy_promotions"] == off["policy_demotions"] == 0
    assert off["policy_nanoseconds"] == 0 and on["policy_nanoseconds"] > 0
    assert on["policy_promotions"] > 0
    if gpu_budget == 4194304:
        assert on["read_bytes"] < off["read_bytes"]
    else:
        assert on["policy_demotions"] > 0


@pytest.mark.skipif(os.environ.get("ARGUS_TEST_CUDA") != "1", reason="set ARGUS_TEST_CUDA=1 with CUDA build")
@pytest.mark.parametrize("gpu_budget", [4194304, 262144])
def test_cuda_mixed_resident_path_preserves_policy_semantics(tmp_path, gpu_budget):
    """Staging cold pages for one read must look to the policy exactly like the staged path."""
    binary = _compile(tmp_path, "tests/cpp/test_llama_paged_attention.cpp", ["-DARGUS_TEST_CUDA_TIER"],
                      ["-lllama", "-lggml-base"])
    env = {**os.environ, "ARGUS_KV_STAGING_BYTES": "4194304", "ARGUS_KV_GPU_BYTES": str(gpu_budget),
           "ARGUS_KV_PINNED_BYTES": "4194304", "ARGUS_TEST_PROMPT_TOKENS": "600",
           "ARGUS_TEST_AUTO_POLICY": "1", "ARGUS_KV_POLICY": "on"}
    env.pop("ARGUS_TEST_REPORT_LOGITS_ONLY", None)
    env.pop("ARGUS_KV_RAM_BYTES", None)
    stats = {}
    for path in ("staged", "batched"):
        result = subprocess.run([binary, MODEL, _storage(), "f16"], env={**env, "ARGUS_KV_ATTENTION_PATH": path},
                                capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stderr[-4000:]
        report = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"kv_type"')))
        assert report["greedy_mismatches"] == 0
        stats[path] = report["stats"]
    staged, mixed = stats["staged"], stats["batched"]
    for key in ("policy_promotions", "policy_demotions", "policy_rejected", "written_bytes", "committed_pages",
                "cuda_attention_calls"):
        assert staged[key] == mixed[key], key
    # Staged 32-cell tiles re-read pages that straddle tiles (576-byte cells); mixed reads each once.
    assert mixed["read_bytes"] <= staged["read_bytes"]
    assert staged["resident_prefill_accepted"] == 0
    details = {k: v for k, v in mixed.items() if k.startswith("resident_") and v}
    if gpu_budget == 4194304:
        assert mixed["resident_prefill_accepted"] > 0 and mixed["resident_prefill_cold_pages"] > 0, details
    else:  # The policy keeps the GPU tier full: cold scratch declines to the staged path.
        assert mixed["resident_prefill_accepted"] == 0 and mixed["resident_prefill_reject_cold_scratch_budget"] > 0, details


@pytest.mark.skipif(os.environ.get("ARGUS_TEST_CUDA") != "1", reason="requires real CUDA")
def test_cuda_gpu_control_has_no_disk_io_and_preserves_lifecycle(tmp_path):
    binary = _compile(tmp_path, "tests/cpp/test_llama_paged_attention.cpp", ["-DARGUS_TEST_CUDA_TIER"],
                      ["-lllama", "-lggml-base"])
    env = {**os.environ, "ARGUS_KV_STAGING_BYTES": "4194304", "ARGUS_KV_GPU_BYTES": "4194304",
           "ARGUS_KV_PINNED_BYTES": "4194304", "ARGUS_TEST_PROMPT_TOKENS": "64",
           "ARGUS_TEST_AUTO_POLICY": "1", "ARGUS_KV_GPU_CONTROL": "1", "ARGUS_KV_POLICY": "off",
           "ARGUS_KV_PROFILE": "1"}
    env.pop("ARGUS_TEST_REPORT_LOGITS_ONLY", None)
    result = subprocess.run([binary, MODEL, _storage(), "f16"], env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-4000:]
    report = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"kv_type"')))
    assert report["compared_steps"] == 19 and report["greedy_mismatches"] == 0
    stats = report["stats"]
    assert stats["read_bytes"] == stats["written_bytes"] == stats["disk_bytes"] == 0
    assert stats["profile_prefill_kernel_gpu_ns"] > 0 and stats["profile_decode_kernel_gpu_ns"] > 0
    assert stats["profile_decode_d2d_bytes"] > 0
    assert stats["peak_gpu_bytes"] <= 4194304
