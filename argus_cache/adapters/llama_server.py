"""llama-server launch planning: binary choice, GGUF layout, MoE offload, argv.

ARGUS serves local models through llama-server directly rather than through
Ollama: the flags that matter on a small GPU (``-ncmoe``, a quantized KV cache,
per-run context sizing) are llama-server flags Ollama does not surface.
"""

from __future__ import annotations

import os
import shutil
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

#: llama-server accepts a wider set than Ollama's env var does, so validating
#: llama-server flags against Ollama's shorter list would reject usable types.
#: Note iq4_nl is the same width as q4_0 -- a non-linear codebook, not a smaller
#: one -- so it cannot buy context, only fidelity at equal size. None of the
#: types beyond f16/q8_0/q4_0 have been measured here.
LLAMA_KV_CACHE_TYPES = (
    "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
)


#: Fallback when the in-tree ARGUS build is absent: a stock llama-server on PATH.
DEFAULT_LLAMA_SERVER = shutil.which("llama-server") or "llama-server"

#: In-tree ARGUS-instrumented llama-server binary (P3 GPU prefill, Hetero_Overlap, S1/S2, WebUI).
ARGUS_LLAMA_SERVER = str(
    Path(__file__).resolve().parent.parent.parent
    / "scratch"
    / "llama.cpp-v050"
    / "build-cuda"
    / "bin"
    / "llama-server"
)

#: KV width llama-server is started with. q4_0 rather than iq4_nl because q4_0
#: is the rung every number in this module was measured on.
DEFAULT_SERVER_LLAMA_KV_TYPE = "q4_0"

#: Largest context measured to load *reliably* on a 4 GB card with the experts
#: on CPU and a q4_0 KV cache: 3420 of 3770 MiB, 2/2 loads. 163840 fits on paper
#: too -- 3696 MiB -- but at 98% occupancy it aborted inside CUDA on a repeat
#: run, so the ceiling arithmetic allows is not the ceiling that ships. At
#: 131072 the measured decode was 15.8-17.6 tok/s. See
#: ``test_default_context_leaves_vram_headroom``.
DEFAULT_LLAMA_SERVER_CTX = 131072

_GGUF_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_KV_BYTES_PER_ELEM = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0625,
    "q5_0": 0.6875,
    "q5_1": 0.75,
    "q4_0": 0.5625,
    "q4_1": 0.625,
    "iq4_nl": 0.5625,
}


def resolve_llama_server_binary(explicit: Optional[str] = None) -> str:
    """Prefer the in-tree ARGUS CUDA llama-server when built, else fallback."""
    if explicit:
        return explicit
    if os.path.isfile(ARGUS_LLAMA_SERVER) and os.access(ARGUS_LLAMA_SERVER, os.X_OK):
        return ARGUS_LLAMA_SERVER
    return DEFAULT_LLAMA_SERVER


def inspect_gguf_layout(model_path: str) -> Dict[str, Any]:
    """Read GGUF header metadata and per-layer MoE vs non-MoE byte sizes in <2 ms."""
    try:
        file_size = os.path.getsize(model_path)
        with open(model_path, "rb") as f:
            magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
            if magic != b"GGUF" or version not in (2, 3):
                return {}

            def read_str() -> str:
                ln = struct.unpack("<Q", f.read(8))[0]
                return f.read(ln).decode("utf-8", "replace")

            def skip_val(vtype: int) -> None:
                if vtype in _GGUF_SCALAR_SIZES:
                    f.read(_GGUF_SCALAR_SIZES[vtype])
                elif vtype == 8:
                    read_str()
                elif vtype == 9:
                    et, cnt = struct.unpack("<IQ", f.read(12))
                    for _ in range(cnt):
                        skip_val(et)

            meta: Dict[str, Any] = {}
            for _ in range(n_kv):
                key = read_str()
                vt = struct.unpack("<I", f.read(4))[0]
                if vt == 8 and key == "general.architecture":
                    meta[key] = read_str()
                elif vt == 4 and key.endswith((
                    ".block_count",
                    ".attention.key_length",
                    ".attention.value_length",
                    ".attention.head_count_kv",
                    ".full_attention_interval",
                )):
                    meta[key] = struct.unpack("<I", f.read(4))[0]
                else:
                    skip_val(vt)

            tensors = []
            for _ in range(n_tensors):
                tname = read_str()
                nd = struct.unpack("<I", f.read(4))[0]
                f.read(8 * nd)
                _, off = struct.unpack("<IQ", f.read(12))
                tensors.append((off, tname))
            data_start = (f.tell() + 31) & ~31
    except (OSError, struct.error, ValueError):
        return {}

    arch = meta.get("general.architecture", "")
    block_count = int(meta.get(f"{arch}.block_count", 0))
    tensors.sort()
    has_mtp = any(".nextn." in name for _, name in tensors)
    non_moe_bytes = 0
    embd_bytes = 0
    moe_by_layer: Dict[int, int] = {}
    for idx, (off, name) in enumerate(tensors):
        nxt = tensors[idx + 1][0] if idx + 1 < len(tensors) else max(off, file_size - data_start)
        sz = max(0, nxt - off)
        if ".nextn." in name or (block_count > 0 and name.startswith(f"blk.{block_count}.")):
            continue
        if "_exps." in name and name.startswith("blk."):
            try:
                il = int(name.split(".")[1])
            except ValueError:
                continue
            if block_count == 0 or il < block_count:
                moe_by_layer[il] = moe_by_layer.get(il, 0) + sz
        else:
            non_moe_bytes += sz
            if name.startswith("token_embd."):
                embd_bytes += sz

    gpu_non_moe_bytes = (non_moe_bytes - embd_bytes) if non_moe_bytes > embd_bytes else non_moe_bytes
    n_moe_layers = block_count if block_count > 0 else (max(moe_by_layer) + 1 if moe_by_layer else 0)
    moe_layers_mib = [moe_by_layer.get(i, 0) / 1048576.0 for i in range(n_moe_layers)] if moe_by_layer else []
    fa_interval = int(meta.get(f"{arch}.full_attention_interval", 0))
    kv_layers = (n_moe_layers // fa_interval) if (fa_interval > 0 and n_moe_layers > 0) else max(n_moe_layers, 1)
    n_head_kv = int(meta.get(f"{arch}.attention.head_count_kv", 2))
    k_len = int(meta.get(f"{arch}.attention.key_length", 256))
    v_len = int(meta.get(f"{arch}.attention.value_length", k_len))

    return {
        "architecture": arch,
        "block_count": n_moe_layers,
        "has_mtp": has_mtp,
        "non_moe_mib": non_moe_bytes / 1048576.0,
        "gpu_non_moe_mib": gpu_non_moe_bytes / 1048576.0,
        "moe_layers_mib": moe_layers_mib,
        "p3_arena_mib": max(moe_layers_mib) if moe_layers_mib else 0.0,
        "kv_elems_per_token": kv_layers * n_head_kv * (k_len + v_len),
    }


def sanitize_spec_type(spec_type: Optional[str], model_path: Optional[str]) -> Optional[str]:
    """Strip draft-mtp when a GGUF contains no MTP nextn tensors, avoiding startup GGML_ASSERT."""
    if not spec_type or "draft-mtp" not in spec_type or not model_path or not os.path.isfile(model_path):
        return spec_type
    layout = inspect_gguf_layout(model_path)
    if not layout or layout.get("has_mtp", True):
        return spec_type
    kept = [part.strip() for part in spec_type.split(",") if part.strip() and part.strip() != "draft-mtp"]
    return ",".join(kept) if kept else None


def plan_moe_offload(
    model_path: str,
    *,
    free_vram_mib: float,
    num_ctx: int = 32768,
    kv_cache_type: str = "q4_0",
    vram_reserve_mib: float = 320.0,
    prefer_p3_prefill: bool = True,
) -> Optional[int]:
    """Choose ``-ncmoe N`` so trailing MoE layers use spare VRAM without freezing the desktop."""
    layout = inspect_gguf_layout(model_path)
    moe_layers = layout.get("moe_layers_mib") or []
    if not moe_layers or free_vram_mib <= 0:
        return None

    bpe = _KV_BYTES_PER_ELEM.get(kv_cache_type, 1.0625)
    kv_mib = (num_ctx * layout.get("kv_elems_per_token", 10240) * bpe) / 1048576.0
    s2_and_cuda_mib = 200.0
    compute_mib = 424.0 + max(0, num_ctx - 32768) * (40.0 / 32768.0)
    p3_mib = layout.get("p3_arena_mib", 0.0) if prefer_p3_prefill else 0.0
    gpu_non_moe = layout.get("gpu_non_moe_mib", layout.get("non_moe_mib", 0.0))
    used_mib = gpu_non_moe + s2_and_cuda_mib + compute_mib + kv_mib + p3_mib + vram_reserve_mib

    n_cpu = len(moe_layers)
    while n_cpu > 0 and (used_mib + moe_layers[n_cpu - 1]) <= free_vram_mib:
        used_mib += moe_layers[n_cpu - 1]
        n_cpu -= 1
    return n_cpu


def llama_server_command(
    *,
    model_path: str,
    port: int,
    num_ctx: int = DEFAULT_LLAMA_SERVER_CTX,
    kv_cache_type: str = "q4_0",
    binary: str = DEFAULT_LLAMA_SERVER,
    host: str = "127.0.0.1",
    spec_type: Optional[str] = None,
    spec_draft_n_max: int = 2,
    draft_model: Optional[str] = None,
    flash_attn: str = "on",
    load_mode: Optional[str] = None,
    spec_draft_p_min: Optional[float] = None,
    n_cpu_moe: Optional[int] = None,
    threads: Optional[int] = None,
    threads_batch: Optional[int] = None,
    webui: bool = False,
    cache_ram: Optional[int] = None,
    mmproj: Optional[str] = None,
) -> List[str]:
    """Argument vector for a llama-server tuned for a small-VRAM machine.

    Ollama cannot serve this configuration: it does not expose ``--cpu-moe``,
    which on a mixture-of-experts model is the difference between 12.09 and
    19.14 tok/s (measured, RTX 3050 Ti 4 GB, qwen3.6-35b-a3b at 32k). Running
    llama-server directly is the only way to reach it.

    The three settings that matter, and why:

    * ``--cpu-moe`` pins the experts to host memory. Only 8 of 256 are active
      per token, so they are poor tenants of scarce VRAM; the attention weights
      and KV cache are what benefit from being resident.
    * ``-ctk``/``-ctv`` quantize the KV cache. Whether that cache fits in VRAM
      is the only thing that moves throughput -- about 18 tok/s while it does,
      8.8 tok/s once it spills -- and q4_0 raised the ceiling from 96k to 160k
      at no measured cost in speed.
    * ``-fa`` is a precondition: llama.cpp needs flash attention for a
      quantized KV cache, and silently ignores the request without it.
    * Speculative decoding (MTP + N-Gram) dramatically boosts token generation
      rate on CPU-MoE setups by verifying multiple tokens per RAM read.

    The default is ``q4_0`` because that is the rung the numbers above were
    measured on. ``iq4_nl`` is the same width with a non-linear codebook and
    should dominate it, but it has not been measured here -- defaulting to it
    would mean shipping an unmeasured setting justified by another setting's
    data.

    On quality: a multi-key retrieval probe -- four codes recalled from 31k
    tokens against eight confusable distractors, each answer required to bind
    the right code to the right project name -- scored 4/4 at f16, q8_0 and
    q4_0 alike, with byte-identical answers
    (``docs/measurements/kv-quantization-retrieval-2026-08-16.json``).

    Read that narrowly. It rules out retrieval damage at this width and
    length, which was the specific worry. It does not measure reasoning, code
    generation, or long-range coherence, it ran at 31k rather than at the
    131072 window shipped, and three identical answers suggest the probe still
    sits inside the model's comfortable range rather than at its margin.
    """
    if kv_cache_type not in LLAMA_KV_CACHE_TYPES:
        raise ValueError(
            f"kv_cache_type must be one of {', '.join(LLAMA_KV_CACHE_TYPES)}; "
            f"got {kv_cache_type!r}"
        )
    effective_spec = sanitize_spec_type(spec_type, model_path)
    cmd = [
        binary,
        "--model", model_path,
        "--port", str(port),
        "--host", host,
    ]
    # Tool calling needs the jinja chat template whether or not the bundled
    # web page is served; the two used to share one flag.
    cmd.append("--jinja")
    if not webui:
        cmd.append("--no-webui")
    cmd.extend([
        "-c", str(num_ctx),
        "-ngl", "99",
    ])
    if n_cpu_moe is None:
        cmd.append("--cpu-moe")
    elif n_cpu_moe > 0:
        cmd.extend(["-ncmoe", str(n_cpu_moe)])
    cmd.extend([
        "-ctk", kv_cache_type,
        "-ctv", kv_cache_type,
        "-fa", flash_attn,
        "-np", "1",
    ])
    if threads is not None and threads > 0:
        cmd.extend(["-t", str(threads)])
    if threads_batch is not None and threads_batch > 0:
        cmd.extend(["-tb", str(threads_batch)])
    if effective_spec:
        cmd.extend(["--spec-type", effective_spec])
        if spec_draft_n_max > 0:
            cmd.extend(["--spec-draft-n-max", str(spec_draft_n_max)])
            if "ngram-mod" in effective_spec:
                cmd.extend([
                    "--spec-ngram-mod-n-min", str(min(2, spec_draft_n_max)),
                    "--spec-ngram-mod-n-max", str(spec_draft_n_max),
                ])
        if spec_draft_p_min is not None:
            cmd.extend(["--spec-draft-p-min", str(spec_draft_p_min)])
        if draft_model:
            cmd.extend(["--spec-draft-model", draft_model])
    if load_mode:
        cmd.extend(["--load-mode", load_mode])
    if cache_ram is not None:
        cmd.extend(["--cache-ram", str(cache_ram)])
    if mmproj:
        cmd.extend(["--mmproj", mmproj])
    return cmd
