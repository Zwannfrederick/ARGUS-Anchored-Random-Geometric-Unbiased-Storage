#!/usr/bin/env python3
"""ARGUS AI Service & Claude Code Gateway Manager (Start / Stop / Status / Chat / Web / Claude).

Supports 4 rigorous operational states:
- OFF      (Processes stopped, Claude Code using standard providers)
- STARTING (Gateway / llama-server booting, health probes in flight)
- READY    (Gateway online + Backend healthy + Model loaded + ARGUS data plane ready)
- ERROR    (Process failure, upstream unreachable, or memory abort)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from argus_cache.adapters.claude_session import (  # noqa: E402
    LOCAL_PROFILE_DIR,
    _CLAUDE_FALLBACK_PATHS,
    claude_launch_argv,
    claude_session_environment,
    claude_settings_env,  # noqa: F401 - re-exported for tests
    cleanup_claude_settings,
    model_label_for,
    reconcile_claude_settings,
    resolve_claude_binary,
    sync_claude_settings,  # noqa: F401 - re-exported for tests
)
from argus_cache.adapters.llama_server import (  # noqa: E402
    ARGUS_LLAMA_SERVER,
    DEFAULT_LLAMA_SERVER_CTX,
    DEFAULT_SERVER_LLAMA_KV_TYPE,
    LLAMA_KV_CACHE_TYPES,
    llama_server_command,
    plan_moe_offload,
    resolve_llama_server_binary,
    sanitize_spec_type,
)

RUNTIME_DIR = Path.home() / ".argus_runtime"
PID_FILE = RUNTIME_DIR / "gateway.pid"
BACKEND_PID_FILE = RUNTIME_DIR / "backend.pid"
LOG_FILE = RUNTIME_DIR / "gateway.log"
STATE_FILE = RUNTIME_DIR / "state.json"
CONFIG_FILE = RUNTIME_DIR / "config.json"

DEFAULT_GATEWAY_PORT = 8090
DEFAULT_UPSTREAM_URL = "http://127.0.0.1:8080/v1"
_DGPU_RUNTIME_STATUS = Path("/sys/bus/pci/devices/0000:01:00.0/power/runtime_status")


def ensure_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def get_stored_pid(pid_file: Path = PID_FILE) -> Optional[int]:
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, OSError):
        try:
            pid_file.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def check_gateway_health(port: int = DEFAULT_GATEWAY_PORT) -> Tuple[bool, Optional[Dict[str, Any]]]:
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(f"http://127.0.0.1:{port}/health")
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and data.get("status") == "ok" and "upstream" in data:
                    return True, data
    except Exception:
        pass
    return False, None


def check_backend_health(upstream_url: str = DEFAULT_UPSTREAM_URL) -> Tuple[bool, str]:
    try:
        models_url = f"{upstream_url.rstrip('/')}/models"
        with httpx.Client(timeout=1.0) as client:
            if client.get(models_url).status_code == 200:
                return True, "llama-server / OpenAI API (Healthy)"
    except Exception:
        pass

    return False, "Unreachable"


def get_vram_metrics(*, allow_wake: bool = True) -> Dict[str, Any]:
    if not allow_wake:
        try:
            if _DGPU_RUNTIME_STATUS.is_file() and _DGPU_RUNTIME_STATUS.read_text().strip() != "active":
                return {"available": False, "gpu_name": "GPU (uyku)", "used_mib": 0, "total_mib": 0, "free_mib": 0, "utilization_pct": 0}
        except OSError:
            pass
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total,name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 4:
                used_mib = float(parts[0])
                free_mib = float(parts[1])
                total_mib = float(parts[2])
                name = parts[3]
                pct = round((used_mib / total_mib) * 100, 1) if total_mib > 0 else 0.0
                return {
                    "available": True,
                    "gpu_name": name,
                    "used_mib": round(used_mib, 1),
                    "total_mib": round(total_mib, 1),
                    "free_mib": round(free_mib, 1),
                    "utilization_pct": pct,
                }
    except Exception:
        pass
    return {
        "available": False,
        "gpu_name": "CPU",
        "used_mib": 0,
        "total_mib": 0,
        "free_mib": 0,
        "utilization_pct": 0,
    }


def get_detailed_status(port: int = DEFAULT_GATEWAY_PORT, upstream_url: str = DEFAULT_UPSTREAM_URL) -> Dict[str, Any]:
    gateway_pid = get_stored_pid(PID_FILE)
    gateway_ok, _ = check_gateway_health(port)
    backend_ok, backend_msg = check_backend_health(upstream_url)
    vram = get_vram_metrics(allow_wake=(gateway_ok or backend_ok))

    if not gateway_pid and not gateway_ok:
        state, status_text, css_class, error_msg = "OFF", "ARGUS ● OFF", "offline", None
    elif gateway_ok and backend_ok:
        state, status_text, css_class, error_msg = "READY", "ARGUS ● READY", "ready", None
    elif gateway_ok and not backend_ok:
        state, status_text, css_class = "STARTING", "ARGUS ◐ STARTING", "starting"
        error_msg = "Gateway online, waiting for LLM Backend / Model load"
    elif gateway_pid and not gateway_ok:
        state, status_text, css_class = "STARTING", "ARGUS ◐ STARTING", "starting"
        error_msg = "Gateway daemon initializing..."
    else:
        state, status_text, css_class = "ERROR", "ARGUS ! ERROR", "error"
        error_msg = "Process failure or port conflict"

    active_upstream = upstream_url

    model_name = "unknown"
    context_length = None

    if "llama-server" in backend_msg:
        try:
            with httpx.Client(timeout=0.4) as c:
                r = c.get(active_upstream.replace("/v1", "") + "/props")
                if r.status_code == 200:
                    pdata = r.json()
                    mpath = pdata.get("model_path") or pdata.get("model_alias", "")
                    mname = Path(mpath).name if mpath else "llama-server model"
                    ftype = pdata.get("model_ftype", "")
                    model_name = f"{mname} ({ftype})" if ftype else mname
                    context_length = pdata.get("default_generation_settings", {}).get("n_ctx")
        except (httpx.HTTPError, ValueError, OSError):
            pass

    effective_kv_type = None
    try:
        effective_kv_type = json.loads(CONFIG_FILE.read_text()).get("kv_cache_type")
    except (OSError, ValueError, AttributeError):
        pass
    effective_kv_type = effective_kv_type or DEFAULT_SERVER_LLAMA_KV_TYPE

    return {
        "state": state,
        "status_text": status_text,
        "css_class": css_class,
        "gateway_pid": gateway_pid,
        "gateway_ok": gateway_ok,
        "backend_ok": backend_ok,
        "backend_type": backend_msg,
        "gateway_url": f"http://127.0.0.1:{port}",
        "upstream_url": active_upstream,
        "model_name": model_name,
        "kv_cache_type": effective_kv_type,
        "kv_managed_by": "ARGUS+backend" if Path(ARGUS_LLAMA_SERVER).is_file() else "backend",
        "context_length": context_length,
        "vram": vram,
        "error_message": error_msg,
    }


_CONFIG_DEFAULTS = {
    "llama_server_model": None,
    "llama_server_port": 8080,
    "num_ctx": DEFAULT_LLAMA_SERVER_CTX,
    "kv_cache_type": DEFAULT_SERVER_LLAMA_KV_TYPE,
    "spec_type": "ngram-mod",
    "spec_draft_n_max": 4,
    "draft_model": None,
    "flash_attn": "on",
    "load_mode": None,
    "spec_draft_p_min": None,
    "n_cpu_moe": None,
    "threads": None,
    "threads_batch": None,
    "cache_ram": None,
    "p3_min_tokens": None,
    "mmap_policy": None,
    # llama-server's bundled page duplicates Studio on :8090; off saves its RAM.
    "webui": False,
    "nice_level": 5,
    "sync_claude": False,
}

_CONFIG_ALIASES = {"workspace": "cwd"}


def apply_config(args, config_path: Path = None):
    """Fill unset arguments from the persisted config, then from defaults."""
    path = CONFIG_FILE if config_path is None else config_path
    config = {}
    try:
        config = json.loads(path.read_text())
        if not isinstance(config, dict):
            config = {}
    except (OSError, json.JSONDecodeError):
        config = {}

    kv_type = getattr(args, "kv_cache_type", None) or config.get("kv_cache_type")
    if kv_type is not None and kv_type not in LLAMA_KV_CACHE_TYPES:
        raise ValueError(
            f"kv_cache_type must be one of {', '.join(LLAMA_KV_CACHE_TYPES)}; "
            f"got {kv_type!r} (from {path})"
        )

    for key, fallback in _CONFIG_DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, config.get(key, fallback))
    for config_key, arg_name in _CONFIG_ALIASES.items():
        if getattr(args, arg_name, None) is None and config_key in config:
            setattr(args, arg_name, config[config_key])
    return args


def _llama_server_answering(port: int) -> bool:
    try:
        with httpx.Client(timeout=1.0) as client:
            return client.get(f"http://127.0.0.1:{port}/health").status_code == 200
    except httpx.HTTPError:
        return False


def _default_cpu_threads() -> Tuple[int, int]:
    logical = os.cpu_count() or 4
    phys = max(2, logical // 2)
    batch = min(logical, max(phys, phys + max(1, (logical - phys) // 2)))
    return phys, batch


def plan_llama_server_launch(args) -> Tuple[List[str], Dict[str, str], Optional[str], str]:
    """argv, environment, load mode and a one-line summary for llama-server.

    Shared by ``argus_ctl.sh start`` and the systemd launcher so the two cannot
    drift apart (they had: the RAM reserve only reached one of them).
    """
    binary = resolve_llama_server_binary(getattr(args, "llama_server_bin", None))
    is_argus_bin = os.path.abspath(binary) == os.path.abspath(ARGUS_LLAMA_SERVER)

    raw_spec = getattr(args, "spec_type", None)
    spec_type = sanitize_spec_type(raw_spec, args.llama_server_model)
    spec_draft_n_max = getattr(args, "spec_draft_n_max", 4)
    draft_model = getattr(args, "draft_model", None)
    flash_attn = getattr(args, "flash_attn", "on")
    load_mode = getattr(args, "load_mode", None)
    if load_mode is None or load_mode == "auto":
        load_mode = "mmap" if is_argus_bin else None
    spec_draft_p_min = getattr(args, "spec_draft_p_min", None)
    webui = bool(getattr(args, "webui", False)) and is_argus_bin

    phys_t, batch_t = _default_cpu_threads()
    threads = getattr(args, "threads", None) or phys_t
    threads_batch = getattr(args, "threads_batch", None) or batch_t

    n_cpu_moe = getattr(args, "n_cpu_moe", None)
    vram = get_vram_metrics()
    if n_cpu_moe is None and vram["available"] and args.llama_server_model:
        n_cpu_moe = plan_moe_offload(
            args.llama_server_model,
            free_vram_mib=vram["free_mib"],
            num_ctx=args.num_ctx,
            kv_cache_type=args.kv_cache_type,
        )

    raw_cache_ram = getattr(args, "cache_ram", None)
    effective_cache_ram = raw_cache_ram if raw_cache_ram is not None else (1536 if is_argus_bin else None)

    cmd = llama_server_command(
        model_path=args.llama_server_model,
        port=args.llama_server_port,
        num_ctx=args.num_ctx,
        kv_cache_type=args.kv_cache_type,
        binary=binary,
        spec_type=spec_type,
        spec_draft_n_max=spec_draft_n_max,
        draft_model=draft_model,
        flash_attn=flash_attn,
        load_mode=load_mode,
        spec_draft_p_min=spec_draft_p_min,
        n_cpu_moe=n_cpu_moe,
        threads=threads,
        threads_batch=threads_batch,
        webui=webui,
        cache_ram=effective_cache_ram,
    )

    env = dict(os.environ)
    if is_argus_bin:
        bin_dir = str(Path(binary).parent)
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:/opt/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
        env.setdefault("ARGUS_AUTO", "1")
        env.setdefault("ARGUS_KV_VRAM_MARGIN", "268435456")
        env.setdefault("ARGUS_P3_LOG", "1")
        p3_min = getattr(args, "p3_min_tokens", None)
        if p3_min is not None and str(p3_min).strip():
            env["ARGUS_P3_MIN_TOKENS"] = str(p3_min).strip()
        mmap_pol = getattr(args, "mmap_policy", None)
        env["ARGUS_MMAP_POLICY"] = str(mmap_pol).strip() if (mmap_pol is not None and str(mmap_pol).strip()) else "normal"
        env.setdefault("ARGUS_DYNAMIC_RAM", "1")
        ram_res = getattr(args, "ram_reserve", None)
        if ram_res is not None and str(ram_res).strip():
            env["ARGUS_RAM_RESERVE_MIB"] = str(ram_res).strip()

    moe_desc = f"ncmoe={n_cpu_moe}" if n_cpu_moe is not None else "cpu-moe"
    spec_desc = f", spec={spec_type}" if spec_type else ""
    lm_desc = f", load={load_mode}" if load_mode else ""
    summary = (
        f"{'ARGUS CUDA' if is_argus_bin else 'stock'}, ctx {args.num_ctx:,}, KV {args.kv_cache_type}, "
        f"{moe_desc}, t={threads}/{threads_batch}{spec_desc}{lm_desc}"
    )
    return cmd, env, load_mode, summary


def warm_model_page_cache(model_path: str) -> None:
    """Read the GGUF into page cache at idle IO and CPU priority, detached.

    With mmap the experts stay on disk until first touch, so the first request
    after boot paid the SATA read: a 14-token prefill took 9.7 s cold against
    0.4 s warm. BFQ honours the idle class, so this only spends disk time
    nobody else wants; the pages are ordinary clean cache the kernel can drop.
    ``setsid -f`` detaches the reader so it is never our zombie.
    """
    try:
        subprocess.run(
            ["setsid", "-f", "ionice", "-c3", "nice", "-n19", "cat", model_path],
            stdout=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"ARGUS warm-up skipped: {exc}", file=sys.stderr)


def _start_llama_server(args) -> None:
    """Launch ARGUS llama-server with anti-freeze VRAM/CPU scheduling."""
    cmd, env, load_mode, summary = plan_llama_server_launch(args)
    print(f"🔄 Starting llama-server [{summary}]...")
    if load_mode == "mmap":
        warm_model_page_cache(args.llama_server_model)

    nice_level = int(getattr(args, "nice_level", 5) or 0)

    def _preexec() -> None:
        if nice_level > 0 and hasattr(os, "nice"):
            try:
                os.nice(nice_level)
            except OSError:
                pass

    log = open(RUNTIME_DIR / "llama-server.log", "a")
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
        preexec_fn=_preexec if os.name == "posix" else None,
    )
    BACKEND_PID_FILE.write_text(str(proc.pid))

    timeout_s = int(getattr(args, "llama_server_timeout", 300) or 300)
    health = f"http://127.0.0.1:{args.llama_server_port}/health"
    for _ in range(timeout_s):
        if proc.poll() is not None:
            print(f"   ❌ llama-server exited during load; see {RUNTIME_DIR / 'llama-server.log'}")
            return
        try:
            with httpx.Client(timeout=1.0) as client:
                if client.get(health).status_code == 200:
                    if load_mode == "none" and hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                        try:
                            mfd = os.open(args.llama_server_model, os.O_RDONLY)
                            try:
                                os.posix_fadvise(mfd, 0, 0, os.POSIX_FADV_DONTNEED)
                            finally:
                                os.close(mfd)
                        except OSError:
                            pass
                    print("   ✅ llama-server ready.")
                    return
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    print(f"   ⚠️  llama-server did not answer within {timeout_s}s.")


def cmd_start(args) -> None:
    ensure_runtime_dir()
    apply_config(args)
    info = get_detailed_status(args.port, args.upstream)

    if info["state"] == "READY":
        print(f"⚠️  ARGUS Studio is already running and READY (Port: {args.port}).")
        return

    backend_ok, _ = check_backend_health(args.upstream)
    if getattr(args, "llama_server_model", None):
        if not _llama_server_answering(args.llama_server_port):
            _start_llama_server(args)
        backend_ok, _ = check_backend_health(args.upstream)


    print(f"🚀 Starting ARGUS Studio WebUI on port {args.port}...")
    log_fp = open(LOG_FILE, "w")
    cmd = [
        sys.executable,
        "-m",
        "argus_cache.webui.server",
        "--port",
        str(args.port),
        "--upstream",
        args.upstream,
    ]
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, start_new_session=True)
    PID_FILE.write_text(str(proc.pid))

    for _ in range(40):
        time.sleep(0.1)
        if get_detailed_status(args.port, args.upstream)["gateway_ok"]:
            break

    final_st = get_detailed_status(args.port, args.upstream)
    print(f"✅ Service State: {final_st['status_text']}")
    print(f"   ARGUS Studio WebUI: http://127.0.0.1:{args.port}")
    print(f"   LLM OpenAI API:     http://127.0.0.1:{args.llama_server_port}")
    print("   Terminal Chat:      ./scripts/argus_ctl.sh chat")


def cmd_stop(args) -> None:
    pid = get_stored_pid(PID_FILE)
    print(f"🛑 Stopping ARGUS Gateway (PID: {pid or 'all'})...")

    backend_pid = get_stored_pid(BACKEND_PID_FILE)
    if backend_pid:
        try:
            os.kill(backend_pid, signal.SIGTERM)
            for _ in range(80):
                time.sleep(0.1)
                os.kill(backend_pid, 0)
        except OSError:
            pass
        BACKEND_PID_FILE.unlink(missing_ok=True)
    elif _llama_server_answering(8080):
        subprocess.run(["fuser", "-k", "-TERM", "8080/tcp"], capture_output=True)
        for _ in range(80):
            if not _llama_server_answering(8080):
                break
            time.sleep(0.1)

    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(15):
                time.sleep(0.1)
                os.kill(pid, 0)
        except OSError:
            pass

    PID_FILE.unlink(missing_ok=True)
    subprocess.run(["fuser", "-k", f"{args.port}/tcp"], capture_output=True)
    cleanup_claude_settings()
    time.sleep(0.2)
    print("✅ ARGUS Gateway stopped (Offline). VRAM & RAM released.")


def cmd_restart(args) -> None:
    cmd_stop(args)
    time.sleep(0.5)
    cmd_start(args)


def cmd_status(args) -> None:
    info = get_detailed_status(args.port, args.upstream)
    healed = reconcile_claude_settings(port=args.port, gateway_ok=info["gateway_ok"])

    if getattr(args, "json", False):
        print(json.dumps(info, indent=2))
        return

    print("\n" + "=" * 60)
    print("                ARGUS AI TELEMETRY & STATUS")
    print("=" * 60)
    print(f"  Operational State:  {info['status_text']}")
    print(f"  Gateway URL:        {info['gateway_url']} ({'🟢 OK' if info['gateway_ok'] else '🔴 Down'})")
    print(f"  LLM Backend:        {info['upstream_url']} ({'🟢 OK' if info['backend_ok'] else '🟡 ' + info['backend_type']})")
    print(f"  Active Model:       {info['model_name']}")
    print(f"  KV Cache Type:      {info['kv_cache_type']} ({info['kv_managed_by']})")
    ctx = info["context_length"]
    print(f"  Context Length:     {ctx:,}" if ctx else "  Context Length:     (no model loaded)")

    v = info["vram"]
    if v["available"]:
        print(f"  GPU Hardware:       {v['gpu_name']}")
        print(f"  VRAM Allocation:    {v['used_mib']} / {v['total_mib']} MiB ({v['utilization_pct']}%, free {v['free_mib']} MiB)")
    else:
        print("  GPU Hardware:       CPU Mode")

    if info["error_message"]:
        print(f"  Notice:             ⚠️  {info['error_message']}")
    if healed:
        print("  Notice:             ℹ️  cleared a stale Claude Code redirect")
    print("=" * 60 + "\n")


def cmd_claude(args) -> int:
    info = get_detailed_status(args.port, args.upstream)
    if not info["gateway_ok"]:
        if getattr(args, "auto_start", False):
            print("🔄 ARGUS kapalı, yerel model otomatik başlatılıyor...")
            cmd_start(args)
            info = get_detailed_status(args.port, args.upstream)
        if not info["gateway_ok"]:
            print(f"❌ No gateway on port {args.port}. Start it with `argus_ctl.sh start`.")
            return 1

    binary = resolve_claude_binary()
    if binary is None:
        print(
            "❌ Claude Code CLI not found. Looked on PATH and in "
            + ", ".join(f"~/{p}" for p in _CLAUDE_FALLBACK_PATHS)
            + f".\n   Install it, or launch with: ANTHROPIC_BASE_URL=http://127.0.0.1:{args.port} claude"
        )
        return 1

    args = apply_config(args)
    workspace = getattr(args, "cwd", None)
    if workspace and not Path(workspace).is_dir():
        print(f"❌ workspace {workspace!r} is not a directory")
        return 1

    label = model_label_for(getattr(args, "llama_server_model", None))
    env = claude_session_environment(
        os.environ,
        port=args.port,
        model_label=label,
        num_ctx=getattr(args, "num_ctx", None) or DEFAULT_LLAMA_SERVER_CTX,
        config_dir=None if getattr(args, "with_plugins", False) else LOCAL_PROFILE_DIR,
    )
    argv = claude_launch_argv(model_label=label, extra=getattr(args, "claude_args", []), binary=binary)
    profile = "your profile" if getattr(args, "with_plugins", False) else "lean profile"
    print(f"🚀 Claude Code -> http://127.0.0.1:{args.port} ({label}, {profile}, this session only)")
    try:
        return subprocess.call(argv, env=env, cwd=workspace or None)
    except FileNotFoundError:
        print(f"❌ could not execute {binary}")
        return 1


def _ensure_backend_for_interactive(args) -> str:
    apply_config(args)
    base_url = args.upstream.rstrip("/")
    if not check_backend_health(base_url)[0]:
        print("🔄 Backend kapalı, ARGUS otomatik başlatılıyor...")
        cmd_start(args)
    return base_url


def cmd_web(args) -> int:
    apply_config(args)
    if not get_detailed_status(args.port, args.upstream)["gateway_ok"]:
        print("🔄 ARGUS Studio kapalı, otomatik başlatılıyor...")
        cmd_start(args)
    ui_url = f"http://127.0.0.1:{args.port}"
    print(f"🌐 ARGUS Studio WebUI: {ui_url}")
    subprocess.Popen(["xdg-open", ui_url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


def cmd_chat(args) -> int:
    """Zero-load interactive streaming terminal chat against the running ARGUS backend."""
    base_url = _ensure_backend_for_interactive(args)
    endpoint = f"{base_url}/chat/completions"
    thinking = bool(getattr(args, "think", False))
    messages = []
    if getattr(args, "system", None):
        messages.append({"role": "system", "content": args.system})

    def _stream_turn(user_text: str) -> None:
        messages.append({"role": "user", "content": user_text})
        payload = {
            "messages": messages,
            "stream": True,
            "temperature": getattr(args, "temp", 0.6),
            "n_predict": -1,
            "reasoning_budget_tokens": -1 if thinking else 0,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        full_reply = []
        t0 = time.perf_counter()
        tok_count = 0
        with httpx.Client(timeout=httpx.Timeout(connect=15.0, read=None, write=60.0, pool=15.0)) as client:
            with client.stream("POST", endpoint, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    r_piece = delta.get("reasoning_content") or ""
                    c_piece = delta.get("content") or ""
                    if r_piece:
                        sys.stdout.write(f"\033[2m{r_piece}\033[0m")
                        sys.stdout.flush()
                        tok_count += 1
                    if c_piece:
                        sys.stdout.write(c_piece)
                        sys.stdout.flush()
                        full_reply.append(c_piece)
                        tok_count += 1
                    timings = chunk.get("timings")
                    if timings and timings.get("predicted_per_second"):
                        tps = timings["predicted_per_second"]
                        pps = timings.get("prompt_per_second", 0.0)
                        sys.stdout.write(f"\n\033[36m[⚡ {tps:.1f} tok/s decode | {pps:.1f} tok/s prefill]\033[0m")
        elapsed = max(1e-3, time.perf_counter() - t0)
        if tok_count and "timings" not in locals():
            sys.stdout.write(f"\n\033[36m[⚡ ~{tok_count / elapsed:.1f} tok/s]\033[0m")
        print("\n")
        messages.append({"role": "assistant", "content": "".join(full_reply)})

    one_shot = " ".join(getattr(args, "prompt", []) or []).strip()
    if one_shot:
        _stream_turn(one_shot)
        return 0

    print(f"⚡ ARGUS Terminal Chat ({base_url}) — '/clear' sıfırlar, '/think' düşünme modunu değiştirir, '/exit' çıkar.\n")
    while True:
        try:
            user_inp = input("\033[1;32msen>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_inp:
            continue
        if user_inp in ("/exit", "/quit", "exit", "quit"):
            return 0
        if user_inp == "/clear":
            messages = [m for m in messages if m["role"] == "system"]
            print("🧹 Bağlam temizlendi.\n")
            continue
        if user_inp == "/think":
            thinking = not thinking
            print(f"🧠 Thinking modu: {'AÇIK' if thinking else 'KAPALI'}\n")
            continue
        try:
            _stream_turn(user_inp)
        except Exception as exc:
            print(f"\n❌ İstek hatası: {exc}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ARGUS AI Service & Claude Code Gateway Controller")
    parser.add_argument("--port", type=int, default=DEFAULT_GATEWAY_PORT, help="Gateway port (default: 8008)")
    parser.add_argument("--upstream", type=str, default=DEFAULT_UPSTREAM_URL, help="Upstream OpenAI API endpoint")
    parser.add_argument("--json", action="store_true", help="Output status in JSON format")

    subparsers = parser.add_subparsers(dest="action", required=True)
    start_parser = subparsers.add_parser("start", help="Start ARGUS Gateway and tuned llama-server")
    start_parser.add_argument("--llama-server-model", type=str, default=None)
    start_parser.add_argument("--llama-server-port", type=int, default=None)
    start_parser.add_argument("--llama-server-bin", type=str, default=None)
    start_parser.add_argument("--llama-server-timeout", type=int, default=300)
    start_parser.add_argument("--num-ctx", type=int, default=None)
    start_parser.add_argument("--kv-cache-type", type=str, default=None)
    start_parser.add_argument("--n-cpu-moe", type=int, default=None, help="CPU MoE layer count (-ncmoe); auto-planned if omitted")
    start_parser.add_argument("--threads", type=int, default=None)
    start_parser.add_argument("--threads-batch", type=int, default=None)
    start_parser.add_argument("--spec-type", type=str, default=None)
    start_parser.add_argument("--spec-draft-n-max", type=int, default=None)
    start_parser.add_argument("--draft-model", type=str, default=None)
    start_parser.add_argument("--flash-attn", type=str, choices=["on", "off", "auto"], default=None)
    start_parser.add_argument("--load-mode", type=str, choices=["auto", "none", "mmap", "mlock", "mmap+mlock"], default=None)
    start_parser.add_argument("--cache-ram", type=int, default=None, help="Prompt cache RAM limit in MiB (default: 2048 for ARGUS)")
    start_parser.add_argument("--p3-min-tokens", type=str, default=None, help="ARGUS_P3_MIN_TOKENS ('auto' or integer e.g. 64, 128, 256, 512)")
    start_parser.add_argument("--mmap-policy", type=str, choices=["random", "normal", "sequential"], default=None)
    start_parser.add_argument("--spec-draft-p-min", type=float, default=None)
    start_parser.add_argument("--no-webui", dest="webui", action="store_false", default=None)
    start_parser.add_argument("--sync-claude", action="store_true", default=None)

    claude_parser = subparsers.add_parser("claude", help="Launch one Claude Code session bound to the gateway")
    claude_parser.add_argument("--llama-server-model", type=str, default=None)
    claude_parser.add_argument("--with-plugins", action="store_true")
    claude_parser.add_argument("--auto-start", action="store_true", help="Auto-start ARGUS if currently offline")
    claude_parser.add_argument("--cwd", type=str, default=None)
    claude_parser.add_argument("claude_args", nargs=argparse.REMAINDER)

    chat_parser = subparsers.add_parser("chat", help="Interactive terminal chat with the loaded ARGUS model")
    chat_parser.add_argument("--think", action="store_true", help="Enable Qwen3.6 reasoning (<think>) mode")
    chat_parser.add_argument("--temp", type=float, default=0.6)
    chat_parser.add_argument("--system", type=str, default=None)
    chat_parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt")

    subparsers.add_parser("web", help="Open the embedded ARGUS WebUI in the default browser")
    subparsers.add_parser("stop", help="Stop ARGUS Gateway & backend")
    status_parser = subparsers.add_parser("status", help="Get operational status and telemetry")
    status_parser.add_argument("--json", action="store_true")
    subparsers.add_parser("restart", help="Restart ARGUS service")

    args = parser.parse_args()
    actions = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "claude": cmd_claude,
        "chat": cmd_chat,
        "web": cmd_web,
        "restart": cmd_restart,
    }
    rc = actions[args.action](args)
    if isinstance(rc, int) and rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
