#!/usr/bin/env python3
"""ARGUS AI Service & Claude Code Gateway Manager (Start / Stop / Status / Restart / Logs / Telemetry).

Supports 4 rigorous operational states:
- OFF      (Processes stopped, Claude Code using standard providers)
- STARTING (Gateway / Ollama / llama-server booting, health probes in flight)
- READY    (Gateway online + Backend healthy + Qwen3.8 loaded + ARGUS data plane ready)
- ERROR    (Process failure, upstream unreachable, or memory abort)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from argus_cache.adapters.ollama import (
    DEFAULT_LLAMA_SERVER_CTX,
    LLAMA_KV_CACHE_TYPES,
    DEFAULT_SERVER_KV_CACHE_TYPE,
    DEFAULT_SERVER_LLAMA_KV_TYPE,
    llama_server_command,
    recommended_server_environment,
)

#: Ollama ships the CUDA ggml backend in its own lib directory.
LLAMA_CUDA_DIR = "/usr/lib/ollama/cuda_v13"

RUNTIME_DIR = Path.home() / ".argus_runtime"
PID_FILE = RUNTIME_DIR / "gateway.pid"
BACKEND_PID_FILE = RUNTIME_DIR / "backend.pid"
LOG_FILE = RUNTIME_DIR / "gateway.log"
STATE_FILE = RUNTIME_DIR / "state.json"

DEFAULT_GATEWAY_PORT = 8008
DEFAULT_UPSTREAM_URL = "http://127.0.0.1:8080/v1"
OLLAMA_API_URL = "http://127.0.0.1:11434"


def ensure_runtime_dir():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def get_stored_pid(pid_file: Path = PID_FILE) -> Optional[int]:
    if not pid_file.exists():
        return None
    try:
        pid_str = pid_file.read_text().strip()
        pid = int(pid_str)
        os.kill(pid, 0)
        return pid
    except (ValueError, OSError):
        if pid_file.exists():
            pid_file.unlink(missing_ok=True)
        return None


def check_gateway_health(port: int = DEFAULT_GATEWAY_PORT) -> Tuple[bool, Optional[Dict[str, Any]]]:
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(f"http://127.0.0.1:{port}/health")
            if resp.status_code == 200:
                data = resp.json()
                # Verify genuine ARGUS Gateway signature, rejecting foreign services on port
                if isinstance(data, dict) and data.get("status") == "ok" and "upstream" in data:
                    return True, data
    except Exception:
        pass
    return False, None


def check_backend_health(upstream_url: str = DEFAULT_UPSTREAM_URL) -> Tuple[bool, str]:
    # Check llama-server or OpenAI-compatible endpoint
    try:
        models_url = f"{upstream_url.rstrip('/')}/models"
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(models_url)
            if resp.status_code == 200:
                return True, "llama-server / OpenAI API (Healthy)"
    except Exception:
        pass

    # Check Ollama endpoint
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(f"{OLLAMA_API_URL}/api/tags")
            if resp.status_code == 200:
                return True, "Ollama Service (Healthy)"
    except Exception:
        pass

    return False, "Unreachable"


def get_vram_metrics() -> Dict[str, Any]:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 3:
                used_mib = float(parts[0])
                total_mib = float(parts[1])
                name = parts[2]
                free_mib = max(0.0, total_mib - used_mib)
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
    gateway_ok, health_data = check_gateway_health(port)
    backend_ok, backend_msg = check_backend_health(upstream_url)
    vram = get_vram_metrics()

    # Determine State: OFF, STARTING, READY, ERROR
    if not gateway_pid and not gateway_ok:
        state = "OFF"
        status_text = "ARGUS ● OFF"
        css_class = "offline"
        error_msg = None
    elif gateway_ok and backend_ok:
        state = "READY"
        status_text = "ARGUS ● READY"
        css_class = "ready"
        error_msg = None
    elif gateway_ok and not backend_ok:
        # Gateway is up but waiting for LLM backend or model loading
        state = "STARTING"
        status_text = "ARGUS ◐ STARTING"
        css_class = "starting"
        error_msg = "Gateway online, waiting for LLM Backend / Model load"
    elif gateway_pid and not gateway_ok:
        # Process started but health check not yet answering (booting or dead)
        state = "STARTING"
        status_text = "ARGUS ◐ STARTING"
        css_class = "starting"
        error_msg = "Gateway daemon initializing..."
    else:
        state = "ERROR"
        status_text = "ARGUS ! ERROR"
        css_class = "error"
        error_msg = "Process failure or port conflict"

    active_upstream = upstream_url if "llama-server" in backend_msg else (f"{OLLAMA_API_URL}/v1" if backend_ok else upstream_url)

    model_name = "unknown"
    context_length = None

    # 1. Discover from llama-server if active
    if "llama-server" in backend_msg:
        try:
            with httpx.Client(timeout=0.4) as c:
                props_url = active_upstream.replace("/v1", "") + "/props"
                r = c.get(props_url)
                if r.status_code == 200:
                    pdata = r.json()
                    mpath = pdata.get("model_path") or pdata.get("model_alias", "")
                    mname = Path(mpath).name if mpath else "llama-server model"
                    ftype = pdata.get("model_ftype", "")
                    model_name = f"{mname} ({ftype})" if ftype else mname
                    gen_settings = pdata.get("default_generation_settings", {})
                    context_length = gen_settings.get("n_ctx")
        except (httpx.HTTPError, ValueError, OSError):
            pass

    # 2. Discover model from Ollama if still unknown
    if model_name == "unknown":
        try:
            with httpx.Client(timeout=0.4) as c:
                r = c.get("http://127.0.0.1:11434/api/tags")
                if r.status_code == 200:
                    models = r.json().get("models", [])
                    if models:
                        m = models[0]
                        sz_gb = round(m.get("size", 0) / (1024**3), 1)
                        model_name = f"{m.get('name')} ({sz_gb} GB)"
        except (httpx.HTTPError, ValueError):
            pass

    # 3. Context length from Ollama if not yet discovered
    if context_length is None:
        try:
            with httpx.Client(timeout=0.4) as c:
                r = c.get("http://127.0.0.1:11434/api/ps")
                if r.status_code == 200:
                    loaded = r.json().get("models", [])
                    if loaded:
                        context_length = loaded[0].get("context_length")
        except (httpx.HTTPError, ValueError):
            pass

    # Determine effective KV cache type
    effective_kv_type = os.environ.get("OLLAMA_KV_CACHE_TYPE")
    if not effective_kv_type:
        try:
            cfg = json.loads(CONFIG_FILE.read_text())
            effective_kv_type = cfg.get("kv_cache_type")
        except Exception:
            pass
    if not effective_kv_type:
        effective_kv_type = DEFAULT_SERVER_KV_CACHE_TYPE

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
        # The backend process owns this cache; ARGUS neither places nor
        # observes its pages. Report the type actually requested of it rather
        # than describing an ARGUS tier layout that is not in play.
        "kv_cache_type": effective_kv_type,
        "kv_managed_by": "backend",
        "context_length": context_length,
        "vram": vram,
        "error_message": error_msg,
    }


#: Keys this service owns in ~/.claude/settings.json. Listed once so start and
#: stop cannot drift apart and leave a stale entry behind.
CLAUDE_SETTINGS_KEYS = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_MODEL_OPTION",
    "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME",
    "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION",
)

CLAUDE_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"


def model_label_for(llama_server_model: Optional[str]) -> str:
    """A readable name for whatever is actually serving.

    llama-server is usually pointed at an Ollama blob, whose filename is a
    sha256. Showing that in a model picker is useless, so a real GGUF filename
    is preferred and the hash degrades to a generic label rather than being
    displayed.
    """
    if not llama_server_model:
        return "local"
    stem = Path(llama_server_model).stem
    if stem.startswith("sha256-") or len(stem) >= 64:
        return "local"
    return stem


def claude_settings_env(*, port: int, model_label: str) -> Dict[str, str]:
    """The environment entries that point Claude Code at this gateway."""
    return {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
        "ANTHROPIC_CUSTOM_MODEL_OPTION": model_label,
        "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": f"ARGUS: {model_label}",
        "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": (
            "Local model served by ARGUS on this machine."
        ),
    }


#: Variables Claude Code exports so child processes know they were spawned by
#: it. A fresh session that inherits them starts in --print mode and dies on
#: "Input must be provided either through stdin or as a prompt argument".
_PARENT_SESSION_PREFIXES = ("CLAUDE_CODE", "CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT")

#: Client request timeout for a local session, in milliseconds. 15 minutes
#: covers the measured worst case: a ~57k-token Claude Code prompt prefilling
#: at roughly 135 tok/s on this hardware is about 7 minutes, and a cold model
#: load adds another minute.
LOCAL_BACKEND_TIMEOUT_MS = 900_000

#: Config profile for local sessions, kept apart from the user's own.
LOCAL_PROFILE_DIR = RUNTIME_DIR / "claude-profile"


def _seed_local_profile(config_dir: Path) -> None:
    """Create the local profile already past Claude Code's first-run setup.

    Onboarding runs once per config directory. ARGUS creates this one, so
    without seeding the user meets a theme picker and welcome screen they did
    not ask for -- which, launched from a bar, reads like a login prompt.

    Existing files are never rewritten: once the profile exists it belongs to
    the user, and a relaunch must not reset choices they have made in it.
    """
    config_dir.mkdir(parents=True, exist_ok=True)

    state = config_dir / ".claude.json"
    if not state.exists():
        state.write_text(json.dumps({"hasCompletedOnboarding": True}, indent=2))

    settings = config_dir / "settings.json"
    if not settings.exists():
        settings.write_text(json.dumps({"theme": "dark"}, indent=2))


def claude_session_environment(
    base: Dict[str, str],
    *,
    port: int,
    model_label: str,
    num_ctx: Optional[int] = None,
    config_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Environment for a single Claude Code session bound to this gateway.

    The safe way to use the local model: the redirect lives in one process's
    environment, affects nothing else on the machine, and cannot outlive that
    process. Prefer this over :func:`sync_claude_settings`, whose reach is the
    whole host.

    Markers of a *parent* Claude Code session are stripped. Launching from a
    terminal that Claude Code itself started -- or from a waybar that such a
    session spawned -- otherwise hands the new session the parent's identity,
    and it exits immediately rather than opening.
    """
    env = {
        key: value
        for key, value in base.items()
        if not key.startswith(_PARENT_SESSION_PREFIXES)
    }
    env.update(claude_settings_env(port=port, model_label=model_label))
    # A local backend prefills Claude Code's system prompt and tool schemas --
    # tens of thousands of tokens -- on CPU, which takes minutes. The default
    # client timeout cancels and retries well before that finishes, and each
    # retry restarts the prefill it just cancelled, so the session never
    # produces a first response. setdefault so an operator's own budget wins.
    env.setdefault("API_TIMEOUT_MS", str(LOCAL_BACKEND_TIMEOUT_MS))
    if config_dir is not None:
        # Plugins, skills, and MCP servers are read from the config directory
        # and rendered into every request. Measured against this project's own
        # setup they were 21k of a 57k-token prompt -- roughly two and a half
        # minutes of CPU prefill on the first turn of every session. A separate
        # profile keeps the local model lean without disturbing the user's real
        # configuration, which their cloud sessions keep using.
        _seed_local_profile(config_dir)
        env["CLAUDE_CONFIG_DIR"] = str(config_dir)
        # A fresh profile holds no credentials, so Claude Code would stop and
        # ask the user to log in. Nothing here authenticates against Anthropic
        # -- requests go to the gateway over loopback -- but the client still
        # requires *a* credential to start. Set only on the lean profile: a key
        # outranks stored OAuth credentials, so setting it on the user's own
        # profile would silently break their real login.
        env.setdefault("ANTHROPIC_API_KEY", "argus-local")
    if num_ctx is not None:
        # A model name Claude Code does not recognise makes it assume a 200k
        # window and schedule auto-compact against that. The backend serves
        # far less, so the session would overrun what llama-server can hold
        # before compaction ever fires. Stating the real figure is the
        # difference between compacting on time and failing mid-request.
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(num_ctx)
    return env


#: Where npm and pipx-style installs put the CLI. Consulted only after PATH,
#: and only because the PATH a compositor hands to waybar is not the PATH a
#: login shell builds.
_CLAUDE_FALLBACK_PATHS = (
    ".npm-global/bin/claude",
    ".local/bin/claude",
    ".bun/bin/claude",
    "node_modules/.bin/claude",
)


def resolve_claude_binary(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Absolute path to the Claude Code CLI, or None if it is not installed.

    PATH is tried first and is usually enough. It is not enough when the
    launcher is waybar: the bar is started by the window manager with whatever
    environment the compositor had at boot, which typically omits
    ``~/.npm-global/bin``. The symptom is "claude not found" from the bar while
    the identical command works in every terminal, so the well-known install
    locations are checked before giving up.
    """
    environ = os.environ if env is None else env
    found = shutil.which("claude", path=environ.get("PATH"))
    if found:
        return found

    home = environ.get("HOME")
    if not home:
        return None
    for relative in _CLAUDE_FALLBACK_PATHS:
        candidate = Path(home) / relative
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def claude_launch_argv(
    *, model_label: str, extra: Optional[list] = None, binary: str = "claude"
) -> list:
    """Command line for a session that actually runs the local model.

    ``ANTHROPIC_CUSTOM_MODEL_OPTION`` only adds an entry to the model picker.
    Without ``--model`` the session opens on its default cloud model, the
    gateway recognises that name as a cloud model and proxies it upstream, and
    the local backend sits idle while the user believes it is serving them.
    """
    forwarded = [arg for arg in (extra or []) if arg != "--"]
    return [binary, "--model", model_label, *forwarded]


def sync_claude_settings(
    *, port: int, model_label: str, settings_path: Path = None, opted_in: bool = False
) -> None:
    """Point *every* Claude Code session on this machine at the local gateway.

    ``~/.claude/settings.json`` is user-global: the security classifier, the
    auto-compact subprocess, and every unrelated project read it. Writing the
    redirect there makes one project's local model the whole machine's
    provider, and while the gateway is down every one of those sessions fails
    against a dead port.

    That blast radius is why this is opt-in (``--sync-claude``) rather than
    something ``start`` does on its own. For ordinary use prefer
    ``argus_ctl.sh claude``, which scopes the redirect to one session via
    :func:`claude_session_environment`.

    Only the keys in :data:`CLAUDE_SETTINGS_KEYS` are touched; the rest of the
    file is the user's and is written back unchanged. A missing settings file
    is left missing rather than created, since fabricating one would impose
    configuration on a machine that never asked for it.
    """
    if not opted_in:
        return
    path = CLAUDE_SETTINGS_PATH if settings_path is None else settings_path
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"   ⚠️  could not read {path}: {exc}")
        return

    data.setdefault("env", {}).update(
        claude_settings_env(port=port, model_label=model_label)
    )
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"   Claude Code -> http://127.0.0.1:{port} ({model_label})")
    except OSError as exc:
        print(f"   ⚠️  could not write {path}: {exc}")


def reconcile_claude_settings(
    *, port: int, gateway_ok: bool, settings_path: Path = None
) -> bool:
    """Drop a redirect that outlived the gateway it pointed at.

    :func:`cleanup_claude_settings` only runs on a graceful stop, so a kill
    -9, an OOM, or a reboot leaves ``ANTHROPIC_BASE_URL`` behind pointing at a
    dead port. Every Claude Code session on the machine then fails, and it
    presents as Claude Code being broken rather than as ARGUS being off.

    Status is polled continuously by the waybar module, so reconciling here
    means the stale entry clears itself within seconds of the crash without
    the user knowing any of this happened.

    Only a redirect matching *this* gateway's URL is removed: a user may point
    Claude Code at their own proxy, and that is not ours to delete.

    Returns whether anything was cleared, so callers emitting machine-readable
    output can stay silent -- waybar parses ``status --json`` and a stray line
    would corrupt it.
    """
    if gateway_ok:
        return False
    path = CLAUDE_SETTINGS_PATH if settings_path is None else settings_path
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    env = data.get("env")
    if not isinstance(env, dict):
        return False
    if env.get("ANTHROPIC_BASE_URL") != f"http://127.0.0.1:{port}":
        return False

    cleanup_claude_settings(settings_path=path)
    return True


def cmd_claude(args):
    """Run one Claude Code session against the gateway, changing nothing else.

    The redirect is passed in this process's environment and inherited only by
    the session being launched, so no other Claude Code process on the machine
    is affected and nothing is left behind if this one dies.
    """
    info = get_detailed_status(args.port, args.upstream)
    if not info["gateway_ok"]:
        print(f"❌ No gateway on port {args.port}. Start it with `argus_ctl.sh start`.")
        return 1

    binary = resolve_claude_binary()
    if binary is None:
        print(
            "❌ Claude Code CLI not found. Looked on PATH and in "
            + ", ".join(f"~/{p}" for p in _CLAUDE_FALLBACK_PATHS)
            + ".\n   Install it, or launch with: "
            f"ANTHROPIC_BASE_URL=http://127.0.0.1:{args.port} claude"
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
    argv = claude_launch_argv(
        model_label=label, extra=getattr(args, "claude_args", []), binary=binary
    )
    profile = "your profile" if getattr(args, "with_plugins", False) else "lean profile"
    print(
        f"🚀 Claude Code -> http://127.0.0.1:{args.port} "
        f"({label}, {profile}, this session only)"
    )
    try:
        return subprocess.call(argv, env=env, cwd=workspace or None)
    except FileNotFoundError:
        print(f"❌ could not execute {binary}")
        return 1


def cmd_status(args):
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
    print(f"  KV Cache Type:      {info['kv_cache_type']} (owned by {info['kv_managed_by']})")
    ctx = info["context_length"]
    print(f"  Context Length:     {ctx:,}" if ctx else "  Context Length:     (no model loaded)")
    
    v = info["vram"]
    if v["available"]:
        print(f"  GPU Hardware:       {v['gpu_name']}")
        print(f"  VRAM Allocation:    {v['used_mib']} MiB / {v['total_mib']} MiB ({v['utilization_pct']}%)")
    else:
        print("  GPU Hardware:       CPU Mode")

    if info["error_message"]:
        print(f"  Notice:             ⚠️  {info['error_message']}")
    if healed:
        print("  Notice:             ℹ️  cleared a stale Claude Code redirect")

    print("=" * 60 + "\n")


CONFIG_FILE = RUNTIME_DIR / "config.json"

#: Settings a launcher may persist, and the value used when neither the config
#: nor a flag supplies one.
_CONFIG_DEFAULTS = {
    "llama_server_model": None,
    "llama_server_port": 8080,
    "num_ctx": DEFAULT_LLAMA_SERVER_CTX,
    # Matches llama_server_command: the measured rung, not the unmeasured one.
    "kv_cache_type": DEFAULT_SERVER_LLAMA_KV_TYPE,
    "spec_type": "ngram-mod",
    "spec_draft_n_max": 2,
    "draft_model": None,
    "flash_attn": "on",
    "load_mode": None,
    "spec_draft_p_min": None,
}

#: Config keys whose name differs from the argument they fill. ``workspace``
#: reads better in a config file than ``cwd`` does, and waybar has no way to
#: pass a directory on the command line.
_CONFIG_ALIASES = {"workspace": "cwd"}


def apply_config(args, config_path: Path = None):
    """Fill unset arguments from the persisted config, then from defaults.

    Waybar and similar launchers invoke ``start`` with no arguments and discard
    the output, so without this the click quietly selects the untuned backend:
    Ollama cannot be told ``--cpu-moe``, which is the difference between 12.09
    and 19.14 tok/s on this hardware. Persisting the choice is what makes the
    button and the command line behave the same.

    Precedence is flag > config > default. A missing config is ordinary; an
    unreadable one degrades to defaults rather than turning a button press into
    a silent no-op.
    """
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
    """Whether a llama-server is already serving on ``port``."""
    try:
        with httpx.Client(timeout=1.0) as client:
            return client.get(f"http://127.0.0.1:{port}/health").status_code == 200
    except httpx.HTTPError:
        return False


def _start_llama_server(args) -> None:
    """Launch llama-server with the tuned flags and wait for it to answer.

    The CUDA backend ships in Ollama's private lib directory, so the loader
    path has to be set explicitly; without it llama-server reports "no usable
    GPU found" and silently runs on CPU, which costs far more than the flags
    gain.

    Startup is slow -- the model is over 20 GB -- so this polls rather than
    sleeping a fixed interval, and reports failure instead of leaving the
    caller to discover a dead backend later.
    """
    spec_type = getattr(args, "spec_type", None)
    spec_draft_n_max = getattr(args, "spec_draft_n_max", 2)
    draft_model = getattr(args, "draft_model", None)
    flash_attn = getattr(args, "flash_attn", "on")
    load_mode = getattr(args, "load_mode", None)
    spec_draft_p_min = getattr(args, "spec_draft_p_min", None)

    cmd = llama_server_command(
        model_path=args.llama_server_model,
        port=args.llama_server_port,
        num_ctx=args.num_ctx,
        kv_cache_type=args.kv_cache_type,
        spec_type=spec_type,
        spec_draft_n_max=spec_draft_n_max,
        draft_model=draft_model,
        flash_attn=flash_attn,
        load_mode=load_mode,
        spec_draft_p_min=spec_draft_p_min,
    )
    spec_info = f", spec {spec_type}" if spec_type else ""
    lm_info = f", load-mode {load_mode}" if load_mode else ""
    pmin_info = f", p-min {spec_draft_p_min}" if spec_draft_p_min is not None else ""
    print(f"🔄 Starting llama-server (ctx {args.num_ctx:,}, KV {args.kv_cache_type}{spec_info}, FA {flash_attn}{lm_info}{pmin_info}, experts on CPU)...")

    env = dict(os.environ)
    cuda_dir = Path(LLAMA_CUDA_DIR)
    if cuda_dir.is_dir():
        env["LD_LIBRARY_PATH"] = f"{cuda_dir}:{cuda_dir.parent}:" + env.get("LD_LIBRARY_PATH", "")
        env["GGML_BACKEND_PATH"] = str(cuda_dir / "libggml-cuda.so")
    else:
        print(f"   ⚠️  {cuda_dir} not found; llama-server will run without CUDA.")

    log = open(RUNTIME_DIR / "llama-server.log", "a")
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env
    )
    BACKEND_PID_FILE.write_text(str(proc.pid))

    health = f"http://127.0.0.1:{args.llama_server_port}/health"
    for _ in range(args.llama_server_timeout):
        if proc.poll() is not None:
            print(f"   ❌ llama-server exited during load; see {RUNTIME_DIR / 'llama-server.log'}")
            return
        try:
            with httpx.Client(timeout=1.0) as client:
                if client.get(health).status_code == 200:
                    print("   ✅ llama-server ready.")
                    return
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    print(f"   ⚠️  llama-server did not answer within {args.llama_server_timeout}s.")


def cmd_start(args):
    ensure_runtime_dir()
    apply_config(args)
    info = get_detailed_status(args.port, args.upstream)

    if info["state"] == "READY":
        print(f"⚠️  ARGUS Gateway is already running and READY (Port: {args.port}).")
        return

    # Bring up a backend if none is answering. llama-server is preferred when
    # a model path was given: it is the only one that accepts --cpu-moe, which
    # was measured at 12.09 -> 19.14 tok/s on this class of GPU.
    backend_ok, _ = check_backend_health(args.upstream)
    # An explicit model path wins over an already-healthy backend. Otherwise a
    # stray Ollama daemon silently satisfies the health check and the tuned
    # llama-server -- the entire reason the flag was passed -- never starts.
    if getattr(args, "llama_server_model", None):
        if not _llama_server_answering(args.llama_server_port):
            _start_llama_server(args)
        backend_ok, _ = check_backend_health(args.upstream)
    if not backend_ok:
        try:
            # Check if ollama binary exists
            ollama_path = subprocess.run(["which", "ollama"], capture_output=True, text=True).stdout.strip()
            if ollama_path:
                print("🔄 Starting Ollama backend service...")
                b_log = open(RUNTIME_DIR / "ollama.log", "a")
                # KV quantization is a startup-only knob: a server booted
                # without it cannot be retuned per request. Skipping it costs
                # roughly 3x decode throughput at long context on a small GPU.
                backend_env = recommended_server_environment(os.environ)
                print(
                    f"   KV cache: {backend_env['OLLAMA_KV_CACHE_TYPE']}, "
                    f"flash attention: {backend_env['OLLAMA_FLASH_ATTENTION']}"
                )
                b_proc = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=b_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=backend_env,
                )
                BACKEND_PID_FILE.write_text(str(b_proc.pid))
                time.sleep(1.0)
        except Exception:
            pass

    print(f"🚀 Starting ARGUS Gateway on port {args.port}...")
    log_fp = open(LOG_FILE, "w")


    cmd = [
        sys.executable,
        "-m",
        "argus_cache.adapters.claude_gateway",
        "--port",
        str(args.port),
        "--upstream",
        args.upstream,
        # Requesting a window the backend was not started with just wastes it,
        # so the gateway is told the same number llama-server was given.
        "--num-ctx",
        str(args.num_ctx),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    PID_FILE.write_text(str(proc.pid))

    # Wait and probe
    for _ in range(30):
        time.sleep(0.1)
        st = get_detailed_status(args.port, args.upstream)
        if st["gateway_ok"]:
            break

    # Machine-wide redirect only on explicit request: see sync_claude_settings
    # for why starting a service is not consent to repoint every session on
    # the host.
    sync_claude_settings(
        port=args.port,
        model_label=model_label_for(getattr(args, "llama_server_model", None)),
        opted_in=getattr(args, "sync_claude", False),
    )

    final_st = get_detailed_status(args.port, args.upstream)
    print(f"✅ Service State: {final_st['status_text']}")
    print(f"   Gateway: http://127.0.0.1:{args.port}")
    print(f"   Logs:    {LOG_FILE}")
    if not getattr(args, "sync_claude", False):
        print(f"   Claude Code: run `argus_ctl.sh claude` to use this gateway")


def cleanup_claude_settings(*, settings_path: Path = None) -> None:
    """Hand Claude Code back to its normal providers.

    Symmetric with :func:`sync_claude_settings` by construction: both work from
    :data:`CLAUDE_SETTINGS_KEYS`, so a key added at start cannot be forgotten
    at stop. A leftover ANTHROPIC_BASE_URL points Claude Code at a dead port,
    which presents as Claude Code being broken rather than ARGUS being off.

    An ``env`` block emptied by this removal is dropped, so stopping restores
    the file to what it was before starting.
    """
    path = CLAUDE_SETTINGS_PATH if settings_path is None else settings_path
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    env = data.get("env")
    if not isinstance(env, dict):
        return
    removed = [key for key in CLAUDE_SETTINGS_KEYS if env.pop(key, None) is not None]
    if not removed:
        return
    if not env:
        data.pop("env", None)
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def cmd_stop(args):
    pid = get_stored_pid(PID_FILE)
    print(f"🛑 Stopping ARGUS Gateway (PID: {pid or 'all'})...")

    # 1. Unload models from Ollama to instantly free GPU VRAM & RAM
    try:
        with httpx.Client(timeout=1.5) as client:
            resp = client.get("http://127.0.0.1:11434/api/tags")
            if resp.status_code == 200:
                for m in resp.json().get("models", []):
                    m_name = m.get("name")
                    if m_name:
                        try:
                            client.post("http://127.0.0.1:11434/api/generate", json={"model": m_name, "keep_alive": 0})
                        except Exception:
                            pass
    except Exception:
        pass

    # 2. Stop spawned backend process if we started it
    backend_pid = get_stored_pid(BACKEND_PID_FILE)
    if backend_pid:
        try:
            os.kill(backend_pid, signal.SIGTERM)
        except OSError:
            pass
        BACKEND_PID_FILE.unlink(missing_ok=True)

    # 3. Stop Gateway daemon
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(15):
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
        except OSError:
            pass

    PID_FILE.unlink(missing_ok=True)
    subprocess.run(["fuser", "-k", f"{args.port}/tcp"], capture_output=True)
    cleanup_claude_settings()
    time.sleep(0.2)
    print("✅ ARGUS Gateway stopped (Offline). VRAM released. Claude Code restored to direct cloud.")


def cmd_restart(args):
    cmd_stop(args)
    time.sleep(0.5)
    cmd_start(args)


def main():
    parser = argparse.ArgumentParser(description="ARGUS AI Service & Claude Code Gateway Controller")
    parser.add_argument("--port", type=int, default=DEFAULT_GATEWAY_PORT, help="Gateway port (default: 8000)")
    parser.add_argument("--upstream", type=str, default=DEFAULT_UPSTREAM_URL, help="Upstream LLM endpoint (default: http://127.0.0.1:8080/v1)")
    parser.add_argument("--json", action="store_true", help="Output status in JSON format")

    subparsers = parser.add_subparsers(dest="action", required=True)
    start_parser = subparsers.add_parser("start", help="Start ARGUS Gateway and backend")
    # Backend options belong to `start`, so they can follow the subcommand the
    # way an operator naturally types them.
    start_parser.add_argument(
        "--llama-server-model",
        type=str,
        default=None,
        help=(
            "Path to a GGUF. When set, start llama-server instead of Ollama -- "
            "the only backend that accepts --cpu-moe, worth ~1.6x decode on a "
            "small GPU."
        ),
    )
    start_parser.add_argument("--llama-server-port", type=int, default=None)
    start_parser.add_argument(
        "--llama-server-timeout", type=int, default=300,
        help="Seconds to wait for llama-server to answer /health.",
    )
    start_parser.add_argument(
        "--num-ctx", type=int, default=None,
        help=f"Context window for llama-server (default: {DEFAULT_LLAMA_SERVER_CTX}).",
    )
    start_parser.add_argument(
        "--kv-cache-type", type=str, default=None,
        help=(
            "KV cache type for llama-server. iq4_nl reached 224k context at "
            "16.64 tok/s where q4_0 topped out at 160k (default: iq4_nl)."
        ),
    )
    start_parser.add_argument(
        "--spec-type", type=str, default=None,
        help="Speculative decoding type (e.g. 'ngram-mod', 'draft-mtp,ngram-mod', 'draft-simple').",
    )
    start_parser.add_argument(
        "--spec-draft-n-max", type=int, default=None,
        help="Maximum speculative draft tokens (default: 2).",
    )
    start_parser.add_argument(
        "--draft-model", type=str, default=None,
        help="Draft model path for speculative decoding (when using draft-simple).",
    )
    start_parser.add_argument(
        "--flash-attn",
        type=str,
        choices=["on", "off", "auto"],
        default=None,
        help="Flash attention mode for llama-server ('on', 'off', 'auto', default: 'on').",
    )
    start_parser.add_argument(
        "--load-mode",
        type=str,
        choices=["auto", "none", "mmap", "mlock", "mmap+mlock"],
        default=None,
        help="Model loading mode ('auto', 'none', 'mmap', 'mlock', 'mmap+mlock').",
    )
    start_parser.add_argument(
        "--spec-draft-p-min",
        type=float,
        default=None,
        help="Minimum speculative decoding probability (default: 0.00).",
    )
    start_parser.add_argument(
        "--sync-claude",
        action="store_true",
        help=(
            "Also repoint EVERY Claude Code session on this machine at the "
            "gateway by writing ~/.claude/settings.json. Off by default: while "
            "the gateway is down those sessions fail against a dead port. "
            "Prefer `argus_ctl.sh claude` for a single scoped session."
        ),
    )
    claude_parser = subparsers.add_parser(
        "claude",
        help="Launch one Claude Code session bound to the gateway (writes nothing)",
    )
    claude_parser.add_argument("--llama-server-model", type=str, default=None)
    claude_parser.add_argument(
        "--with-plugins",
        action="store_true",
        help=(
            "Use your own Claude Code profile instead of ARGUS's lean one. "
            "Restores plugins, skills, and MCP servers -- and the ~21k tokens "
            "they add to every request, which on a local backend is about two "
            "and a half extra minutes of prefill on the first turn."
        ),
    )
    claude_parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help=(
            "Directory to open the session in. Defaults to the `workspace` key "
            "in the service config, else the current directory. waybar cannot "
            "pass one, so set `workspace` to control what the click opens."
        ),
    )
    claude_parser.add_argument(
        "claude_args", nargs=argparse.REMAINDER, help="Arguments passed to claude"
    )
    subparsers.add_parser("stop", help="Stop ARGUS Gateway")
    status_parser = subparsers.add_parser("status", help="Get operational status and telemetry")
    status_parser.add_argument("--json", action="store_true", help="Output status in JSON format")
    subparsers.add_parser("restart", help="Restart ARGUS service")

    args = parser.parse_args()
    actions = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "claude": cmd_claude,
        "restart": cmd_restart,
    }
    actions[args.action](args)


if __name__ == "__main__":
    main()
