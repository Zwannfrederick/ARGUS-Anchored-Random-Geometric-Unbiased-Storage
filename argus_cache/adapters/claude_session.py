"""Claude Code local session & settings handoff for ARGUS Gateway."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

RUNTIME_DIR = Path.home() / ".argus_runtime"

#: Keys this service owns in ~/.claude/settings.json. Listed once so start and
#: stop cannot drift apart and leave a stale entry behind.
CLAUDE_SETTINGS_KEYS = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_MODEL_OPTION",
    "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME",
    "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION",
    "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY",
)

ARGUS_PICKER_PREFIX = "ARGUS: "
ARGUS_MODEL_DESCRIPTION = "Local model served by ARGUS on this machine."

CLAUDE_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

#: Variables Claude Code exports so child processes know they were spawned by
#: it. A fresh session that inherits them starts in --print mode and dies on
#: "Input must be provided either through stdin or as a prompt argument".
_PARENT_SESSION_PREFIXES = ("CLAUDE_CODE", "CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT")

#: Client request timeout for a local session, in milliseconds. 15 minutes
#: covers worst-case large prompt prefill plus cold model load.
LOCAL_BACKEND_TIMEOUT_MS = 900_000

#: Config profile for local sessions, kept apart from the user's own.
LOCAL_PROFILE_DIR = RUNTIME_DIR / "claude-profile"

#: Where npm and pipx-style installs put the CLI. Consulted only after PATH,
#: because a compositor-launched waybar often omits ~/.npm-global/bin.
_CLAUDE_FALLBACK_PATHS = (
    ".npm-global/bin/claude",
    ".local/bin/claude",
    ".bun/bin/claude",
    "node_modules/.bin/claude",
)

#: Bundled native CLI inside VS Code/Cursor extensions and Claude Desktop.
_CLAUDE_FALLBACK_GLOBS = (
    ".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude",
    ".cursor/extensions/anthropic.claude-code-*/resources/native-binary/claude",
    ".antigravity/extensions/anthropic.claude-code-*/resources/native-binary/claude",
    ".config/Claude/claude-code/*/claude",
)


def model_label_for(llama_server_model: Optional[str]) -> str:
    """A readable name for whatever is actually serving."""
    if not llama_server_model:
        return "local"
    stem = Path(llama_server_model).stem
    if stem.startswith("sha256-") or len(stem) >= 64:
        return "local"
    return stem


def claude_model_picker_option(model_label: str) -> Dict[str, str]:
    """Structured /model picker entry for Claude Code settings.json."""
    return {
        "model": model_label,
        "label": f"{ARGUS_PICKER_PREFIX}{model_label}",
        "description": ARGUS_MODEL_DESCRIPTION,
        "behavesAs": "claude-sonnet-4-6",
    }


def _upsert_model_picker(data: Dict[str, object], model_label: str) -> None:
    picker = data.get("modelPicker")
    if not isinstance(picker, dict):
        picker = {"options": [], "replaceBuiltInOptions": False}
        data["modelPicker"] = picker
    existing = picker.get("options") if isinstance(picker.get("options"), list) else []
    filtered = [
        opt for opt in existing
        if not (isinstance(opt, dict) and str(opt.get("label", "")).startswith(ARGUS_PICKER_PREFIX))
    ]
    filtered.append(claude_model_picker_option(model_label))
    picker["options"] = filtered
    picker.setdefault("replaceBuiltInOptions", False)


def claude_settings_env(*, port: int, model_label: str) -> Dict[str, str]:
    """The environment entries that point Claude Code at this gateway."""
    return {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
        "ANTHROPIC_CUSTOM_MODEL_OPTION": model_label,
        "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": f"{ARGUS_PICKER_PREFIX}{model_label}",
        "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": ARGUS_MODEL_DESCRIPTION,
    }


def _seed_local_profile(config_dir: Path, *, model_label: Optional[str] = None) -> None:
    """Create the local profile already past Claude Code's first-run setup."""
    config_dir.mkdir(parents=True, exist_ok=True)

    state = config_dir / ".claude.json"
    if not state.exists():
        state.write_text(
            json.dumps(
                {
                    "hasCompletedOnboarding": True,
                    "customApiKeyResponses": {"approved": ["argus-local"], "rejected": []},
                },
                indent=2,
            )
        )

    settings = config_dir / "settings.json"
    settings_data: Dict[str, object] = {"theme": "dark"}
    if settings.exists():
        try:
            loaded = json.loads(settings.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                settings_data = loaded
        except (OSError, json.JSONDecodeError):
            pass
    if model_label:
        _upsert_model_picker(settings_data, model_label)
    if not settings.exists() or model_label:
        settings.write_text(json.dumps(settings_data, indent=2), encoding="utf-8")


def claude_session_environment(
    base: Dict[str, str],
    *,
    port: int,
    model_label: str,
    num_ctx: Optional[int] = None,
    config_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Environment for a single Claude Code session bound to this gateway."""
    env = {
        key: value
        for key, value in base.items()
        if not key.startswith(_PARENT_SESSION_PREFIXES)
    }
    env.update(claude_settings_env(port=port, model_label=model_label))
    env.setdefault("ANTHROPIC_SMALL_FAST_MODEL", model_label)
    env.setdefault("ANTHROPIC_DEFAULT_HAIKU_MODEL", model_label)
    env.setdefault("API_TIMEOUT_MS", str(LOCAL_BACKEND_TIMEOUT_MS))
    if config_dir is not None:
        _seed_local_profile(config_dir, model_label=model_label)
        env["CLAUDE_CONFIG_DIR"] = str(config_dir)
        env.setdefault("ANTHROPIC_API_KEY", "argus-local")
    if num_ctx is not None:
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(num_ctx)
        env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", str(min(16384, max(4096, num_ctx // 8))))
    return env


def resolve_claude_binary(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Absolute path to the Claude Code CLI, or None if it is not installed."""
    environ = os.environ if env is None else env
    found = shutil.which("claude", path=environ.get("PATH"))
    if found:
        return found

    home = environ.get("HOME")
    if not home:
        return None
    home_path = Path(home)
    for relative in _CLAUDE_FALLBACK_PATHS:
        candidate = home_path / relative
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    for pattern in _CLAUDE_FALLBACK_GLOBS:
        for candidate in sorted(home_path.glob(pattern), reverse=True):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def claude_launch_argv(
    *, model_label: str, extra: Optional[list] = None, binary: str = "claude"
) -> list:
    """Command line for a session that actually runs the local model."""
    forwarded = [arg for arg in (extra or []) if arg != "--"]
    return [binary, "--model", model_label, *forwarded]


def sync_claude_settings(
    *, port: int, model_label: str, settings_path: Path = None, opted_in: bool = False
) -> None:
    """Point every Claude Code session on this machine at the local gateway (opt-in)."""
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

    env_block = data.setdefault("env", {})
    env_block.update(claude_settings_env(port=port, model_label=model_label))
    env_block["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
    _upsert_model_picker(data, model_label)
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"   Claude Code -> http://127.0.0.1:{port} ({model_label})")
    except OSError as exc:
        print(f"   ⚠️  could not write {path}: {exc}")


def cleanup_claude_settings(*, settings_path: Path = None) -> None:
    """Hand Claude Code back to its normal providers."""
    path = CLAUDE_SETTINGS_PATH if settings_path is None else settings_path
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    changed = False
    env = data.get("env")
    if isinstance(env, dict):
        removed = [key for key in CLAUDE_SETTINGS_KEYS if env.pop(key, None) is not None]
        if removed:
            changed = True
            if not env:
                data.pop("env", None)

    picker = data.get("modelPicker")
    if isinstance(picker, dict) and isinstance(picker.get("options"), list):
        kept = [
            opt for opt in picker["options"]
            if not (isinstance(opt, dict) and str(opt.get("label", "")).startswith(ARGUS_PICKER_PREFIX))
        ]
        if len(kept) != len(picker["options"]):
            changed = True
            if kept:
                picker["options"] = kept
            elif set(picker.keys()) <= {"options", "replaceBuiltInOptions"} and not picker.get("replaceBuiltInOptions"):
                data.pop("modelPicker", None)
            else:
                picker["options"] = []

    if not changed:
        return
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def reconcile_claude_settings(
    *, port: int, gateway_ok: bool, settings_path: Path = None
) -> bool:
    """Drop a redirect that outlived the gateway it pointed at."""
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

