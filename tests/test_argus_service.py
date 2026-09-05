"""Service configuration resolution.

The waybar module and any other launcher that cannot pass arguments call
``argus_ctl.sh start`` bare. Without a persisted configuration that invocation
silently takes the untuned path -- Ollama instead of llama-server -- which
gives up the largest measured speedup (12.09 -> 19.14 tok/s from ``--cpu-moe``)
precisely when a user clicks the button rather than typing a command.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "argus_service",
    Path(__file__).resolve().parent.parent / "scripts" / "argus_service.py",
)
argus_service = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(argus_service)


class _Args:
    """Stand-in for the argparse namespace, with everything unset."""

    def __init__(self, **kwargs):
        self.llama_server_model = None
        self.num_ctx = None
        self.kv_cache_type = None
        self.llama_server_port = None
        self.cwd = None
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_config_supplies_settings_when_no_flags_were_given(tmp_path):
    """A bare `start` picks up the persisted backend, which is what waybar does."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "llama_server_model": "/models/qwen.gguf",
        "num_ctx": 229376,
        "kv_cache_type": "iq4_nl",
    }))

    args = argus_service.apply_config(_Args(), config_path=config)

    assert args.llama_server_model == "/models/qwen.gguf"
    assert args.num_ctx == 229376
    assert args.kv_cache_type == "iq4_nl"


def test_explicit_flags_win_over_the_config(tmp_path):
    """Typing a flag must override the stored value, not be silently ignored."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"num_ctx": 229376, "kv_cache_type": "iq4_nl"}))

    args = argus_service.apply_config(
        _Args(num_ctx=32768, kv_cache_type="q8_0"), config_path=config
    )

    assert args.num_ctx == 32768
    assert args.kv_cache_type == "q8_0"


def test_absent_config_leaves_built_in_defaults(tmp_path):
    """No config is a normal state, not an error."""
    args = argus_service.apply_config(_Args(), config_path=tmp_path / "nope.json")

    assert args.llama_server_model is None
    assert args.num_ctx == argus_service.DEFAULT_LLAMA_SERVER_CTX


def test_unreadable_config_does_not_prevent_startup(tmp_path):
    """A corrupt config must degrade to defaults rather than crash the service.

    This runs from a window-manager click with output discarded, so an
    exception here would look like the button simply not working.
    """
    config = tmp_path / "config.json"
    config.write_text("{ this is not json")

    args = argus_service.apply_config(_Args(), config_path=config)

    assert args.num_ctx == argus_service.DEFAULT_LLAMA_SERVER_CTX


def test_config_rejects_an_unusable_kv_type(tmp_path):
    """A typo in the stored KV type is reported, not carried into llama-server.

    llama.cpp fails late and obscurely on an unknown cache type, so the value
    is checked where it can still be attributed to the config file.
    """
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"kv_cache_type": "q3_k"}))

    with pytest.raises(ValueError, match="kv_cache_type"):
        argus_service.apply_config(_Args(), config_path=config)


# ── Claude Code settings handoff ────────────────────────────────────────────


def test_settings_point_claude_code_at_the_gateway():
    """Claude Code reaches the local model only through ANTHROPIC_BASE_URL."""
    env = argus_service.claude_settings_env(port=8000, model_label="qwen3.6-35b-a3b")

    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"


def test_settings_name_the_model_that_is_actually_serving():
    """The label follows the running backend, not whatever Ollama happens to hold.

    The previous version read Ollama's first tag even when llama-server was the
    backend, so the name shown could belong to a model that was not answering.
    """
    env = argus_service.claude_settings_env(port=8000, model_label="qwen3.6-35b-a3b")

    assert "qwen3.6-35b-a3b" in env["ANTHROPIC_CUSTOM_MODEL_OPTION"]


def test_model_label_is_derived_from_a_llama_server_path():
    """A GGUF path becomes a readable name rather than a blob hash."""
    label = argus_service.model_label_for(
        llama_server_model="/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    )

    assert label == "Qwen3.6-35B-A3B-UD-Q4_K_M"


def test_model_label_falls_back_when_no_model_is_configured():
    assert argus_service.model_label_for(llama_server_model=None) == "local"


def test_start_then_stop_restores_the_original_settings(tmp_path):
    """Turning ARGUS off must leave settings byte-for-byte as they were.

    A stale ANTHROPIC_BASE_URL after shutdown silently points Claude Code at a
    dead port, which looks like Claude Code itself being broken.
    """
    settings = tmp_path / "settings.json"
    original = {"env": {"EDITOR": "vim"}, "theme": "dark"}
    settings.write_text(json.dumps(original, indent=2))

    argus_service.sync_claude_settings(
        port=8000,
        model_label="qwen3.6-35b-a3b",
        settings_path=settings,
        opted_in=True,
    )
    after_start = json.loads(settings.read_text())
    assert after_start["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"

    argus_service.cleanup_claude_settings(settings_path=settings)

    assert json.loads(settings.read_text()) == original


def test_sync_preserves_unrelated_settings(tmp_path):
    """Writing our keys must not disturb the user's own configuration."""
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"env": {"EDITOR": "vim"}, "model": "opus"}))

    argus_service.sync_claude_settings(
        port=8000, model_label="x", settings_path=settings, opted_in=True
    )

    data = json.loads(settings.read_text())
    assert data["env"]["EDITOR"] == "vim"
    assert data["model"] == "opus"


def test_sync_without_a_settings_file_is_a_noop(tmp_path):
    """A machine with no Claude settings is not an error state."""
    missing = tmp_path / "nope.json"

    argus_service.sync_claude_settings(
        port=8000, model_label="x", settings_path=missing, opted_in=True
    )

    assert not missing.exists()


# ── self-healing after an unclean shutdown ──────────────────────────────────


def test_stale_redirect_is_removed_when_the_gateway_is_gone(tmp_path):
    """A crash leaves the redirect behind; the next status poll must clear it.

    ``cleanup_claude_settings`` only runs on a graceful stop. After a kill -9,
    an OOM, or a reboot the redirect survives and every Claude Code session on
    the machine points at a dead port -- which presents as Claude Code being
    broken rather than as ARGUS being down.
    """
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"env": {"EDITOR": "vim"}}))
    argus_service.sync_claude_settings(
        port=8000, model_label="x", settings_path=settings, opted_in=True
    )

    argus_service.reconcile_claude_settings(
        port=8000, gateway_ok=False, settings_path=settings
    )

    assert json.loads(settings.read_text()) == {"env": {"EDITOR": "vim"}}


def test_live_gateway_keeps_its_redirect(tmp_path):
    """Reconciliation must not fight a healthy gateway."""
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"env": {}}))
    argus_service.sync_claude_settings(
        port=8000, model_label="x", settings_path=settings, opted_in=True
    )

    argus_service.reconcile_claude_settings(
        port=8000, gateway_ok=True, settings_path=settings
    )

    env = json.loads(settings.read_text())["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"


def test_a_redirect_we_do_not_own_is_left_alone(tmp_path):
    """Someone else's ANTHROPIC_BASE_URL is not ours to delete.

    A user may point Claude Code at their own proxy or a colleague's host.
    Reconciliation keys off the exact gateway URL so an unrelated redirect
    survives ARGUS being down.
    """
    settings = tmp_path / "settings.json"
    foreign = {"env": {"ANTHROPIC_BASE_URL": "https://proxy.example.com"}}
    settings.write_text(json.dumps(foreign))

    argus_service.reconcile_claude_settings(
        port=8000, gateway_ok=False, settings_path=settings
    )

    assert json.loads(settings.read_text()) == foreign


def test_reconciliation_without_a_settings_file_is_a_noop(tmp_path):
    argus_service.reconcile_claude_settings(
        port=8000, gateway_ok=False, settings_path=tmp_path / "nope.json"
    )


# ── global settings are opt-in ──────────────────────────────────────────────


def test_start_does_not_redirect_every_session_by_default(tmp_path):
    """Starting ARGUS must not repoint Claude Code machine-wide.

    ``~/.claude/settings.json`` is read by every Claude Code process on the
    host, including the security classifier and the auto-compact subprocess.
    Writing ANTHROPIC_BASE_URL there on start means one project's local model
    hijacks unrelated sessions, and the moment the gateway is down -- a crash,
    a restart, a test -- all of them fail against a dead port.
    """
    settings = tmp_path / "settings.json"
    original = {"env": {"EDITOR": "vim"}}
    settings.write_text(json.dumps(original))

    argus_service.sync_claude_settings(
        port=8000,
        model_label="x",
        settings_path=settings,
        opted_in=False,
    )

    assert json.loads(settings.read_text()) == original


def test_explicit_opt_in_still_redirects(tmp_path):
    """A user who asks for the machine-wide redirect still gets it."""
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"env": {}}))

    argus_service.sync_claude_settings(
        port=8000,
        model_label="x",
        settings_path=settings,
        opted_in=True,
    )

    env = json.loads(settings.read_text())["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"


def test_session_environment_points_at_the_gateway_without_touching_disk(tmp_path):
    """The isolated launcher path configures one session and writes nothing.

    This is the safe way to use the local model: scoped to the process that
    asked for it, and it cannot outlive that process.
    """
    settings = tmp_path / "settings.json"
    original = {"env": {"EDITOR": "vim"}}
    settings.write_text(json.dumps(original))

    env = argus_service.claude_session_environment(
        {"PATH": "/usr/bin"}, port=8000, model_label="qwen"
    )

    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"
    assert env["PATH"] == "/usr/bin"
    assert json.loads(settings.read_text()) == original


def test_session_environment_drops_parent_session_markers():
    """A session launched from inside Claude Code must not inherit its markers.

    Claude Code advertises itself to child processes via CLAUDECODE and the
    CLAUDE_CODE_* family. A new session that inherits them believes it is a
    nested tool invocation and starts in --print mode, which fails immediately
    with "Input must be provided either through stdin or as a prompt
    argument". Launching from waybar inherits the same variables whenever the
    bar itself was started from such a session.
    """
    base = {
        "PATH": "/usr/bin",
        "CLAUDECODE": "1",
        "CLAUDE_CODE_ENTRYPOINT": "cli",
        "CLAUDE_CODE_SESSION_ID": "abc",
        "CLAUDE_PID": "123",
        "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
    }

    env = argus_service.claude_session_environment(base, port=8000, model_label="qwen")

    assert env["PATH"] == "/usr/bin"
    assert not [k for k in env if k.startswith("CLAUDE_CODE")]
    assert "CLAUDECODE" not in env
    assert "CLAUDE_PID" not in env
    # Ours replaces whatever the parent had pointed at.
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"


def test_session_selects_the_local_model():
    """The launched session must start on the local model, not merely offer it.

    ANTHROPIC_CUSTOM_MODEL_OPTION adds an entry to the model picker; it does
    not select one. Without an explicit --model the session opens on its
    default cloud model, the gateway recognises the name as a cloud model and
    proxies it upstream -- so the local backend sits idle while the user
    believes they are running it.
    """
    argv = argus_service.claude_launch_argv(model_label="qwen3.6-35b-a3b")

    assert argv[0] == "claude"
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "qwen3.6-35b-a3b"


def test_extra_arguments_are_forwarded_after_the_model():
    argv = argus_service.claude_launch_argv(
        model_label="qwen", extra=["--", "--resume"]
    )

    assert argv[-1] == "--resume"
    assert "--" not in argv


def test_claude_binary_is_found_on_the_path(tmp_path):
    """The ordinary case: the launcher honours PATH."""
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)

    found = argus_service.resolve_claude_binary({"PATH": str(tmp_path)})

    assert found == str(binary)


def test_claude_binary_is_found_when_the_path_is_a_gui_stub(tmp_path):
    """waybar is started by the window manager, not by a login shell.

    Its PATH is whatever the compositor inherited at boot, which typically
    omits ~/.npm-global/bin and ~/.local/bin. Resolving through PATH alone
    reports "claude not found" from the bar while the same command works in
    any terminal.
    """
    home = tmp_path / "home"
    (home / ".npm-global" / "bin").mkdir(parents=True)
    binary = home / ".npm-global" / "bin" / "claude"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)

    found = argus_service.resolve_claude_binary(
        {"PATH": "/usr/bin:/bin", "HOME": str(home)}
    )

    assert found == str(binary)


def test_missing_claude_reports_absence_rather_than_guessing(tmp_path):
    assert argus_service.resolve_claude_binary(
        {"PATH": str(tmp_path), "HOME": str(tmp_path)}
    ) is None


def test_launch_argv_uses_the_resolved_binary():
    argv = argus_service.claude_launch_argv(
        model_label="qwen", binary="/opt/claude/bin/claude"
    )

    assert argv[0] == "/opt/claude/bin/claude"


def test_session_declares_the_real_context_window():
    """Claude Code must be told the window the backend actually has.

    An unrecognised model name makes Claude Code assume a 200k window and
    schedule auto-compact against it. The backend here serves 131072, so the
    session would keep filling past what llama-server can hold and the request
    fails before compaction ever triggers. Declaring the real number moves
    compaction to the right place.
    """
    env = argus_service.claude_session_environment(
        {}, port=8000, model_label="local", num_ctx=131072
    )

    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"


def test_context_window_is_not_declared_when_unknown():
    """Guessing a window would be worse than letting Claude Code decide."""
    env = argus_service.claude_session_environment({}, port=8000, model_label="local")

    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in env


def test_workspace_comes_from_the_config_when_no_flag_is_given(tmp_path):
    """waybar cannot pass a directory, so the stored one decides.

    Launched from a bar the process inherits the compositor's working
    directory -- normally $HOME -- and Claude Code then asks the user to trust
    their entire home folder. A persisted workspace is how the click opens
    where the user actually works.
    """
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"workspace": "/srv/project"}))

    args = argus_service.apply_config(_Args(cwd=None), config_path=config)

    assert args.cwd == "/srv/project"


def test_explicit_cwd_wins_over_the_configured_workspace(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"workspace": "/srv/project"}))

    args = argus_service.apply_config(_Args(cwd="/tmp/other"), config_path=config)

    assert args.cwd == "/tmp/other"


def test_session_extends_the_client_timeout_for_a_local_backend():
    """Claude Code must wait out a local model's prefill instead of retrying it.

    A 4 GB card prefills Claude Code's ~57k-token system prompt at roughly
    135 tok/s, so the first turn takes minutes. The default client timeout
    cancels and retries well before that, and each retry restarts the prefill
    it just cancelled -- the session never produces a first response.
    """
    env = argus_service.claude_session_environment(
        {}, port=8000, model_label="local", num_ctx=131072
    )

    assert int(env["API_TIMEOUT_MS"]) >= 600_000


def test_an_explicit_client_timeout_is_left_alone():
    """An operator who set their own budget keeps it."""
    env = argus_service.claude_session_environment(
        {"API_TIMEOUT_MS": "60000"}, port=8000, model_label="local"
    )

    assert env["API_TIMEOUT_MS"] == "60000"


def test_local_session_uses_its_own_config_profile(tmp_path):
    """The local session gets a separate Claude Code profile, not the user's.

    Plugins and MCP servers are loaded from the config directory and land in
    every request: measured here, they are 21k of a 57k-token prompt. On a
    cloud model that costs nothing noticeable; against a local backend
    prefilling at ~135 tok/s it is about two and a half minutes per first
    turn. A dedicated profile drops the prompt to 36k without touching the
    user's own configuration.
    """
    env = argus_service.claude_session_environment(
        {}, port=8000, model_label="local", config_dir=tmp_path / "profile"
    )

    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "profile")
    assert (tmp_path / "profile").is_dir()


def test_the_users_own_profile_can_be_kept():
    """Opting out restores the full plugin/MCP set."""
    env = argus_service.claude_session_environment(
        {}, port=8000, model_label="local", config_dir=None
    )

    assert "CLAUDE_CONFIG_DIR" not in env


def test_lean_profile_carries_a_placeholder_key(tmp_path):
    """A fresh profile has no credentials, so Claude Code would ask to log in.

    The local backend authenticates nothing -- requests go to the gateway on
    loopback -- but Claude Code still refuses to start without *some*
    credential. A placeholder satisfies that check and keeps the session
    non-interactive.
    """
    env = argus_service.claude_session_environment(
        {}, port=8000, model_label="local", config_dir=tmp_path / "p"
    )

    assert env["ANTHROPIC_API_KEY"]


def test_the_users_own_profile_is_never_given_a_key():
    """A key would shadow the user's OAuth credentials and break their login.

    Anthropic's credential resolution puts ANTHROPIC_API_KEY ahead of any
    stored profile, so setting one here would silently redirect an
    authenticated user's session to an empty credential.
    """
    env = argus_service.claude_session_environment(
        {}, port=8000, model_label="local", config_dir=None
    )

    assert "ANTHROPIC_API_KEY" not in env


def test_lean_profile_is_seeded_past_onboarding(tmp_path):
    """A brand-new profile otherwise stops at the first-run setup screen.

    Claude Code runs its onboarding (theme picker, welcome) once per config
    directory. Since ARGUS creates that directory itself, the user would meet
    a setup wizard they never asked for every time the profile is reset --
    and from a waybar-launched terminal it reads like a login prompt.
    """
    profile = tmp_path / "p"
    argus_service.claude_session_environment(
        {}, port=8000, model_label="local", config_dir=profile
    )

    state = json.loads((profile / ".claude.json").read_text())
    assert state["hasCompletedOnboarding"] is True
    assert json.loads((profile / "settings.json").read_text())["theme"]


def test_seeding_never_overwrites_an_existing_profile(tmp_path):
    """Re-launching must not reset a profile the user has since customized."""
    profile = tmp_path / "p"
    profile.mkdir()
    (profile / ".claude.json").write_text(json.dumps({"theme": "light", "mine": True}))

    argus_service.claude_session_environment(
        {}, port=8000, model_label="local", config_dir=profile
    )

    assert json.loads((profile / ".claude.json").read_text()) == {
        "theme": "light",
        "mine": True,
    }
