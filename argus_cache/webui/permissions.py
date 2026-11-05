"""Permission gate and unified-diff builder for local file read/write tools."""

from __future__ import annotations

import asyncio
import difflib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Set

_SENSITIVE_DIR_NAMES = {".ssh", ".gnupg", ".aws", ".kube"}
_SENSITIVE_FILE_NAMES = {
    ".env", ".env.local", ".env.production", "id_rsa", "id_ed25519",
    "shadow", "gshadow", "credentials", "credentials.json", ".netrc",
}
_SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


def is_sensitive_path(path: Path) -> bool:
    resolved = path.expanduser().resolve(strict=False)
    parts = {p.lower() for p in resolved.parts}
    if parts & _SENSITIVE_DIR_NAMES:
        return True
    name = resolved.name.lower()
    if name in _SENSITIVE_FILE_NAMES or resolved.suffix.lower() in _SENSITIVE_SUFFIXES:
        return True
    if str(resolved).startswith(("/etc/shadow", "/etc/gshadow", "/etc/sudoers")):
        return True
    return False


def is_within_dir(target: Path, parent: Path) -> bool:
    t = target.expanduser().resolve(strict=False)
    p = parent.expanduser().resolve(strict=False)
    return t == p or p in t.parents


def build_unified_diff(path: Path, old_text: Optional[str], new_text: str, max_lines: int = 120) -> str:
    old_lines = (old_text or "").splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{path.name}" if old_text is not None else "/dev/null",
            tofile=f"b/{path.name}",
            n=3,
        )
    )
    if not diff:
        return "(Değişiklik yok)"
    if len(diff) > max_lines:
        shown = "".join(diff[:max_lines])
        return f"{shown}\n... (+{len(diff) - max_lines} satır daha)"
    return "".join(diff).rstrip()


@dataclass
class PendingPermission:
    req_id: str
    session_id: str
    operation: str  # "read" | "write" | "edit"
    target_path: str
    summary: str
    diff_preview: str = ""
    decision: str = "pending"  # "allow_once" | "allow_dir" | "deny"
    event: asyncio.Event = field(default_factory=asyncio.Event)

    def to_event_dict(self) -> Dict[str, str]:
        return {
            "req_id": self.req_id,
            "session_id": self.session_id,
            "operation": self.operation,
            "target_path": self.target_path,
            "summary": self.summary,
            "diff_preview": self.diff_preview,
        }


class PermissionBroker:
    """Tracks per-session directory grants and coordinates interactive UI approval."""

    def __init__(self, safe_notes_dir: Path) -> None:
        self.safe_notes_dir = safe_notes_dir.expanduser().resolve(strict=False)
        self.safe_notes_dir.mkdir(parents=True, exist_ok=True)
        self._trusted_read: Dict[str, Set[Path]] = {}
        self._trusted_write: Dict[str, Set[Path]] = {}
        self._trusted_exec: Dict[str, Set[Path]] = {}
        self._pending: Dict[str, PendingPermission] = {}

    def can_read_immediately(self, session_id: str, target: Path, workspace_dir: Optional[str]) -> bool:
        resolved = target.expanduser().resolve(strict=False)
        if is_within_dir(resolved, self.safe_notes_dir):
            return True
        if workspace_dir:
            ws = Path(workspace_dir).expanduser().resolve(strict=False)
            if is_within_dir(resolved, ws):
                return True
        for allowed in self._trusted_read.get(session_id, set()):
            if is_within_dir(resolved, allowed):
                return True
        return False

    def can_write_immediately(self, session_id: str, target: Path) -> bool:
        resolved = target.expanduser().resolve(strict=False)
        if is_within_dir(resolved, self.safe_notes_dir):
            return True
        for allowed in self._trusted_write.get(session_id, set()):
            if is_within_dir(resolved, allowed):
                return True
        return False

    def can_exec_immediately(self, session_id: str, target: Path) -> bool:
        resolved = target.expanduser().resolve(strict=False)
        for allowed in self._trusted_exec.get(session_id, set()):
            if is_within_dir(resolved, allowed):
                return True
        return False

    async def ensure_permission(
        self,
        *,
        session_id: str,
        operation: str,
        target: Path,
        workspace_dir: Optional[str],
        summary: str,
        diff_preview: str,
        emit_event: Callable[[str, Dict[str, object]], None],
        timeout_s: float = 180.0,
    ) -> bool:
        resolved = target.expanduser().resolve(strict=False)
        if operation not in ("exec", "exec_critical", "desktop") and is_sensitive_path(resolved):
            raise PermissionError(f"Hassas dosya/dizin erişimi güvenlik politikası gereği engellendi: {resolved}")

        if operation == "read" and self.can_read_immediately(session_id, resolved, workspace_dir):
            return True
        if operation in ("write", "edit") and self.can_write_immediately(session_id, resolved):
            return True
        if operation == "exec" and self.can_exec_immediately(session_id, resolved):
            return True

        req = PendingPermission(
            req_id=uuid.uuid4().hex[:12],
            session_id=session_id,
            operation=operation,
            target_path=str(resolved),
            summary=summary,
            diff_preview=diff_preview,
        )
        self._pending[req.req_id] = req
        emit_event("permission_request", req.to_event_dict())

        try:
            await asyncio.wait_for(req.event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            emit_event("permission_resolved", {"req_id": req.req_id, "decision": "timeout"})
            return False
        finally:
            self._pending.pop(req.req_id, None)
        if req.decision == "allow_dir":
            grant_dir = resolved if resolved.is_dir() else resolved.parent
            self._trusted_read.setdefault(session_id, set()).add(grant_dir)
            if operation in ("write", "edit"):
                self._trusted_write.setdefault(session_id, set()).add(grant_dir)
            elif operation == "exec":
                self._trusted_exec.setdefault(session_id, set()).add(grant_dir)
            return True
        return req.decision == "allow_once"

    def resolve(self, req_id: str, decision: str) -> Optional[Dict[str, str]]:
        req = self._pending.get(req_id)
        if req is None:
            return None
        if decision not in ("allow_once", "allow_dir", "deny"):
            raise ValueError(f"Geçersiz izin kararı: {decision}")
        req.decision = decision
        req.event.set()
        return {"req_id": req_id, "decision": decision, "target_path": req.target_path}

    def cancel_session_pending(self, session_id: str) -> int:
        cancelled = 0
        for req_id, req in list(self._pending.items()):
            if req.session_id == session_id:
                req.decision = "deny"
                req.event.set()
                self._pending.pop(req_id, None)
                cancelled += 1
        return cancelled


