"""Sandbox policy checks for commands, filesystem, clipboard, and websites."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
import json
import re
import shlex
import time


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str = ""


class SandboxPolicyEngine:
    """Evaluates sandbox policy decisions from runtime config."""

    _STRICT_DEFAULT_COMMAND_PREFIXES: list[list[str]] = [
        ["date"],
        ["pwd"],
        ["ls"],
        ["cat"],
        ["head"],
        ["tail"],
        ["wc"],
        ["echo"],
        ["rg"],
        ["find"],
        ["jq"],
        ["python3", "-c"],
    ]

    def __init__(self, *, config: Any, workspace_root: Path, event_logger: Any = None) -> None:
        self.config = config
        self.workspace_root = workspace_root.expanduser().resolve()
        self.event_logger = event_logger
        self._audit_log_path: Optional[Path] = None

    @property
    def enabled(self) -> bool:
        return bool(getattr(getattr(self.config, "sandbox", None), "enabled", True))

    @property
    def preset(self) -> str:
        return str(getattr(getattr(self.config, "sandbox", None), "preset", "standard")).strip().lower()

    @property
    def mode(self) -> str:
        return str(getattr(getattr(self.config, "sandbox", None), "mode", "enforce")).strip().lower()

    @property
    def enforce(self) -> bool:
        return self.enabled and self.mode == "enforce"

    @property
    def audit_enabled(self) -> bool:
        return bool(
            getattr(
                getattr(getattr(self.config, "sandbox", None), "audit", None),
                "enabled",
                True,
            )
        )

    def set_audit_log_path(self, path: Optional[Path]) -> None:
        """Attach or clear run-scoped audit logging."""
        if not path:
            self._audit_log_path = None
            return
        try:
            resolved = Path(path).expanduser().resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self._audit_log_path = resolved
        except Exception:
            self._audit_log_path = None

    def command_timeout_seconds(self) -> int:
        value = getattr(getattr(getattr(self.config, "sandbox", None), "command", None), "max_runtime_seconds", 30)
        try:
            return max(1, int(value))
        except Exception:
            return 30

    def check_command(self, command: str) -> PolicyDecision:
        if not self.enabled:
            decision = PolicyDecision(True)
            self._record_decision(
                check_type="command",
                target=str(command or ""),
                decision=decision,
                details={"sandbox_enabled": False},
            )
            return decision
        if self.preset == "locked":
            decision = PolicyDecision(False, "Command execution is disabled in locked preset.")
            self._record_decision(check_type="command", target=str(command or ""), decision=decision)
            return decision

        normalized = str(command or "").strip()
        if not normalized:
            decision = PolicyDecision(False, "Command is empty.")
            self._record_decision(check_type="command", target=normalized, decision=decision)
            return decision

        configured = self._configured_command_prefixes()
        if configured:
            decision = self._check_command_against_prefixes(normalized, configured)
            self._record_decision(
                check_type="command",
                target=normalized,
                decision=decision,
                details={"source": "configured_prefixes"},
            )
            return decision

        if self.preset == "strict":
            decision = self._check_command_against_prefixes(normalized, self._STRICT_DEFAULT_COMMAND_PREFIXES)
            self._record_decision(
                check_type="command",
                target=normalized,
                decision=decision,
                details={"source": "strict_defaults"},
            )
            return decision

        # No explicit prefixes configured for this preset: block.
        decision = PolicyDecision(False, "No allowed command prefixes configured.")
        self._record_decision(
            check_type="command",
            target=normalized,
            decision=decision,
            details={"source": "no_prefix_config"},
        )
        return decision

    def check_path(self, path: Path, *, operation: str) -> PolicyDecision:
        if not self.enabled:
            decision = PolicyDecision(True)
            self._record_decision(
                check_type="path",
                target=str(path),
                decision=decision,
                details={"operation": operation, "sandbox_enabled": False},
            )
            return decision

        allowed_roots = self._resolved_allowed_roots()
        try:
            resolved = path.expanduser().resolve()
        except Exception as e:
            decision = PolicyDecision(False, f"Invalid path: {e}")
            self._record_decision(
                check_type="path",
                target=str(path),
                decision=decision,
                details={"operation": operation},
            )
            return decision

        for root in allowed_roots:
            if self._is_within(resolved, root):
                decision = PolicyDecision(True)
                self._record_decision(
                    check_type="path",
                    target=str(resolved),
                    decision=decision,
                    details={"operation": operation, "matched_root": str(root)},
                )
                return decision
        roots_list = ", ".join(str(root) for root in allowed_roots)
        decision = PolicyDecision(False, f"{operation} path is outside allowed roots: {roots_list}")
        self._record_decision(
            check_type="path",
            target=str(resolved),
            decision=decision,
            details={"operation": operation},
        )
        return decision

    def check_clipboard_read(self) -> PolicyDecision:
        if not self.enabled:
            decision = PolicyDecision(True)
            self._record_decision(
                check_type="clipboard_read",
                target="system",
                decision=decision,
                details={"sandbox_enabled": False},
            )
            return decision
        if self.preset == "locked":
            decision = PolicyDecision(False, "Clipboard access is disabled in locked preset.")
            self._record_decision(check_type="clipboard_read", target="system", decision=decision)
            return decision

        allow_read = bool(
            getattr(
                getattr(getattr(self.config, "sandbox", None), "clipboard", None),
                "allow_read",
                True,
            )
        )
        if not allow_read:
            decision = PolicyDecision(False, "Clipboard read is disabled by sandbox policy.")
            self._record_decision(check_type="clipboard_read", target="system", decision=decision)
            return decision
        decision = PolicyDecision(True)
        self._record_decision(check_type="clipboard_read", target="system", decision=decision)
        return decision

    def check_url(self, url: str) -> PolicyDecision:
        if not self.enabled:
            decision = PolicyDecision(True)
            self._record_decision(
                check_type="url",
                target=str(url or ""),
                decision=decision,
                details={"sandbox_enabled": False},
            )
            return decision
        if self.preset == "locked":
            decision = PolicyDecision(False, "Website navigation is disabled in locked preset.")
            self._record_decision(check_type="url", target=str(url or ""), decision=decision)
            return decision

        raw = str(url or "").strip()
        if not raw:
            decision = PolicyDecision(False, "URL is empty.")
            self._record_decision(check_type="url", target=raw, decision=decision)
            return decision

        parsed = urlparse(raw)
        if parsed.scheme == "about" and raw == "about:blank":
            decision = PolicyDecision(True)
            self._record_decision(
                check_type="url",
                target=raw,
                decision=decision,
                details={"scheme": "about"},
            )
            return decision
        if parsed.scheme not in {"http", "https"}:
            decision = PolicyDecision(False, f"Unsupported URL scheme: {parsed.scheme or 'unknown'}")
            self._record_decision(
                check_type="url",
                target=raw,
                decision=decision,
                details={"scheme": parsed.scheme or "unknown"},
            )
            return decision

        host = (parsed.hostname or "").strip().lower()
        if not host:
            decision = PolicyDecision(False, "URL host is missing.")
            self._record_decision(check_type="url", target=raw, decision=decision)
            return decision

        allowed_domains = self._configured_allowed_domains()
        if not allowed_domains:
            if self._allow_empty_web_allowlist():
                decision = PolicyDecision(True)
                self._record_decision(
                    check_type="url",
                    target=raw,
                    decision=decision,
                    details={"host": host, "source": "empty_allowlist_permitted"},
                )
                return decision
            if self.preset == "strict":
                decision = PolicyDecision(
                    False,
                    "No allowed domains configured for strict preset.",
                )
            else:
                decision = PolicyDecision(False, "No allowed domains configured.")
            self._record_decision(
                check_type="url",
                target=raw,
                decision=decision,
                details={"host": host, "source": "empty_allowlist_blocked"},
            )
            return decision

        for pattern in allowed_domains:
            if self._domain_matches(host, pattern):
                decision = PolicyDecision(True)
                self._record_decision(
                    check_type="url",
                    target=raw,
                    decision=decision,
                    details={"host": host, "matched_pattern": pattern},
                )
                return decision
        decision = PolicyDecision(False, f"Domain '{host}' is not in allowed_domains.")
        self._record_decision(check_type="url", target=raw, decision=decision, details={"host": host})
        return decision

    @staticmethod
    def summarize_web_policy(config: Any) -> str:
        """Return a concise, mode-aware summary of effective web policy."""
        sandbox = getattr(config, "sandbox", None)
        if not sandbox or not bool(getattr(sandbox, "enabled", True)):
            return "disabled"

        preset = str(getattr(sandbox, "preset", "standard") or "standard").strip().lower() or "standard"
        mode = str(getattr(sandbox, "mode", "enforce") or "enforce").strip().lower() or "enforce"
        web_cfg = getattr(sandbox, "web", None)
        raw_domains = list(getattr(web_cfg, "allowed_domains", []) or [])
        domains = [str(item).strip() for item in raw_domains if str(item).strip()]
        allow_empty = bool(getattr(web_cfg, "allow_empty_allowlist", False))
        allow_empty_effective = allow_empty and preset not in {"strict", "locked"}

        if preset == "locked":
            web_scope = "web blocked" if mode == "enforce" else "policy-flagged (locked preset)"
        elif domains:
            web_scope = f"allowlist ({len(domains)} domains)"
        elif allow_empty_effective:
            web_scope = "all domains (empty allowlist)"
        else:
            reason = "strict requires domains" if preset == "strict" else "empty allowlist"
            web_scope = (
                f"web blocked ({reason})"
                if mode == "enforce"
                else f"policy-flagged ({reason})"
            )
        return f"{preset}/{mode}: {web_scope}"

    def render_prompt_policy_block(self) -> str:
        if not self.enabled:
            return "Sandbox policy is disabled."

        allowed_domains = self._configured_allowed_domains()
        if allowed_domains:
            domains_text = ", ".join(allowed_domains)
        elif self._allow_empty_web_allowlist():
            domains_text = "(none configured; all domains allowed)"
        elif self.preset == "strict":
            domains_text = "(none configured; strict preset requires explicit domains; web navigation blocked)"
        else:
            domains_text = "(none configured; web navigation blocked)"
        roots_text = ", ".join(str(root) for root in self._resolved_allowed_roots())

        configured_prefixes = self._configured_command_prefixes()
        if configured_prefixes:
            cmd_prefixes = [" ".join(prefix) for prefix in configured_prefixes]
        elif self.preset == "strict":
            cmd_prefixes = [" ".join(prefix) for prefix in self._STRICT_DEFAULT_COMMAND_PREFIXES]
        elif self.preset == "locked":
            cmd_prefixes = ["(commands disabled)"]
        else:
            cmd_prefixes = ["(none configured; commands blocked)"]

        command_text = ", ".join(cmd_prefixes)
        clipboard_allowed = bool(
            getattr(
                getattr(getattr(self.config, "sandbox", None), "clipboard", None),
                "allow_read",
                True,
            )
        )

        return (
            "SANDBOX POLICY CONSTRAINTS:\n"
            f"- preset={self.preset}, mode={self.mode}\n"
            f"- allowed websites: {domains_text}\n"
            f"- allowed workspace roots: {roots_text}\n"
            f"- allowed command prefixes: {command_text}\n"
            f"- clipboard read allowed: {'yes' if clipboard_allowed else 'no'}\n"
            "If an action would violate policy, choose a different allowed action."
        )

    def _resolved_allowed_roots(self) -> list[Path]:
        configured = list(
            getattr(
                getattr(getattr(self.config, "sandbox", None), "fs", None),
                "allowed_roots",
                [],
            )
            or []
        )
        if not configured:
            configured = ["{agent.workspace_root}"]

        roots: list[Path] = []
        for raw in configured:
            text = str(raw or "").strip()
            if not text:
                continue
            text = text.replace("{agent.workspace_root}", str(self.workspace_root))
            try:
                roots.append(Path(text).expanduser().resolve())
            except Exception:
                continue
        if not roots:
            return [self.workspace_root]
        return roots

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            return path == root or root in path.parents
        except Exception:
            return False

    def _configured_command_prefixes(self) -> list[list[str]]:
        raw = getattr(getattr(getattr(self.config, "sandbox", None), "command", None), "allowed_prefixes", [])
        prefixes: list[list[str]] = []
        for item in raw or []:
            if not isinstance(item, (list, tuple)):
                continue
            cleaned = [str(token).strip() for token in item if str(token).strip()]
            if cleaned:
                prefixes.append(cleaned)
        return prefixes

    def _check_command_against_prefixes(self, command: str, prefixes: list[list[str]]) -> PolicyDecision:
        segments = [seg.strip() for seg in re.split(r"\|\||&&|[|;]", command) if seg.strip()]
        if not segments:
            return PolicyDecision(False, "Command contains no executable segment.")

        for segment in segments:
            try:
                tokens = shlex.split(segment)
            except ValueError:
                return PolicyDecision(False, f"Unable to parse command segment: {segment}")
            if not tokens:
                continue
            if not any(self._has_prefix(tokens, prefix) for prefix in prefixes):
                allowed_list = ", ".join(" ".join(prefix) for prefix in prefixes)
                return PolicyDecision(False, f"Command segment '{segment}' not allowed. Allowed prefixes: {allowed_list}")
        return PolicyDecision(True)

    @staticmethod
    def _has_prefix(tokens: list[str], prefix: list[str]) -> bool:
        if len(tokens) < len(prefix):
            return False
        return tokens[: len(prefix)] == prefix

    def _configured_allowed_domains(self) -> list[str]:
        raw = getattr(getattr(getattr(self.config, "sandbox", None), "web", None), "allowed_domains", [])
        domains: list[str] = []
        for item in raw or []:
            text = str(item or "").strip().lower()
            if text:
                domains.append(text)
        return domains

    def _allow_empty_web_allowlist(self) -> bool:
        if self.preset in {"strict", "locked"}:
            return False
        return bool(
            getattr(
                getattr(getattr(self.config, "sandbox", None), "web", None),
                "allow_empty_allowlist",
                False,
            )
        )

    def _record_decision(
        self,
        *,
        check_type: str,
        target: str,
        decision: PolicyDecision,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        payload = {
            "timestamp": time.time(),
            "check_type": check_type,
            "target": str(target or ""),
            "allowed": bool(decision.allowed),
            "reason": str(decision.reason or ""),
            "preset": self.preset,
            "mode": self.mode,
        }
        if details:
            payload.update(details)

        if self.event_logger:
            try:
                self.event_logger.sandbox_decision(
                    check_type=check_type,
                    target=payload["target"],
                    allowed=payload["allowed"],
                    reason=payload["reason"],
                    preset=payload["preset"],
                    mode=payload["mode"],
                )
                if not decision.allowed:
                    self.event_logger.sandbox_blocked_action(
                        check_type=check_type,
                        target=payload["target"],
                        reason=payload["reason"],
                        preset=payload["preset"],
                        mode=payload["mode"],
                    )
            except Exception:
                pass

        if not self.audit_enabled or not self._audit_log_path:
            return
        try:
            self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")
        except Exception:
            pass

    @staticmethod
    def _domain_matches(host: str, pattern: str) -> bool:
        normalized_pattern = pattern.strip().lower()
        if not normalized_pattern:
            return False
        if normalized_pattern.startswith("*."):
            base = normalized_pattern[2:]
            return host == base or host.endswith(f".{base}")
        return host == normalized_pattern or host.endswith(f".{normalized_pattern}")
