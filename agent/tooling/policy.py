from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import json
import time

from .effects import Effect
from .types import ToolManifest


class PolicyMode(str, Enum):
    ENFORCE = "ENFORCE"
    OBSERVE = "OBSERVE"


class PolicyPreset(str, Enum):
    FULL = "FULL"
    WEB_SAFE = "WEB_SAFE"
    LOCKED_DOWN = "LOCKED_DOWN"


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str = ""


class EffectPolicyEngine:
    """Effect-level governance layered on top of sandbox policy checks."""

    _DENY_BY_PRESET: dict[PolicyPreset, frozenset[Effect]] = {
        PolicyPreset.FULL: frozenset(),
        PolicyPreset.WEB_SAFE: frozenset({
            Effect.READ_HOST,
            Effect.WRITE_HOST,
            Effect.RUN_COMMAND,
            Effect.SEND_EXTERNAL,
        }),
        PolicyPreset.LOCKED_DOWN: frozenset({
            Effect.READ_PAGE,
            Effect.WRITE_PAGE,
            Effect.NAVIGATE_WEB,
            Effect.TAB_MANAGEMENT,
            Effect.DIALOG_MANAGEMENT,
            Effect.READ_HOST,
            Effect.WRITE_HOST,
            Effect.RUN_COMMAND,
            Effect.SEND_EXTERNAL,
        }),
    }

    _ALLOW_ONLY_BY_PRESET: dict[PolicyPreset, Optional[frozenset[Effect]]] = {
        PolicyPreset.FULL: None,
        PolicyPreset.WEB_SAFE: None,
        PolicyPreset.LOCKED_DOWN: frozenset({
            Effect.USER_IO,
            Effect.CONTROL_FLOW,
        }),
    }

    def __init__(
        self,
        *,
        preset: PolicyPreset = PolicyPreset.FULL,
        mode: PolicyMode = PolicyMode.ENFORCE,
        event_logger: Any = None,
    ) -> None:
        self.preset = PolicyPreset(str(preset).upper()) if not isinstance(preset, PolicyPreset) else preset
        self.mode = PolicyMode(str(mode).upper()) if not isinstance(mode, PolicyMode) else mode
        self.event_logger = event_logger
        self._audit_path: Optional[Path] = None

    @property
    def enforce(self) -> bool:
        return self.mode == PolicyMode.ENFORCE

    def set_audit_log_path(self, path: Optional[Path]) -> None:
        if path is None:
            self._audit_path = None
            return
        try:
            resolved = Path(path).expanduser().resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self._audit_path = resolved
        except Exception:
            self._audit_path = None

    def evaluate(self, manifest: ToolManifest, *, phase: str, record: bool = True) -> PolicyDecision:
        effects = frozenset(manifest.effects or frozenset())
        if not effects:
            return PolicyDecision(False, "Tool has no declared effects.")

        deny = self._DENY_BY_PRESET.get(self.preset, frozenset())
        blocked = sorted(effect.value for effect in effects if effect in deny)
        if blocked:
            decision = PolicyDecision(False, f"Effects denied by preset {self.preset.value}: {', '.join(blocked)}")
            if record:
                self._record(manifest, phase=phase, decision=decision)
            return decision

        allow_only = self._ALLOW_ONLY_BY_PRESET.get(self.preset)
        if allow_only is not None:
            disallowed = sorted(effect.value for effect in effects if effect not in allow_only)
            if disallowed:
                decision = PolicyDecision(
                    False,
                    f"Effects not allowed in preset {self.preset.value}: {', '.join(disallowed)}",
                )
                if record:
                    self._record(manifest, phase=phase, decision=decision)
                return decision

        decision = PolicyDecision(True, "")
        if record:
            self._record(manifest, phase=phase, decision=decision)
        return decision

    def render_prompt_policy_block(self) -> str:
        deny = self._DENY_BY_PRESET.get(self.preset, frozenset())
        allow_only = self._ALLOW_ONLY_BY_PRESET.get(self.preset)
        denied_text = ", ".join(sorted(effect.value for effect in deny)) or "none"
        allow_text = ", ".join(sorted(effect.value for effect in (allow_only or set()))) if allow_only else "all"
        return (
            "EFFECT POLICY CONSTRAINTS:\n"
            f"- preset={self.preset.value}, mode={self.mode.value}\n"
            f"- allowed effects: {allow_text}\n"
            f"- denied effects: {denied_text}\n"
            "If a tool effect is disallowed, choose a compliant tool instead."
        )

    def _record(self, manifest: ToolManifest, *, phase: str, decision: PolicyDecision) -> None:
        payload = {
            "timestamp": time.time(),
            "tool": manifest.name,
            "effects": sorted(effect.value for effect in manifest.effects),
            "phase": phase,
            "allowed": bool(decision.allowed),
            "reason": str(decision.reason or ""),
            "preset": self.preset.value,
            "mode": self.mode.value,
        }
        if self.event_logger is not None:
            try:
                if decision.allowed:
                    self.event_logger.system_debug("Effect policy allowed tool", **payload)
                else:
                    self.event_logger.system_warning("Effect policy blocked tool", **payload)
            except Exception:
                pass

        if not self._audit_path:
            return
        try:
            with self._audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")
        except Exception:
            pass
