from __future__ import annotations

from typing import Any, Optional

from pydantic import ValidationError

from .context import ToolContext
from .policy import EffectPolicyEngine
from .types import ThinkControl, ThinkNextAction, ToolOutcome, ToolOutput


class ToolEngine:
    """Executes registered tools with effect policy enforcement."""

    def __init__(self, *, registry: Any, policy_engine: EffectPolicyEngine, event_logger: Any = None) -> None:
        self.registry = registry
        self.policy_engine = policy_engine
        self.event_logger = event_logger

    def execute(self, tool_name: str, raw_args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        spec = self.registry.get(tool_name)
        if spec is None:
            return ToolOutcome(output=ToolOutput(success=False, error=f"Unknown tool: {tool_name}", summary="Unknown tool"))

        policy_decision = self.policy_engine.evaluate(spec.manifest, phase="runtime")
        if not policy_decision.allowed and self.policy_engine.enforce:
            return ToolOutcome(
                output=ToolOutput(
                    success=False,
                    error=policy_decision.reason,
                    summary=f"{tool_name} blocked by effect policy: {policy_decision.reason}",
                )
            )

        try:
            args = spec.args_model.model_validate(raw_args or {})
        except ValidationError as e:
            return ToolOutcome(
                output=ToolOutput(
                    success=False,
                    error=str(e),
                    summary=f"{tool_name} arguments invalid",
                )
            )

        try:
            result = spec.fn(ctx, args)
            outcome = self._coerce_outcome(result)
            if not policy_decision.allowed and not self.policy_engine.enforce:
                msg = f"effect_policy(observe): {policy_decision.reason}"
                if outcome.output.summary:
                    outcome.output.summary = f"{outcome.output.summary} | {msg}"
                else:
                    outcome.output.summary = msg
            return outcome
        except Exception as e:
            if self.event_logger is not None:
                try:
                    self.event_logger.system_error(f"Tool execution error ({tool_name}): {e}")
                except Exception:
                    pass
            return ToolOutcome(
                output=ToolOutput(
                    success=False,
                    error=str(e),
                    summary=f"{tool_name} execution failed",
                )
            )

    def _coerce_outcome(self, value: Any) -> ToolOutcome:
        if isinstance(value, ToolOutcome):
            return value
        if isinstance(value, ToolOutput):
            return ToolOutcome(output=value)
        if isinstance(value, bool):
            return ToolOutcome(output=ToolOutput(success=value, summary="success" if value else "failed"))
        if isinstance(value, str):
            return ToolOutcome(output=ToolOutput(success=True, summary=value))
        if isinstance(value, dict):
            success = bool(value.get("success", True))
            summary = str(value.get("summary", "")).strip()
            error = value.get("error")
            data = value.get("data")
            control = self._coerce_control(value.get("control"))
            return ToolOutcome(
                output=ToolOutput(success=success, summary=summary, error=error, data=data if isinstance(data, dict) else None),
                control=control,
            )
        return ToolOutcome(output=ToolOutput(success=True, summary="Tool completed"))

    @staticmethod
    def _coerce_control(value: Any) -> Optional[ThinkControl]:
        if value is None:
            return None
        if isinstance(value, ThinkControl):
            return value
        if not isinstance(value, dict):
            return None
        raw_next = value.get("next_action")
        if raw_next is None:
            return None
        try:
            next_action = (
                raw_next
                if isinstance(raw_next, ThinkNextAction)
                else ThinkNextAction(str(raw_next).strip().lower())
            )
        except Exception:
            return None
        loop_count = value.get("loop_count")
        try:
            loop_count_val = int(loop_count) if loop_count is not None else None
        except Exception:
            loop_count_val = None
        completed_rounds = value.get("completed_rounds", 0)
        try:
            completed_rounds_val = max(0, int(completed_rounds or 0))
        except Exception:
            completed_rounds_val = 0
        return ThinkControl(
            next_action=next_action,
            loop_count=loop_count_val,
            loop_mode=str(value.get("loop_mode", "") or "").strip() or "counted",
            loop_description=str(value.get("loop_description", "") or "").strip() or None,
            loop_exit_condition=str(value.get("loop_exit_condition", "") or "").strip() or None,
            completed_rounds=completed_rounds_val,
            hint_message=str(value.get("hint_message", "") or "").strip() or None,
            done_reasoning=str(value.get("done_reasoning", "") or "").strip() or None,
        )
