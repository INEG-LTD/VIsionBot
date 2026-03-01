from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from agent.tooling import (
    DialogPolicy,
    Effect,
    ProgressPolicy,
    ThinkControl,
    ThinkNextAction,
    ToolManifest,
    tool,
)
from agent.tooling.context import ToolContext
from agent.tooling.types import ToolOutcome

from ._helpers import delegate_builtin


class StuckPattern(str, Enum):
    ACTION_LOOP = "action_loop"
    NO_STATE_CHANGE = "no_state_change"
    FAILURE_CLUSTER = "failure_cluster"
    NAVIGATION_LOOP = "navigation_loop"
    ELEMENT_NOT_FOUND = "element_not_found"
    OTHER = "other"


class ThinkArgs(BaseModel):
    reasoning: str
    next_action: ThinkNextAction
    recommended_next_step: Optional[str] = None
    loop_count: Optional[int] = Field(default=None, ge=1)
    loop_description: Optional[str] = None
    stuck_pattern: Optional[StuckPattern] = None
    memory_evidence_ids: Optional[list[str]] = None

    @model_validator(mode="after")
    def _validate_contract(self) -> "ThinkArgs":
        if self.next_action == ThinkNextAction.START_LOOP:
            if self.loop_count is None:
                raise ValueError("loop_count is required when next_action=start_loop")
            if not str(self.loop_description or "").strip():
                raise ValueError("loop_description is required when next_action=start_loop")
        if self.next_action == ThinkNextAction.STUCK and self.stuck_pattern is None:
            raise ValueError("stuck_pattern is required when next_action=stuck")
        return self


class AssertConditionArgs(BaseModel):
    condition: str
    reasoning: str


class FlagArgs(BaseModel):
    message: str
    reasoning: str


class WaitForArgs(BaseModel):
    condition: str
    timeout_seconds: int = Field(default=10, ge=1, le=30)
    reasoning: str


@tool(
    manifest=ToolManifest(
        name="think",
        description="Stop and reason about what to do next.",
        effects=frozenset({Effect.CONTROL_FLOW}),
        dialog_policy=DialogPolicy.ALLOW_WHEN_DIALOG,
        progress_policy=ProgressPolicy.NON_USER_FACING,
        tags=frozenset({"cognitive"}),
    ),
    args_model=ThinkArgs,
)
def think(ctx: ToolContext, args: ThinkArgs) -> ToolOutcome:
    outcome = delegate_builtin("think", ctx, args)
    hint_message = None
    done_reasoning = None
    if args.next_action == ThinkNextAction.STUCK:
        hint_message = str(args.recommended_next_step or "").strip() or str(args.reasoning or "").strip()
    elif args.next_action == ThinkNextAction.DONE:
        done_reasoning = str(args.reasoning or "Mission complete").strip() or "Mission complete"

    outcome.control = ThinkControl(
        next_action=args.next_action,
        loop_count=int(args.loop_count) if args.loop_count is not None else None,
        loop_description=str(args.loop_description or "").strip() or None,
        hint_message=hint_message,
        done_reasoning=done_reasoning,
    )
    if not outcome.output.summary:
        outcome.output.summary = f"think(next_action={args.next_action.value})"
    return outcome


@tool(
    manifest=ToolManifest(
        name="assert_condition",
        description="Check whether an expected condition is true.",
        effects=frozenset({Effect.READ_PAGE}),
        dialog_policy=DialogPolicy.ALLOW_WHEN_DIALOG,
        progress_policy=ProgressPolicy.NON_USER_FACING,
        tags=frozenset({"cognitive"}),
    ),
    args_model=AssertConditionArgs,
)
def assert_condition(ctx: ToolContext, args: AssertConditionArgs) -> ToolOutcome:
    return delegate_builtin("assert_condition", ctx, args)


@tool(
    manifest=ToolManifest(
        name="flag",
        description="Send a non-blocking heads-up to user.",
        effects=frozenset({Effect.USER_IO}),
        dialog_policy=DialogPolicy.ALLOW_WHEN_DIALOG,
        progress_policy=ProgressPolicy.NON_USER_FACING,
        tags=frozenset({"cognitive", "communication"}),
    ),
    args_model=FlagArgs,
)
def flag(ctx: ToolContext, args: FlagArgs) -> ToolOutcome:
    return delegate_builtin("flag", ctx, args)


@tool(
    manifest=ToolManifest(
        name="wait_for",
        description="Wait for a condition before continuing.",
        effects=frozenset({Effect.CONTROL_FLOW}),
        dialog_policy=DialogPolicy.ALLOW_WHEN_DIALOG,
        progress_policy=ProgressPolicy.NON_USER_FACING,
        tags=frozenset({"cognitive"}),
    ),
    args_model=WaitForArgs,
)
def wait_for(ctx: ToolContext, args: WaitForArgs) -> ToolOutcome:
    return delegate_builtin("wait_for", ctx, args)
