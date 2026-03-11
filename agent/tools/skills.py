from __future__ import annotations

from pydantic import BaseModel

from agent.tooling import (
    DialogPolicy,
    Effect,
    ProgressPolicy,
    ToolManifest,
    tool,
)
from agent.tooling.context import ToolContext
from agent.tooling.types import ToolOutcome

from ._helpers import delegate_builtin


class ActivateSkillArgs(BaseModel):
    skill_name: str
    reasoning: str


@tool(
    manifest=ToolManifest(
        name="activate_skill",
        description="Activate a skill and load its full SKILL.md instructions.",
        effects=frozenset({Effect.READ_HOST, Effect.CONTROL_FLOW}),
        dialog_policy=DialogPolicy.ALLOW_WHEN_DIALOG,
        progress_policy=ProgressPolicy.NON_USER_FACING,
        tags=frozenset({"skills", "control"}),
    ),
    args_model=ActivateSkillArgs,
)
def activate_skill(ctx: ToolContext, args: ActivateSkillArgs) -> ToolOutcome:
    return delegate_builtin("activate_skill", ctx, args)
