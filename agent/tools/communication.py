from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

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


class AskUserArgs(BaseModel):
    question: str
    context: str = ""
    options: list[str] = Field(default_factory=list)
    multi_select: bool = False
    yes_no: bool = False
    reasoning: str


class SendEmailArgs(BaseModel):
    to: str
    subject: str
    body: str
    reasoning: str


class BashArgs(BaseModel):
    command: str
    reasoning: str


@tool(
    manifest=ToolManifest(
        name="ask_user",
        description="Ask the user a clarifying question.",
        effects=frozenset({Effect.USER_IO}),
        dialog_policy=DialogPolicy.ALLOW_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"communication"}),
    ),
    args_model=AskUserArgs,
)
def ask_user(ctx: ToolContext, args: AskUserArgs) -> ToolOutcome:
    return delegate_builtin("ask_user", ctx, args)


@tool(
    manifest=ToolManifest(
        name="send_email",
        description="Send an email via configured provider.",
        effects=frozenset({Effect.SEND_EXTERNAL}),
        dialog_policy=DialogPolicy.ALLOW_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"communication", "external"}),
    ),
    args_model=SendEmailArgs,
)
def send_email(ctx: ToolContext, args: SendEmailArgs) -> ToolOutcome:
    return delegate_builtin("send_email", ctx, args)


@tool(
    manifest=ToolManifest(
        name="bash",
        description="Run a bash command on local host.",
        effects=frozenset({Effect.RUN_COMMAND}),
        dialog_policy=DialogPolicy.ALLOW_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"communication", "host"}),
    ),
    args_model=BashArgs,
)
def bash(ctx: ToolContext, args: BashArgs) -> ToolOutcome:
    return delegate_builtin("bash", ctx, args)
