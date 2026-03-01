from __future__ import annotations

from typing import Optional

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


class SwitchTabArgs(BaseModel):
    tab_id: str
    reasoning: str


class CloseTabArgs(BaseModel):
    tab_id: str
    reasoning: str


class OpenTabArgs(BaseModel):
    url: Optional[str] = None
    reasoning: str


class DismissDialogArgs(BaseModel):
    accept: bool
    input_text: Optional[str] = None
    reasoning: str


@tool(
    manifest=ToolManifest(
        name="switch_tab",
        description="Switch to a different browser tab.",
        effects=frozenset({Effect.TAB_MANAGEMENT}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"tabs"}),
    ),
    args_model=SwitchTabArgs,
)
def switch_tab(ctx: ToolContext, args: SwitchTabArgs) -> ToolOutcome:
    return delegate_builtin("switch_tab", ctx, args)


@tool(
    manifest=ToolManifest(
        name="close_tab",
        description="Close a browser tab.",
        effects=frozenset({Effect.TAB_MANAGEMENT}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"tabs"}),
    ),
    args_model=CloseTabArgs,
)
def close_tab(ctx: ToolContext, args: CloseTabArgs) -> ToolOutcome:
    return delegate_builtin("close_tab", ctx, args)


@tool(
    manifest=ToolManifest(
        name="open_tab",
        description="Open a new browser tab.",
        effects=frozenset({Effect.TAB_MANAGEMENT}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"tabs"}),
    ),
    args_model=OpenTabArgs,
)
def open_tab(ctx: ToolContext, args: OpenTabArgs) -> ToolOutcome:
    return delegate_builtin("open_tab", ctx, args)


@tool(
    manifest=ToolManifest(
        name="dismiss_dialog",
        description="Dismiss or accept blocking JavaScript dialog.",
        effects=frozenset({Effect.DIALOG_MANAGEMENT}),
        dialog_policy=DialogPolicy.ALLOW_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"tabs"}),
    ),
    args_model=DismissDialogArgs,
)
def dismiss_dialog(ctx: ToolContext, args: DismissDialogArgs) -> ToolOutcome:
    return delegate_builtin("dismiss_dialog", ctx, args)
