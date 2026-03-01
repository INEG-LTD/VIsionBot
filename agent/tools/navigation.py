from __future__ import annotations

from enum import Enum
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


class ScrollAmount(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class ScrollDirection(str, Enum):
    UP = "up"
    DOWN = "down"


class OpenUrlArgs(BaseModel):
    url: str
    reasoning: str


class GoBackArgs(BaseModel):
    steps: int = Field(default=1, ge=1)
    reasoning: str


class GoForwardArgs(BaseModel):
    steps: int = Field(default=1, ge=1)
    reasoning: str


class ScrollDownArgs(BaseModel):
    amount: ScrollAmount = ScrollAmount.MEDIUM
    reasoning: str


class ScrollUpArgs(BaseModel):
    amount: ScrollAmount = ScrollAmount.MEDIUM
    reasoning: str


class ScrollContainerArgs(BaseModel):
    element_id: int = Field(ge=1)
    direction: ScrollDirection
    amount: ScrollAmount = ScrollAmount.MEDIUM
    reasoning: str


class ScrollToElementArgs(BaseModel):
    element_id: int = Field(ge=1)
    reasoning: str


@tool(
    manifest=ToolManifest(
        name="open_url",
        description="Navigate to a specific URL.",
        effects=frozenset({Effect.NAVIGATE_WEB}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"navigation"}),
    ),
    args_model=OpenUrlArgs,
)
def open_url(ctx: ToolContext, args: OpenUrlArgs) -> ToolOutcome:
    return delegate_builtin("open_url", ctx, args)


@tool(
    manifest=ToolManifest(
        name="go_back",
        description="Navigate back in browser history.",
        effects=frozenset({Effect.NAVIGATE_WEB}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"navigation"}),
    ),
    args_model=GoBackArgs,
)
def go_back(ctx: ToolContext, args: GoBackArgs) -> ToolOutcome:
    return delegate_builtin("go_back", ctx, args)


@tool(
    manifest=ToolManifest(
        name="go_forward",
        description="Navigate forward in browser history.",
        effects=frozenset({Effect.NAVIGATE_WEB}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"navigation"}),
    ),
    args_model=GoForwardArgs,
)
def go_forward(ctx: ToolContext, args: GoForwardArgs) -> ToolOutcome:
    return delegate_builtin("go_forward", ctx, args)


@tool(
    manifest=ToolManifest(
        name="scroll_down",
        description="Scroll main page down.",
        effects=frozenset({Effect.WRITE_PAGE}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"navigation"}),
    ),
    args_model=ScrollDownArgs,
)
def scroll_down(ctx: ToolContext, args: ScrollDownArgs) -> ToolOutcome:
    return delegate_builtin("scroll_down", ctx, args)


@tool(
    manifest=ToolManifest(
        name="scroll_up",
        description="Scroll main page up.",
        effects=frozenset({Effect.WRITE_PAGE}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"navigation"}),
    ),
    args_model=ScrollUpArgs,
)
def scroll_up(ctx: ToolContext, args: ScrollUpArgs) -> ToolOutcome:
    return delegate_builtin("scroll_up", ctx, args)


@tool(
    manifest=ToolManifest(
        name="scroll_container",
        description="Scroll a specific container.",
        effects=frozenset({Effect.WRITE_PAGE}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"navigation"}),
    ),
    args_model=ScrollContainerArgs,
)
def scroll_container(ctx: ToolContext, args: ScrollContainerArgs) -> ToolOutcome:
    return delegate_builtin("scroll_container", ctx, args)


@tool(
    manifest=ToolManifest(
        name="scroll_to_element",
        description="Scroll target element into view.",
        effects=frozenset({Effect.WRITE_PAGE}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"navigation"}),
    ),
    args_model=ScrollToElementArgs,
)
def scroll_to_element(ctx: ToolContext, args: ScrollToElementArgs) -> ToolOutcome:
    return delegate_builtin("scroll_to_element", ctx, args)
