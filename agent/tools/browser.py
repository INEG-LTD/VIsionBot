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


class ClickElementType(str, Enum):
    BUTTON = "button"
    LINK = "link"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    TAB = "tab"
    ICON = "icon"
    MENU_ITEM = "menu item"
    CARD = "card"
    IMAGE = "image"
    TEXT = "text"
    INPUT = "input"


class PressKey(str, Enum):
    ENTER = "Enter"
    ESCAPE = "Escape"
    TAB = "Tab"
    ARROW_DOWN = "ArrowDown"
    ARROW_UP = "ArrowUp"
    ARROW_LEFT = "ArrowLeft"
    ARROW_RIGHT = "ArrowRight"
    BACKSPACE = "Backspace"
    DELETE = "Delete"
    PAGE_DOWN = "PageDown"
    PAGE_UP = "PageUp"
    HOME = "Home"
    END = "End"
    SPACE = "Space"


class ClickArgs(BaseModel):
    element_id: int = Field(ge=1)
    element_type: ClickElementType
    description: str
    reasoning: str
    memory_evidence_ids: Optional[list[str]] = None


class TypeTextArgs(BaseModel):
    text: str
    field_description: str
    reasoning: str
    element_id: Optional[int] = Field(default=None, ge=1)
    memory_evidence_ids: Optional[list[str]] = None


class ClearTextArgs(BaseModel):
    field_description: str
    reasoning: str
    element_id: Optional[int] = Field(default=None, ge=1)


class SelectOptionArgs(BaseModel):
    option: str
    dropdown_description: str
    reasoning: str
    element_id: Optional[int] = Field(default=None, ge=1)


class UploadFileArgs(BaseModel):
    file_path: str
    target_description: str
    reasoning: str


class SetDateTimeArgs(BaseModel):
    value: str
    picker_description: str
    reasoning: str


class PressKeyArgs(BaseModel):
    key: PressKey
    reasoning: str


@tool(
    manifest=ToolManifest(
        name="click",
        description="Click on an interactive element.",
        effects=frozenset({Effect.WRITE_PAGE}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"browser"}),
    ),
    args_model=ClickArgs,
)
def click(ctx: ToolContext, args: ClickArgs) -> ToolOutcome:
    return delegate_builtin("click", ctx, args)


@tool(
    manifest=ToolManifest(
        name="type_text",
        description="Type text into an input field.",
        effects=frozenset({Effect.WRITE_PAGE}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"browser"}),
    ),
    args_model=TypeTextArgs,
)
def type_text(ctx: ToolContext, args: TypeTextArgs) -> ToolOutcome:
    return delegate_builtin("type_text", ctx, args)


@tool(
    manifest=ToolManifest(
        name="clear_text",
        description="Clear text from an input field.",
        effects=frozenset({Effect.WRITE_PAGE}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"browser"}),
    ),
    args_model=ClearTextArgs,
)
def clear_text(ctx: ToolContext, args: ClearTextArgs) -> ToolOutcome:
    return delegate_builtin("clear_text", ctx, args)


@tool(
    manifest=ToolManifest(
        name="select_option",
        description="Select an option from a dropdown.",
        effects=frozenset({Effect.WRITE_PAGE}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"browser"}),
    ),
    args_model=SelectOptionArgs,
)
def select_option(ctx: ToolContext, args: SelectOptionArgs) -> ToolOutcome:
    return delegate_builtin("select_option", ctx, args)


@tool(
    manifest=ToolManifest(
        name="upload_file",
        description="Upload a file to a file input.",
        effects=frozenset({Effect.WRITE_PAGE}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"browser"}),
    ),
    args_model=UploadFileArgs,
)
def upload_file(ctx: ToolContext, args: UploadFileArgs) -> ToolOutcome:
    return delegate_builtin("upload_file", ctx, args)


@tool(
    manifest=ToolManifest(
        name="set_datetime",
        description="Set a date/time value in picker.",
        effects=frozenset({Effect.WRITE_PAGE}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"browser"}),
    ),
    args_model=SetDateTimeArgs,
)
def set_datetime(ctx: ToolContext, args: SetDateTimeArgs) -> ToolOutcome:
    return delegate_builtin("set_datetime", ctx, args)


@tool(
    manifest=ToolManifest(
        name="press_key",
        description="Press a keyboard key.",
        effects=frozenset({Effect.WRITE_PAGE}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"browser"}),
    ),
    args_model=PressKeyArgs,
)
def press_key(ctx: ToolContext, args: PressKeyArgs) -> ToolOutcome:
    return delegate_builtin("press_key", ctx, args)
