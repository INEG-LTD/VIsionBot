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


class ExtractFormat(str, Enum):
    TEXT = "text"
    LIST = "list"
    TABLE = "table"
    STRUCTURED = "structured"


class WriteMode(str, Enum):
    OVERWRITE = "overwrite"
    APPEND = "append"


class WriteFormat(str, Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"


class ExtractDataArgs(BaseModel):
    data_description: str
    format_hint: ExtractFormat = ExtractFormat.TEXT
    reasoning: str
    memory_evidence_ids: Optional[list[str]] = None


class ReportDataArgs(BaseModel):
    payload: str
    reasoning: str


class WriteDataArgs(BaseModel):
    data: str
    path: Optional[str] = None
    file_name: Optional[str] = None
    mode: WriteMode = WriteMode.OVERWRITE
    format_hint: WriteFormat = WriteFormat.TEXT
    reasoning: str


class ReadFileArgs(BaseModel):
    path: str
    start_line: Optional[int] = Field(default=None, ge=1)
    end_line: Optional[int] = Field(default=None, ge=1)
    reasoning: str


class FindFilesArgs(BaseModel):
    pattern: str
    directory: str = "~"
    recursive: bool = True
    reasoning: str


class ReadClipboardArgs(BaseModel):
    reasoning: str


@tool(
    manifest=ToolManifest(
        name="extract_data",
        description="Extract data from the current page.",
        effects=frozenset({Effect.READ_PAGE}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"data"}),
    ),
    args_model=ExtractDataArgs,
)
def extract_data(ctx: ToolContext, args: ExtractDataArgs) -> ToolOutcome:
    return delegate_builtin("extract_data", ctx, args)


@tool(
    manifest=ToolManifest(
        name="report_data",
        description="Report textual data back to host application.",
        effects=frozenset({Effect.USER_IO}),
        dialog_policy=DialogPolicy.ALLOW_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"data", "communication"}),
    ),
    args_model=ReportDataArgs,
)
def report_data(ctx: ToolContext, args: ReportDataArgs) -> ToolOutcome:
    return delegate_builtin("report_data", ctx, args)


@tool(
    manifest=ToolManifest(
        name="write_data",
        description="Write textual data to local filesystem.",
        effects=frozenset({Effect.WRITE_HOST}),
        dialog_policy=DialogPolicy.ALLOW_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"data", "host"}),
    ),
    args_model=WriteDataArgs,
)
def write_data(ctx: ToolContext, args: WriteDataArgs) -> ToolOutcome:
    return delegate_builtin("write_data", ctx, args)


@tool(
    manifest=ToolManifest(
        name="read_file",
        description="Read local file contents.",
        effects=frozenset({Effect.READ_HOST}),
        dialog_policy=DialogPolicy.ALLOW_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"data", "host"}),
    ),
    args_model=ReadFileArgs,
)
def read_file(ctx: ToolContext, args: ReadFileArgs) -> ToolOutcome:
    return delegate_builtin("read_file", ctx, args)


@tool(
    manifest=ToolManifest(
        name="find_files",
        description="Find files by pattern in local filesystem.",
        effects=frozenset({Effect.READ_HOST}),
        dialog_policy=DialogPolicy.ALLOW_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"data", "host"}),
    ),
    args_model=FindFilesArgs,
)
def find_files(ctx: ToolContext, args: FindFilesArgs) -> ToolOutcome:
    return delegate_builtin("find_files", ctx, args)


@tool(
    manifest=ToolManifest(
        name="read_clipboard",
        description="Read system clipboard.",
        effects=frozenset({Effect.READ_HOST}),
        dialog_policy=DialogPolicy.ALLOW_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"data", "host"}),
    ),
    args_model=ReadClipboardArgs,
)
def read_clipboard(ctx: ToolContext, args: ReadClipboardArgs) -> ToolOutcome:
    return delegate_builtin("read_clipboard", ctx, args)
