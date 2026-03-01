from __future__ import annotations

from typing import Any

from agent.tooling.context import ToolContext
from agent.tooling.types import ToolOutcome


def delegate_builtin(tool_name: str, ctx: ToolContext, args: Any) -> ToolOutcome:
    runner = getattr(ctx.runtime_state, "execute_builtin_tool", None)
    if not callable(runner):
        raise RuntimeError("runtime_state does not expose execute_builtin_tool")
    return runner(tool_name, args)
