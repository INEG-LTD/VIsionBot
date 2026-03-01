from __future__ import annotations

from typing import Any, Callable, Type

from pydantic import BaseModel

from .types import ToolManifest


def tool(*, manifest: ToolManifest, args_model: Type[BaseModel]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Attach tool metadata to a handler function.

    Tool runtime signature must be: fn(ctx: ToolContext, args: args_model).
    """

    if not isinstance(manifest, ToolManifest):
        raise TypeError("manifest must be a ToolManifest")
    if not isinstance(args_model, type) or not issubclass(args_model, BaseModel):
        raise TypeError("args_model must be a Pydantic BaseModel type")

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(fn, "__tool_manifest__", manifest)
        setattr(fn, "__tool_args_model__", args_model)
        return fn

    return _decorator
