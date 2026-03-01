from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel

from .effects import Effect


class DialogPolicy(str, Enum):
    ALLOW_WHEN_DIALOG = "ALLOW_WHEN_DIALOG"
    BLOCK_WHEN_DIALOG = "BLOCK_WHEN_DIALOG"


class ProgressPolicy(str, Enum):
    USER_FACING = "USER_FACING"
    NON_USER_FACING = "NON_USER_FACING"


class ThinkNextAction(str, Enum):
    CONTINUE = "continue"
    START_LOOP = "start_loop"
    ADVANCE = "advance"
    END_LOOP = "end_loop"
    DONE = "done"
    STUCK = "stuck"


@dataclass(frozen=True)
class ToolManifest:
    name: str
    description: str
    effects: frozenset[Effect]
    dialog_policy: DialogPolicy = DialogPolicy.BLOCK_WHEN_DIALOG
    progress_policy: ProgressPolicy = ProgressPolicy.NON_USER_FACING
    tags: frozenset[str] = field(default_factory=frozenset)


@dataclass
class ToolOutput:
    success: bool
    summary: str = ""
    error: Optional[str] = None
    data: Optional[dict[str, Any]] = None


@dataclass
class ThinkControl:
    next_action: ThinkNextAction
    loop_count: Optional[int] = None
    loop_description: Optional[str] = None
    hint_message: Optional[str] = None
    done_reasoning: Optional[str] = None


@dataclass
class ToolOutcome:
    output: ToolOutput
    control: Optional[ThinkControl] = None


@dataclass(frozen=True)
class ToolSpec:
    manifest: ToolManifest
    args_model: Type[BaseModel]
    fn: Callable[..., Any]


# Cross-cutting planner contract fields.
NARRATIVE_FIELD = {
    "narrative": {
        "type": "string",
        "description": (
            "A short first-person description of what you're doing for a non-technical observer."
        ),
    }
}

NEXT_HINT_FIELD = {
    "next_hint_json": {
        "type": "string",
        "description": (
            "Required JSON envelope for planner hint handoff. Compact keys: "
            "s(status), f(function_name), a(function_arguments), oi(overlay_index), "
            "c(confidence), r(reason), id(optional candidate_id)."
        ),
    }
}

BUDGET_FIELDS = {
    "budget_spent": {
        "type": "integer",
        "minimum": 0,
        "description": "Actions spent so far in mission.",
    },
    "budget_remaining": {
        "type": "integer",
        "minimum": 0,
        "description": "Actions remaining in mission.",
    },
    "budget_total": {
        "type": "integer",
        "minimum": 1,
        "description": "Total actions budget for mission.",
    },
}
