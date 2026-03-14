"""
Mission progress policy interfaces and payloads.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ActionResultPayload:
    """Structured details about one completed action."""

    action_name: str
    action_args: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    result_data: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    current_url: str = ""
    page_title: str = ""
    accepted_event_names: tuple[str, ...] = ()
    accepted_events: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class FinishAttemptPayload:
    """Structured details about a mission-finish attempt."""

    kind: str
    current_url: str = ""
    page_title: str = ""
    accepted_event_names: tuple[str, ...] = ()
    event_name: str = ""
    event_data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinishDecision:
    """Policy decision for whether a finish attempt is allowed."""

    allow: bool
    reason: str = ""
    hint: str = ""


class MissionProgressPolicy(Protocol):
    """App-defined callbacks for mission-specific progress tracking."""

    def on_mission_start(
        self,
        *,
        mission: str,
        starting_url: str,
        base_knowledge: list[str],
    ) -> dict[str, Any]:
        ...

    def on_action_result(
        self,
        *,
        progress_state: dict[str, Any],
        action_result: ActionResultPayload,
    ) -> tuple[dict[str, Any], list[str]]:
        ...

    def get_progress_context(
        self,
        *,
        progress_state: dict[str, Any],
    ) -> str:
        ...

    def on_finish_attempt(
        self,
        *,
        progress_state: dict[str, Any],
        finish_attempt: FinishAttemptPayload,
    ) -> FinishDecision:
        ...
