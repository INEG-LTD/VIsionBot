from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.agent_context import EnvironmentState
from agent.memory import NarrativeMemory
from models import PageElements, PageInfo
from models.models import ActionStep


@dataclass
class ToolContext:
    page: Any
    elements: PageElements
    page_info: PageInfo
    environment_state: EnvironmentState
    memory_store: NarrativeMemory
    event_logger: Any
    sandbox_policy: Any
    action_step: ActionStep
    runtime_state: Any
