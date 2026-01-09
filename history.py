from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from models.core_models import ActionStep


@dataclass
class HistoryEntry:
    """Single recorded step in the agent's history."""
    step_number: int
    goal_description: str
    action_steps: List[ActionStep]
    reasoning: str
    success: bool
    overlay_index: Optional[int]
    page_url: str
    page_title: str
    timestamp: float = field(default_factory=time.time)

    def summary_line(self, max_reasoning: int = 120) -> str:
        """Generate a human-readable line for this history entry."""
        action_text = ", ".join(
            f"{step.action.value}#{step.overlay_index or 'N'}" for step in self.action_steps if step.action
        )
        action_text = action_text or self.goal_description
        reasoning = (self.reasoning or "").replace("\n", " ").strip()
        if len(reasoning) > max_reasoning:
            reasoning = reasoning[:max_reasoning].rstrip() + "…"
        status = "✅" if self.success else "❌"
        return f"{self.step_number}. {status} {action_text} @ {self.page_title or 'page'} ({self.page_url}) — {reasoning}"


class HistoryManager:
    """Keeps a rolling list of HistoryEntry objects."""

    def __init__(self, max_items: int = 20, summary_length: int = 120):
        self.max_items = max_items
        self.summary_length = summary_length
        self._entries: List[HistoryEntry] = []
        self._next_step = 1

    def add_entry(
        self,
        goal_description: str,
        action_steps: List[ActionStep],
        reasoning: str,
        success: bool,
        overlay_index: Optional[int],
        page_url: str,
        page_title: str,
    ) -> None:
        entry = HistoryEntry(
            step_number=self._next_step,
            goal_description=goal_description,
            action_steps=action_steps,
            reasoning=reasoning or "",
            success=success,
            overlay_index=overlay_index,
            page_url=page_url or "",
            page_title=page_title or "",
        )
        self._next_step += 1
        self._entries.append(entry)
        if len(self._entries) > self.max_items:
            self._entries = self._entries[-self.max_items:]

    def history_block(self, limit: Optional[int] = None) -> str:
        """Return a formatted history block for prompts."""
        entries = self._entries if limit is None else self._entries[-limit:]
        if not entries:
            return "HISTORY: <none yet>"
        lines = [entry.summary_line(max_reasoning=self.summary_length) for entry in entries]
        return "HISTORY:\n" + "\n".join(lines)

    def last_entry(self) -> Optional[HistoryEntry]:
        return self._entries[-1] if self._entries else None

    def __len__(self) -> int:
        return len(self._entries)
