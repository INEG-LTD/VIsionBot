"""Narrative memory system for agent execution.

This module is the single source of truth for what the agent did, why it did it,
and what happened after each action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
from typing import Any, Dict, List, Optional


class InteractionType(str, Enum):
    CLICK = "click"
    TYPE = "type"
    CLEAR_TEXT = "clear_text"
    SCROLL = "scroll"
    PRESS = "press"
    SELECT = "select"
    UPLOAD = "upload"
    DATETIME = "datetime"
    NAVIGATION = "navigation"
    EXTRACT = "extract"
    THINK = "think"
    ASSERT = "assert"
    FLAG = "flag"
    WAIT_FOR = "wait_for"
    ASK = "ask"
    MARK_PROGRESS = "mark_progress"


class MemoryOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    NO_CHANGE = "no_change"
    BLOCKED = "blocked"


@dataclass
class MemoryState:
    timestamp: float
    url: str
    title: str
    page_width: int
    page_height: int
    scroll_x: int
    scroll_y: int
    visible_text: str = ""
    screenshot: Optional[bytes] = None


@dataclass
class MemoryEntry:
    id: str
    timestamp: float
    turn_number: int
    i_did: str
    because: str
    and_then: str
    action_type: str
    action_params: Dict[str, Any]
    outcome: str
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    mission: str
    task: str
    memory_tags: List[str] = field(default_factory=list)
    reference_turns: List[int] = field(default_factory=list)


class NarrativeMemory:
    """In-memory narrative log for the current run."""

    def __init__(self, browser: Optional[Any] = None):
        self.browser = browser
        self.entries: List[MemoryEntry] = []
        self.current_mission: str = ""
        self.current_task: str = ""
        self.turn_counter: int = 0
        self.base_knowledge: List[str] = []
        self.question_answer_pairs: List[Dict[str, str]] = []
        self.url_history: List[str] = []
        self.url_pointer: int = -1
        self._current_action_reasoning: Optional[str] = None
        self._current_action_evidence_turns: List[int] = []
        self._current_action_evidence_summary: str = ""
        self._current_action_stuck_pattern: Optional[str] = None
        self._last_overlay_index: Optional[int] = None

    # ---------------------------------------------------------------------
    # Context setters
    # ---------------------------------------------------------------------

    def attach_browser(self, browser: Any) -> None:
        self.browser = browser

    def start_mission(self, mission: str) -> None:
        self.current_mission = (mission or "").strip()

    def start_task(self, task: str) -> None:
        self.current_task = (task or "").strip()

    def set_base_knowledge(self, knowledge: List[str]) -> None:
        self.base_knowledge = knowledge or []

    def set_user_prompt(self, prompt: str) -> None:
        # Kept to avoid duplicating mission/task setters at call sites.
        self.current_mission = (prompt or "").strip()

    def add_question_answer(self, question: str, answer: str) -> None:
        self.question_answer_pairs.append(
            {"question": question or "", "answer": answer or ""}
        )

    def set_current_action_reasoning(self, reasoning: Optional[str]) -> None:
        self._current_action_reasoning = reasoning

    def get_current_action_reasoning(self) -> Optional[str]:
        reasoning = self._current_action_reasoning
        self._current_action_reasoning = None
        return reasoning

    def set_current_action_context(
        self,
        *,
        reasoning: Optional[str] = None,
        memory_evidence_turns: Optional[List[int]] = None,
        memory_evidence_summary: Optional[str] = None,
        stuck_pattern: Optional[str] = None,
    ) -> None:
        self._current_action_reasoning = reasoning
        turns = memory_evidence_turns or []
        self._current_action_evidence_turns = [t for t in turns if isinstance(t, int) and t >= 0]
        self._current_action_evidence_summary = (memory_evidence_summary or "").strip()
        self._current_action_stuck_pattern = (stuck_pattern or "").strip() or None

    def _consume_current_action_context(
        self,
    ) -> tuple[Optional[str], List[int], str, Optional[str]]:
        reasoning = self._current_action_reasoning
        turns = list(self._current_action_evidence_turns)
        summary = self._current_action_evidence_summary
        stuck_pattern = self._current_action_stuck_pattern

        self._current_action_reasoning = None
        self._current_action_evidence_turns = []
        self._current_action_evidence_summary = ""
        self._current_action_stuck_pattern = None

        return reasoning, turns, summary, stuck_pattern

    # ---------------------------------------------------------------------
    # State capture
    # ---------------------------------------------------------------------

    def _capture_current_state(self) -> MemoryState:
        page = getattr(self.browser, "page", None)
        timestamp = time.time()
        if page is None:
            return MemoryState(
                timestamp=timestamp,
                url="",
                title="",
                page_width=0,
                page_height=0,
                scroll_x=0,
                scroll_y=0,
                visible_text="",
                screenshot=None,
            )

        try:
            url = page.url or ""
        except Exception:
            url = ""

        try:
            title = page.title() or ""
        except Exception:
            title = ""

        try:
            viewport_size = page.viewport_size or {"width": 0, "height": 0}
            page_width = int(viewport_size.get("width", 0) or 0)
            page_height = int(viewport_size.get("height", 0) or 0)
        except Exception:
            page_width = 0
            page_height = 0

        try:
            scroll_x = int(page.evaluate("window.scrollX || 0") or 0)
            scroll_y = int(page.evaluate("window.scrollY || 0") or 0)
        except Exception:
            scroll_x = 0
            scroll_y = 0

        try:
            visible_text = (page.evaluate("document.body ? document.body.innerText : ''") or "")[:2000]
        except Exception:
            visible_text = ""

        screenshot = None
        try:
            screenshot = page.screenshot(full_page=False)
        except Exception:
            screenshot = None

        return MemoryState(
            timestamp=timestamp,
            url=url,
            title=title,
            page_width=page_width,
            page_height=page_height,
            scroll_x=scroll_x,
            scroll_y=scroll_y,
            visible_text=visible_text,
            screenshot=screenshot,
        )

    def _state_to_dict(self, state: Optional[MemoryState]) -> Dict[str, Any]:
        if not state:
            return {}
        return {
            "url": state.url,
            "title": state.title,
            "scroll_x": state.scroll_x,
            "scroll_y": state.scroll_y,
            "page_width": state.page_width,
            "page_height": state.page_height,
            "visible_text": (state.visible_text or "")[:500],
        }

    def _has_meaningful_change(
        self,
        before_state: Optional[MemoryState],
        after_state: Optional[MemoryState],
    ) -> bool:
        if not before_state or not after_state:
            return False

        if before_state.url != after_state.url:
            return True
        if before_state.title != after_state.title:
            return True

        if abs((before_state.scroll_y or 0) - (after_state.scroll_y or 0)) > 10:
            return True
        if abs((before_state.scroll_x or 0) - (after_state.scroll_x or 0)) > 10:
            return True

        before_text = " ".join((before_state.visible_text or "").split())
        after_text = " ".join((after_state.visible_text or "").split())
        return before_text != after_text

    def _determine_outcome(
        self,
        success: bool,
        before_state: Optional[MemoryState],
        after_state: Optional[MemoryState],
        error_message: Optional[str] = None,
    ) -> MemoryOutcome:
        if not success:
            if error_message and "dialog" in error_message.lower():
                return MemoryOutcome.BLOCKED
            return MemoryOutcome.FAILURE

        if not self._has_meaningful_change(before_state, after_state):
            return MemoryOutcome.NO_CHANGE

        return MemoryOutcome.SUCCESS

    def _describe_outcome(
        self,
        outcome: MemoryOutcome,
        before_state: Optional[MemoryState],
        after_state: Optional[MemoryState],
        error_message: Optional[str] = None,
    ) -> str:
        if outcome == MemoryOutcome.FAILURE:
            if error_message:
                return f"It failed: {error_message}."
            return "It failed."
        if outcome == MemoryOutcome.BLOCKED:
            return "I was blocked and could not continue."
        if outcome == MemoryOutcome.NO_CHANGE:
            return "It completed but the page state did not meaningfully change."

        if before_state and after_state and before_state.url != after_state.url:
            return f"It navigated to {after_state.url or 'a new page'}."
        return "It worked and moved the task forward."

    def _format_action(self, action_type: str, action_params: Dict[str, Any]) -> str:
        action_type = (action_type or "action").lower()
        description = (
            action_params.get("description")
            or action_params.get("action")
            or action_params.get("condition")
            or action_params.get("message")
            or action_params.get("url")
            or ""
        )

        if action_type == "click":
            return f"I clicked {description or 'an element'}"
        if action_type == "type":
            text_input = action_params.get("text_input") or action_params.get("text")
            target = description or "an input field"
            if text_input:
                return f"I typed '{text_input}' into {target}"
            return f"I typed into {target}"
        if action_type == "navigation":
            url = action_params.get("navigation_url") or action_params.get("url")
            if url:
                return f"I navigated to {url}"
            return "I navigated"
        if action_type == "extract":
            return f"I extracted {description or 'data'}"
        if action_type == "think":
            return "I thought about what to do next"
        if action_type == "mark_progress":
            return f"I marked progress: {description or 'completed one unit'}"
        return f"I executed {action_type}"

    def _update_url_history(self, after_state: Optional[MemoryState]) -> None:
        if not after_state or not after_state.url:
            return
        current = after_state.url
        if not self.url_history:
            self.url_history = [current]
            self.url_pointer = 0
            return
        if self.url_history[-1] != current:
            self.url_history.append(current)
            self.url_pointer = len(self.url_history) - 1

    # ---------------------------------------------------------------------
    # Memory writes
    # ---------------------------------------------------------------------

    def record_action(
        self,
        *,
        action_type: str,
        action_params: Optional[Dict[str, Any]] = None,
        reasoning: Optional[str] = None,
        before_state: Optional[MemoryState] = None,
        after_state: Optional[MemoryState] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        mission: Optional[str] = None,
        task: Optional[str] = None,
        memory_tags: Optional[List[str]] = None,
        reference_turns: Optional[List[int]] = None,
    ) -> MemoryEntry:
        params = action_params or {}
        because = (reasoning or "I judged this as the best next step.").strip()
        outcome = self._determine_outcome(success, before_state, after_state, error_message)
        and_then = self._describe_outcome(outcome, before_state, after_state, error_message)
        i_did = self._format_action(action_type, params)

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            turn_number=self.turn_counter,
            i_did=i_did,
            because=because,
            and_then=and_then,
            action_type=(action_type or "unknown").lower(),
            action_params=params,
            outcome=outcome.value,
            state_before=self._state_to_dict(before_state),
            state_after=self._state_to_dict(after_state),
            mission=(mission if mission is not None else self.current_mission),
            task=(task if task is not None else self.current_task),
            memory_tags=memory_tags or [],
            reference_turns=reference_turns or [],
        )

        self.entries.append(entry)
        self.turn_counter += 1

        overlay_index = params.get("overlay_index")
        if isinstance(overlay_index, int):
            self._last_overlay_index = overlay_index

        self._update_url_history(after_state)
        return entry

    # Unified interaction writer used by executor/controller actions.
    def record_interaction(
        self,
        interaction_type: Any,
        before_state: Optional[MemoryState] = None,
        after_state: Optional[MemoryState] = None,
        **kwargs: Any,
    ) -> MemoryEntry:
        action_type = (
            interaction_type.value
            if hasattr(interaction_type, "value")
            else str(interaction_type)
        )

        params: Dict[str, Any] = {}
        target_info = kwargs.get("target_element_info")
        if isinstance(target_info, dict):
            params.update(target_info)

        for key in (
            "coordinates",
            "text_input",
            "keys_pressed",
            "scroll_direction",
            "scroll_axis",
            "target_x",
            "target_y",
            "navigation_url",
            "extraction_prompt",
            "extracted_data",
            "notes",
            "error_message",
        ):
            if kwargs.get(key) is not None:
                params[key] = kwargs.get(key)

        (
            buffered_reasoning,
            buffered_turns,
            buffered_summary,
            buffered_stuck_pattern,
        ) = self._consume_current_action_context()

        reasoning = kwargs.get("reasoning") or buffered_reasoning
        success = bool(kwargs.get("success", True))
        error_message = kwargs.get("error_message")
        reference_turns = kwargs.get("memory_evidence_turns") or buffered_turns or []

        memory_evidence_summary = kwargs.get("memory_evidence_summary")
        if memory_evidence_summary is None:
            memory_evidence_summary = buffered_summary
        if memory_evidence_summary:
            params["memory_evidence_summary"] = str(memory_evidence_summary)

        if reference_turns:
            params["memory_evidence_turns"] = reference_turns

        stuck_pattern = kwargs.get("stuck_pattern")
        if stuck_pattern is None:
            stuck_pattern = buffered_stuck_pattern
        if stuck_pattern:
            params["stuck_pattern"] = str(stuck_pattern)

        return self.record_action(
            action_type=action_type,
            action_params=params,
            reasoning=reasoning,
            before_state=before_state,
            after_state=after_state,
            success=success,
            error_message=error_message,
            mission=self.current_mission,
            task=self.current_task,
            reference_turns=reference_turns,
        )

    # ---------------------------------------------------------------------
    # Memory reads
    # ---------------------------------------------------------------------

    def get_turn(self, turn_number: int) -> Optional[MemoryEntry]:
        for entry in self.entries:
            if entry.turn_number == turn_number:
                return entry
        return None

    def get_range(self, start_turn: int, end_turn: int) -> List[MemoryEntry]:
        lo = min(start_turn, end_turn)
        hi = max(start_turn, end_turn)
        return [e for e in self.entries if lo <= e.turn_number <= hi]

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        if n <= 0:
            return []
        return self.entries[-n:]

    def search(
        self,
        action_type: Optional[str] = None,
        outcome: Optional[str] = None,
        text: Optional[str] = None,
    ) -> List[MemoryEntry]:
        text_norm = (text or "").lower().strip()
        results: List[MemoryEntry] = []

        for entry in self.entries:
            if action_type and entry.action_type != action_type:
                continue
            if outcome and entry.outcome != outcome:
                continue
            if text_norm:
                hay = f"{entry.i_did} {entry.because} {entry.and_then}".lower()
                if text_norm not in hay:
                    continue
            results.append(entry)

        return results

    def get_narrative(
        self,
        n: int = 10,
        start_turn: Optional[int] = None,
        end_turn: Optional[int] = None,
    ) -> str:
        if start_turn is not None or end_turn is not None:
            lo = start_turn if start_turn is not None else 0
            hi = end_turn if end_turn is not None else (self.turn_counter - 1)
            selected = self.get_range(lo, hi)
        else:
            selected = self.get_recent(n)

        if not selected:
            return "No memory entries recorded yet."

        lines = []
        for entry in selected:
            lines.append(
                f"[{entry.turn_number}] {entry.i_did} because {entry.because}. {entry.and_then}"
            )
        return "\n".join(lines)

    def get_latest_strategy(self) -> str:
        for entry in reversed(self.entries):
            if entry.action_type != InteractionType.THINK.value:
                continue
            next_action = str(entry.action_params.get("next_action", "")).lower()
            if next_action in {"continue", "stuck"} and entry.because:
                return entry.because
        return ""

    def get_stuck_pattern_hints(self) -> List[str]:
        hints: List[str] = []
        recent = self.get_recent(6)
        if len(recent) < 3:
            return hints

        # Pattern: repeated same action type
        types = [e.action_type for e in recent[-3:]]
        if len(set(types)) == 1:
            hints.append("action_loop")

        # Pattern: no state change cluster
        if sum(1 for e in recent[-3:] if e.outcome == MemoryOutcome.NO_CHANGE.value) >= 2:
            hints.append("no_state_change")

        # Pattern: failures cluster
        if all(e.outcome in {MemoryOutcome.FAILURE.value, MemoryOutcome.BLOCKED.value} for e in recent[-3:]):
            hints.append("failure_cluster")

        # Pattern: navigation loop
        urls = [e.state_after.get("url", "") for e in recent if e.state_after.get("url")]
        if len(urls) >= 4 and len(set(urls[-4:])) <= 2:
            hints.append("navigation_loop")

        return hints

    def get_last_interaction_overlay_index(self) -> Optional[int]:
        return self._last_overlay_index

__all__ = [
    "InteractionType",
    "MemoryOutcome",
    "MemoryState",
    "MemoryEntry",
    "NarrativeMemory",
]
