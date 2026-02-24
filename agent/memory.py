"""Narrative memory system for agent execution.

This module is the single source of truth for what the agent did, why it did it,
and what happened after each action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
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
    REPORT = "report"
    WRITE_DATA = "write_data"
    MARK_PROGRESS = "mark_progress"


class MemoryOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    NO_CHANGE = "no_change"
    BLOCKED = "blocked"


class MemoryEntryKind(str, Enum):
    EXECUTED_ACTION = "executed_action"
    REFLECTION = "reflection"
    PROGRESS = "progress"
    AUXILIARY = "auxiliary"
    PLANNING = "planning"


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
    memory_id: str
    timestamp: float
    memory_entry_index: int
    i_did: str
    because: str
    and_then: str
    action_type: str
    action_params: Dict[str, Any]
    outcome: str
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    mission: str
    entry_kind: str = MemoryEntryKind.EXECUTED_ACTION.value
    memory_tags: List[str] = field(default_factory=list)
    reference_memory_ids: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Backward-compatible alias."""
        return self.memory_id


class NarrativeMemory:
    """In-memory narrative log for the current run."""

    def __init__(self, browser: Optional[Any] = None):
        self.browser = browser
        self.entries: List[MemoryEntry] = []
        self._entries_by_id: Dict[str, MemoryEntry] = {}
        self.current_mission: str = ""
        self.memory_entry_counter: int = 1
        self.base_knowledge: List[str] = []
        self.question_answer_pairs: List[Dict[str, str]] = []
        self.url_history: List[str] = []
        self.url_pointer: int = -1
        self._current_action_reasoning: Optional[str] = None
        self._current_action_evidence_ids: List[str] = []
        self._current_action_stuck_pattern: Optional[str] = None
        self._last_overlay_index: Optional[int] = None

    # ---------------------------------------------------------------------
    # Context setters
    # ---------------------------------------------------------------------

    def attach_browser(self, browser: Any) -> None:
        self.browser = browser

    def start_mission(self, mission: str) -> None:
        self.current_mission = (mission or "").strip()

    def set_base_knowledge(self, knowledge: List[str]) -> None:
        self.base_knowledge = knowledge or []

    def set_user_prompt(self, prompt: str) -> None:
        # Kept to avoid duplicating mission setters at call sites.
        self.current_mission = (prompt or "").strip()

    def add_question_answer(self, question: str, answer: str) -> None:
        self.question_answer_pairs.append(
            {"question": question or "", "answer": answer or ""}
        )

    def get_recent_question_answers(self, n: int = 5) -> List[Dict[str, str]]:
        """Return the most recent user question/answer pairs."""
        if n <= 0:
            return []
        recent_pairs = self.question_answer_pairs[-n:]
        normalized: List[Dict[str, str]] = []
        for pair in recent_pairs:
            if not isinstance(pair, dict):
                continue
            question = str(pair.get("question", "") or "").strip()
            answer = str(pair.get("answer", "") or "").strip()
            if not question and not answer:
                continue
            normalized.append({"question": question, "answer": answer})
        return normalized

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
        memory_evidence_ids: Optional[List[str]] = None,
        stuck_pattern: Optional[str] = None,
    ) -> None:
        self._current_action_reasoning = reasoning
        memory_ids = memory_evidence_ids or []
        self._current_action_evidence_ids = self.normalize_memory_ids(memory_ids)
        self._current_action_stuck_pattern = (stuck_pattern or "").strip() or None

    def _consume_current_action_context(
        self,
    ) -> tuple[Optional[str], List[str], Optional[str]]:
        reasoning = self._current_action_reasoning
        memory_ids = list(self._current_action_evidence_ids)
        stuck_pattern = self._current_action_stuck_pattern

        self._current_action_reasoning = None
        self._current_action_evidence_ids = []
        self._current_action_stuck_pattern = None

        return (
            reasoning,
            memory_ids,
            stuck_pattern,
        )

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
        return "It worked and moved the mission forward."

    def _format_action(self, action_type: str, action_params: Dict[str, Any]) -> str:
        action_type = (action_type or "action").lower()
        description = (
            action_params.get("description")
            or action_params.get("action")
            or action_params.get("condition")
            or action_params.get("message")
            or action_params.get("payload")
            or action_params.get("resolved_path")
            or action_params.get("path")
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
        if action_type == "send_email":
            recipients = action_params.get("to") or ""
            if isinstance(recipients, list):
                recipient_text = ", ".join(
                    str(item).strip() for item in recipients if str(item).strip()
                )
            else:
                recipient_text = str(recipients).strip()
            subject = str(action_params.get("subject") or "").strip()
            body_preview = str(action_params.get("body_preview") or "").strip()
            if recipient_text and subject and body_preview:
                return (
                    f"I sent an email to {recipient_text} with subject '{subject}' "
                    f"and body '{body_preview}'"
                )
            if recipient_text and subject:
                return f"I sent an email to {recipient_text} with subject '{subject}'"
            if recipient_text:
                return f"I sent an email to {recipient_text}"
            return "I sent an email"
        if action_type == "think":
            return "I thought about what to do next"
        if action_type == "report":
            return f"I reported data to the user: {description or 'text payload'}"
        if action_type == "write_data":
            return f"I wrote data to {description or 'a file'}"
        if action_type == "mark_progress":
            return f"I marked progress: {description or 'completed one unit'}"
        return f"I executed {action_type}"

    def _classify_entry_kind(
        self,
        action_type: str
    ) -> MemoryEntryKind:
        action_type = (action_type or "").lower()

        if action_type == InteractionType.THINK.value:
            return MemoryEntryKind.REFLECTION
        if action_type == InteractionType.MARK_PROGRESS.value:
            return MemoryEntryKind.PROGRESS
        if action_type in {
            InteractionType.ASSERT.value,
            InteractionType.FLAG.value,
            InteractionType.WAIT_FOR.value,
            InteractionType.ASK.value,
        }:
            return MemoryEntryKind.AUXILIARY
        return MemoryEntryKind.EXECUTED_ACTION

    def _update_url_history(
        self,
        after_state: Optional[MemoryState],
        *,
        before_state: Optional[MemoryState] = None,
        action_type: Optional[str] = None,
        action_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not after_state or not after_state.url:
            return

        current_url = after_state.url
        if not self.url_history:
            self.url_history = [current_url]
            self.url_pointer = 0
            return

        pointer = self.url_pointer
        if pointer < 0 or pointer >= len(self.url_history):
            pointer = len(self.url_history) - 1
            self.url_pointer = pointer

        # If the current pointer does not match the recorded before URL,
        # realign to the nearest matching URL when possible.
        before_url = (before_state.url if before_state else "").strip()
        if before_url and self.url_history[pointer] != before_url:
            matching_indices = [
                idx for idx, url in enumerate(self.url_history) if url == before_url
            ]
            if matching_indices:
                pointer = min(matching_indices, key=lambda idx: abs(idx - pointer))
                self.url_pointer = pointer

        if self.url_history[self.url_pointer] == current_url:
            return

        params = action_params or {}
        normalized_action_type = (action_type or "").strip().lower()
        direction = str(params.get("direction", "")).strip().lower()

        # Explicit history traversal for go_back/go_forward.
        if (
            normalized_action_type == InteractionType.NAVIGATION.value
            and direction in {"back", "forward"}
        ):
            if direction == "back":
                for idx in range(self.url_pointer - 1, -1, -1):
                    if self.url_history[idx] == current_url:
                        self.url_pointer = idx
                        return
            else:
                for idx in range(self.url_pointer + 1, len(self.url_history)):
                    if self.url_history[idx] == current_url:
                        self.url_pointer = idx
                        return

        # Infer traversal even when direction metadata is absent.
        if (
            self.url_pointer > 0
            and self.url_history[self.url_pointer - 1] == current_url
        ):
            self.url_pointer -= 1
            return
        if (
            self.url_pointer < len(self.url_history) - 1
            and self.url_history[self.url_pointer + 1] == current_url
        ):
            self.url_pointer += 1
            return

        # New navigation from current pointer: discard stale forward branch.
        if self.url_pointer < len(self.url_history) - 1:
            self.url_history = self.url_history[: self.url_pointer + 1]

        if self.url_history and self.url_history[-1] == current_url:
            self.url_pointer = len(self.url_history) - 1
            return

        self.url_history.append(current_url)
        self.url_pointer = len(self.url_history) - 1

    def normalize_memory_ids(self, memory_ids: Optional[List[Any]]) -> List[str]:
        """Normalize, deduplicate, and sequence-sort known memory ids."""
        if not memory_ids:
            return []

        cleaned: List[str] = []
        seen: set[str] = set()
        for value in memory_ids:
            if not isinstance(value, str):
                continue
            memory_id = value.strip()
            if not memory_id or memory_id in seen:
                continue
            seen.add(memory_id)
            cleaned.append(memory_id)

        if not cleaned:
            return []

        order_map = {
            entry.memory_id: entry.memory_entry_index
            for entry in self.entries
        }
        known = [memory_id for memory_id in cleaned if memory_id in order_map]
        known.sort(key=lambda memory_id: order_map[memory_id])
        return known

    def _next_memory_id(self) -> str:
        return f"mem_{self.memory_entry_counter:06d}"

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
        memory_tags: Optional[List[str]] = None,
        reference_memory_ids: Optional[List[str]] = None,
    ) -> MemoryEntry:
        params = action_params or {}
        because = (reasoning or "I judged this as the best next step.").strip()
        outcome = self._determine_outcome(success, before_state, after_state, error_message)
        and_then = self._describe_outcome(outcome, before_state, after_state, error_message)
        i_did = self._format_action(action_type, params)
        entry_kind = self._classify_entry_kind(action_type).value
        refs = self.normalize_memory_ids(reference_memory_ids)
        memory_id = self._next_memory_id()

        entry = MemoryEntry(
            memory_id=memory_id,
            timestamp=time.time(),
            memory_entry_index=self.memory_entry_counter,
            i_did=i_did,
            because=because,
            and_then=and_then,
            action_type=(action_type or "unknown").lower(),
            action_params=params,
            outcome=outcome.value,
            state_before=self._state_to_dict(before_state),
            state_after=self._state_to_dict(after_state),
            mission=(mission if mission is not None else self.current_mission),
            entry_kind=entry_kind,
            memory_tags=memory_tags or [],
            reference_memory_ids=refs,
        )

        self.entries.append(entry)
        self._entries_by_id[entry.memory_id] = entry
        self.memory_entry_counter += 1

        overlay_index = params.get("overlay_index")
        if isinstance(overlay_index, int):
            self._last_overlay_index = overlay_index

        self._update_url_history(
            after_state,
            before_state=before_state,
            action_type=action_type,
            action_params=params,
        )
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
            "next_action",
            "recommended_next_step",
        ):
            if kwargs.get(key) is not None:
                params[key] = kwargs.get(key)

        (
            buffered_reasoning,
            buffered_memory_ids,
            buffered_stuck_pattern,
        ) = self._consume_current_action_context()

        reasoning = kwargs.get("reasoning") or buffered_reasoning
        success = bool(kwargs.get("success", True))
        error_message = kwargs.get("error_message")
        reference_memory_ids = (
            kwargs.get("memory_evidence_ids")
            or buffered_memory_ids
            or []
        )
        reference_memory_ids = self.normalize_memory_ids(reference_memory_ids)

        if reference_memory_ids:
            params["memory_evidence_ids"] = reference_memory_ids

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
            reference_memory_ids=reference_memory_ids,
        )

    # ---------------------------------------------------------------------
    # Memory reads
    # ---------------------------------------------------------------------

    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        if not isinstance(memory_id, str):
            return None
        return self._entries_by_id.get(memory_id.strip())

    def has_memory_id(self, memory_id: str) -> bool:
        return self.get_memory_by_id(memory_id) is not None

    def get_memory_entry(self, memory_entry_index: int) -> Optional[MemoryEntry]:
        for entry in self.entries:
            if entry.memory_entry_index == memory_entry_index:
                return entry
        return None

    def get_range(self, start_memory_entry_index: int, end_memory_entry_index: int) -> List[MemoryEntry]:
        lo = min(start_memory_entry_index, end_memory_entry_index)
        hi = max(start_memory_entry_index, end_memory_entry_index)
        return [e for e in self.entries if lo <= e.memory_entry_index <= hi]

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        if n <= 0:
            return []
        return self.entries[-n:]

    def has_entries(self) -> bool:
        return bool(self.entries)

    def get_entries_by_kind(
        self,
        kind: MemoryEntryKind | str,
        n: Optional[int] = None,
    ) -> List[MemoryEntry]:
        kind_value = kind.value if isinstance(kind, MemoryEntryKind) else str(kind)
        matches = [entry for entry in self.entries if entry.entry_kind == kind_value]
        if n is None or n <= 0:
            return matches
        return matches[-n:]

    def get_recent_executed_actions(self, n: int = 10) -> List[MemoryEntry]:
        return self.get_entries_by_kind(MemoryEntryKind.EXECUTED_ACTION, n=n)

    def get_recent_reflections(self, n: int = 10) -> List[MemoryEntry]:
        return self.get_entries_by_kind(MemoryEntryKind.REFLECTION, n=n)

    def get_recent_executed_action_ids(self, n: int = 10) -> List[str]:
        return [entry.memory_id for entry in self.get_recent_executed_actions(n=n)]

    def get_recent_reflection_ids(self, n: int = 10) -> List[str]:
        return [entry.memory_id for entry in self.get_recent_reflections(n=n)]

    def get_executed_action_ledger(self, n: int = 12) -> str:
        rows: List[str] = []
        for entry in self.get_recent_executed_actions(n=n):
            description = str(
                entry.action_params.get("description")
                or entry.action_params.get("action")
                or entry.action_params.get("resolved_path")
                or entry.action_params.get("path")
                or entry.action_params.get("url")
                or entry.action_params.get("condition")
                or ""
            ).strip()
            if entry.action_type == "send_email":
                recipients = entry.action_params.get("to") or []
                if isinstance(recipients, list):
                    recipient_text = ", ".join(
                        str(item).strip() for item in recipients if str(item).strip()
                    )
                else:
                    recipient_text = str(recipients).strip()
                subject = str(entry.action_params.get("subject") or "").strip()
                body_preview = str(entry.action_params.get("body_preview") or "").strip()
                if recipient_text:
                    description = f"to={recipient_text}"
                if subject:
                    description = f"{description} | subject={subject}" if description else f"subject={subject}"
                if body_preview:
                    description = f"{description} | body={body_preview}" if description else f"body={body_preview}"
            url_after = str(entry.state_after.get("url", "")).strip()
            suffix = f" | target={description}" if description else ""
            if url_after:
                suffix += f" | url={url_after}"
            rows.append(
                f"[{entry.memory_id}] {entry.action_type} -> {entry.outcome}{suffix}"
            )

        return "\n".join(rows) if rows else "No executed browser actions yet."

    def get_latest_recommended_next_step(self) -> tuple[Optional[str], Optional[str]]:
        latest_executed_index = 0
        for entry in self.entries:
            if entry.entry_kind == MemoryEntryKind.EXECUTED_ACTION.value:
                latest_executed_index = max(latest_executed_index, entry.memory_entry_index)

        for entry in reversed(self.entries):
            if entry.entry_kind != MemoryEntryKind.REFLECTION.value:
                continue
            value = str(entry.action_params.get("recommended_next_step", "")).strip()
            if value:
                # One-shot behavior: once any executed action happens after a recommendation,
                # treat that recommendation as consumed and do not resurface it.
                if latest_executed_index > entry.memory_entry_index:
                    continue
                return value, entry.memory_id
        return None, None

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
        start_memory_entry_index: Optional[int] = None,
        end_memory_entry_index: Optional[int] = None,
        entry_kind: Optional[str] = None,
    ) -> str:
        if start_memory_entry_index is not None or end_memory_entry_index is not None:
            lo = start_memory_entry_index if start_memory_entry_index is not None else 1
            hi = end_memory_entry_index if end_memory_entry_index is not None else (self.memory_entry_counter - 1)
            selected = self.get_range(lo, hi)
        else:
            selected = self.get_recent(n)

        if entry_kind:
            selected = [entry for entry in selected if entry.entry_kind == entry_kind]

        if not selected:
            return "No memory entries recorded yet."

        lines = []
        for entry in selected:
            lines.append(
                f"[{entry.memory_id}] {entry.i_did} because {entry.because}. {entry.and_then}"
            )
        return "\n".join(lines)

    def get_latest_strategy(self) -> str:
        for entry in reversed(self.entries):
            if entry.entry_kind != MemoryEntryKind.REFLECTION.value:
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
    "MemoryEntryKind",
    "MemoryState",
    "MemoryEntry",
    "NarrativeMemory",
]
