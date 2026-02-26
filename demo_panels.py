"""demo_panels.py — Reactive panel framework + all UI panels.

Adding a new panel (2 steps):
  1. Create a class here that inherits AgentPanel
  2. Add `yield MyPanel()` inside AgentView.compose() in demo_ui.py

Auto-subscription:
  When mounted, AgentPanel walks up the DOM to find an AgentView ancestor
  with _reactive_state and _event_stream, then subscribes automatically.
  No manual wiring needed.

AgentState fields are typed — use state.field_name directly, no string keys.
Add new fields to AgentState to expose new data to all panels.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Input, Label, Select, Switch, TextArea, Tree

from core.config import get_config_option_catalog, get_config_section_catalog
from utils.event_logger import BotEvent, EventType


# ---------------------------------------------------------------------------
# AgentState — typed snapshot of everything the UI needs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentState:
    """Immutable snapshot of a session's UI-visible state.

    Frozen so ReactiveState can detect changes via == comparison.
    Add new fields here to expose new data to all panels — no other wiring needed.
    """

    tab_id: str = ""
    label: str = ""
    status: str = "idle"
    last_message: str = "Ready"
    current_mission: str = ""
    worker_alive: bool = False
    queue_depth: int = 0
    dropped_event_count: int = 0
    agent_attached: bool = False
    current_iteration: int = 0
    paused: bool = False
    cancel_requested: bool = False
    llm_total_cost_usd: float = 0.0
    llm_total_tokens: int = 0
    total_actions: int = 0
    actions_since_progress: int = 0
    user_facing_actions_since_progress: int = 0
    budget_total: int = 0
    budget_spent: int = 0
    budget_remaining: int = 0
    budget_phase: str = "normal"
    low_budget_mode: bool = False
    budget_constraints_enabled: bool = True
    planning_batch_limit: int = 0
    iteration_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    navigation_latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    image_count: int = 0
    tool_calls: int = 0
    retries: int = 0
    avg_iteration_ms: float = 0.0
    p95_iteration_ms: float = 0.0
    avg_llm_ms: float = 0.0
    avg_tool_ms: float = 0.0
    avg_tokens_in: float = 0.0
    avg_tokens_out: float = 0.0
    avg_images_per_call: float = 0.0
    retries_per_mission: float = 0.0
    mission_ms: float = 0.0
    failure_code: str = ""
    failure_stage: str = ""
    checkpoint_pending: bool = False
    last_action_summary: str = ""
    in_loop: bool = False
    loop_count: int = 0
    loop_round: int = 0
    loop_description: str = ""
    mission_success: bool | None = None
    mission_reasoning: str = ""
    mission_final_url: str = ""
    last_known_url: str = ""
    sandbox_web_policy: str = ""
    thinking_active: bool = False
    thinking_text: str = "Thinking..."
    render_frame: int = 0  # incremented by poll timer to drive animations
    pending_question: str = ""
    pending_options: tuple[str, ...] = ()
    pending_multi_select: bool = False
    pending_yes_no: bool = False
    allow_custom: bool = True
    allow_skip: bool = True


# ---------------------------------------------------------------------------
# ReactiveState — notifies subscribers when state changes
# ---------------------------------------------------------------------------


class ReactiveState:
    """Holds an AgentState and notifies all subscribers whenever it changes.

    Call update() only from the main thread (e.g., inside a Textual set_interval
    callback). Subscribers are called synchronously on the same thread.
    """

    def __init__(self, initial: AgentState) -> None:
        self._state = initial
        self._subscribers: list[Callable[[AgentState], None]] = []

    @property
    def state(self) -> AgentState:
        return self._state

    def update(self, new_state: AgentState) -> None:
        """Replace state and notify all subscribers if anything changed."""
        if new_state == self._state:
            return
        self._state = new_state
        for cb in self._subscribers:
            try:
                cb(new_state)
            except Exception:
                pass

    def subscribe(self, callback: Callable[[AgentState], None]) -> None:
        """Subscribe and immediately invoke callback with the current state."""
        self._subscribers.append(callback)
        try:
            callback(self._state)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# EventStream — pub/sub bus with replay buffer
# ---------------------------------------------------------------------------


class EventStream:
    """Event bus that replays buffered events to late subscribers.

    push() must be called from the main thread only.
    Subscribers are called synchronously during push().
    """

    def __init__(self, maxlen: int = 5000) -> None:
        self._buffer: deque[BotEvent] = deque(maxlen=maxlen)
        self._subscribers: list[
            tuple[frozenset[EventType], Callable[[BotEvent], None]]
        ] = []
        self.dropped_count: int = 0

    def push(self, event: BotEvent) -> None:
        """Publish an event. Buffered and forwarded to all matching subscribers."""
        if len(self._buffer) == self._buffer.maxlen:
            self.dropped_count += 1
        self._buffer.append(event)
        for event_types, cb in self._subscribers:
            if not event_types or event.event_type in event_types:
                try:
                    cb(event)
                except Exception:
                    pass

    def subscribe(
        self,
        callback: Callable[[BotEvent], None],
        event_types: list[EventType] | None = None,
    ) -> None:
        """Subscribe and replay buffered events that match event_types."""
        key = frozenset(event_types) if event_types else frozenset()
        self._subscribers.append((key, callback))
        for event in list(self._buffer):
            if not key or event.event_type in key:
                try:
                    callback(event)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# AgentPanel — base class for all reactive panels
# ---------------------------------------------------------------------------


class AgentPanel(Widget):
    """Base class for panels that auto-subscribe to ReactiveState and EventStream.

    How to create a new panel:

        class MyPanel(AgentPanel):
            listens_to = [EventType.ACTION_DETERMINED]  # events you want

            def compose(self) -> ComposeResult:
                yield Label("", classes="my-label")

            def panel_ready(self) -> None:
                # runs after compose — use instead of on_mount
                self._label = self.query_one(".my-label", Label)

            def on_state_update(self, state: AgentState) -> None:
                # called whenever AgentState changes
                self._label.update(state.last_message)

            def on_agent_event(self, event: BotEvent) -> None:
                # called for events matching listens_to only
                ...

    Auto-subscription:
        When mounted, this panel walks up the DOM to find an ancestor with
        _reactive_state and _event_stream (set by AgentView). Subscribes
        automatically — just mount the panel inside AgentView.compose().

    Validation:
        If on_agent_event is overridden but listens_to is empty, a warning
        is logged because no events will be delivered.
    """

    listens_to: list[EventType] = []

    @staticmethod
    def _to_class_token(value: str) -> str:
        """Convert arbitrary text into a safe CSS class token."""
        raw = str(value or "").strip().lower()
        chars: list[str] = []
        last_dash = False
        for ch in raw:
            if ch.isalnum():
                chars.append(ch)
                last_dash = False
                continue
            if not last_dash:
                chars.append("-")
                last_dash = True
        token = "".join(chars).strip("-")
        return token or "unknown"

    @classmethod
    def apply_variant_class(cls, widget: Widget, prefix: str, value: str) -> str:
        """Set one `prefix*` class on a widget and remove old variants."""
        token = cls._to_class_token(value)
        target_class = f"{prefix}{token}"
        for existing in list(widget.classes):
            existing_name = str(existing)
            if existing_name.startswith(prefix) and existing_name != target_class:
                widget.remove_class(existing_name)
        widget.add_class(target_class)
        return target_class

    def on_mount(self) -> None:
        # panel_ready() must run first so subclasses can grab widget references
        # (subscribe() immediately calls on_state_update() with current state)
        self.panel_ready()

        reactive: ReactiveState | None = None
        stream: EventStream | None = None
        node = self.parent
        while node is not None:
            if reactive is None and hasattr(node, "_reactive_state"):
                reactive = node._reactive_state  # type: ignore[attr-defined]
            if stream is None and hasattr(node, "_event_stream"):
                stream = node._event_stream  # type: ignore[attr-defined]
            if reactive and stream:
                break
            node = getattr(node, "parent", None)

        if reactive is not None:
            reactive.subscribe(self._on_state_change)
        if stream is not None:
            stream.subscribe(self._on_stream_event, self.listens_to or None)

        if (
            type(self).on_agent_event is not AgentPanel.on_agent_event
            and not self.listens_to
        ):  # noqa: SIM102
            self.log.warning(
                f"{type(self).__name__}.on_agent_event is defined but listens_to is "
                "empty — no events will be delivered. Set listens_to = [EventType.XXX]."
            )

    def _on_state_change(self, state: AgentState) -> None:
        self.on_state_update(state)

    def _on_stream_event(self, event: BotEvent) -> None:
        self.on_agent_event(event)

    # ---- Overridable hooks -----------------------------------------------

    def panel_ready(self) -> None:
        """Called after auto-subscription completes. Override instead of on_mount."""

    def on_state_update(self, state: AgentState) -> None:
        """Called whenever AgentState changes. Update your widgets here.

        NOTE: Do NOT name this render() — Textual reserves render() for its
        own paint cycle and calls it with no arguments, which would crash.
        """

    def on_agent_event(self, event: BotEvent) -> None:
        """Called for events matching listens_to. Handle event-driven updates here."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(value: object, max_len: int = 72) -> str:
    text = str(value or "").strip()
    if not text:
        return "n/a"
    return text if len(text) <= max_len else f"{text[:max_len - 3]}..."


def _path_token(path: str) -> str:
    safe = []
    for ch in str(path):
        safe.append(ch if ch.isalnum() else "-")
    return "".join(safe).strip("-")


def _section_for_path(path: str) -> str:
    return str(path).split(".", 1)[0]


_STATUS_META: dict[str, tuple[str, str]] = {
    "idle": (" IDLE ", "idle"),
    "starting": (" STARTING ", "starting"),
    "running": (" RUNNING ", "running"),
    "completed": (" SUCCESS ", "completed"),
    "failed": (" FAILED ", "failed"),
    "stopping": (" STOPPING ", "stopping"),
    "stopped": (" STOPPED ", "stopped"),
    "paused": (" PAUSED ", "paused"),
    "queued": (" QUEUED ", "queued"),
    "asking": (" ASKING ", "asking"),
    "cancel requested": (" CANCELLING ", "cancel-requested"),
    "error": (" ERROR ", "error"),
}


# ---------------------------------------------------------------------------
# TimelineEntry — immutable record produced from a BotEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimelineEntry:
    kind: str  # "line", "collapsible", or "callout"
    text: str = ""
    title: str = ""
    body: str = ""
    variant: str = ""
    markup: bool = False


# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------


class StatusPill(Widget):
    """Reusable status chip with decoupled background and text layers."""

    DEFAULT_CSS = """
    StatusPill { 
        height: 1; 
        max-width: 30;
    }
    StatusPill .pill-text.status--idle { background: #6B7280; color: #f8fafc; }
    StatusPill .pill-text.status--starting { background: #A78BFA; color: #0b1220; }
    StatusPill .pill-text.status--running { background: #22C55E; color: #052e16; }
    StatusPill .pill-text.status--completed { background: #14B8A6; color: #042f2e; }
    StatusPill .pill-text.status--failed { background: #EF4444; color: #431407; }
    StatusPill .pill-text.status--stopping { background: #F97316; color: #431407; }
    StatusPill .pill-text.status--stopped { background: #475569; color: #431407; }
    StatusPill .pill-text.status--paused { background: #EAB308; color: #422006; }
    StatusPill .pill-text.status--queued { background: #3B82F6; color: #0b1220; }
    StatusPill .pill-text.status--asking { background: #60A5FA; color: #0b1220; }
    StatusPill .pill-text.status--cancel-requested { background: #FB7185; color: #431407; }
    StatusPill .pill-text.status--error { background: #DC2626; color: #431407; }
    StatusPill .pill-text.status--unknown { background: #64748b; color: #431407; }

    """

    def __init__(
        self,
        *,
        label: str = " IDLE ",
        variant: str = "idle",
        classes: str | None = None,
    ) -> None:
        super().__init__(classes=classes)
        self._initial_label = str(label or "IDLE")
        self._initial_variant = str(variant or "idle")

    def compose(self) -> ComposeResult:
        with Container():
            # yield Container(classes=f"pill-bg status--{AgentPanel._to_class_token(self._initial_variant)}")
            yield Label(self._initial_label, markup=False, classes="pill-text")

    def on_mount(self) -> None:
        # self._bg = self.query_one(".pill-bg", Container)
        self._text = self.query_one(".pill-text", Label)
        self.set_status(self._initial_label, self._initial_variant)

    def set_status(self, label: str, variant: str) -> None:
        self._text.update(str(label or "UNKNOWN"))
        # AgentPanel.apply_variant_class(self._bg, "status--", str(variant or "unknown"))
        AgentPanel.apply_variant_class(self._text, "status--", str(variant or "unknown"))


# ---------------------------------------------------------------------------
# Concrete panels
# ---------------------------------------------------------------------------


class StatusRow(AgentPanel):
    """Horizontal row: colored status pill."""

    DEFAULT_CSS = """
    StatusRow { height: auto; width: 99%; }
    StatusRow Horizontal { height: 1; width: 100%; }
    """
    
    label = "IDLE"

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield StatusPill(label=self.label, variant="idle")

    def panel_ready(self) -> None:
        self._pill = self.query_one(StatusPill)

    def on_state_update(self, state: AgentState) -> None:
        status_key = str(state.status or "").strip().lower()
        fallback_label = status_key.upper() if status_key else "UNKNOWN"
        label_text, variant = _STATUS_META.get(status_key, (fallback_label, "unknown"))
        self._pill.set_status(label_text, variant)
        self.label = label_text


class IntroBanner(AgentPanel):
    """Welcome banner. Hides automatically once a mission starts."""

    DEFAULT_CSS = """
    IntroBanner {
        height: auto; width: 99%;
        padding-top: 2; padding-bottom: 2; margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        from art import text2art

        yield Label(text2art("The Big Browser Agent", font="serifcap"))
        yield Label("[bold]Welcome to The Big Browser Agent Research Project[/bold]")
        yield Label(
            "[gray]An agent that lives in the browser and supercharges your productivity. "
            "Works best with targeted, specific tasks.[/gray]"
        )
        yield Label(
            "[gray][bold]Tip:[/bold] Tell the agent exactly what to do — "
            "specific and direct works best.[/gray]"
        )

    def on_state_update(self, state: AgentState) -> None:
        started = bool(state.current_mission) or state.status not in {"idle"}
        self.display = not started


class TimelinePanel(AgentPanel):
    """Scrollable list of timeline entries. Appends a widget per event."""

    listens_to = list(EventType)
    max_entries = 600

    DEFAULT_CSS = """
    TimelinePanel {
        height: 1fr; 
        min-height: 0; 
        width: 99%;
        layers: below above;
    }
    .timeline-block {
        height: 1fr;
        min-height: 0;
        border: solid #9C9C9C;
    }
    .timeline-thinking-label {
        color: #F0D264;
        height: 1;
        width: 100%;
        text-align: left;
        padding: 0 1;
        border-top: solid #4B5563;
    }
    TimelinePanel VerticalScroll {
        height: 1fr;
        padding: 0 0 0 1;
        overflow-y: auto;
    }

    TimelinePanel Label {
        width: 100%;
        height: auto;
    }
    
    .timeline-events {
        layout: vertical;
        width: 100%;
        height: auto;
        layers: above;
    }
    .timeline-card {
        width: 60;
        height: auto;
        border: round #6b7280;
        background: rgba(148, 163, 184, 0.08);
        padding: 1 1;
        margin: 1 0 1 2;
        padding-left: 1;
    }
    .timeline-card-header {
        width: 100%;
        text-style: bold;
    }
    .timeline-card-body {
        width: 100%;
        color: #d1d5db;
        padding-left: 1;
    }
    .timeline-card--ask {
        border: round #60A5FA;
    }
    .timeline-card--ask .timeline-card-header {
        color: #bfdbfe;
    }
    .timeline-card--answered {
        border: round #34D399;
        background: rgba(52, 211, 153, 0.14);
    }
    .timeline-card--answered .timeline-card-header {
        color: #a7f3d0;
    }
    .timeline-card--skipped {
        border: round #FBBF24;
        background: rgba(251, 191, 36, 0.13);
    }
    .timeline-card--skipped .timeline-card-header {
        color: #fde68a;
    }
    .timeline-card--error {
        border: round #F87171;
        background: rgba(248, 113, 113, 0.13);
    }
    .timeline-card--error .timeline-card-header {
        color: #fecaca;
    }
    
    .timeline-intro-text {
        align: center middle;
        layer: below;
        text-align: center;
    }
    
    .timeline-intro-text Label {
        text-align: center;
        width: 100%
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(classes="timeline-block"):
            with VerticalScroll(classes="timeline-scroll"):
                yield Container(classes="timeline-events")
                with Container(classes="timeline-intro-text"):
                    yield Label(
                        "[gray]The history of agent work will appear here[/gray]"
                    )
                    yield Label("[darkgray]TIMELINE[/darkgray]", classes="timeline-intro-text-label")
            yield Label("", classes="timeline-thinking-label")

    def panel_ready(self) -> None:
        self._thinking = self.query_one(".timeline-thinking-label", Label)
        self._scroll = self.query_one(".timeline-scroll", VerticalScroll)
        self._events = self.query_one(".timeline-events", Container)
        self._intro = self.query_one(".timeline-intro-text", Container)

    def on_state_update(self, state: AgentState) -> None:
        if state.thinking_active:
            frame = state.render_frame % 4
            dots = "." * frame + " " * (3 - frame)
            text = _truncate(state.thinking_text or "Thinking...", 84)
            self._thinking.update(f"THINKING: 🤔 {text}{dots}")
        else:
            self._thinking.update("")

    def on_agent_event(self, event: BotEvent) -> None:
        entry = self._format_event(event)
        if entry is None:
            return

        follow_tail = self._is_near_bottom()
        self._intro.display = False

        if entry.kind == "collapsible":
            body_label = Label(entry.body, markup=False, classes="collapsible-entry-body")
            widget: Widget = Collapsible(
                body_label,
                title=entry.title or "Details",
                collapsed=True,
                classes="collapsible-entry",
            )
        elif entry.kind == "callout":
            # header_label = Label(entry.title or "NOTICE", markup=False, classes="timeline-card-header")
            body_label = Label(
                entry.body or entry.text or "",
                markup=False,
                classes="timeline-card-body",
            )
            card = Container(
                # header_label,
                body_label,
                classes="timeline-card",
            )
            card.border_title = entry.title or "NOTICE"
            AgentPanel.apply_variant_class(card, "timeline-card--", entry.variant or "default")
            widget = card
        else:
            widget = Label(entry.text or "", markup=entry.markup)
            widget.styles.width = "100%"
            widget.styles.height = "auto"
            widget.styles.padding = (0, 1, 0, 0)

        self._events.mount(widget)
        self._trim_entries()
        self._scroll.refresh(layout=True)

        if entry.kind == "collapsible":
            def _set_body_styles(lbl: Label = body_label) -> None:
                lbl.styles.width = "100%"
                lbl.styles.height = "auto"
            self.call_after_refresh(_set_body_styles)

        if follow_tail:
            self.call_after_refresh(lambda: self._scroll.scroll_end(animate=False))

    def _trim_entries(self) -> None:
        while len(self._events.children) > self.max_entries:
            oldest = self._events.children[0]
            try:
                oldest.remove()
            except Exception:
                break

    def _is_near_bottom(self, threshold: int = 3) -> bool:
        """Return True when viewport is close enough to bottom to keep auto-follow enabled."""
        try:
            max_scroll = float(getattr(self._scroll, "max_scroll_y", 0.0) or 0.0)
        except Exception:
            max_scroll = 0.0

        current_scroll: float | None = None
        try:
            current_scroll = float(getattr(self._scroll, "scroll_y"))
        except Exception:
            current_scroll = None

        if current_scroll is None:
            try:
                offset = getattr(self._scroll, "scroll_offset")
                current_scroll = float(getattr(offset, "y"))
            except Exception:
                current_scroll = None

        if current_scroll is None:
            return True
        if max_scroll <= 0:
            return True
        return (max_scroll - current_scroll) <= float(threshold)

    @staticmethod
    def _format_event_details(details: dict[str, Any]) -> str:
        lines: list[str] = []
        for key, value in details.items():
            if value is None or key in {"timestamp", "timestamp_iso"}:
                continue
            if isinstance(value, (str, int, float, bool)):
                lines.append(f"{key}: {value}")
                continue
            if isinstance(value, list):
                lines.append(f"{key}: list[{len(value)}]")
                continue
            if isinstance(value, dict):
                lines.append(f"{key}: dict[{len(value)}]")
                continue
            lines.append(f"{key}: {type(value).__name__}")
        return "\n".join(lines)

    def _format_event(self, event: BotEvent) -> TimelineEntry | None:
        details = event.details or {}
        t = event.event_type

        if t == EventType.ITERATION_START:
            if details.get("iteration", 0) == 1:
                return TimelineEntry(
                    kind="line",
                    text=f"> {details.get('mission', 'unknown')}",
                )

        elif t == EventType.ACTION_DETERMINED:
            narrative = details.get("narrative") or "unknown"
            reasoning = details.get("reasoning") or "no reasoning"
            return TimelineEntry(
                kind="collapsible", title=str(narrative), body=str(reasoning)
            )

        elif t == EventType.LOOP_STATE_CHANGED:
            change = details.get("change", "update")
            round_num = details.get("loop_round", "?")
            total = details.get("loop_count", "?")
            return TimelineEntry(
                kind="line", text=f"LOOP {change}: round {round_num}/{total}"
            )

        elif t == EventType.ASK_REQUESTED:
            question = str(details.get("question", "")).strip()
            return TimelineEntry(
                kind="callout",
                title="QUESTION",
                body=question or "The agent asked a question.",
                variant="ask",
            )

        elif t == EventType.ASK_COMMAND_ANSWERED:
            response = str(details.get("response", "")).strip() or "(empty)"
            return TimelineEntry(
                kind="callout",
                title="ANSWERED",
                body=response,
                variant="answered",
            )

        elif t == EventType.ASK_COMMAND_SKIPPED:
            return TimelineEntry(
                kind="callout",
                title="SKIPPED",
                body="You didn't answer the question.",
                variant="skipped",
            )

        elif t == EventType.ASK_COMMAND_FAILURE:
            error = str(details.get("error", "")).strip() or "Unknown ask callback error."
            return TimelineEntry(
                kind="callout",
                title="ASK ERROR",
                body=f"{error}",
                variant="error",
            )

        elif t == EventType.SYSTEM_WARNING:
            return TimelineEntry(kind="line", text=f"WARNING {event.message}")

        elif t == EventType.SYSTEM_ERROR:
            return TimelineEntry(kind="line", text=f"ERROR {event.message}")

        elif t == EventType.SYSTEM_DEBUG:
            source = str(details.get("source", "")).strip().lower()
            message = str(event.message or "").strip()
            if source == "dprint":
                return TimelineEntry(kind="line", text=message, markup=False)
            return TimelineEntry(kind="line", text=f"DEBUG {message}", markup=False)

        elif t == EventType.SYSTEM_INFO:
            message = str(event.message or "").strip()
            return TimelineEntry(kind="line", text=f"INFO {message}", markup=False)

        label = str(t.value or "event").upper()
        message = str(event.message or "").strip()
        if not message:
            message = label
        detail_text = self._format_event_details(details)
        if detail_text:
            return TimelineEntry(
                kind="collapsible",
                title=f"{label}: {message}",
                body=detail_text,
            )
        return TimelineEntry(kind="line", text=f"{label}: {message}", markup=False)

class ConfigButtons(AgentPanel):
    """Buttons for the agent configuration."""

    DEFAULT_CSS = """
    ConfigButtons { height: auto; width: 100%; margin-top: 1; }
    ConfigButtons Horizontal { height: auto; width: 100%; }
    ConfigButtons Button {
        width: 1fr;
        min-width: 12;
        height: 3;
        background: transparent;
        border: solid #9C9C9C;
    }
    """
    
    def __init__(self, tab_id: str) -> None:
        super().__init__()
        self._tab_id = tab_id
        self.config_button_id = f"open-config-{tab_id}"

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Button("Config", classes="config-button", id=self.config_button_id)
            
    def panel_ready(self) -> None:
        self._config_button = self.query_one(".config-button", Button)
        self._config_button.styles.border = ("round", "white")
        
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == self.config_button_id:
            self._config_button.pressed = True


class AgentConfigEditor(Widget):
    """Per-agent config editor with a section tree and dynamic detail pane."""

    DEFAULT_CSS = """
    AgentConfigEditor {
        width: 100%;
        height: 100%;
    }
    .agent-config-root {
        layout: horizontal;
        width: 100%;
        height: 100%;
    }
    .config-sidebar {
        width: 42;
        min-width: 36;
        border: solid #808080;
        padding: 1 1;
    }
    .config-sidebar-title {
        text-style: bold;
        margin-bottom: 1;
    }
    .config-tree {
        height: 1fr;
        width: 100%;
    }
    .config-detail-scroll {
        width: 1fr;
        height: 100%;
        min-height: 0;
        border: solid #808080;
        border-left: none;
        padding: 1 2;
    }
    .config-detail-container {
        width: 100%;
        height: auto;
        min-height: 100%;
    }
    .config-empty-state {
        width: 100%;
        height: 100%;
        align: center middle;
    }
    .config-empty-text {
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 1;
    }
    .config-empty-actions {
        width: 100%;
        align: center middle;
    }
    .back-to-agent-button {
        min-width: 18;
        width: auto;
    }
    .config-detail-header {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }
    .config-detail-header-row {
        width: 100%;
        height: auto;
    }
    .config-detail-title-wrap {
        width: 1fr;
        height: auto;
    }
    .config-detail-title {
        text-style: bold;
    }
    .config-detail-description {
        color: #a0a0a0;
    }
    .config-next-mission-note {
        color: #bcae4a;
        margin-top: 1;
        margin-bottom: 1;
    }
    .config-option-card {
        width: 100%;
        height: auto;
        border: round #666666;
        padding: 1;
        margin-bottom: 1;
    }
    .config-option-card-selected {
        border: round #ffffff;
        background: rgba(255, 255, 255, 0.06);
    }
    .config-option-header-row {
        width: 100%;
        height: auto;
    }
    .config-option-copy {
        width: 1fr;
        height: auto;
    }
    .config-option-name {
        text-style: bold;
    }
    .config-option-description {
        color: #a0a0a0;
    }
    .config-option-readonly {
        color: #8ea8c7;
    }
    .config-option-control {
        width: 40;
        height: auto;
        align: right top;
    }
    .config-option-control Input,
    .config-option-control Select,
    .config-option-control TextArea {
        width: 100%;
    }
    .config-option-control TextArea {
        height: 5;
    }
    .config-clear-button {
        margin-top: 1;
        width: auto;
        min-width: 10;
    }
    .config-option-error {
        color: #f38ba8;
        width: 100%;
        margin-top: 1;
    }
    """

    READ_ONLY_PATHS = {
        "logging.screenshot_dir",
        "logging.screenshot_stream_dir",
        "browser.user_data_dir",
    }

    _ENUM_FALLBACK_OPTIONS = {
        "model.image_detail": ["low", "high", "auto"],
        "browser.provider_type": ["local", "remote", "persistent", "mock"],
        "execution.wait_for_load_state": ["load", "domcontentloaded", "networkidle"],
    }

    _LIST_TEXTAREA_PATHS = {"browser.extra_args"}

    def __init__(
        self,
        *,
        tab_id: str,
        get_config: Callable[[], Any],
        apply_change: Callable[[str, Any, bool], tuple[bool, str]],
        on_back: Callable[[], None],
        id: Optional[str] = None,
    ) -> None:
        super().__init__(id=id)
        self._tab_id = tab_id
        self._get_config = get_config
        self._apply_change = apply_change
        self._on_back = on_back

        self.tree_id = f"config-tree-{tab_id}"
        self.detail_id = f"config-detail-{tab_id}"
        self.back_button_id = f"back-to-agent-{tab_id}"

        self._tree: Tree | None = None
        self._detail: Container | None = None

        self._sections: dict[str, dict[str, str]] = {}
        self._options_by_section: dict[str, list[dict[str, Any]]] = {}
        self._option_meta: dict[str, dict[str, Any]] = {}

        self._control_to_path: dict[str, str] = {}
        self._clear_to_path: dict[str, str] = {}
        self._error_labels: dict[str, Label] = {}
        self._option_rows: dict[str, Container] = {}

        self._selected_section: str | None = None
        self._selected_path: str | None = None
        self._hydrating_controls: bool = False

    def compose(self) -> ComposeResult:
        with Container(classes="agent-config-root"):
            with Vertical(classes="config-sidebar"):
                yield Label("Config Options", classes="config-sidebar-title")
                yield Tree("Config Options", id=self.tree_id, classes="config-tree")
            with VerticalScroll(classes="config-detail-scroll"):
                yield Container(id=self.detail_id, classes="config-detail-container")

    def on_mount(self) -> None:
        self._tree = self.query_one(f"#{self.tree_id}", Tree)
        self._detail = self.query_one(f"#{self.detail_id}", Container)
        try:
            self._tree.show_root = False
        except Exception:
            pass
        self._load_catalog()
        self._populate_tree()
        self._show_empty_state()

    def refresh_from_config(self) -> None:
        if self._selected_section:
            self._render_section(self._selected_section, focus_path=self._selected_path)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        node_data = getattr(event.node, "data", None)
        if not isinstance(node_data, dict):
            return

        kind = str(node_data.get("kind", ""))
        section = str(node_data.get("section", "")).strip()
        if not section:
            return

        if kind == "section":
            try:
                event.node.expand()
            except Exception:
                pass
            self._selected_section = section
            self._selected_path = None
            self._render_section(section, focus_path=None)
            return

        if kind == "option":
            try:
                if event.node.parent is not None:
                    event.node.parent.expand()
            except Exception:
                pass
            path = str(node_data.get("path", "")).strip()
            if not path:
                return
            self._selected_section = section
            self._selected_path = path
            self._render_section(section, focus_path=path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id or "")
        if button_id == self.back_button_id:
            self._on_back()
            event.stop()
            return

        path = self._clear_to_path.get(button_id)
        if not path:
            return

        success, error_text = self._apply_change(path, None, True)
        if success:
            self._clear_error(path)
            section = _section_for_path(path)
            self._selected_section = section
            self._selected_path = path
            self._render_section(section, focus_path=path)
        else:
            self._set_error(path, error_text or "Unable to clear this setting.")
        event.stop()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if self._hydrating_controls:
            return
        control_id = str(event.switch.id or "")
        path = self._control_to_path.get(control_id)
        if not path or path in self.READ_ONLY_PATHS:
            return
        success, error_text = self._apply_change(path, bool(event.value), False)
        if success:
            self._clear_error(path)
        else:
            self._set_error(path, error_text or "Invalid value.")
            section = _section_for_path(path)
            self._render_section(section, focus_path=path)

    def on_select_changed(self, event: Select.Changed) -> None:
        if self._hydrating_controls:
            return
        control_id = str(event.select.id or "")
        path = self._control_to_path.get(control_id)
        if not path or path in self.READ_ONLY_PATHS:
            return
        selected = event.value
        blank_token = getattr(Select, "BLANK", None)
        if blank_token is not None and selected == blank_token:
            selected = None
        success, error_text = self._apply_change(path, selected, False)
        if success:
            self._clear_error(path)
        else:
            self._set_error(path, error_text or "Invalid value.")
            section = _section_for_path(path)
            self._render_section(section, focus_path=path)

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._hydrating_controls:
            return
        control_id = str(event.input.id or "")
        path = self._control_to_path.get(control_id)
        if not path or path in self.READ_ONLY_PATHS:
            return

        option_meta = self._option_meta.get(path, {})
        control_kind = str(option_meta.get("control_kind", "string"))
        parsed_value, parse_error = self._parse_input_value(control_kind, event.value)
        if parse_error:
            self._set_error(path, parse_error)
            return

        success, error_text = self._apply_change(path, parsed_value, False)
        if success:
            self._clear_error(path)
        else:
            self._set_error(path, error_text or "Invalid value.")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if self._hydrating_controls:
            return
        text_area = event.text_area
        control_id = str(text_area.id or "")
        path = self._control_to_path.get(control_id)
        if not path or path in self.READ_ONLY_PATHS:
            return
        text_value = str(getattr(text_area, "text", "") or "")
        values = [line.strip() for line in text_value.splitlines() if line.strip()]
        success, error_text = self._apply_change(path, values, False)
        if success:
            self._clear_error(path)
        else:
            self._set_error(path, error_text or "Invalid list value.")

    def _load_catalog(self) -> None:
        section_catalog = get_config_section_catalog()
        option_catalog = get_config_option_catalog()

        self._sections = {
            str(item["section"]): {
                "section": str(item["section"]),
                "section_type": str(item.get("section_type", "") or ""),
                "section_name": str(item.get("section_name", "") or ""),
                "section_description": str(item.get("section_description", "") or ""),
            }
            for item in section_catalog
        }
        self._options_by_section = {}
        for option in option_catalog:
            path = str(option.get("path", "")).strip()
            if not path:
                continue
            section = str(option.get("section", "")).strip() or _section_for_path(path)
            if section not in self._sections:
                self._sections[section] = {
                    "section": section,
                    "section_type": f"{_truncate(section, 64)}Config",
                    "section_name": section.capitalize(),
                    "section_description": "",
                }
            option_copy = dict(option)
            option_copy["path"] = path
            option_copy["section"] = section
            option_copy["option_key"] = path.split(".")[-1]
            self._options_by_section.setdefault(section, []).append(option_copy)

        for options in self._options_by_section.values():
            options.sort(key=lambda item: str(item.get("option_key", "")))

    def _populate_tree(self) -> None:
        if self._tree is None:
            return
        root = self._tree.root
        try:
            root.expand()
        except Exception:
            pass

        section_keys = sorted(
            self._options_by_section.keys(),
            key=lambda key: str(self._sections.get(key, {}).get("section_type", key)).lower(),
        )
        for section in section_keys:
            options = self._options_by_section.get(section, [])
            if not options:
                continue
            section_meta = self._sections.get(section, {})
            section_label = str(section_meta.get("section_type", "") or section)
            section_node = root.add(section_label, data={"kind": "section", "section": section})
            for option in options:
                option_key = str(option.get("option_key", "") or option.get("path", ""))
                section_node.add_leaf(
                    option_key,
                    data={
                        "kind": "option",
                        "section": section,
                        "path": str(option.get("path", "")),
                    },
                )

    def _clear_detail(self) -> None:
        if self._detail is None:
            return
        for child in list(self._detail.children):
            try:
                child.remove()
            except Exception:
                pass

    def _show_empty_state(self) -> None:
        if self._detail is None:
            return
        self._clear_detail()
        empty = Vertical(
            Label(
                "Select a config type/option to customise here",
                classes="config-empty-text",
            ),
            Horizontal(
                Button(
                    "Back to Agent",
                    id=self.back_button_id,
                    classes="back-to-agent-button",
                ),
                classes="config-empty-actions",
            ),
            classes="config-empty-state",
        )
        self._detail.mount(empty)

    def _render_section(self, section: str, *, focus_path: str | None) -> None:
        if self._detail is None:
            return

        section_meta = self._sections.get(section)
        if section_meta is None:
            self._show_empty_state()
            return

        self._hydrating_controls = True
        try:
            self._control_to_path.clear()
            self._clear_to_path.clear()
            self._error_labels.clear()
            self._option_rows.clear()
            self._option_meta.clear()
            self._clear_detail()

            section_title = str(section_meta.get("section_type", "") or section)
            section_description = str(section_meta.get("section_description", "") or "").strip()
            if not section_description:
                section_description = f"Configure {section_title} values."

            header = Container(
                Horizontal(
                    Vertical(
                        Label(section_title, classes="config-detail-title"),
                        Label(section_description, classes="config-detail-description"),
                        classes="config-detail-title-wrap",
                    ),
                    Button(
                        "Back to Agent",
                        id=self.back_button_id,
                        classes="back-to-agent-button",
                    ),
                    classes="config-detail-header-row",
                ),
                Label(
                    "Changes apply on next mission only.",
                    classes="config-next-mission-note",
                ),
                classes="config-detail-header",
            )
            self._detail.mount(header)

            options = self._options_by_section.get(section, [])
            current_config = self._get_config()
            for option in options:
                path = str(option.get("path", ""))
                control_kind = self._resolve_control_kind(option, path)
                option_with_kind = dict(option)
                option_with_kind["control_kind"] = control_kind
                self._option_meta[path] = option_with_kind

                copy_widgets: list[Widget] = [
                    Label(str(option.get("option_key", path)), classes="config-option-name")
                ]
                description = str(option.get("description", "") or "").strip()
                if description:
                    copy_widgets.append(Label(description, classes="config-option-description"))
                if path in self.READ_ONLY_PATHS:
                    copy_widgets.append(
                        Label("Read-only during runtime.", classes="config-option-readonly")
                    )
                copy_block = Vertical(*copy_widgets, classes="config-option-copy")

                current_value = self._read_path_value(current_config, path)
                control_widget, clear_button = self._build_control_widget(
                    path=path,
                    option=option,
                    control_kind=control_kind,
                    current_value=current_value,
                )
                control_widgets: list[Widget] = [control_widget]
                if clear_button is not None:
                    control_widgets.append(clear_button)
                control_block = Vertical(*control_widgets, classes="config-option-control")

                header_row = Horizontal(
                    copy_block,
                    control_block,
                    classes="config-option-header-row",
                )
                error_label = Label("", classes="config-option-error")
                row = Container(
                    header_row,
                    error_label,
                    classes="config-option-card",
                )
                if focus_path and path == focus_path:
                    row.add_class("config-option-card-selected")

                self._error_labels[path] = error_label
                self._option_rows[path] = row
                self._detail.mount(row)
        finally:
            self.call_after_refresh(self._finish_hydration)

        if focus_path and focus_path in self._option_rows:
            target = self._option_rows[focus_path]
            self.call_after_refresh(lambda: target.scroll_visible(animate=False))

    def _build_control_widget(
        self,
        *,
        path: str,
        option: dict[str, Any],
        control_kind: str,
        current_value: Any,
    ) -> tuple[Widget, Button | None]:
        control_id = f"cfg-value-{self._tab_id}-{_path_token(path)}"
        self._control_to_path[control_id] = path
        read_only = path in self.READ_ONLY_PATHS
        clear_button: Button | None = None

        if control_kind == "bool":
            widget = Switch(value=bool(current_value), id=control_id)
            widget.disabled = read_only
        elif control_kind == "enum":
            choices = self._enum_choices(path, option)
            current_token = self._enum_token(current_value)
            if current_token and current_token not in choices:
                choices = [*choices, current_token]
            if not choices:
                choices = [current_token] if current_token else [""]
            option_tuples = [(str(choice), str(choice)) for choice in choices]
            initial_value = current_token
            if not initial_value:
                initial_value = str(choices[0])
            if initial_value not in choices:
                lower_lookup = {str(choice).lower(): str(choice) for choice in choices}
                initial_value = lower_lookup.get(initial_value.lower(), str(choices[0]))
            widget = Select(
                options=option_tuples,
                value=initial_value,
                id=control_id,
            )
            widget.disabled = read_only
        elif control_kind == "list":
            lines = []
            if isinstance(current_value, list):
                lines = [str(item) for item in current_value]
            elif current_value is not None:
                lines = [str(current_value)]
            widget = TextArea(text="\n".join(lines), id=control_id)
            widget.disabled = read_only
        else:
            text_value = "" if current_value is None else str(current_value)
            widget = Input(value=text_value, id=control_id)
            widget.disabled = read_only

        constraints = option.get("constraints", {})
        is_nullable = bool(isinstance(constraints, dict) and constraints.get("nullable"))
        if is_nullable and not read_only:
            clear_id = f"cfg-clear-{self._tab_id}-{_path_token(path)}"
            clear_button = Button("Clear", id=clear_id, classes="config-clear-button")
            self._clear_to_path[clear_id] = path

        return widget, clear_button

    def _resolve_control_kind(self, option: dict[str, Any], path: str) -> str:
        type_name = str(option.get("type", "") or "").lower()
        if path in self._LIST_TEXTAREA_PATHS or "array" in type_name:
            return "list"
        if self._enum_choices(path, option):
            return "enum"
        if "bool" in type_name or "boolean" in type_name:
            return "bool"
        if "integer" in type_name:
            return "integer"
        if "number" in type_name:
            return "number"
        return "string"

    def _enum_choices(self, path: str, option: dict[str, Any]) -> list[str]:
        constraints = option.get("constraints")
        if isinstance(constraints, dict):
            enum_values = constraints.get("enum")
            if isinstance(enum_values, list) and enum_values:
                normalized: list[str] = []
                for value in enum_values:
                    token = self._enum_token(value)
                    if token and token not in normalized:
                        normalized.append(token)
                if normalized:
                    return normalized
        fallback = self._ENUM_FALLBACK_OPTIONS.get(path)
        if fallback:
            return list(fallback)
        return []

    @staticmethod
    def _enum_token(value: Any) -> str:
        if value is None:
            return ""
        member_value = getattr(value, "value", None)
        if member_value is not None:
            return str(member_value)
        return str(value)

    @staticmethod
    def _read_path_value(config: Any, path: str) -> Any:
        current = config
        for part in str(path).split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
        return current

    @staticmethod
    def _parse_input_value(control_kind: str, raw: str) -> tuple[Any, str | None]:
        value = str(raw or "")
        if control_kind == "integer":
            trimmed = value.strip()
            if not trimmed:
                return None, "Please enter an integer value."
            try:
                return int(trimmed), None
            except ValueError:
                return None, "Please enter a valid integer."
        if control_kind == "number":
            trimmed = value.strip()
            if not trimmed:
                return None, "Please enter a number."
            try:
                return float(trimmed), None
            except ValueError:
                return None, "Please enter a valid number."
        return value, None

    def _set_error(self, path: str, message: str) -> None:
        label = self._error_labels.get(path)
        if label is not None:
            label.update(str(message or "Invalid value."))

    def _clear_error(self, path: str) -> None:
        label = self._error_labels.get(path)
        if label is not None:
            label.update("")

    def _finish_hydration(self) -> None:
        self._hydrating_controls = False

class TelemetryPanel(AgentPanel):
    """Right-side panel showing raw agent telemetry data."""

    DEFAULT_CSS = """
    TelemetryPanel {
        width: 100%;
        height: 1fr;
        min-height: 0;
        border: solid darkgray;
        padding-left: 2;
        padding-top: 1;
        padding-bottom: 1
    }

    TelemetryPanel VerticalScroll {
        width: 100%;
        height: 100%;
    }

    TelemetryPanel Label {
        width: 100%;
        height: auto;
    }

    .telemetry-intro {
        align: left bottom;
    }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="telemetry-scroll"):
            yield Label(
                "Telemetry: waiting for mission state",
                markup=False,
                classes="telemetry-label",
            )
            with Container(classes="telemetry-intro"):
                yield Label("[gray]Internal agent state will appear here[/gray]")
                yield Label("[darkgray]TELEMETRY[/darkgray]")

    def panel_ready(self) -> None:
        self._label = self.query_one(".telemetry-label", Label)
        self._scroll = self.query_one(".telemetry-scroll", VerticalScroll)
        self._intro = self.query_one(".telemetry-intro", Container)
        self._label.styles.width = "100%"
        self._label.styles.height = "auto"

    def on_state_update(self, state: AgentState) -> None:
        if state.agent_attached:
            self._intro.display = False
        self._label.update(self._build_text(state))
        self._scroll.refresh(layout=True)

    def _build_text(self, state: AgentState) -> str:
        if not state.agent_attached and state.status == "idle":
            return "\n".join(
                [
                    f"Status: {state.status}",
                    f"Message: {state.last_message or 'n/a'}",
                    f"Queue: {state.queue_depth}",
                    f"Worker: {'running' if state.worker_alive else 'stopped'}",
                    "Iteration: n/a",
                    "Budget: 0/0 (phase: normal)",
                    "Cost: $0.0000",
                    "Tokens: 0",
                    f"Sandbox Web: {state.sandbox_web_policy or 'n/a'}",
                    f"Dropped Events: {state.dropped_event_count}",
                    "Hint: enter a mission and press Run.",
                ]
            )

        loop_line = "off"
        if state.in_loop:
            loop_line = f"round {state.loop_round}/{state.loop_count or '?'} ({state.loop_description or ''})"

        result_line = (
            "Result: n/a"
            if state.mission_success is None
            else f"Result: {'success' if state.mission_success else 'failed'}"
        )

        current_url = state.last_known_url or state.mission_final_url or "n/a"

        return "\n".join(
            [
                f"Status: {state.status}",
                f"Message: {state.last_message or 'n/a'}",
                f"Queue: {state.queue_depth}",
                f"Worker: {'running' if state.worker_alive else 'stopped'}",
                f"Agent: {'attached' if state.agent_attached else 'no'}",
                f"Iteration: {state.current_iteration or '-'}",
                f"Paused: {'yes' if state.paused else 'no'}",
                f"Cancel Req: {'yes' if state.cancel_requested else 'no'}",
                f"Loop: {loop_line}",
                f"Checkpoint: {'pending' if state.checkpoint_pending else 'clear'}",
                f"Actions: {state.total_actions}",
                f"Progress Gap: {state.actions_since_progress}",
                f"User Actions: {state.user_facing_actions_since_progress}",
                f"Budget: {state.budget_remaining}/{state.budget_total} remaining (spent={state.budget_spent}, phase={state.budget_phase})",
                f"Budget Constraints: {'on' if state.budget_constraints_enabled else 'off'}",
                f"Low Budget Mode: {'on' if state.low_budget_mode else 'off'}",
                f"Plan Batch Limit: {state.planning_batch_limit or '-'}",
                f"Cost: ${state.llm_total_cost_usd:.4f}",
                f"Tokens (all): {state.llm_total_tokens}",
                f"Iteration ms: {state.iteration_ms:.2f}",
                f"LLM ms: {state.llm_latency_ms:.2f}",
                f"Tool ms: {state.tool_latency_ms:.2f}",
                f"Navigation ms: {state.navigation_latency_ms:.2f}",
                f"Tokens in/out (iter): {state.tokens_in}/{state.tokens_out}",
                f"Images (iter): {state.image_count}",
                f"Tool calls (iter): {state.tool_calls}",
                f"Retries (iter): {state.retries}",
                f"Avg iteration ms: {state.avg_iteration_ms:.2f}",
                f"P95 iteration ms: {state.p95_iteration_ms:.2f}",
                f"Avg LLM ms: {state.avg_llm_ms:.2f}",
                f"Avg tool ms: {state.avg_tool_ms:.2f}",
                f"Avg tokens in/out: {state.avg_tokens_in:.2f}/{state.avg_tokens_out:.2f}",
                f"Avg images/call: {state.avg_images_per_call:.2f}",
                f"Retries/mission: {state.retries_per_mission:.2f}",
                f"Mission ms: {state.mission_ms:.2f}",
                f"Sandbox Web: {state.sandbox_web_policy or 'n/a'}",
                f"URL: {current_url}",
                f"Last Action: {state.last_action_summary or 'n/a'}",
                f"Failure code: {state.failure_code or 'n/a'}",
                f"Failure stage: {state.failure_stage or 'n/a'}",
                f"Dropped Events: {state.dropped_event_count}",
                result_line,
            ]
        )


class MissionControls(AgentPanel):
    """Mission input field + Run button. IDs are tab-scoped for multi-tab support."""

    DEFAULT_CSS = """
    MissionControls { height: auto; width: 99%; }
    MissionControls Horizontal { height: auto; }
    MissionControls Input { width: 1fr; }
    MissionControls .choice-area { height: auto; width: 100%; }
    MissionControls .choice-options { height: auto; width: 100%; }
    MissionControls .choice-actions { height: auto; width: 100%; }
    MissionControls .choice-question { color: #D4D4D4; padding: 0 0 1 0; }
    """

    def __init__(self, tab_id: str) -> None:
        super().__init__()
        self._tab_id = tab_id
        self.input_id = f"mission-input-{tab_id}"
        self.button_id = f"run-mission-{tab_id}"
        self.main_controls_id = f"main-controls-{tab_id}"
        self.choice_area_id = f"choice-area-{tab_id}"
        self._confirm_button_id = f"choice-confirm-{tab_id}"
        self._selected_indices: set[int] = set()
        self._current_options: list[str] = []
        self._current_multi_select: bool = False
        self._current_yes_no: bool = False
        self._choice_render_key: tuple[Any, ...] = ()

    def compose(self) -> ComposeResult:
        with Horizontal(classes="controls", id=self.main_controls_id):
            yield Input(
                placeholder="Describe the mission for this agent...",
                id=self.input_id,
            )
            yield Button("RUN", id=self.button_id)
        yield Container(id=self.choice_area_id, classes="choice-area")

    def panel_ready(self) -> None:
        inp = self.query_one(f"#{self.input_id}", Input)
        btn = self.query_one(f"#{self.button_id}", Button)
        area = self.query_one(f"#{self.choice_area_id}", Container)
        inp.styles.border = ("solid", "#9C9C9C")
        inp.styles.background = "transparent"
        inp.styles.height = "auto"
        inp.styles.padding = (0, 1, 0, 1)
        btn.styles.width = "auto"
        btn.styles.height = "3"
        btn.styles.border = ("solid", "green")
        area.display = False

    def clear_input(self) -> None:
        try:
            inp = self.query_one(f"#{self.input_id}", Input)
            inp.value = ""
        except Exception:
            pass

    def get_option_text(self, option_index: int) -> str | None:
        if option_index < 0 or option_index >= len(self._current_options):
            return None
        return self._current_options[option_index]

    def toggle_option(self, option_index: int) -> None:
        if not self._current_multi_select:
            return
        if option_index < 0 or option_index >= len(self._current_options):
            return
        if option_index in self._selected_indices:
            self._selected_indices.remove(option_index)
        else:
            self._selected_indices.add(option_index)
        self._sync_choice_button_styles()
        self._sync_confirm_button_state()

    def get_selected_options(self) -> list[str]:
        if not self._current_options:
            return []
        selected: list[str] = []
        for idx in sorted(self._selected_indices):
            if 0 <= idx < len(self._current_options):
                selected.append(self._current_options[idx])
        return selected

    def _set_mode_widgets(self, *, show_main_controls: bool) -> tuple[Input, Button, Container]:
        inp = self.query_one(f"#{self.input_id}", Input)
        btn = self.query_one(f"#{self.button_id}", Button)
        main_controls = self.query_one(f"#{self.main_controls_id}", Horizontal)
        choice_area = self.query_one(f"#{self.choice_area_id}", Container)
        main_controls.display = show_main_controls
        inp.display = show_main_controls
        btn.display = show_main_controls
        return inp, btn, choice_area

    def _clear_choice_area(self, choice_area: Container) -> None:
        try:
            choice_area.remove_children()
        except Exception:
            for child in list(choice_area.children):
                try:
                    child.remove()
                except Exception:
                    pass
        self._selected_indices.clear()
        self._current_options = []
        self._current_multi_select = False
        self._current_yes_no = False
        self._choice_render_key = ()

    def _style_choice_button(self, button: Button, option_text: str, selected: bool) -> None:
        _ = option_text
        base_color = "#8C8C8C"
        selected_color = "#2563EB"
        selected_text = "white"
        if selected:
            button.styles.border = ("solid", selected_color)
            button.styles.background = selected_color
            button.styles.color = selected_text
        else:
            button.styles.border = ("solid", base_color)
            button.styles.background = "transparent"
            button.styles.color = "white"

    def _sync_choice_button_styles(self) -> None:
        for idx, option_text in enumerate(self._current_options):
            button_id = f"choice-opt-{idx}-{self._tab_id}"
            try:
                button = self.query_one(f"#{button_id}", Button)
            except Exception:
                continue
            self._style_choice_button(button, option_text, idx in self._selected_indices)

    def _sync_confirm_button_state(self) -> None:
        try:
            confirm_btn = self.query_one(f"#{self._confirm_button_id}", Button)
        except Exception:
            return
        confirm_btn.disabled = len(self._selected_indices) == 0

    def _rebuild_choice_area(self, state: AgentState, choice_area: Container) -> None:
        self._clear_choice_area(choice_area)
        self._current_options = [str(opt) for opt in state.pending_options]
        self._current_multi_select = bool(state.pending_multi_select)
        self._current_yes_no = bool(state.pending_yes_no)

        question_text = str(state.pending_question or "Choose an option")
        question_label = Label(
            f"Question: {_truncate(question_text, 120)}",
            classes="choice-question",
        )
        choice_area.mount(question_label)

        options_row = Horizontal(classes="choice-options")
        choice_area.mount(options_row)
        for idx, option_text in enumerate(self._current_options):
            button = Button(option_text, id=f"choice-opt-{idx}-{self._tab_id}")
            button.styles.height = "3"
            if self._current_yes_no:
                button.styles.width = "8"
            else:
                button.styles.width = "auto"
            self._style_choice_button(button, option_text, selected=False)
            options_row.mount(button)
        if state.allow_skip:
            skip_btn = Button("SKIP", id=f"choice-skip-{self._tab_id}")
            skip_btn.styles.height = "3"
            skip_btn.styles.width = "8"
            skip_btn.styles.border = ("solid", "#8C8C8C")
            options_row.mount(skip_btn)

        needs_actions_row = self._current_multi_select
        if needs_actions_row:
            actions_row = Horizontal(classes="choice-actions")
            choice_area.mount(actions_row)
            if self._current_multi_select:
                confirm_btn = Button("CONFIRM", id=self._confirm_button_id)
                confirm_btn.styles.height = "3"
                confirm_btn.styles.width = "auto"
                confirm_btn.styles.border = ("solid", "#60A5FA")
                confirm_btn.disabled = True
                actions_row.mount(confirm_btn)

    def on_state_update(self, state: AgentState) -> None:
        try:
            inp, btn, choice_area = self._set_mode_widgets(show_main_controls=True)
        except Exception:
            return

        is_asking = state.status == "asking"
        has_options = bool(state.pending_options)

        if not is_asking:
            choice_area.display = False
            self._clear_choice_area(choice_area)
            inp.placeholder = "Describe the mission for this agent..."
            btn.label = "RUN"
            btn.styles.border = ("solid", "green")
            return

        if not has_options:
            choice_area.display = False
            self._clear_choice_area(choice_area)
            if state.pending_question:
                inp.placeholder = f"Answer: {_truncate(state.pending_question, 72)}"
            elif state.allow_skip:
                inp.placeholder = "Type your answer, or leave empty to skip..."
            else:
                inp.placeholder = "Type your answer..."
            btn.label = "ANSWER"
            btn.styles.border = ("solid", "#60A5FA")
            return

        choice_area.display = True
        render_key = (
            state.pending_question,
            state.pending_options,
            state.pending_multi_select,
            state.pending_yes_no,
            state.allow_custom,
            state.allow_skip,
        )
        if render_key != self._choice_render_key:
            self._rebuild_choice_area(state, choice_area)
            self._choice_render_key = render_key
        self._sync_choice_button_styles()
        self._sync_confirm_button_state()

        if state.allow_custom:
            inp.placeholder = (
                "Custom answer (blank to skip)..."
                if state.allow_skip
                else "Custom answer..."
            )
            btn.label = "ANSWER"
            btn.styles.border = ("solid", "#60A5FA")
            self._set_mode_widgets(show_main_controls=True)
        else:
            self._set_mode_widgets(show_main_controls=False)
