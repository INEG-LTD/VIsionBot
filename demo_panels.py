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
from typing import Callable

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Input, Label

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
    thinking_active: bool = False
    thinking_text: str = "Thinking..."
    render_frame: int = 0  # incremented by poll timer to drive animations


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


_STATUS_COLORS: dict[str, tuple[str, str]] = {
    "idle": ("IDLE", "red"),
    "running": ("RUNNING", "#4BF538"),
    "completed": ("SUCCESS", "#4BF538"),
    "failed": ("FAILED", "red"),
    "stopping": ("STOPPING", "#FF9798"),
    "stopped": ("STOPPED", "red"),
    "paused": ("PAUSED", "yellow"),
    "queued": ("QUEUED", "#F5A623"),
    "cancel requested": ("CANCELLING", "#FF9798"),
    "error": ("ERROR", "red"),
}


# ---------------------------------------------------------------------------
# TimelineEntry — immutable record produced from a BotEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimelineEntry:
    kind: str  # "line" or "collapsible"
    text: str = ""
    title: str = ""
    body: str = ""


# ---------------------------------------------------------------------------
# Concrete panels
# ---------------------------------------------------------------------------


class StatusRow(AgentPanel):
    """Horizontal row: colored status pill + thinking status label."""

    DEFAULT_CSS = """
    StatusRow { height: auto; width: 99%; }
    StatusRow Horizontal { height: auto; }
    StatusRow .status-pill {
        height: 1; width: auto; padding: 0 1;
        background: red; color: white;
    }
    StatusRow .thinking-label {
        height: auto; width: 1fr; color: #F0D264;
        padding: 0 1; text-align: right;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label("[bold]IDLE[/bold]", classes="status-pill")
            yield Label("THINKING: idle", markup=False, classes="thinking-label")

    def panel_ready(self) -> None:
        self._pill = self.query_one(".status-pill", Label)
        self._thinking = self.query_one(".thinking-label", Label)

    def on_state_update(self, state: AgentState) -> None:
        label_text, color = _STATUS_COLORS.get(
            state.status, (state.status.upper(), "gray")
        )
        self._pill.update(f"[bold]{label_text}[/bold]")
        self._pill.styles.background = color

        if state.thinking_active:
            frame = state.render_frame % 4
            dots = "." * frame + " " * (3 - frame)
            text = _truncate(state.thinking_text or "Thinking...", 84)
            self._thinking.update(f"THINKING: 🤔 {text}{dots}")
        else:
            self._thinking.update("THINKING: idle")


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

    listens_to = [
        EventType.ITERATION_START,
        EventType.ACTION_DETERMINED,
        EventType.LOOP_STATE_CHANGED,
        EventType.ASK_REQUESTED,
        EventType.SYSTEM_ERROR,
    ]

    DEFAULT_CSS = """
    TimelinePanel { height: 1fr; min-height: 0; width: 99%; }
    TimelinePanel VerticalScroll {
        height: 1fr; border: solid #9C9C9C;
        padding: 0 0 0 1; overflow-y: auto;
    }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="timeline-scroll"):
            yield Container(classes="timeline-events")
            with Container(classes="timeline-intro-text"):
                yield Label(
                    "[gray]The history of agent work will appear here[/gray]"
                )
                yield Label("[darkgray]TIMELINE[/darkgray]")

    def panel_ready(self) -> None:
        self._scroll = self.query_one(".timeline-scroll", VerticalScroll)
        self._events = self.query_one(".timeline-events", Container)
        self._intro = self.query_one(".timeline-intro-text", Container)

    def on_agent_event(self, event: BotEvent) -> None:
        entry = self._format_event(event)
        if entry is None:
            return

        self._intro.display = False

        if entry.kind == "collapsible":
            widget: Widget = Collapsible(
                Label(entry.body, markup=False, classes="collapsible-entry-body"),
                title=entry.title or "Details",
                collapsed=True,
                classes="collapsible-entry",
            )
        else:
            widget = Label(entry.text or "", markup=False)

        self._events.mount(widget)
        self._scroll.scroll_end(animate=False)

    def _format_event(self, event: BotEvent) -> TimelineEntry | None:
        details = event.details or {}
        t = event.event_type

        if t == EventType.ITERATION_START:
            if details.get("iteration", 0) == 1:
                return TimelineEntry(
                    kind="line",
                    text=f"Mission: {details.get('mission', 'unknown')}",
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
                kind="line",
                text=f"{question}\nThe agent is asking a question. Please respond...",
            )

        elif t == EventType.SYSTEM_ERROR:
            return TimelineEntry(kind="line", text=f"ERROR {event.message}")

        return None


class TelemetryPanel(AgentPanel):
    """Right-side panel showing raw agent telemetry data."""

    DEFAULT_CSS = """
    TelemetryPanel {
        width: 48; height: 100%; overflow-y: auto;
        border: solid darkgray; padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
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
        self._intro = self.query_one(".telemetry-intro", Container)

    def on_state_update(self, state: AgentState) -> None:
        if state.agent_attached:
            self._intro.display = False
        self._label.update(self._build_text(state))

    def _build_text(self, state: AgentState) -> str:
        if not state.agent_attached and state.status == "idle":
            return "\n".join(
                [
                    f"Status: {state.status}",
                    f"Message: {_truncate(state.last_message, 64)}",
                    f"Queue: {state.queue_depth}",
                    f"Worker: {'running' if state.worker_alive else 'stopped'}",
                    "Iteration: n/a",
                    "Cost: $0.0000",
                    "Tokens: 0",
                    f"Dropped Events: {state.dropped_event_count}",
                    "Hint: enter a mission and press Run.",
                ]
            )

        loop_line = "off"
        if state.in_loop:
            desc = _truncate(state.loop_description, 54)
            loop_line = f"round {state.loop_round}/{state.loop_count or '?'} ({desc})"

        result_line = (
            "Result: n/a"
            if state.mission_success is None
            else f"Result: {'success' if state.mission_success else 'failed'}"
        )

        current_url = _truncate(state.last_known_url or state.mission_final_url, 70)

        return "\n".join(
            [
                f"Status: {state.status}",
                f"Message: {_truncate(state.last_message, 64)}",
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
                f"Cost: ${state.llm_total_cost_usd:.4f}",
                f"Tokens: {state.llm_total_tokens}",
                f"URL: {current_url}",
                f"Last Action: {_truncate(state.last_action_summary, 70)}",
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
    """

    def __init__(self, tab_id: str) -> None:
        super().__init__()
        self._tab_id = tab_id
        self.input_id = f"mission-input-{tab_id}"
        self.button_id = f"run-mission-{tab_id}"

    def compose(self) -> ComposeResult:
        with Horizontal(classes="controls"):
            yield Input(
                placeholder="Describe the mission for this agent...",
                id=self.input_id,
            )
            yield Button("RUN", id=self.button_id)

    def panel_ready(self) -> None:
        inp = self.query_one(f"#{self.input_id}", Input)
        btn = self.query_one(f"#{self.button_id}", Button)
        inp.styles.border = ("solid", "#9C9C9C")
        inp.styles.background = "transparent"
        inp.styles.height = "auto"
        inp.styles.padding = (0, 1, 0, 1)
        btn.styles.width = 16
        btn.styles.height = "3"
        btn.styles.background = "transparent"
        btn.styles.border = ("solid", "#9C9C9C")
