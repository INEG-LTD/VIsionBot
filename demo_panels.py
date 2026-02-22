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
    "cancel requested": (" CANCELLING ", "cancel-requested"),
    "error": (" ERROR ", "error"),
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
    """Horizontal row: colored status pill + thinking status label."""

    DEFAULT_CSS = """
    StatusRow { height: auto; width: 99%; }
    StatusRow Horizontal { height: 1; width: 100%; }
    .thinking-label {
        color: #F0D264;
        text-align: right;
        width: 1fr;
    }
    """
    
    label = "IDLE"

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield StatusPill(label=self.label, variant="idle")
            yield Label("THINKING: idle", classes="thinking-label")

    def panel_ready(self) -> None:
        self._pill = self.query_one(StatusPill)
        self._thinking = self.query_one(".thinking-label", Label)

    def on_state_update(self, state: AgentState) -> None:
        status_key = str(state.status or "").strip().lower()
        fallback_label = status_key.upper() if status_key else "UNKNOWN"
        label_text, variant = _STATUS_META.get(status_key, (fallback_label, "unknown"))
        self._pill.set_status(label_text, variant)
        self.label = label_text
        if state.thinking_active:
            frame = state.render_frame % 4
            dots = "." * frame + " " * (3 - frame)
            text = _truncate(state.thinking_text or "Thinking...", 84)
            self._thinking.update(f"THINKING: 🤔 {text}{dots}")
        else:
            self._thinking.update("")


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
    TimelinePanel {
        height: 1fr; 
        min-height: 0; 
        width: 99%;
        layers: below above;
    }
    TimelinePanel VerticalScroll {
        height: 1fr; 
        border: solid #9C9C9C;
        padding: 0 0 0 1; 
        overflow-y: auto;
    }
    
    .timeline-events {
        layers: above;
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
        with VerticalScroll(classes="timeline-scroll"):
            yield Container(classes="timeline-events")
            with Container(classes="timeline-intro-text"):
                yield Label(
                    "[gray]The history of agent work will appear here[/gray]"
                )
                yield Label("[darkgray]TIMELINE[/darkgray]", classes="timeline-intro-text-label")

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
                    text=f"Your agent is currently: {details.get('mission', 'unknown')}",
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
        self.settings_button_id = f"settings-button-{tab_id}"
        self.config_button_id = f"open-config-{tab_id}"

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Button("Settings", classes="settings-button", id=self.settings_button_id)
            yield Button("Config", classes="config-button", id=self.config_button_id)
            
    def panel_ready(self) -> None:
        self._settings_button = self.query_one(".settings-button", Button)
        self._config_button = self.query_one(".config-button", Button)
        self._settings_button.styles.border = ("round", "white")
        self._config_button.styles.border = ("round", "white")
        
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == self.settings_button_id:
            self._settings_button.pressed = True
            self._config_button.pressed = False
        elif event.button.id == self.config_button_id:
            self._config_button.pressed = True
            self._settings_button.pressed = False


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
        "error_handling.screenshot_dir",
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
        overflow-y: auto;
        border: solid darkgray; 
        padding-left: 2;
        padding-top: 1;
        padding-bottom: 1
    }
    
    .telemetry-intro {
        align: left bottom;
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
                    "Budget: 0/0 (phase: normal)",
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
                f"Budget: {state.budget_remaining}/{state.budget_total} remaining (spent={state.budget_spent}, phase={state.budget_phase})",
                f"Budget Constraints: {'on' if state.budget_constraints_enabled else 'off'}",
                f"Low Budget Mode: {'on' if state.low_budget_mode else 'off'}",
                f"Plan Batch Limit: {state.planning_batch_limit or '-'}",
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
        btn.styles.width = "auto"
        btn.styles.height = "3"
        # btn.styles.background = "transparent"
        btn.styles.border = ("solid", "green")
