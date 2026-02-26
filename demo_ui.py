"""demo_ui.py — Agent session management, layout, and the main Textual app.

You normally only need to edit this file to:
  - Change the overall layout (add a sidebar, rearrange panes)
  - Modify how agent sessions are started, stopped, or reset
  - Add new keyboard bindings

To add a new panel:    edit demo_panels.py, then add `yield MyPanel()` in AgentView.compose()
To change config/callbacks: edit demo_config.py
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, ContentSwitcher, Footer, Input, Label, Tab, Tabs

from agent.agent_controller import Agent
from core.agent_workspace import AgentWorkspaceManager
from core.config import Config
from core.executor import Executor
from core.sandbox_policy import SandboxPolicyEngine
from utils.debug_print import (
    DebugPrintRecord,
    register_print_callback,
    unregister_print_callback,
)
from utils.event_logger import BotEvent, EventType, LogLevel

from demo_config import (
    STARTING_URL,
    config as base_config,
    on_data_reported,
    setup_interceptors,
)
from demo_panels import (
    AgentConfigEditor,
    AgentState,
    ConfigButtons,
    EventStream,
    IntroBanner,
    MissionControls,
    ReactiveState,
    StatusRow,
    TelemetryPanel,
    TimelinePanel,
)


def _clone_base_config() -> Config:
    """Create an isolated per-agent config from demo_config defaults."""
    try:
        return base_config.model_copy(deep=True)
    except AttributeError:
        # Compatibility fallback for older Pydantic versions.
        return base_config.copy(deep=True)


@dataclass(frozen=True)
class LoadCandidate:
    """One resumable agent+run pair shown in the load picker."""

    agent_id: str
    run_id: str
    mission: str
    status: str
    active: bool
    started_at: float


class LoadAgentScreen(ModalScreen[LoadCandidate | None]):
    """Simple modal picker for selecting an agent run to load."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, candidates: list[LoadCandidate]) -> None:
        super().__init__()
        self._candidates = list(candidates)

    @staticmethod
    def _candidate_label(index: int, item: LoadCandidate) -> str:
        active_marker = "active" if item.active else "inactive"
        mission_preview = (item.mission or "").strip().replace("\n", " ")
        if len(mission_preview) > 88:
            mission_preview = f"{mission_preview[:85]}..."
        mission_text = mission_preview or "(no mission text)"
        return (
            f"[{index + 1}] {item.agent_id} / {item.run_id} "
            f"({item.status or 'unknown'}, {active_marker})\n{mission_text}"
        )

    def compose(self) -> ComposeResult:
        with Container(id="load-agent-modal"):
            yield Label("Load Agent Checkpoint", id="load-agent-title")
            if not self._candidates:
                yield Label("No resumable checkpoints were found.")
            else:
                yield Label("Choose which checkpoint to load:")
                for idx, item in enumerate(self._candidates):
                    yield Button(
                        self._candidate_label(idx, item),
                        id=f"load-candidate-{idx}",
                    )
            yield Button("Cancel", id="load-cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id or "")
        if button_id == "load-cancel":
            self.dismiss(None)
            return
        if not button_id.startswith("load-candidate-"):
            return
        raw_index = button_id.replace("load-candidate-", "", 1)
        try:
            idx = int(raw_index)
        except ValueError:
            return
        if idx < 0 or idx >= len(self._candidates):
            return
        self.dismiss(self._candidates[idx])


# ---------------------------------------------------------------------------
# ThinkingBorderManager — animated blue border on the browser page
# ---------------------------------------------------------------------------


class ThinkingBorderManager:
    _JS_INIT = """
    (function() {
        if (window.__agentThinkingBorder) return;
        window.__agentThinkingBorder = {
            overlay: null,
            blockingOverlay: null,
            init: function() {
                if (this.overlay) return;
                const style = document.createElement('style');
                style.textContent = `
                    @keyframes agent-thinking-pulse {
                        0%, 100% { box-shadow: inset 0 0 60px 20px rgba(59, 130, 246, 0.6); }
                        50% { box-shadow: inset 0 0 80px 30px rgba(59, 130, 246, 0.8); }
                    }
                    @keyframes agent-thinking-fadeout {
                        from { opacity: 1; }
                        to { opacity: 0; }
                    }
                `;
                document.head.appendChild(style);

                this.overlay = document.createElement('div');
                this.overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;pointer-events:none;z-index:2147483647;display:none;opacity:0;';
                document.body.appendChild(this.overlay);

                this.blockingOverlay = document.createElement('div');
                this.blockingOverlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:transparent;pointer-events:auto;z-index:2147483646;display:none;cursor:not-allowed;';
                document.body.appendChild(this.blockingOverlay);
            },
            start: function() {
                this.init();
                this.overlay.style.display = 'block';
                this.overlay.style.opacity = '1';
                this.overlay.style.animation = 'agent-thinking-pulse 1.5s ease-in-out infinite';
            },
            stop: function() {
                if (!this.overlay) return;
                this.overlay.style.animation = 'agent-thinking-fadeout 0.5s ease-out forwards';
                setTimeout(() => { this.overlay.style.display = 'none'; }, 500);
            },
            enableBlocking: function() {
                this.init();
                this.blockingOverlay.style.display = 'block';
            },
            disableBlocking: function() {
                if (!this.blockingOverlay) return;
                this.blockingOverlay.style.display = 'none';
            }
        };
    })();
    """

    def __init__(self, agent: Agent):
        self.agent = agent
        self._enabled = not agent.config.logging.debug_mode
        self._last_page_id = None

    def _ensure_init(self):
        if not self._enabled or not self.agent.browser.page:
            return
        page_id = id(self.agent.browser.page)
        needs_init = self._last_page_id != page_id
        if not needs_init:
            try:
                needs_init = self.agent.browser.page.evaluate(
                    "() => !window.__agentThinkingBorder || !window.__agentThinkingBorder.overlay"
                )
            except Exception:
                needs_init = True
        if needs_init:
            try:
                self.agent.browser.page.evaluate(self._JS_INIT)
            except Exception:
                pass
        self._last_page_id = page_id

    def start(self):
        if not self._enabled:
            return
        self._ensure_init()
        try:
            self.agent.browser.page.evaluate(
                "if(window.__agentThinkingBorder) window.__agentThinkingBorder.start();"
            )
        except Exception:
            pass

    def stop(self):
        if not self._enabled:
            return
        try:
            self.agent.browser.page.evaluate(
                "if(window.__agentThinkingBorder) window.__agentThinkingBorder.stop();"
            )
        except Exception:
            pass

    def enable_blocking(self):
        if not self._enabled:
            return
        self._ensure_init()
        try:
            self.agent.browser.page.evaluate(
                "if(window.__agentThinkingBorder) window.__agentThinkingBorder.enableBlocking();"
            )
        except Exception:
            pass
        try:
            self.start()
        except Exception:
            pass

    def disable_blocking(self):
        if not self._enabled:
            return
        try:
            self.agent.browser.page.evaluate(
                "if(window.__agentThinkingBorder) window.__agentThinkingBorder.disableBlocking();"
            )
        except Exception:
            pass


def apply_thinking_border(agent: Agent) -> ThinkingBorderManager:
    """Patch Agent and Executor to show a pulsing blue border while thinking."""
    manager = ThinkingBorderManager(agent)
    agent._thinking_border_manager = manager
    try:
        agent.browser._thinking_border_manager = manager
    except Exception:
        pass

    if hasattr(agent.browser.page, "on"):
        def _handle_frame_navigation(frame):
            try:
                if frame != agent.browser.page.main_frame:
                    return
                manager._last_page_id = None
                manager.start()
            except Exception:
                pass
        try:
            agent.browser.page.on("framenavigated", _handle_frame_navigation)
        except Exception:
            pass

    if not getattr(Agent, "_thinking_border_capture_patched", False):
        original_capture = Agent._capture_snapshot

        def patched_capture(self, *args, **kwargs):
            current_manager = getattr(self, "_thinking_border_manager", None)
            if current_manager:
                current_manager.start()
            return original_capture(self, *args, **kwargs)

        Agent._capture_snapshot = patched_capture
        Agent._thinking_border_capture_patched = True

    if not getattr(Executor, "_thinking_border_act_patched", False):
        original_act = Executor.act

        def patched_act(self, *args, **kwargs):
            current_manager = getattr(self.browser, "_thinking_border_manager", None)
            if current_manager:
                current_manager.stop()
                current_manager.disable_blocking()
            try:
                return original_act(self, *args, **kwargs)
            finally:
                if current_manager:
                    current_manager.enable_blocking()

        Executor.act = patched_act
        Executor._thinking_border_act_patched = True

    if not getattr(Executor, "_thinking_border_extract_patched", False):
        original_extract = Executor.extract

        def patched_extract(self, *args, **kwargs):
            current_manager = getattr(self.browser, "_thinking_border_manager", None)
            if current_manager:
                current_manager.stop()
                current_manager.disable_blocking()
            try:
                return original_extract(self, *args, **kwargs)
            finally:
                if current_manager:
                    current_manager.enable_blocking()

        Executor.extract = patched_extract
        Executor._thinking_border_extract_patched = True

    if not getattr(Agent, "_thinking_border_run_loop_patched", False):
        original_run_loop = Agent._run_execution_loop

        def patched_run_loop(self, *args, **kwargs):
            current_manager = getattr(self, "_thinking_border_manager", None)
            if current_manager:
                current_manager.enable_blocking()
            try:
                return original_run_loop(self, *args, **kwargs)
            finally:
                if current_manager:
                    current_manager.disable_blocking()

        Agent._run_execution_loop = patched_run_loop
        Agent._thinking_border_run_loop_patched = True

    return manager


# ---------------------------------------------------------------------------
# AgentSession — per-tab runtime state (lives on the main thread)
# ---------------------------------------------------------------------------


@dataclass
class AgentSession:
    """Per-tab runtime state. Populated gradually as missions execute."""

    tab_id: str
    label: str
    config: Config = field(default_factory=_clone_base_config)
    config_revision: int = 0
    agent_config_revision: int = -1
    status: str = "idle"
    agent: Agent | None = None
    agent_thread_id: int | None = None
    load_agent_id: str = ""
    load_run_id: str = ""
    resume_ready: bool = False
    current_mission: str = ""
    worker_thread: threading.Thread | None = None
    mission_queue: queue.Queue = field(default_factory=queue.Queue)
    worker_stop_event: threading.Event = field(default_factory=threading.Event)
    last_snapshot: dict | None = None
    last_known_url: str = ""
    last_message: str = "Ready"
    dropped_event_count: int = 0
    thinking_active: bool = False
    thinking_text: str = "Thinking..."
    pending_question: str = ""
    pending_options: list[str] = field(default_factory=list)
    pending_multi_select: bool = False
    pending_yes_no: bool = False
    pending_answer: str = ""
    awaiting_answer: bool = False
    answer_event: threading.Event = field(default_factory=threading.Event)
    # Wired by AgentView after it is mounted:
    reactive: ReactiveState | None = None
    stream: EventStream | None = None

    def answer_question(self, text: str) -> None:
        """Unblock a waiting ask callback with user-provided text (or empty to skip)."""
        self.pending_answer = str(text or "").strip()
        self.pending_question = ""
        self.pending_options = []
        self.pending_multi_select = False
        self.pending_yes_no = False
        self.awaiting_answer = False
        if self.status == "asking":
            self.status = "running"
        self.answer_event.set()


def _make_session_question_callback(session: AgentSession) -> Callable[[str, dict, list[str], bool, bool], str]:
    """Build a per-session ask callback that blocks until the UI submits an answer."""

    def _on_user_question(
        question: str,
        context: dict,
        options: list[str] | None = None,
        multi_select: bool = False,
        yes_no: bool = False,
    ) -> str:
        _ = context  # reserved for future context-aware prompts
        normalized_options = [str(opt).strip() for opt in (options or []) if str(opt).strip()]
        yes_no_mode = bool(yes_no)
        if yes_no_mode:
            normalized_options = ["Yes", "No"]
            multi_select = False
        session.pending_options = normalized_options
        session.pending_multi_select = bool(multi_select and normalized_options)
        session.pending_yes_no = yes_no_mode
        session.pending_question = str(question or "").strip()
        session.awaiting_answer = True

        while not session.answer_event.wait(timeout=0.1):
            if session.worker_stop_event.is_set():
                session.pending_question = ""
                session.pending_options = []
                session.pending_multi_select = False
                session.pending_yes_no = False
                session.awaiting_answer = False
                session.pending_answer = ""
                session.answer_event.clear()
                return ""

        answer = session.pending_answer
        session.pending_answer = ""
        session.awaiting_answer = False
        session.answer_event.clear()
        return answer

    return _on_user_question


# ---------------------------------------------------------------------------
# AgentView — layout container; exposes _reactive_state and _event_stream
#             so AgentPanel children can auto-subscribe
# ---------------------------------------------------------------------------


class AgentView(Vertical):
    """One view per agent tab. Houses the panel layout and owns reactive state."""

    def __init__(self, session: AgentSession, view_id: str) -> None:
        super().__init__(id=view_id)
        self._session = session
        self._main_view_id = f"agent-main-{session.tab_id}"
        self._config_view_id = f"agent-config-{session.tab_id}"
        self._switcher_id = f"agent-view-switcher-{session.tab_id}"

        # Create reactive state + event stream, wire to session immediately
        initial = AgentState(tab_id=session.tab_id, label=session.label)
        self._reactive_state = ReactiveState(initial)
        self._event_stream = EventStream()
        session.reactive = self._reactive_state
        session.stream = self._event_stream

    def compose(self) -> ComposeResult:
        tab_id = self._session.tab_id
        with ContentSwitcher(
            initial=self._main_view_id,
            id=self._switcher_id,
            classes="agent-view-switcher",
        ):
            # Panels inside here auto-subscribe by walking up to find
            # _reactive_state and _event_stream on this AgentView.
            with Container(id=self._main_view_id, classes="agent-main-view"):
                with Container(classes="agent-page"):
                    with Vertical(classes="content-container"):
                        yield StatusRow()
                        yield TimelinePanel()
                        yield IntroBanner()
                        yield MissionControls(tab_id=tab_id)
                    with Vertical(classes="config-container"):
                        yield TelemetryPanel()
                        yield ConfigButtons(tab_id=tab_id)
            yield AgentConfigEditor(
                tab_id=tab_id,
                id=self._config_view_id,
                get_config=self._get_session_config,
                apply_change=self._apply_config_change,
                on_back=self.show_agent_main,
            )

    def on_mount(self) -> None:
        main_view = self.query_one(f"#{self._main_view_id}", Container)
        config_container = main_view.query_one(".config-container", Vertical)
        config_container.styles.layout = "vertical"
        config_container.styles.height = "100%"
        config_container.styles.min_height = 0
        config_container.styles.width = "48"
        switcher = self.query_one(f"#{self._switcher_id}", ContentSwitcher)
        switcher.styles.width = "100%"
        switcher.styles.height = "100%"
        self.styles.width = "100%"
        self.styles.height = "100%"
        self.styles.padding = (1, 2, 1, 2)

    def on_button_pressed(self, event) -> None:
        button_id = str(event.button.id or "")
        if button_id == f"open-config-{self._session.tab_id}":
            self.show_config_editor()
            event.stop()
        elif button_id == f"settings-button-{self._session.tab_id}":
            event.stop()

    def show_config_editor(self) -> None:
        switcher = self.query_one(f"#{self._switcher_id}", ContentSwitcher)
        switcher.current = self._config_view_id
        try:
            editor = switcher.query_one(f"#{self._config_view_id}", AgentConfigEditor)
            editor.refresh_from_config()
        except Exception:
            pass

    def show_agent_main(self) -> None:
        switcher = self.query_one(f"#{self._switcher_id}", ContentSwitcher)
        switcher.current = self._main_view_id

    def _get_session_config(self) -> Config:
        return self._session.config

    @staticmethod
    def _set_path_value(target: dict[str, Any], dotted_path: str, value: Any) -> None:
        parts = [part for part in str(dotted_path).split(".") if part]
        if not parts:
            return
        node = target
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value

    @staticmethod
    def _format_validation_error(exc: Exception) -> str:
        errors = getattr(exc, "errors", None)
        if callable(errors):
            try:
                details = errors()
                if isinstance(details, list) and details:
                    message = str(details[0].get("msg", "") or "").strip()
                    if message:
                        return message
            except Exception:
                pass
        text = str(exc).strip()
        return text or "Invalid value."

    def _apply_config_change(
        self,
        dotted_path: str,
        value: Any,
        clear: bool,
    ) -> tuple[bool, str]:
        path = str(dotted_path or "").strip()
        if not path:
            return False, "Missing config path."

        try:
            candidate_data = self._session.config.model_dump(mode="python")
        except Exception:
            return False, "Unable to read current config."

        self._set_path_value(candidate_data, path, None if clear else value)
        try:
            updated_config = Config.model_validate(candidate_data)
        except Exception as exc:
            return False, self._format_validation_error(exc)

        try:
            before_dump = self._session.config.model_dump(mode="python")
            after_dump = updated_config.model_dump(mode="python")
            if before_dump == after_dump:
                return True, ""
        except Exception:
            pass

        self._session.config = updated_config
        self._session.config_revision += 1
        return True, ""


# ---------------------------------------------------------------------------
# BrowserAgentApp — the main Textual application
# ---------------------------------------------------------------------------


class BrowserAgentApp(App):
    """Multi-tab browser agent TUI."""

    CSS_PATH = "dashboard_layout.tcss"

    BINDINGS = [
        Binding("ctrl+s", "stop_agent", "Stop Agent"),
        Binding("ctrl+r", "resume_agent", "Resume Agent"),
        Binding("ctrl+p", "pause_agent", "Pause Agent", show=True),
        Binding("ctrl+c", "cancel_agent", "Cancel Agent"),
        Binding("ctrl+x", "exit_agent", "Exit Agent"),
        Binding("a", "add", "Add Agent"),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._add_tab_id = "agent-add"
        self._load_tab_id = "agent-load"
        self._switcher_id = "agent-switcher"
        self._resume_queue_token = "__resume_loaded_checkpoint__"
        self._tab_to_view: dict[str, str] = {"agent-1": "data-1"}
        self._tab_label: dict[str, str] = {"agent-1": "Agent 1"}
        self._sessions: dict[str, AgentSession] = {}
        self._create_session("agent-1")
        self._max_queue_depth = 40
        self._min_refresh_interval_s = 0.12
        self._debug_print_callback = self._capture_debug_print

    # ---- Session lifecycle ------------------------------------------------

    def _create_session(self, tab_id: str) -> AgentSession:
        session = AgentSession(
            tab_id=tab_id,
            label=self._tab_label.get(tab_id, tab_id),
        )
        self._sessions[tab_id] = session
        return session

    def _ensure_session(self, tab_id: str) -> AgentSession:
        return self._sessions.get(tab_id) or self._create_session(tab_id)

    def get_session(self, tab_id: str) -> AgentSession | None:
        return self._sessions.get(tab_id)

    def get_active_session(self) -> AgentSession | None:
        tabs = self.query_one(Tabs)
        active_tab_id = tabs.active
        return self._sessions.get(active_tab_id) if active_tab_id else None

    # ---- App lifecycle ----------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Tabs(
            Tab("Agent 1", id="agent-1"),
            Tab("+ Load Agent", id=self._load_tab_id),
            Tab("+ New Agent", id=self._add_tab_id),
        )
        with ContentSwitcher(initial="data-1", id=self._switcher_id):
            yield self._make_agent_view("agent-1")
        yield Footer()

    def on_mount(self) -> None:
        try:
            switcher = self.query_one(f"#{self._switcher_id}", ContentSwitcher)
            switcher.styles.width = "100%"
            switcher.styles.height = "1fr"
        except Exception:
            pass
        register_print_callback(self._debug_print_callback)
        self.set_interval(0.5, self._poll_sessions)

    def on_unmount(self) -> None:
        unregister_print_callback(self._debug_print_callback)
        for session in self._sessions.values():
            session.worker_stop_event.set()
            if session.awaiting_answer or session.pending_question:
                session.answer_question("")
            try:
                session.mission_queue.put_nowait(None)
            except Exception:
                pass
        for session in self._sessions.values():
            worker = session.worker_thread
            if worker and worker.is_alive():
                try:
                    worker.join(timeout=3.0)
                except Exception:
                    pass

    def _session_for_worker_thread(self, thread_id: int) -> AgentSession | None:
        if thread_id <= 0:
            return None
        for session in self._sessions.values():
            if session.agent_thread_id == thread_id:
                return session
        return None

    def _capture_debug_print(self, record: DebugPrintRecord) -> None:
        # EventLogger debug events already flow through structured callbacks.
        # Skip their console mirror lines to avoid duplicate timeline entries.
        if record.module == "utils.event_logger" and record.function == "_print_event":
            return

        session = self._session_for_worker_thread(record.thread_id)
        if session is None:
            return

        text = str(record.text or "").replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            return

        try:
            self.call_from_thread(
                self._record_debug_print_line,
                session,
                text,
                record.module,
                record.function,
            )
        except Exception:
            pass

    def _record_debug_print_line(
        self,
        session: AgentSession,
        text: str,
        source_module: str,
        source_function: str,
    ) -> None:
        for raw_line in str(text).split("\n"):
            if not raw_line.strip():
                continue
            event = BotEvent(
                event_type=EventType.SYSTEM_DEBUG,
                message=raw_line,
                level=LogLevel.DEBUG,
                details={
                    "source": "dprint",
                    "module": source_module,
                    "function": source_function,
                },
            )
            self._record_session_event(session, event)

    # ---- View factory ----------------------------------------------------

    def _make_agent_view(self, tab_id: str) -> AgentView:
        session = self._ensure_session(tab_id)
        view_id = self._tab_to_view[tab_id]
        view = AgentView(session=session, view_id=view_id)
        view.styles.width = "100%"
        view.styles.height = "1fr"
        return view

    # ---- Poll timer — builds AgentState and pushes to ReactiveState ------

    def _poll_sessions(self) -> None:
        for tab_id, session in list(self._sessions.items()):
            if session.reactive is None:
                continue
            agent = session.agent
            if agent is not None:
                try:
                    session.last_snapshot = agent.get_state_snapshot()
                    paused = bool(session.last_snapshot.get("paused", False))
                    if paused and session.status in {"running", "queued", "starting"}:
                        session.status = "paused"
                    elif not paused and session.status == "paused":
                        session.status = "running"
                except Exception as exc:
                    session.last_message = f"Snapshot error: {exc}"
            state = self._build_agent_state(session)
            session.reactive.update(state)

    @staticmethod
    def _sv(source: object, key: str, default=None):
        """Safe value getter for dicts and objects."""
        if source is None:
            return default
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    def _build_agent_state(self, session: AgentSession) -> AgentState:
        """Build a typed AgentState from a session's current runtime data."""
        snapshot = session.last_snapshot or {}
        sv = self._sv
        execution_state = sv(snapshot, "execution_state")
        mission_result = sv(snapshot, "mission_result")
        active_agent = session.agent
        effective_config = getattr(active_agent, "config", session.config) if active_agent else session.config
        interaction_cfg = getattr(effective_config, "user_interaction", None)
        iteration_samples = list(sv(execution_state, "iteration_ms_samples", []) or [])
        llm_samples = list(sv(execution_state, "llm_latency_ms_samples", []) or [])
        tool_samples = list(sv(execution_state, "tool_latency_ms_samples", []) or [])

        def _mean(values: list[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        def _p95(values: list[float]) -> float:
            if not values:
                return 0.0
            ordered = sorted(float(v) for v in values)
            idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95))))
            return float(ordered[idx])

        llm_call_count = int(sv(execution_state, "llm_call_count", 0) or 0)
        tokens_in_total = int(sv(execution_state, "tokens_in_total", 0) or 0)
        tokens_out_total = int(sv(execution_state, "tokens_out_total", 0) or 0)
        image_count_total = int(sv(execution_state, "image_count_total", 0) or 0)
        retry_count = int(sv(execution_state, "retry_count", 0) or 0)
        avg_tokens_in_live = float(tokens_in_total) / float(llm_call_count) if llm_call_count > 0 else 0.0
        avg_tokens_out_live = float(tokens_out_total) / float(llm_call_count) if llm_call_count > 0 else 0.0
        avg_images_live = float(image_count_total) / float(llm_call_count) if llm_call_count > 0 else 0.0
        retries_per_mission_live = float(retry_count) / max(1.0, float(sv(snapshot, "current_iteration", 0) or 1))
        avg_iteration_value = float(sv(mission_result, "avg_iteration_ms", 0.0) or _mean(iteration_samples))
        p95_iteration_value = float(sv(mission_result, "p95_iteration_ms", 0.0) or _p95(iteration_samples))
        avg_llm_value = float(sv(mission_result, "avg_llm_ms", 0.0) or _mean(llm_samples))
        avg_tool_value = float(sv(mission_result, "avg_tool_ms", 0.0) or _mean(tool_samples))
        avg_tokens_in_value = float(sv(mission_result, "avg_tokens_in", 0.0) or avg_tokens_in_live)
        avg_tokens_out_value = float(sv(mission_result, "avg_tokens_out", 0.0) or avg_tokens_out_live)
        avg_images_value = float(sv(mission_result, "avg_images_per_call", 0.0) or avg_images_live)
        retries_per_mission_value = float(sv(mission_result, "retries_per_mission", 0.0) or retries_per_mission_live)
        mission_ms_value = float(sv(mission_result, "mission_ms", 0.0) or sum(iteration_samples))

        return AgentState(
            tab_id=session.tab_id,
            label=session.label,
            status=session.status,
            last_message=session.last_message,
            current_mission=session.current_mission,
            worker_alive=bool(
                session.worker_thread and session.worker_thread.is_alive()
            ),
            queue_depth=session.mission_queue.qsize(),
            dropped_event_count=session.stream.dropped_count if session.stream else session.dropped_event_count,
            agent_attached=session.agent is not None,
            current_iteration=int(sv(snapshot, "current_iteration", 0) or 0),
            paused=bool(sv(snapshot, "paused", False)),
            cancel_requested=bool(sv(snapshot, "cancel_requested", False)),
            llm_total_cost_usd=float(sv(snapshot, "llm_total_cost_usd", 0.0) or 0.0),
            llm_total_tokens=int(sv(snapshot, "llm_total_tokens", 0) or 0),
            total_actions=int(sv(execution_state, "total_actions", 0) or 0),
            actions_since_progress=int(
                sv(execution_state, "actions_since_progress", 0) or 0
            ),
            user_facing_actions_since_progress=int(
                sv(execution_state, "user_facing_actions_since_progress", 0) or 0
            ),
            budget_total=int(sv(execution_state, "budget_total", 0) or 0),
            budget_spent=int(sv(execution_state, "budget_spent", 0) or 0),
            budget_remaining=int(sv(execution_state, "budget_remaining", 0) or 0),
            budget_phase=str(sv(execution_state, "budget_phase", "normal") or "normal"),
            low_budget_mode=bool(sv(execution_state, "low_budget_mode", False)),
            budget_constraints_enabled=bool(
                sv(execution_state, "budget_constraints_enabled", True)
            ),
            planning_batch_limit=int(sv(execution_state, "planning_batch_limit", 0) or 0),
            iteration_ms=float(sv(execution_state, "iteration_ms", 0.0) or 0.0),
            llm_latency_ms=float(sv(execution_state, "llm_latency_ms", 0.0) or 0.0),
            tool_latency_ms=float(sv(execution_state, "tool_latency_ms", 0.0) or 0.0),
            navigation_latency_ms=float(sv(execution_state, "navigation_latency_ms", 0.0) or 0.0),
            tokens_in=int(sv(execution_state, "tokens_in", 0) or 0),
            tokens_out=int(sv(execution_state, "tokens_out", 0) or 0),
            image_count=int(sv(execution_state, "image_count", 0) or 0),
            tool_calls=int(sv(execution_state, "tool_calls", 0) or 0),
            retries=int(sv(execution_state, "retries", 0) or 0),
            avg_iteration_ms=avg_iteration_value,
            p95_iteration_ms=p95_iteration_value,
            avg_llm_ms=avg_llm_value,
            avg_tool_ms=avg_tool_value,
            avg_tokens_in=avg_tokens_in_value,
            avg_tokens_out=avg_tokens_out_value,
            avg_images_per_call=avg_images_value,
            retries_per_mission=retries_per_mission_value,
            mission_ms=mission_ms_value,
            failure_code=str(sv(mission_result, "failure_code", "") or sv(execution_state, "failure_code", "") or ""),
            failure_stage=str(sv(mission_result, "failure_stage", "") or sv(execution_state, "failure_stage", "") or ""),
            checkpoint_pending=bool(sv(execution_state, "checkpoint_pending", False)),
            last_action_summary=str(sv(execution_state, "last_action_summary", "") or ""),
            in_loop=bool(sv(execution_state, "in_loop", False)),
            loop_count=int(sv(execution_state, "loop_count", 0) or 0),
            loop_round=int(sv(execution_state, "loop_round", 0) or 0),
            loop_description=str(sv(execution_state, "loop_description", "") or ""),
            mission_success=sv(mission_result, "success", None),
            mission_reasoning=str(sv(mission_result, "reasoning", "") or ""),
            mission_final_url=str(sv(mission_result, "final_url", "") or ""),
            last_known_url=session.last_known_url,
            sandbox_web_policy=SandboxPolicyEngine.summarize_web_policy(effective_config),
            thinking_active=session.thinking_active,
            thinking_text=session.thinking_text,
            render_frame=int(time.monotonic() * 3) % 4 if session.thinking_active else 0,
            pending_question=session.pending_question,
            pending_options=tuple(session.pending_options),
            pending_multi_select=session.pending_multi_select,
            pending_yes_no=session.pending_yes_no,
            allow_custom=bool(getattr(interaction_cfg, "allow_custom", True)),
            allow_skip=bool(getattr(interaction_cfg, "allow_skip", True)),
        )

    # ---- Event recording — called from worker thread via call_from_thread --

    def _record_session_event(self, session: AgentSession, event: BotEvent) -> None:
        """Update session derived state from an event, then push to stream."""
        details = event.details or {}

        # Update last_known_url
        if event.event_type == EventType.BROWSER_NAVIGATION:
            url = str(details.get("url", "")).strip()
            if url:
                session.last_known_url = url

        # Track thinking state
        if event.event_type == EventType.ITERATION_START:
            iteration = details.get("iteration", "?")
            max_iterations = details.get("max_iterations", "?")
            session.thinking_active = True
            session.thinking_text = f"Thinking through iteration {iteration}/{max_iterations}"
        elif event.event_type == EventType.SYSTEM_INFO:
            message = str(event.message or "").strip()
            if session.thinking_active and message:
                if message.startswith("🤔 Agent thinking:"):
                    session.thinking_text = (
                        message.replace("🤔 Agent thinking:", "", 1).strip()
                        or "Thinking..."
                    )
                elif message.startswith("⏳ Waiting for:"):
                    session.thinking_text = message
        elif event.event_type == EventType.ASK_REQUESTED:
            session.thinking_active = False
            question = str(details.get("question", "")).strip()
            raw_options = details.get("options", [])
            options: list[str] = []
            if isinstance(raw_options, list):
                options = [str(opt).strip() for opt in raw_options if str(opt).strip()]
            yes_no = bool(details.get("yes_no", False))
            if yes_no:
                options = ["Yes", "No"]
            session.pending_options = options
            session.pending_multi_select = bool(details.get("multi_select", False)) and bool(options) and not yes_no
            session.pending_yes_no = yes_no
            session.pending_question = question
            if session.worker_stop_event.is_set() or session.status in {
                "stopping",
                "cancel requested",
            }:
                session.answer_question("")
            else:
                session.awaiting_answer = True
                session.status = "asking"
        elif event.event_type in {
            EventType.ASK_COMMAND_ANSWERED,
            EventType.ASK_COMMAND_SKIPPED,
            EventType.ASK_COMMAND_FAILURE,
        }:
            session.thinking_active = False
            session.pending_question = ""
            session.pending_options = []
            session.pending_multi_select = False
            session.pending_yes_no = False
            session.awaiting_answer = False
            if session.status == "asking":
                session.status = "running"
        elif event.event_type in {
            EventType.ACTION_DETERMINED,
            EventType.ACTION_COMPLETE,
            EventType.AGENT_COMPLETE,
            EventType.AGENT_ERROR,
            EventType.SYSTEM_ERROR,
        }:
            session.thinking_active = False

        # Push to stream (we are already on the main thread via call_from_thread)
        if session.stream is not None:
            session.stream.push(event)

    # ---- Agent / worker lifecycle -----------------------------------------

    def _ensure_agent_for_session(self, session: AgentSession) -> Agent:
        current_thread_id = threading.get_ident()
        if (
            session.agent is not None
            and session.agent_thread_id == current_thread_id
            and session.agent_config_revision == session.config_revision
        ):
            return session.agent

        # Different thread or config revision mismatch — teardown old agent.
        if session.agent is not None:
            try:
                session.agent.cancel()
            except Exception:
                pass
            try:
                if getattr(session.agent, "browser", None):
                    session.agent.browser.end()
            except Exception:
                pass
            session.agent = None
            session.agent_thread_id = None
            session.agent_config_revision = -1

        agent = Agent(
            config=session.config,
            user_question_callback=_make_session_question_callback(session),
            data_report_callback=on_data_reported,
            agent_id=(session.load_agent_id or None),
        )
        agent._start()
        setup_interceptors(agent)
        if session.load_agent_id and session.resume_ready:
            loaded, message = agent.load_checkpoint(run_id=(session.load_run_id or None))
            if loaded:
                session.resume_ready = True
                session.current_mission = agent.loaded_resume_mission or session.current_mission
                session.status = "paused"
                session.last_message = f"{message} Press Resume to continue."
                try:
                    session.last_snapshot = agent.get_state_snapshot()
                except Exception:
                    pass
            else:
                session.resume_ready = False
                session.status = "error"
                session.last_message = f"Checkpoint load failed: {message}"
                try:
                    agent.browser.page.goto(STARTING_URL)
                except Exception:
                    pass
        elif not session.load_agent_id:
            agent.browser.page.goto(STARTING_URL)
        apply_thinking_border(agent)

        def _capture_event(event: BotEvent) -> None:
            self.call_from_thread(self._record_session_event, session, event)

        agent.event_logger.register_callback(_capture_event)
        session.agent = agent
        session.agent_thread_id = current_thread_id
        session.agent_config_revision = session.config_revision
        return agent

    def _start_worker_for_session(self, session: AgentSession) -> None:
        worker = session.worker_thread
        if worker is not None and worker.is_alive():
            return
        session.worker_stop_event.clear()
        session.worker_thread = threading.Thread(
            target=self._run_session_worker,
            args=(session.tab_id,),
            daemon=True,
            name=f"session-worker-{session.tab_id}",
        )
        session.worker_thread.start()

    def _run_session_worker(self, tab_id: str) -> None:
        session = self._ensure_session(tab_id)
        agent: Agent | None = None
        try:
            while not session.worker_stop_event.is_set():
                try:
                    queued_mission = session.mission_queue.get(timeout=0.25)
                except queue.Empty:
                    continue

                if queued_mission is None:
                    session.mission_queue.task_done()
                    break

                mission = str(queued_mission).strip()
                is_resume_request = mission == self._resume_queue_token
                if not mission and not is_resume_request:
                    session.mission_queue.task_done()
                    continue

                try:
                    pending = session.mission_queue.qsize()
                    session.status = "running"
                    if is_resume_request:
                        session.last_message = f"Resuming mission ({pending} queued)..."
                    else:
                        session.current_mission = mission
                        session.last_message = f"Initializing mission ({pending} queued)..."

                    agent = self._ensure_agent_for_session(session)
                    if is_resume_request:
                        if not agent.has_loaded_checkpoint:
                            session.status = "error"
                            session.last_message = "No checkpoint is loaded for resume."
                            session.resume_ready = False
                            continue
                        session.current_mission = (
                            agent.loaded_resume_mission or session.current_mission
                        )
                        session.last_message = "Resuming loaded mission..."
                        result = agent.resume_loaded_mission()
                        session.resume_ready = False
                    else:
                        session.last_message = "Mission executing..."
                        result = agent.execute_mission(mission)

                    session.last_snapshot = agent.get_state_snapshot()
                    session.status = "completed" if result.success else "failed"
                    session.last_message = result.reasoning or (
                        "Mission complete" if result.success else "Mission failed"
                    )
                    session.last_known_url = str(
                        getattr(result, "final_url", "") or session.last_known_url
                    )
                    severity = "information" if result.success else "warning"
                    label = "mission complete" if result.success else "mission failed"
                    self.call_from_thread(
                        self._emit_notice, f"{session.label}: {label}", severity
                    )
                except Exception as exc:
                    session.status = "error"
                    session.last_message = f"Runtime error: {exc}"
                    error_event = BotEvent(
                        event_type=EventType.SYSTEM_ERROR,
                        message=f"Runtime error: {exc}",
                        details={
                            "source": "worker",
                            "tab_id": tab_id,
                            "mission": session.current_mission,
                        },
                    )
                    self.call_from_thread(
                        self._record_session_event, session, error_event
                    )
                    self.call_from_thread(
                        self._emit_notice, f"{session.label}: runtime error", "error"
                    )
                finally:
                    session.mission_queue.task_done()
        finally:
            if agent is not None:
                try:
                    agent.cancel()
                except Exception:
                    pass
                try:
                    if getattr(agent, "browser", None):
                        agent.browser.end()
                except Exception:
                    pass
            session.agent = None
            session.agent_thread_id = None
            session.agent_config_revision = -1
            session.worker_thread = None
            if session.status not in {"error"}:
                session.status = "idle"
                session.last_message = "Worker stopped"

    # ---- Mission queueing ------------------------------------------------

    def _enqueue_mission_for_tab(self, tab_id: str, mission: str) -> bool:
        session = self._ensure_session(tab_id)
        mission_text = (mission or "").strip()

        self._start_worker_for_session(session)
        current_depth = session.mission_queue.qsize()
        if current_depth >= self._max_queue_depth:
            session.last_message = (
                f"Queue full ({current_depth}/{self._max_queue_depth}). "
                "Wait for missions to finish."
            )
            self._emit_notice(f"{session.label}: queue is full.", "warning")
            return False

        if not mission_text:
            if session.resume_ready:
                session.mission_queue.put(self._resume_queue_token)
                pending = session.mission_queue.qsize()
                if session.status not in {"running", "paused", "cancel requested", "stopping"}:
                    session.status = "queued"
                session.last_message = f"Resume queued ({pending} pending)."
                return True
            session.last_message = "Please enter a mission before running."
            self._emit_notice(f"{session.label}: mission text is required.", "warning")
            return False

        # User entered a fresh mission, so this session is no longer in "resume checkpoint" mode.
        session.resume_ready = False
        session.load_run_id = ""
        session.mission_queue.put(mission_text)
        pending = session.mission_queue.qsize()
        if session.status not in {"running", "paused", "cancel requested", "stopping"}:
            session.status = "queued"
        session.last_message = f"Mission queued ({pending} pending)."
        return True

    def _drain_pending_missions(self, session: AgentSession) -> int:
        drained = 0
        while True:
            try:
                queued = session.mission_queue.get_nowait()
            except queue.Empty:
                break
            if queued is not None:
                drained += 1
            try:
                session.mission_queue.task_done()
            except Exception:
                pass
        return drained

    def _emit_notice(self, message: str, severity: str = "information") -> None:
        try:
            self.notify(message, severity=severity, timeout=2.5)
        except Exception:
            pass

    # ---- Tab management --------------------------------------------------

    def _fallback_tab_id(self) -> str:
        for tab_id in self._tab_to_view.keys():
            return tab_id
        return "agent-1"

    def _next_tab_identity(self) -> tuple[int, str, str, str]:
        """Return (index, label, tab_id, view_id) for the next agent tab."""
        new_index = len(self._tab_to_view) + 1
        new_label = f"Agent {new_index}"
        new_tab_id = f"agent-{new_index}"
        view_id = f"data-{new_index}"
        return new_index, new_label, new_tab_id, view_id

    def _discover_load_candidates(self) -> list[LoadCandidate]:
        """Discover resumable checkpoints across all known agents."""
        manager = AgentWorkspaceManager(base_dir=base_config.storage.base_dir)
        candidates: list[LoadCandidate] = []
        for agent_id in manager.list_agent_ids():
            workspace = manager.create_agent(
                persistence_mode=base_config.storage.default_persistence_mode,
                agent_id=agent_id,
            )
            run_id = manager.resolve_resume_run_id(workspace, prefer_active=True)
            if not run_id:
                continue
            checkpoint = manager.read_run_checkpoint(workspace, run_id=run_id)
            if not checkpoint:
                continue
            runs = manager.list_runs(workspace)
            run_record = next(
                (item for item in runs if str(item.get("run_id", "")) == run_id),
                {},
            )
            mission = str(
                checkpoint.get("mission", "")
                or run_record.get("mission", "")
                or ""
            ).strip()
            candidates.append(
                LoadCandidate(
                    agent_id=agent_id,
                    run_id=run_id,
                    mission=mission,
                    status=str(run_record.get("status", "") or ""),
                    active=bool(run_record.get("active", False)),
                    started_at=float(run_record.get("started_at", 0.0) or 0.0),
                )
            )

        candidates.sort(
            key=lambda item: (
                1 if item.active else 0,
                item.started_at,
                item.agent_id,
                item.run_id,
            ),
            reverse=True,
        )
        return candidates

    def _open_loaded_candidate_tab(self, candidate: LoadCandidate) -> None:
        tabs = self.query_one(Tabs)
        switcher = self.query_one(ContentSwitcher)
        _, new_label, new_tab_id, view_id = self._next_tab_identity()
        new_label = f"{new_label} ({candidate.agent_id})"
        self._tab_label[new_tab_id] = new_label
        self._tab_to_view[new_tab_id] = view_id
        session = self._create_session(new_tab_id)
        session.load_agent_id = candidate.agent_id
        session.load_run_id = candidate.run_id
        session.resume_ready = True
        session.current_mission = candidate.mission
        session.status = "paused"
        session.last_message = (
            f"Checkpoint selected: {candidate.agent_id}/{candidate.run_id}. "
            "Press Resume to continue."
        )

        tabs.add_tab(Tab(new_label, id=new_tab_id), before=self._load_tab_id)

        def _mount_and_show() -> None:
            switcher.mount(self._make_agent_view(new_tab_id))
            tabs.active = new_tab_id
            switcher.current = view_id

        self.call_after_refresh(_mount_and_show)

    def action_load(self) -> None:
        candidates = self._discover_load_candidates()
        if not candidates:
            self._emit_notice("No resumable checkpoints found.", "warning")
            tabs = self.query_one(Tabs)
            tabs.active = self._fallback_tab_id()
            return

        def _on_dismissed(selection: LoadCandidate | None) -> None:
            if selection is None:
                tabs = self.query_one(Tabs)
                tabs.active = self._fallback_tab_id()
                return
            self._open_loaded_candidate_tab(selection)

        self.push_screen(LoadAgentScreen(candidates), _on_dismissed)

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        if event.tab is None:
            return
        if event.tab.id == self._load_tab_id:
            self.action_load()
            return
        if event.tab.id == self._add_tab_id:
            self.action_add()
            return
        self._ensure_session(event.tab.id)
        switcher = self.query_one(ContentSwitcher)
        view_id = self._tab_to_view.get(event.tab.id)
        if not view_id:
            return
        try:
            switcher.query_one(f"#{view_id}")
        except Exception:
            return
        switcher.current = view_id

    def action_add(self) -> None:
        tabs = self.query_one(Tabs)
        switcher = self.query_one(ContentSwitcher)
        _, new_label, new_tab_id, view_id = self._next_tab_identity()
        self._tab_label[new_tab_id] = new_label
        self._tab_to_view[new_tab_id] = view_id
        self._create_session(new_tab_id)
        tabs.add_tab(Tab(new_label, id=new_tab_id), before=self._load_tab_id)

        def _mount_and_show() -> None:
            switcher.mount(self._make_agent_view(new_tab_id))
            tabs.active = new_tab_id
            switcher.current = view_id

        self.call_after_refresh(_mount_and_show)

    # ---- Input handling --------------------------------------------------

    @staticmethod
    def _extract_tab_id(widget_id: str | None, prefix: str) -> str | None:
        if not widget_id or not widget_id.startswith(prefix):
            return None
        return widget_id[len(prefix):] or None

    @staticmethod
    def _extract_choice_option_target(widget_id: str | None) -> tuple[str, int] | None:
        prefix = "choice-opt-"
        if not widget_id or not widget_id.startswith(prefix):
            return None
        remainder = widget_id[len(prefix):]
        if "-" not in remainder:
            return None
        index_part, tab_id = remainder.split("-", 1)
        try:
            option_index = int(index_part)
        except ValueError:
            return None
        if not tab_id:
            return None
        return tab_id, option_index

    @staticmethod
    def _interaction_policy(session: AgentSession) -> tuple[bool, bool]:
        active_agent = session.agent
        effective_config = getattr(active_agent, "config", session.config) if active_agent else session.config
        interaction_cfg = getattr(effective_config, "user_interaction", None)
        allow_custom = bool(getattr(interaction_cfg, "allow_custom", True))
        allow_skip = bool(getattr(interaction_cfg, "allow_skip", True))
        return allow_custom, allow_skip

    def _resolve_mission_controls(self, tab_id: str) -> MissionControls | None:
        view_id = self._tab_to_view.get(tab_id)
        if not view_id:
            return None
        try:
            switcher = self.query_one(ContentSwitcher)
            view = switcher.query_one(f"#{view_id}", AgentView)
            return view.query_one(MissionControls)
        except Exception:
            return None

    def on_button_pressed(self, event) -> None:
        button_id = event.button.id

        choice_target = self._extract_choice_option_target(button_id)
        if choice_target is not None:
            tab_id, option_index = choice_target
            session = self._sessions.get(tab_id)
            if session is None or session.status != "asking":
                return
            controls = self._resolve_mission_controls(tab_id)
            if controls is None:
                self._emit_notice(f"{session.label}: question view is unavailable.", "error")
                return
            option_text = controls.get_option_text(option_index)
            if option_text is None:
                return
            if session.pending_multi_select:
                controls.toggle_option(option_index)
                return
            session.answer_question(option_text)
            controls.clear_input()
            return

        confirm_tab_id = self._extract_tab_id(button_id, "choice-confirm-")
        if confirm_tab_id is not None:
            session = self._sessions.get(confirm_tab_id)
            if session is None or session.status != "asking":
                return
            controls = self._resolve_mission_controls(confirm_tab_id)
            if controls is None:
                self._emit_notice(f"{session.label}: question view is unavailable.", "error")
                return
            selected_options = controls.get_selected_options()
            if not selected_options:
                self._emit_notice("Select at least one option before confirming.", "warning")
                return
            session.answer_question(", ".join(selected_options))
            controls.clear_input()
            return

        skip_tab_id = self._extract_tab_id(button_id, "choice-skip-")
        if skip_tab_id is not None:
            session = self._sessions.get(skip_tab_id)
            if session is None or session.status != "asking":
                return
            _, allow_skip = self._interaction_policy(session)
            if not allow_skip:
                self._emit_notice("Skipping is disabled for this prompt.", "warning")
                return
            controls = self._resolve_mission_controls(skip_tab_id)
            if controls is not None:
                controls.clear_input()
            session.answer_question("")
            return

        tab_id = self._extract_tab_id(button_id, "run-mission-")
        if tab_id is None:
            return
        session = self._sessions.get(tab_id)
        controls = self._resolve_mission_controls(tab_id)
        if session is not None and session.status == "asking":
            if controls is None:
                self._emit_notice(f"{session.label}: question view is unavailable.", "error")
                return
            try:
                inp = controls.query_one(f"#{controls.input_id}", Input)
            except Exception as exc:
                session.last_message = f"Unable to submit answer: {exc}"
                self._emit_notice(f"{session.label}: unable to submit answer.", "error")
                return
            answer = str(inp.value or "").strip()
            allow_custom, allow_skip = self._interaction_policy(session)
            if session.pending_options and not allow_custom:
                self._emit_notice("Choose one of the provided options.", "warning")
                return
            if not answer:
                if not allow_skip:
                    self._emit_notice("An answer is required for this prompt.", "warning")
                    inp.focus()
                    return
                session.answer_question("")
            else:
                session.answer_question(answer)
            inp.value = ""
            inp.focus()
            return
        if controls is None:
            return
        mission = ""
        try:
            inp = controls.query_one(f"#{controls.input_id}", Input)
            mission = inp.value
        except Exception:
            pass
        enqueued = self._enqueue_mission_for_tab(tab_id, mission)
        if enqueued:
            try:
                inp.value = ""
                inp.focus()
            except Exception:
                pass

    def on_input_submitted(self, event) -> None:
        tab_id = self._extract_tab_id(event.input.id, "mission-input-")
        if tab_id is None:
            return
        session = self._sessions.get(tab_id)
        if session is not None and session.status == "asking":
            answer = str(event.value or "").strip()
            allow_custom, allow_skip = self._interaction_policy(session)
            if session.pending_options and not allow_custom:
                self._emit_notice("Choose one of the provided options.", "warning")
                event.input.focus()
                return
            if not answer:
                if not allow_skip:
                    self._emit_notice("An answer is required for this prompt.", "warning")
                    event.input.focus()
                    return
                session.answer_question("")
            else:
                session.answer_question(answer)
            event.input.value = ""
            event.input.focus()
            return
        enqueued = self._enqueue_mission_for_tab(tab_id, event.value)
        if enqueued:
            event.input.value = ""
            event.input.focus()

    # ---- Keyboard actions ------------------------------------------------

    def action_stop_agent(self) -> None:
        session = self.get_active_session()
        if session is None:
            return
        drained = self._drain_pending_missions(session)
        session.worker_stop_event.set()
        try:
            session.mission_queue.put_nowait(None)
        except Exception:
            pass
        if session.agent is not None:
            try:
                session.agent.cancel()
            except Exception:
                pass
        session.status = "stopping"
        session.last_message = f"Stop requested. Cleared {drained} queued mission(s)."
        if session.awaiting_answer or session.pending_question:
            session.answer_question("")

    def action_resume_agent(self) -> None:
        session = self.get_active_session()
        if session is None:
            return
        if session.resume_ready:
            self._start_worker_for_session(session)
            current_depth = session.mission_queue.qsize()
            if current_depth >= self._max_queue_depth:
                session.status = "error"
                session.last_message = (
                    f"Unable to resume: queue full ({current_depth}/{self._max_queue_depth})."
                )
                return
            session.mission_queue.put(self._resume_queue_token)
            pending = session.mission_queue.qsize()
            session.status = "queued"
            session.last_message = f"Resume queued ({pending} pending)."
            return
        if session.agent is None:
            session.last_message = "No active mission to resume."
            return
        try:
            session.agent.resume()
            session.status = "running"
            session.last_message = "Mission resumed."
        except Exception as exc:
            session.status = "error"
            session.last_message = f"Resume failed: {exc}"

    def action_pause_agent(self) -> None:
        session = self.get_active_session()
        if session is None:
            return
        if session.agent is None:
            session.last_message = "No active mission to pause."
            return
        try:
            session.agent.pause("Paused from keyboard")
            session.status = "paused"
            session.last_message = "Mission paused."
        except Exception as exc:
            session.status = "error"
            session.last_message = f"Pause failed: {exc}"

    def action_cancel_agent(self) -> None:
        session = self.get_active_session()
        if session is None:
            return
        drained = self._drain_pending_missions(session)
        if session.agent is None:
            session.status = "idle" if drained > 0 else session.status
            session.last_message = (
                f"Cleared {drained} queued mission(s)."
                if drained > 0
                else "No active mission to cancel."
            )
            return
        try:
            session.agent.cancel()
            session.status = "cancel requested"
            session.last_message = f"Cancel requested. Cleared {drained} queued mission(s)."
            if session.awaiting_answer or session.pending_question:
                session.answer_question("")
        except Exception as exc:
            session.status = "error"
            session.last_message = f"Cancel failed: {exc}"

    def action_exit_agent(self) -> None:
        self.exit()
