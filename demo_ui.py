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

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import ContentSwitcher, Footer, Input, Tab, Tabs

from agent.agent_controller import Agent
from core.executor import Executor
from utils.event_logger import BotEvent, EventType

from demo_config import STARTING_URL, config, on_data_reported, on_user_question, setup_interceptors
from demo_panels import (
    AgentState,
    EventStream,
    IntroBanner,
    MissionControls,
    ReactiveState,
    StatusRow,
    TelemetryPanel,
    TimelinePanel,
)


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
    status: str = "idle"
    agent: Agent | None = None
    agent_thread_id: int | None = None
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
    # Wired by AgentView after it is mounted:
    reactive: ReactiveState | None = None
    stream: EventStream | None = None


# ---------------------------------------------------------------------------
# AgentView — layout container; exposes _reactive_state and _event_stream
#             so AgentPanel children can auto-subscribe
# ---------------------------------------------------------------------------


class AgentView(Vertical):
    """One view per agent tab. Houses the panel layout and owns reactive state."""

    def __init__(self, session: AgentSession, view_id: str) -> None:
        super().__init__(id=view_id)
        self._session = session

        # Create reactive state + event stream, wire to session immediately
        initial = AgentState(tab_id=session.tab_id, label=session.label)
        self._reactive_state = ReactiveState(initial)
        self._event_stream = EventStream()
        session.reactive = self._reactive_state
        session.stream = self._event_stream

    def compose(self) -> ComposeResult:
        tab_id = self._session.tab_id
        # Panels inside here auto-subscribe by walking up to find
        # _reactive_state and _event_stream on this AgentView.
        with Container(classes="agent-page"):
            with Vertical(classes="content-container"):
                yield StatusRow()
                yield TimelinePanel()
                yield IntroBanner()
                yield MissionControls(tab_id=tab_id)
            yield TelemetryPanel()

    def on_mount(self) -> None:
        self.styles.width = "100%"
        self.styles.height = "100%"
        self.styles.padding = (2, 2, 2, 2)


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
        self._switcher_id = "agent-switcher"
        self._tab_to_view: dict[str, str] = {"agent-1": "data-1"}
        self._tab_label: dict[str, str] = {"agent-1": "Agent 1"}
        self._sessions: dict[str, AgentSession] = {}
        self._create_session("agent-1")
        self._max_queue_depth = 40
        self._min_refresh_interval_s = 0.12

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
        yield Tabs(Tab("Agent 1", id="agent-1"))
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
        self.set_interval(0.5, self._poll_sessions)

    def on_unmount(self) -> None:
        for session in self._sessions.values():
            session.worker_stop_event.set()
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
            thinking_active=session.thinking_active,
            thinking_text=session.thinking_text,
            render_frame=int(time.monotonic() * 3) % 4 if session.thinking_active else 0,
        )

    # ---- Event recording — called from worker thread via call_from_thread --

    def _record_session_event(self, session: AgentSession, event: BotEvent) -> None:
        """Update session derived state from an event, then push to stream."""
        # Update last_known_url
        if event.event_type == EventType.BROWSER_NAVIGATION:
            url = str((event.details or {}).get("url", "")).strip()
            if url:
                session.last_known_url = url

        # Track thinking state
        if event.event_type == EventType.ITERATION_START:
            iteration = (event.details or {}).get("iteration", "?")
            max_iterations = (event.details or {}).get("max_iterations", "?")
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
        elif event.event_type in {
            EventType.ACTION_DETERMINED,
            EventType.ACTION_COMPLETE,
            EventType.ASK_REQUESTED,
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
        if session.agent is not None and session.agent_thread_id == current_thread_id:
            return session.agent

        # Different thread — teardown old agent
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

        agent = Agent(
            config=config,
            user_question_callback=on_user_question,
            data_report_callback=on_data_reported,
        )
        agent._start()
        setup_interceptors(agent)
        agent.browser.page.goto(STARTING_URL)
        apply_thinking_border(agent)

        def _capture_event(event: BotEvent) -> None:
            self.call_from_thread(self._record_session_event, session, event)

        agent.event_logger.register_callback(_capture_event)
        session.agent = agent
        session.agent_thread_id = current_thread_id
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
                if not mission:
                    session.mission_queue.task_done()
                    continue

                try:
                    pending = session.mission_queue.qsize()
                    session.status = "running"
                    session.current_mission = mission
                    session.last_message = f"Initializing mission ({pending} queued)..."

                    agent = self._ensure_agent_for_session(session)
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
                        details={"source": "worker", "tab_id": tab_id, "mission": mission},
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
            session.worker_thread = None
            if session.status not in {"error"}:
                session.status = "idle"
                session.last_message = "Worker stopped"

    # ---- Mission queueing ------------------------------------------------

    def _enqueue_mission_for_tab(self, tab_id: str, mission: str) -> bool:
        session = self._ensure_session(tab_id)
        mission_text = (mission or "").strip()
        if not mission_text:
            session.last_message = "Please enter a mission before running."
            self._emit_notice(f"{session.label}: mission text is required.", "warning")
            return False

        self._start_worker_for_session(session)
        current_depth = session.mission_queue.qsize()
        if current_depth >= self._max_queue_depth:
            session.last_message = (
                f"Queue full ({current_depth}/{self._max_queue_depth}). "
                "Wait for missions to finish."
            )
            self._emit_notice(f"{session.label}: queue is full.", "warning")
            return False

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

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        if event.tab is None:
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
        new_index = len(self._tab_to_view) + 1
        new_label = f"Agent {new_index}"
        new_tab_id = f"agent-{new_index}"
        view_id = f"data-{new_index}"
        self._tab_label[new_tab_id] = new_label
        self._tab_to_view[new_tab_id] = view_id
        self._create_session(new_tab_id)
        tabs.add_tab(Tab(new_label, id=new_tab_id))

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

    def on_button_pressed(self, event) -> None:
        tab_id = self._extract_tab_id(event.button.id, "run-mission-")
        if tab_id is None:
            return
        view_id = self._tab_to_view.get(tab_id)
        if not view_id:
            return
        mission = ""
        try:
            switcher = self.query_one(ContentSwitcher)
            view = switcher.query_one(f"#{view_id}", AgentView)
            controls = view.query_one(MissionControls)
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

    def action_resume_agent(self) -> None:
        session = self.get_active_session()
        if session is None:
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
        except Exception as exc:
            session.status = "error"
            session.last_message = f"Cancel failed: {exc}"

    def action_exit_agent(self) -> None:
        self.exit()
