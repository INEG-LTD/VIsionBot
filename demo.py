"""

"""
from time import sleep
import sys
import threading
import os
import queue
import time
from datetime import datetime
from dataclasses import dataclass, field
import chime
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from yaspin import yaspin
from art import text2art
from pydantic import BaseModel
from agent.agent_controller import Agent
from browser.provider import BrowserConfig
from core.config import Config, ModelConfig, ExecutionConfig, ElementConfig, DebugConfig, UserMessagesConfig
from core.config import ActFunctionConfig
from core.executor import Executor
from lib.ai import ReasoningLevel
from text_animator import TextAnimator
from utils.event_logger import BotEvent, EventType
from agent.interceptor_manager import Interceptor, InterceptorMode, InterceptorContext
import random
from prompt_toolkit import HTML, print_formatted_text as print
from rich.console import Console
from rich.prompt import Prompt
from textual.app import App, ComposeResult
from textual.widgets import Button, ContentSwitcher, Footer, Header, Input, Label, Tab, ListView, ListItem
from textual.widgets import Tabs

console = Console()

os.environ.setdefault("PLAYWRIGHT_CHROMIUM_DISABLE_CRASHPAD", "1")

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
        if not self._enabled: return
        self._ensure_init()
        try:
            self.agent.browser.page.evaluate("if(window.__agentThinkingBorder) window.__agentThinkingBorder.start();")
        except Exception:
            pass

    def stop(self):
        if not self._enabled: return
        try:
            self.agent.browser.page.evaluate("if(window.__agentThinkingBorder) window.__agentThinkingBorder.stop();")
        except Exception:
            pass

    def enable_blocking(self):
        if not self._enabled: return
        self._ensure_init()
        try:
            self.agent.browser.page.evaluate("if(window.__agentThinkingBorder) window.__agentThinkingBorder.enableBlocking();")
        except Exception:
            pass
        try:
            self.start()
        except Exception:
            pass

    def disable_blocking(self):
        if not self._enabled: return
        try:
            self.agent.browser.page.evaluate("if(window.__agentThinkingBorder) window.__agentThinkingBorder.disableBlocking();")
        except Exception:
            pass

def apply_thinking_border(agent: Agent):
    manager = ThinkingBorderManager(agent)
    agent._thinking_border_manager = manager
    # Executor checks browser for this manager (e.g., ask flow unblocks UI via browser attr).
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
                result = original_run_loop(self, *args, **kwargs)
                return result
            finally:
                if current_manager:
                    current_manager.disable_blocking()

        Agent._run_execution_loop = patched_run_loop
        Agent._thinking_border_run_loop_patched = True

    return manager

def setup_interceptors(agent: Agent):
    
    from typing import Optional
    class DropdownSelection(BaseModel):
        recommended_option: str
        confidence: float = 1.0
        reasoning: Optional[str] = None

    class IsDropdownVisible(BaseModel):
        is_visible: bool
    dropdown_trigger_select = Interceptor(
        action_type="select",
        target_regex=r"(?i)dropdown|select|combobox"
    )
    dropdown_trigger_click = Interceptor(
        action_type="click",
        target_regex=r"(?i)dropdown|select|combobox"
    )

    def select_dropdown_handler(context: InterceptorContext):
        print("🎯 Running dropdown selection interceptor...")

        try:
            current_action = context.action

            if not current_action:
                print("❌ No current action available for dropdown selection")
                return

            print(f"📍 Action to execute: {current_action}")

            action_part = ""

            if current_action.startswith('select:'):
                action_part = current_action[7:].strip()
                print(f"🎯 Need to select: '{action_part}'")
                action_part = action_part.replace("dropdown", "").replace("combobox", "").replace("select", "")
            elif current_action.startswith('click:'):
                action_part = current_action[6:].strip()
                print(f"🎯 Clicked on: '{action_part}'")
                action_part = action_part.replace("dropdown", "").replace("combobox", "").replace("select", "")

            analysis_prompt = f"""
                Based on the current page state, what is the best option to select for this select field with the placeholder: "{action_part}"?
                Consider the overall task context and what would be the most logical selection.
                """
            selection_info: DropdownSelection = context.ask_question_structured(
                analysis_prompt,
                DropdownSelection
            )

            dropdown_prompt = f"""
                Based on the current page state, is the dropdown with the placeholder: "{action_part}" visible?
                """
            dropdown_visible: IsDropdownVisible = context.ask_question_structured(
                dropdown_prompt,
                IsDropdownVisible
            )

            print(f"🤖 AI Analysis: Select '{selection_info.recommended_option}'")
            print(f"   Confidence: {selection_info.confidence}")

            if selection_info.confidence < 0.3:
                print("⚠️ AI confidence too low, skipping selection")
                return

            if not dropdown_visible.is_visible:
                agent.action_executor.act(f"type: {selection_info.recommended_option} in {action_part}")
                sleep(5)
            agent.action_executor.act(f"click: {selection_info.recommended_option}")
        except Exception as e:
            print(f"❌ Unexpected error in dropdown handler: {e}")

    agent.register_interceptor(
        trigger=dropdown_trigger_click,
        mode=InterceptorMode.SCRIPTED,
        handler=select_dropdown_handler
    )
    agent.register_interceptor(
        trigger=dropdown_trigger_select,
        mode=InterceptorMode.SCRIPTED,
        handler=select_dropdown_handler
    )
  
def simple_event_callback(event: BotEvent):
    text_animator = TextAnimator("Thinking...", effect="pulse")

    if event.event_type == EventType.ITERATION_START:
        iteration = event.details.get('iteration', '?')
        max_iterations = event.details.get('max_iterations', '?')
        print(HTML(f"\n<b>∞ Iteration {iteration}/{max_iterations}</b>"))
          
        text_animator.start()

    elif event.event_type == EventType.ACTION_DETERMINED:
        text_animator.stop()
        action = event.details.get('action', '?')
        narrative = event.details.get('narrative', '')
        reasoning = event.details.get('reasoning', '')

        if reasoning:
            print(HTML(f"<gray>> Here's the agent's internal reasoning [{action}]: {reasoning}</gray>"))
        print(f"    ⚡ {narrative}")

    elif event.event_type == EventType.AGENT_COMPLETE and event.details.get('success', False):            
        print("\n✅ I have now completed the mission!")

def ask_user_for_help(question: str, context: dict) -> str:
    chime.warning()
    print(f"\n❓ Agent asks: {question}")
    print("   (Press Enter to skip, or type your answer)")

    try:
        answer = input("   Your answer: ").strip()
        return answer
    except (KeyboardInterrupt, EOFError):
        return ""
    except Exception as e:
        print(f"❌ Error asking user for help: {e}")
        return ""

def receive_reported_data(payload: str, context: dict) -> None:
    print("\n📦 Agent reported data:")
    print(f"{payload}")
    current_url = context.get("current_url", "")
    if current_url:
        print(f"URL: {current_url}")

config = Config(
    model=ModelConfig(
        agent_model="gpt-5-mini",
        command_model="gpt-5-mini",
        agent_reasoning_level=ReasoningLevel.HIGH
    ),
    execution=ExecutionConfig(
        max_iterations=500,
        max_actions_per_plan=1,
        wait_for_load_before_iteration=True,
        wait_for_load_state="networkidle",
        wait_for_load_timeout_ms=5000,
    ),
    elements=ElementConfig(
        selection_fallback_model="gemini/gemini-2.5-flash-lite",
        selection_retry_attempts=2,
        crops_per_gallery=6,
    ),
    logging=DebugConfig(
        debug_mode=False,
        show_overlay_candidates=True,
        show_llm_costs=False,
        save_screenshots=True,
    ),
    browser=BrowserConfig(
        provider_type="local",
        headless=False,
        apply_stealth=True,
    ),
    act_function=ActFunctionConfig(
        enable_target_context_guard=False,
        enable_modifier=True,
        enable_additional_context=True
    ),
    user_messages=UserMessagesConfig(
        file_upload_prompt="⏸️ Please select the file you would like to upload. Press [Enter] when done...",
        file_upload_interrupted="⚠️ Upload interrupted. Please try again."
    )
)

class StatusPill(Widget):
    def __init__(self, status: str, color: str) -> None:
        super().__init__()
        self.status = status
        self.color = color

    def on_mount(self) -> None:
        self.mount(Label(f"[bold]{self.status}[/bold]"))
        
        # self.styles.padding = (1, 1, 1, 1) .
        self.styles.background = self.color
        self.styles.color = "white"
        self.styles.height = "1"
        self.styles.width = "auto"
        self.styles.padding = (0, 1, 0, 1)
        
    @classmethod
    def idle(cls) -> "StatusPill":
        return cls("IDLE", "red")
    @classmethod
    def running(cls) -> "StatusPill":
        return cls("RUNNING", "#4BF538")
    @classmethod
    def success(cls) -> "StatusPill":
        return cls("SUCCESS", "#4BF538")
    @classmethod
    def failed(cls) -> "StatusPill":
        return cls("FAILED", "red")
    @classmethod
    def stopping(cls) -> "StatusPill":
        return cls("STOPPING", "#FF9798")
    @classmethod
    def stopped(cls) -> "StatusPill":
        return cls("STOPPED", "red")
    @classmethod
    def paused(cls) -> "StatusPill":
        return cls("PAUSED", "yellow")
    @classmethod
    def cancelled(cls) -> "StatusPill":
        return cls("CANCELLED", "red")
    @classmethod
    def asking(cls) -> "StatusPill":
        return cls("ASKING", "blue")

class AgentView(Vertical):
    """A per-agent view whose children are mounted only after this view is mounted."""

    Art = text2art("The Big Browser Agent", font="serifcap")
    
    def __init__(self, agent_name: str, view_id: str, tab_id: str) -> None:
        super().__init__(id=view_id)
        self.agent_name = agent_name
        self.tab_id = tab_id
        self.mission_input_id = f"mission-input-{tab_id}"
        self.run_button_id = f"run-mission-{tab_id}"
        self.intro_banner_id = f"intro-banner-{tab_id}"
        self.timeline_scroll_id = f"timeline-scroll-{tab_id}"
        self.timeline_label_id = f"session-timeline-{tab_id}"
        self.telemetry_label_id = f"session-telemetry-{tab_id}"
        self.runtime_root_id = f"runtime-root-{tab_id}"
        self.left_pane_id = f"left-pane-{tab_id}"
        self.controls_row_id = f"controls-row-{tab_id}"

    def on_mount(self) -> None:
        # Group timeline, controls, and telemetry in a single container for unified mounting/layout.
        self.mount((
            Container(
                Container(
                    StatusPill.stopping(),
                    VerticalScroll(
                        Label("Timeline: waiting for mission events", id=self.timeline_label_id, markup=False),
                        Container(
                            Label("[gray]The history of work done by the agent will be displayed here[/gray]"),
                            Label("[darkgray]TIMELINE[/darkgray]"),
                            id="timeline-intro-text"
                        ),
                        id=self.timeline_scroll_id,
                        classes="timeline",
                    ),
                    Container(
                        Label(self.Art),
                        Label("[bold]Welcome to The Big Browser Agent Research Project[/bold]"),
                        Label("[gray]This is an agent that lives in the browser and supercharges the amount of work you can do. The agent works well when given a targeted specific task to perform.[/gray]"),
                        Label("[gray][bold]Guiding tip[/bold]: Imagine you were telling someone who's never done what you're asking before, provide just enough detail but not too much`[/gray]"),
                        id=self.intro_banner_id,
                        classes="intro-banner",
                    ),
                    Horizontal(
                        Input(
                            placeholder="Describe the mission for this agent tab...",
                            id=self.mission_input_id,
                        ),
                        Button("RUN", id=self.run_button_id, variant="success"),
                        id=self.controls_row_id,
                        classes="controls",
                    ),
                    classes="content-container",
                ),
                Container(
                    Label("Telemetry: waiting for mission state", id=self.telemetry_label_id, markup=False),
                    Container(
                        Label("[gray]Internal information about the agent's state will be displayed[/gray]"),
                        Label("[darkgray]TELEMETRY[/darkgray]"),
                        id="telemetry-intro-text"
                    ),
                    id=f"telemetry-panel-{self.tab_id}",
                    classes="right-pane",
                ),
                classes="agent-page",
            )
        ))
        # Screen grid is defined in dashboard_layout.tcss; avoid overriding here

        intro_banner = self.query_one(f"#{self.intro_banner_id}")
        intro_banner.styles.margin_bottom = 1
        # intro_banner.border_title = "The Big Browser Agent"
        # intro_banner.styles.border = ("round", "gray")
        # intro_banner.styles.padding = (0, 1, 0, 1)
        intro_banner.styles.height = "auto"

        timeline_scroll = self.query_one(f"#{self.timeline_scroll_id}")
        timeline_scroll.styles.border = ("solid", "#9C9C9C")
        timeline_scroll.styles.padding = (1, 1, 1, 1)
        timeline_scroll.styles.height = "1fr"
        # timeline_scroll.styles.background = "cyan"
        timeline_scroll.styles.padding = (0, 0, 0, 1)

        controls_row = self.query_one(f"#{self.controls_row_id}")
        controls_row.styles.padding = (0, 0, 0, 0)
        controls_row.styles.height = "auto"

        mission_input = self.query_one(f"#{self.mission_input_id}", Input)
        mission_input.styles.border = ("solid", "#9C9C9C")
        mission_input.styles.background = "transparent"
        mission_input.styles.height = "auto"
        mission_input.styles.width = "1fr"
        mission_input.styles.padding = (0, 1, 0, 1)

        run_button = self.query_one(f"#{self.run_button_id}", Button)
        run_button.styles.width = 16
        run_button.styles.height = "3"
        run_button.styles.background = "transparent"
        run_button.styles.pointer = "grab"
        run_button.styles.border = ("solid", "#9C9C9C")

        telemetry_panel = self.query_one(f"#telemetry-panel-{self.tab_id}")
        telemetry_panel.styles.border = ("solid", "darkgray")
        telemetry_panel.styles.padding = (0, 1, 0, 1)

        self.styles.width = "100%"
        self.styles.height = "100%"
        self.styles.padding = (2, 2, 2, 2)

    def set_runtime(
        self,
        session: "AgentSession",
        timeline_text: str,
        telemetry_text: str,
    ) -> None:
        """Refresh timeline + telemetry and hide intro once missions begin."""
        try:
            timeline_label = self.query_one(f"#{self.timeline_label_id}", Label)
            timeline_label.update(timeline_text)
            timeline_scroll = self.query_one(f"#{self.timeline_scroll_id}", VerticalScroll)
            timeline_scroll.scroll_end(animate=False)
        except Exception:
            pass

        try:
            telemetry_label = self.query_one(f"#{self.telemetry_label_id}", Label)
            telemetry_label.update(telemetry_text)
        except Exception:
            pass

        try:
            intro_banner = self.query_one(f"#{self.intro_banner_id}")
            has_started_mission = bool((session.current_mission or "").strip()) or session.status not in {"idle"}
            intro_banner.styles.display = "none" if has_started_mission else "block"
        except Exception:
            pass


@dataclass
class AgentSession:
    """Per-tab runtime state. Filled out gradually as TUI phases are implemented."""

    tab_id: str
    label: str
    status: str = "idle"
    agent: Agent | None = None
    agent_thread_id: int | None = None
    current_mission: str = ""
    worker_thread: threading.Thread | None = None
    mission_queue: queue.Queue = field(default_factory=queue.Queue)
    worker_stop_event: threading.Event = field(default_factory=threading.Event)
    event_buffer: list[BotEvent] = field(default_factory=list)
    event_lock: threading.Lock = field(default_factory=threading.Lock)
    last_snapshot: dict | None = None
    last_known_url: str = ""
    last_message: str = "Ready"
    dropped_event_count: int = 0
    last_render_key: tuple | None = None
    last_render_at_monotonic: float = 0.0
               
class BrowserAgentApp(App):
    """Textual UI wrapper that can host multiple agent tabs/views."""
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

        # Tab labels
        self.tabs: list[str] = ["Agent 1"]
        self.agents: list[str] = ["Agent 1"]
        self._switcher_id = "agent-switcher"

        # Map tab_id -> ContentSwitcher child id
        self._tab_to_view: dict[str, str] = {"agent-1": "data-1"}
        # Map tab_id -> display label
        self._tab_label: dict[str, str] = {"agent-1": "Agent 1"}
        # Map tab_id -> isolated runtime session state
        self._sessions: dict[str, AgentSession] = {}
        self._create_session("agent-1")
        self._snapshot_timer = None
        self._max_event_buffer = 5000
        self._max_queue_depth = 40
        self._min_refresh_interval_s = 0.12

    def _create_session(self, tab_id: str) -> AgentSession:
        session = AgentSession(
            tab_id=tab_id,
            label=self._tab_label.get(tab_id, tab_id),
        )
        self._sessions[tab_id] = session
        return session

    def _ensure_session(self, tab_id: str) -> AgentSession:
        session = self._sessions.get(tab_id)
        if session is not None:
            return session
        return self._create_session(tab_id)

    def get_session(self, tab_id: str) -> AgentSession | None:
        return self._sessions.get(tab_id)

    def get_active_session(self) -> AgentSession | None:
        tabs = self.query_one(Tabs)
        active_tab_id = tabs.active
        if active_tab_id is None:
            return None
        return self._sessions.get(active_tab_id)

    def on_mount(self) -> None:
        try:
            switcher = self.query_one(f"#{self._switcher_id}", ContentSwitcher)
            switcher.styles.width = "100%"
            switcher.styles.height = "1fr"
        except Exception:
            pass
        self._snapshot_timer = self.set_interval(0.5, self._poll_sessions)
        self._refresh_all_views()

    def on_unmount(self) -> None:
        for session in self._sessions.values():
            session.worker_stop_event.set()
            try:
                session.mission_queue.put_nowait(None)
            except Exception:
                pass

        for session in self._sessions.values():
            worker = session.worker_thread
            if not worker or not worker.is_alive():
                continue
            try:
                worker.join(timeout=3.0)
            except Exception:
                pass

    def _make_agent_view(self, tab_id: str) -> Vertical:
        self._ensure_session(tab_id)
        view_id = self._tab_to_view[tab_id]
        label = self._tab_label.get(tab_id, tab_id)
        view = AgentView(agent_name=label, view_id=view_id, tab_id=tab_id)
        view.styles.width = "100%"
        view.styles.height = "1fr"
        return view

    def compose(self) -> ComposeResult:
        yield Tabs(Tab("Agent 1", id="agent-1"))

        with ContentSwitcher(initial="data-1", id=self._switcher_id):
            # Yield children during compose; don't call .mount() here.
            yield self._make_agent_view("agent-1")

        yield Footer()

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        if event.tab is None:
            return
        self._ensure_session(event.tab.id)

        switcher = self.query_one(ContentSwitcher)
        view_id = self._tab_to_view.get(event.tab.id)
        if not view_id:
            return

        # Only switch if that view already exists (new tabs mount later)
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
            # mount first so data-N exists
            switcher.mount(self._make_agent_view(new_tab_id))

            # then activate (will trigger handler, but now it's safe)
            tabs.active = new_tab_id

            # and explicitly switch
            switcher.current = view_id

        self.call_after_refresh(_mount_and_show)

    def _extract_tab_id(self, widget_id: str | None, prefix: str) -> str | None:
        if not widget_id or not widget_id.startswith(prefix):
            return None
        tab_id = widget_id[len(prefix):]
        return tab_id or None

    def _emit_notice(self, message: str, severity: str = "information") -> None:
        try:
            self.notify(message, severity=severity, timeout=2.5)
        except Exception:
            # Keep notifications best-effort; never break runtime flow.
            pass

    @staticmethod
    def _state_value(source: object, key: str, default=None):
        if source is None:
            return default
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    def _snapshot_signature(self, snapshot: dict | None) -> tuple:
        if not snapshot:
            return ()

        execution_state = self._state_value(snapshot, "execution_state")
        mission_result = self._state_value(snapshot, "mission_result")

        return (
            self._state_value(snapshot, "paused", False),
            self._state_value(snapshot, "cancel_requested", False),
            self._state_value(snapshot, "current_iteration", 0),
            round(float(self._state_value(snapshot, "llm_total_cost_usd", 0.0) or 0.0), 6),
            int(self._state_value(snapshot, "llm_total_tokens", 0) or 0),
            bool(self._state_value(execution_state, "in_loop", False)),
            self._state_value(execution_state, "loop_round", 0),
            self._state_value(execution_state, "loop_count", 0),
            bool(self._state_value(execution_state, "checkpoint_pending", False)),
            int(self._state_value(execution_state, "total_actions", 0) or 0),
            int(self._state_value(execution_state, "actions_since_progress", 0) or 0),
            int(self._state_value(execution_state, "user_facing_actions_since_progress", 0) or 0),
            str(self._state_value(execution_state, "last_action_summary", "") or ""),
            self._state_value(mission_result, "success", None),
            str(self._state_value(mission_result, "reasoning", "") or ""),
            str(self._state_value(mission_result, "final_url", "") or ""),
        )

    @staticmethod
    def _truncate_text(value: object, max_len: int = 72) -> str:
        text = str(value or "").strip()
        if not text:
            return "n/a"
        if len(text) <= max_len:
            return text
        return f"{text[:max_len - 3]}..."

    def _build_render_key(self, session: AgentSession) -> tuple:
        with session.event_lock:
            event_count = len(session.event_buffer)
            last_event_ts = session.event_buffer[-1].timestamp if event_count else 0.0

        queue_depth = session.mission_queue.qsize()
        worker_alive = bool(session.worker_thread and session.worker_thread.is_alive())
        snapshot_signature = self._snapshot_signature(session.last_snapshot)
        mission = (session.current_mission or "").strip()

        return (
            session.status,
            session.last_message,
            mission,
            queue_depth,
            worker_alive,
            session.last_known_url,
            session.dropped_event_count,
            event_count,
            last_event_ts,
            snapshot_signature,
        )

    def _get_view_and_input(self, tab_id: str) -> tuple[AgentView | None, Input | None]:
        view_id = self._tab_to_view.get(tab_id)
        if not view_id:
            return None, None
        try:
            switcher = self.query_one(ContentSwitcher)
            view = switcher.query_one(f"#{view_id}", AgentView)
            mission_input = view.query_one(f"#{view.mission_input_id}", Input)
            return view, mission_input
        except Exception:
            return None, None

    def _clear_and_focus_mission_input(self, tab_id: str) -> None:
        _, mission_input = self._get_view_and_input(tab_id)
        if mission_input is None:
            return
        mission_input.value = ""
        mission_input.focus()

    def _build_telemetry_text(self, session: AgentSession) -> str:
        snapshot = session.last_snapshot or {}
        execution_state = self._state_value(snapshot, "execution_state")
        mission_result = self._state_value(snapshot, "mission_result")

        iteration = self._state_value(snapshot, "current_iteration", "-")
        paused = bool(self._state_value(snapshot, "paused", False))
        cancel_requested = bool(self._state_value(snapshot, "cancel_requested", False))
        total_cost = float(self._state_value(snapshot, "llm_total_cost_usd", 0.0) or 0.0)
        total_tokens = int(self._state_value(snapshot, "llm_total_tokens", 0) or 0)

        queue_depth = session.mission_queue.qsize()
        worker_alive = bool(session.worker_thread and session.worker_thread.is_alive())
        agent_attached = session.agent is not None

        in_loop = bool(self._state_value(execution_state, "in_loop", False))
        loop_round = self._state_value(execution_state, "loop_round", 0)
        loop_count = self._state_value(execution_state, "loop_count", None)
        loop_desc = self._truncate_text(self._state_value(execution_state, "loop_description", ""), 54)
        checkpoint_pending = bool(self._state_value(execution_state, "checkpoint_pending", False))
        total_actions = int(self._state_value(execution_state, "total_actions", 0) or 0)
        actions_since_progress = int(self._state_value(execution_state, "actions_since_progress", 0) or 0)
        user_facing_actions = int(
            self._state_value(execution_state, "user_facing_actions_since_progress", 0) or 0
        )
        last_action_summary = self._truncate_text(
            self._state_value(execution_state, "last_action_summary", ""),
            70,
        )

        final_url = self._state_value(mission_result, "final_url", "")
        current_url = session.last_known_url or final_url
        current_url = self._truncate_text(current_url, 70)

        mission_success = self._state_value(mission_result, "success", None)
        if mission_success is None:
            mission_result_line = "Result: n/a"
        else:
            mission_result_line = f"Result: {'success' if mission_success else 'failed'}"

        loop_line = "off"
        if in_loop:
            loop_line = f"round {loop_round}/{loop_count or '?'} ({loop_desc})"

        if not snapshot:
            idle_lines = [
                f"Status: {session.status}",
                f"Message: {self._truncate_text(session.last_message, 64)}",
                f"Queue Depth: {queue_depth}",
                f"Worker: {'running' if worker_alive else 'stopped'}",
                "Iteration: n/a",
                "Cost USD: 0.0000",
                "Tokens: 0",
                f"Dropped Events: {session.dropped_event_count}",
                "Hint: enter a mission and press Run.",
            ]
            return "\n".join(idle_lines)

        lines = [
            f"Status: {session.status}",
            f"Message: {self._truncate_text(session.last_message, 64)}",
            f"Queue Depth: {queue_depth}",
            f"Worker: {'running' if worker_alive else 'stopped'}",
            f"Agent Attached: {'yes' if agent_attached else 'no'}",
            f"Iteration: {iteration}",
            f"Paused: {'yes' if paused else 'no'}",
            f"Cancel Req: {'yes' if cancel_requested else 'no'}",
            f"Loop: {loop_line}",
            f"Checkpoint: {'pending' if checkpoint_pending else 'clear'}",
            f"Actions: total={total_actions}",
            f"Progress Gap: {actions_since_progress}",
            f"User Actions: {user_facing_actions}",
            f"Cost USD: {total_cost:.4f}",
            f"Tokens: {total_tokens}",
            f"URL: {current_url}",
            f"Last Action: {last_action_summary}",
            f"Dropped Events: {session.dropped_event_count}",
            mission_result_line,
        ]
        return "\n".join(lines)

    def _refresh_view_for_tab(self, tab_id: str, *, force: bool = False) -> None:
        session = self._sessions.get(tab_id)
        view_id = self._tab_to_view.get(tab_id)
        if not session or not view_id:
            return
        now = time.monotonic()
        render_key = self._build_render_key(session)
        if not force:
            if render_key == session.last_render_key:
                return
            if session.last_render_at_monotonic and (now - session.last_render_at_monotonic) < self._min_refresh_interval_s:
                return
        try:
            switcher = self.query_one(ContentSwitcher)
            view = switcher.query_one(f"#{view_id}", AgentView)
            timeline_text = self._build_timeline_text(session)
            telemetry_text = self._build_telemetry_text(session)
            view.set_runtime(
                session,
                timeline_text,
                telemetry_text,
            )
            session.last_render_key = render_key
            session.last_render_at_monotonic = now
        except Exception:
            pass

    def _refresh_all_views(self) -> None:
        for tab_id in list(self._sessions.keys()):
            self._refresh_view_for_tab(tab_id, force=True)

    def _format_timeline_event(self, event: BotEvent) -> str | None:
        event_type = event.event_type
        details = event.details or {}
        ts = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")

        if event_type == EventType.ITERATION_START:
            iteration = details.get("iteration", "?")
            max_iterations = details.get("max_iterations", "?")
            return f"[{ts}] ITERATION {iteration}/{max_iterations}"

        if event_type == EventType.ACTION_DETERMINED:
            action = details.get("action") or "unknown action"
            return f"[{ts}] ACTION -> {action}"

        if event_type == EventType.ACTION_COMPLETE:
            tool = details.get("tool") or "tool"
            success = bool(details.get("success", False))
            duration_ms = float(details.get("duration_ms", 0.0) or 0.0)
            result = "ok" if success else "failed"
            return f"[{ts}] RESULT {tool}: {result} ({duration_ms:.0f} ms)"

        if event_type == EventType.LOOP_STATE_CHANGED:
            change = details.get("change", "update")
            round_num = details.get("loop_round", "?")
            total = details.get("loop_count", "?")
            return f"[{ts}] LOOP {change}: round {round_num}/{total}"

        if event_type == EventType.ASK_REQUESTED:
            question = str(details.get("question", "")).strip()
            if len(question) > 110:
                question = f"{question[:107]}..."
            return f"[{ts}] ASK {question}"

        if event_type == EventType.AGENT_COMPLETE:
            success = bool(details.get("success", False))
            status = "SUCCESS" if success else "FAILED"
            reasoning = str(details.get("reasoning", "")).strip()
            if reasoning and len(reasoning) > 100:
                reasoning = f"{reasoning[:97]}..."
            if reasoning:
                return f"[{ts}] {status} {reasoning}"
            return f"[{ts}] {status}"

        if event_type == EventType.SYSTEM_ERROR:
            return f"[{ts}] ERROR {event.message}"

        return None

    def _build_timeline_text(self, session: AgentSession) -> str:
        with session.event_lock:
            events = list(session.event_buffer)

        lines: list[str] = []
        if session.dropped_event_count > 0:
            lines.append(f"[history] dropped {session.dropped_event_count} older event(s) to cap memory")

        for event in events:
            formatted = self._format_timeline_event(event)
            if formatted:
                lines.append(formatted)

        # if not lines:
        #     return (
        #         "Timeline is empty.\n"
        #         "Start a mission to see live execution events here.\n"
        #         "Controls: Ctrl+P pause, Ctrl+R resume, Ctrl+C cancel, Ctrl+S stop."
            # )
        return "\n".join(lines)

    def _ensure_agent_for_session(self, session: AgentSession) -> Agent:
        current_thread_id = threading.get_ident()
        if session.agent is not None and session.agent_thread_id == current_thread_id:
            return session.agent

        # Safety fallback: if an agent exists but belongs to another thread,
        # do not reuse it. Re-create on the current worker thread.
        if session.agent is not None and session.agent_thread_id != current_thread_id:
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
            user_question_callback=None,
            data_report_callback=receive_reported_data,
        )
        agent._start()
        setup_interceptors(agent)
        agent.browser.page.goto("https://example.com")
        apply_thinking_border(agent)

        # Store a rolling event buffer used by the live timeline panel.
        def _capture_event(event: BotEvent) -> None:
            with session.event_lock:
                if event.event_type == EventType.BROWSER_NAVIGATION:
                    url = str((event.details or {}).get("url", "")).strip()
                    if url:
                        session.last_known_url = url
                session.event_buffer.append(event)
                if len(session.event_buffer) > self._max_event_buffer:
                    session.event_buffer.pop(0)
                    session.dropped_event_count += 1

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
                    self.call_from_thread(self._refresh_view_for_tab, tab_id, force=True)

                    agent = self._ensure_agent_for_session(session)
                    session.last_message = "Mission executing..."
                    self.call_from_thread(self._refresh_view_for_tab, tab_id, force=True)

                    result = agent.execute_mission(mission)
                    session.last_snapshot = agent.get_state_snapshot()
                    session.status = "completed" if result.success else "failed"
                    session.last_message = result.reasoning or ("Mission complete" if result.success else "Mission failed")
                    session.last_known_url = str(getattr(result, "final_url", "") or session.last_known_url)
                    if result.success:
                        self.call_from_thread(self._emit_notice, f"{session.label}: mission complete", "information")
                    else:
                        self.call_from_thread(self._emit_notice, f"{session.label}: mission failed", "warning")
                except Exception as exc:
                    session.status = "error"
                    session.last_message = f"Runtime error: {exc}"
                    self.call_from_thread(self._emit_notice, f"{session.label}: runtime error", "error")
                finally:
                    session.mission_queue.task_done()
                    try:
                        self.call_from_thread(self._refresh_view_for_tab, tab_id, force=True)
                    except Exception:
                        pass
        finally:
            agent = session.agent
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
            try:
                self.call_from_thread(self._refresh_view_for_tab, tab_id, force=True)
            except Exception:
                pass

    def _enqueue_mission_for_tab(self, tab_id: str, mission: str) -> bool:
        session = self._ensure_session(tab_id)
        mission_text = (mission or "").strip()
        if not mission_text:
            session.last_message = "Please enter a mission before running."
            self._emit_notice(f"{session.label}: mission text is required.", "warning")
            self._refresh_view_for_tab(tab_id, force=True)
            return False

        self._start_worker_for_session(session)
        current_depth = session.mission_queue.qsize()
        if current_depth >= self._max_queue_depth:
            session.last_message = (
                f"Queue full ({current_depth}/{self._max_queue_depth}). Wait for missions to finish."
            )
            self._emit_notice(f"{session.label}: queue is full.", "warning")
            self._refresh_view_for_tab(tab_id, force=True)
            return False

        session.mission_queue.put(mission_text)
        pending = session.mission_queue.qsize()
        if session.status not in {"running", "paused", "cancel requested", "stopping"}:
            session.status = "queued"
        session.last_message = f"Mission queued ({pending} pending)."
        self._refresh_view_for_tab(tab_id, force=True)
        return True

    def _drain_pending_missions(self, session: AgentSession) -> int:
        drained = 0
        while True:
            try:
                queued = session.mission_queue.get_nowait()
            except queue.Empty:
                break
            else:
                # Ignore sentinel bookkeeping details; we only track user missions.
                if queued is not None:
                    drained += 1
                try:
                    session.mission_queue.task_done()
                except Exception:
                    pass
        return drained

    def _poll_sessions(self) -> None:
        for tab_id, session in list(self._sessions.items()):
            agent = session.agent
            if agent is None:
                self._refresh_view_for_tab(tab_id)
                continue
            try:
                session.last_snapshot = agent.get_state_snapshot()
                paused = bool(session.last_snapshot.get("paused", False))
                if paused and session.status in {"running", "queued", "starting"}:
                    session.status = "paused"
                elif not paused and session.status == "paused":
                    session.status = "running"
            except Exception as exc:
                session.last_message = f"Snapshot error: {exc}"
            self._refresh_view_for_tab(tab_id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        tab_id = self._extract_tab_id(event.button.id, "run-mission-")
        if tab_id is None:
            return
        view_id = self._tab_to_view.get(tab_id)
        if not view_id:
            return
        mission = ""
        _, mission_input = self._get_view_and_input(tab_id)
        if mission_input is not None:
            mission = mission_input.value
        enqueued = self._enqueue_mission_for_tab(tab_id, mission)
        if enqueued:
            if mission_input is not None:
                mission_input.value = ""
                mission_input.focus()
            else:
                self._clear_and_focus_mission_input(tab_id)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        tab_id = self._extract_tab_id(event.input.id, "mission-input-")
        if tab_id is None:
            return
        enqueued = self._enqueue_mission_for_tab(tab_id, event.value)
        if enqueued:
            event.input.value = ""
            event.input.focus()

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
        self._refresh_view_for_tab(session.tab_id, force=True)

    def action_resume_agent(self) -> None:
        session = self.get_active_session()
        if session is None:
            return

        if session.agent is None:
            session.last_message = "No active mission to resume."
            self._refresh_view_for_tab(session.tab_id, force=True)
            return

        try:
            session.agent.resume()
            session.status = "running"
            session.last_message = "Mission resumed."
        except Exception as exc:
            session.status = "error"
            session.last_message = f"Resume failed: {exc}"
        self._refresh_view_for_tab(session.tab_id, force=True)

    def action_pause_agent(self) -> None:
        session = self.get_active_session()
        if session is None:
            return

        if session.agent is None:
            session.last_message = "No active mission to pause."
            self._refresh_view_for_tab(session.tab_id, force=True)
            return

        try:
            session.agent.pause("Paused from keyboard")
            session.status = "paused"
            session.last_message = "Mission paused."
        except Exception as exc:
            session.status = "error"
            session.last_message = f"Pause failed: {exc}"
        self._refresh_view_for_tab(session.tab_id, force=True)

    def action_cancel_agent(self) -> None:
        session = self.get_active_session()
        if session is None:
            return

        drained = self._drain_pending_missions(session)
        if session.agent is None:
            if drained > 0:
                session.status = "idle"
                session.last_message = f"Cleared {drained} queued mission(s)."
            else:
                session.last_message = "No active mission to cancel."
            self._refresh_view_for_tab(session.tab_id, force=True)
            return

        try:
            session.agent.cancel()
            session.status = "cancel requested"
            session.last_message = f"Cancel requested. Cleared {drained} queued mission(s)."
        except Exception as exc:
            session.status = "error"
            session.last_message = f"Cancel failed: {exc}"
        self._refresh_view_for_tab(session.tab_id, force=True)

    def action_exit_agent(self) -> None:
        self.exit()

if __name__ == "__main__":
    # main()
    
    app = BrowserAgentApp()
    app.run()
