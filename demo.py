"""

"""
from time import sleep
import sys
import threading
import os
import queue
from datetime import datetime
from dataclasses import dataclass, field
import chime
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
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

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    clear_screen()
    Art = text2art("The Big Browser Agent", font="graceful")
    print(Art)
    print(HTML("<b>Welcome to The Big Browser Agent [RESEARCH TOOL]</b>"))
    print(HTML("The Big Browser Agent helps you perform long running complex tasks (called 'missions' by the agent)."))
    print(HTML("It is a research tool to help you perform tasks that would be too manual or time consuming to do yourself."))
    print(HTML("    ➤ Too manual: Tasks that require a lot of repitition/context switching/tab opening and closing."))
    print(HTML("    ➤ Too time consuming: Tasks that take at least 5 minutes of clicking and typing to complete."))
    
    print(HTML("The agent works best when you give it a targeted specific task to perform."))
    print(HTML("The agent is built to perform a task and is not conversational.\n"))
    print(HTML("<b>Guiding Tip:</b> Imagine you were telling someone who's never done what you're asking before, provide just enough detail but not too much."))
    print(HTML("When you are done, you can exit by typing 'exit'."))
    
    agent_loading_spinner = yaspin(text="Loading agent...", color="cyan")
    agent_loading_spinner.start()
    with Agent(
        config=config,
        user_question_callback=ask_user_for_help,
        data_report_callback=receive_reported_data,
    ) as agent:
        agent_loading_spinner.stop()
        setup_interceptors(agent)

        agent.event_logger.register_callback(simple_event_callback)
        agent.browser.page.goto("https://example.com")

        apply_thinking_border(agent)
        while True:
            what_to_do = Prompt.ask("\n ↦ What do you want to do?", case_sensitive=False)
            if what_to_do == "exit":
                clear_screen()
                print(HTML("<b>Thank you for using The Big Browser Agent [RESEARCH TOOL]</b>"))
                break
            result = agent.execute_mission(what_to_do)
            if result.success:
                chime.success()
            else:
                print("\n❌ Mission failed")
                chime.error()

            input("Press Enter to continue...")
            
class AgentView(VerticalScroll):
    """A per-agent view whose children are mounted only after this view is mounted."""

    def __init__(self, agent_name: str, view_id: str, tab_id: str) -> None:
        super().__init__(id=view_id)
        self.agent_name = agent_name
        self.tab_id = tab_id
        self.mission_input_id = f"mission-input-{tab_id}"
        self.run_button_id = f"run-mission-{tab_id}"
        self.status_label_id = f"session-status-{tab_id}"
        self.snapshot_label_id = f"session-snapshot-{tab_id}"
        self.timeline_label_id = f"session-timeline-{tab_id}"

    def on_mount(self) -> None:
        art = text2art("The Big Browser Agent", font="graceful")

        self.mount(Label(art))
        self.mount(
            Container(
                Label("The Big Browser Agent helps you perform long running complex tasks (called 'missions' by the agent)."),
                Label("\nAn agent in the browser helps you perform tasks that would be too manual or time consuming to do yourself"),
                Label("[bold]    ➤ Too manual: Tasks that require a lot of repitition/context switching/tab opening and closing.[/bold]"),
                Label("[bold]    ➤ Too time consuming: Tasks that take at least 5 minutes of clicking and typing to complete.[/bold]"),
                Label("The agent works best when you give it a targeted specific task to perform."),
                Label("The agent is built to perform a task and is not conversational.\n"),
                Label("[bold]Guiding Tip:[/bold] Imagine you were telling someone who's never done what you're asking before, provide just enough detail but not too much."),
                Label("When you are done, you can exit by typing 'exit'."),
                id="header-content"
            ),
        )
        
        self.mount(
            Container(
                Label("\nStatus: idle", id=self.status_label_id),
                Label("Snapshot: waiting for mission", id=self.snapshot_label_id),
                Input(
                    placeholder="Describe the mission for this agent tab...",
                    id=self.mission_input_id,
                ),
                Button("Run Mission", id=self.run_button_id, variant="success"),
                id=f"mission-controls-{self.tab_id}",
            ),
        )
        self.mount(
            Container(
                Label("Timeline: waiting for mission events", id=self.timeline_label_id),
                id=f"timeline-panel-{self.tab_id}",
            ),
        )
        self.mount(Container())
        self

        # mission_controls = self.query_one(f"#mission-controls-{self.tab_id}")
        # mission_controls.border_title = f"{self.agent_name} Controls"
        # mission_controls.styles.border = ("round", "green")
        # mission_controls.styles.padding = (1, 2, 1, 2)
        # mission_controls.styles.max_width = "75%"

        # timeline_panel = self.query_one(f"#timeline-panel-{self.tab_id}")
        # timeline_panel.border_title = f"{self.agent_name} Timeline"
        # timeline_panel.styles.border = ("round", "cyan")
        # timeline_panel.styles.padding = (1, 2, 1, 2)
        # timeline_panel.styles.max_width = "85%"

        header_content = self.query_one("#header-content")
        header_content.border_title = "The Big Browser Agent (RESEARCH TOOL)"
        header_content.styles.border = ("round", "gray")
        header_content.styles.padding = (1, 3, 1, 3)
        header_content.styles.max_width = "50%"

    def set_runtime(self, session: "AgentSession", snapshot_line: str, timeline_text: str) -> None:
        """Refresh status/snapshot labels using session state."""
        try:
            status_label = self.query_one(f"#{self.status_label_id}", Label)
            mission = (session.current_mission or "").strip()
            mission_preview = mission if len(mission) <= 64 else f"{mission[:61]}..."
            if mission_preview:
                status_label.update(f"Status: {session.status} | Mission: {mission_preview}")
            else:
                status_label.update(f"Status: {session.status}")
        except Exception:
            pass

        try:
            snapshot_label = self.query_one(f"#{self.snapshot_label_id}", Label)
            snapshot_label.update(snapshot_line)
        except Exception:
            pass

        try:
            timeline_label = self.query_one(f"#{self.timeline_label_id}", Label)
            timeline_label.update(timeline_text)
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
    last_message: str = "Ready"
               
class BrowserAgentApp(App):
    """Textual UI wrapper that can host multiple agent tabs/views."""

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

        # Map tab_id -> ContentSwitcher child id
        self._tab_to_view: dict[str, str] = {"agent-1": "data-1"}
        # Map tab_id -> display label
        self._tab_label: dict[str, str] = {"agent-1": "Agent 1"}
        # Map tab_id -> isolated runtime session state
        self._sessions: dict[str, AgentSession] = {}
        self._create_session("agent-1")
        self._snapshot_timer = None

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

    def _make_agent_view(self, tab_id: str) -> VerticalScroll:
        self._ensure_session(tab_id)
        view_id = self._tab_to_view[tab_id]
        label = self._tab_label.get(tab_id, tab_id)
        return AgentView(agent_name=label, view_id=view_id, tab_id=tab_id)

    def compose(self) -> ComposeResult:
        yield Tabs(Tab("Agent 1", id="agent-1"))

        with ContentSwitcher(initial="data-1"):
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

    def _format_snapshot_line(self, session: AgentSession) -> str:
        snapshot = session.last_snapshot
        if not snapshot:
            return f"Snapshot: waiting for mission | {session.last_message}"

        iteration = snapshot.get("current_iteration", "?")
        paused = bool(snapshot.get("paused", False))
        cancel_requested = bool(snapshot.get("cancel_requested", False))
        total_cost = float(snapshot.get("llm_total_cost_usd", 0.0) or 0.0)
        total_tokens = int(snapshot.get("llm_total_tokens", 0) or 0)
        mode = "paused" if paused else "running"
        if cancel_requested:
            mode = "cancel requested"
        return (
            f"Snapshot: iter={iteration} | mode={mode} | "
            f"cost=${total_cost:.4f} | tokens={total_tokens} | {session.last_message}"
        )

    def _refresh_view_for_tab(self, tab_id: str) -> None:
        session = self._sessions.get(tab_id)
        view_id = self._tab_to_view.get(tab_id)
        if not session or not view_id:
            return
        try:
            switcher = self.query_one(ContentSwitcher)
            view = switcher.query_one(f"#{view_id}", AgentView)
            timeline_text = self._build_timeline_text(session)
            view.set_runtime(session, self._format_snapshot_line(session), timeline_text)
        except Exception:
            pass

    def _refresh_all_views(self) -> None:
        for tab_id in list(self._sessions.keys()):
            self._refresh_view_for_tab(tab_id)

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
        for event in events:
            formatted = self._format_timeline_event(event)
            if formatted:
                lines.append(formatted)

        if not lines:
            return "Timeline: waiting for mission events"
        return "\n".join(lines[-24:])

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
                session.event_buffer.append(event)
                if len(session.event_buffer) > 300:
                    session.event_buffer.pop(0)

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
                    self.call_from_thread(self._refresh_view_for_tab, tab_id)

                    agent = self._ensure_agent_for_session(session)
                    session.last_message = "Mission executing..."
                    self.call_from_thread(self._refresh_view_for_tab, tab_id)

                    result = agent.execute_mission(mission)
                    session.last_snapshot = agent.get_state_snapshot()
                    session.status = "completed" if result.success else "failed"
                    session.last_message = result.reasoning or ("Mission complete" if result.success else "Mission failed")
                except Exception as exc:
                    session.status = "error"
                    session.last_message = f"Runtime error: {exc}"
                finally:
                    session.mission_queue.task_done()
                    try:
                        self.call_from_thread(self._refresh_view_for_tab, tab_id)
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
                self.call_from_thread(self._refresh_view_for_tab, tab_id)
            except Exception:
                pass

    def _enqueue_mission_for_tab(self, tab_id: str, mission: str) -> None:
        session = self._ensure_session(tab_id)
        mission_text = (mission or "").strip()
        if not mission_text:
            session.last_message = "Please enter a mission before running."
            self._refresh_view_for_tab(tab_id)
            return

        self._start_worker_for_session(session)

        session.mission_queue.put(mission_text)
        pending = session.mission_queue.qsize()
        if session.status != "running":
            session.status = "queued"
        session.last_message = f"Mission queued ({pending} pending)."
        self._refresh_view_for_tab(tab_id)

    def _poll_sessions(self) -> None:
        for tab_id, session in self._sessions.items():
            agent = session.agent
            if agent is None:
                self._refresh_view_for_tab(tab_id)
                continue
            try:
                session.last_snapshot = agent.get_state_snapshot()
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
        try:
            switcher = self.query_one(ContentSwitcher)
            view = switcher.query_one(f"#{view_id}", AgentView)
            mission_input = view.query_one(f"#{view.mission_input_id}", Input)
            mission = mission_input.value
        except Exception:
            pass
        self._enqueue_mission_for_tab(tab_id, mission)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        tab_id = self._extract_tab_id(event.input.id, "mission-input-")
        if tab_id is None:
            return
        self._enqueue_mission_for_tab(tab_id, event.value)

    # Your existing actions (stubs here so bindings don’t crash)
    def action_stop_agent(self) -> None: ...
    def action_resume_agent(self) -> None: ...
    def action_pause_agent(self) -> None: ...
    def action_cancel_agent(self) -> None: ...
    def action_exit_agent(self) -> None:
        self.exit()

if __name__ == "__main__":
    # main()
    
    app = BrowserAgentApp()
    app.run()
