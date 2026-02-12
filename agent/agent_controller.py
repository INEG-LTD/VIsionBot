import time
import threading
from typing import Optional, List, Dict, Any, Tuple, Callable, Union, Type, TYPE_CHECKING
import hashlib

from browser.dom import build_page_elements
from core.browser import ExecutionTimer
from core.executor.base import Executor
from core.session import BrowserState, SessionTracker
from models import PageElements, PageInfo
from models.models import (
    MissionPlan,
    Task,
    TaskStatus,
    TaskCompletionStatus,
    NotebookEntryType,
    FailedAction,
)
from agent.results import MissionResult, TaskResult
from agent.agent_context import EnvironmentState
from agent.notebook import Notebook
from agent.planning.orchestrator import TaskOrchestrator
from utils.debug_print import dprint
from lib.ai import (
    ReasoningLevel,
    get_default_agent_reasoning_level,
    set_default_model,
    set_default_reasoning_level,
    set_default_agent_model,
    set_default_agent_reasoning_level,
)
from pydantic import BaseModel

from utils.event_logger import set_event_logger

from .interceptor_manager import InterceptorManager, Interceptor, InterceptorMode, InterceptorContext
from core.config import Config
from execution.result import ActionResult
from utils import PageUtils
from core.browser import Browser

# Type alias for user question callback (ask: command handler)
# Callback receives: question (str), context (dict) -> returns user's answer (str) or None to skip
UserQuestionCallback = Callable[[str, dict], str]



"""
Agent Controller - Task-Based Execution

Runs tasks through the task orchestration and mini-loop execution engine.
"""

class Agent:
    """
    Agent controller running task-based execution.

    Provides:
    - Task decomposition (Normal and Sequential tasks)
    - Sequential task iteration with Sequence Planner
    - Result tracking and accumulation
    """
    
    def __init__(
        self,
        config: Config,
        base_knowledge: Optional[List[str]] = None,
        clarification_callback: Optional[Any] = None,
        # User question callback for ask: command
        user_question_callback: Optional[UserQuestionCallback] = None,
        # Agent talk callback for talk: command
        agent_talk_callback: Optional[Callable[[str], None]] = None,
        # Completion callback for complete: command
        completion_callback: Optional[Callable[[str], None]] = None,
    ):
        self.config = config
        self.mission_result = MissionResult()

        # Set global print mode based on config (affects all dprint calls and API debug logging)
        from utils.debug_print import set_print_mode, PrintMode
        set_print_mode(PrintMode.DEBUG if config.logging.debug_mode else PrintMode.NORMAL)

        # Access event logger from agent
        from utils.event_logger import EventLogger
        self.event_logger = EventLogger(debug_mode=True, show_overlay_candidates=config.logging.show_overlay_candidates, show_llm_costs=config.logging.show_llm_costs)
        set_event_logger(self.event_logger)  # Set as global
        
        self.iteration_delay = 0.5
        self.task_start_url: Optional[str] = None
        self.task_start_time: Optional[float] = None
        self.base_knowledge = base_knowledge or []  # Base knowledge rules that guide agent behavior
        self.failed_actions: List[str] = []  # Track actions that failed AND didn't yield any change
        self.ineffective_actions: List[str] = []  # Track actions that succeeded BUT didn't yield any change
        self.notebook: Notebook = Notebook()
        self._task_tracker: Dict[str, Dict[str, Any]] = {}
        # Completion / evaluation behavior
        self.clarification_callback = clarification_callback
        self._user_inputs: List[Dict[str, Any]] = []
        self._temp_user_inputs: List[Dict[str, Any]] = []  # Single-use suggestions
        self._original_user_prompt: str = ""

        # Store user question callback for ask: command
        self.user_question_callback = user_question_callback
        self.agent_talk_callback = agent_talk_callback
        self._last_ask_iteration: int = -2  # Track last iteration where ask: was answered (to prevent consecutive asks)

        # Store completion callback for complete: command
        self.completion_callback = completion_callback
        self._screenshot_counter = 0  # Counter for naming screenshots

        self.agent_model_name: str = config.model.agent_model
        self.agent_reasoning_level: ReasoningLevel = config.model.agent_reasoning_level
        self.command_model_name: str = self.config.model.command_model
        self.command_reasoning_level: ReasoningLevel = self.config.model.command_reasoning_level
        self.image_detail: str = config.model.image_detail
        self.max_iterations = config.execution.max_iterations
        set_default_model(self.command_model_name)
        set_default_reasoning_level(self.command_reasoning_level)
        set_default_agent_model(self.agent_model_name)
        set_default_agent_reasoning_level(self.agent_reasoning_level)
        
        # Pause functionality: Allows pausing agent execution between actions
        self._paused = False
        self._pause_lock = threading.Lock()  # Thread-safe access to pause state
        self._pause_event = threading.Event()  # Event to block execution when paused
        self._pause_event.set()  # Initially not paused (event is set = not blocking)
        self._pause_message = "Paused"

        self.interceptor_stack: List[Dict[str, Any]] = []  # Stack of active interceptors

        # Execution timer for tracking task, iteration, and action timings
        self.execution_timer = ExecutionTimer()
        
        self.show_llm_costs = config.logging.show_llm_costs
        _show_overlay_candidates = config.logging.show_overlay_candidates
        self.save_screenshots = config.logging.save_screenshots
        self.screenshot_dir = config.logging.screenshot_dir

        self._current_iteration = 0

        # Initialize task-based execution system
        self._initialize_task_system(
            auto_complete_extract_commands=self.config.execution.auto_complete_extract_commands,
        )

    def __enter__(self) -> 'Agent':
        """
        Context manager entry point. Automatically calls start().
        
        Example:
            >>> with Agent(browser, config) as agent:
            ...     agent.run_task("Click the button")
            # Automatically calls end() on exit
        """
        self._start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.event_logger.system_info("Stopping agent")
        # Clean up browser if it exists
        if self.browser:
            try:
                self.browser.end()
                self.event_logger.system_info("Browser stopped")
            except Exception:
                pass
        self.event_logger.system_info("Agent stopped")
    def _start(self) -> None:
        """Start the agent"""
        # Create browser without context manager - we'll manage its lifecycle
        # through the Agent's context manager instead
        self.browser = Browser(config=self.config)
        self.browser.started = True
        
        # Initialize components
        self.session_tracker: SessionTracker = SessionTracker(self.browser)
        
        self.page_utils = PageUtils(self.browser.page)

        # Initialize action executor
        self.action_executor: Executor = Executor(
            self.browser,
            self.session_tracker,
            self.notebook,
            self.page_utils,
            user_question_callback=self.user_question_callback,
            agent_talk_callback=self.agent_talk_callback,
            user_messages_config=self.config.user_messages if self.config else None,
        )
        
        # State tracking
        self.current_attempt = 0
        self.last_screenshot_hash = None
        self.last_dom_signature = None
        self.agent_controller = None  # Current agent controller instance (set during execute_task)
        
        # Screenshot caching for performance optimization
        self._cached_clean_screenshot = None
        self._cached_dom_signature = None
        
        self.interceptor_manager = InterceptorManager(self.browser)

        # Tab manager for multi-tab support
        self.tab_manager = self.browser.tab_manager

        # Plan generator for AI planning prompts
        self.started = True
      
    def execute_mission(
        self,
        user_prompt: str,
    ) -> bool:   
        
        # Register all pre-registered interceptors with the new controller
        for interceptor_data in self.interceptor_stack:
            self.register_interceptor(
                trigger=interceptor_data["trigger"],
                mode=interceptor_data["mode"],
                handler=interceptor_data["handler"]
            )
        
        # Run the mission (Decompose user request into tasks and execute them sequentially)
        mission_result = self._run_mission(user_prompt)

        self.event_logger.agent_complete(mission_result.success, mission_result.reasoning)
        

        self.mission_result = mission_result
        return mission_result.success
    
    def register_interceptor(
        self,
        trigger: Interceptor,
        mode: InterceptorMode,
        handler: Optional[Callable[[InterceptorContext], None]] = None
    ):
        """Register an interceptor trigger and handler."""
        self.interceptor_manager.register_interceptor(trigger, mode, handler)

    def _handle_interceptor_trigger(self, entry: Dict[str, Any], action: Optional[str] = None, action_step: Optional[Any] = None) -> bool:
        """Process a triggered interceptor"""
        if len(self.interceptor_stack) >= self.interceptor_manager.recursion_limit:
            self.event_logger.system_warning(
                f"Interceptor recursion limit reached ({self.interceptor_manager.recursion_limit})"
            )
            return False

        # Add instruction if missing
        if "instruction" not in entry:
            entry["instruction"] = f"Interact with: {action}"
        
        self.interceptor_stack.append(entry)
        self.event_logger.system_info(f"🎯 Interceptor Active: {entry['instruction']}")

        if entry["mode"] != InterceptorMode.SCRIPTED:
            self.interceptor_stack.pop()
            self.event_logger.system_warning("Unsupported interceptor mode (scripted-only)")
            return False
        try:
            self.interceptor_manager.execute_scripted(entry, self, action_step, action)
            self.interceptor_stack.pop()
            self.event_logger.system_info("✅ Scripted Interceptor Complete")
            return True
        except Exception as e:
            self.interceptor_stack.pop()
            self.event_logger.system_error(f"❌ Scripted Interceptor Failed: {e}")
            return False

    def _maybe_wait_for_turn_load(self, reason: str = "turn") -> None:
        if not self.config.execution.wait_for_load_before_turn:
            return
        try:
            if getattr(self.browser, "page", None) and not self.browser.page.is_closed():
                try:
                    self.event_logger.system_debug(
                        f"Waiting for page load before {reason} (state={self.wait_for_load_state})"
                    )
                except Exception:
                    pass
                self.browser.wait_for_load(
                    timeout=self.wait_for_load_timeout_ms,
                    state=self.wait_for_load_state,
                )
        except Exception:
            # Best-effort wait; do not block on load wait errors.
            pass
    
    def _run_mission(self, user_mission: str) -> MissionResult:
        """
        Execute a task autonomously.
        
        Args:
            user_mission: User's high-level request
            detected_elements: List of detected elements on the page
        Returns:
            MissionResult indicating success or failure
        """

        self._task_tracker = {}
        self._original_user_mission = user_mission
        # Initialize global iteration counter for mission-wide tracking
        self._current_iteration = 0
        self._current_sequential_iteration = 1
        self.event_logger.agent_start(user_mission)
        
        # Initialize task tracking
        # Ensure we're using the current page (may have been switched)
        try:
            self.task_start_url = self.browser.page.url
        except Exception:
            self.task_start_url = "unknown"
        self.task_start_time = time.time()
        
        # Start task timer
        self.execution_timer.start_task()
        
        # Set base knowledge on goal monitor for goal evaluation
        if self.base_knowledge:
            self.session_tracker.set_base_knowledge(self.base_knowledge)
        
        # Check if starting from a blank page. We don't want to start from a blank page.
        if self.browser.page.url.startswith("about:blank"):
            self.event_logger.agent_error("Page is on initial blank page.")

            # End task timer if it was started
            if self.execution_timer.task_start_time is not None:
                self.execution_timer.end_task()
                
            self.mission_result.success = False
            
            self.event_logger.agent_complete(success=False, reasoning="Page is blank")
            return self.mission_result

        # Decompose user request into tasks
        self.task_list = self._decompose_user_mission_into_tasks(user_mission)
        # Execute task list
        task_results = self._execute_task_list(user_mission)

        if self.execution_timer.task_start_time is not None:
            self.execution_timer.end_task()
            
        return task_results

    def _capture_snapshot(self, full_page: bool = False) -> BrowserState:
        """
        Capture current browser state snapshot.
        
        Args:
            full_page: If True, capture full page screenshot (for exploration mode)
                      If False, capture viewport only (normal mode)
        """
        snapshot = self.session_tracker._capture_current_state()
        
        # Always capture screenshot - agent needs it to see the page
        try:
            if full_page:
                snapshot.screenshot = self.browser.page.screenshot(full_page=True)
                dprint("📸 Using full-page screenshot for exploration mode")
            else:
                # Capture viewport screenshot (agent needs this to see what's visible)
                snapshot.screenshot = self.browser.page.screenshot(full_page=False)
        except Exception as e:
            dprint(f"⚠️ Failed to capture screenshot: {e}")
            snapshot.screenshot = None

        # Save screenshot for debugging if enabled
        if self.save_screenshots and snapshot.screenshot:
            try:
                from pathlib import Path
                from datetime import datetime

                # Create directory if it doesn't exist
                screenshot_path = Path(self.screenshot_dir)
                screenshot_path.mkdir(parents=True, exist_ok=True)

                # Generate filename with iteration and timestamp
                self._screenshot_counter += 1
                timestamp = datetime.now().strftime("%H%M%S")
                filename = f"iter{self._current_iteration:03d}_snap{self._screenshot_counter:03d}_{timestamp}.png"
                filepath = screenshot_path / filename

                # Save the screenshot
                with open(filepath, "wb") as f:
                    f.write(snapshot.screenshot)
                dprint(f"📸 Saved screenshot: {filepath}")
            except Exception as e:
                dprint(f"⚠️ Failed to save screenshot: {e}")

        # Compute screenshot hash for change detection
        screenshot_data = getattr(snapshot, "screenshot", None)
        screenshot_hash = None
        if screenshot_data:
            try:
                screenshot_hash = hashlib.md5(screenshot_data).hexdigest()
            except Exception:
                screenshot_hash = None
        setattr(snapshot, "screenshot_hash", screenshot_hash)
        
        return snapshot

    # ===== Task-Based Execution Methods (merged from TaskBasedExecution) =====

    def _initialize_task_system(
        self,
        auto_complete_extract_commands: bool = True,
    ) -> None:
        """
        Initialize task-based execution system.

        Args:
            auto_complete_extract_commands: If False, require explicit complete: commands after extract: actions
        """
        self.auto_complete_extract_commands = auto_complete_extract_commands

        # Task orchestrator for decomposition
        self.task_orchestrator = TaskOrchestrator(
            model_name=self.agent_model_name,
            reasoning_level=self.agent_reasoning_level,
        )

        # Current task list (set during execution)
        self.task_list: Optional[MissionPlan] = None
        self._extraction_model_cache: Dict[tuple[str, ...], Type[BaseModel]] = {}

    def _decompose_user_mission_into_tasks(
        self,
        user_mission: str) -> MissionPlan:
        """
        Decompose user mission into a MissionPlan.

        Args:
            user_mission: User's mission

        Returns:
            MissionPlan with Normal and Sequential tasks
        """
        # Build initial context
        context = {}
        try:
            context["url"] = self.browser.page.url
            context["page_title"] = self.browser.page.title()
        except Exception:
            pass

        # Capture current viewport screenshot for task decomposition grounding (best effort)
        screenshot_bytes = None
        try:
            snapshot_for_tasks = self._capture_snapshot(full_page=False)
            screenshot_bytes = getattr(snapshot_for_tasks, "screenshot", None)
        except Exception:
            screenshot_bytes = None

        # Call task orchestrator
        task_list = self.task_orchestrator.decompose_mission(
            user_mission=user_mission,
            initial_context=context,
            screenshot=screenshot_bytes,
        )

        # Log task decomposition
        try:
            self.event_logger.system_info(f"Task decomposition: {len(task_list.tasks)} tasks generated")
            for i, task in enumerate(task_list.tasks, 1):
                target_display = task.target if isinstance(task.target, str) else f"{task.target}x"
                self.event_logger.system_debug(f"  Task {i} (target={target_display}): {task.goal}")
        except Exception:
            pass

        return task_list

    def _execute_task_list(
        self,
        user_mission: str,
    ) -> MissionResult:
        """Execute all tasks in the mission sequentially."""
        mission_result = MissionResult()
        previous_task_context = []  # Context from previous tasks to flow forward

        # Execute tasks sequentially
        while self.task_list.get_current_task() is not None:
            current_task: Task = self.task_list.get_current_task()

            try:
                self.event_logger.task_start(
                    task_id=current_task.task_id,
                    task=current_task.goal,
                    task_type=f"target={current_task.target}",
                )
            except Exception:
                pass

            # Execute the task using unified execution
            result = self._execute_task(current_task, user_mission)

            self.mission_result.task_results.append(result)

            # Update task status
            if result.success:
                current_task.status = TaskStatus.COMPLETED
                current_task.completed_at = time.time()
                self.event_logger.task_complete(current_task.task_id, task_type=f"target={current_task.target}")

                # Add task history to context for next task
                if current_task.history:
                    previous_task_context.extend(current_task.history)

                # Move on to the next task
                self.task_list.current_task_index += 1
            else:
                current_task.status = TaskStatus.FAILED
                current_task.completed_at = time.time()
                self.event_logger.task_fail(current_task.task_id, error=result.reasoning, task_type=f"target={current_task.target}")

                # When a task fails, end the mission
                mission_result.success = False
                mission_result.reasoning = result.reasoning
                return mission_result

        # All tasks completed successfully
        completed_tasks = self.task_list.get_completed_tasks()

        if len(completed_tasks) > 0:
            mission_result.success = all(result.status == TaskStatus.COMPLETED for result in completed_tasks)
        else:
            mission_result.success = False
            mission_result.reasoning = "No tasks completed"

        return mission_result

    def _execute_task(self, task: Task, user_mission: str) -> TaskResult:
        """
        Execute a unified task (single or repetitive).

        Args:
            task: The task to execute (with target indicating repetition)
            user_mission: Original user mission for context

        Returns:
            TaskResult indicating success/failure and completion status
        """
        task.status = TaskStatus.IN_PROGRESS

        # Execute using the unified task loop
        result = self._run_unified_task_loop(task, user_mission)

        # Update task with final completion status
        if result.completion_status:
            task.completion_status = result.completion_status

        return result

    @staticmethod
    def _build_action_summary(action_step, result_str: str) -> str:
        """Build a natural-language summary of an action for the reflection block.

        Uses the action_step's function_name and arguments to produce a brief,
        first-person description of what happened.
        """
        fn = getattr(action_step, "function_name", None)
        args = getattr(action_step, "function_arguments", {}) or {}

        if fn == "click":
            elem = args.get("element_type", "element")
            desc = args.get("description", "")
            return f"You clicked the {elem} \"{desc}\". Result: {result_str}."
        elif fn == "type_text":
            text = args.get("text", "")
            field = args.get("field_description", "input field")
            return f"You typed \"{text}\" into {field}. Result: {result_str}."
        elif fn == "clear_text":
            field = args.get("field_description", "input field")
            return f"You cleared the text in {field}. Result: {result_str}."
        elif fn == "select_option":
            option = args.get("option", "")
            dropdown = args.get("dropdown_description", "dropdown")
            return f"You selected \"{option}\" in {dropdown}. Result: {result_str}."
        elif fn == "scroll_page":
            direction = args.get("direction", "down")
            return f"You scrolled {direction}. Result: {result_str}."
        elif fn == "press_key":
            key = args.get("key", "")
            return f"You pressed {key}. Result: {result_str}."
        elif fn == "open_url":
            url = args.get("url", "")
            return f"You navigated to {url}. Result: {result_str}."
        elif fn == "go_back":
            return f"You went back in browser history. Result: {result_str}."
        elif fn == "go_forward":
            return f"You went forward in browser history. Result: {result_str}."
        elif fn == "upload_file":
            file_path = args.get("file_path", "")
            return f"You uploaded file \"{file_path}\". Result: {result_str}."
        elif fn == "set_datetime":
            value = args.get("value", "")
            picker = args.get("picker_description", "date picker")
            return f"You set {picker} to \"{value}\". Result: {result_str}."
        elif fn == "extract_data":
            desc = args.get("data_description", "data")
            return f"You extracted: \"{desc}\". Result: {result_str}."
        elif fn == "wait_for":
            condition = args.get("condition", "")
            return f"You waited for: \"{condition}\". Result: {result_str}."
        else:
            # Fallback: use the action string
            action_str = getattr(action_step, "action", str(action_step))
            return f"You performed: {action_str}. Result: {result_str}."

    def _run_unified_task_loop(self, task: Task, original_prompt: str) -> TaskResult:
        """
        Unified task execution loop - handles both single and repetitive tasks.

        This replaces both _execute_normal_task and _execute_sequential_task with one unified loop.
        The agent uses mark_progress to signal completion of each unit of work.

        Args:
            task: Task with goal and target (1, N, or "all")
            original_prompt: Original user mission for context

        Returns:
            TaskResult with completion status
        """
        from agent.action_planner import ActionPlanner

        # Get config values
        max_actions_per_task = self.config.task_execution.max_actions_per_task
        stuck_threshold = self.config.task_execution.stuck_threshold

        # Track progress
        actions_since_progress = 0
        browser_actions_since_progress = 0
        total_actions = 0
        failed_elements: List[FailedAction] = []
        checkpoint_pending = False  # After every browser action, force next action to be think or mark_progress
        suppress_mark_progress = False  # After mark_progress is called, suppress it until next browser action
        active_strategy: Optional[str] = None  # Persistent reasoning from think(continue), shown every turn
        last_action_summary: Optional[str] = None  # Brief description of last action + result

        # Determine numeric target (None for "all")
        numeric_target = task.target if isinstance(task.target, int) else None

        while total_actions < max_actions_per_task:
            total_actions += 1
            self._current_iteration += 1

            # Check if target reached
            if numeric_target is not None and task.progress >= numeric_target:
                return TaskResult(
                    success=True,
                    completion_status=TaskCompletionStatus.COMPLETED,
                    progress=task.progress,
                    target=task.target,
                    history=task.history,
                    reasoning=f"Target reached: {task.progress}/{numeric_target}",
                )

            # Capture current state
            try:
                snapshot = self._capture_snapshot(full_page=False)
                page_info = self.page_utils.get_page_info()
                detected_elements = build_page_elements(self.browser.page, page_info)
            except Exception as e:
                return TaskResult(
                    success=False,
                    completion_status=TaskCompletionStatus.STUCK,
                    progress=task.progress,
                    target=task.target,
                    history=task.history,
                    reasoning=f"Failed to capture state: {str(e)}",
                )

            # Build environment state
            environment_state = EnvironmentState(
                browser_state=snapshot,
                interaction_history=self.session_tracker.interaction_history,
                user_prompt=original_prompt,
                task_start_url=self.task_start_url,
                task_start_time=self.task_start_time,
                current_url=snapshot.url,
                page_title=snapshot.title,
                visible_text=snapshot.visible_text,
                url_history=self.session_tracker.url_history.copy(),
                url_pointer=self.session_tracker.url_pointer
            )

            # Gather tab state for prompt injection
            tab_bar = None
            dialog_notice = None
            tab_events = []
            dialog_pending = False
            if self.tab_manager:
                self.tab_manager.refresh_metadata()
                tab_bar = self.tab_manager.build_tab_bar()
                dialog_notice = self.tab_manager.build_dialog_notice()
                tab_events = self.tab_manager.drain_events()
                dialog_pending = self.tab_manager.has_pending_dialog_on_active()

            # Create action planner with force_think if stuck
            force_think = (actions_since_progress >= stuck_threshold)
            action_planner = ActionPlanner(
                task.goal,
                self.session_tracker,
                base_knowledge=self.base_knowledge,
                model_name=self.agent_model_name,
                reasoning_level=self.agent_reasoning_level,
                image_detail=self.config.model.image_detail,
                max_actions_per_plan=self.config.execution.max_actions_per_plan,
                task_target=task.target,
                task_progress=task.progress,
                task_history=task.history,
                force_think=force_think,
                current_iteration=self._current_iteration,
                browser_actions_in_round=browser_actions_since_progress,
                checkpoint_mode=checkpoint_pending,
                suppress_mark_progress=suppress_mark_progress,
                active_strategy=active_strategy,
                last_action_summary=last_action_summary,
                tab_bar=tab_bar,
                dialog_notice=dialog_notice,
                tab_events=tab_events,
                dialog_pending=dialog_pending,
            )

            # Generate next actions
            try:
                actions_list, error = action_planner.get_next_actions_with_function_calling(
                    environment_state=environment_state,
                    screenshot=snapshot.screenshot,
                    notebook=self.notebook,
                    element_data=detected_elements,
                )

                if not actions_list:
                    return TaskResult(
                        success=False,
                        completion_status=TaskCompletionStatus.STUCK,
                        progress=task.progress,
                        target=task.target,
                        history=task.history,
                        reasoning=error or "No action generated",
                    )

                # Execute each action
                for action_step in actions_list:
                    current_action = action_step.action

                    # Handle mark_progress (intercepted by controller, not executor)
                    if current_action and current_action.lower().startswith("mark_progress:"):
                        # Gate: only count progress if real browser work was done since last mark
                        if browser_actions_since_progress == 0:
                            self.event_logger.system_debug("⚠ mark_progress ignored — no browser actions since last progress")
                            last_action_summary = (
                                f"mark_progress BLOCKED: No browser actions since your last progress mark "
                                f"({task.progress}/{task.target}). Do a browser action first "
                                f"(click, type, go_back, etc.) before marking progress again."
                            )
                            actions_since_progress += 1
                            continue

                        # Parse: "mark_progress: description | count=1 | done=false"
                        parts = current_action.split(":", 1)[1].strip()
                        description = parts
                        count = 1
                        done = False

                        # Parse parameters
                        if "|" in parts:
                            desc_part = parts.split("|")[0].strip()
                            description = desc_part
                            for param in parts.split("|")[1:]:
                                if "count=" in param:
                                    count = int(param.split("=")[1].strip())
                                elif "done=" in param:
                                    done = param.split("=")[1].strip().lower() == "true"

                        # Update task
                        task.progress += count
                        task.history.append(description)
                        actions_since_progress = 0
                        browser_actions_since_progress = 0
                        checkpoint_pending = False  # Exit checkpoint mode
                        suppress_mark_progress = True  # Suppress mark_progress until next browser action
                        active_strategy = None  # Strategy fulfilled, clear it
                        last_action_summary = f"You marked progress: \"{description}\" ({task.progress}/{task.target})"

                        self.event_logger.system_info(f"✓ Progress: {description} ({task.progress}/{task.target})")

                        # Check completion — only finish when target is actually reached
                        # For numeric targets: ignore done flag, trust the counter
                        # For "all" targets: done flag is the only way to signal completion
                        target_reached = numeric_target and task.progress >= numeric_target
                        done_with_open_target = done and numeric_target is None

                        if target_reached or done_with_open_target:
                            return TaskResult(
                                success=True,
                                completion_status=TaskCompletionStatus.COMPLETED,
                                progress=task.progress,
                                target=task.target,
                                history=task.history,
                                reasoning=f"Completed: {task.progress}/{task.target}",
                            )
                        continue

                    # Handle revise_target (intercepted by controller)
                    if current_action and current_action.lower().startswith("revise_target:"):
                        # Parse: "revise_target: new_target | reason"
                        parts = current_action.split(":", 1)[1].strip()
                        new_target_str = parts.split("|")[0].strip()
                        reason = parts.split("|")[1].strip() if "|" in parts else "Target revised"

                        # Parse new target
                        if new_target_str.lower() == "all":
                            task.target = "all"
                            numeric_target = None
                        else:
                            task.target = int(new_target_str)
                            numeric_target = task.target

                        self.event_logger.system_info(f"→ Target revised to {task.target}: {reason}")
                        last_action_summary = f"You revised the target to {task.target}: \"{reason}\""
                        continue

                    # Handle think (with next_action decision)
                    if current_action and current_action.lower().startswith("think:"):
                        # Execute via executor (records in session tracker)
                        self.action_executor.act(
                            action_step=action_step,
                            detected_elements=detected_elements,
                            page_info=page_info,
                            environment_state=environment_state,
                            current_iteration=self._current_iteration,
                        )
                        actions_since_progress += 1

                        # Extract the reasoning text (before any | params)
                        think_reasoning = current_action.split("|")[0].split(":", 1)[1].strip() if ":" in current_action else ""

                        # Parse next_action from think command
                        # Format: "think: reasoning text | next_action=done"
                        think_next_action = "continue"
                        if "| next_action=" in current_action:
                            think_next_action = current_action.split("| next_action=")[-1].strip().lower()

                        if think_next_action in ("mark_progress", "done"):
                            checkpoint_pending = False
                            active_strategy = None  # Strategy fulfilled
                            if browser_actions_since_progress == 0:
                                self.event_logger.system_debug(f"⚠ think next_action={think_next_action} ignored — no browser actions since last progress")
                                last_action_summary = (
                                    f"mark_progress BLOCKED: No browser actions since your last progress mark "
                                    f"({task.progress}/{task.target}). Do a browser action first "
                                    f"(click, type, go_back, etc.) before marking progress again."
                                )
                            else:
                                reasoning = think_reasoning or "Completed"
                                task.progress += 1
                                task.history.append(reasoning)
                                actions_since_progress = 0
                                browser_actions_since_progress = 0
                                self.event_logger.system_info(f"✓ Progress (via think): {reasoning} ({task.progress}/{task.target})")
                                last_action_summary = f"You marked progress via think: \"{reasoning}\" ({task.progress}/{task.target})"

                                # Complete if target reached
                                if numeric_target and task.progress >= numeric_target:
                                    return TaskResult(
                                        success=True,
                                        completion_status=TaskCompletionStatus.COMPLETED,
                                        progress=task.progress,
                                        target=task.target,
                                        history=task.history,
                                        reasoning=f"Completed: {task.progress}/{task.target}",
                                    )

                                # For "done": only actually finish if target is "all" (non-numeric)
                                # For numeric targets, treat done same as mark_progress — the agent
                                # can't prematurely end a task that hasn't reached its target
                                if think_next_action == "done" and numeric_target is None:
                                    return TaskResult(
                                        success=True,
                                        completion_status=TaskCompletionStatus.COMPLETED,
                                        progress=task.progress,
                                        target=task.target,
                                        history=task.history,
                                        reasoning=reasoning,
                                    )

                        elif think_next_action == "stuck":
                            # Agent decided it's stuck
                            active_strategy = None  # Clear strategy
                            reasoning = think_reasoning or "Stuck"
                            self.event_logger.system_info(f"✗ Agent stuck: {reasoning}")
                            return TaskResult(
                                success=False,
                                completion_status=TaskCompletionStatus.STUCK,
                                progress=task.progress,
                                target=task.target,
                                history=task.history,
                                reasoning=reasoning,
                            )

                        elif think_next_action == "continue":
                            # Set active strategy from the think reasoning
                            active_strategy = think_reasoning
                            last_action_summary = f"You thought: \"{think_reasoning}\""

                        # Release checkpoint and keep looping
                        checkpoint_pending = False
                        continue

                    # Handle assert/flag (non-browser actions)
                    if current_action and (current_action.lower().startswith("assert:") or
                                          current_action.lower().startswith("flag:")):
                        # Execute via executor (records in session tracker)
                        self.action_executor.act(
                            action_step=action_step,
                            detected_elements=detected_elements,
                            page_info=page_info,
                            environment_state=environment_state,
                            current_iteration=self._current_iteration,
                        )
                        action_type = "assert" if current_action.lower().startswith("assert:") else "flag"
                        action_content = current_action.split(":", 1)[1].split("|")[0].strip() if ":" in current_action else ""
                        last_action_summary = f"You called {action_type}: \"{action_content}\""
                        actions_since_progress += 1
                        continue

                    # Handle tab management actions (intercepted by controller)
                    if current_action and current_action.lower().startswith("switch_tab:"):
                        if self.tab_manager:
                            tab_id = current_action.split(":", 1)[1].strip()
                            try:
                                new_page = self.tab_manager.switch_to(tab_id)
                                self.action_executor.set_page(new_page)
                                title = ""
                                try:
                                    title = new_page.title()
                                except Exception:
                                    pass
                                last_action_summary = f"Switched to tab [{tab_id}]: \"{title}\""
                                browser_actions_since_progress += 1
                                suppress_mark_progress = False
                                checkpoint_pending = True
                            except ValueError as e:
                                last_action_summary = f"switch_tab FAILED: {e}"
                        else:
                            last_action_summary = "switch_tab FAILED: Tab management not available"
                        actions_since_progress += 1
                        continue

                    if current_action and current_action.lower().startswith("close_tab:"):
                        if self.tab_manager:
                            tab_id = current_action.split(":", 1)[1].strip()
                            try:
                                new_page = self.tab_manager.close_tab(tab_id)
                                self.action_executor.set_page(new_page)
                                active = self.tab_manager.get_active()
                                active_id = active.id if active else "?"
                                last_action_summary = f"Closed tab [{tab_id}]. Now on tab [{active_id}]"
                                browser_actions_since_progress += 1
                                suppress_mark_progress = False
                                checkpoint_pending = True
                            except ValueError as e:
                                last_action_summary = f"close_tab FAILED: {e}"
                        else:
                            last_action_summary = "close_tab FAILED: Tab management not available"
                        actions_since_progress += 1
                        continue

                    if current_action and current_action.lower().startswith("open_tab:"):
                        if self.tab_manager:
                            url = current_action.split(":", 1)[1].strip() or None
                            try:
                                new_page = self.tab_manager.open_tab(url)
                                self.action_executor.set_page(new_page)
                                active = self.tab_manager.get_active()
                                active_id = active.id if active else "?"
                                last_action_summary = f"Opened new tab [{active_id}]"
                                if url:
                                    last_action_summary += f" at {url}"
                                browser_actions_since_progress += 1
                                suppress_mark_progress = False
                                checkpoint_pending = True
                            except Exception as e:
                                last_action_summary = f"open_tab FAILED: {e}"
                        else:
                            last_action_summary = "open_tab FAILED: Tab management not available"
                        actions_since_progress += 1
                        continue

                    if current_action and current_action.lower().startswith("dismiss_dialog:"):
                        if self.tab_manager and self.tab_manager.pending_dialog:
                            # Parse: "dismiss_dialog: accept=True | input_text=..."
                            parts_str = current_action.split(":", 1)[1].strip()
                            accept = "accept=true" in parts_str.lower()
                            input_text = None
                            if "input_text=" in parts_str:
                                input_text = parts_str.split("input_text=", 1)[1].strip()
                            self.tab_manager.dismiss_dialog(accept, input_text)
                            action_word = "accepted" if accept else "dismissed"
                            last_action_summary = f"Dialog {action_word}"
                            checkpoint_pending = False  # Unblock the agent
                        else:
                            last_action_summary = "dismiss_dialog: No dialog pending"
                        actions_since_progress += 1
                        continue

                    # Execute browser action
                    result = self.action_executor.act(
                        action_step=action_step,
                        detected_elements=detected_elements,
                        page_info=page_info,
                        environment_state=environment_state,
                        failed_actions=[],
                        base_knowledge=self.base_knowledge,
                        current_iteration=self._current_iteration,
                    )

                    actions_since_progress += 1

                    # Build last_action_summary from the action
                    result_str = "success" if result.success else "failed"
                    last_action_summary = self._build_action_summary(action_step, result_str)

                    # Force checkpoint after any browser action (success or failure)
                    # On failure: browser_actions_since_progress stays 0, so mark_progress is blocked
                    # — the agent can only think, forcing it to reason about the failure
                    checkpoint_pending = True

                    if result.success:
                        browser_actions_since_progress += 1
                        suppress_mark_progress = False  # Reset suppression - browser action completed

                    # Sync tab manager after browser actions (click may have opened a new tab)
                    if result.success and self.tab_manager:
                        active_tab = self.tab_manager.get_active()
                        if active_tab and self.browser.page is not active_tab.page:
                            self.action_executor.set_page(active_tab.page)

                    # Track failures
                    if not result.success:
                        try:
                            failed_action = FailedAction(
                                action=current_action,
                                overlay_index=result.metadata.get("overlay_index") if result.metadata else None,
                                url=page_info.url if page_info else "",
                                page_title=page_info.title if page_info else None,
                                timestamp=time.time()
                            )
                            failed_elements.append(failed_action)
                        except Exception:
                            pass

            except Exception as e:
                return TaskResult(
                    success=False,
                    completion_status=TaskCompletionStatus.STUCK,
                    progress=task.progress,
                    target=task.target,
                    history=task.history,
                    reasoning=f"Error: {str(e)}",
                )

        # Max actions reached
        if task.progress > 0:
            return TaskResult(
                success=False,
                completion_status=TaskCompletionStatus.PARTIAL,
                progress=task.progress,
                target=task.target,
                history=task.history,
                reasoning=f"Partial completion: {task.progress}/{task.target} (max actions reached)",
            )
        else:
            return TaskResult(
                success=False,
                completion_status=TaskCompletionStatus.STUCK,
                progress=task.progress,
                target=task.target,
                history=task.history,
                reasoning="Stuck: no progress made",
            )

    def _validate_extraction_schema(
        self,
        extracted_data: Any,
        schema: Dict[str, Any],
        action: str
    ) -> Dict[str, Any]:
        """
        Validate extracted data against the expected schema.

        Args:
            extracted_data: The data extracted by the agent
            schema: The JSON schema to validate against
            action: The extraction action (for logging)

        Returns:
            Dict with keys:
                - valid: bool indicating if validation passed
                - error: str with error message if validation failed
                - expected_fields: list of expected field names
                - actual_fields: list of actual field names
        """
        expected_fields = schema.get("required", [])

        # Handle different data formats
        if isinstance(extracted_data, BaseModel):
            extracted_data = extracted_data.model_dump()

        if isinstance(extracted_data, dict):
            # Single extraction - filter out metadata fields (start with _)
            actual_fields = [k for k in extracted_data.keys() if not k.startswith('_')]
        elif isinstance(extracted_data, list) and len(extracted_data) > 0:
            # List of extractions - check first item
            if isinstance(extracted_data[0], dict):
                actual_fields = [k for k in extracted_data[0].keys() if not k.startswith('_')]
            else:
                actual_fields = []
        else:
            actual_fields = []

        # Check if all required fields are present
        missing_fields = [f for f in expected_fields if f not in actual_fields]
        # Only report extra fields that are NOT metadata (don't start with _)
        extra_fields = [f for f in actual_fields if f not in expected_fields]

        if missing_fields or extra_fields:
            error_parts = []
            if missing_fields:
                error_parts.append(f"Missing fields: {missing_fields}")
            if extra_fields:
                error_parts.append(f"Unexpected fields: {extra_fields}")

            return {
                "valid": False,
                "error": "; ".join(error_parts),
                "expected_fields": expected_fields,
                "actual_fields": actual_fields,
            }

        return {
            "valid": True,
            "error": None,
            "expected_fields": expected_fields,
            "actual_fields": actual_fields,
        }

    def run_task_loop(
        self,
        task_instruction: str,
        original_prompt: str,
        extraction_schema: Optional[Dict[str, Any]] = None,
        current_sequence_task: Optional[str] = None,
    ) -> TaskResult:
        """
        Run a task loop for a specific instruction.

        This executes actions until the agent issues a complete: command or max global iterations reached.
        It handles:
        - Context integration (previous task results)
        - History-based completion checks
        - Snapshot execution
        - Planning and Action execution
        - Extraction validation

        Args:
            task_instruction: The specific instruction to execute
            original_prompt: Original user prompt (for context)
            extraction_schema: Optional schema for extraction validation
            context: Optional context dictionary (e.g. previous task results)

        Returns:
            TaskResult indicating success/failure
        """

        from agent.action_planner import ActionPlanner

        # Track task-specific failed/ineffective actions
        task_failed_actions: List[str] = []
        failed_elements: List[FailedAction] = []

        # Track action history for function calling context
        action_history = []
        
        last_extracted_data = None

        # Use global mission-wide iteration counter
        while self._current_iteration < self.max_iterations:
            self._current_iteration += 1
            iteration = self._current_iteration

            try:
                page_info = self.page_utils.get_page_info()
                detected_elements = build_page_elements(self.browser.page, page_info)
            except Exception as e:
                self.event_logger.system_error("Error building page elements", error=e)
                detected_elements = None
                
            # Debug: Log iteration start
            self.event_logger.system_debug(
                f"[Task Loop] Starting global iteration {iteration}/{self.max_iterations} for task: '{task_instruction}'"
            )
            self.event_logger.iteration_start(task_instruction, iteration, self.max_iterations)


            # Capture current state
            try:
                if hasattr(self, "_maybe_wait_for_turn_load"):
                    self._maybe_wait_for_turn_load(reason="mini-loop iteration")
                snapshot = self._capture_snapshot(full_page=False)
            except Exception as e:
                return TaskResult(
                    success=False,
                    confidence=0.0,
                    reasoning=f"Failed to capture snapshot: {str(e)}",
                    evidence={"error": str(e)},
                )

            # Build environment state
            environment_state = EnvironmentState(
                browser_state=snapshot,
                interaction_history=self.session_tracker.interaction_history,
                user_prompt=original_prompt,
                task_start_url=self.task_start_url,
                task_start_time=self.task_start_time,
                current_url=snapshot.url,
                page_title=snapshot.title,
                visible_text=snapshot.visible_text,
                url_history=self.session_tracker.url_history.copy() if self.session_tracker.url_history else [],
                url_pointer=self.session_tracker.url_pointer
            )

            # Create action planner for this task
            action_planner = ActionPlanner(
                task_instruction,
                self.session_tracker,
                base_knowledge=self.base_knowledge if hasattr(self, "base_knowledge") else [],
                model_name=self.agent_model_name,
                reasoning_level=self.agent_reasoning_level,
                image_detail=self.config.model.image_detail,
                interaction_summary_limit=None,
                current_iteration=iteration,
            )

            # Generate next action using function calling
            try:
                actions_list, error = action_planner.get_next_actions_with_function_calling(
                    environment_state=environment_state,
                    screenshot=snapshot.screenshot,
                    notebook=self.notebook,
                    element_data=detected_elements,
                )
                
                for action_step in actions_list:
                    # Debug logging
                    try:
                        self.event_logger.system_debug(
                            f"Generated action: {action_step.action if action_step else 'N/A'}"
                        )

                        if not action_step:
                            try:
                                self.event_logger.iteration_fail(task_instruction, iteration, error="no_action")
                            except Exception:
                                pass

                            return TaskResult(
                                success=False,
                                confidence=0.0,
                                reasoning=error or "No action generated",
                                evidence={"task_instruction": task_instruction},
                            )

                        current_action = action_step.action
                            
                        # Check for complete_sequence command
                        if current_action and current_action.lower().startswith("complete_sequence"):
                            self._temp_end_sequence = True
                            self._temp_end_sequence_reasoning = current_action.split(":", 1)[1].strip() if ":" in current_action else "Sequence completed"
                            print("Sequence complete command received")
                            # self.event_logger.sequence_complete(task_instruction)
                            return TaskResult(
                                success=True,
                                confidence=1.0,
                                reasoning=self._temp_end_sequence_reasoning,
                                evidence={
                                    "iterations": iteration,
                                    "actions_tried": iteration,
                                    "extracted_data": last_extracted_data,
                                },
                            )

                        # Check for complete: command
                        if current_action and current_action.lower().startswith("complete:"):
                            self._temp_end_sequence = True
                            self._temp_end_sequence_reasoning = current_action.split(":", 1)[1].strip() if ":" in current_action else "Sequence completed"
                            completion_reasoning = current_action.split(":", 1)[1].strip() if ":" in current_action else "Task completed"
                            self.event_logger.iteration_complete(task_instruction, iteration, self.max_iterations)
                            
                            return TaskResult(
                                success=True,
                                confidence=1.0,
                                reasoning=completion_reasoning,
                                evidence={
                                    "iterations": iteration,
                                    "actions_tried": iteration,
                                    "extracted_data": last_extracted_data,
                                },
                            )
                        # Execute the action
                        try:
                            result = self.action_executor.act(
                                action_step=action_step,
                                detected_elements=detected_elements,
                                page_info=page_info,
                                extraction_schema=extraction_schema,
                                environment_state=environment_state,
                                failed_actions=task_failed_actions,
                                base_knowledge=self.base_knowledge,
                                current_iteration=iteration,
                                current_sequential_iteration=self._current_sequential_iteration,
                            )

                            # Track ineffective/failed actions
                            if not result.success:
                                task_failed_actions.append(current_action)
                                # Get overlay_index from result metadata or from last interaction
                                overlay_index = None
                                if result.metadata and "overlay_index" in result.metadata:
                                    overlay_index = result.metadata.get("overlay_index")
                                else:
                                    # Fallback: get from session tracker's last interaction
                                    try:
                                        overlay_index = self.action_executor.session_tracker.get_last_interaction_overlay_index()
                                    except Exception:
                                        pass
                                
                                # Create FailedAction with page context
                                try:
                                    failed_action = FailedAction(
                                        action=current_action,
                                        overlay_index=overlay_index,
                                        url=page_info.url if page_info else "",
                                        page_title=page_info.title if page_info else None,
                                        timestamp=time.time()
                                    )
                                    failed_elements.append(failed_action)
                                except Exception as e:
                                    # Fallback: create minimal FailedAction if page_info is unavailable
                                    try:
                                        failed_action = FailedAction(
                                            action=current_action,
                                            overlay_index=overlay_index,
                                            url="",
                                            page_title=None,
                                            timestamp=time.time()
                                        )
                                        failed_elements.append(failed_action)
                                    except Exception:
                                        pass

                                # Log failure
                                self.event_logger.system_debug(
                                        f"   ✗ Action failed: {result.error or result.message}"
                                    )

                                # Action failed, continue to next iteration to generate new action
                                continue

                            if result.success and current_action.lower().startswith("extract:"):
                                last_extracted_data = result.data if hasattr(result, "data") else None
                                try:
                                    extraction_prompt = current_action.split(":", 1)[1].strip()
                                    self.event_logger.extraction_success(extraction_prompt, result=result.data)
                                except Exception:
                                    pass
                                # Validate extraction against schema if available
                                if extraction_schema and hasattr(result, 'data') and result.data:
                                    validation_result = self._validate_extraction_schema(
                                        result.data,
                                        extraction_schema,
                                        current_action
                                    )
                                    if not validation_result["valid"]:
                                        # Log validation failure but don't fail the task
                                        # The agent can try again in the next iteration
                                        try:
                                            self.event_logger.system_warning(
                                                f"⚠️ Extraction schema validation failed: {validation_result['error']}"
                                            )
                                            self.event_logger.system_info(
                                                f"   Expected fields: {validation_result['expected_fields']}"
                                            )
                                            self.event_logger.system_info(
                                                f"   Actual fields: {validation_result['actual_fields']}"
                                            )
                                        except Exception:
                                            pass
                                        # Mark as failed so agent can retry with correct schema
                                        task_failed_actions.append(current_action)

                                        # Continue to next iteration to retry with correct schema
                                        continue
                                    else:
                                        # Validation passed
                                        try:
                                            self.event_logger.system_info(
                                                "✓ Extraction schema validation passed"
                                            )
                                        except Exception:
                                            pass

                                # Respect configuration to disable auto-completion on extract actions
                                if not self.auto_complete_extract_commands:
                                    try:
                                        self.event_logger.system_debug(
                                            "[Extraction auto-complete disabled] Waiting for complete: command"
                                        )
                                    except Exception:
                                        pass
                                    # Skip auto-complete and continue normal loop behavior
                                    continue

                                # Define all possible action verbs
                                action_verbs = [
                                    "click", "type", "clear_text", "select", "scroll", "press", "open",
                                    "back", "forward", "wait", "defer", "upload", "datetime",
                                    "form", "interceptor"
                                ]

                                # Check if task has any non-extraction action verbs
                                task_lower = task_instruction.lower()
                                has_other_actions = any(verb in task_lower for verb in action_verbs)

                                # Debug logging for extraction auto-complete decision
                                try:
                                    self.event_logger.system_debug(
                                        f"[Extraction auto-complete check] "
                                        f"has_other_actions={has_other_actions}, "
                                        f"task='{task_instruction[:80]}...'"
                                    )
                                except Exception:
                                    pass

                                # If this is a pure extraction task (no other actions), auto-complete
                                if not has_other_actions:
                                    # Log mini-loop completion
                                    try:
                                        self.event_logger.system_info(
                                            f"✓ Pure extraction task auto-completed: {current_action}"
                                        )
                                    except Exception:
                                        pass

                                    # Extraction succeeded, task is complete
                                    return TaskResult(
                                        success=True,
                                        confidence=1.0,
                                        reasoning=f"Extraction completed successfully: {current_action}",
                                        evidence={
                                            "iterations": iteration,
                                            "actions_tried": iteration,
                                            "extracted_data": result.data if hasattr(result, "data") else None,
                                        },
                                    )
                                else:
                                    # Multi-action task, agent must issue complete: command
                                    try:
                                        self.event_logger.system_debug(
                                            "[Extraction] Multi-action task detected, waiting for agent to issue complete: command"
                                        )
                                    except Exception:
                                        pass
                                # else: Multi-action task, let agent issue complete: command

                        except Exception as e:
                            task_failed_actions.append(current_action)
                            # Continue to next iteration after exception
                            continue

                        # Add action to history for context in next iteration
                        action_history.append(action_step)
                    except Exception:
                        pass
            except Exception as e:
                # Debug logging
                try:
                    self.event_logger.system_debug(
                        f"[Iteration {iteration}] Exception in get_next_action_with_function_calling: {str(e)}"
                    )
                except Exception:
                    pass

                return TaskResult(
                    success=False,
                    confidence=0.0,
                    reasoning=f"Failed to determine action: {str(e)}",
                    evidence={"error": str(e)},
                )

        # Max iterations reached without completion
        try:
            self.event_logger.iteration_fail(task_instruction, self.max_iterations, error="max_iterations_reached")
        except Exception:
            pass
        return TaskResult(
            success=False,
            confidence=0.0,
            reasoning=f"Mission reached maximum global iterations ({self.max_iterations}) without completing task",
            evidence={"max_iterations": self.max_iterations, "task_instruction": task_instruction},
        )

