"""Agent controller with mission/task orchestration.

Architecture overview:
- Planner (plan_next) proposes tasks with required_tools_for_completion.
- Controller executes tools, records memory, and marks progress per task.
- Mission completion: planner declares done, controller trusts it.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable, Union, Type, TYPE_CHECKING
import hashlib

from browser.dom import build_page_elements
from core.browser import ExecutionTimer
from core.executor.base import Executor
from agent.memory import InteractionType, MemoryState, NarrativeMemory
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
from agent.action_tools import PLANNING_TOOLS
from agent.prompts import (
    DecisionContext,
    PLANNER_DEVELOPER_POLICY,
    PLANNER_MEMORY_CONTRACT,
    SHARED_CONTRADICTION_GATE,
)
from utils.debug_print import dprint
from lib.ai import (
    ReasoningLevel,
    generate_action_with_tools,
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


@dataclass
class TaskExecutionState:
    """Mutable per-task execution state for the unified loop."""
    total_actions: int = 0
    actions_since_progress: int = 0
    browser_actions_since_progress: int = 0
    checkpoint_pending: bool = False
    suppress_mark_progress: bool = False
    last_action_summary: Optional[str] = None
    progress_notes: List[str] = field(default_factory=list)
    failed_elements: List[FailedAction] = field(default_factory=list)
    validation_failures: int = 0


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
        self.memory_store: NarrativeMemory = NarrativeMemory(self.browser)
        self.memory_store.set_base_knowledge(self.base_knowledge)
        
        self.page_utils = PageUtils(self.browser.page)

        # Initialize action executor
        self.action_executor: Executor = Executor(
            self.browser,
            self.memory_store,
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

    def _maybe_wait_for_iteration_load(self, reason: str = "iteration") -> None:
        if not self.config.execution.wait_for_load_before_iteration:
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
        Execute a mission using incremental agent-driven planning.

        Instead of decomposing all tasks upfront, the agent plans one task at a
        time: after each task completes (or fails), it sees the current screenshot
        and completed work, then decides the next task or declares done.

        Args:
            user_mission: User's high-level request
        Returns:
            MissionResult indicating success or failure
        """
        self._task_tracker = {}
        self._original_user_mission = user_mission
        self._current_iteration = 0
        self._current_sequential_iteration = 1
        self.event_logger.agent_start(user_mission)

        # Initialize task tracking
        try:
            self.task_start_url = self.browser.page.url
        except Exception:
            self.task_start_url = "unknown"
        self.task_start_time = time.time()

        self.execution_timer.start_task()

        if self.base_knowledge:
            self.memory_store.set_base_knowledge(self.base_knowledge)
        self.memory_store.start_mission(user_mission)

        # Check if starting from a blank page
        if self.browser.page.url.startswith("about:blank"):
            self.event_logger.agent_error("Page is on initial blank page.")
            if self.execution_timer.task_start_time is not None:
                self.execution_timer.end_task()
            self.mission_result.success = False
            self.event_logger.agent_complete(success=False, reasoning="Page is blank")
            return self.mission_result

        # ── Incremental planning loop ──
        max_planning_iterations = self.config.task_execution.max_tasks_per_mission
        completed_tasks: List[Dict[str, Any]] = []
        planning_iteration = 0
        planning_feedback: Optional[str] = None
        mission_declared_complete = False
        self._latest_final_answer_draft = ""

        while planning_iteration < max_planning_iterations:
            planning_iteration += 1

            # Run a planning iteration — the planner sees the current screen + history
            plan = self._run_planning_iteration(
                user_mission,
                completed_tasks,
                planning_iteration,
                planning_feedback=planning_feedback,
            )
            planning_feedback = None

            if plan is None:
                # Planning call failed
                self.event_logger.system_error("Planning iteration returned None")
                break

            self._latest_final_answer_draft = str(plan.get("final_answer_draft", "") or "").strip()

            if plan.get("mission_complete"):
                if not self._latest_final_answer_draft:
                    # Reject: planner must provide a final answer summarizing
                    # the result so the user gets actionable output.
                    planning_feedback = (
                        "Mission declared complete but no final_answer_draft provided. "
                        "If the mission asks for data (get/find/extract), use extract_data "
                        "to capture it first, then declare complete with final_answer_draft "
                        "containing the answer. Otherwise, re-declare complete with a "
                        "final_answer_draft summarizing what was accomplished."
                    )
                    self.event_logger.system_info(
                        "Mission completion rejected: no final_answer_draft"
                    )
                    continue
                mission_declared_complete = True
                self.event_logger.system_info(
                    f"Mission declared complete after {len(completed_tasks)} tasks"
                )
                break

            # Build a Task from the plan
            task_goal = plan["task"]
            task_target = plan.get("target", 1)
            start_hint = plan.get("start_hint", "")
            raw_required_tools = plan.get("required_tools_for_completion", [])
            if isinstance(raw_required_tools, list):
                required_tools_for_completion = [
                    str(tool).strip()
                    for tool in raw_required_tools
                    if isinstance(tool, str) and str(tool).strip()
                ]
            else:
                required_tools_for_completion = []

            task = Task(
                goal=task_goal,
                target=task_target,
                required_tools_for_completion=required_tools_for_completion,
                start_hint=start_hint or None,
                task_id=f"task_{planning_iteration}",
                created_at=time.time(),
            )

            self.event_logger.task_start(
                task_id=task.task_id,
                task=task.goal,
                task_type=(
                    f"target={task.target}, "
                    f"required_tools={','.join(task.required_tools_for_completion) or 'none'}"
                ),
            )

            # Start checkpoint mode once per mission (first task only).
            start_in_checkpoint = (planning_iteration == 1)

            # Execute the task
            result = self._execute_task(
                task,
                user_mission,
                planning_iteration=planning_iteration,
                start_in_checkpoint=start_in_checkpoint,
            )
            self.mission_result.task_results.append(result)

            # Build a concise summary for the planner's context
            summary: Dict[str, Any] = {
                "task": task_goal,
                "target": task_target,
                "success": result.success,
                "progress": result.progress,
                "required_tools_for_completion": list(task.required_tools_for_completion),
            }
            if result.reasoning:
                summary["reasoning"] = result.reasoning[:200]

            completed_tasks.append(summary)

            if result.success:
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                self.event_logger.task_complete(
                    task.task_id,
                    task_type=(
                        f"target={task.target}, "
                        f"required_tools={','.join(task.required_tools_for_completion) or 'none'}"
                    ),
                )
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                self.event_logger.task_fail(
                    task.task_id,
                    error=result.reasoning,
                    task_type=(
                        f"target={task.target}, "
                        f"required_tools={','.join(task.required_tools_for_completion) or 'none'}"
                    ),
                )
                # Don't abort — let the planner see the failure and decide what to do next

        # ── Finalize ──
        if self.execution_timer.task_start_time is not None:
            self.execution_timer.end_task()

        self.mission_result.final_answer_draft = self._latest_final_answer_draft

        if mission_declared_complete:
            self.mission_result.success = True
            self.mission_result.reasoning = f"Mission complete after {len(completed_tasks)} tasks"
        elif not self.mission_result.task_results:
            self.mission_result.success = False
            self.mission_result.reasoning = "No tasks were executed"
        else:
            any_success = any(r.success for r in self.mission_result.task_results)
            all_failed = all(not r.success for r in self.mission_result.task_results)
            if all_failed:
                self.mission_result.success = False
                self.mission_result.reasoning = "All tasks failed"
            else:
                self.mission_result.success = any_success
                self.mission_result.reasoning = (
                    f"Completed {sum(1 for r in self.mission_result.task_results if r.success)}/"
                    f"{len(self.mission_result.task_results)} tasks"
                )

        return self.mission_result

    def _run_planning_iteration(
        self,
        user_mission: str,
        completed_tasks: List[Dict[str, Any]],
        planning_iteration: int,
        planning_feedback: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Run a single planning iteration: capture screenshot, call LLM with PLANNING_TOOLS,
        parse the result into a task description (or mission-complete signal).

        Returns:
            Dict with keys: mission_complete (bool), task (str), target, start_hint,
            required_tools_for_completion, final_answer_draft
            or None on error.
        """
        self.event_logger.planning_iteration_start(planning_iteration, user_mission)

        # Capture a lean screenshot (no overlays, no element detection)
        try:
            screenshot = self.browser.page.screenshot(full_page=False)
        except Exception:
            screenshot = None

        # Get current page context
        try:
            current_url = self.browser.page.url
            page_title = self.browser.page.title()
        except Exception:
            current_url = "unknown"
            page_title = ""

        system_prompt = self._build_planning_system_prompt()
        user_prompt = self._build_planning_user_prompt(
            user_mission,
            current_url,
            page_title,
            completed_tasks,
            planning_iteration,
            planning_feedback=planning_feedback,
        )

        try:
            tool_calls = generate_action_with_tools(
                prompt=user_prompt,
                tools=PLANNING_TOOLS,
                system_prompt=system_prompt,
                developer_prompt=PLANNER_DEVELOPER_POLICY,
                image=screenshot,
                image_detail=self.config.model.image_detail,
                model=self.agent_model_name,
                reasoning_level=self.agent_reasoning_level,
                tool_choice="required",
                parallel_tool_calls=False,
            )
        except Exception as e:
            self.event_logger.system_error(f"Planning LLM call failed: {e}")
            return None

        # Parse the first tool call result
        if not tool_calls:
            self.event_logger.system_error("Planning returned no tool calls")
            return None

        call = tool_calls[0]
        fn_name = call.get("function_name", "")
        args = call.get("arguments", {})

        if fn_name != "plan_next":
            self.event_logger.system_error(f"Unexpected planning function: {fn_name}")
            return None

        task_text = args.get("task", "").strip()
        reasoning = args.get("reasoning", "")
        raw_target = args.get("target", 1)
        target, target_note, target_note_is_warning = self._normalize_planning_target(raw_target)
        if target_note:
            if target_note_is_warning:
                self.event_logger.system_warning(target_note)
            else:
                self.event_logger.system_debug(target_note)
        start_hint = args.get("start_hint", "")
        required_tools_for_completion = args.get("required_tools_for_completion", [])
        if not isinstance(required_tools_for_completion, list):
            required_tools_for_completion = []
        final_answer_draft = str(args.get("final_answer_draft", "") or "").strip()

        # Empty task = mission complete
        mission_complete = task_text == ""

        self.event_logger.planning_iteration_complete(
            planning_iteration,
            task=task_text,
            mission_complete=mission_complete,
            reasoning=reasoning,
        )

        if mission_complete:
            self.event_logger.system_info(f"Planner reasoning (done): {reasoning}")
        else:
            target_display = target if isinstance(target, str) else f"{target}x" if target > 1 else ""
            self.event_logger.system_info(
                f"Planner → Task {planning_iteration}: \"{task_text}\""
                + (f" (target={target_display})" if target_display else "")
            )

        return {
            "mission_complete": mission_complete,
            "task": task_text,
            "target": target,
            "start_hint": start_hint,
            "required_tools_for_completion": required_tools_for_completion,
            "final_answer_draft": final_answer_draft,
        }

    @staticmethod
    def _normalize_planning_target(raw_target: Any) -> Tuple[Union[int, str], Optional[str], bool]:
        """
        Normalize planner target to either:
        - positive integer (1+), or
        - "all"

        Returns:
            (normalized_target, note, note_is_warning)
        """
        if raw_target is None:
            return 1, "Planner target missing; defaulted to 1.", False

        # bool is a subclass of int; handle it explicitly as invalid
        if isinstance(raw_target, bool):
            return 1, f"Invalid planner target {raw_target!r} (bool); defaulted to 1.", True

        if isinstance(raw_target, int):
            if raw_target >= 1:
                return raw_target, None, False
            return 1, f"Invalid planner target {raw_target!r} (<1); defaulted to 1.", True

        if isinstance(raw_target, float):
            if raw_target.is_integer() and raw_target >= 1:
                normalized = int(raw_target)
                return normalized, f"Coerced planner target {raw_target!r} to integer {normalized}.", False
            return 1, f"Invalid planner target {raw_target!r} (non-integer float); defaulted to 1.", True

        if isinstance(raw_target, str):
            cleaned = raw_target.strip().lower()
            if cleaned == "all":
                return "all", None, False
            if cleaned.isdigit():
                normalized = int(cleaned)
                if normalized >= 1:
                    return normalized, f"Coerced planner target {raw_target!r} to integer {normalized}.", False
            return 1, f"Invalid planner target {raw_target!r}; expected positive integer or 'all'. Defaulted to 1.", True

        return 1, f"Invalid planner target type {type(raw_target).__name__}; defaulted to 1.", True

    def _build_planning_system_prompt(self) -> str:
        """Build the system prompt for the incremental planner."""
        return f"""You are a mission planner for a browser automation agent.

Goal:
• Decide the NEXT single task, or declare mission complete with task="".

Core planning rules:
1. One concern per task. Do not combine navigation + loop in one task.
2. Use small actionable tasks grounded in tools (click/type/press/scroll/open/extract_data).
3. target controls repetition. Get it right:
   When target > 1, write the task as a GENERIC action — never name a specific item.
   The agent repeats the task description each round; naming a specific element causes it to act on that same element every round.

   Ordered missions ("first 3", "top 5") — use "the next" to signal sequential progression:
   ✓ Mission: "Click the first 3 job postings" → task="Click the next job posting", target=3
   ✓ Mission: "Star the top 5 repos"           → task="Star the next repo", target=5
   ✓ Mission: "Open the first 4 emails"        → task="Open the next email", target=4

   Unordered missions (any N items) — use "a/an" for any item:
   ✓ Mission: "Like 5 photos"    → task="Like a photo", target=5
   ✓ Mission: "Delete 10 emails" → task="Delete an email", target=10

   WRONG — naming a specific item with target > 1:
   ✗ "Click the first job posting", target=3   → clicks the SAME posting 3 times
   ✗ "Star the top repo", target=5             → stars the same repo 5 times
   ✗ "Like the first photo", target=4          → likes the same photo 4 times

   WRONG — using target > 1 for distinct actions that need separate tasks:
   ✗ "Fill out a form field", target=3         → each field needs different values; use target=1 per field
   ✗ "Click a navigation menu item", target=3  → each item is a different destination; use target=1 each
   ✗ "Complete a checkout step", target=4       → each step has different inputs; use target=1 each

4. Skip already-completed work visible in current page + memory evidence.
5. Use target="all" only when count is genuinely open-ended.
6. If prior task failed, adapt strategy instead of repeating blindly.
7. Declare mission complete (task="") only when all user-requested work is done. When the mission asks to get/find/extract data, you MUST create a task with required_tools_for_completion=['extract_data'] to capture the data before declaring complete. Data visible on screen is NOT extracted until extract_data is called.
8. When task requires specific tool usage before completion, set required_tools_for_completion.
9. When task="" (mission complete), you MUST include final_answer_draft with a concrete summary of the result/answer for the user. This is required, not optional.

Output schema:
• reasoning: why this is the right next task now.
• task: next task text (empty string means mission complete).
• target: integer >=1 or "all".
• start_hint: optional immediate first step.
• required_tools_for_completion: optional strict tools required for this task's completion.
• final_answer_draft: optional final user-facing answer when mission complete.

{PLANNER_MEMORY_CONTRACT}
{SHARED_CONTRADICTION_GATE}
"""

    def _build_planning_user_prompt(
        self,
        user_mission: str,
        current_url: str,
        page_title: str,
        completed_tasks: List[Dict[str, Any]],
        planning_iteration: int,
        planning_feedback: Optional[str] = None,
    ) -> str:
        """Build the user prompt for the incremental planner."""
        lines = [f"PLANNING ITERATION: {planning_iteration}"]
        lines.append(f"MISSION: {user_mission}")
        lines.append(f"CURRENT PAGE: {current_url} — {page_title}")
        if planning_feedback:
            lines.append(f"RUNTIME FEEDBACK: {planning_feedback}")
        lines.append("")
        lines.append("RECENT MEMORY NARRATIVE (memory IDs):")
        lines.append(self.memory_store.get_narrative(n=12))
        lines.append("")
        lines.append("RECENT EXECUTED ACTION LEDGER (facts only, memory IDs):")
        lines.append(self.memory_store.get_executed_action_ledger(n=12))

        if completed_tasks:
            lines.append("")
            lines.append("COMPLETED TASK SUMMARIES (TS#):")
            for i, t in enumerate(completed_tasks, 1):
                status = "SUCCESS" if t["success"] else "FAILED"
                progress_str = ""
                target = t.get("target", 1)
                progress = t.get("progress", 0)
                if isinstance(target, int) and target > 1:
                    progress_str = f" [{progress}/{target}]"
                elif target == "all":
                    progress_str = f" [{progress} done]"

                line = f"  TS{i}. [{status}]{progress_str} {t['task']}"
                required_tools = t.get("required_tools_for_completion") or []
                if isinstance(required_tools, list) and required_tools:
                    line += f" (required_tools={','.join(str(tool) for tool in required_tools)})"
                if not t["success"] and t.get("reasoning"):
                    line += f" — {t['reasoning']}"
                lines.append(line)
        else:
            lines.append("")
            lines.append("COMPLETED TASK SUMMARIES (TS#): None yet — this is the first planning iteration.")

        return "\n".join(lines)

    def _capture_snapshot(self, full_page: bool = False) -> MemoryState:
        """
        Capture current browser state snapshot.
        
        Args:
            full_page: If True, capture full page screenshot (for exploration mode)
                      If False, capture viewport only (normal mode)
        """
        snapshot = self.memory_store._capture_current_state()
        
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

        # Current task list (set during execution)
        self.task_list: Optional[MissionPlan] = None
        self._extraction_model_cache: Dict[tuple[str, ...], Type[BaseModel]] = {}

    def _execute_task(
        self,
        task: Task,
        user_mission: str,
        planning_iteration: int,
        *,
        start_in_checkpoint: bool = False,
    ) -> TaskResult:
        """
        Execute a unified task (single or repetitive).

        Args:
            task: The task to execute (with target indicating repetition)
            user_mission: Original user mission for context
            start_in_checkpoint: Whether to begin this task in checkpoint mode

        Returns:
            TaskResult indicating success/failure and completion status
        """
        self.memory_store.start_task(task.goal)
        task.status = TaskStatus.IN_PROGRESS

        # Execute using the unified task loop
        result = self._run_unified_task_loop(
            task,
            user_mission,
            planning_iteration=planning_iteration,
            start_in_checkpoint=start_in_checkpoint,
        )

        # Update task with final completion status
        if result.completion_status:
            task.completion_status = result.completion_status

        return result

    def _get_latest_recommended_next_step(self) -> Tuple[Optional[str], Optional[str]]:
        """Return newest recommended step plus source memory ID."""
        return self.memory_store.get_latest_recommended_next_step()

    def _record_controller_action(
        self,
        *,
        action_type: str,
        action_step: Any,
        success: bool,
        error_message: Optional[str] = None,
        action_params: Optional[Dict[str, Any]] = None,
        before_state: Optional[MemoryState] = None,
        after_state: Optional[MemoryState] = None,
    ) -> None:
        """Record controller-handled actions into narrative memory."""
        try:
            args = getattr(action_step, "function_arguments", {}) or {}
            params = dict(action_params or {})

            evidence_ids = args.get("memory_evidence_ids")
            if isinstance(evidence_ids, list):
                normalized_ids = self.memory_store.normalize_memory_ids(evidence_ids)
                if normalized_ids:
                    params["memory_evidence_ids"] = normalized_ids

            if getattr(action_step, "function_name", None):
                params["tool"] = action_step.function_name

            if before_state is None:
                before_state = self.memory_store._capture_current_state()
            if after_state is None:
                after_state = self.memory_store._capture_current_state()

            self.memory_store.record_action(
                action_type=action_type,
                action_params=params,
                reasoning=args.get("reasoning") or getattr(action_step, "reasoning", None),
                before_state=before_state,
                after_state=after_state,
                success=success,
                error_message=error_message,
                mission=self.memory_store.current_mission,
                task=self.memory_store.current_task,
                reference_memory_ids=normalized_ids if isinstance(evidence_ids, list) else [],
            )
        except Exception:
            pass

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

    def _run_unified_task_loop(
        self,
        task: Task,
        original_prompt: str,
        planning_iteration: int,
        *,
        start_in_checkpoint: bool = False,
    ) -> TaskResult:
        """
        Unified task execution loop - handles both single and repetitive tasks.

        This replaces both _execute_normal_task and _execute_sequential_task with one unified loop.
        The agent uses mark_progress to signal completion of each unit of work.

        Args:
            task: Task with goal and target (1, N, or "all")
            original_prompt: Original user mission for context
            start_in_checkpoint: Whether this task should start in checkpoint mode

        Returns:
            TaskResult with completion status
        """
        from agent.action_planner import ActionPlanner

        # Get config values
        max_actions_per_task = self.config.task_execution.max_actions_per_task

        state = TaskExecutionState(
            checkpoint_pending=bool(start_in_checkpoint),
        )

        # Determine numeric target (None for "all")
        numeric_target = task.target if isinstance(task.target, int) else None
        required_tools_for_completion = [
            str(tool).strip().lower()
            for tool in (task.required_tools_for_completion or [])
            if isinstance(tool, str) and str(tool).strip()
        ]
        tools_used_since_progress: set[str] = set()

        def _record_progress(
            description: str,
            *,
            count: int = 1,
            done: bool = False,
            source: str = "mark_progress",
        ) -> Tuple[bool, Optional[TaskResult]]:
            """
            Record unit completion consistently for both direct mark_progress and think(next_action=mark_progress).
            Returns (recorded, completion_result). completion_result is set when task finishes immediately.
            """
            if state.browser_actions_since_progress == 0:
                self.event_logger.system_debug(
                    f"⚠ {source} ignored — no browser actions since last progress"
                )
                state.last_action_summary = (
                    f"mark_progress BLOCKED: No browser actions since your last progress mark "
                    f"({task.progress}/{task.target}). Do a browser action first "
                    f"(click, type, go_back, etc.) before marking progress again."
                )
                return False, None

            if required_tools_for_completion:
                missing_tools = [
                    tool
                    for tool in required_tools_for_completion
                    if tool not in tools_used_since_progress
                ]
                if missing_tools:
                    state.last_action_summary = (
                        "mark_progress BLOCKED: missing required tools for this task unit: "
                        + ", ".join(missing_tools)
                    )
                    return False, None

            task.progress += count
            state.progress_notes.append(description)
            state.actions_since_progress = 0
            state.browser_actions_since_progress = 0
            state.checkpoint_pending = False
            state.suppress_mark_progress = True
            tools_used_since_progress.clear()

            if source == "think":
                self.event_logger.system_info(
                    f"✓ Progress (via think): {description} ({task.progress}/{task.target})"
                )
                state.last_action_summary = (
                    f"You marked progress via think: \"{description}\" "
                    f"({task.progress}/{task.target})"
                )
            else:
                self.event_logger.system_info(
                    f"✓ Progress: {description} ({task.progress}/{task.target})"
                )
                state.last_action_summary = (
                    f"You marked progress: \"{description}\" "
                    f"({task.progress}/{task.target})"
                )

            target_reached = numeric_target is not None and task.progress >= numeric_target
            done_with_open_target = done and numeric_target is None
            if target_reached or done_with_open_target:
                return True, TaskResult(
                    success=True,
                    completion_status=TaskCompletionStatus.COMPLETED,
                    progress=task.progress,
                    target=task.target,
                    reasoning=f"Completed: {task.progress}/{task.target}",
                )

            return True, None

        while state.total_actions < max_actions_per_task:
            state.total_actions += 1
            self._current_iteration += 1

            # Check if target reached
            if numeric_target is not None and task.progress >= numeric_target:
                return TaskResult(
                    success=True,
                    completion_status=TaskCompletionStatus.COMPLETED,
                    progress=task.progress,
                    target=task.target,
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
                    reasoning=f"Failed to capture state: {str(e)}",
                )

            # Build environment state
            memory_recent = self.memory_store.get_recent(20)
            recent_executed_ids = self.memory_store.get_recent_executed_action_ids(n=20)
            recent_reflection_ids = self.memory_store.get_recent_reflection_ids(n=20)
            recommended_step, recommended_step_source_id = self._get_latest_recommended_next_step()

            decision_context = DecisionContext(
                planning_iteration=planning_iteration,
                action_iteration=state.total_actions,
                mission=original_prompt,
                task=task.goal,
                current_url=snapshot.url,
                page_title=snapshot.title,
                recommended_next_step=recommended_step,
                recommended_from_memory_id=recommended_step_source_id,
                executed_memory_ids=recent_executed_ids,
                reflection_memory_ids=recent_reflection_ids,
            )

            environment_state = EnvironmentState(
                browser_state=snapshot,
                memory_narrative=self.memory_store.get_narrative(n=20),
                memory_recent_ids=[entry.memory_id for entry in memory_recent],
                user_prompt=original_prompt,
                task_start_url=self.task_start_url,
                task_start_time=self.task_start_time,
                current_url=snapshot.url,
                page_title=snapshot.title,
                visible_text=snapshot.visible_text,
                url_history=self.memory_store.url_history.copy(),
                url_pointer=self.memory_store.url_pointer,
                decision_context=decision_context,
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
            force_think = False
            active_strategy = self.memory_store.get_latest_strategy() or None
            action_planner = ActionPlanner(
                task.goal,
                self.memory_store,
                base_knowledge=self.base_knowledge,
                model_name=self.agent_model_name,
                reasoning_level=self.agent_reasoning_level,
                image_detail=self.config.model.image_detail,
                max_actions_per_plan=self.config.execution.max_actions_per_plan,
                task_target=task.target,
                task_progress=task.progress,
                task_history=state.progress_notes,
                force_think=force_think,
                current_iteration=self._current_iteration,
                browser_actions_in_round=state.browser_actions_since_progress,
                checkpoint_mode=state.checkpoint_pending,
                suppress_mark_progress=state.suppress_mark_progress,
                active_strategy=active_strategy,
                last_action_summary=state.last_action_summary,
                tab_bar=tab_bar,
                dialog_notice=dialog_notice,
                tab_events=tab_events,
                dialog_pending=dialog_pending,
                start_hint=task.start_hint,
                recommended_next_step=recommended_step,
                recommended_next_step_source_id=recommended_step_source_id,
                decision_context=decision_context,
                required_tools_for_completion=required_tools_for_completion,
                tools_used_since_progress=tools_used_since_progress,
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
                    state.validation_failures += 1
                    if (
                        state.validation_failures
                        <= self.config.execution.validation_failure_escalation_limit
                    ):
                        state.last_action_summary = (
                            f"Planner/action validation issue: {error or 'No action generated'}. "
                            "Retrying planning."
                        )
                        state.checkpoint_pending = False
                        continue
                    return TaskResult(
                        success=False,
                        completion_status=TaskCompletionStatus.BLOCKED,
                        progress=task.progress,
                        target=task.target,
                        reasoning=(
                            "Repeated planning/action validation failures: "
                            f"{error or 'No action generated'}"
                        ),
                    )
                state.validation_failures = 0

                # Execute each action
                for action_step in actions_list:
                    function_name = (getattr(action_step, "function_name", None) or "").strip()
                    action_args = getattr(action_step, "function_arguments", {}) or {}
                    current_action = getattr(action_step, "action", "") or function_name

                    # Handle mark_progress (intercepted by controller, not executor)
                    if function_name == "mark_progress":
                        before_state = self.memory_store._capture_current_state()
                        description = str(action_args.get("description", "")).strip() or "Completed one unit"
                        count_raw = action_args.get("count", 1)
                        done = bool(action_args.get("done", False))
                        try:
                            count = int(count_raw)
                        except Exception:
                            count = 1
                        if count < 1:
                            count = 1

                        recorded, maybe_completion = _record_progress(
                            description,
                            count=count,
                            done=done,
                            source="mark_progress",
                        )
                        after_state = self.memory_store._capture_current_state()
                        self._record_controller_action(
                            action_type=InteractionType.MARK_PROGRESS.value,
                            action_step=action_step,
                            success=recorded,
                            error_message=None if recorded else "No browser actions since last progress mark",
                            action_params={
                                "description": description,
                                "count": count,
                                "done": done,
                            },
                            before_state=before_state,
                            after_state=after_state,
                        )
                        if not recorded:
                            state.actions_since_progress += 1
                            state.checkpoint_pending = False
                            continue
                        if maybe_completion:
                            return maybe_completion
                        continue

                    # Handle revise_target (intercepted by controller)
                    if function_name == "revise_target":
                        before_state = self.memory_store._capture_current_state()
                        new_target_raw = action_args.get("new_target", task.target)
                        reason = str(action_args.get("reason", "Target revised")).strip() or "Target revised"

                        if isinstance(new_target_raw, str) and new_target_raw.strip().lower() == "all":
                            task.target = "all"
                            numeric_target = None
                        else:
                            task.target = int(new_target_raw)
                            numeric_target = task.target

                        self.event_logger.system_info(f"→ Target revised to {task.target}: {reason}")
                        state.last_action_summary = f"You revised the target to {task.target}: \"{reason}\""
                        state.checkpoint_pending = True
                        self._record_controller_action(
                            action_type="revise_target",
                            action_step=action_step,
                            success=True,
                            action_params={
                                "new_target": task.target,
                                "reason": reason,
                            },
                            before_state=before_state,
                            after_state=self.memory_store._capture_current_state(),
                        )
                        continue

                    # Handle think (with next_action decision)
                    if function_name == "think":
                        self.action_executor.act(
                            action_step=action_step,
                            detected_elements=detected_elements,
                            page_info=page_info,
                            environment_state=environment_state,
                            current_iteration=self._current_iteration,
                        )
                        state.actions_since_progress += 1

                        think_reasoning = str(action_args.get("reasoning", "")).strip()
                        think_next_action = str(action_args.get("next_action", "continue")).strip().lower()
                        recommended_next_step = str(action_args.get("recommended_next_step", "")).strip()

                        if think_next_action in ("mark_progress", "done"):
                            reasoning = think_reasoning or "Completed"
                            recorded, maybe_completion = _record_progress(
                                reasoning,
                                count=1,
                                done=(think_next_action == "done"),
                                source="think",
                            )
                            if not recorded:
                                state.checkpoint_pending = False
                            if maybe_completion:
                                return maybe_completion
                        elif think_next_action == "stuck":
                            replacement_strategy = think_reasoning or "I'm stuck with my prior approach, so I'll try a different strategy."
                            state.checkpoint_pending = False
                            state.last_action_summary = f"Strategy switch (stuck): \"{replacement_strategy}\""
                            if recommended_next_step:
                                state.last_action_summary += f" | recommended_next_step={recommended_next_step}"
                            else:
                                state.last_action_summary += " | recommended_next_step=<none acknowledged>"
                            self.event_logger.system_info(f"↺ Strategy switched via stuck: {replacement_strategy}")
                        elif think_next_action == "continue":
                            state.last_action_summary = f"You thought: \"{think_reasoning}\""
                            if recommended_next_step:
                                state.last_action_summary += f" | recommended_next_step={recommended_next_step}"
                            else:
                                state.last_action_summary += " | recommended_next_step=<none acknowledged>"
                            state.checkpoint_pending = False
                        else:
                            state.checkpoint_pending = False
                        continue

                    # Handle assert/flag (non-browser actions)
                    if function_name in {"assert_condition", "flag"}:
                        self.action_executor.act(
                            action_step=action_step,
                            detected_elements=detected_elements,
                            page_info=page_info,
                            environment_state=environment_state,
                            current_iteration=self._current_iteration,
                        )
                        action_type = "assert" if function_name == "assert_condition" else "flag"
                        action_content = str(
                            action_args.get("condition") if function_name == "assert_condition" else action_args.get("message", "")
                        ).strip()
                        state.last_action_summary = f"You called {action_type}: \"{action_content}\""
                        state.actions_since_progress += 1
                        state.checkpoint_pending = True
                        continue

                    # Handle tab management actions (intercepted by controller)
                    if function_name == "switch_tab":
                        before_state = self.memory_store._capture_current_state()
                        action_success = False
                        action_error: Optional[str] = None
                        tab_id = str(action_args.get("tab_id", "")).strip()
                        if self.tab_manager:
                            try:
                                new_page = self.tab_manager.switch_to(tab_id)
                                self.action_executor.set_page(new_page)
                                title = ""
                                try:
                                    title = new_page.title()
                                except Exception:
                                    pass
                                state.last_action_summary = f"Switched to tab [{tab_id}]: \"{title}\""
                                state.browser_actions_since_progress += 1
                                state.suppress_mark_progress = False
                                state.checkpoint_pending = True
                                action_success = True
                                tools_used_since_progress.add(function_name.lower())
                            except ValueError as e:
                                state.last_action_summary = f"switch_tab FAILED: {e}"
                                action_error = str(e)
                        else:
                            state.last_action_summary = "switch_tab FAILED: Tab management not available"
                            action_error = "Tab management not available"
                        self._record_controller_action(
                            action_type=InteractionType.NAVIGATION.value,
                            action_step=action_step,
                            success=action_success,
                            error_message=action_error,
                            action_params={"operation": "switch_tab", "tab_id": tab_id},
                            before_state=before_state,
                            after_state=self.memory_store._capture_current_state(),
                        )

                        state.actions_since_progress += 1
                        continue

                    if function_name == "close_tab":
                        before_state = self.memory_store._capture_current_state()
                        action_success = False
                        action_error: Optional[str] = None
                        tab_id = str(action_args.get("tab_id", "")).strip()
                        if self.tab_manager:
                            try:
                                new_page = self.tab_manager.close_tab(tab_id)
                                self.action_executor.set_page(new_page)
                                active = self.tab_manager.get_active()
                                active_id = active.id if active else "?"
                                state.last_action_summary = f"Closed tab [{tab_id}]. Now on tab [{active_id}]"
                                state.browser_actions_since_progress += 1
                                state.suppress_mark_progress = False
                                state.checkpoint_pending = True
                                action_success = True
                                tools_used_since_progress.add(function_name.lower())
                            except ValueError as e:
                                state.last_action_summary = f"close_tab FAILED: {e}"
                                action_error = str(e)
                        else:
                            state.last_action_summary = "close_tab FAILED: Tab management not available"
                            action_error = "Tab management not available"
                        self._record_controller_action(
                            action_type=InteractionType.NAVIGATION.value,
                            action_step=action_step,
                            success=action_success,
                            error_message=action_error,
                            action_params={"operation": "close_tab", "tab_id": tab_id},
                            before_state=before_state,
                            after_state=self.memory_store._capture_current_state(),
                        )

                        state.actions_since_progress += 1
                        continue

                    if function_name == "open_tab":
                        before_state = self.memory_store._capture_current_state()
                        action_success = False
                        action_error: Optional[str] = None
                        url = str(action_args.get("url", "")).strip() or None
                        if self.tab_manager:
                            try:
                                new_page = self.tab_manager.open_tab(url)
                                self.action_executor.set_page(new_page)
                                active = self.tab_manager.get_active()
                                active_id = active.id if active else "?"
                                state.last_action_summary = f"Opened new tab [{active_id}]"
                                if url:
                                    state.last_action_summary += f" at {url}"
                                state.browser_actions_since_progress += 1
                                state.suppress_mark_progress = False
                                state.checkpoint_pending = True
                                action_success = True
                                tools_used_since_progress.add(function_name.lower())
                            except Exception as e:
                                state.last_action_summary = f"open_tab FAILED: {e}"
                                action_error = str(e)
                        else:
                            state.last_action_summary = "open_tab FAILED: Tab management not available"
                            action_error = "Tab management not available"
                        self._record_controller_action(
                            action_type=InteractionType.NAVIGATION.value,
                            action_step=action_step,
                            success=action_success,
                            error_message=action_error,
                            action_params={"operation": "open_tab", "url": url},
                            before_state=before_state,
                            after_state=self.memory_store._capture_current_state(),
                        )

                        state.actions_since_progress += 1
                        continue

                    if function_name == "dismiss_dialog":
                        before_state = self.memory_store._capture_current_state()
                        action_success = False
                        action_error: Optional[str] = None
                        accept = bool(action_args.get("accept", False))
                        input_text_raw = action_args.get("input_text")
                        input_text = str(input_text_raw).strip() if input_text_raw not in (None, "") else None
                        if self.tab_manager and self.tab_manager.pending_dialog:
                            self.tab_manager.dismiss_dialog(accept, input_text)
                            action_word = "accepted" if accept else "dismissed"
                            state.last_action_summary = f"Dialog {action_word}"
                            state.checkpoint_pending = True
                            action_success = True
                            tools_used_since_progress.add(function_name.lower())
                        else:
                            state.last_action_summary = "dismiss_dialog: No dialog pending"
                            state.checkpoint_pending = True
                            action_error = "No dialog pending"
                        self._record_controller_action(
                            action_type="dismiss_dialog",
                            action_step=action_step,
                            success=action_success,
                            error_message=action_error,
                            action_params={"accept": accept, "input_text": input_text},
                            before_state=before_state,
                            after_state=self.memory_store._capture_current_state(),
                        )

                        state.actions_since_progress += 1
                        continue

                    # Execute browser action
                    result = self.action_executor.act(
                        action_step=action_step,
                        detected_elements=detected_elements,
                        page_info=page_info,
                        environment_state=environment_state,
                        base_knowledge=self.base_knowledge,
                        current_iteration=self._current_iteration,
                    )

                    state.actions_since_progress += 1
                    result_str = "success" if result.success else "failed"
                    state.last_action_summary = self._build_action_summary(action_step, result_str)
                    state.checkpoint_pending = True

                    if result.success:
                        state.browser_actions_since_progress += 1
                        state.suppress_mark_progress = False
                        if function_name:
                            tools_used_since_progress.add(function_name.lower())


                    # Sync tab manager after browser actions (click may have opened a new tab)
                    if result.success and self.tab_manager:
                        active_tab = self.tab_manager.get_active()
                        if active_tab and self.browser.page is not active_tab.page:
                            self.action_executor.set_page(active_tab.page)

                    if not result.success:
                        try:
                            overlay_index = result.metadata.get("overlay_index") if result.metadata else None
                            if overlay_index is None:
                                overlay_index = action_args.get("overlay_index")
                            failed_action = FailedAction(
                                action=current_action,
                                overlay_index=overlay_index,
                                url=page_info.url if page_info else "",
                                page_title=page_info.title if page_info else None,
                                timestamp=time.time()
                            )
                            state.failed_elements.append(failed_action)
                        except Exception:
                            pass

            except Exception as e:
                return TaskResult(
                    success=False,
                    completion_status=TaskCompletionStatus.STUCK,
                    progress=task.progress,
                    target=task.target,
                    reasoning=f"Error: {str(e)}",
                )

        # Max actions reached
        if task.progress > 0:
            return TaskResult(
                success=False,
                completion_status=TaskCompletionStatus.PARTIAL,
                progress=task.progress,
                target=task.target,
                reasoning=f"Partial completion: {task.progress}/{task.target} (max actions reached)",
            )
        else:
            return TaskResult(
                success=False,
                completion_status=TaskCompletionStatus.STUCK,
                progress=task.progress,
                target=task.target,
                reasoning="Stuck: no progress made",
            )
