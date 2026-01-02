import json
import time
import threading
from typing import Optional, List, Dict, Any, Tuple, Set, TYPE_CHECKING, Callable
from enum import Enum
import re
import hashlib
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

from session_tracker import BrowserState
from .task_result import TaskResult
from agent.completion_contract import CompletionContract, EnvironmentState, CompletionEvaluation
from agent.reactive_goal_determiner import ReactiveGoalDeterminer
from agent.agent_context import AgentContext
from agent.sub_agent_controller import SubAgentController
from agent.sub_agent_result import SubAgentResult
from typing import Callable

# Type alias for user question callback (ask: command handler)
# Callback receives: question (str), context (dict) -> returns user's answer (str) or None to skip
UserQuestionCallback = Callable[[str, dict], Optional[str]]
from tab_management import TabDecisionEngine, TabAction
from ai_utils import (
    generate_model,
    ReasoningLevel,
    get_default_agent_model,
    get_default_agent_reasoning_level,
)
from pydantic import BaseModel, ConfigDict, Field

from .mini_goal_manager import MiniGoalManager, MiniGoalTrigger, MiniGoalMode, MiniGoalScriptContext

if TYPE_CHECKING:
    from .agent_controller import AgentController
    from vision_bot import BrowserVisionBot

_REQUIREMENT_KEYWORD_MAP = {
    "lede": ["lede", "introduction", "intro", "opening paragraph"],
    "infobox": ["infobox"],
    "milestones": ["milestone", "timeline", "chronology", "history"],
    "founding": ["founding", "founded", "established", "origin", "formation"],
    "leadership": ["leadership", "executive", "ceo", "chairman", "president", "founder"],
    "products": ["product", "service", "achievement", "mission", "program"],
    "controversies": ["controversy", "controversies", "criticism", "lawsuit", "scandal", "challenge"],
    "summary": ["summary", "summaries", "report", "overview"],
    "bullet_list": ["bullet", "list", "milestones", "timeline", "chronology"],
    "sources": ["source", "citation", "url", "link", "reference", "section title"],
    "revision": ["revision", "last updated", "last-edited", "last edited", "edit history"],
}

class _SubAgentPolicyYesNo(BaseModel):
    """Lightweight yes/no response for sub-agent policy decision"""
    model_config = ConfigDict(extra="forbid")
    needs_sub_agents: bool = Field(description="True if sub-agents should be used, False otherwise")


class _SubAgentPolicyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy_level: str
    policy_score: float
    rationale: str


class _SubAgentTaskPlan(BaseModel):
    name: str
    instruction: str
    suggested_url: Optional[str] = None
    expected_output: Optional[str] = None


class _ParallelWorkPlan(BaseModel):
    main_agent_focus: str
    sub_tasks: List[_SubAgentTaskPlan]
    integration_notes: Optional[str] = None


class _RetargetDecision(BaseModel):
    action: Optional[str]
    rationale: str


class SubAgentPolicyLevel(Enum):
    """Adaptive utilization policy levels for sub-agent usage."""
    SINGLE_THREADED = "single_threaded"
    PARALLELIZED = "parallelized"


"""
Agent Controller - Step 2: LLM-Based Completion + Reactive Goal Determination

Implements agentic mode with:
- Reactive loop: observe → check completion (LLM) → determine goal → act → repeat
- LLM-based completion evaluation (CompletionContract)
- EnvironmentState for full context
- Reactive goal determination (what to do now)
"""

class AgentController:
    """
    Basic reactive agent controller.
    
    Step 1 implementation: Minimal viable agent that:
    1. Observes browser state
    2. Checks simple completion criteria
    3. Generates actions using existing PlanGenerator
    4. Executes actions using existing ActionExecutor
    5. Repeats until done or max iterations
    """
    
    def __init__(
        self,
        bot,
        track_ineffective_actions: bool = True,
        base_knowledge: Optional[List[str]] = None,
        allow_partial_completion: bool = False,
        parallel_completion_and_action: bool = True,
        show_completion_reasoning_every_iteration: bool = False,
        strict_mode: bool = False,
        clarification_callback: Optional[Any] = None,
        max_clarification_rounds: int = 0,
        # User question callback for ask: command
        user_question_callback: Optional[UserQuestionCallback] = None,
        # Act function parameter configuration
        act_enable_target_context_guard: bool = True,
        act_enable_modifier: bool = True,
        act_enable_additional_context: bool = True,
        # Overlay inclusion in agent context
        include_overlays_in_agent_context: bool = True,
        # Interaction summarization
        interaction_summary_limit_completion: Optional[int] = None,
        interaction_summary_limit_action: Optional[int] = None,
    ):
        """
        Initialize agent controller.
        
        Args:
            bot: BrowserVisionBot instance to control
            track_ineffective_actions: If True, track and avoid repeating actions that didn't yield page changes.
                                       Default: True (recommended for better performance)
            base_knowledge: Optional list of knowledge rules/instructions that guide the agent's behavior.
                           Example: ["just press enter after you've typed a search term into a search field"]
            allow_partial_completion: If True, allow partial completion of tasks.
            parallel_completion_and_action: If True, run completion check and next action determination in parallel
                                            for faster feedback. Default: True.
                                            Note: Set to False primarily for debugging purposes, as sequential execution
                                            makes it easier to trace the execution flow and understand which LLM call
                                            is running at any given time.
            act_enable_target_context_guard: If True, allow the agent to use target_context_guard parameter in act() calls.
                                            This parameter enables contextual element filtering. Default: True.
            act_enable_modifier: If True, allow the agent to use modifier parameter in act() calls.
                                This parameter enables ordinal selection (e.g., "first", "second"). Default: True.
            act_enable_additional_context: If True, allow the agent to use additional_context parameter in act() calls.
                                          This parameter provides supplementary information for planning. Default: True.
            interaction_summary_limit_completion: Max interactions to feed into completion evaluation.
                                                  None means include all interactions. Default: None.
            interaction_summary_limit_action: Max interactions to feed into action determination.
                                              None means include all interactions. Default: None.
        """
        self.bot = bot
        # Access event logger from bot
        self.event_logger = getattr(bot, 'event_logger', None)
        if self.event_logger is None:
            from utils.event_logger import get_event_logger
            self.event_logger = get_event_logger()
        
        # Ensure event_logger is never None - create a dummy one if needed
        if self.event_logger is None:
            from utils.event_logger import EventLogger
            self.event_logger = EventLogger(debug_mode=True)
        self.max_iterations = 50
        self.iteration_delay = 0.5
        self.task_start_url: Optional[str] = None
        self.task_start_time: Optional[float] = None
        self.allow_non_clickable_clicks = True  # Allow clicking non-clickable elements (configurable)
        self.track_ineffective_actions = track_ineffective_actions  # Track actions that didn't yield page changes
        self.detect_ineffective_actions = track_ineffective_actions
        self.base_knowledge = base_knowledge or []  # Base knowledge rules that guide agent behavior
        self.failed_actions: List[str] = []  # Track actions that failed AND didn't yield any change
        self.ineffective_actions: List[str] = []  # Track actions that succeeded BUT didn't yield any change
        self._consecutive_page_changes: int = 0  # Track consecutive page state changes for phase-out
        self.extracted_data: Dict[str, Any] = {}  # Store extracted data (key: extraction prompt, value: result)
        self.sub_agent_results: List[SubAgentResult] = []  # Store completed sub-agent results
        self.orchestration_events: List[Dict[str, Any]] = []  # Track orchestration events for reporting
        self.sub_agent_policy_level: SubAgentPolicyLevel = SubAgentPolicyLevel.SINGLE_THREADED
        self.sub_agent_policy_rationale: str = "Default adaptive policy (single-threaded)."
        self._sub_agent_policy_override: Optional[SubAgentPolicyLevel] = None
        self._sub_agent_policy_score: float = 0.0
        self.sub_agent_spawn_count: int = 0
        self._spawns_this_iteration = 0
        self._current_iteration = 0
        self._parallel_plan_done: bool = False
        self._parallel_plan: Optional[_ParallelWorkPlan] = None
        self._main_agent_focus: Optional[str] = None
        self._integration_notes: Optional[str] = None
        self._extraction_failures: Dict[str, int] = {}
        self._completed_extractions: Set[str] = set()
        self._extraction_prompt_map: Dict[str, str] = {}
        self._retarget_attempts: Dict[str, int] = {}
        self._queued_action: Optional[str] = None
        self._queued_action_reason: Optional[str] = None
        self._task_tracker: Dict[str, Dict[str, Any]] = {}
        self.allow_partial_completion = allow_partial_completion
        self.parallel_completion_and_action = parallel_completion_and_action
        # Completion / evaluation behavior
        self.show_completion_reasoning_every_iteration = show_completion_reasoning_every_iteration
        self.strict_mode = strict_mode
        self.clarification_callback = clarification_callback
        self.max_clarification_rounds = max_clarification_rounds
        self.interaction_summary_limit_completion = interaction_summary_limit_completion
        self.interaction_summary_limit_action = interaction_summary_limit_action
        self._user_inputs: List[Dict[str, Any]] = []
        self._temp_user_inputs: List[Dict[str, Any]] = []  # Single-use suggestions
        self._requirement_flags: Dict[str, bool] = {}
        self._original_user_prompt: str = ""
        self._primary_output_tasks_initialized: bool = False
        self._last_action_summary: Optional[Dict[str, Any]] = None
        self._current_task_prompt: Optional[str] = None  # Rewritten prompt when stuck
        self._last_completion_reasoning: Optional[str] = None  # Store for stuck detection at start of next iteration
        self._last_completion_evaluation: Optional[CompletionEvaluation] = None
        self._last_screenshot_hash: Optional[str] = None  # Store screenshot hash for phase-out tracking

        # Store user question callback for ask: command
        self.user_question_callback = user_question_callback
        self._last_ask_iteration: int = -2  # Track last iteration where ask: was answered (to prevent consecutive asks)
        
        # Store act function parameter configuration
        # These flags control which parameters are passed to bot.act() during execution
        self.act_enable_target_context_guard = act_enable_target_context_guard
        self.act_enable_modifier = act_enable_modifier
        self.act_enable_additional_context = act_enable_additional_context
        
        # Store overlay inclusion configuration
        # Controls whether overlay element data is included in agent's context for action determination
        self.include_overlays_in_agent_context = include_overlays_in_agent_context

        self.agent_model_name: str = getattr(bot, "agent_model_name", get_default_agent_model())
        agent_reasoning = getattr(bot, "agent_reasoning_level", None)
        if agent_reasoning is None:
            agent_reasoning = ReasoningLevel.coerce(get_default_agent_reasoning_level())
        else:
            agent_reasoning = ReasoningLevel.coerce(agent_reasoning)
        self.agent_reasoning_level: ReasoningLevel = agent_reasoning
        
        # Initialize TabDecisionEngine if TabManager is available
        self.tab_decision_engine: Optional[TabDecisionEngine] = None
        if hasattr(bot, 'tab_manager') and bot.tab_manager:
            try:
                self.tab_decision_engine = TabDecisionEngine(
                    bot.tab_manager,
                    model_name=self.agent_model_name,
                    reasoning_level=self.agent_reasoning_level,
                )
            except Exception as e:
                print(f"⚠️ Failed to initialize TabDecisionEngine: {e}")
                self.tab_decision_engine = None
        
        # Phase 3: Sub-agent support
        self.agent_context: Optional[AgentContext] = None
        self.sub_agent_controller: Optional[SubAgentController] = None
        
        # Pause functionality: Allows pausing agent execution between actions
        self._paused = False
        self._pause_lock = threading.Lock()  # Thread-safe access to pause state
        self._pause_event = threading.Event()  # Event to block execution when paused
        self._pause_event.set()  # Initially not paused (event is set = not blocking)
        self._pause_message = "Paused"

        self.mini_goal_manager = MiniGoalManager(self.bot)
        self.mini_goal_stack: List[Dict[str, Any]] = []  # Stack of active mini goals

    def register_mini_goal(
        self,
        trigger: MiniGoalTrigger,
        mode: MiniGoalMode,
        handler: Optional[Callable[[MiniGoalScriptContext], None]] = None,
        instruction_override: Optional[str] = None
    ):
        """Register a mini goal trigger and handler"""
        self.mini_goal_manager.register_mini_goal(trigger, mode, handler, instruction_override)

    def _handle_mini_goal_trigger(self, entry: Dict[str, Any], action: Optional[str] = None, action_step: Optional[Any] = None) -> bool:
        """Process a triggered mini goal"""
        if len(self.mini_goal_stack) >= self.mini_goal_manager.recursion_limit:
            self.event_logger.system_warning(f"Mini goal recursion limit reached ({self.mini_goal_manager.recursion_limit})")
            return False

        # Add instruction if missing
        if "instruction" not in entry:
            entry["instruction"] = entry.get("instruction_override") or f"Interact with: {action}"
        
        self.mini_goal_stack.append(entry)
        self.event_logger.system_info(f"🎯 Mini Goal Active: {entry['instruction']}")

        if entry["mode"] == MiniGoalMode.SCRIPTED:
            try:
                self.mini_goal_manager.execute_scripted(entry, self, action_step, action)
                self.mini_goal_stack.pop()
                self.event_logger.system_info(f"✅ Scripted Mini Goal Complete")
                return True
            except Exception as e:
                self.mini_goal_stack.pop()
                self.event_logger.system_error(f"❌ Scripted Mini Goal Failed: {e}")
                return False
        
        # Autonomy mode remains on the stack and will be handled by the main loop
        return True

    def _get_current_prompt(self, base_prompt: str) -> str:
        """Returns the current task instruction, considering the mini-goal stack"""
        if self.mini_goal_stack:
            return self.mini_goal_stack[-1]["instruction"]
        return base_prompt

    def _is_nav_action(self, action: str) -> bool:
        """Identify if an action involves navigating away or returning back/forward"""
        if not action: return False
        act = action.lower().strip()
        nav_cmds = ["navigate:", "back", "forward", "open:"]
        return any(act.startswith(cmd) for cmd in nav_cmds)

    def pause(self, message: str = "Paused") -> None:
        """
        Pause the agent execution between actions.
        
        When paused, the agent will wait before executing the next action, allowing for:
        - Manual inspection of page state
        - Debugging action sequences
        - User intervention when needed
        - Verification of intermediate results
        
        The pause occurs between actions (not between iterations), providing fine-grained control.
        This means you can pause after a specific action completes and inspect the result.
        
        Args:
            message: Optional message to display when paused (default: "Paused")
        """
        with self._pause_lock:
            self._paused = True
            self._pause_message = message
            self._pause_event.clear()  # Clear event to block execution
        try:
            self.event_logger.system_info(f"⏸️  Agent paused: {message}")
        except Exception:
            pass
    
    def resume(self) -> None:
        """
        Resume the agent execution after a pause.
        
        Unblocks the agent to continue executing actions. If the agent is not paused,
        this method has no effect.
        """
        with self._pause_lock:
            was_paused = self._paused
            self._paused = False
            self._pause_event.set()  # Set event to unblock execution
        if was_paused:
            try:
                self.event_logger.system_info("▶️  Agent resumed")
            except Exception:
                pass
    
    def is_paused(self) -> bool:
        """
        Check if the agent is currently paused.
        
        Returns:
            True if the agent is paused, False otherwise
        """
        with self._pause_lock:
            return self._paused
    
    def _check_pause(self, action_description: str = None) -> None:
        """
        Internal method to check pause state and wait if paused.
        
        This is called before each action execution to respect pause state.
        The method blocks execution until resume() is called if the agent is paused.
        
        Args:
            action_description: Optional description of the action about to be executed
                               (used for display purposes)
        
        Why this approach:
        - Thread-safe: Uses locks and events for safe concurrent access
        - Non-blocking when not paused: Event is set by default, so no overhead when running
        - Granular control: Pauses between actions, not just iterations
        - User-friendly: Provides clear feedback about what action is being paused
        """
        # Check pause state (quick check without lock first for performance)
        if not self._paused:
            return
        
        # Get pause message with lock
        with self._pause_lock:
            if not self._paused:
                return  # Double-check after acquiring lock
            message = self._pause_message
            action_desc = action_description or "next action"
        
        # Display pause information
        print(f"\n⏸️  {message}")
        if action_desc:
            print(f"   Waiting before: {action_desc}")
        try:
            current_url = self.bot.page.url if self.bot.page else 'N/A'
            print(f"   URL: {current_url}")
        except Exception:
            pass
        
        # Wait until resume() is called (this blocks the execution thread)
        self._pause_event.wait()
    
    def run_execute_task(self, user_prompt: str, agent_context: Optional[AgentContext] = None) -> TaskResult:
        """
        Execute a task autonomously.
        
        Args:
            user_prompt: User's high-level request
            agent_context: Optional agent context (for sub-agents)
            
        Returns:
            TaskResult indicating success or failure
        """
        # Reset per-run state
        role = "Main agent"
        if agent_context and agent_context.parent_agent_id:
            role = f"Sub-agent helper (task: {agent_context.instruction})"
        try:
            self.event_logger.system_info(f"Active role: {role}")
        except Exception:
            pass

        self.sub_agent_results = []
        self.orchestration_events = []
        self.sub_agent_policy_level = SubAgentPolicyLevel.SINGLE_THREADED
        self.sub_agent_policy_rationale = "Default adaptive policy (single-threaded)."
        self._sub_agent_policy_override = None
        self._sub_agent_policy_score = 0.0
        self.sub_agent_spawn_count = 0
        self._spawns_this_iteration = 0
        self._current_iteration = 0
        self._task_tracker = {}
        self._original_user_prompt = user_prompt
        self._current_task_prompt = None  # Reset rewritten prompt
        self._primary_output_tasks_initialized = False
        self._initialize_requirement_flags(user_prompt)
        self._ensure_primary_output_tasks(user_prompt)
        # Reset page change counter for new task
        self._consecutive_page_changes = 0
        # Reset screenshot hash tracking
        self._last_screenshot_hash = None
        
        # Phase 3: Set agent context
        if agent_context:
            self.agent_context = agent_context
            # Initialize sub-agent controller if this is main agent
            if agent_context.parent_agent_id is None:
                if not self.sub_agent_controller:
                    self.sub_agent_controller = SubAgentController(
                        self.bot,
                        agent_context,
                        controller_factory=self._spawn_child_controller,
                        track_ineffective_actions=self.detect_ineffective_actions,
                        allow_partial_completion=self.allow_partial_completion
                    )
        else:
            # Create main agent context if not provided
            if self.bot.tab_manager:
                current_tab = self.bot.tab_manager.get_active_tab()
                if current_tab:
                    self.agent_context = AgentContext.create_main_agent(
                        tab_id=current_tab.tab_id,
                        instruction=user_prompt
                    )
                    if not self.sub_agent_controller:
                        self.sub_agent_controller = SubAgentController(
                            self.bot,
                            self.agent_context,
                            controller_factory=self._spawn_child_controller,
                            track_ineffective_actions=self.detect_ineffective_actions,
                            allow_partial_completion=self.allow_partial_completion
                        )
        
        agent_type = "Sub-agent" if (self.agent_context and self.agent_context.parent_agent_id) else "Main agent"
        self.event_logger.agent_start(user_prompt, agent_type=agent_type, max_iterations=self.max_iterations)

        self._log_event(
            "agent_start",
            agent_type=agent_type,
            prompt=user_prompt,
            tab_id=self.agent_context.tab_id if self.agent_context else None,
        )
        
        # Initialize task tracking
        # Ensure we're using the current page (may have been switched)
        try:
            self.task_start_url = self.bot.page.url
        except Exception:
            self.task_start_url = "unknown"
        self.task_start_time = time.time()
        
        # Start task timer
        self.bot.execution_timer.start_task()
        
        # Set base knowledge on goal monitor for goal evaluation
        if self.base_knowledge:
            self.bot.session_tracker.set_base_knowledge(self.base_knowledge)
        
        # Create completion contract (Step 2: LLM-based)
        completion_contract = CompletionContract(
            user_prompt,
            allow_partial_completion=self.allow_partial_completion,
            show_task_completion_reason=self.show_completion_reasoning_every_iteration,
            strict_mode=self.strict_mode,
            model_name=self.agent_model_name,
            reasoning_level=self.agent_reasoning_level,
            interaction_summary_limit=self.interaction_summary_limit_completion,
        )
        
        if not self.bot.started:
            self.event_logger.system_error("Bot not started. Call bot.start() first.")
            self._log_event("agent_complete", status="failed", reason="bot_not_started")
            # End task timer if it was started
            if self.bot.execution_timer.task_start_time is not None:
                self.bot.execution_timer.end_task()
                self.bot.execution_timer.log_summary(self.event_logger)
            result = TaskResult(
                success=False,
                confidence=1.0,
                reasoning="Bot not started",
                evidence=self._build_evidence()
            )
            self.event_logger.agent_complete(success=False, reasoning="Bot not started")
            return result
        
        if self.bot.page.url.startswith("about:blank"):
            self.event_logger.system_error("Page is on initial blank page.")
            self._log_event("agent_complete", status="failed", reason="blank_page")
            # End task timer if it was started
            if self.bot.execution_timer.task_start_time is not None:
                self.bot.execution_timer.end_task()
                self.bot.execution_timer.log_summary(self.event_logger)
            result = TaskResult(
                success=False,
                confidence=1.0,
                reasoning="Page is blank",
                evidence=self._build_evidence()
            )
            self.event_logger.agent_complete(success=False, reasoning="Page is blank")
            return result
        
        # Main reactive loop
        for iteration in range(self.max_iterations):
            # Start iteration timer
            self.bot.execution_timer.start_iteration()
            
            self._log_event("iteration_start", iteration=iteration + 1)
            self._current_iteration = iteration + 1
            self._spawns_this_iteration = 0

            # Drain any completed sub-agent results before continuing
            self._drain_sub_agent_results()
            
            # 1. Observe: Capture browser state (start with viewport)
            snapshot = self._capture_snapshot(full_page=False)
            self.event_logger.agent_iteration(iteration + 1, self.max_iterations, url=snapshot.url, title=snapshot.title)
            
            # --- MINI GOAL INTEGRATION: Observational Trigger ---
            if len(self.mini_goal_stack) < self.mini_goal_manager.recursion_limit:
                matching_goal = self.mini_goal_manager.find_matching_goal(visible_text=snapshot.visible_text)
                if matching_goal:
                    self._handle_mini_goal_trigger(matching_goal)
            # --------------------------------------------------

            # 2. Use active instruction from stack if available
            active_user_prompt = self._get_current_prompt(user_prompt)
            
            # Calculate screenshot hash for phase-out tracking (at start of iteration)
            if snapshot.screenshot:
                current_screenshot_hash = hashlib.md5(snapshot.screenshot).hexdigest()
                # Compare with previous iteration's screenshot hash
                if self._last_screenshot_hash and current_screenshot_hash != self._last_screenshot_hash:
                    # Screenshot changed from previous iteration
                    self._consecutive_page_changes += 1
                    # After 2 clear page state changes (screenshot hash differences), phase out failed/ineffective actions
                    if self._consecutive_page_changes >= 2:
                        if self.failed_actions or self.ineffective_actions:
                            total = len(self.failed_actions) + len(self.ineffective_actions)
                            try:
                                self.event_logger.system_info(f"{self._consecutive_page_changes} consecutive screenshot changes detected - phasing out {total} failed/ineffective action(s)")
                            except Exception:
                                pass
                            self.failed_actions.clear()
                            self.ineffective_actions.clear()
                            self._consecutive_page_changes = 0  # Reset counter after phase-out
                elif self._last_screenshot_hash:
                    # Screenshot didn't change, reset counter
                    self._consecutive_page_changes = 0
                # Update screenshot hash for next iteration
                self._last_screenshot_hash = current_screenshot_hash
            
            # 2. Prepare environment state for parallel LLM calls
            environment_state = EnvironmentState(
                browser_state=snapshot,
                interaction_history=self.bot.session_tracker.interaction_history,
                user_prompt=user_prompt,
                task_start_url=self.task_start_url,
                task_start_time=self.task_start_time,
                current_url=snapshot.url,
                page_title=snapshot.title,
                visible_text=snapshot.visible_text,
                url_history=self.bot.session_tracker.url_history.copy() if self.bot.session_tracker.url_history else [],
                url_pointer=getattr(self.bot.session_tracker, "url_pointer", None)
            )
            
            # 2.1. Prepare goal determiner with current prompt
            dynamic_prompt = self._build_current_task_prompt(user_prompt)
            
            # Combine base_knowledge with temporary user guidance
            combined_knowledge = list(self.base_knowledge) if self.base_knowledge else []
            if self._temp_user_inputs:
                for entry in self._temp_user_inputs:
                    response = entry.get("response", "").strip()
                    if response:
                        combined_knowledge.append(f"User instruction for this specific state: {response}")
            
            goal_determiner = ReactiveGoalDeterminer(
                dynamic_prompt,
                base_knowledge=combined_knowledge,
                model_name=self.agent_model_name,
                reasoning_level=self.agent_reasoning_level,
                interaction_summary_limit=self.interaction_summary_limit_action,
                include_overlays_in_agent_context=self.include_overlays_in_agent_context,
            )
            
            # 2.3. Check for queued action first (doesn't need LLM)
            current_action: Optional[str] = None
            
            if self._queued_action:
                current_action = self._queued_action
                print(f"🎯 Using queued action: {current_action}")
                if self._queued_action_reason:
                    print(f"   Reason: {self._queued_action_reason}")
                self._log_event(
                    "queued_action_consumed",
                    action=current_action,
                    reason=self._queued_action_reason,
                )
                self._queued_action = None
                self._queued_action_reason = None
            
            # 2.4. Run completion check and next action determination
            # (only if we don't have a queued action)
            policy_updated_in_parallel = False
            # Store completion reasoning and evaluation for stuck detector (for next iteration)
            latest_completion_reasoning: Optional[str] = None
            latest_evaluation: Optional[CompletionEvaluation] = None
            if current_action is None:
                if self.parallel_completion_and_action:
                    try:
                        self.event_logger.system_debug("Running completion check, next action determination, and subagent policy check in parallel...")
                    except Exception:
                        pass
                    
                    def run_completion_check():
                        """Run completion evaluation"""
                        return completion_contract.evaluate(
                            environment_state,
                            screenshot=snapshot.screenshot,
                            user_prompt=active_user_prompt
                        )
                    
                    def run_next_action_determination():
                        """Run next action determination"""
                        if self.track_ineffective_actions:
                            try:
                                # Filter out scroll actions from noise
                                failed_non_scroll = [a for a in self.failed_actions if not a.lower().startswith("scroll:")]
                                ineffective_non_scroll = [a for a in self.ineffective_actions if not a.lower().startswith("scroll:")]
                                if failed_non_scroll:
                                    self.event_logger.system_warning(f"Previously failed actions (failed + no change): {', '.join(failed_non_scroll)}")
                                if ineffective_non_scroll:
                                    self.event_logger.system_warning(f"Previously ineffective actions (succeeded but no change): {', '.join(ineffective_non_scroll)}")
                            except Exception:
                                pass
                        action, reasoning = goal_determiner.determine_next_action(
                            environment_state,
                            screenshot=snapshot.screenshot,
                            failed_actions=failed_non_scroll if self.track_ineffective_actions else [],
                            ineffective_actions=ineffective_non_scroll if self.track_ineffective_actions else []
                        )
                        # Store reasoning for next interaction
                        if reasoning:
                            self.bot.session_tracker.set_current_action_reasoning(reasoning)
                        return action
                    
                    def run_subagent_policy_check(completion_reasoning_ref):
                        """Run subagent policy check (will use completion_reasoning after completion check finishes)"""
                        # Wait for completion_reasoning to be available
                        completion_reasoning = completion_reasoning_ref[0] if completion_reasoning_ref else None
                        return self._compute_sub_agent_policy(
                            user_prompt=user_prompt,
                            completion_reasoning=completion_reasoning or ""
                        )
                    
                    # Execute all three in parallel
                    # Use a list to share completion_reasoning between threads
                    completion_reasoning_ref = [None]
                    
                    with ThreadPoolExecutor(max_workers=3) as executor:
                        completion_future = executor.submit(run_completion_check)
                        next_action_future = executor.submit(run_next_action_determination)
                        
                        # Wait for completion check first (we need to know if we're done)
                        is_complete, completion_reasoning, evaluation = completion_future.result()
                        completion_reasoning_ref[0] = completion_reasoning  # Share with subagent policy check
                        latest_completion_reasoning = completion_reasoning
                        latest_evaluation = evaluation
                        
                        # Start subagent policy check now that we have completion_reasoning
                        subagent_policy_future = executor.submit(run_subagent_policy_check, completion_reasoning_ref)
                        
                        if is_complete:
                            self.event_logger.completion_check(is_complete=True, reasoning=completion_reasoning, confidence=evaluation.confidence, evidence=evaluation.evidence)
                            
                            # --- MINI GOAL INTEGRATION: Pop stack if complete ---
                            if self.mini_goal_stack:
                                self.event_logger.system_info(f"🎯 Mini Goal Achieved: {active_user_prompt}")
                                self.mini_goal_stack.pop()
                                # Wait for subagent policy check (optional finish)
                                try: subagent_policy_future.result(timeout=1)
                                except Exception: pass
                                
                                time.sleep(self.iteration_delay)
                                continue # NEXT ITERATION of main loop
                            # ----------------------------------------------------
                            
                            evidence_dict: Dict[str, Any] = {}
                            if evaluation.evidence:
                                try:
                                    import json
                                    evidence_dict = json.loads(evaluation.evidence) if isinstance(evaluation.evidence, str) else evaluation.evidence
                                except Exception:
                                    evidence_dict = {"evidence": evaluation.evidence}
                            
                            evidence_dict = self._build_evidence(evidence_dict)
                            self._log_event(
                                "agent_complete",
                                status="achieved",
                                confidence=evaluation.confidence,
                                reasoning=completion_reasoning,
                            )
                            # End task timer and log summary
                            self.bot.execution_timer.end_task()
                            self.bot.execution_timer.log_summary(self.event_logger)
                            # Wait for subagent policy check to complete (for logging)
                            try:
                                subagent_policy_future.result(timeout=5)
                            except Exception:
                                pass  # Ignore timeout, task is complete anyway
                            # Next action result will be ignored (task is complete)
                            self.event_logger.agent_complete(success=True, reasoning=completion_reasoning, confidence=evaluation.confidence)
                            return TaskResult(
                                success=True,
                                confidence=evaluation.confidence,
                                reasoning=completion_reasoning,
                                evidence=evidence_dict
                            )
                        else:
                            self.event_logger.completion_check(is_complete=False, reasoning=completion_reasoning)
                            # Task not complete, so we need the next action - wait for it
                            self.event_logger.system_debug("Determining next action based on current viewport...")
                            current_action = next_action_future.result()
                            
                            # Wait for subagent policy check and update policy
                            try:
                                new_level, new_score, new_rationale = subagent_policy_future.result()
                                
                                # Check for changes before updating
                                level_changed = new_level != self.sub_agent_policy_level
                                score_changed = abs(new_score - self._sub_agent_policy_score) >= 0.1
                                rationale_changed = new_rationale != self.sub_agent_policy_rationale
                                
                                # Update policy
                                self.sub_agent_policy_level = new_level
                                self._sub_agent_policy_score = new_score
                                self.sub_agent_policy_rationale = new_rationale
                                
                                if level_changed or score_changed or rationale_changed:
                                    try:
                                        self.event_logger.sub_agent_policy(
                                            policy=self._policy_display_name(new_level),
                                            score=new_score,
                                            reason=new_rationale
                                        )
                                    except Exception:
                                        pass
                                    self._log_event(
                                        "sub_agent_policy_update",
                                        policy=new_level.value,
                                        score=new_score,
                                        rationale=new_rationale,
                                        override=self._sub_agent_policy_override.value if self._sub_agent_policy_override else None
                                    )
                                policy_updated_in_parallel = True
                            except Exception as e:
                                print(f"⚠️ Subagent policy check failed: {e}")
                                # Fallback: update policy sequentially
                                self._update_sub_agent_policy(
                                    user_prompt=user_prompt,
                                    completion_reasoning=completion_reasoning
                                )
                                policy_updated_in_parallel = True
                else:
                    # Sequential execution (parallel disabled)
                    print("🔄 Running completion check...")
                    latest_evaluation = completion_contract.evaluate(
                        environment_state,
                        screenshot=snapshot.screenshot,
                        user_prompt=active_user_prompt
                    )
                    latest_completion_reasoning = latest_evaluation.reasoning if latest_evaluation else None
                    is_complete = latest_evaluation.is_complete if latest_evaluation else False
                    
                    if is_complete:
                        self.event_logger.completion_check(
                            is_complete=True, 
                            reasoning=latest_completion_reasoning, 
                            confidence=latest_evaluation.confidence, 
                            evidence=latest_evaluation.evidence
                        )
                        
                        # --- MINI GOAL INTEGRATION: Pop stack if complete ---
                        if self.mini_goal_stack:
                            self.event_logger.system_info(f"🎯 Mini Goal Achieved: {active_user_prompt}")
                            self.mini_goal_stack.pop()
                            # Restart iteration to check main goal or next mini-goal
                            time.sleep(self.iteration_delay)
                            continue 
                        # --------------------------------------------------

                        current_action = "stop"
                    else:
                        if latest_evaluation:
                            self.event_logger.completion_check(is_complete=False, reasoning=latest_completion_reasoning)
                        
                        # Task not complete, determine next action
                        self.event_logger.system_debug("Determining next action based on current viewport...")
                        
                        failed_non_scroll = [a for a in self.failed_actions if not a.lower().startswith("scroll:")]
                        ineffective_non_scroll = [a for a in self.ineffective_actions if not a.lower().startswith("scroll:")]
                        
                        current_action, reasoning = goal_determiner.determine_next_action(
                            environment_state,
                            screenshot=snapshot.screenshot,
                            failed_actions=failed_non_scroll if self.track_ineffective_actions else [],
                            ineffective_actions=ineffective_non_scroll if self.track_ineffective_actions else []
                        )
                        # Store reasoning for next interaction
                        if reasoning:
                            self.bot.session_tracker.set_current_action_reasoning(reasoning)

            # --- MINI GOAL INTEGRATION: Action/Historical Trigger ---
            if current_action and not current_action.startswith("mini_goal:"):
                matching_goal = self.mini_goal_manager.find_matching_goal(action=current_action)
                if matching_goal:
                    # Don't trigger if this specific goal (by identity or instruction) is already active
                    is_already_active = False
                    if self.mini_goal_stack:
                        top = self.mini_goal_stack[-1]
                        # Use instruction_override or instruction for comparison
                        top_instr = top.get("instruction")
                        match_instr = matching_goal.get("instruction_override") or f"Interact with: {current_action}"
                        if top_instr == match_instr:
                            is_already_active = True
                    
                    if not is_already_active:
                        triggered = self._handle_mini_goal_trigger(matching_goal, action=current_action)
                        if triggered:
                            print(f"⚡ Action intercepted by Mini Goal: {current_action}")
                        # If autonomy mode, it stays on stack and we restart iteration with new prompt
                        # If scripted mode, it was executed and popped, we continue?
                        # Scripted mode counts as 1 iteration, so we should skip execution of current_action
                        if matching_goal["mode"] == MiniGoalMode.SCRIPTED:
                            time.sleep(self.iteration_delay)
                            continue
                        else:
                            # Autonomy mode - spawn sub-agent to handle the mini goal
                            self.event_logger.system_info(f"🤖 Starting Autonomy Mini Goal: {matching_goal.get('instruction_override', current_action)}")
                            try:
                                result = self.mini_goal_manager.execute_autonomous(matching_goal, self, current_action)
                                self.mini_goal_stack.pop()
                                self.event_logger.system_info(f"✅ Autonomy Mini Goal Complete: {result.success}")
                            except Exception as e:
                                self.mini_goal_stack.pop()
                                self.event_logger.system_error(f"❌ Autonomy Mini Goal Failed: {e}")
                            time.sleep(self.iteration_delay)
                            continue

            if current_action and current_action.startswith("mini_goal:"):
                instruction = current_action.split(":", 1)[1].strip()
                self._handle_mini_goal_trigger({"mode": MiniGoalMode.AUTONOMY, "instruction": instruction}, action=current_action)
                time.sleep(self.iteration_delay)
                continue
            # ----------------------------------------------------------
            
            # --- ASK COMMAND: Agent asks user for help ---
            if current_action and current_action.lower().startswith("ask:"):
                # Block consecutive ask: commands - must act on previous answer first
                if iteration <= self._last_ask_iteration + 1:
                    print("⚠️ Cannot ask again - must act on previous answer first")
                    self._log_event(
                        "ask_command_blocked",
                        reason="consecutive_ask_blocked",
                        iteration=iteration,
                    )
                    # Force agent to continue without asking - it has the answer in base_knowledge
                    current_action = None
                else:
                    question = current_action.split(":", 1)[1].strip()
                    handled = self._handle_ask_command(question, iteration, environment_state)
                    if handled:
                        # User provided answer, retry with new knowledge
                        time.sleep(self.iteration_delay)
                        continue
                    # User skipped or no callback - agent should try something else
            
            # Block navigation in autonomy mode
            if self.mini_goal_stack and self._is_nav_action(current_action):
                self.event_logger.system_warning(f"🚫 Iteration {iteration+1}: Navigation blocked by active Mini Goal.")
                current_action = None # Skip this action
            
            # Wait, if current_action is STOP and we have a stack, pop it
            if current_action == "stop" and self.mini_goal_stack:
                 self.event_logger.system_info(f"🎯 Mini Goal requested STOP, finishing: {active_user_prompt}")
                 self.mini_goal_stack.pop()
                 time.sleep(self.iteration_delay)
                 continue
            else:
                # We have a queued action, so we still need to check completion
                self.event_logger.system_debug("Running completion check (next action already queued)...")
                is_complete, completion_reasoning, evaluation = completion_contract.evaluate(
                    environment_state,
                    screenshot=snapshot.screenshot,
                    user_prompt=active_user_prompt
                )
                latest_completion_reasoning = completion_reasoning
                latest_evaluation = evaluation
                
                if is_complete:
                    self.event_logger.completion_check(is_complete=True, reasoning=completion_reasoning, confidence=evaluation.confidence, evidence=evaluation.evidence)
                    evidence_dict: Dict[str, Any] = {}
                    if evaluation.evidence:
                        try:
                            import json
                            evidence_dict = json.loads(evaluation.evidence) if isinstance(evaluation.evidence, str) else evaluation.evidence
                        except Exception:
                            evidence_dict = {"evidence": evaluation.evidence}
                    
                    evidence_dict = self._build_evidence(evidence_dict)
                    self._log_event(
                        "agent_complete",
                        status="achieved",
                        confidence=evaluation.confidence,
                        reasoning=completion_reasoning,
                    )
                    # End task timer and log summary
                    self.bot.execution_timer.end_task()
                    self.bot.execution_timer.log_summary(self.event_logger)
                    self.event_logger.agent_complete(success=True, reasoning=completion_reasoning, confidence=evaluation.confidence)
                    return TaskResult(
                        success=True,
                        confidence=evaluation.confidence,
                        reasoning=completion_reasoning,
                        evidence=evidence_dict
                    )
                else:
                    self.event_logger.completion_check(is_complete=False, reasoning=completion_reasoning)
            
            # Update adaptive sub-agent utilization policy (only if not already done in parallel path)
            # In parallel path, policy is updated after next_action_future.result()
            # In sequential/queued paths, update it here
            if not policy_updated_in_parallel:
                self._update_sub_agent_policy(
                    user_prompt=user_prompt,
                    completion_reasoning=completion_reasoning
                )
            
            if (
                self.sub_agent_policy_level == SubAgentPolicyLevel.PARALLELIZED
                and not self._parallel_plan_done
                and (not self.agent_context or self.agent_context.parent_agent_id is None)
            ):
                self._orchestrate_parallel_work(user_prompt)
            # 2.5. Check tab management decisions (Phase 2)
            if self.tab_decision_engine and self.bot.tab_manager:
                current_tab = self.bot.tab_manager.get_active_tab()
                if current_tab:
                    try:
                        # Make tab decision
                        tab_decision = self.tab_decision_engine.make_decision(
                            current_tab_id=current_tab.tab_id,
                            user_prompt=user_prompt,
                            current_action=current_action,  # Now we have the action
                            task_context={
                                "iteration": iteration + 1,
                                "max_iterations": self.max_iterations,
                                    "completion_reasoning": completion_reasoning,
                                    "sub_agent_policy": self.sub_agent_policy_level.value,
                                    "sub_agent_policy_score": round(self._sub_agent_policy_score, 2),
                                    "sub_agent_policy_reason": self.sub_agent_policy_rationale,
                                    "sub_agent_policy_override": self._sub_agent_policy_override.value if self._sub_agent_policy_override else None,
                                    "sub_agent_spawn_count": self.sub_agent_spawn_count,
                                    "sub_agent_spawns_this_iteration": self._spawns_this_iteration,
                            }
                        )
                        
                        # Execute decision if needed
                        if tab_decision.should_take_action:
                            executed = self._execute_tab_decision(tab_decision)
                            if executed:
                                # Tab switch/close happened, continue to next iteration
                                print("🔄 Tab action executed, continuing to next iteration...")
                                time.sleep(self.iteration_delay)
                                continue
                    except Exception as e:
                        print(f"⚠️ Error in tab decision making: {e}")
            
            # 3. Process next action (if we got one from parallel execution)
            if current_action is not None:
                
                # Safeguard: Filter out actions that exactly match failed or ineffective actions (only if tracking enabled)
                if (
                    self.track_ineffective_actions
                    and current_action
                    and current_action.lower().startswith("click:")
                    and (current_action in self.failed_actions or current_action in self.ineffective_actions)
                ):
                    print(f"⚠️ Generated action matches a failed/ineffective action, forcing None: {current_action}")
                    current_action = None
            
            # 4. Fallback if determiner fails
            if not current_action:
                current_action = self._determine_next_action(user_prompt, snapshot)
            
            if not current_action:
                print("⚠️ Cannot determine next action")
                if iteration == self.max_iterations - 1:
                    self._log_event(
                        "agent_complete",
                        status="failed",
                        reason="no_action",
                        iterations=iteration + 1,
                    )
                    # End task timer and log summary
                    self.bot.execution_timer.end_task()
                    self.bot.execution_timer.log_summary(self.event_logger)
                    return TaskResult(
                        success=False,
                        confidence=0.5,
                        reasoning="Could not determine next action",
                        evidence=self._build_evidence({"iterations": iteration + 1})
                    )
                time.sleep(self.iteration_delay)
                continue
            
            try:
                self.event_logger.system_info(f"Next action: {current_action}")
            except Exception:
                pass
            
            # 4. Check if this is an extraction command or detect extraction needs from natural language
            extraction_prompt = self._detect_extraction_need(current_action, user_prompt)
            
            if extraction_prompt:
                self._update_requirement_flags_from_text(extraction_prompt)
                subject = self._infer_extraction_subject(extraction_prompt)
                normalized_key_source = subject if subject else extraction_prompt
                normalized_key = self._normalize_extraction_prompt(normalized_key_source)
                canonical_prompt = self._build_comprehensive_extraction_prompt(
                    extraction_prompt,
                    subject,
                    self._original_user_prompt or user_prompt,
                    snapshot.url if snapshot else None
                )
                self._extraction_prompt_map[normalized_key] = canonical_prompt
                task_description = self._build_extraction_task_description(subject, extraction_prompt)
                self._register_task(normalized_key, task_description, "extraction")
                try:
                    self.event_logger.extraction_detected(canonical_prompt)
                except Exception:
                    pass
                already_completed = normalized_key in self._completed_extractions
                task_entry = self._task_tracker.get(normalized_key)
                if task_entry and not already_completed:
                    task_entry["status"] = "running"
                    task_entry["updated_at"] = time.time()
                
                if already_completed:
                    try:
                        self.event_logger.system_info(f"Extraction skipped (already completed): {canonical_prompt}")
                    except Exception:
                        pass
                    if task_entry:
                        task_entry["status"] = "completed"
                    self._activate_primary_output_task("Required Wikipedia fields already captured.")
                    time.sleep(self.iteration_delay)
                    continue
                if self._should_skip_extraction(normalized_key):
                    try:
                        self.event_logger.system_warning(f"Extraction skipped after repeated failures: {canonical_prompt}")
                    except Exception:
                        pass
                    time.sleep(self.iteration_delay)
                    continue
                
                try:
                    # Call extract() directly - simpler and more efficient
                    result = self.bot.extract(
                        prompt=canonical_prompt,
                        output_format="json",
                        scope="viewport"
                    )
                    if result.success:
                        self._record_extraction_success(normalized_key)
                        self._completed_extractions.add(normalized_key)
                        
                        self._extraction_prompt_map[normalized_key] = canonical_prompt
                        
                        # Store extracted data (the actual dict/text)
                        data = result.data
                        self.extracted_data[canonical_prompt] = data
                        
                        fields = []
                        if isinstance(data, dict):
                            fields = list(data.keys())
                        elif isinstance(data, str):
                            fields = ["text"]
                            
                        self._mark_task_completed(
                            normalized_key,
                            {
                                "type": "extraction",
                                "fields": fields,
                            }
                        )
                        self._activate_primary_output_task("Extraction completed successfully.")
                        try:
                            self.event_logger.extraction_success(canonical_prompt, result=result)
                        except Exception:
                            pass
                    else:
                        # Handle extraction failure
                        error_msg = result.error or result.message
                        try:
                            self.event_logger.system_warning(f"Extraction failed: {canonical_prompt} - {error_msg}")
                        except Exception:
                            pass
                        # We don't raise here, just let it continue to next iteration
                    
                    # Continue to next iteration (extraction is complete)
                    time.sleep(self.iteration_delay)
                    continue
                except Exception as e:
                    try:
                        self.event_logger.extraction_failure(canonical_prompt, error=str(e))
                    except Exception:
                        pass
                    import traceback
                    traceback.print_exc()
                    self._record_extraction_failure(normalized_key)
                    self._mark_task_failed(normalized_key, str(e))
                    
                    retarget_decision = self._retarget_after_extraction_failure(
                        prompt_key=normalized_key,
                        extraction_prompt=canonical_prompt,
                        snapshot=snapshot,
                        environment_state=environment_state,
                        user_prompt=user_prompt
                    )
                    if retarget_decision and retarget_decision.action:
                        action_lower = retarget_decision.action.lower()
                        if action_lower != "none":
                            self._queued_action = retarget_decision.action
                            self._queued_action_reason = retarget_decision.rationale
                            try:
                                self.event_logger.system_info(f"Retargeting after extraction failure → {retarget_decision.action}")
                                self.event_logger.system_info(f"   Rationale: {retarget_decision.rationale}")
                            except Exception:
                                pass
                            self._log_event(
                                "retarget_suggested",
                                action=retarget_decision.action,
                                rationale=retarget_decision.rationale,
                                prompt=canonical_prompt,
                                failures=self._extraction_failures.get(normalized_key, 0),
                            )
                        else:
                            print(f"ℹ️ Retarget assessment: no navigation change needed ({retarget_decision.rationale})")
                    
                    # Continue to next iteration anyway
                    time.sleep(self.iteration_delay)
                    continue
            
            # 4.1. Handle internal control commands (e.g., policy overrides)
            if self._handle_internal_command(current_action, user_prompt):
                time.sleep(self.iteration_delay)
                continue
            
            # 4. Filter out navigation commands - convert to click commands instead
            current_action = self._filter_navigation_commands(current_action)
            
            # 5. Parse action to extract parameters for act()
            act_params = self._parse_action_for_act_params(current_action, user_prompt)
            
            # Log parameters being passed to act()
            try:
                params = {
                    'goal_description': act_params['goal_description'],
                    'additional_context': act_params['additional_context'],
                    'target_context_guard': str(act_params['target_context_guard']) if act_params['target_context_guard'] else None,
                    'max_attempts': 5,
                    'allow_non_clickable_clicks': self.allow_non_clickable_clicks
                }
                self.event_logger.action_params(params)
            except Exception:
                pass
            
            # Capture page state BEFORE action
            state_before = self._get_page_state()
            
            # Check for pause before executing action
            # Why pause between actions: This allows inspection after each action completes,
            # providing fine-grained control over execution flow. Unlike pausing between
            # iterations, this lets you see the immediate result of each action before
            # the agent decides what to do next.
            self._check_pause(current_action)
            
            # 6. Use existing act() function - it handles goal creation, planning, execution
            # This leverages all the existing infrastructure (goal creation, plan generation, etc.)
            try:
                action_result = self.bot.act(
                    goal_description=act_params["goal_description"],
                    additional_context=act_params["additional_context"],
                    target_context_guard=act_params["target_context_guard"],
                    max_attempts=5,  # Allow 5 attempts per action for better reliability
                    allow_non_clickable_clicks=self.allow_non_clickable_clicks,  # Pass configurable setting
                    base_knowledge=self.base_knowledge,  # Pass base knowledge to act() for planning
                )
                
                # Extract success from ActionResult
                success = action_result.success
                
                # Capture page state AFTER action
                state_after = self._get_page_state()
                defer_input_consumer = getattr(self.bot, "consume_last_defer_input", None)
                if callable(defer_input_consumer):
                    defer_payload = defer_input_consumer()
                    if defer_payload:
                        self._handle_defer_input(defer_payload, current_action)
                
                # Check if action yielded any change
                page_changed = True
                if self.detect_ineffective_actions:
                    page_changed = self._page_state_changed(state_before, state_after)
                lower_action = current_action.lower()
                is_click_action = lower_action.startswith("click:")
                is_scroll_action = lower_action.startswith("scroll:")
                
                
                # Only track ineffective actions if the feature is enabled
                if self.track_ineffective_actions:
                    if success:
                        if page_changed:
                            # Successful command that changed the page - clear both failed and ineffective actions
                            if self.failed_actions or self.ineffective_actions:
                                total = len(self.failed_actions) + len(self.ineffective_actions)
                                print(f"   ✅ Command succeeded and changed page - clearing {total} ineffective action(s) from memory")
                                self.failed_actions.clear()
                                self.ineffective_actions.clear()
                                # Reset counter since we cleared the lists
                                self._consecutive_page_changes = 0
                        else:
                            # Successful command but no page change - add to ineffective actions
                            if is_click_action and self.detect_ineffective_actions:
                                try:
                                    self.event_logger.action_state_change(
                                        f"Action succeeded but did not yield any change: {current_action}",
                                        url_before=state_before['url'],
                                        url_after=state_after['url'],
                                        dom_changed=False
                                    )
                                except Exception:
                                    pass
                                # Add to ineffective actions list (nudge to try something different)
                                if current_action not in self.ineffective_actions:
                                    self.ineffective_actions.append(current_action)
                                    try:
                                        self.event_logger.system_info("Added to ineffective actions list (will try different approach in future iterations)")
                                    except Exception:
                                        pass
                            else:
                                if self.detect_ineffective_actions:
                                    try:
                                        self.event_logger.system_info(f"Non-click action succeeded without visible change: {current_action}")
                                    except Exception:
                                        pass
                    else:
                        # Failed command - check if it yielded any change
                        if not page_changed:
                            try:
                                self.event_logger.system_warning(f"Action failed and did not yield any change: {current_action}")
                            except Exception:
                                pass
                            try:
                                self.event_logger.action_state_change(
                                    f"Action failed and did not yield any change: {current_action}",
                                    url_before=state_before['url'],
                                    url_after=state_after['url'],
                                    dom_changed=False
                                )
                            except Exception:
                                pass
                            # Add to failed actions list (avoid trying this again)
                            if not is_scroll_action and current_action not in self.failed_actions:
                                self.failed_actions.append(current_action)
                                try:
                                    self.event_logger.system_info("Added to failed actions list (will avoid in future iterations)")
                                except Exception:
                                    pass
                        else:
                            # Action failed but page changed - still add to failed actions since the action itself failed
                            try:
                                self.event_logger.system_warning(f"Action failed (page changed but action execution failed): {current_action}")
                            except Exception:
                                pass
                            if current_action not in self.failed_actions:
                                self.failed_actions.append(current_action)
                                try:
                                    self.event_logger.system_info("Added to failed actions list (will avoid in future iterations)")
                                except Exception:
                                    pass
                
                self._record_action_outcome(
                    current_action,
                    success,
                    state_before,
                    state_after,
                    page_changed,
                )
                
                if success:
                    # Only show in debug mode
                    if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
                        print(f"✅ Action completed: {current_action}")
                    self._log_event(
                        "action_completed",
                        action=current_action,
                        success=True,
                        url_after=state_after["url"],
                    )
                else:
                    try:
                        self.event_logger.action_failure(f"Action failed: {current_action}")
                    except Exception:
                        pass
                    self._log_event(
                        "action_completed",
                        action=current_action,
                        success=False,
                        url_after=state_after["url"],
                    )
                    # Continue to next iteration (will try again with fresh state)
                    
            except Exception as e:
                print(f"⚠️ Error executing action: {e}")
                import traceback
                traceback.print_exc()
            
            # Update stuck detector with this iteration's results
            # Capture fresh environment state after action execution
            snapshot_after = self._capture_snapshot(full_page=False)
            environment_state_after = EnvironmentState(
                browser_state=snapshot_after,
                interaction_history=self.bot.session_tracker.interaction_history,
                user_prompt=user_prompt,
                task_start_url=self.task_start_url,
                task_start_time=self.task_start_time,
                current_url=snapshot_after.url,
                page_title=snapshot_after.title,
                visible_text=snapshot_after.visible_text,
                url_history=self.bot.session_tracker.url_history.copy() if self.bot.session_tracker.url_history else [],
                url_pointer=getattr(self.bot.session_tracker, "url_pointer", None)
            )
            # End iteration timer
            self.bot.execution_timer.end_iteration()
            
            # Small delay between iterations
            time.sleep(self.iteration_delay)
        
        # Drain any remaining sub-agent results before reporting failure
        self._drain_sub_agent_results()
        self._log_event(
            "agent_complete",
            status="failed",
            reason="max_iterations",
            iterations=self.max_iterations,
        )
        # End task timer and log summary
        self.bot.execution_timer.end_task()
        self.bot.execution_timer.log_summary()
        # Max iterations reached
        reasoning = f"Max iterations ({self.max_iterations}) reached without completion"
        self.event_logger.agent_complete(success=False, reasoning=reasoning)
        return TaskResult(
            success=False,
            confidence=0.5,
            reasoning=reasoning,
            evidence=self._build_evidence({"max_iterations": self.max_iterations})
        )
    
    def _capture_snapshot(self, full_page: bool = False) -> BrowserState:
        """
        Capture current browser state snapshot.
        
        Args:
            full_page: If True, capture full page screenshot (for exploration mode)
                      If False, capture viewport only (normal mode)
        """
        snapshot = self.bot.session_tracker._capture_current_state()
        
        # Always capture screenshot - agent needs it to see the page
        try:
            if full_page:
                snapshot.screenshot = self.bot.page.screenshot(full_page=True)
                print("📸 Using full-page screenshot for exploration mode")
            else:
                # Capture viewport screenshot (agent needs this to see what's visible)
                snapshot.screenshot = self.bot.page.screenshot(full_page=False)
        except Exception as e:
            print(f"⚠️ Failed to capture screenshot: {e}")
            snapshot.screenshot = None
        
        # Compute screenshot hash for change detection
        screenshot_data = getattr(snapshot, "screenshot", None)
        screenshot_hash = None
        if screenshot_data:
            try:
                screenshot_hash = hashlib.md5(screenshot_data).hexdigest()
            except Exception:
                screenshot_hash = None
        setattr(snapshot, "screenshot_hash", screenshot_hash)
        if hasattr(self.bot, "page"):
            try:
                setattr(self.bot.page, "_last_screenshot_hash", screenshot_hash)
            except Exception:
                pass
        
        return snapshot
    
    def _determine_next_action(
        self, 
        user_prompt: str, 
        snapshot: BrowserState,
        completion_evaluation: Optional[CompletionEvaluation] = None
    ) -> Optional[str]:
        """
        Determine what action needs to be done next.
        
        Step 2: Enhanced with state awareness - can use completion evaluation hints.
        Later steps will do full LLM-based reactive goal determination.
        
        Args:
            user_prompt: Original user request
            snapshot: Current browser state (viewport snapshot)
            completion_evaluation: Optional completion evaluation with remaining_steps hints
            
        Returns:
            Specific action description (e.g., "type: John Doe in name field")
            or None if cannot determine
        """
        # Step 2: Use completion evaluation hints if available
        if completion_evaluation and completion_evaluation.remaining_steps:
            # Use the first remaining step as the next action
            next_step = completion_evaluation.remaining_steps[0]
            print(f"🎯 Using suggested next step: {next_step}")
            
            # Special handling: if the step suggests scrolling, prioritize it
            # This ensures we scroll when elements aren't visible
            if next_step.lower().startswith("scroll:"):
                print("📍 Scrolling to reveal more content")
                return next_step
            
            return next_step
        
        # Fallback: use prompt directly (Step 2 still basic for goal determination)
        return user_prompt
    
    def _url_matches_target(self, url: Optional[str], target: str) -> bool:
        if not url or not target:
            return False
        url_lower = url.lower()
        target_lower = target.lower().strip()
        if not target_lower:
            return False

        if target_lower in url_lower:
            return True

        try:
            parsed = urlparse(url_lower)
            domain = parsed.netloc
            if domain and target_lower in domain:
                return True
            path = parsed.path
            if path and target_lower in path:
                return True
        except Exception:
            pass

        return False

    def _history_steps_to_target(self, history: List[str], pointer: int, target: str, direction: str) -> Optional[int]:
        if not history:
            return None

        if direction == "back":
            indices = range(pointer - 1, -1, -1)
        else:
            indices = range(pointer + 1, len(history))

        steps = 0
        for idx in indices:
            steps += 1
            if self._url_matches_target(history[idx], target):
                return steps
        return None

    def _filter_navigation_commands(self, action_command: str) -> str:
        """
        Filter navigate commands so they are only used when no simpler navigation path exists.
        
        Args:
            action_command: The action command to filter
            
        Returns:
            Potentially transformed action command
        """
        if action_command.lower().startswith("navigate:"):
            print(f"ℹ️ Navigation command preserved: {action_command}")
            return action_command
        
        return action_command
    
    def _handle_internal_command(self, current_action: Optional[str], user_prompt: str) -> bool:
        """
        Handle controller-level internal commands that should not be forwarded to act().
        
        Returns True if the command was handled (iteration should continue), False otherwise.
        """
        if not current_action:
            return False
        
        command = current_action.strip()
        command_lower = command.lower()
        
        if command_lower.startswith("subagents:"):
            directive = command.split(":", 1)[1].strip() if ":" in command else ""
            if not directive:
                print("⚠️ Sub-agent override command provided without a directive. "
                      "Use one of: single, parallel, reset.")
                self._log_event(
                    "sub_agent_policy_override_requested",
                    directive=None,
                    recognized=False,
                    source="command",
                    user_prompt_excerpt=user_prompt[:120]
                )
                return True
            
            handled = self._apply_sub_agent_override(directive)
            self._log_event(
                "sub_agent_policy_override_requested",
                directive=directive,
                recognized=handled,
                source="command",
                user_prompt_excerpt=user_prompt[:120]
            )
            if handled:
                print("🛠️ Sub-agent utilization override updated.")
            else:
                print(f"⚠️ Unknown sub-agent override directive: '{directive}'. "
                      "Valid options: single, parallel, aggressive, max, reset.")
            return True
        
        return False
    
    @staticmethod
    def _policy_display_name(level: SubAgentPolicyLevel) -> str:
        """Return human-readable label for a policy level."""
        return level.value.replace("_", " ").title()
    
    def _apply_sub_agent_override(self, directive: str) -> bool:
        """Apply a user-specified override for the sub-agent utilization policy."""
        directive_normalized = directive.strip().lower()
        reset_keywords = {"reset", "clear", "default", "auto", "adaptive"}
        override_map = {
            "single": SubAgentPolicyLevel.SINGLE_THREADED,
            "single-threaded": SubAgentPolicyLevel.SINGLE_THREADED,
            "solo": SubAgentPolicyLevel.SINGLE_THREADED,
            "off": SubAgentPolicyLevel.SINGLE_THREADED,
            "conservative": SubAgentPolicyLevel.SINGLE_THREADED,
            "parallel": SubAgentPolicyLevel.PARALLELIZED,
            "parallelized": SubAgentPolicyLevel.PARALLELIZED,
            "aggressive": SubAgentPolicyLevel.PARALLELIZED,
            "max": SubAgentPolicyLevel.PARALLELIZED,
        }
        score_map = {
            SubAgentPolicyLevel.SINGLE_THREADED: 0.0,
            SubAgentPolicyLevel.PARALLELIZED: 1.0,
        }
        
        if directive_normalized in reset_keywords:
            was_overridden = self._sub_agent_policy_override is not None
            self._sub_agent_policy_override = None
            self.sub_agent_policy_level = SubAgentPolicyLevel.SINGLE_THREADED
            self._sub_agent_policy_score = 0.0
            self.sub_agent_policy_rationale = "Override cleared; returning to adaptive policy (single-threaded baseline)."
            if was_overridden:
                print("🔄 Sub-agent utilization override cleared. Adaptive policy re-enabled.")
            else:
                print("ℹ️ Sub-agent utilization is already using adaptive policy.")
            return True
        
        if directive_normalized in override_map:
            level = override_map[directive_normalized]
            self._sub_agent_policy_override = level
            self.sub_agent_policy_level = level
            self._sub_agent_policy_score = score_map[level]
            self.sub_agent_policy_rationale = (
                f"Override active: forced to {self._policy_display_name(level)}."
            )
            print(f"✅ Sub-agent utilization override -> {self._policy_display_name(level)}")
            return True
        
        return False
    
    def _detect_extraction_need(self, current_action: Optional[str], user_prompt: str) -> Optional[str]:
        """
        Detect if extraction is needed from current action ONLY.
        
        We should only extract if the current action explicitly indicates extraction.
        Do NOT extract based on user prompt if the action is something else (like click).
        
        Args:
            current_action: The current action command (e.g., "extract: product price")
            user_prompt: The original user prompt (not used for detection, only for context)
        
        Returns:
            Extraction prompt if extraction is needed, None otherwise
        """
        if not current_action:
            return None
        
        # Check for explicit "extract:" command
        if current_action.startswith("extract:"):
            return current_action.replace("extract:", "").strip()
        
        # Check if current action indicates extraction (even if not explicitly "extract:")
        action_lower = current_action.lower()
        extraction_keywords = [
            "extract", "get", "find", "note", "collect", "gather", "retrieve", "pull", "fetch",
            "list", "show", "display", "return", "output", "print", "read", "scan", "capture",
            "obtain", "acquire", "pick up", "take", "grab", "pull out", "pull up", "bring up",
            "present", "report", "summarize", "detail", "enumerate", "itemize", "catalog",
            "record", "document", "save", "export", "download", "copy", "quote", "cite"
        ]
        
        # Only check if the action itself contains extraction keywords
        # Don't extract if the action is clearly something else (like "click", "type", "scroll")
        action_type_keywords = [
            "click",
            "type",
            "press",
            "scroll",
            "select",
            "navigate",
            "upload",
            "back",
            "forward",
            "form",
            "defer",
            "datetime",
            "subagents"
        ]
        
        # If action starts with a non-extraction action type, don't extract
        for action_type in action_type_keywords:
            if action_lower.startswith(f"{action_type}:"):
                return None  # This is clearly a different action, not extraction
        
        # If action contains extraction keywords, extract the description
        for keyword in extraction_keywords:
            if keyword in action_lower:
                # Try to extract what needs to be extracted
                # Pattern: "extract/get/find: <description>"
                pattern = rf"{keyword}:\s*(.+?)(?:\s|$)"
                match = re.search(pattern, action_lower)
                if match:
                    return match.group(1).strip()
        
        # Don't fall back to checking user prompt - only use current_action
        return None

    @staticmethod
    def _normalize_extraction_prompt(prompt: str) -> str:
        """
        Normalize extraction prompt text so equivalent requests share the same key.
        """
        return " ".join(prompt.lower().strip().split())

    @staticmethod
    def _normalize_task_name(name: str) -> str:
        return " ".join(name.lower().strip().split())

    def _initialize_requirement_flags(self, user_prompt: str) -> None:
        """
        Seed requirement flags from the user's original prompt.
        """
        self._requirement_flags = {}
        lowered = user_prompt.lower()
        for key, keywords in _REQUIREMENT_KEYWORD_MAP.items():
            self._requirement_flags[key] = any(keyword in lowered for keyword in keywords)

    def _update_requirement_flags_from_text(self, text: str) -> None:
        """
        Strengthen requirement flags if new instructions mention additional deliverables.
        """
        if not text:
            return
        lowered = text.lower()
        for key, keywords in _REQUIREMENT_KEYWORD_MAP.items():
            if self._requirement_flags.get(key):
                continue
            if any(keyword in lowered for keyword in keywords):
                self._requirement_flags[key] = True

    def _infer_extraction_subject(self, extraction_prompt: str) -> Optional[str]:
        """
        Attempt to infer which subject/entity the extraction prompt targets.
        """
        if not extraction_prompt:
            return None
        match = re.search(r"from the ([^.,\\n]+?)(?: wikipedia| article| page| site)", extraction_prompt, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"for (.+?)(?: page| article| wikipedia)", extraction_prompt, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _build_extraction_task_description(self, subject: Optional[str], extraction_prompt: str) -> str:
        """
        Produce a concise description for the task tracker.
        """
        subject_label = subject or "the target subject"
        sanitized_prompt = extraction_prompt.strip().replace("\n", " ")
        return f"Extract the data requested for {subject_label}: {sanitized_prompt}"

    def _build_comprehensive_extraction_prompt(
        self,
        extraction_prompt: str,
        subject: Optional[str],
        user_prompt: str,
        current_url: Optional[str] = None
    ) -> str:
        """
        Wrap the extraction prompt with light guidance while keeping scope tight.
        """
        scope_note = f"Focus on the subject '{subject}'." if subject else "Focus on the primary subject of this page."
        lines = [
            "Extract exactly the information described below.",
            "Use the entire page content (not just the visible viewport) and avoid adding extra categories.",
            scope_note,
            "Return JSON that mirrors the requested fields only.",
        ]
        if current_url:
            lines.append(f"Current page URL: {current_url}")
        lines.extend([
            "",
            "Requested extraction:",
            extraction_prompt.strip(),
            "",
            "Reference (original user prompt):",
            user_prompt.strip()
        ])
        return "\n".join(lines)

    def _extract_requested_subjects(self, user_prompt: str) -> List[str]:
        """
        Heuristic extraction of distinct subjects mentioned in the user prompt.
        """
        lower = user_prompt.lower()
        alias_pairs = [
            ("elon musk", "Elon Musk"),
            ("space x", "SpaceX"),
            ("spacex", "SpaceX"),
            ("tesla stock", "Tesla Stock"),
            ("tesla, inc.", "Tesla, Inc."),
            ("tesla inc.", "Tesla, Inc."),
            ("tesla inc", "Tesla, Inc."),
            ("tesla", "Tesla, Inc."),
            ("yahoo finance", "Yahoo Finance"),
            ("google docs", "Google Docs"),
        ]

        found: List[Tuple[int, str]] = []
        for alias, canonical in alias_pairs:
            for match in re.finditer(re.escape(alias), lower):
                found.append((match.start(), canonical))

        found.sort(key=lambda item: item[0])
        subjects: List[str] = []
        for _, canonical in found:
            if canonical and canonical not in subjects:
                subjects.append(canonical)
        return subjects

    def _build_manual_parallel_plan(
        self,
        subjects: List[str],
        user_prompt: str
    ) -> Optional[_ParallelWorkPlan]:
        """
        Construct a deterministic parallel plan that mirrors the requested sources.
        """
        if not subjects:
            return None

        lower_prompt = user_prompt.lower()
        sub_tasks: List[_SubAgentTaskPlan] = []
        for subject in subjects:
            subject_lower = subject.lower()
            if subject_lower == "google docs":
                # Leave Google Docs work to the main agent.
                continue

            if subject_lower == "yahoo finance":
                sub_tasks.append(
                    _SubAgentTaskPlan(
                        name="Tesla stock snapshot",
                        instruction=(
                            "Navigate to https://finance.yahoo.com/quote/TSLA and capture the Tesla stock "
                            "performance metrics the user requested (price, change, market summary, key stats). "
                            "Stay focused on the Yahoo Finance data only."
                        ),
                        suggested_url="https://finance.yahoo.com/quote/TSLA",
                        expected_output="Structured Yahoo Finance data for Tesla stock."
                    )
                )
                continue

            slug = re.sub(r"[^\w]+", "_", subject).strip("_")
            suggested_url = None
            if "wikipedia" in lower_prompt:
                suggested_url = f"https://en.wikipedia.org/wiki/{slug}"

            sub_tasks.append(
                _SubAgentTaskPlan(
                    name=f"Wikipedia – {subject}",
                    instruction=(
                        f"Open the Wikipedia page for {subject} and gather only the information requested in the user prompt. "
                        "Do not add unrelated details."
                    ),
                    suggested_url=suggested_url,
                    expected_output=f"Structured Wikipedia data for {subject}."
                )
            )

        if not sub_tasks:
            return None

        main_focus = "Integrate the sub-agent outputs and compose the final write-up using Google Docs."
        integration_notes = "Combine the extracted data from each sub-task to fulfill the original request."
        return _ParallelWorkPlan(
            main_agent_focus=main_focus,
            sub_tasks=sub_tasks,
            integration_notes=integration_notes
        )

    def _handle_defer_input(self, payload: Dict[str, Any], action_command: str) -> None:
        """
        Record user-provided input collected via a defer-input command.
        """
        if not payload:
            return
        response = payload.get("response")
        if response is None:
            return
        prompt = payload.get("prompt", "")
        timestamp = payload.get("timestamp", time.time())
        entry = {
            "prompt": prompt,
            "response": response,
            "timestamp": timestamp,
            "action": action_command,
            "page_url": payload.get("page_url"),
            "page_title": payload.get("page_title"),
            "temporary": True,  # Mark as single-use suggestion
        }
        # Store defer inputs as temporary (single-use) suggestions
        self._temp_user_inputs.append(entry)
        print(f"📝 Captured user input from defer: {response}")
        self._log_event(
            "defer_input_received",
            prompt=prompt,
            response=response,
            action=action_command,
            timestamp=timestamp,
            page_url=entry.get("page_url"),
            page_title=entry.get("page_title"),
        )

    def _handle_ask_command(
        self,
        question: str,
        iteration: int,
        environment_state: "EnvironmentState"
    ) -> bool:
        """
        Handle the ask: command - agent asking user for help/clarification.
        
        Args:
            question: The question the agent wants to ask
            iteration: Current iteration number
            environment_state: Current environment state
            
        Returns:
            True if user provided an answer (added to base_knowledge), False otherwise
        """
        if not self.user_question_callback:
            print("⚠️ Agent wants to ask a question but no callback configured")
            self._log_event(
                "ask_command_no_callback",
                question=question,
                iteration=iteration,
            )
            return False
        
        # Build context for the callback
        context = {
            "iteration": iteration,
            "current_url": environment_state.current_url,
            "page_title": environment_state.page_title,
        }

        try:
            # Disable page blocking while asking user question
            if hasattr(self.bot, '_thinking_border_manager'):
                self.bot._thinking_border_manager.disable_blocking()

            try:
                answer = self.user_question_callback(question, context)
            finally:
                # Re-enable page blocking after user responds
                if hasattr(self.bot, '_thinking_border_manager'):
                    self.bot._thinking_border_manager.enable_blocking()
            
            if answer:
                # Add user guidance as temporary context for the next action only
                # This provides guidance for the immediate next command without persisting
                temp_guidance = {
                    "prompt": question,
                    "response": answer,
                    "timestamp": time.time(),
                    "action": "ask_response",
                    "temporary": True
                }
                self._temp_user_inputs.append(temp_guidance)
                # Track that we just got an answer (to block consecutive asks)
                self._last_ask_iteration = iteration
                print(f"📝 User guidance added for next command: {answer}")
                self._log_event(
                    "ask_command_answered",
                    question=question,
                    answer=answer,
                    iteration=iteration,
                )
                return True
            else:
                print("⏭️ User skipped the question")
                self._log_event(
                    "ask_command_skipped",
                    question=question,
                    iteration=iteration,
                )
                return False
                
        except Exception as e:
            print(f"⚠️ Error in ask callback: {e}")
            self._log_event(
                "ask_command_error",
                question=question,
                error=str(e),
                iteration=iteration,
            )
            return False

    def _ensure_primary_output_tasks(self, user_prompt: str) -> None:
        """
        Register downstream deliverable tasks (e.g., Google Docs report) so the determiner
        focuses on them once prerequisite extractions are satisfied.
        """
        if self._primary_output_tasks_initialized:
            return

        lowered = (user_prompt or "").lower()
        tasks_added = False

        if any(keyword in lowered for keyword in ["google doc", "google docs", "google document"]):
            self._register_task(
                "google_docs_report",
                "Create a Google Docs report using the extracted Wikipedia basics (open docs.google.com, start a document, insert and organize the captured data).",
                "document_output",
            )
            entry = self._task_tracker.get("google_docs_report")
            if entry:
                details = entry.setdefault("details", {})
                details["requires_extractions"] = True
                details["ready"] = False
            tasks_added = True

        # Prevent redundant initialization checks even if no tasks were added.
        if tasks_added or True:
            self._primary_output_tasks_initialized = True

    def _activate_primary_output_task(self, reason: Optional[str] = None) -> None:
        """
        Flag the Google Docs output task as ready so the goal determiner pivots away from extraction.
        """
        entry = self._task_tracker.get("google_docs_report")
        if not entry:
            return
        details = entry.setdefault("details", {})
        already_ready = details.get("ready", False)
        details["ready"] = True
        if reason:
            details["ready_reason"] = reason
        entry["updated_at"] = time.time()
        if entry["status"] == "pending" and not already_ready:
            print("📝 Google Docs task unlocked: extraction prerequisites satisfied.")

    def _register_task(self, task_id: str, description: str, task_type: str) -> None:
        entry = self._task_tracker.get(task_id)
        if not entry:
            self._task_tracker[task_id] = {
                "description": description,
                "type": task_type,
                "status": "pending",
                "attempts": 0,
                "last_error": None,
                "details": {},
                "updated_at": time.time(),
            }
        else:
            if description and description != entry.get("description"):
                entry["description"] = description
            entry.setdefault("type", task_type)
        self._update_requirement_flags_from_text(description)

    def _mark_task_completed(self, task_id: str, details: Optional[Dict[str, Any]] = None) -> None:
        entry = self._task_tracker.get(task_id)
        if not entry:
            return
        entry["status"] = "completed"
        entry["updated_at"] = time.time()
        entry["last_error"] = None
        if details is not None:
            entry["details"] = details

    def _mark_task_failed(self, task_id: str, error: Optional[str] = None) -> None:
        entry = self._task_tracker.get(task_id)
        if not entry:
            return
        entry["status"] = "pending"
        entry["attempts"] = entry.get("attempts", 0) + 1
        entry["updated_at"] = time.time()
        if error:
            entry["last_error"] = error

    def _record_action_outcome(
        self,
        action: str,
        success: bool,
        state_before: Optional[Dict[str, Any]],
        state_after: Optional[Dict[str, Any]],
        page_changed: bool,
    ) -> None:
        """
        Store a lightweight summary of the last executed action so the determiner
        can see recent context and blockers can react accordingly.
        """
        summary = {
            "action": action,
            "success": success,
            "page_changed": page_changed,
            "url_before": (state_before or {}).get("url"),
            "url_after": (state_after or {}).get("url"),
            "timestamp": time.time(),
        }
        if not success:
            summary["note"] = "Action execution failed."
        elif not page_changed:
            summary["note"] = "No visible page change detected."
        self._last_action_summary = summary
        self._update_task_blockers(summary)
        
        # Clear temporary user inputs only if action succeeded and was NOT an ask commission
        # Why: Persists user guidance until it's actually used by a successful action
        if success and not action.lower().startswith("ask:"):
            if self._temp_user_inputs:
                # print(f"✨ Successful action '{action}' executed, clearing temporary user guidance.")
                self._temp_user_inputs.clear()

    def _update_task_blockers(self, summary: Dict[str, Any]) -> None:
        """
        Annotate downstream tasks with blocker notes (e.g., authentication walls).
        """
        url_after = (summary.get("url_after") or "").strip()
        if not url_after:
            return

        parsed = urlparse(url_after)
        host = (parsed.netloc or "").lower()
        entry = self._task_tracker.get("google_docs_report")
        if not entry:
            return

        details = entry.setdefault("details", {})
        blockers: Dict[str, Any] = details.setdefault("blockers", {})
        blocker_key = "google_auth_wall"
        blocker_message = "Google Docs is gated by a sign-in screen. Use defer-input or manual authentication."
        blocker_added = False
        blocker_removed = False

        if "accounts.google.com" in host or "signin" in host:
            blocker = blockers.get(blocker_key)
            if not blocker:
                blockers[blocker_key] = {
                    "message": blocker_message,
                    "url": url_after,
                    "timestamp": summary.get("timestamp"),
                }
                blocker_added = True
            entry["last_error"] = blocker_message
            details.setdefault("ready", False)
        else:
            if blockers.pop(blocker_key, None) is not None:
                blocker_removed = True
            if entry.get("last_error") == blocker_message:
                entry["last_error"] = None

        if blocker_added or blocker_removed:
            entry["updated_at"] = time.time()

    def _build_current_task_prompt(self, original_prompt: str) -> str:
        # Use rewritten prompt if available (from stuck detector)
        if self._current_task_prompt:
            return self._current_task_prompt
        
        if not self._task_tracker:
            return original_prompt

        pending_items = [
            (task_id, data)
            for task_id, data in self._task_tracker.items()
            if data.get("status") != "completed"
        ]
        completed_items = [
            (task_id, data)
            for task_id, data in self._task_tracker.items()
            if data.get("status") == "completed"
        ]

        lines = [original_prompt.strip(), "", "---- Task Progress ----"]

        if completed_items:
            lines.append("Completed objectives:")
            for _, data in completed_items:
                lines.append(f"- ✅ {data.get('description')}")
        else:
            lines.append("Completed objectives: (none yet)")

        if pending_items:
            lines.append("")
            lines.append("Current focus (pending):")
            for task_id, data in pending_items:
                status = data.get("status", "pending")
                attempts = data.get("attempts", 0)
                descriptor = data.get("description")
                status_note = " (in progress)" if status == "running" else ""
                attempt_note = f" [retries: {attempts}]" if attempts else ""
                lines.append(f"- ⏳ {descriptor}{status_note}{attempt_note}")
                last_error = data.get("last_error")
                if last_error:
                    lines.append(f"    ↳ last error: {last_error}")
                details = data.get("details") or {}
                if task_id == "google_docs_report" and details.get("ready"):
                    reason = details.get("ready_reason", "Extraction prerequisites satisfied.")
                    lines.append(f"    ↳ Next step: open Google Docs and compose the report. ({reason})")
                blockers = (details.get("blockers") or {})
                if blockers:
                    for blocker in blockers.values():
                        message = blocker.get("message", "Blocked")
                        lines.append(f"    ↳ Blocked: {message}")
                        blocker_url = blocker.get("url")
                        if blocker_url:
                            lines.append(f"       url: {blocker_url}")
            lines.append("")
            lines.append("Tip: If the required content is on a different site, use `navigate: <url>` to open it directly.")
        # Collect all user inputs (permanent and temporary)
        all_user_inputs = self._user_inputs + self._temp_user_inputs
        if all_user_inputs:
            lines.append("")
            lines.append("User-provided clarifications (most recent first):")
            # We don't clear self._temp_user_inputs here anymore; handled in _record_action_outcome
            for entry in reversed(all_user_inputs[-3:]):
                prompt = (entry.get("prompt") or "").strip()
                response = (entry.get("response") or "").strip()
                if prompt:
                    lines.append(f"- {prompt}: {response}")
                    # Also print to terminal for user verification
                    print(f"🔹 [DEBUG] Including user guidance in prompt: {prompt} -> {response}")
                else:
                    lines.append(f"- {response}")
                    print(f"🔹 [DEBUG] Including user guidance in prompt: {response}")

        if self._last_action_summary:
            summary = self._last_action_summary
            lines.append("")
            lines.append("Recent action summary:")
            action = summary.get("action", "Unknown action")
            result_word = "succeeded" if summary.get("success") else "failed"
            change_note = "changed the page" if summary.get("page_changed") else "no visible change"
            lines.append(f"- {action} → {result_word} ({change_note}).")
            url_after = summary.get("url_after")
            if url_after:
                lines.append(f"  URL after: {url_after}")
            note = summary.get("note")
            if note:
                lines.append(f"  Note: {note}")

        return "\n".join(lines)

    def _should_skip_extraction(self, prompt_key: str) -> bool:
        failures = self._extraction_failures.get(prompt_key, 0)
        return failures >= 2

    def _record_extraction_failure(self, prompt_key: str) -> None:
        self._extraction_failures[prompt_key] = self._extraction_failures.get(prompt_key, 0) + 1

    def _record_extraction_success(self, prompt_key: str) -> None:
        if prompt_key in self._extraction_failures:
            del self._extraction_failures[prompt_key]
        if prompt_key in self._retarget_attempts:
            del self._retarget_attempts[prompt_key]

    def _retarget_after_extraction_failure(
        self,
        prompt_key: str,
        extraction_prompt: str,
        snapshot: BrowserState,
        environment_state: EnvironmentState,
        user_prompt: str,
    ) -> Optional[_RetargetDecision]:
        """
        Use an LLM to determine if we should retarget the task after repeated extraction failures.
        """
        attempts = self._retarget_attempts.get(prompt_key, 0)
        if attempts >= 2:
            return None
        self._retarget_attempts[prompt_key] = attempts + 1

        visible_text = snapshot.visible_text or ""
        snippet = visible_text[:1200]

        history_summary = ""
        if environment_state.url_history:
            history_summary = " → ".join(environment_state.url_history[-5:])

        prompt = (
            "The agent attempted to extract structured data but the extraction failed.\n"
            "Decide whether the current page contains the requested information, and if not, provide a single retargeting command.\n\n"
            f"User task: {user_prompt}\n"
            f"Extraction request: {extraction_prompt}\n"
            f"Current URL: {snapshot.url}\n"
            f"Page title: {snapshot.title}\n"
            f"Recent URL history: {history_summary or 'N/A'}\n"
            f"Visible text snippet (truncated):\n{snippet}\n\n"
            "If the page is irrelevant or missing the requested subject, output a command such as:\n"
            "- navigate: <url>\n"
            "- search: <query>\n"
            "- click: <selector or link text>\n"
            "Return 'none' if the page is correct and the agent should try extraction again without navigating.\n"
            "Avoid scroll actions, and prefer precise navigation targets when suggesting URLs."
        )

        try:
            decision = generate_model(
                prompt=prompt,
                system_prompt=(
                    "You help a web-browsing automation agent recover from failed data extractions. "
                    "Evaluate the current page and produce the best single next action. "
                    "Respond with a short command (or 'none') and a concise rationale."
                ),
                model_object_type=_RetargetDecision,
                model=self.agent_model_name,
                reasoning_level=self.agent_reasoning_level,
            )
            return decision
        except Exception as e:
            print(f"⚠️ Failed to obtain retarget decision: {e}")
            return None
    
    def _parse_action_for_act_params(
        self,
        action_command: str,
        user_prompt: str
    ) -> dict:
        """
        Parse action command to extract parameters for act() function.
        
        Extracts:
        - Ordinal information (first, second, third, etc.) → modifier
        - Collection hints (article, button, link, etc.) → additional_context
        - Target context guard (for filtering elements)
        
        Args:
            action_command: The action command (e.g., "click: first article")
            user_prompt: The original user prompt for context
            
        Returns:
            Dictionary with act() parameters
        """
        import re
        from utils.intent_parsers import ORDINAL_WORDS
        
        params = {
            "goal_description": action_command,
            "additional_context": "",
            "target_context_guard": None,
            "modifier": None,
        }
        
        # Extract ordinal information
        action_lower = action_command.lower()
        ordinal_word = None
        ordinal_index = None
        
        # Check for ordinal words
        for word, idx in ORDINAL_WORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", action_lower):
                ordinal_word = word
                ordinal_index = idx
                break
        
        # Also check for numeric ordinals (1st, 2nd, etc.)
        if ordinal_index is None:
            match = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", action_lower)
            if match:
                ordinal_index = max(int(match.group(1)) - 1, 0)
                ordinal_word = f"{ordinal_index + 1}"
        
        # If ordinal found, add to modifier (only if modifier is enabled)
        if ordinal_index is not None and self.act_enable_modifier:
            # Format: "first" -> ["ordinal:0"], "second" -> ["ordinal:1"]
            params["modifier"] = [f"ordinal:{ordinal_index}"]
            
            # Add to additional_context for better planning (only if additional_context is enabled)
            if self.act_enable_additional_context:
                params["additional_context"] = f"Target is the {ordinal_word} element in the list/collection. "
        
        # Extract collection hints (article, button, link, etc.) - only if additional_context is enabled
        if self.act_enable_additional_context:
            collection_patterns = {
                "article": r"\barticle\b",
                "button": r"\bbutton\b",
                "link": r"\blink\b",
                "item": r"\bitem\b",
                "entry": r"\bentry\b",
                "row": r"\brow\b",
            }
            
            found_collections = []
            for collection, pattern in collection_patterns.items():
                if re.search(pattern, action_lower):
                    found_collections.append(collection)
            
            if found_collections:
                collection_context = f"Looking for a {', '.join(found_collections)}. "
                params["additional_context"] += collection_context
        
        # Keyword mode is the default (click:, type:, etc.) - no need to specify
        
        # Add target context guard for ordinal selection (only if target_context_guard is enabled)
        # This helps the plan generator filter to the correct ordinal position
        if ordinal_index is not None and self.act_enable_target_context_guard:
            # Guard: element must be at the specified position in a list/collection
            params["target_context_guard"] = f"Element must be the {ordinal_word} in the list/collection"
        
        # If any parameter is disabled, ensure it's set to None/empty
        if not self.act_enable_target_context_guard:
            params["target_context_guard"] = None
        if not self.act_enable_modifier:
            params["modifier"] = None
        if not self.act_enable_additional_context:
            params["additional_context"] = ""
        
        # NOTE: We do NOT add the original user prompt to additional_context
        # The plan generator should focus ONLY on the immediate goal at hand,
        # not the overall task. This prevents the plan generator from trying to
        # accomplish multiple goals at once (e.g., searching for "hacker news"
        # when the goal is just "click: Google Search")
        
        return params
    
    def _determine_scroll_direction_from_position(
        self,
        viewport_snapshot: BrowserState,
        full_page_snapshot: BrowserState,
        target_description: Optional[str]
    ) -> str:
        """
        Deterministically determine scroll direction by detecting elements and comparing positions.
        
        This method:
        1. Detects all form elements on the page
        2. Finds the target element (e.g., "email field")
        3. Gets its absolute Y position (scroll_y + element.rect.top)
        4. Compares with current viewport bounds
        5. Returns "scroll: up" or "scroll: down"
        """
        try:
            # Get current scroll position and viewport bounds
            current_scroll_y = full_page_snapshot.scroll_y
            viewport_height = full_page_snapshot.page_height
            viewport_top = current_scroll_y
            viewport_bottom = current_scroll_y + viewport_height
            
            # Detect all form elements on the page
            page = self.bot.page
            if not page:
                # Fallback to simple heuristic
                return "scroll: down" if current_scroll_y == 0 else "scroll: up"
            
            # Get all interactive elements with their positions
            # This includes: form fields, buttons, links, and other clickable elements
            js_code = """
            (function() {
                const elements = [];
                
                // Get all potentially interactive elements
                const selectors = [
                    'input, select, textarea',  // Form fields
                    'button, [role="button"]',  // Buttons
                    'a[href]',                  // Links
                    '[onclick]',                // Elements with onclick handlers
                    '[tabindex]:not([tabindex="-1"])'  // Focusable elements
                ];
                
                const allElements = new Set();
                selectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => allElements.add(el));
                });
                
                allElements.forEach((element) => {
                    const rect = element.getBoundingClientRect();
                    
                    // Skip hidden elements
                    if (rect.width === 0 || rect.height === 0 || 
                        getComputedStyle(element).display === 'none' ||
                        getComputedStyle(element).visibility === 'hidden') {
                        return;
                    }
                    
                    // Calculate absolute Y position (viewport-relative + scroll offset)
                    const absoluteY = window.scrollY + rect.top;
                    const centerY = absoluteY + (rect.height / 2);
                    
                    // Get text content for matching
                    const textContent = element.textContent?.trim() || '';
                    const innerText = element.innerText?.trim() || '';
                    
                    elements.push({
                        tagName: element.tagName.toLowerCase(),
                        type: element.type || '',
                        name: element.name || '',
                        id: element.id || '',
                        className: element.className || '',
                        placeholder: element.placeholder || '',
                        label: element.labels?.[0]?.textContent || '',
                        textContent: textContent,
                        innerText: innerText,
                        href: element.href || '',
                        role: element.getAttribute('role') || '',
                        ariaLabel: element.getAttribute('aria-label') || '',
                        absoluteY: absoluteY,
                        centerY: centerY,
                        rect: {
                            top: rect.top,
                            left: rect.left,
                            height: rect.height,
                            width: rect.width
                        }
                    });
                });
                
                return elements;
            })();
            """
            
            elements = page.evaluate(js_code)
            if not elements:
                # Fallback if no elements found
                return "scroll: down" if current_scroll_y == 0 else "scroll: up"
            
            # Find target element based on description
            # This matches against various element attributes (name, id, type, text, href, etc.)
            target_element = None
            if target_description:
                target_lower = target_description.lower()
                
                # Extract keywords from target description
                keywords = []
                if 'email' in target_lower or 'mail' in target_lower:
                    keywords.extend(['email', 'mail'])
                if 'name' in target_lower:
                    keywords.append('name')
                if 'password' in target_lower or 'pass' in target_lower:
                    keywords.extend(['password', 'pass'])
                if 'button' in target_lower:
                    keywords.append('button')
                if 'link' in target_lower:
                    keywords.append('link')
                if 'submit' in target_lower:
                    keywords.extend(['submit', 'button'])
                if 'click' in target_lower:
                    # For click actions, try to extract what to click
                    if 'hacker' in target_lower and 'news' in target_lower:
                        keywords.extend(['hacker', 'news', 'ycombinator'])
                
                # Score each element based on how well it matches
                best_match = None
                best_score = 0
                
                for element in elements:
                    score = 0
                    
                    # Check various attributes
                    searchable_text = ' '.join([
                        element.get('name', ''),
                        element.get('id', ''),
                        element.get('type', ''),
                        element.get('textContent', ''),
                        element.get('innerText', ''),
                        element.get('placeholder', ''),
                        element.get('label', ''),
                        element.get('ariaLabel', ''),
                        element.get('href', ''),
                        element.get('className', '')
                    ]).lower()
                    
                    # Score based on keyword matches
                    for keyword in keywords:
                        if keyword in searchable_text:
                            score += 1
                            # Higher weight for exact matches in key fields
                            if keyword in element.get('name', '').lower():
                                score += 2
                            if keyword in element.get('id', '').lower():
                                score += 2
                            if keyword in element.get('type', '').lower():
                                score += 2
                    
                    # Bonus for exact type matches
                    if 'email' in keywords and element.get('type') == 'email':
                        score += 3
                    if 'password' in keywords and element.get('type') == 'password':
                        score += 3
                    
                    # Bonus for button/link tag matches
                    if 'button' in keywords and element.get('tagName') == 'button':
                        score += 2
                    if 'link' in keywords and element.get('tagName') == 'a':
                        score += 2
                    
                    if score > best_score:
                        best_score = score
                        best_match = element
                
                if best_match and best_score > 0:
                    target_element = best_match
                    print(f"   🎯 Matched target element (score: {best_score}): {best_match.get('tagName', 'unknown')} - {best_match.get('name', '') or best_match.get('id', '') or best_match.get('textContent', '')[:50]}")
            
            # If target element found, compare position
            if target_element:
                target_y = target_element['centerY']  # Use center Y for better accuracy
                
                print(f"   📍 Target element found at absolute Y: {target_y}")
                print(f"   📍 Viewport bounds: Y={viewport_top} to Y={viewport_bottom}")
                
                if target_y < viewport_top:
                    print("   ⬆️ Target is above viewport → scroll: up")
                    return "scroll: up"
                elif target_y > viewport_bottom:
                    print("   ⬇️ Target is below viewport → scroll: down")
                    return "scroll: down"
                else:
                    # Target is in viewport - shouldn't happen in exploration mode
                    print("   ✅ Target is in viewport (should not be in exploration mode)")
                    return "scroll: down"  # Default
            
            # Fallback: if we can't find target, use simple heuristic
            print("   ⚠️ Target element not found, using heuristic")
            if current_scroll_y > 0:
                return "scroll: up"  # Already scrolled, try up first
            else:
                return "scroll: down"  # At top, scroll down
                
        except Exception as e:
            print(f"⚠️ Error determining scroll direction: {e}")
            # Final fallback
            if full_page_snapshot.scroll_y > 0:
                return "scroll: up"
            else:
                return "scroll: down"
    
    def _is_in_exploration_mode(self, recent_actions) -> bool:
        """
        Detect if agent is in exploration mode (searching for missing elements).
        
        Exploration mode is triggered when:
        - Recent actions are primarily scrolling
        - Multiple scroll actions in a row suggest we're looking for something
        """
        if not recent_actions:
            return False
        
        # Count scroll actions in recent history
        scroll_count = sum(
            1 for action in recent_actions
            if action.interaction_type.value == "scroll"
        )
        
        # If 2+ scrolls in last 3 actions, we're exploring
        if scroll_count >= 2:
            return True
        
        # Also check if last action was scroll (actively searching)
        if recent_actions and recent_actions[-1].interaction_type.value == "scroll":
            # Check if we've scrolled multiple times total
            total_scrolls = sum(
                1 for action in self.bot.session_tracker.interaction_history
                if action.interaction_type.value == "scroll"
            )
            # If we've scrolled 3+ times total, likely exploring
            if total_scrolls >= 3:
                return True
        
        return False
    
    def _simple_completion_check(
        self,
        user_prompt: str,
        snapshot: BrowserState
    ) -> tuple[bool, str]:
        """
        Simple fallback completion check (kept for compatibility).
        Step 2 uses LLM-based completion contract instead.
        
        Simple rule-based completion check.
        
        Checks:
        - URL matches if prompt mentions navigation
        - Page text contains expected keywords
        - URL patterns match expected destinations
        
        Returns:
            (is_complete, reasoning)
        """
        prompt_lower = user_prompt.lower()
        url_lower = snapshot.url.lower()
        title_lower = (snapshot.title or "").lower()
        text_lower = (snapshot.visible_text or "").lower()
        
        # Check for navigation completion
        nav_keywords = ["navigate", "go to", "visit", "open", "goto"]
        if any(keyword in prompt_lower for keyword in nav_keywords):
            # Try to extract target URL/domain from prompt
            # Simple pattern: "go to example.com" or "navigate to https://example.com"
            url_patterns = re.findall(
                r'(?:https?://)?([a-z0-9\-]+(?:\.[a-z0-9\-]+)+)',
                prompt_lower
            )
            if url_patterns:
                target_domain = url_patterns[0]
                if target_domain in url_lower:
                    return True, f"Navigated to target domain: {target_domain}"
            
            # Check for common navigation success indicators
            if "error" not in text_lower and "404" not in url_lower:
                # If we're on a different page than before, might be success
                # (This is basic - could be improved)
                pass
        
        # Check for completion keywords in page content
        completion_keywords = [
            "success", "complete", "done", "submitted", "confirmed",
            "thank you", "application received", "sent successfully"
        ]
        for keyword in completion_keywords:
            if keyword in text_lower or keyword in title_lower:
                return True, f"Completion keyword found: '{keyword}'"
        
        # Check for error indicators (task likely not complete)
        error_keywords = ["error", "failed", "invalid", "not found", "404"]
        for keyword in error_keywords:
            if keyword in text_lower or keyword in title_lower:
                # Don't mark complete if errors are present
                return False, f"Error indicator found: '{keyword}'"
        
        # Default: not complete
        return False, "Completion criteria not met"
    
    def _get_page_state(self) -> dict:
        """
        Get current page state for change detection.

        Returns:
            Dictionary with url, dom_signature, and UI state information
        """
        try:
            url = self.bot.page.url
            # Get element count as a simple DOM change indicator
            element_count = self.bot.page.evaluate("() => document.querySelectorAll('*').length")
            sig_src = f"{url}|{element_count}"
            dom_signature = hashlib.md5(sig_src.encode("utf-8")).hexdigest()
            screenshot_hash = getattr(self.bot.page, "_last_screenshot_hash", None)

            # Get additional UI state information for better change detection
            ui_state = self._get_ui_state_info()

            return {
                "url": url,
                "dom_signature": dom_signature,
                "screenshot_hash": screenshot_hash,
                **ui_state
            }
        except Exception:
            # Fallback to just URL
            try:
                url = self.bot.page.url
                dom_signature = hashlib.md5(url.encode("utf-8")).hexdigest()
                screenshot_hash = getattr(self.bot.page, "_last_screenshot_hash", None)
                return {
                    "url": url,
                    "dom_signature": dom_signature,
                    "screenshot_hash": screenshot_hash,
                    "visible_elements": [],
                    "overlay_elements": []
                }
            except Exception:
                return {
                    "url": "",
                    "dom_signature": "",
                    "screenshot_hash": None,
                    "visible_elements": [],
                    "overlay_elements": []
                }

    def _get_ui_state_info(self) -> dict:
        """
        Get UI state information for change detection.

        Returns:
            Dictionary with visible elements and overlay information
        """
        try:
            # Get visible elements (simplified version for performance)
            visible_elements = self.bot.page.evaluate("""
                () => {
                    const elements = [];
                    const allElements = document.querySelectorAll('*');

                    for (let i = 0; i < Math.min(allElements.length, 200); i++) {  // Limit for performance
                        const el = allElements[i];
                        const rect = el.getBoundingClientRect();

                        // Check if element is visible in viewport
                        if (rect.width > 0 && rect.height > 0 &&
                            rect.bottom > 0 && rect.right > 0 &&
                            rect.top < window.innerHeight && rect.left < window.innerWidth) {

                            elements.push({
                                tagName: el.tagName,
                                textContent: (el.textContent || '').trim().substring(0, 50),
                                id: el.id || '',
                                className: el.className || '',
                                role: el.getAttribute('role') || '',
                                type: el.type || ''
                            });

                            if (elements.length >= 100) break;  // Limit visible elements
                        }
                    }

                    return elements;
                }
            """)

            # Get overlay elements if available
            overlay_elements = []
            if hasattr(self.bot, '_cached_overlay_data') and self.bot._cached_overlay_data:
                # Simplify overlay data for comparison
                overlay_elements = [{
                    'tag': elem.get('tagName', ''),
                    'text': elem.get('textContent', '')[:30],
                    'id': elem.get('id', '')
                } for elem in self.bot._cached_overlay_data[:50]]  # Limit overlay elements

            return {
                "visible_elements": visible_elements or [],
                "overlay_elements": overlay_elements
            }

        except Exception:
            return {
                "visible_elements": [],
                "overlay_elements": []
            }

    def _spawn_child_controller(
        self,
        bot,
        *,
        base_knowledge: Optional[List[str]] = None,
        track_ineffective_actions: Optional[bool] = None,
        allow_partial_completion: Optional[bool] = None
    ) -> "AgentController":
        """
        Factory method passed to SubAgentController to ensure sub-agents inherit configuration.
        """
        track = self.detect_ineffective_actions if track_ineffective_actions is None else track_ineffective_actions
        partial = self.allow_partial_completion if allow_partial_completion is None else allow_partial_completion
        return AgentController(
            bot=bot,
            track_ineffective_actions=track,
            base_knowledge=base_knowledge,
            allow_partial_completion=partial
        )
    
    def _drain_sub_agent_results(self) -> None:
        """
        Retrieve any newly completed sub-agent results and integrate them.
        """
        if not self.sub_agent_controller:
            return
        results = self.sub_agent_controller.pop_completed_results()
        if not results:
            return
        
        for result in results:
            self.sub_agent_results.append(result)
            status_icon = "✅" if result.success else "⚠️"
            duration = max(0.0, result.completed_at - result.started_at)
            print(f"{status_icon} Sub-agent [{result.agent_id}] ({result.instruction}) -> {result.status} "
                  f"(confidence={result.confidence:.2f}, duration={duration:.2f}s)")
            
            # Merge extracted data into main agent store if provided
            evidence = result.evidence or {}
            extracted = evidence.get("extracted_data")
            if isinstance(extracted, dict):
                for key, value in extracted.items():
                    if key not in self.extracted_data:
                        self.extracted_data[key] = value
            self._log_event(
                "sub_agent_result_recorded",
                agent_id=result.agent_id,
                success=result.success,
                status=result.status,
            )

    def _build_evidence(self, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Construct evidence dictionaries including extracted data and sub-agent results.
        """
        evidence: Dict[str, Any] = {}
        if base:
            evidence.update(base)
        if self.extracted_data and "extracted_data" not in evidence:
            evidence["extracted_data"] = self.extracted_data.copy()
        if self.sub_agent_results:
            evidence["sub_agents"] = [result.to_dict() for result in self.sub_agent_results]
        orchestration = {
            "events": list(self.orchestration_events),
            "tabs": self._build_tab_summary(),
        }
        # Only include sub-agent summaries if not already present
        if self.sub_agent_results:
            orchestration["sub_agents"] = [result.to_dict() for result in self.sub_agent_results]
        evidence["orchestration"] = orchestration
        evidence["sub_agent_policy"] = {
            "level": self.sub_agent_policy_level.value,
            "score": round(self._sub_agent_policy_score, 2),
            "rationale": self.sub_agent_policy_rationale,
            "override": self._sub_agent_policy_override.value if self._sub_agent_policy_override else None,
            "spawn_count": self.sub_agent_spawn_count,
        }
        if self._parallel_plan:
            evidence["parallel_plan"] = {
                "main_agent_focus": self._main_agent_focus,
                "integration_notes": self._integration_notes,
                "sub_tasks": [task.dict() for task in self._parallel_plan.sub_tasks],
            }
        if self._task_tracker:
            evidence["task_progress"] = {
                task_id: {
                    "description": data.get("description"),
                    "type": data.get("type"),
                    "status": data.get("status"),
                    "attempts": data.get("attempts"),
                    "last_error": data.get("last_error"),
                    "updated_at": data.get("updated_at"),
                    "details": data.get("details"),
                }
                for task_id, data in self._task_tracker.items()
            }
        return evidence

    def _rewrite_task_prompt_using_completion_reasoning(
        self,
        original_user_prompt: str,
        current_task_prompt: str,
        completion_reasoning: str,
    ) -> Optional[str]:
        """
        Rewrite the task prompt using completion reasoning to focus on what's still needed.
        
        Uses LLM to create a new, focused prompt that doesn't mention being stuck.
        """
        try:
            from pydantic import BaseModel, Field, ConfigDict
            
            class RewrittenTaskPrompt(BaseModel):
                model_config = ConfigDict(extra="forbid")
                rewritten_prompt: str = Field(
                    description="New task prompt that keeps the original goal but narrows it to the next concrete subtask. "
                               "Do NOT mention that the agent is stuck or looping. Write it as a fresh instruction."
                )
            
            system_prompt = """You rewrite high-level web automation tasks into focused subtasks.

You are given:
- The original user task
- The current task prompt (which may already be a rewrite)
- A completion evaluator's reasoning explaining why the task is NOT complete

Your job:
- Keep the original ultimate goal in mind
- Produce a short rewritten task prompt that focuses the agent on the NEXT concrete sub-goal needed right now
- Do NOT mention that the agent is stuck or looping
- Avoid meta commentary; write it as a plain instruction the agent would normally receive
- Be specific about what needs to be done next based on the completion reasoning

Examples:
- Original: "Collect all quotes from the first 3 pages"
  Completion reasoning: "Only quotes from page 1 collected; pages 2 and 3 not visited."
  Rewritten: "From the current page, navigate to pages 2 and 3 of the quote site and extract all quotes from each page."

- Original: "Fill out the contact form"
  Completion reasoning: "Form fields are visible but not yet filled."
  Rewritten: "Fill in all visible form fields on the contact form and submit it."

Output JSON with just the field 'rewritten_prompt'."""
            
            user_prompt_text = f"""ORIGINAL USER PROMPT:
{original_user_prompt}

CURRENT TASK PROMPT:
{current_task_prompt}

COMPLETION REASONING (not complete):
{completion_reasoning}

Write a new, focused task prompt that the agent should follow next.
Do NOT mention being stuck or looping. Write it as a fresh instruction."""
            
            result = generate_model(
                prompt=user_prompt_text,
                model_object_type=RewrittenTaskPrompt,
                system_prompt=system_prompt,
                model=self.agent_model_name,
                reasoning_level=ReasoningLevel.LOW,  # Use low reasoning for prompt rewriting
            )
            
            return result.rewritten_prompt if result else None
            
        except Exception as e:
            print(f"⚠️ Failed to rewrite task prompt: {e}")
            return None

    def _build_tab_summary(self) -> List[Dict[str, Any]]:
        tab_manager = getattr(self.bot, "tab_manager", None)
        if not tab_manager:
            return []
        try:
            tabs = tab_manager.list_tabs()
        except Exception:
            return []
        summary = []
        for tab in tabs:
            if hasattr(tab, "to_dict"):
                summary.append(tab.to_dict())
            else:
                summary.append({"tab_id": getattr(tab, "tab_id", None)})
        return summary

    def _page_state_changed(self, state_before: dict, state_after: dict) -> bool:
        """
        Check if page state changed after an action.

        Args:
            state_before: Page state before action
            state_after: Page state after action

        Returns:
            True if page state changed, False otherwise
        """
        # Check if URL changed
        if state_before.get("url") != state_after.get("url"):
            return True

        # Check if DOM signature changed
        if state_before.get("dom_signature") != state_after.get("dom_signature"):
            return True

        # Check if screenshot hash changed (if available)
        if state_before.get("screenshot_hash") != state_after.get("screenshot_hash"):
            return True

        # Check for UI state changes that DOM signature might miss
        try:
            ui_changes = self._detect_ui_state_changes(state_before, state_after)
            if ui_changes:
                if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
                    print(f"   🔄 Detected UI state changes: {ui_changes}")
                return True
        except Exception as e:
            if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
                print(f"   ⚠️ Error checking UI state changes: {e}")

        return False

    def _detect_ui_state_changes(self, state_before: dict, state_after: dict) -> list[str]:
        """
        Detect subtle UI changes that basic DOM comparison might miss.

        Returns:
            List of detected changes (empty if no changes)
        """
        changes = []

        try:
            # Check for changes in visible elements count
            visible_before = len(state_before.get("visible_elements", []))
            visible_after = len(state_after.get("visible_elements", []))

            if abs(visible_before - visible_after) > 2:  # Allow small variations
                changes.append(f"visible elements: {visible_before} → {visible_after}")

            # Check for new interactive elements appearing
            interactive_before = set()
            interactive_after = set()

            for elem in state_before.get("visible_elements", []):
                if elem.get("tagName", "").lower() in ["button", "input", "select", "a"] or elem.get("role") in ["button", "link", "option", "combobox"]:
                    interactive_before.add(f"{elem.get('tagName', '')}:{elem.get('textContent', '')[:20]}")

            for elem in state_after.get("visible_elements", []):
                if elem.get("tagName", "").lower() in ["button", "input", "select", "a"] or elem.get("role") in ["button", "link", "option", "combobox"]:
                    interactive_after.add(f"{elem.get('tagName', '')}:{elem.get('textContent', '')[:20]}")

            new_interactive = interactive_after - interactive_before
            if new_interactive:
                changes.append(f"new interactive elements: {len(new_interactive)}")

            # Check for changes in overlay count (indicates dropdowns/modals)
            overlays_before = len(state_before.get("overlay_elements", []))
            overlays_after = len(state_after.get("overlay_elements", []))

            if overlays_before != overlays_after:
                changes.append(f"overlay elements: {overlays_before} → {overlays_after}")

            # Check for form state changes (expanded forms, new inputs)
            form_elements_before = sum(1 for elem in state_before.get("visible_elements", [])
                                     if elem.get("tagName", "").lower() in ["input", "select", "textarea"])
            form_elements_after = sum(1 for elem in state_after.get("visible_elements", [])
                                    if elem.get("tagName", "").lower() in ["input", "select", "textarea"])

            if abs(form_elements_before - form_elements_after) > 1:
                changes.append(f"form elements: {form_elements_before} → {form_elements_after}")

        except Exception:
            # Don't let UI change detection break the main flow
            pass

        return changes
    
    def _update_sub_agent_policy(self, user_prompt: str, completion_reasoning: Optional[str]) -> None:
        """
        Evaluate and update the adaptive sub-agent utilization policy.
        """
        new_level, new_score, new_rationale = self._compute_sub_agent_policy(
            user_prompt=user_prompt,
            completion_reasoning=completion_reasoning or "",
            suppress_quick_check_print=True  # Suppress quick check print, we'll print the full policy update instead
        )
        
        level_changed = new_level != self.sub_agent_policy_level
        score_changed = abs(new_score - self._sub_agent_policy_score) >= 0.1
        rationale_changed = new_rationale != self.sub_agent_policy_rationale
        
        self.sub_agent_policy_level = new_level
        self._sub_agent_policy_score = new_score
        self.sub_agent_policy_rationale = new_rationale
        
        if level_changed or score_changed or rationale_changed:
            try:
                self.event_logger.sub_agent_policy(
                    policy=self._policy_display_name(new_level),
                    score=new_score,
                    reason=new_rationale
                )
            except Exception:
                pass
            self._log_event(
                "sub_agent_policy_update",
                policy=new_level.value,
                score=new_score,
                rationale=new_rationale,
                override=self._sub_agent_policy_override.value if self._sub_agent_policy_override else None
            )
    
    def _compute_sub_agent_policy(
        self,
        user_prompt: str,
        completion_reasoning: str,
        suppress_quick_check_print: bool = False
    ) -> Tuple[SubAgentPolicyLevel, float, str]:
        """
        Compute the appropriate sub-agent utilization policy score and level using an LLM.
        
        Args:
            suppress_quick_check_print: If True, don't print the quick check message (used when
                                       called from _update_sub_agent_policy to avoid duplicate prints)
        """
        if not self.sub_agent_controller or not getattr(self.bot, "tab_manager", None):
            try:
                self.event_logger.sub_agent_policy(
                    policy="Single Threaded",
                    score=0.0,
                    reason="Sub-agent policy decision: controller or tab manager unavailable (forcing single-threaded)."
                )
            except Exception:
                pass
            return (
                SubAgentPolicyLevel.SINGLE_THREADED,
                0.0,
                "Sub-agent controller or tab manager unavailable."
            )
        
        if self._sub_agent_policy_override:
            level = self._sub_agent_policy_override
            score_override = {
                SubAgentPolicyLevel.SINGLE_THREADED: 0.0,
                SubAgentPolicyLevel.PARALLELIZED: 1.0
            }[level]
            try:
                self.event_logger.sub_agent_policy(
                    policy=self._policy_display_name(level),
                    score=score_override,
                    reason="Sub-agent policy override active"
                )
            except Exception:
                pass
            return (
                level,
                score_override,
                f"Override active: forced to {self._policy_display_name(level)}."
            )
        
        # First, do a lightweight yes/no check
        needs_sub_agents = self._query_sub_agent_policy_yes_no(
            user_prompt=user_prompt,
            completion_reasoning=completion_reasoning
        )
        
        # If no sub-agents needed, return immediately without detailed evaluation
        if not needs_sub_agents:
            if not suppress_quick_check_print:
                try:
                    self.event_logger.sub_agent_policy(
                        policy="Single Threaded",
                        score=0.0,
                        reason="Quick check: no sub-agents needed"
                    )
                except Exception:
                    pass
            return (
                SubAgentPolicyLevel.SINGLE_THREADED,
                0.0,
                "Quick check: no sub-agents needed"
            )
        
        # Only do detailed evaluation if yes/no check says sub-agents are needed
        policy_spec, score, rationale = self._query_sub_agent_policy_llm(
            user_prompt=user_prompt,
            completion_reasoning=completion_reasoning
        )
        
        if not isinstance(policy_spec, str):
            policy_spec = SubAgentPolicyLevel.SINGLE_THREADED.value
        
        policy_spec_normalized = policy_spec.strip().lower()
        level_map = {
            "single": SubAgentPolicyLevel.SINGLE_THREADED,
            "single_threaded": SubAgentPolicyLevel.SINGLE_THREADED,
            "single-threaded": SubAgentPolicyLevel.SINGLE_THREADED,
            "assistive": SubAgentPolicyLevel.PARALLELIZED,
            "balanced": SubAgentPolicyLevel.PARALLELIZED,
            "default": SubAgentPolicyLevel.SINGLE_THREADED,
            "parallel": SubAgentPolicyLevel.PARALLELIZED,
            "parallelized": SubAgentPolicyLevel.PARALLELIZED,
            "aggressive": SubAgentPolicyLevel.PARALLELIZED,
        }
        policy_level = level_map.get(policy_spec_normalized, SubAgentPolicyLevel.SINGLE_THREADED)
        
        score_clamped = max(0.0, min(1.0, score if isinstance(score, (int, float)) else (1.0 if policy_level == SubAgentPolicyLevel.PARALLELIZED else 0.0)))
        rationale_text = rationale or "LLM policy evaluation."
        
        try:
            self.event_logger.sub_agent_policy(
                policy=self._policy_display_name(policy_level),
                score=score_clamped,
                reason=rationale_text
            )
        except Exception:
            pass
        
        return policy_level, score_clamped, rationale_text
    
    def _query_sub_agent_policy_yes_no(
        self,
        user_prompt: str,
        completion_reasoning: str
    ) -> bool:
        """
        Lightweight yes/no check to determine if sub-agents are needed.
        Returns True if sub-agents are needed, False otherwise.
        """
        prompt = self._build_sub_agent_policy_yes_no_prompt(
            user_prompt=user_prompt,
            completion_reasoning=completion_reasoning
        )
        try:
            result = generate_model(
                prompt=prompt,
                model_object_type=_SubAgentPolicyYesNo,
                system_prompt=self._build_sub_agent_policy_yes_no_system_prompt(),
                model=self.agent_model_name,
                reasoning_level=ReasoningLevel.LOW,  # Use low reasoning for quick check
            )
            return result.needs_sub_agents
        except Exception as e:
            print(f"⚠️ Failed to evaluate sub-agent policy yes/no check via LLM: {e}")
            # Fallback: assume no sub-agents needed (conservative)
            return False
    
    @staticmethod
    def _build_sub_agent_policy_yes_no_system_prompt() -> str:
        return """You are making a quick yes/no decision: should sub-agents be used for this task?

Sub-agents are useful when:
- Task requires parallel research across multiple sources/websites
- Task involves collecting data from multiple independent sources
- Task would benefit from simultaneous work in separate tabs
- User explicitly mentions multiple sources or parallel work

Sub-agents are NOT needed when:
- Task is a simple sequential navigation (visit one page)
- Task can be completed by the main agent alone
- Task is straightforward (single form, single page interaction)
- No indication of parallel research needs

Respond with just needs_sub_agents (true/false)."""
    
    def _build_sub_agent_policy_yes_no_prompt(
        self,
        user_prompt: str,
        completion_reasoning: str
    ) -> str:
        """Build a lightweight prompt for yes/no sub-agent decision"""
        return f"""
QUICK SUB-AGENT DECISION
========================

USER PROMPT:
{user_prompt}

COMPLETION REASONING:
{completion_reasoning or "N/A"}

QUESTION: Does this task need sub-agents (parallel work across multiple tabs/sources)?

Answer YES only if:
- Multiple independent sources need to be researched
- Parallel data collection from different websites
- Task explicitly benefits from simultaneous work

Answer NO if:
- Simple sequential task (one page, one form)
- Main agent can handle it alone
- No parallel research needs

Respond with needs_sub_agents (true/false).
"""
    
    def _query_sub_agent_policy_llm(
        self,
        user_prompt: str,
        completion_reasoning: str
    ) -> Tuple[str, float, str]:
        """
        Ask an LLM to evaluate whether sub-agents should be used and at what intensity.
        """
        policy_prompt = self._build_sub_agent_policy_prompt(
            user_prompt=user_prompt,
            completion_reasoning=completion_reasoning
        )
        try:
            result = generate_model(
                prompt=policy_prompt,
                model_object_type=_SubAgentPolicyResponse,
                system_prompt=self._build_sub_agent_policy_system_prompt(),
                model=self.agent_model_name,
                reasoning_level=self.agent_reasoning_level,
            )
        except Exception as e:
            print(f"⚠️ Failed to evaluate sub-agent policy via LLM: {e}")
            return SubAgentPolicyLevel.SINGLE_THREADED.value, 0.0, "Fallback adaptive policy (LLM error)."
        
        return result.policy_level, result.policy_score, result.rationale
    
    @staticmethod
    def _build_sub_agent_policy_system_prompt() -> str:
        return """You help a browser automation supervisor decide whether to delegate work to sub-agents.

Consider task urgency, need for parallel research, blockers, focus requirements, and historical success of sub-agents.
Return a concise policy recommendation."""
    
    def _build_sub_agent_policy_prompt(
        self,
        user_prompt: str,
        completion_reasoning: str
    ) -> str:
        active_sub_agents = 0
        if self.sub_agent_controller:
            active_sub_agents = sum(
                1 for ctx in self.sub_agent_controller.sub_agents.values()
                if ctx.status in {"pending", "running"}
            )
        
        history = self.sub_agent_controller.get_execution_history() if self.sub_agent_controller else []
        successes = sum(1 for result in history if getattr(result, "success", False))
        failures = len(history) - successes
        
        tab_manager = getattr(self.bot, "tab_manager", None)
        active_tab_url = ""
        if tab_manager and tab_manager.get_active_tab():
            try:
                active_tab_url = tab_manager.get_active_tab().url
            except Exception:
                active_tab_url = ""
        
        prompt_flags = {
            "multi_tab_emphasis": "spread across multiple tabs" if "multiple tabs" in user_prompt.lower() else "unspecified",
            "urgency": "yes" if any(word in user_prompt.lower() for word in ["quick", "fast", "urgent", "asap"]) else "no"
        }
        
        return f"""
SUB-AGENT UTILIZATION DECISION
==============================

USER PROMPT:
{user_prompt}

COMPLETION REASONING:
{completion_reasoning or "N/A"}

ACTIVE TAB URL:
{active_tab_url or "unknown"}

CURRENT POLICY:
- Override: {self._sub_agent_policy_override.value if self._sub_agent_policy_override else "None"}
- Previous level: {self.sub_agent_policy_level.value}
- Spawn count this run: {self.sub_agent_spawn_count}
- Spawns this iteration: {self._spawns_this_iteration}
- Active sub-agents: {active_sub_agents}
- Recent sub-agent successes: {successes}
- Recent sub-agent failures: {failures}

TASK CONTEXT FLAGS:
- Multi-tab emphasis: {prompt_flags["multi_tab_emphasis"]}
- Urgency mentioned: {prompt_flags["urgency"]}

DECISION GUIDELINES:
- single_threaded: keep work on the main agent; avoid spawning sub-agents.
- parallelized: aggressively use sub-agents; create dedicated tabs for parallel research or extraction.

Provide:
1. policy_level (single_threaded or parallelized)
2. policy_score (0.0-1.0 representing intensity of delegation)
3. rationale (one or two sentences)
"""

    def _normalize_suggested_url(self, url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        parts = url.strip().split()
        for part in parts:
            candidate = part.strip()
            if candidate.lower().startswith(("http://", "https://")):
                return candidate
        return None

    def _run_orchestrated_task(self, task: _SubAgentTaskPlan, plan_label: str) -> None:
        if not self.sub_agent_controller or not getattr(self.bot, "tab_manager", None):
            return
        if self.agent_context and self.agent_context.parent_agent_id and plan_label != "parallel_plan":
            print(f"ℹ️ {plan_label}: sub-agents do not orchestrate additional helpers (task '{task.name}' skipped).")
            return

        task_key = self._normalize_task_name(task.name)
        self._register_task(task_key, task.instruction or task.name, plan_label)
        task_entry = self._task_tracker.get(task_key)
        if task_entry:
            task_entry["status"] = "running"
            task_entry["updated_at"] = time.time()

        metadata = {
            "orchestrated": True,
            "plan": plan_label,
            "task_name": task.name,
            "expected_output": task.expected_output,
        }

        normalized_url = self._normalize_suggested_url(task.suggested_url)
        if task.suggested_url and not normalized_url:
            print(f"⚠️ {plan_label}: suggested URL '{task.suggested_url}' is invalid; skipping automatic navigation.")

        new_tab_id = self.bot.tab_manager.open_new_tab(
            purpose=task.name,
            url=normalized_url,
            metadata=metadata
        )
        if not new_tab_id:
            print(f"⚠️ {plan_label}: failed to open tab for task '{task.name}'.")
            self._log_event(f"{plan_label}_tab_failed", task=task.dict())
            self._mark_task_failed(task_key, "Unable to open browser tab for sub-task.")
            return

        sub_agent_id = self.sub_agent_controller.spawn_sub_agent(
            tab_id=new_tab_id,
            instruction=task.instruction,
            metadata=metadata
        )
        if not sub_agent_id:
            print(f"⚠️ {plan_label}: failed to spawn sub-agent for task '{task.name}'.")
            self._log_event(
                f"{plan_label}_spawn_failed",
                task=task.dict(),
                tab_id=new_tab_id
            )
            self._mark_task_failed(task_key, "Failed to spawn sub-agent for sub-task.")
            return

        if not (self.agent_context and self.agent_context.parent_agent_id):
            self.sub_agent_spawn_count += 1
            self._spawns_this_iteration += 1

        self._log_event(
            f"{plan_label}_task_spawned",
            task=task.dict(),
            tab_id=new_tab_id,
            agent_id=sub_agent_id
        )
        print(f"🤝 {plan_label}: running sub-agent '{task.name}' → {task.instruction}")

        result = self.sub_agent_controller.execute_sub_agent(sub_agent_id)
        self._log_event(
            f"{plan_label}_task_completed",
            task=task.dict(),
            tab_id=new_tab_id,
            agent_id=sub_agent_id,
            result=result
        )
        success = False
        confidence = None
        reasoning = None
        error_reason = None
        if isinstance(result, dict):
            success = bool(result.get("success"))
            confidence = result.get("confidence")
            reasoning = result.get("reasoning")
            error_reason = result.get("error")
        else:
            success = bool(getattr(result, "success", False))
            confidence = getattr(result, "confidence", None)
            reasoning = getattr(result, "reasoning", None)
            error_reason = getattr(result, "error", None)

        if success:
            self._mark_task_completed(
                task_key,
                {
                    "type": plan_label,
                    "success": True,
                    "confidence": confidence,
                    "reasoning": reasoning,
                }
            )
        else:
            self._mark_task_failed(task_key, error_reason or "Sub-agent task did not complete successfully.")
        self._drain_sub_agent_results()
    
    def _orchestrate_parallel_work(self, user_prompt: str) -> None:
        if self._parallel_plan_done:
            return
        if not self.sub_agent_controller or not getattr(self.bot, "tab_manager", None):
            self._parallel_plan_done = True
            return
        if self.agent_context and self.agent_context.parent_agent_id:
            self._parallel_plan_done = True
            return
        
        plan = self._create_parallel_plan(user_prompt)
        if not plan or not plan.sub_tasks:
            print("⚠️ Parallel work plan could not be generated. Continuing without orchestration.")
            self._parallel_plan_done = True
            return
        
        self._parallel_plan = plan
        self._main_agent_focus = plan.main_agent_focus
        self._integration_notes = plan.integration_notes
        self._log_event("parallel_plan_created", plan=plan.dict())
        for sub_task in plan.sub_tasks:
            task_key = self._normalize_task_name(sub_task.name)
            self._register_task(task_key, sub_task.instruction or sub_task.name, "parallel_task")
        if plan.main_agent_focus:
            focus_key = self._normalize_task_name(f"focus:{plan.main_agent_focus}")
            self._register_task(focus_key, plan.main_agent_focus, "integration")
        elif self._integration_notes:
            notes_key = self._normalize_task_name(f"notes:{self._integration_notes}")
            self._register_task(notes_key, self._integration_notes, "integration")
        
        if plan.integration_notes:
            print(f"🧩 Parallel integration notes: {plan.integration_notes}")
        if plan.main_agent_focus:
            print(f"🧭 Main agent focus set to: {plan.main_agent_focus}")
        elif self._main_agent_focus:
            print(f"🧭 Main agent continuing focus: {self._main_agent_focus}")
        
        original_tab_id = self.agent_context.tab_id if self.agent_context else None
        
        for task in plan.sub_tasks:
            self._run_orchestrated_task(task, "parallel_plan")
        
        if original_tab_id and self.bot.tab_manager:
            main_tab = self.bot.tab_manager.get_tab_info(original_tab_id)
            if main_tab:
                self.bot.tab_manager.switch_to_tab(original_tab_id)
                self.bot.switch_to_page(main_tab.page)
        
        self._parallel_plan_done = True
    
    def _create_parallel_plan(self, user_prompt: str) -> Optional[_ParallelWorkPlan]:
        subjects = self._extract_requested_subjects(user_prompt)
        manual_plan = self._build_manual_parallel_plan(subjects, user_prompt)
        if manual_plan:
            return manual_plan

        prompt = self._build_parallel_plan_prompt(user_prompt)
        try:
            plan = generate_model(
                prompt=prompt,
                model_object_type=_ParallelWorkPlan,
                system_prompt=self._build_parallel_plan_system_prompt(),
                model=self.agent_model_name,
                reasoning_level=self.agent_reasoning_level,
            )
            return plan
        except Exception as e:
            print(f"⚠️ Failed to generate parallel work plan: {e}")
            return None
    
    @staticmethod
    def _build_parallel_plan_system_prompt() -> str:
        return """You are the coordinator for a multi-agent browser automation system.
Break the user's high-level goal into:
- A concise focus for the main agent (the coordinator)
- 2-4 sub-agent tasks, each with clear instructions, an optional starting URL, and an expected output description.
Ensure sub-tasks are independent and collectively satisfy the overall request.
Create one sub-task per distinct source or deliverable; do not bundle multiple independent targets into a single task.
Return structured data in the provided schema only."""
    
    def _build_parallel_plan_prompt(self, user_prompt: str) -> str:
        tab_summary = self._build_tab_summary()
        tab_summary_text = json.dumps(tab_summary, indent=2) if tab_summary else "[]"
        base_rules = "\n".join(self.base_knowledge) if self.base_knowledge else "None provided."
        recent_results = [
            result.to_dict() for result in self.sub_agent_results[-5:]
        ] if self.sub_agent_results else []
        recent_results_text = json.dumps(recent_results, indent=2) if recent_results else "[]"
        
        return f"""
PARALLEL WORK PLANNING
======================

USER PROMPT:
{user_prompt}

CURRENT POLICY LEVEL: {self.sub_agent_policy_level.value} (score {self._sub_agent_policy_score:.2f})
OVERRIDE: {self._sub_agent_policy_override.value if self._sub_agent_policy_override else "None"}

KNOWN TABS:
{tab_summary_text}

RECENT SUB-AGENT RESULTS (if any):
{recent_results_text}

BASE KNOWLEDGE RULES:
{base_rules}

Provide a plan that:
- Gives 'main_agent_focus' (the primary coordinator's responsibility).
- Lists 'sub_tasks' (each with name, instruction, optional suggested_url, and expected_output). Each sub-task should target exactly one independent source or deliverable (e.g., one website, document, or dataset).
- Optionally adds 'integration_notes' explaining how the outputs should be combined.
- If the user mentions multiple sources (e.g., several websites or datasets), create a separate sub-task for each source so they can run in parallel or sequentially without overlap.
"""
    
    def _can_spawn_sub_agent(self) -> Tuple[bool, str]:
        """
        Determine if a sub-agent can be spawned under the current policy and budgets.
        """
        if not self.sub_agent_controller:
            return False, "Sub-agent controller not initialized."
        
        if self.sub_agent_policy_level == SubAgentPolicyLevel.SINGLE_THREADED:
            return False, "Policy set to single-threaded."
        
        return True, ""
    
    def _execute_tab_decision(self, decision) -> bool:
        """
        Execute a tab management decision.
        
        Args:
            decision: TabDecision object with action to take
        
        Returns:
            True if action was executed, False otherwise
        """
        if not self.bot.tab_manager:
            return False
        
        try:
            if decision.action == TabAction.SWITCH:
                if not decision.target_tab_id:
                    print("⚠️ Tab decision: SWITCH but no target_tab_id provided")
                    return False
                
                target_tab = self.bot.tab_manager.get_tab_info(decision.target_tab_id)
                if not target_tab:
                    print(f"⚠️ Tab decision: SWITCH to {decision.target_tab_id} but tab not found")
                    return False
                
                print(f"🔀 Tab Decision: Switching to tab {decision.target_tab_id} ({target_tab.purpose})")
                print(f"   Reasoning: {decision.reasoning}")
                print(f"   Confidence: {decision.confidence:.2f}")
                
                # Switch tab
                if self.bot.tab_manager.switch_to_tab(decision.target_tab_id):
                    # Switch bot to the new page
                    self.bot.switch_to_page(target_tab.page)
                    return True
                else:
                    print(f"⚠️ Failed to switch to tab {decision.target_tab_id}")
                    return False
            
            elif decision.action == TabAction.CLOSE:
                if not decision.target_tab_id:
                    print("⚠️ Tab decision: CLOSE but no target_tab_id provided")
                    return False
                
                target_tab = self.bot.tab_manager.get_tab_info(decision.target_tab_id)
                if not target_tab:
                    print(f"⚠️ Tab decision: CLOSE {decision.target_tab_id} but tab not found")
                    return False
                
                print(f"🗑️ Tab Decision: Closing tab {decision.target_tab_id} ({target_tab.purpose})")
                print(f"   Reasoning: {decision.reasoning}")
                print(f"   Confidence: {decision.confidence:.2f}")
                
                # Find another tab to switch to if closing active tab
                current_tab = self.bot.tab_manager.get_active_tab()
                switch_to = None
                if current_tab and current_tab.tab_id == decision.target_tab_id:
                    # Closing active tab, find another
                    other_tabs = [t for t in self.bot.tab_manager.list_tabs() if t.tab_id != decision.target_tab_id]
                    if other_tabs:
                        switch_to = other_tabs[0].tab_id
                
                # Close tab
                if self.bot.tab_manager.close_tab(decision.target_tab_id, switch_to=switch_to):
                    # If we switched to another tab, update bot
                    if switch_to:
                        new_tab = self.bot.tab_manager.get_tab_info(switch_to)
                        if new_tab:
                            self.bot.switch_to_page(new_tab.page)
                    return True
                else:
                    print(f"⚠️ Failed to close tab {decision.target_tab_id}")
                    return False
            
            elif decision.action == TabAction.CONTINUE:
                # No action needed, just log
                print("✅ Tab Decision: Continue on current tab")
                print(f"   Reasoning: {decision.reasoning}")
                return False  # Don't skip iteration, just continue normally
            
            elif decision.action == TabAction.SPAWN_SUB_AGENT:
                # Phase 3: Spawn sub-agent for another tab
                if not decision.target_tab_id:
                    print("⚠️ Tab decision: SPAWN_SUB_AGENT but no target_tab_id provided")
                    return False
                
                if not self.sub_agent_controller:
                    print("⚠️ Cannot spawn sub-agent: SubAgentController not initialized")
                    return False
                
                policy_allowed, policy_reason = self._can_spawn_sub_agent()
                if not policy_allowed:
                    print(f"🚫 Sub-agent spawn skipped ({policy_reason})")
                    self._log_event(
                        "sub_agent_spawn_blocked",
                        policy=self.sub_agent_policy_level.value,
                        reason=policy_reason,
                        override=self._sub_agent_policy_override.value if self._sub_agent_policy_override else None,
                        target_tab_id=decision.target_tab_id,
                        iteration=self._current_iteration,
                        spawn_count=self.sub_agent_spawn_count
                    )
                    return False
                
                target_tab: Optional[Any] = None
                created_new_tab = False
                
                if decision.target_tab_id:
                    target_tab = self.bot.tab_manager.get_tab_info(decision.target_tab_id)
                    if not target_tab:
                        print(f"⚠️ Tab decision: SPAWN_SUB_AGENT for {decision.target_tab_id} but tab not found")
                        return False
                else:
                    purpose = decision.target_purpose or "Sub-agent task"
                    metadata = {
                        "spawned_by_decision": True,
                        "decision_reasoning": decision.reasoning,
                    }
                    if self._sub_agent_policy_override:
                        metadata["policy_override"] = self._sub_agent_policy_override.value
                    new_tab_id = self.bot.tab_manager.open_new_tab(
                        purpose=purpose,
                        url=decision.target_url,
                        metadata=metadata
                    )
                    if not new_tab_id:
                        print("⚠️ Failed to create new tab for sub-agent.")
                        return False
                    decision.target_tab_id = new_tab_id
                    created_new_tab = True
                    target_tab = self.bot.tab_manager.get_tab_info(new_tab_id)
                    if not target_tab:
                        print(f"⚠️ Failed to retrieve newly created tab {new_tab_id} for sub-agent.")
                        return False
                
                print(f"🤖 Tab Decision: Spawning sub-agent for tab {decision.target_tab_id} ({target_tab.purpose})")
                print(f"   Reasoning: {decision.reasoning}")
                print(f"   Confidence: {decision.confidence:.2f}")
                if created_new_tab:
                    print(f"   🆕 Created new tab {decision.target_tab_id} (URL: {target_tab.url}) for the sub-agent.")
                
                # Extract instruction from reasoning or use a default
                # The LLM should provide instruction in metadata or we derive from reasoning
                instruction = decision.reasoning  # Default: use reasoning as instruction
                if decision.target_tab_id in self.bot.tab_manager.tabs:
                    tab_metadata = self.bot.tab_manager.tabs[decision.target_tab_id].metadata
                    if "sub_agent_instruction" in tab_metadata:
                        instruction = tab_metadata["sub_agent_instruction"]
                    else:
                        tab_metadata["sub_agent_instruction"] = instruction
                
                # Spawn sub-agent
                sub_agent_id = self.sub_agent_controller.spawn_sub_agent(
                    tab_id=decision.target_tab_id,
                    instruction=instruction,
                    metadata={"spawned_by_decision": True, "reasoning": decision.reasoning}
                )
                
                if sub_agent_id:
                    print(f"   ✅ Sub-agent spawned: {sub_agent_id}")
                    self._log_event(
                        "sub_agent_spawned",
                        agent_id=sub_agent_id,
                        instruction=instruction,
                        tab_id=decision.target_tab_id,
                    )
                    self.sub_agent_spawn_count += 1
                    self._spawns_this_iteration += 1
                    # Execute sub-agent immediately
                    result = self.sub_agent_controller.execute_sub_agent(sub_agent_id)
                    if result.get("success"):
                        print("   ✅ Sub-agent completed successfully")
                        self._log_event(
                            "sub_agent_execution",
                            agent_id=sub_agent_id,
                            success=True,
                            status=result.get("status"),
                        )
                    else:
                        print(f"   ⚠️ Sub-agent failed: {result.get('error', 'Unknown error')}")
                        self._log_event(
                            "sub_agent_execution",
                            agent_id=sub_agent_id,
                            success=False,
                            status=result.get("status"),
                            error=result.get("error"),
                        )
                    
                    # Pull results into controller state
                    self._drain_sub_agent_results()
                    
                    # Switch back to main agent's tab
                    if self.agent_context:
                        main_tab = self.bot.tab_manager.get_tab_info(self.agent_context.tab_id)
                        if main_tab:
                            self.bot.tab_manager.switch_to_tab(self.agent_context.tab_id)
                            self.bot.switch_to_page(main_tab.page)
                    
                    return True
                else:
                    print("   ⚠️ Failed to spawn sub-agent")
                    return False
            
            else:
                print(f"⚠️ Unknown tab action: {decision.action}")
                return False
                
        except Exception as e:
            print(f"⚠️ Error executing tab decision: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_sub_agent_results(self) -> List[SubAgentResult]:
        """Return completed sub-agent results for this controller run."""
        return list(self.sub_agent_results)

    def _log_event(self, event_type: str, **data: Any) -> None:
        event = {
            "type": event_type,
            "timestamp": time.time(),
        }
        if data:
            event.update(data)
        self.orchestration_events.append(event)

