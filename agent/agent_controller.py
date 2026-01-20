import time
import threading
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING, Callable
import re
import hashlib

from core.session import BrowserState, Interaction, InteractionType
from models.models import NotebookEntryType
from agent.results import TaskResult
from agent.agent_context import EnvironmentState
from agent.notebook import Notebook
from utils.debug_print import dprint, PrintMode
# Type alias for user question callback (ask: command handler)
# Callback receives: question (str), context (dict) -> returns user's answer (str) or None to skip
UserQuestionCallback = Callable[[str, dict], Optional[str]]
from lib.ai import (
    generate_model,
    ReasoningLevel,
    get_default_agent_model,
    get_default_agent_reasoning_level,
)
from pydantic import BaseModel, ConfigDict, Field

from .interceptor_manager import InterceptorManager, Interceptor, InterceptorMode, InterceptorContext
from agent.task_based_execution import TaskBasedExecution
from core.config import SequentialTaskConfig

if TYPE_CHECKING:
    from .agent_controller import Agent
    from core.browser import Browser

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
        # Completion callback for complete: command
        completion_callback: Optional[Callable[[str], None]] = None,
        # Act function parameter configuration
        act_enable_target_context_guard: bool = True,
        act_enable_modifier: bool = True,
        act_enable_additional_context: bool = True,
        # Overlay inclusion in agent context
        include_overlays_in_agent_context: bool = True,
        # Visible text inclusion in agent context
        include_visible_text_in_agent_context: bool = False,
        # Interaction summarization
        interaction_summary_limit_completion: Optional[int] = None,
        interaction_summary_limit_action: Optional[int] = None,
        # Max actions per plan
        max_actions_per_plan: int = 6,
        # Per-turn load wait
        wait_for_load_before_turn: Optional[bool] = None,
        wait_for_load_state: Optional[str] = None,
        wait_for_load_timeout_ms: Optional[int] = None,
        # Image detail level for vision API
        image_detail: str = "high",
        # Screenshot saving for debugging
        save_screenshots: bool = False,
        screenshot_dir: str = "agent_screenshots",
        # Completion history images
        completion_history_image_limit: int = 3,
        # Task-based execution
        auto_complete_extract_commands: bool = True,
        sequential_task_config: Optional[SequentialTaskConfig] = None,
    ):
        """
        Initialize agent controller.
        
        Args:
            bot: Browser instance to control
            track_ineffective_actions: If True, track and avoid repeating actions that didn't yield page changes.
                                       Default: True (recommended for better performance)
            base_knowledge: Optional list of knowledge rules/instructions that guide the agent's behavior.
                           Example: ["just press enter after you've typed a search term into a search field"]
            allow_partial_completion: If True, allow partial completion of tasks.
            parallel_completion_and_action: Legacy flag (no effect when completion is agent-signaled).
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
            self.event_logger = EventLogger(debug_mode=True, show_overlay_candidates=False)
        self.max_iterations = 50
        self.iteration_delay = 0.5
        self.task_start_url: Optional[str] = None
        self.task_start_time: Optional[float] = None
        self.allow_non_clickable_clicks = True  # Allow clicking non-clickable elements (configurable)
        self.track_ineffective_actions = track_ineffective_actions  # Track actions that didn't yield page changes
        self.completion_history_image_limit = completion_history_image_limit
        self.detect_ineffective_actions = track_ineffective_actions
        self.base_knowledge = base_knowledge or []  # Base knowledge rules that guide agent behavior
        self.failed_actions: List[str] = []  # Track actions that failed AND didn't yield any change
        self.ineffective_actions: List[str] = []  # Track actions that succeeded BUT didn't yield any change
        self._consecutive_page_changes: int = 0  # Track consecutive page state changes for phase-out
        self.notebook: Notebook = Notebook()
        self.orchestration_events: List[Dict[str, Any]] = []  # Track orchestration events for reporting
        self._task_tracker: Dict[str, Dict[str, Any]] = {}
        self.allow_partial_completion = allow_partial_completion
        self.parallel_completion_and_action = parallel_completion_and_action
        self.completion_mode = "external_only"
        # Completion / evaluation behavior
        self.show_completion_reasoning_every_iteration = show_completion_reasoning_every_iteration
        self.strict_mode = strict_mode
        self.clarification_callback = clarification_callback
        self.max_clarification_rounds = max_clarification_rounds
        self.interaction_summary_limit_completion = interaction_summary_limit_completion
        self.interaction_summary_limit_action = interaction_summary_limit_action
        self.max_actions_per_plan = max_actions_per_plan
        self.wait_for_load_before_turn = (
            wait_for_load_before_turn
            if wait_for_load_before_turn is not None
            else getattr(bot, "wait_for_load_before_turn", False)
        )
        self.wait_for_load_state = (
            wait_for_load_state
            if wait_for_load_state is not None
            else getattr(bot, "wait_for_load_state", "networkidle")
        )
        self.wait_for_load_timeout_ms = (
            wait_for_load_timeout_ms
            if wait_for_load_timeout_ms is not None
            else getattr(bot, "wait_for_load_timeout_ms", 30000)
        )
        self._user_inputs: List[Dict[str, Any]] = []
        self._temp_user_inputs: List[Dict[str, Any]] = []  # Single-use suggestions
        self._original_user_prompt: str = ""
        self._last_screenshot_hash: Optional[str] = None  # Store screenshot hash for phase-out tracking

        # Store user question callback for ask: command
        self.user_question_callback = user_question_callback
        self._last_ask_iteration: int = -2  # Track last iteration where ask: was answered (to prevent consecutive asks)

        # Store completion callback for complete: command
        self.completion_callback = completion_callback

        # Store act function parameter configuration
        # These flags control which parameters are passed to bot.act() during execution
        self.act_enable_target_context_guard = act_enable_target_context_guard
        self.act_enable_modifier = act_enable_modifier
        self.act_enable_additional_context = act_enable_additional_context
        
        # Store overlay inclusion configuration
        # Controls whether overlay element data is included in agent's context for action determination
        self.include_overlays_in_agent_context = include_overlays_in_agent_context

        # Store visible text inclusion configuration
        # Controls whether visible text is included in agent's context for action determination
        self.include_visible_text_in_agent_context = include_visible_text_in_agent_context

        # Store image detail level for vision API
        self.image_detail = image_detail

        # Store screenshot saving configuration
        self.save_screenshots = save_screenshots
        self.screenshot_dir = screenshot_dir
        self._screenshot_counter = 0  # Counter for naming screenshots

        self.agent_model_name: str = getattr(bot, "agent_model_name", get_default_agent_model())
        agent_reasoning = getattr(bot, "agent_reasoning_level", None)
        if agent_reasoning is None:
            agent_reasoning = ReasoningLevel.coerce(get_default_agent_reasoning_level())
        else:
            agent_reasoning = ReasoningLevel.coerce(agent_reasoning)
        self.agent_reasoning_level: ReasoningLevel = agent_reasoning
        
        # Pause functionality: Allows pausing agent execution between actions
        self._paused = False
        self._pause_lock = threading.Lock()  # Thread-safe access to pause state
        self._pause_event = threading.Event()  # Event to block execution when paused
        self._pause_event.set()  # Initially not paused (event is set = not blocking)
        self._pause_message = "Paused"

        self.interceptor_manager = InterceptorManager(self.bot)
        self.interceptor_stack: List[Dict[str, Any]] = []  # Stack of active interceptors

        self.task_execution = TaskBasedExecution(
            self,
            sequential_task_config=sequential_task_config,
            auto_complete_extract_commands=auto_complete_extract_commands,
        )

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
        try:
            self.event_logger.agent_pause(message)
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
            try:
                self.event_logger.agent_resume()
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
        dprint(f"\n⏸️  {message}")
        if action_desc:
            dprint(f"   Waiting before: {action_desc}")
        try:
            current_url = self.bot.page.url if self.bot.page else 'N/A'
            dprint(f"   URL: {current_url}")
        except Exception:
            pass
        
        # Wait until resume() is called (this blocks the execution thread)
        self._pause_event.wait()

    def _maybe_wait_for_turn_load(self, reason: str = "turn") -> None:
        if not self.wait_for_load_before_turn:
            return
        try:
            if getattr(self.bot, "page", None) and not self.bot.page.is_closed():
                try:
                    self.event_logger.system_debug(
                        f"Waiting for page load before {reason} (state={self.wait_for_load_state})"
                    )
                except Exception:
                    pass
                self.bot.wait_for_load(
                    timeout=self.wait_for_load_timeout_ms,
                    state=self.wait_for_load_state,
                )
        except Exception:
            # Best-effort wait; do not block on load wait errors.
            pass
    
    def run_execute_task(self, user_prompt: str) -> TaskResult:
        """
        Execute a task autonomously.
        
        Args:
            user_prompt: User's high-level request
            
        Returns:
            TaskResult indicating success or failure
        """
        # Reset per-run state
        try:
            self.event_logger.system_info("Active role: Main agent")
        except Exception:
            pass

        self.orchestration_events = []
        self._task_tracker = {}
        self._original_user_prompt = user_prompt
        # Reset page change counter for new task
        self._consecutive_page_changes = 0
        # Reset screenshot hash tracking
        self._last_screenshot_hash = None

        self.event_logger.agent_start(user_prompt, agent_type="Main agent", max_iterations=self.max_iterations)

        self._log_event(
            "agent_start",
            agent_type="Main agent",
            prompt=user_prompt,
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
        
        if not self.bot.started:
            self.event_logger.system_error("Bot not started. Call bot.start() first.")
            try:
                self.event_logger.agent_error("Bot not started")
            except Exception:
                pass
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
            try:
                self.event_logger.agent_error("Page is blank")
            except Exception:
                pass
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

        # Task-based execution (default)
        result = self.task_execution.run(user_prompt)

        if self.bot.execution_timer.task_start_time is not None:
            self.bot.execution_timer.end_task()
            self.bot.execution_timer.log_summary(self.event_logger)

        if result.success:
            try:
                self.event_logger.agent_completed(result.reasoning)
            except Exception:
                pass
        else:
            self.event_logger.agent_complete(
                success=False,
                reasoning=result.reasoning,
                confidence=result.confidence,
            )

        self._log_event(
            "agent_complete",
            status="completed" if result.success else "failed",
            reason=result.reasoning,
            confidence=result.confidence,
        )

        return result

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
                dprint("📸 Using full-page screenshot for exploration mode")
            else:
                # Capture viewport screenshot (agent needs this to see what's visible)
                snapshot.screenshot = self.bot.page.screenshot(full_page=False)
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
        if hasattr(self.bot, "page"):
            try:
                setattr(self.bot.page, "_last_screenshot_hash", screenshot_hash)
            except Exception:
                pass
        
        return snapshot
    
    @staticmethod
    def _normalize_task_name(name: str) -> str:
        return " ".join(name.lower().strip().split())

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
        dprint(f"📝 Captured user input from defer: {response}")
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
            dprint("⚠️ Agent wants to ask a question but no callback configured")
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
                dprint(f"📝 User guidance added for next command: {answer}")
                self._log_event(
                    "ask_command_answered",
                    question=question,
                    answer=answer,
                    iteration=iteration,
                )
                return True
            else:
                dprint("⏭️ User skipped the question")
                self._log_event(
                    "ask_command_skipped",
                    question=question,
                    iteration=iteration,
                )
                return False
                
        except Exception as e:
            dprint(f"⚠️ Error in ask callback: {e}")
            self._log_event(
                "ask_command_error",
                question=question,
                error=str(e),
                iteration=iteration,
            )
            return False

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
            "command": action_command,
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
                    dprint(f"   🎯 Matched target element (score: {best_score}): {best_match.get('tagName', 'unknown')} - {best_match.get('name', '') or best_match.get('id', '') or best_match.get('textContent', '')[:50]}")
            
            # If target element found, compare position
            if target_element:
                target_y = target_element['centerY']  # Use center Y for better accuracy
                
                dprint(f"   📍 Target element found at absolute Y: {target_y}")
                dprint(f"   📍 Viewport bounds: Y={viewport_top} to Y={viewport_bottom}")
                
                if target_y < viewport_top:
                    dprint("   ⬆️ Target is above viewport → scroll: up")
                    return "scroll: up"
                elif target_y > viewport_bottom:
                    dprint("   ⬇️ Target is below viewport → scroll: down")
                    return "scroll: down"
                else:
                    # Target is in viewport - shouldn't happen in exploration mode
                    dprint("   ✅ Target is in viewport (should not be in exploration mode)")
                    return "scroll: down"  # Default
            
            # Fallback: if we can't find target, use simple heuristic
            dprint("   ⚠️ Target element not found, using heuristic")
            if current_scroll_y > 0:
                return "scroll: up"  # Already scrolled, try up first
            else:
                return "scroll: down"  # At top, scroll down
                
        except Exception as e:
            dprint(f"⚠️ Error determining scroll direction: {e}")
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

            return {
                "visible_elements": visible_elements or [],
                "overlay_elements": overlay_elements
            }

        except Exception:
            return {
                "visible_elements": [],
                "overlay_elements": []
            }

    def _build_evidence(self, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Construct evidence dictionaries including extracted data.
        """
        evidence: Dict[str, Any] = {}
        if base:
            evidence.update(base)
        if self.notebook and "notebook" not in evidence:
            evidence["notebook"] = (
                self.notebook.to_list()
                if hasattr(self.notebook, "to_list")
                else list(self.notebook)
            )
        orchestration = {
            "events": list(self.orchestration_events),
        }
        evidence["orchestration"] = orchestration
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

    def _format_interaction_history_detailed(self, interaction_history: List[Interaction]) -> str:
        """
        Format interaction history with full details about WHAT was interacted with.

        Returns a detailed string representation showing:
        - What action was performed
        - What element/target was interacted with
        - What text was typed
        - What was extracted
        - Success/failure status
        """
        if not interaction_history:
            return "No actions performed yet."

        lines = []
        for i, interaction in enumerate(interaction_history, 1):
            action_type = interaction.interaction_type.value
            parts = [f"{i}. {action_type.upper()}"]

            # Add details based on interaction type
            if interaction.interaction_type == InteractionType.CLICK:
                # Show what was clicked with as much detail as possible
                click_target = None
                if interaction.reasoning:
                    # Extract target from reasoning if available
                    # e.g., "The user wants to click the 5th article link"
                    parts.append(f"reason: {interaction.reasoning[:150]}")
                    click_target = "described in reasoning"
                elif interaction.target_element_info:
                    elem_desc = interaction.target_element_info.get('description', '')
                    if elem_desc:
                        parts.append(f"target: {elem_desc}")
                        click_target = elem_desc
                    elif 'overlay_index' in interaction.target_element_info:
                        overlay_idx = interaction.target_element_info['overlay_index']
                        parts.append(f"target: element #{overlay_idx}")
                        click_target = f"element #{overlay_idx}"

                if interaction.coordinates and not click_target:
                    parts.append(f"at coordinates ({interaction.coordinates[0]}, {interaction.coordinates[1]})")
                elif interaction.coordinates and click_target:
                    parts.append(f"at ({interaction.coordinates[0]}, {interaction.coordinates[1]})")

            elif interaction.interaction_type == InteractionType.TYPE:
                if interaction.text_input:
                    parts.append(f"text: '{interaction.text_input}'")
                if interaction.target_element_info:
                    elem_desc = interaction.target_element_info.get('description', '')
                    if elem_desc:
                        parts.append(f"into: {elem_desc}")

            elif interaction.interaction_type == InteractionType.PRESS:
                if interaction.keys_pressed:
                    parts.append(f"key: {interaction.keys_pressed}")

            elif interaction.interaction_type == InteractionType.SCROLL:
                if interaction.scroll_direction:
                    parts.append(f"direction: {interaction.scroll_direction}")
                if interaction.scroll_axis:
                    parts.append(f"axis: {interaction.scroll_axis}")

            elif interaction.interaction_type == InteractionType.NAVIGATION:
                if interaction.navigation_url:
                    parts.append(f"to: {interaction.navigation_url}")
                elif interaction.target_element_info:
                    direction = interaction.target_element_info.get('direction', '')
                    if direction:
                        parts.append(f"direction: {direction}")

            elif interaction.interaction_type == InteractionType.EXTRACT:
                if interaction.extraction_prompt:
                    parts.append(f"prompt: {interaction.extraction_prompt}")
                if interaction.extracted_data:
                    # Show summary of extracted data
                    if isinstance(interaction.extracted_data, dict):
                        if 'items' in interaction.extracted_data:
                            item_count = len(interaction.extracted_data.get('items', []))
                            parts.append(f"extracted {item_count} items")
                        else:
                            parts.append(f"extracted data with {len(interaction.extracted_data)} fields")

            elif interaction.interaction_type == InteractionType.DEFER:
                if interaction.text_input:
                    parts.append(f"reason: {interaction.text_input}")

            # Add reasoning if available (but not if already included above in click target)
            if interaction.reasoning and interaction.interaction_type != InteractionType.CLICK:
                parts.append(f"reason: {interaction.reasoning[:100]}")

            # Add success status
            if not interaction.success:
                status = "❌ FAILED"
                if interaction.error_message:
                    status += f": {interaction.error_message}"
                parts.append(status)
            else:
                parts.append("✓ SUCCESS")

            lines.append(" | ".join(parts))

        return "\n".join(lines)

    def _check_completion_from_history(
        self,
        user_prompt: str,
        interaction_history: List[Interaction],
        notebook: Notebook
    ) -> Optional[str]:
        """
        Check if task is complete based ONLY on interaction history.

        This method evaluates completion WITHOUT looking at the current page state,
        current screenshot, or any observations. It only considers what actions have
        been performed and what data has been extracted.

        Args:
            user_prompt: The user's original goal
            interaction_history: List of all actions performed
            notebook: Extracted data collected during execution

        Returns:
            str: Completion reasoning if task is complete, None if incomplete
        """
        # Define response model
        class CompletionCheck(BaseModel):
            model_config = ConfigDict(extra="forbid")
            is_complete: bool = Field(
                description="True if the user's goal has been accomplished based on the actions performed, False otherwise"
            )
            reasoning: str = Field(
                description="Explain why the task is complete or what is still missing. Be specific about what was accomplished or what remains to be done."
            )

        # Format the interaction history
        history_text = self._format_interaction_history_detailed(interaction_history)

        # Debug: Log what is being evaluated
        try:
            self.event_logger.system_debug(f"[Completion Check] Evaluating task: {user_prompt}")
            self.event_logger.system_debug(f"[Completion Check] Formatted history:\n{history_text}")
        except Exception:
            pass

        # Collect recent screenshots (oldest → newest) for completion context
        recent_images: List[bytes] = []
        recent_image_captions: List[str] = []
        max_images = getattr(self, "completion_history_image_limit", 0) or 0
        if max_images > 0 and interaction_history:
            for interaction in reversed(interaction_history):
                state = getattr(interaction, "before_state", None)
                screenshot = getattr(state, "screenshot", None) if state else None
                if screenshot:
                    target_desc = ""
                    if interaction.target_element_info:
                        target_desc = interaction.target_element_info.get("description") or ""
                        if not target_desc and interaction.target_element_info.get("overlay_index") is not None:
                            target_desc = f"element #{interaction.target_element_info.get('overlay_index')}"
                    reason_snippet = ""
                    if interaction.reasoning:
                        reason_snippet = interaction.reasoning[:80]
                    caption_parts = [
                        f"Before {interaction.interaction_type.value.upper()}",
                        f"url: {state.url or 'unknown'}",
                    ]
                    if target_desc:
                        caption_parts.append(f"target: {target_desc}")
                    if reason_snippet:
                        caption_parts.append(f"reason: {reason_snippet}")
                    caption_parts.append("result: " + ("success" if interaction.success else "failed"))
                    recent_images.append(screenshot)
                    recent_image_captions.append(" | ".join(caption_parts))
                if len(recent_images) >= max_images:
                    break
            if recent_images:
                recent_images.reverse()
                recent_image_captions.reverse()

        # Format notebook entries
        notebook_text = ""
        notebook_entries = notebook.to_list() if hasattr(notebook, 'to_list') else (list(notebook) if notebook else [])
        if notebook_entries:
            notebook_lines = ["EXTRACTED DATA:"]
            for i, entry in enumerate(notebook_entries, 1):
                prompt = entry.get("prompt", "unknown")
                data = entry.get("data", {})

                if isinstance(data, dict) and "items" in data:
                    items = data["items"]
                    notebook_lines.append(f"{i}. [{prompt}] - {len(items)} items extracted")
                    # Show first 2 samples
                    for j, item in enumerate(items[:2]):
                        item_str = str(item)
                        if len(item_str) > 150:
                            item_str = item_str[:150] + "..."
                        notebook_lines.append(f"   Sample {j+1}: {item_str}")
                    if len(items) > 2:
                        notebook_lines.append(f"   ... and {len(items) - 2} more items")
                else:
                    data_str = str(data)
                    if len(data_str) > 200:
                        data_str = data_str[:200] + "..."
                    notebook_lines.append(f"{i}. [{prompt}] - {data_str}")

            notebook_text = "\n".join(notebook_lines)
        else:
            notebook_text = "EXTRACTED DATA:\nNo data extracted yet."

        # Build the prompt
        system_prompt = """You are evaluating whether a web automation task is complete based ONLY on the history of actions performed.

CRITICAL RULES:
1. You MUST NOT consider the current page state, current screenshot, or current observations
2. You can ONLY use the ACTIONS PERFORMED and EXTRACTED DATA to make your decision
3. Check if the SPECIFIC action requested in the goal was performed
4. Look at the "target" and "reason" fields in the action history to understand WHAT was clicked/typed/etc
5. If the goal asks for action X on target Y, verify that BOTH the action type AND target match

ACTION TYPE MAPPING (Natural Language → Actual Command):
Map natural language task descriptions to the actual action commands that accomplish them:

INTERACTION ACTIONS (all map to CLICK command):
- "tap", "press", "hit", "select", "choose", "open" → CLICK
- "click on", "click the", "click" → CLICK

INPUT ACTIONS (all map to TYPE command):
- "enter", "input", "fill", "type", "write" + text/value → TYPE
- "fill in", "fill out", "enter text", "type in" → TYPE

DATA RETRIEVAL ACTIONS (all map to EXTRACT command):
- "extract", "get", "retrieve", "fetch", "collect", "scrape" → EXTRACT
- "list", "summarize", "explain", "show", "find", "obtain" → EXTRACT
- "copy", "save", "record", "note", "capture" → EXTRACT
- For data retrieval tasks, focus on WHETHER the extraction happened, NOT the specific wording
- The extraction prompt may differ from the task wording - that's fine as long as the intent matches

NAVIGATION ACTIONS:
- "go to", "navigate to", "visit", "open" + URL → NAVIGATE
- "go back", "return", "previous page" → BACK
- "go forward", "next page" → FORWARD

COMPLETION CRITERIA:
- Map the task's natural language to the corresponding action command (see mapping above)
- Match the TARGET (what was clicked, where text was typed, what was extracted)
- Use the "reason" field to understand the intent behind the action
- If the reason explicitly mentions fulfilling the goal, that's strong evidence of completion
- For EXTRACT actions: Don't worry about exact wording - if data was extracted that matches the intent, it's complete

Examples:
- Goal: "Tap the 5th article link"
  History shows: "CLICK | target: link GLM-4.7-Flash | reason: The user wants to click the 5th article link | ✓ SUCCESS"
  Decision: COMPLETE ("tap" maps to CLICK, and reason confirms it's the 5th article)

- Goal: "Enter 'john@example.com' into the email field"
  History shows: "TYPE | text: 'john@example.com' | target: email input | ✓ SUCCESS"
  Decision: COMPLETE ("enter" maps to TYPE, correct text and target)

- Goal: "Summarize the product features"
  History shows: "EXTRACT | prompt: product features | extracted 1 item | ✓ SUCCESS"
  Decision: COMPLETE ("summarize" maps to EXTRACT - the data was retrieved, summary is just presentation)

- Goal: "List all the job titles"
  History shows: "EXTRACT | prompt: job titles | extracted 5 items | ✓ SUCCESS"
  Decision: COMPLETE ("list" maps to EXTRACT - the data was extracted, listing is just formatting)

- Goal: "Get the price from the 2nd listing"
  History shows: "EXTRACT | prompt: price from 2nd listing | extracted 1 item | ✓ SUCCESS"
  Decision: COMPLETE ("get" maps to EXTRACT)

- Goal: "Click the 5th article link"
  History shows: "CLICK | target: element #40 | ✓ SUCCESS"
  Decision: INCOMPLETE (click happened but no confirmation it's the 5th article)

- Goal: "Click the Apply button"
  History shows: "CLICK | target: Apply button | ✓ SUCCESS"
  Decision: COMPLETE (correct action and correct target)

- Goal: "Type 'John Doe' in the name field"
  History shows: "TYPE | text: 'John Doe' | target: name input field | ✓ SUCCESS"
  Decision: COMPLETE (correct text typed into correct field)

- Goal: "Click the login button"
  History shows: "SCROLL | direction: down | ✓ SUCCESS"
  Decision: INCOMPLETE (wrong action type - scroll vs click)

- Goal: "Press the submit button"
  History shows: "CLICK | target: cancel button | ✓ SUCCESS"
  Decision: INCOMPLETE ("press" maps to CLICK but wrong target - cancel vs submit)

- Goal: "Extract products from the first 3 pages"
  History shows: Only 1 extraction action
  Decision: INCOMPLETE (need multiple extractions for multiple pages)
"""

        screenshots_text = ""
        if recent_image_captions:
            screenshots_text = "SCREENSHOTS (oldest→newest):\n" + "\n".join(
                f"{i+1}. {caption}" for i, caption in enumerate(recent_image_captions)
            )

        user_prompt_text = f"""Evaluate if the following task is complete based ONLY on the actions performed:

USER GOAL:
{user_prompt}

ACTIONS PERFORMED:
{history_text}

{notebook_text}

{screenshots_text if screenshots_text else "SCREENSHOTS: None available"}

EVALUATION INSTRUCTIONS:
- Review the USER GOAL carefully
- Check if the ACTIONS PERFORMED accomplish that specific goal
- Check if the EXTRACTED DATA contains what the user requested
- Remember: You can ONLY use the history above, NOT the current page state
- Be strict: only mark complete if the goal is clearly satisfied
- Keep the reasoning under 60 words

Is the task complete based on the actions performed?"""

        try:
            result = generate_model(
                prompt=user_prompt_text,
                model_object_type=CompletionCheck,
                system_prompt=system_prompt,
                multi_image=recent_images if recent_images else None,
                image_detail=self.image_detail,
                model=self.agent_model_name,
                reasoning_level=ReasoningLevel.LOW,
            )

            if isinstance(result, CompletionCheck):
                if result.is_complete:
                    try:
                        self.event_logger.system_debug(f"[Completion Check] LLM says COMPLETE: {result.reasoning}")
                    except Exception:
                        pass
                    return result.reasoning
                else:
                    try:
                        self.event_logger.system_debug(f"[Completion Check] LLM says INCOMPLETE: {result.reasoning}")
                    except Exception:
                        pass
                    return None
            else:
                # Failed to get structured response
                try:
                    self.event_logger.system_debug(f"[Completion Check] Failed to get structured response: {result}")
                except Exception:
                    pass
                return None

        except Exception as e:
            dprint(f"⚠️ Error checking completion from history: {e}")
            return None

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
                    dprint(f"   🔄 Detected UI state changes: {ui_changes}")
                return True
        except Exception as e:
            if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
                dprint(f"   ⚠️ Error checking UI state changes: {e}")

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
    
    def _log_event(self, event_type: str, **data: Any) -> None:
        event = {
            "type": event_type,
            "timestamp": time.time(),
        }
        if data:
            event.update(data)
        self.orchestration_events.append(event)
