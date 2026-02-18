"""Agent controller — mission execution with inline loops.

Architecture overview:
- Mission goes directly to the execution loop (no planner).
- Agent uses browser actions to accomplish the mission.
- When repetition is needed, the agent declares a loop inline via think(start_loop).
- Loop rounds are advanced with think(advance) and exited with think(end_loop) or think(done).
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable, Union, Type, TYPE_CHECKING
import hashlib

from browser.dom import build_page_elements
from browser.annotate import build_element_index, build_crop_gallery
from core.browser import ExecutionTimer
from core.executor.base import Executor
from agent.memory import InteractionType, MemoryState, NarrativeMemory
from models import PageElements, PageInfo
from models.models import FailedAction
from agent.results import MissionResult
from agent.agent_context import EnvironmentState
from agent.notebook import Notebook
from agent.action_planner import strip_targeting_data
from agent.prompts import (
    DecisionContext,
)
from utils.debug_print import dprint
from lib.ai import (
    ReasoningLevel,
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
class ExecutionState:
    """Mutable execution state for the mission loop."""
    total_actions: int = 0
    actions_since_progress: int = 0
    browser_actions_since_progress: int = 0
    checkpoint_pending: bool = False
    last_action_summary: Optional[str] = None
    failed_elements: List[FailedAction] = field(default_factory=list)
    validation_failures: int = 0
    # Loop state
    in_loop: bool = False
    loop_count: Optional[int] = None
    loop_round: int = 0
    loop_description: str = ""
    # Recent action log (compact summaries, last ~10)
    recent_actions: List[str] = field(default_factory=list)


@dataclass
class ScreenshotPreparation:
    """Result of _prepare_screenshot_for_mode()."""
    screenshot_bytes: bytes
    gallery_images: Optional[List[bytes]] = None
    element_index_text: Optional[str] = None


"""
Agent Controller - Mission Execution

Runs missions through the execution loop with inline loops for repetition.
"""

class Agent:
    """
    Agent controller running mission execution.
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
        self.mission_start_url: Optional[str] = None
        self.mission_start_time: Optional[float] = None
        self.base_knowledge = base_knowledge or []  # Base knowledge rules that guide agent behavior
        self.notebook: Notebook = Notebook()
        self._mission_tracker: Dict[str, Dict[str, Any]] = {}
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

        # Execution timer for tracking mission, iteration, and action timings
        self.execution_timer = ExecutionTimer()
        
        self.show_llm_costs = config.logging.show_llm_costs
        _show_overlay_candidates = config.logging.show_overlay_candidates
        self.save_screenshots = config.logging.save_screenshots
        self.screenshot_dir = config.logging.screenshot_dir

        self._current_iteration = 0

        # Initialize execution system
        self.auto_complete_extract_commands = self.config.execution.auto_complete_extract_commands
        self._extraction_model_cache: Dict[tuple[str, ...], Type[BaseModel]] = {}

    def __enter__(self) -> 'Agent':
        """
        Context manager entry point. Automatically calls start().
        
        Example:
            >>> with Agent(browser, config) as agent:
            ...     agent.run_mission("Click the button")
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
        
        # Run the mission
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
        Execute a mission directly.

        The agent executes browser actions to accomplish the mission.
        When repetition is needed, it declares loops inline via think(start_loop).

        Args:
            user_mission: User's high-level request
        Returns:
            MissionResult indicating success or failure
        """
        self._mission_tracker = {}
        self._original_user_mission = user_mission
        self._current_iteration = 0
        self.event_logger.agent_start(user_mission)

        # Initialize tracking
        try:
            self.mission_start_url = self.browser.page.url
        except Exception:
            self.mission_start_url = "unknown"
        self.mission_start_time = time.time()

        self.execution_timer.start_mission()

        if self.base_knowledge:
            self.memory_store.set_base_knowledge(self.base_knowledge)
        self.memory_store.start_mission(user_mission)

        # Check if starting from a blank page
        if self.browser.page.url.startswith("about:blank"):
            self.event_logger.agent_error("Page is on initial blank page.")
            if self.execution_timer.mission_start_time is not None:
                self.execution_timer.end_mission()
            self.mission_result.success = False
            self.event_logger.agent_complete(success=False, reasoning="Page is blank")
            return self.mission_result

        # Execute the mission directly
        self.mission_result = self._run_execution_loop(
            user_mission,
            start_in_checkpoint=True,
        )

        if self.execution_timer.mission_start_time is not None:
            self.execution_timer.end_mission()

        return self.mission_result




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

    def _prepare_screenshot_for_mode(
        self,
        snapshot,
        detected_elements: PageElements,
        page_info: PageInfo,
    ) -> ScreenshotPreparation:
        """Prepare screenshot, element index, and crop gallery for the LLM.

        Returns a ScreenshotPreparation with the clean screenshot bytes,
        optional gallery images, and element index text.
        """
        screenshot = snapshot.screenshot

        if not screenshot:
            return ScreenshotPreparation(screenshot_bytes=screenshot)

        result = build_element_index(
            detected_elements.elements,
            max_elements=self.config.elements.max_index_elements,
            viewport_only=True,
        )
        gallery_images = None
        if result.text_poor_elements:
            crops_per = self.config.elements.crops_per_gallery
            gallery_images = build_crop_gallery(
                screenshot,
                result.text_poor_elements,
                crops_per_page=crops_per,
            )
            dprint(f"📸 Built {len(gallery_images)} gallery page(s) for {len(result.text_poor_elements)} text-poor elements")

        # Save debug screenshots
        if self.save_screenshots:
            try:
                from pathlib import Path
                from datetime import datetime
                ss_dir = Path(self.screenshot_dir)
                ss_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%H%M%S")
                clean_path = str(ss_dir / f"iter{self._current_iteration:03d}_clean_{ts}.png")
                with open(clean_path, "wb") as f:
                    f.write(screenshot)
                if gallery_images:
                    for gi_idx, gi_bytes in enumerate(gallery_images):
                        gp = str(ss_dir / f"iter{self._current_iteration:03d}_gallery{gi_idx + 1}_{ts}.png")
                        with open(gp, "wb") as f:
                            f.write(gi_bytes)
                dprint(f"📸 Saved clean + {len(gallery_images or [])} gallery screenshot(s)")
            except Exception as e:
                dprint(f"⚠️ Could not save debug screenshots: {e}")

        return ScreenshotPreparation(
            screenshot_bytes=screenshot,
            gallery_images=gallery_images,
            element_index_text=result.index_text,
        )

    def _run_execution_loop(
        self,
        mission: str,
        *,
        start_in_checkpoint: bool = False,
    ) -> MissionResult:
        """
        Unified mission execution loop.

        The agent uses browser actions to accomplish the mission directly.
        When repetition is needed, the agent declares loops inline via think(start_loop).

        Args:
            mission: The mission string to execute
            start_in_checkpoint: Whether to begin in checkpoint mode

        Returns:
            MissionResult with success/failure and reasoning
        """
        from agent.action_planner import ActionPlanner

        max_actions = self.config.execution.max_actions_per_mission

        state = ExecutionState(
            checkpoint_pending=bool(start_in_checkpoint),
        )

        self.memory_store.start_mission(mission)

        def _append_recent_action(summary: str) -> None:
            """Append a compact action summary to recent_actions, capped at 10."""
            state.recent_actions.append(summary)
            if len(state.recent_actions) > 10:
                state.recent_actions.pop(0)

        def _exit_loop() -> None:
            """Clean up loop state and DOM markers."""
            state.in_loop = False
            state.loop_count = None
            state.loop_round = 0
            state.loop_description = ""
            # Clean up DOM done markers
            try:
                self.action_executor.clear_done_markers()
            except Exception:
                pass

        while state.total_actions < max_actions:
            state.total_actions += 1
            self._current_iteration += 1
            self.event_logger.iteration_start(
                iteration=self._current_iteration,
                max_iterations=max_actions,
            )

            # Capture current state
            try:
                snapshot = self._capture_snapshot(full_page=False)
                page_info = self.page_utils.get_page_info()
                detected_elements = build_page_elements(self.browser.page, page_info)
            except Exception as e:
                return MissionResult(
                    success=False,

                    reasoning=f"Failed to capture state: {str(e)}",
                )

            # Prepare screenshot + supporting data for the LLM
            prep = self._prepare_screenshot_for_mode(snapshot, detected_elements, page_info)
            annotated_screenshot_bytes = prep.screenshot_bytes
            element_index_text = prep.element_index_text
            gallery_images = prep.gallery_images

            # Build environment state
            memory_recent = self.memory_store.get_recent(20)
            recent_executed_ids = self.memory_store.get_recent_executed_action_ids(n=20)
            recent_reflection_ids = self.memory_store.get_recent_reflection_ids(n=20)
            recommended_step, recommended_step_source_id = self._get_latest_recommended_next_step()

            decision_context = DecisionContext(
                action_iteration=state.total_actions,
                mission=mission,
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
                user_prompt=mission,
                mission_start_url=self.mission_start_url,
                mission_start_time=self.mission_start_time,
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

            # Create action planner
            active_strategy = self.memory_store.get_latest_strategy() or None
            action_planner = ActionPlanner(
                mission,
                self.memory_store,
                base_knowledge=self.base_knowledge,
                model_name=self.agent_model_name,
                reasoning_level=self.agent_reasoning_level,
                image_detail=self.config.model.image_detail,
                max_actions_per_plan=self.config.execution.max_actions_per_plan,
                checkpoint_mode=state.checkpoint_pending,
                active_strategy=active_strategy,
                last_action_summary=state.last_action_summary,
                tab_bar=tab_bar,
                dialog_notice=dialog_notice,
                tab_events=tab_events,
                dialog_pending=dialog_pending,
                recommended_next_step=recommended_step,
                recommended_next_step_source_id=recommended_step_source_id,
                decision_context=decision_context,
                element_index_text=element_index_text,
                gallery_images=None if state.checkpoint_pending else gallery_images,
                current_iteration=self._current_iteration,
                browser_actions_in_round=state.browser_actions_since_progress,
                # Loop state
                in_loop=state.in_loop,
                loop_round=state.loop_round,
                loop_count=state.loop_count,
                loop_description=state.loop_description,
                recent_actions=state.recent_actions,
            )

            # Generate next actions
            try:
                actions_list, error = action_planner.get_next_actions_with_function_calling(
                    environment_state=environment_state,
                    screenshot=annotated_screenshot_bytes,
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
                            f"Action validation issue: {error or 'No action generated'}. Retrying."
                        )
                        state.checkpoint_pending = False
                        continue
                    return MissionResult(
                        success=False,
                        reasoning=(
                            "Repeated action validation failures: "
                            f"{error or 'No action generated'}"
                        ),
                    )
                state.validation_failures = 0

                # Execute each action
                for action_step in actions_list:
                    function_name = (getattr(action_step, "function_name", None) or "").strip()
                    action_args = getattr(action_step, "function_arguments", {}) or {}
                    current_action = getattr(action_step, "action", "") or function_name
                    reasoning = action_args.get("reasoning", "")
                    self.event_logger.action_determined(
                        action=current_action,
                        reasoning=reasoning,
                    )

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
                        recommended_next_step_arg = str(action_args.get("recommended_next_step", "")).strip()

                        if think_next_action == "start_loop":
                            # Enter loop mode
                            loop_count_raw = action_args.get("loop_count")
                            loop_desc = str(action_args.get("loop_description", "")).strip()
                            try:
                                loop_count = int(loop_count_raw)
                            except (TypeError, ValueError):
                                loop_count = 1

                            state.in_loop = True
                            state.loop_count = loop_count
                            state.loop_round = 2  # Round 1 was the pre-loop action
                            state.loop_description = loop_desc or think_reasoning
                            state.checkpoint_pending = False
                            state.browser_actions_since_progress = 0

                            state.last_action_summary = (
                                f"Loop started: {loop_desc or think_reasoning} "
                                f"(round 2 of {loop_count})"
                            )
                            _append_recent_action(f"[LOOP START] {loop_desc} — {loop_count} total rounds")
                            self.event_logger.system_info(
                                f"⟳ Loop started: \"{loop_desc}\" — round 2 of {loop_count}"
                            )

                        elif think_next_action == "advance":
                            # Advance loop round
                            if not state.in_loop:
                                state.last_action_summary = "advance ignored — not in a loop"
                                state.checkpoint_pending = False
                            elif state.browser_actions_since_progress == 0:
                                state.last_action_summary = (
                                    "advance BLOCKED: No browser actions since last advance. "
                                    "Do a browser action first."
                                )
                                state.checkpoint_pending = False
                            else:
                                state.loop_round += 1
                                state.browser_actions_since_progress = 0
                                state.checkpoint_pending = False

                                # Check if loop is complete
                                if state.loop_count and state.loop_round > state.loop_count:
                                    self.event_logger.system_info(
                                        f"⟳ Loop complete: all {state.loop_count} rounds done"
                                    )
                                    state.last_action_summary = (
                                        f"Loop complete — all {state.loop_count} rounds done"
                                    )
                                    _append_recent_action(f"[LOOP COMPLETE] {state.loop_count} rounds done")
                                    _exit_loop()
                                else:
                                    remaining = (
                                        state.loop_count - state.loop_round + 1
                                        if state.loop_count
                                        else "?"
                                    )
                                    state.last_action_summary = (
                                        f"Advanced to loop round {state.loop_round} of {state.loop_count} "
                                        f"({remaining} remaining)"
                                    )
                                    _append_recent_action(
                                        f"[ADVANCE] Round {state.loop_round} of {state.loop_count}"
                                    )
                                    self.event_logger.system_info(
                                        f"⟳ Loop round {state.loop_round} of {state.loop_count}"
                                    )

                        elif think_next_action == "end_loop":
                            if state.in_loop:
                                self.event_logger.system_info(
                                    f"⟳ Loop ended early at round {state.loop_round} of {state.loop_count}"
                                )
                                state.last_action_summary = (
                                    f"Loop ended early at round {state.loop_round} of {state.loop_count}"
                                )
                                _append_recent_action("[LOOP END] early exit")
                                _exit_loop()
                            else:
                                state.last_action_summary = "end_loop ignored — not in a loop"
                            state.checkpoint_pending = False

                        elif think_next_action == "done":
                            # Mission complete
                            if state.in_loop:
                                _exit_loop()
                            return MissionResult(
                                success=True,
                                reasoning=think_reasoning or "Mission complete",
                            )

                        elif think_next_action == "stuck":
                            replacement_strategy = think_reasoning or "Trying a different strategy."
                            state.checkpoint_pending = False
                            state.last_action_summary = f"Strategy switch (stuck): \"{replacement_strategy}\""
                            if recommended_next_step_arg:
                                cleaned = strip_targeting_data(recommended_next_step_arg)
                                state.last_action_summary += f" | recommended_next_step={cleaned}"
                            _append_recent_action(f"[STUCK] Strategy switch")
                            self.event_logger.system_info(f"↺ Strategy switched: {replacement_strategy}")

                        elif think_next_action == "continue":
                            state.last_action_summary = f"You thought: \"{think_reasoning}\""
                            if recommended_next_step_arg:
                                cleaned = strip_targeting_data(recommended_next_step_arg)
                                state.last_action_summary += f" | recommended_next_step={cleaned}"
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
                        _append_recent_action(f"{action_type}: {action_content}")
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
                                state.checkpoint_pending = True
                                action_success = True
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
                        _append_recent_action(state.last_action_summary)
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
                                state.checkpoint_pending = True
                                action_success = True
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
                        _append_recent_action(state.last_action_summary)
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
                                state.checkpoint_pending = True
                                action_success = True
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
                        _append_recent_action(state.last_action_summary)
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
                        _append_recent_action(state.last_action_summary)
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
                    _append_recent_action(state.last_action_summary)
                    state.checkpoint_pending = True

                    if result.success:
                        state.browser_actions_since_progress += 1

                        # Mark element as done in DOM if we're in a loop
                        if state.in_loop:
                            overlay_index = None
                            if result.metadata:
                                overlay_index = result.metadata.get("overlay_index")
                            if overlay_index is None:
                                overlay_index = action_args.get("element_id")
                            if overlay_index is not None:
                                try:
                                    self.action_executor.mark_element_done(int(overlay_index))
                                except (TypeError, ValueError):
                                    pass

                    # Sync tab manager after browser actions
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
                return MissionResult(
                    success=False,

                    reasoning=f"Error: {str(e)}",
                )

        # Max actions reached
        return MissionResult(
            success=False,
            reasoning=f"Max actions ({max_actions}) reached without completion",
        )
