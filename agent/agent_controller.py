"""Agent controller — mission execution with inline loops.

Architecture overview:
- Mission goes directly to the execution loop (no planner).
- Agent uses browser actions to accomplish the mission.
- When repetition is needed, the agent declares a loop inline via think(start_loop).
- Loop rounds are advanced with think(advance) and exited with think(end_loop) or think(done).
"""

import os
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable, Union, Type, TYPE_CHECKING
import hashlib
import copy

from browser.dom import build_page_elements
from browser.annotate import build_element_index, build_crop_gallery
from core.browser import ExecutionTimer
from core.executor.base import Executor
from core.agent_workspace import AgentWorkspace, AgentWorkspaceManager
from core.sandbox_policy import SandboxPolicyEngine
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
from utils.screenshot_store import ScreenshotStore, ScreenshotMeta

from .interceptor_manager import InterceptorManager, Interceptor, InterceptorMode, InterceptorContext
from core.config import Config
from execution.result import ActionResult
from utils import PageUtils
from core.browser import Browser

# Type alias for user question callback (ask: command handler)
# Callback receives: question (str), context (dict) -> returns user's answer (str) or None to skip
UserQuestionCallback = Callable[[str, dict], str]
# Type alias for reported text callback (report_data: command handler)
# Callback receives: payload (str), context (dict)
DataReportCallback = Callable[[str, dict], None]


@dataclass
class ExecutionState:
    """Mutable execution state for the mission loop."""
    total_actions: int = 0
    actions_since_progress: int = 0
    user_facing_actions_since_progress: int = 0
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
    # Budget telemetry
    budget_total: int = 0
    budget_spent: int = 0
    budget_remaining: int = 0
    budget_phase: str = "normal"
    low_budget_mode: bool = False
    budget_constraints_enabled: bool = True
    planning_batch_limit: int = 0


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
        # Data report callback for report_data: command
        data_report_callback: Optional[DataReportCallback] = None,
        # Callback to request a hint when the agent declares itself stuck
        on_stuck_callback: Optional[Callable[[str, int], Optional[str]]] = None,
    ):
        self.config = config
        self.mission_result = MissionResult()

        # Set global print mode based on config (affects all dprint calls and API debug logging)
        from utils.debug_print import set_print_mode, PrintMode
        set_print_mode(PrintMode.DEBUG if config.logging.debug_mode else PrintMode.NORMAL)

        # Access event logger from agent
        from utils.event_logger import EventLogger
        self.event_logger = EventLogger(
            debug_mode=config.logging.debug_mode,
            show_overlay_candidates=config.logging.show_overlay_candidates,
            show_llm_costs=config.logging.show_llm_costs,
        )
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
        self.data_report_callback = data_report_callback
        self.on_stuck_callback = on_stuck_callback
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
        self._cancel_event = threading.Event()
        self._hints_lock = threading.Lock()
        self._pending_hints: List[str] = []

        self.interceptor_stack: List[Dict[str, Any]] = []  # Stack of active interceptors

        # Execution timer for tracking mission, iteration, and action timings
        self.execution_timer = ExecutionTimer()

        # Per-agent storage workspace and temp-only cleanup.
        self.workspace_manager = AgentWorkspaceManager(
            base_dir=self.config.storage.base_dir,
            cleanup_enabled=self.config.storage.cleanup.enabled,
            temp_ttl_days=self.config.storage.cleanup.temp_ttl_days,
            temp_keep_last_runs=self.config.storage.cleanup.temp_keep_last_runs,
            temp_max_runs=self.config.storage.cleanup.temp_max_runs,
            max_disk_mb=self.config.storage.cleanup.max_disk_mb,
            event_logger=self.event_logger,
        )
        self.agent_workspace: AgentWorkspace = self.workspace_manager.create_agent(
            persistence_mode=self.config.storage.default_persistence_mode,
        )
        self.workspace_manager.cleanup_temp_runs(exclude_agent_id=self.agent_workspace.agent_id)
        self._active_run_id: Optional[str] = None
        self._active_run_open: bool = False

        # Route storage defaults into this agent workspace.
        self._apply_workspace_paths()

        self.show_llm_costs = config.logging.show_llm_costs
        _show_overlay_candidates = config.logging.show_overlay_candidates
        self.save_screenshots = config.logging.save_screenshots
        self.screenshot_dir = config.logging.screenshot_dir

        self._current_iteration = 0
        self.execution_state: Optional[ExecutionState] = None

        # Initialize execution system
        self._extraction_model_cache: Dict[tuple[str, ...], Type[BaseModel]] = {}
        self.screenshot_store = ScreenshotStore(
            max_in_memory_items=self.config.logging.screenshot_stream_in_memory_items,
            max_in_memory_mb=self.config.logging.screenshot_stream_in_memory_mb,
            persist_to_disk=self.config.logging.screenshot_stream_persist_to_disk,
            disk_dir=self.config.logging.screenshot_stream_dir,
            max_disk_files=self.config.logging.screenshot_stream_max_disk_files,
        )
        self.sandbox_policy = SandboxPolicyEngine(
            config=self.config,
            workspace_root=self.agent_workspace.workspace_root,
            event_logger=self.event_logger,
        )

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

    def pause(self, message: str = "Paused") -> None:
        with self._pause_lock:
            self._paused = True
            self._pause_message = message
            self._pause_event.clear()

    def resume(self) -> None:
        with self._pause_lock:
            self._paused = False
            self._pause_event.set()

    def cancel(self) -> None:
        self._cancel_event.set()
        # Unblock wait() if currently paused so cancellation can be observed immediately.
        with self._pause_lock:
            self._paused = False
            self._pause_event.set()

    def inject_hint(self, text: str) -> None:
        hint = (text or "").strip()
        if not hint:
            return
        with self._hints_lock:
            self._pending_hints.append(hint)

    def get_state_snapshot(self) -> Dict[str, Any]:
        with self._hints_lock:
            pending_hints = list(self._pending_hints)
        with self._pause_lock:
            paused = self._paused
            pause_message = self._pause_message

        return {
            "paused": paused,
            "pause_message": pause_message,
            "cancel_requested": self._cancel_event.is_set(),
            "current_iteration": self._current_iteration,
            "pending_hints": pending_hints,
            "execution_state": copy.deepcopy(self.execution_state),
            "mission_result": copy.deepcopy(self.mission_result),
            "llm_total_cost_usd": self.event_logger.total_cost_usd,
            "llm_total_tokens": self.event_logger.total_tokens,
        }

    def get_screenshot_bytes(self, screenshot_id: str) -> Optional[bytes]:
        return self.screenshot_store.get_bytes(screenshot_id)

    def get_screenshot_meta(self, screenshot_id: str) -> Optional[ScreenshotMeta]:
        return self.screenshot_store.get_meta(screenshot_id)

    def list_recent_screenshots(self, limit: int = 20) -> List[ScreenshotMeta]:
        return self.screenshot_store.list_recent(limit)

    def get_latest_screenshot_bytes(self) -> Optional[bytes]:
        return self.screenshot_store.get_latest_bytes()

    def get_latest_screenshot_meta(self) -> Optional[ScreenshotMeta]:
        return self.screenshot_store.get_latest_meta()

    def clear_screenshot_cache(self) -> None:
        self.screenshot_store.clear()

    def _apply_workspace_paths(self) -> None:
        """Route default storage paths into this agent's workspace."""
        ws = self.agent_workspace
        self.config.browser.user_data_dir = str(ws.browser_profile_dir)
        self.config.logging.screenshot_dir = str(ws.screenshots_dir)
        self.config.logging.screenshot_stream_dir = str(ws.stream_screenshots_dir)
        self.config.error_handling.screenshot_dir = str(ws.screenshots_dir)

        for path in (
            ws.workspace_root,
            ws.written_data_dir,
            ws.browser_profile_dir,
            ws.browser_downloads_dir,
            ws.screenshots_dir,
            ws.stream_screenshots_dir,
            ws.runs_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _current_page_url(self) -> str:
        """Best-effort current page URL from active browser page."""
        try:
            if self.browser and self.browser.page:
                return str(self.browser.page.url or "").strip()
        except Exception:
            pass
        return ""

    def _enforce_current_page_policy(self, *, source: str) -> Tuple[bool, Optional[str], str]:
        """
        Validate current page URL against sandbox policy.

        Returns:
            (allowed, warning_message_or_none, current_url)
        """
        current_url = self._current_page_url()
        if not current_url:
            return True, None, current_url

        decision = self.sandbox_policy.check_url(current_url)
        if decision.allowed:
            return True, None, current_url

        warning = f"{source} landed on disallowed URL: {decision.reason}"
        self.event_logger.system_warning(
            warning,
            source=source,
            current_url=current_url,
            sandbox_preset=self.sandbox_policy.preset,
            sandbox_mode=self.sandbox_policy.mode,
        )
        if self.sandbox_policy.enforce:
            return False, warning, current_url
        return True, warning, current_url

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
            data_report_callback=self.data_report_callback,
            user_messages_config=self.config.user_messages if self.config else None,
            workspace_paths={
                "written_data_dir": str(self.agent_workspace.written_data_dir),
            },
            sandbox_policy=self.sandbox_policy,
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

        # Emit navigation events for main-frame URL changes.
        try:
            def _on_frame_navigated(frame):
                try:
                    if self.browser and self.browser.page and frame == self.browser.page.main_frame:
                        self.event_logger.browser_navigation(url=frame.url)
                except Exception:
                    pass

            if hasattr(self.browser.page, "on"):
                self.browser.page.on("framenavigated", _on_frame_navigated)
        except Exception:
            pass

        # Plan generator for AI planning prompts
        self.started = True
      
    def execute_mission(
        self,
        user_prompt: str,
    ) -> MissionResult:
        
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
        return mission_result
    
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
        self.execution_state = None
        self._cancel_event.clear()
        with self._pause_lock:
            self._paused = False
            self._pause_message = "Paused"
            self._pause_event.set()
        self.event_logger.agent_start(user_mission)
        try:
            self._active_run_id = self.workspace_manager.start_run(
                self.agent_workspace,
                mission=user_mission,
            )
            self._active_run_open = True
            if self.config.sandbox.audit.enabled:
                self.sandbox_policy.set_audit_log_path(self.agent_workspace.sandbox_audit_path)
            else:
                self.sandbox_policy.set_audit_log_path(None)
        except Exception as e:
            self._active_run_id = None
            self._active_run_open = False
            self.sandbox_policy.set_audit_log_path(None)
            self.event_logger.system_warning(f"Failed to initialize run workspace: {e}")

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
            self.mission_result = self._build_mission_result(
                success=False,
                reasoning="Page is blank",
                narrative="Page is blank",
                state=self.execution_state,
            )
            return self.mission_result

        # Execute the mission directly
        self.mission_result = self._run_execution_loop(
            user_mission,
            start_in_checkpoint=True,
        )

        if self.execution_timer.mission_start_time is not None:
            self.execution_timer.end_mission()

        return self.mission_result

    def _build_mission_result(
        self,
        *,
        success: bool,
        reasoning: str,
        narrative: str,
        state: Optional[ExecutionState],
    ) -> MissionResult:
        duration_s = 0.0
        if self.mission_start_time is not None:
            duration_s = max(0.0, time.time() - self.mission_start_time)

        final_url = ""
        try:
            final_url = self.browser.page.url if self.browser and self.browser.page else ""
        except Exception:
            final_url = ""

        result = MissionResult(
            success=success,
            reasoning=reasoning,
            narrative=narrative,
            total_iterations=self._current_iteration,
            total_actions=state.actions_since_progress if state else 0,
            final_url=final_url,
            duration_s=duration_s,
            total_cost_usd=self.event_logger.total_cost_usd,
            budget_total=int(getattr(state, "budget_total", 0) or 0),
            budget_spent=int(getattr(state, "budget_spent", 0) or 0),
            budget_remaining=int(getattr(state, "budget_remaining", 0) or 0),
            budget_phase=str(getattr(state, "budget_phase", "normal") or "normal"),
        )
        if self._active_run_open:
            try:
                self.workspace_manager.finish_run(
                    self.agent_workspace,
                    success=bool(success),
                    reasoning=reasoning,
                    total_actions=int(getattr(state, "actions_since_progress", 0) or 0),
                    total_iterations=int(self._current_iteration or 0),
                    final_url=final_url,
                    duration_s=duration_s,
                )
            except Exception as e:
                self.event_logger.system_warning(f"Failed to finalize run workspace: {e}")
            finally:
                self._active_run_open = False
                self.sandbox_policy.set_audit_log_path(None)
                try:
                    self.workspace_manager.cleanup_temp_runs()
                except Exception:
                    pass
        else:
            self.sandbox_policy.set_audit_log_path(None)

        return result

    @staticmethod
    def _compute_budget_phase(remaining: int, total: int) -> str:
        """Derive a simple budget phase from remaining vs total actions."""
        total_i = max(1, int(total or 1))
        remaining_i = max(0, int(remaining or 0))
        critical_threshold = max(1, int(total_i * 0.03))
        caution_threshold = max(3, int(total_i * 0.10))
        if caution_threshold <= critical_threshold:
            caution_threshold = critical_threshold + 1
        if remaining_i <= critical_threshold:
            return "critical"
        if remaining_i <= caution_threshold:
            return "caution"
        return "normal"

    @staticmethod
    def _clamp_loop_count_to_budget(requested_loop_count: Any, budget_remaining: int) -> Tuple[int, bool]:
        """Clamp loop count to remaining budget. Returns (effective_count, clamped)."""
        try:
            count = int(requested_loop_count)
        except (TypeError, ValueError):
            count = 1
        count = max(1, count)
        remaining_i = max(0, int(budget_remaining or 0))
        if remaining_i > 0 and count > remaining_i:
            return remaining_i, True
        return count, False




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

        if snapshot.screenshot and self.config.logging.stream_screenshots:
            try:
                screenshot_meta = self.screenshot_store.put(
                    snapshot.screenshot,
                    iteration=self._current_iteration,
                    url=getattr(snapshot, "url", "") or "",
                    title=getattr(snapshot, "title", "") or "",
                )
                setattr(snapshot, "screenshot_id", screenshot_meta.screenshot_id)
                self.event_logger.screenshot_captured(
                    screenshot_id=screenshot_meta.screenshot_id,
                    iteration=screenshot_meta.iteration,
                    byte_size=screenshot_meta.byte_size,
                    sha256=screenshot_meta.sha256,
                    path=screenshot_meta.path,
                    in_memory=screenshot_meta.in_memory,
                    url=screenshot_meta.url,
                )
            except Exception as e:
                self.event_logger.system_warning(f"Failed to stream screenshot metadata: {e}")

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
        elif fn == "report_data":
            payload = args.get("payload", "")
            return f"You reported data to the user: \"{payload}\". Result: {result_str}."
        elif fn == "write_data":
            resolved = args.get("path", "") or args.get("file_name", "") or "default location"
            return f"You wrote data to {resolved}. Result: {result_str}."
        elif fn == "wait_for":
            condition = args.get("condition", "")
            return f"You waited for: \"{condition}\". Result: {result_str}."
        elif fn == "send_email":
            to = args.get("to", "")
            subject = args.get("subject", "")
            body = args.get("body", "")
            return f"You sent an email to {to} with the subject \"{subject}\" and the body \"{body}\". Result: {result_str}."
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
            budget_constraints_enabled=bool(self.config.execution.budget_constraints_enabled),
        )
        state.budget_total = max_actions
        state.planning_batch_limit = self.config.execution.max_actions_per_plan
        self.execution_state = state
        self.memory_store.start_mission(mission)

        last_checkpoint_pending = state.checkpoint_pending
        self.event_logger.checkpoint_changed(pending=state.checkpoint_pending)

        def _set_checkpoint_pending(value: bool) -> None:
            nonlocal last_checkpoint_pending
            new_value = bool(value)
            state.checkpoint_pending = new_value
            if new_value != last_checkpoint_pending:
                last_checkpoint_pending = new_value
                self.event_logger.checkpoint_changed(pending=new_value)

        def _append_recent_action(summary: str) -> None:
            state.recent_actions.append(summary)
            if len(state.recent_actions) > 10:
                state.recent_actions.pop(0)

        def _refresh_budget_state() -> None:
            state.budget_spent = max(0, int(state.total_actions or 0))
            state.budget_total = max(1, int(max_actions or 1))
            state.budget_remaining = max(0, state.budget_total - state.budget_spent)
            state.budget_phase = self._compute_budget_phase(
                remaining=state.budget_remaining,
                total=state.budget_total,
            )
            state.low_budget_mode = state.budget_phase in {"caution", "critical"}
            state.planning_batch_limit = (
                1
                if (state.budget_constraints_enabled and state.low_budget_mode)
                else self.config.execution.max_actions_per_plan
            )

        def _exit_loop() -> None:
            state.in_loop = False
            state.loop_count = None
            state.loop_round = 0
            state.loop_description = ""
            try:
                self.action_executor.clear_done_markers()
            except Exception:
                pass

        _refresh_budget_state()

        while state.total_actions < max_actions:
            if self._cancel_event.is_set():
                return self._build_mission_result(
                    success=False,
                    reasoning="Mission cancelled",
                    narrative="Mission cancelled",
                    state=state,
                )

            self._pause_event.wait()
            if self._cancel_event.is_set():
                return self._build_mission_result(
                    success=False,
                    reasoning="Mission cancelled",
                    narrative="Mission cancelled",
                    state=state,
                )

            iteration_started_at = time.time()
            state.total_actions += 1
            _refresh_budget_state()
            self._current_iteration += 1
            self.execution_state = state
            self.event_logger.iteration_start(
                iteration=self._current_iteration,
                max_iterations=max_actions,
                mission=mission,
                budget_spent=state.budget_spent,
                budget_remaining=state.budget_remaining,
                budget_phase=state.budget_phase,
                low_budget_mode=state.low_budget_mode,
            )

            try:
                try:
                    snapshot = self._capture_snapshot(full_page=False)
                    snapshot_url = str(getattr(snapshot, "url", "") or "").strip()
                    if snapshot_url:
                        current_url_decision = self.sandbox_policy.check_url(snapshot_url)
                        if not current_url_decision.allowed:
                            warning = f"Current page blocked by sandbox: {current_url_decision.reason}"
                            self.event_logger.system_warning(
                                warning,
                                current_url=snapshot_url,
                                sandbox_preset=self.sandbox_policy.preset,
                                sandbox_mode=self.sandbox_policy.mode,
                            )
                            if self.sandbox_policy.enforce:
                                state.last_action_summary = f"sandbox FAILED: {warning} ({snapshot_url})"
                                _append_recent_action(state.last_action_summary)
                                return self._build_mission_result(
                                    success=False,
                                    reasoning=f"{warning} ({snapshot_url})",
                                    narrative="Sandbox blocked disallowed current page",
                                    state=state,
                                )
                            state.last_action_summary = f"sandbox(observe): {warning} ({snapshot_url})"
                            _append_recent_action(state.last_action_summary)
                    page_info = self.page_utils.get_page_info()
                    detected_elements = build_page_elements(self.browser.page, page_info)
                except Exception as e:
                    return self._build_mission_result(
                        success=False,
                        reasoning=f"Failed to capture state: {str(e)}",
                        state=state,
                        narrative="Failed to capture state",
                    )

                elements = getattr(detected_elements, "elements", []) or []
                text_rich = sum(
                    1 for elem in elements
                    if int(getattr(elem, "text_presence_score", 0) or 0) >= 2
                )
                text_poor = sum(
                    1 for elem in elements
                    if int(getattr(elem, "text_presence_score", 0) or 0) <= 1
                )
                self.event_logger.element_capture(
                    total=len(elements),
                    text_rich=text_rich,
                    text_poor=text_poor,
                )

                prep = self._prepare_screenshot_for_mode(snapshot, detected_elements, page_info)
                annotated_screenshot_bytes = prep.screenshot_bytes
                element_index_text = prep.element_index_text
                gallery_images = prep.gallery_images

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
                    budget_spent=state.budget_spent,
                    budget_remaining=state.budget_remaining,
                    budget_total=state.budget_total,
                    budget_phase=state.budget_phase,
                    low_budget_mode=state.low_budget_mode,
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

                with self._hints_lock:
                    pending_hints = list(self._pending_hints)
                    self._pending_hints.clear()

                active_strategy = self.memory_store.get_latest_strategy() or None
                policy_constraints_block = None
                if self.config.sandbox.prompt.include_policy_block:
                    policy_constraints_block = self.sandbox_policy.render_prompt_policy_block()
                if (
                    self.config.logging.debug_mode
                    and state.budget_constraints_enabled
                    and state.low_budget_mode
                ):
                    self.event_logger.system_debug(
                        "Low-budget mode active; using single-action planning",
                        budget_phase=state.budget_phase,
                        budget_remaining=state.budget_remaining,
                        planning_batch_limit=state.planning_batch_limit,
                    )
                action_planner = ActionPlanner(
                    mission,
                    self.memory_store,
                    base_knowledge=self.base_knowledge,
                    model_name=self.agent_model_name,
                    reasoning_level=self.agent_reasoning_level,
                    image_detail=self.config.model.image_detail,
                    max_actions_per_plan=state.planning_batch_limit,
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
                    user_facing_actions_in_round=state.user_facing_actions_since_progress,
                    user_hints=pending_hints,
                    policy_constraints_block=policy_constraints_block,
                    in_loop=state.in_loop,
                    loop_round=state.loop_round,
                    loop_count=state.loop_count,
                    loop_description=state.loop_description,
                    recent_actions=state.recent_actions,
                    iterations_remaining=state.budget_remaining,
                    max_iterations=state.budget_total,
                    budget_spent=state.budget_spent,
                    budget_phase=state.budget_phase,
                    low_budget_mode=state.low_budget_mode,
                    budget_constraints_enabled=state.budget_constraints_enabled,
                )

                try:
                    actions_list, error = action_planner.get_next_actions_with_function_calling(
                        environment_state=environment_state,
                        screenshot=annotated_screenshot_bytes,
                        notebook=self.notebook,
                        element_data=detected_elements,
                    )
                except Exception as e:
                    return self._build_mission_result(
                        success=False,
                        reasoning=f"Error: {str(e)}",
                        state=state,
                        narrative="Failed to get next actions",
                    )

                if not actions_list:
                    state.validation_failures += 1
                    if state.validation_failures <= self.config.execution.validation_failure_escalation_limit:
                        state.last_action_summary = f"Action validation issue: {error or 'No action generated'}. Retrying."
                        _set_checkpoint_pending(False)
                        continue
                    return self._build_mission_result(
                        success=False,
                        reasoning=f"Repeated action validation failures: {error or 'No action generated'}",
                        state=state,
                        narrative="Repeated action validation failures",
                    )

                for action_step in actions_list:
                    if self._cancel_event.is_set():
                        return self._build_mission_result(
                            success=False,
                            reasoning="Mission cancelled",
                            narrative="Mission cancelled",
                            state=state,
                        )
                    self._pause_event.wait()
                    if self._cancel_event.is_set():
                        return self._build_mission_result(
                            success=False,
                            reasoning="Mission cancelled",
                            narrative="Mission cancelled",
                            state=state,
                        )

                    function_name = (getattr(action_step, "function_name", None) or "").strip()
                    action_args = getattr(action_step, "function_arguments", {}) or {}
                    current_action = getattr(action_step, "action", "") or function_name
                    reasoning = action_args.get("reasoning", "")
                    narrative = action_args.get("narrative", "")
                    policy_observe_warning: Optional[str] = None
                    if self.config.logging.debug_mode:
                        self.event_logger.system_debug(
                            "Budget telemetry",
                            tool=function_name,
                            budget_phase=state.budget_phase,
                            budget_remaining=state.budget_remaining,
                            budget_spent=state.budget_spent,
                            budget_total=state.budget_total,
                        )

                    self.event_logger.action_determined(
                        action=current_action,
                        reasoning=reasoning,
                        narrative=narrative,
                        tool=function_name,
                        budget_phase=state.budget_phase,
                        budget_remaining=state.budget_remaining,
                        budget_spent=state.budget_spent,
                        budget_total=state.budget_total,
                        in_loop=state.in_loop,
                        loop_round=state.loop_round if state.in_loop else None,
                        loop_count=state.loop_count if state.in_loop else None,
                    )

                    if function_name == "think":
                        result = self.action_executor.act(
                            action_step=action_step,
                            detected_elements=detected_elements,
                            page_info=page_info,
                            environment_state=environment_state,
                            current_iteration=self._current_iteration,
                        )
                        duration_ms = float((result.metadata or {}).get("duration_ms", 0.0))
                        state.actions_since_progress += 1

                        think_reasoning = str(action_args.get("reasoning", "")).strip()
                        think_next_action = str(action_args.get("next_action", "continue")).strip().lower()
                        recommended_next_step_arg = str(action_args.get("recommended_next_step", "")).strip()

                        if think_next_action == "start_loop":
                            loop_count_raw = action_args.get("loop_count")
                            loop_desc = str(action_args.get("loop_description", "")).strip()
                            loop_count, clamped_loop = self._clamp_loop_count_to_budget(
                                requested_loop_count=loop_count_raw,
                                budget_remaining=(
                                    state.budget_remaining
                                    if state.budget_constraints_enabled
                                    else 0
                                ),
                            )

                            state.in_loop = True
                            state.loop_count = loop_count
                            state.loop_round = 2
                            state.loop_description = loop_desc or think_reasoning
                            _set_checkpoint_pending(False)
                            state.user_facing_actions_since_progress = 0

                            state.last_action_summary = f"Loop started: {loop_desc or think_reasoning} (round 2 of {loop_count})"
                            if clamped_loop:
                                state.last_action_summary += f" [clamped to budget remaining={state.budget_remaining}]"
                            _append_recent_action(f"[LOOP START] {loop_desc} — {loop_count} total rounds")
                            if clamped_loop and self.config.logging.debug_mode:
                                self.event_logger.system_debug(
                                    "Loop count clamped to budget",
                                    requested_loop_count=loop_count_raw,
                                    effective_loop_count=loop_count,
                                    budget_remaining=state.budget_remaining,
                                )
                            self.event_logger.loop_state_changed(
                                change="start",
                                loop_round=state.loop_round,
                                loop_count=state.loop_count,
                                loop_description=state.loop_description,
                            )

                        elif think_next_action == "advance":
                            if not state.in_loop:
                                state.last_action_summary = "advance ignored — not in a loop"
                                _set_checkpoint_pending(False)
                            elif state.user_facing_actions_since_progress == 0:
                                state.last_action_summary = "advance BLOCKED: No user-facing actions since last advance. Do a user-facing action first."
                                _set_checkpoint_pending(False)
                            else:
                                state.loop_round += 1
                                state.user_facing_actions_since_progress = 0
                                _set_checkpoint_pending(False)
                                if state.loop_count and state.loop_round > state.loop_count:
                                    state.last_action_summary = f"Loop complete — all {state.loop_count} rounds done"
                                    _append_recent_action(f"[LOOP COMPLETE] {state.loop_count} rounds done")
                                    self.event_logger.loop_state_changed(
                                        change="end",
                                        loop_round=state.loop_count,
                                        loop_count=state.loop_count,
                                        loop_description=state.loop_description,
                                    )
                                    _exit_loop()
                                else:
                                    remaining = state.loop_count - state.loop_round + 1 if state.loop_count else "?"
                                    state.last_action_summary = (
                                        f"Advanced to loop round {state.loop_round} of {state.loop_count} ({remaining} remaining)"
                                    )
                                    _append_recent_action(f"[ADVANCE] Round {state.loop_round} of {state.loop_count}")
                                    self.event_logger.loop_state_changed(
                                        change="advance",
                                        loop_round=state.loop_round,
                                        loop_count=state.loop_count,
                                        loop_description=state.loop_description,
                                    )

                        elif think_next_action == "end_loop":
                            if state.in_loop:
                                state.last_action_summary = f"Loop ended early at round {state.loop_round} of {state.loop_count}"
                                _append_recent_action("[LOOP END] early exit")
                                self.event_logger.loop_state_changed(
                                    change="end_early",
                                    loop_round=state.loop_round,
                                    loop_count=state.loop_count,
                                    loop_description=state.loop_description,
                                )
                                _exit_loop()
                            else:
                                state.last_action_summary = "end_loop ignored — not in a loop"
                            _set_checkpoint_pending(False)

                        elif think_next_action == "done":
                            if state.in_loop:
                                self.event_logger.loop_state_changed(
                                    change="end",
                                    loop_round=state.loop_round,
                                    loop_count=state.loop_count,
                                    loop_description=state.loop_description,
                                )
                                _exit_loop()
                            result_str = "success" if result.success else "failed"
                            self.event_logger.action_complete(
                                tool=function_name,
                                narrative=narrative,
                                success=bool(result.success),
                                result_str=result_str,
                                duration_ms=duration_ms,
                                iteration=self._current_iteration,
                            )
                            return self._build_mission_result(
                                success=True,
                                reasoning=think_reasoning or "Mission complete",
                                narrative=narrative,
                                state=state,
                            )

                        elif think_next_action == "stuck":
                            replacement_strategy = think_reasoning or "Trying a different strategy."
                            _set_checkpoint_pending(False)
                            state.last_action_summary = f"Strategy switch (stuck): \"{replacement_strategy}\""
                            if recommended_next_step_arg:
                                cleaned = strip_targeting_data(recommended_next_step_arg)
                                state.last_action_summary += f" | recommended_next_step={cleaned}"
                            _append_recent_action("[STUCK] Strategy switch")
                            if self.on_stuck_callback:
                                try:
                                    hint = self.on_stuck_callback(replacement_strategy, self._current_iteration)
                                    if hint:
                                        with self._hints_lock:
                                            self._pending_hints.append(hint.strip())
                                except Exception as e:
                                    self.event_logger.system_warning(f"on_stuck_callback failed: {e}")

                        elif think_next_action == "continue":
                            state.last_action_summary = f"You thought: \"{think_reasoning}\""
                            if recommended_next_step_arg:
                                cleaned = strip_targeting_data(recommended_next_step_arg)
                                state.last_action_summary += f" | recommended_next_step={cleaned}"
                            _set_checkpoint_pending(False)

                        else:
                            _set_checkpoint_pending(False)

                        result_str = "success" if result.success else "failed"
                        self.event_logger.action_complete(
                            tool=function_name,
                            narrative=narrative,
                            success=bool(result.success),
                            result_str=result_str,
                            duration_ms=duration_ms,
                            iteration=self._current_iteration,
                        )
                        continue

                    if function_name in {"assert_condition", "flag"}:
                        result = self.action_executor.act(
                            action_step=action_step,
                            detected_elements=detected_elements,
                            page_info=page_info,
                            environment_state=environment_state,
                            current_iteration=self._current_iteration,
                        )
                        duration_ms = float((result.metadata or {}).get("duration_ms", 0.0))
                        action_type = "assert" if function_name == "assert_condition" else "flag"
                        action_content = str(
                            action_args.get("condition") if function_name == "assert_condition" else action_args.get("message", "")
                        ).strip()
                        state.last_action_summary = f"You called {action_type}: \"{action_content}\""
                        _append_recent_action(f"{action_type}: {action_content}")
                        state.actions_since_progress += 1
                        _set_checkpoint_pending(True)
                        result_str = "success" if result.success else "failed"
                        self.event_logger.action_complete(
                            tool=function_name,
                            narrative=narrative,
                            success=bool(result.success),
                            result_str=result_str,
                            duration_ms=duration_ms,
                            iteration=self._current_iteration,
                        )
                        continue

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
                                policy_allowed, policy_warning, current_url = self._enforce_current_page_policy(
                                    source="switch_tab"
                                )
                                _set_checkpoint_pending(True)
                                if not policy_allowed:
                                    action_error = f"{policy_warning} ({current_url})"
                                    state.last_action_summary = f"switch_tab FAILED: {action_error}"
                                else:
                                    if policy_warning:
                                        state.last_action_summary += f" | sandbox(observe): {policy_warning}"
                                    state.user_facing_actions_since_progress += 1
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
                        self.event_logger.action_complete(
                            tool=function_name,
                            narrative=narrative,
                            success=action_success,
                            result_str="success" if action_success else "failed",
                            duration_ms=0.0,
                            iteration=self._current_iteration,
                        )
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
                                policy_allowed, policy_warning, current_url = self._enforce_current_page_policy(
                                    source="close_tab"
                                )
                                _set_checkpoint_pending(True)
                                if not policy_allowed:
                                    action_error = f"{policy_warning} ({current_url})"
                                    state.last_action_summary = f"close_tab FAILED: {action_error}"
                                else:
                                    if policy_warning:
                                        state.last_action_summary += f" | sandbox(observe): {policy_warning}"
                                    state.user_facing_actions_since_progress += 1
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
                        self.event_logger.action_complete(
                            tool=function_name,
                            narrative=narrative,
                            success=action_success,
                            result_str="success" if action_success else "failed",
                            duration_ms=0.0,
                            iteration=self._current_iteration,
                        )
                        continue

                    if function_name == "open_tab":
                        before_state = self.memory_store._capture_current_state()
                        action_success = False
                        action_error: Optional[str] = None
                        url = str(action_args.get("url", "")).strip() or None
                        observe_warning: Optional[str] = None
                        if url:
                            url_decision = self.sandbox_policy.check_url(url)
                            if not url_decision.allowed:
                                warning = f"open_tab blocked by sandbox: {url_decision.reason}"
                                self.event_logger.system_warning(warning)
                                if self.sandbox_policy.enforce:
                                    action_error = warning
                                    state.last_action_summary = f"open_tab FAILED: {url_decision.reason}"
                                else:
                                    observe_warning = warning

                        if action_error is None and self.tab_manager:
                            try:
                                new_page = self.tab_manager.open_tab(url)
                                self.action_executor.set_page(new_page)
                                active = self.tab_manager.get_active()
                                active_id = active.id if active else "?"
                                state.last_action_summary = f"Opened new tab [{active_id}]"
                                if url:
                                    state.last_action_summary += f" at {url}"
                                if observe_warning:
                                    state.last_action_summary += f" | sandbox(observe): {observe_warning}"
                                _set_checkpoint_pending(True)
                                policy_allowed, policy_warning, current_url = self._enforce_current_page_policy(
                                    source="open_tab"
                                )
                                if not policy_allowed:
                                    action_error = f"{policy_warning} ({current_url})"
                                    state.last_action_summary = f"open_tab FAILED: {action_error}"
                                else:
                                    if policy_warning:
                                        state.last_action_summary += f" | sandbox(observe): {policy_warning}"
                                    state.user_facing_actions_since_progress += 1
                                    action_success = True
                            except Exception as e:
                                state.last_action_summary = f"open_tab FAILED: {e}"
                                action_error = str(e)
                        elif action_error is None:
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
                        self.event_logger.action_complete(
                            tool=function_name,
                            narrative=narrative,
                            success=action_success,
                            result_str="success" if action_success else "failed",
                            duration_ms=0.0,
                            iteration=self._current_iteration,
                        )
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
                            _set_checkpoint_pending(True)
                            action_success = True
                        else:
                            state.last_action_summary = "dismiss_dialog: No dialog pending"
                            _set_checkpoint_pending(True)
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
                        if action_success:
                            state.user_facing_actions_since_progress += 1
                        self.event_logger.action_complete(
                            tool=function_name,
                            narrative=narrative,
                            success=action_success,
                            result_str="success" if action_success else "failed",
                            duration_ms=0.0,
                            iteration=self._current_iteration,
                        )
                        continue

                    # Send email via Resend API (controller-only; no browser action).
                    if function_name == "send_email":
                        before_state = self.memory_store._capture_current_state()
                        action_success = False
                        action_error: Optional[str] = None
                        duplicate_of: Optional[str] = None
                        message_id: Optional[str] = None

                        raw_to = action_args.get("to")
                        if raw_to:
                            to_list = [str(raw_to).strip()]
                        else:
                            to_list = []

                        subject = str(action_args.get("subject", "")).strip()
                        body = str(action_args.get("body", "")).strip()
                        body_preview = body if len(body) <= 200 else f"{body[:197]}..."
                        body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest() if body else ""
                        effective_from_email = "Acme <onboarding@resend.dev>"

                        canonical_to = sorted({email.lower() for email in to_list})
                        signature_source = f"{'|'.join(canonical_to)}\n{subject.lower()}\n{body}"
                        email_signature = hashlib.sha256(signature_source.encode("utf-8")).hexdigest()

                        for entry in reversed(self.memory_store.entries):
                            if entry.action_type != "send_email":
                                continue
                            if entry.outcome not in {"success", "no_change"}:
                                continue
                            if str(entry.action_params.get("email_signature", "")).strip() == email_signature:
                                duplicate_of = entry.memory_id
                                action_success = True
                                state.last_action_summary = (
                                    f"send_email skipped: identical email already sent ({duplicate_of}). "
                                    "If the mission was only this email, call think(next_action=done)."
                                )
                                break

                        try:
                            if duplicate_of is None:
                                import resend
                                api_key = os.environ.get("RESEND_API_KEY")
                                if not api_key:
                                    action_error = "RESEND_API_KEY is not set"
                                    state.last_action_summary = f"send_email FAILED: {action_error}"
                                elif not to_list:
                                    action_error = "No recipients (to) provided"
                                    state.last_action_summary = f"send_email FAILED: {action_error}"
                                elif not subject:
                                    action_error = "Subject is required"
                                    state.last_action_summary = f"send_email FAILED: {action_error}"
                                elif not body:
                                    action_error = "Body is required"
                                    state.last_action_summary = f"send_email FAILED: {action_error}"
                                else:
                                    resend.api_key = api_key
                                    params = {
                                        "from": effective_from_email,
                                        "to": to_list,
                                        "subject": subject,
                                        "html": body,
                                    }
                                    send_result = resend.Emails.send(params)
                                    raw_message_id = (
                                        send_result.get("id")
                                        if isinstance(send_result, dict)
                                        else getattr(send_result, "id", None)
                                    )
                                    if raw_message_id:
                                        message_id = str(raw_message_id).strip()

                                    state.last_action_summary = (
                                        f"Email sent to {', '.join(to_list)}: \"{subject}\" "
                                        f"| body=\"{body_preview}\""
                                    )
                                    if message_id:
                                        state.last_action_summary += f" | message_id={message_id}"
                                    action_success = True
                        except Exception as e:
                            action_error = str(e)
                            state.last_action_summary = f"send_email FAILED: {action_error}"
                        self._record_controller_action(
                            action_type="send_email",
                            action_step=action_step,
                            success=action_success,
                            error_message=action_error,
                            action_params={
                                "operation": "send_email",
                                "to": to_list,
                                "subject": subject,
                                "body_preview": body_preview,
                                "body_hash": body_hash,
                                "from_email": effective_from_email,
                                "email_signature": email_signature,
                                "duplicate_of": duplicate_of,
                                "message_id": message_id,
                            },
                            before_state=before_state,
                            after_state=self.memory_store._capture_current_state(),
                        )
                        if action_success:
                            with self._hints_lock:
                                self._pending_hints.append(
                                    "You already sent the requested email. Do not send it again. "
                                    "If the mission is complete, call think(next_action=done)."
                                )
                        _append_recent_action(state.last_action_summary)
                        state.actions_since_progress += 1
                        if action_success and duplicate_of is None:
                            state.user_facing_actions_since_progress += 1
                        _set_checkpoint_pending(True)
                        self.event_logger.action_complete(
                            tool=function_name,
                            narrative=narrative,
                            success=action_success,
                            result_str="success" if action_success else "failed",
                            duration_ms=0.0,
                            iteration=self._current_iteration,
                        )
                        continue

                    if function_name == "bash":
                        import subprocess

                        before_state = self.memory_store._capture_current_state()
                        command = str(action_args.get("command", "")).strip()
                        command_timeout = self.sandbox_policy.command_timeout_seconds()
                        action_success = False
                        action_error: Optional[str] = None
                        observe_warning: Optional[str] = None

                        if not command:
                            action_error = "No command provided"
                            state.last_action_summary = "bash FAILED: No command provided"
                        else:
                            command_decision = self.sandbox_policy.check_command(command)
                            if not command_decision.allowed:
                                warning = f"bash blocked by sandbox: {command_decision.reason}"
                                self.event_logger.system_warning(warning)
                                if self.sandbox_policy.enforce:
                                    action_error = warning
                                    state.last_action_summary = f"bash FAILED: {command_decision.reason}"
                                else:
                                    observe_warning = warning

                            if action_error is None:
                                try:
                                    proc = subprocess.run(
                                        ["bash", "-lc", command],
                                        capture_output=True,
                                        text=True,
                                        timeout=command_timeout,
                                    )
                                    stdout = (proc.stdout or "").rstrip()
                                    stderr = (proc.stderr or "").rstrip()
                                    exit_code = proc.returncode

                                    parts = [f"bash: `{command}`", f"exit_code={exit_code}"]
                                    if stdout:
                                        preview = stdout if len(stdout) <= 2000 else f"{stdout[:2000]}\n... (truncated)"
                                        parts.append(f"stdout:\n{preview}")
                                    else:
                                        parts.append("stdout: (no output)")
                                    if stderr:
                                        parts.append(f"stderr: {stderr[:500]}")

                                    state.last_action_summary = "\n".join(parts)
                                    if observe_warning:
                                        state.last_action_summary += f"\nsandbox(observe): {observe_warning}"
                                    action_success = True  # command ran; agent sees exit_code in summary
                                except subprocess.TimeoutExpired:
                                    action_error = f"Command timed out after {command_timeout}s"
                                    state.last_action_summary = f"bash FAILED: {action_error}"
                                except Exception as e:
                                    action_error = str(e)
                                    state.last_action_summary = f"bash FAILED: {action_error}"

                        self._record_controller_action(
                            action_type="bash",
                            action_step=action_step,
                            success=action_success,
                            error_message=action_error,
                            action_params={"command": command},
                            before_state=before_state,
                            after_state=self.memory_store._capture_current_state(),
                        )
                        _append_recent_action(f"bash: {command}" if command else "bash: (empty)")
                        state.actions_since_progress += 1
                        if action_success:
                            state.user_facing_actions_since_progress += 1
                        _set_checkpoint_pending(True)
                        self.event_logger.action_complete(
                            tool=function_name,
                            narrative=narrative,
                            success=action_success,
                            result_str="success" if action_success else "failed",
                            duration_ms=0.0,
                            iteration=self._current_iteration,
                        )
                        continue

                    if function_name == "read_file":
                        from pathlib import Path

                        before_state = self.memory_store._capture_current_state()
                        path_arg = str(action_args.get("path", "")).strip()
                        start_line_raw = action_args.get("start_line")
                        end_line_raw = action_args.get("end_line")
                        start_line: Optional[int] = None
                        end_line: Optional[int] = None
                        action_success = False
                        action_error: Optional[str] = None
                        observe_warning: Optional[str] = None

                        if start_line_raw is not None:
                            try:
                                start_line = max(1, int(start_line_raw))
                            except (TypeError, ValueError):
                                start_line = None
                        if end_line_raw is not None:
                            try:
                                end_line = max(1, int(end_line_raw))
                            except (TypeError, ValueError):
                                end_line = None

                        if not path_arg:
                            action_error = "No path provided"
                            state.last_action_summary = "read_file FAILED: No path provided"
                        else:
                            try:
                                resolved = Path(path_arg).expanduser().resolve()
                                path_decision = self.sandbox_policy.check_path(resolved, operation="read")
                                if not path_decision.allowed:
                                    warning = f"read_file blocked by sandbox: {path_decision.reason}"
                                    self.event_logger.system_warning(warning)
                                    if self.sandbox_policy.enforce:
                                        action_error = warning
                                        state.last_action_summary = f"read_file FAILED: {path_decision.reason}"
                                    else:
                                        observe_warning = warning

                                if action_error is not None:
                                    raise RuntimeError(action_error)
                                lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
                                total_lines = len(lines)

                                if start_line and end_line and start_line > end_line:
                                    action_error = f"Invalid line range: start_line={start_line} > end_line={end_line}"
                                    state.last_action_summary = f"read_file FAILED: {action_error}"
                                else:
                                    start_idx = (start_line - 1) if start_line else 0
                                    end_idx = end_line if end_line else total_lines
                                    start_idx = min(max(start_idx, 0), total_lines)
                                    end_idx = min(max(end_idx, start_idx), total_lines)

                                    content = "\n".join(lines[start_idx:end_idx])
                                    if len(content) > 4000:
                                        content = f"{content[:4000]}\n... (truncated)"

                                    if start_line or end_line:
                                        range_note = f" (lines {start_idx + 1}-{end_idx} of {total_lines})"
                                    else:
                                        range_note = f" ({total_lines} lines)"

                                    state.last_action_summary = f"read_file: {resolved}{range_note}\n{content}"
                                    if observe_warning:
                                        state.last_action_summary += f"\nsandbox(observe): {observe_warning}"
                                    action_success = True
                            except FileNotFoundError:
                                action_error = f"File not found: {path_arg}"
                                state.last_action_summary = f"read_file FAILED: {action_error}"
                            except Exception as e:
                                action_error = str(e)
                                state.last_action_summary = f"read_file FAILED: {action_error}"

                        self._record_controller_action(
                            action_type="read_file",
                            action_step=action_step,
                            success=action_success,
                            error_message=action_error,
                            action_params={
                                "path": path_arg,
                                "start_line": start_line,
                                "end_line": end_line,
                            },
                            before_state=before_state,
                            after_state=self.memory_store._capture_current_state(),
                        )
                        _append_recent_action(f"read_file: {path_arg}")
                        state.actions_since_progress += 1
                        if action_success:
                            state.user_facing_actions_since_progress += 1
                        _set_checkpoint_pending(True)
                        self.event_logger.action_complete(
                            tool=function_name,
                            narrative=narrative,
                            success=action_success,
                            result_str="success" if action_success else "failed",
                            duration_ms=0.0,
                            iteration=self._current_iteration,
                        )
                        continue

                    if function_name == "find_files":
                        from pathlib import Path

                        before_state = self.memory_store._capture_current_state()
                        pattern = str(action_args.get("pattern", "")).strip()
                        directory = str(action_args.get("directory", "~")).strip() or "~"
                        recursive = bool(action_args.get("recursive", True))
                        action_success = False
                        action_error: Optional[str] = None
                        observe_warning: Optional[str] = None

                        if not pattern:
                            action_error = "No pattern provided"
                            state.last_action_summary = "find_files FAILED: No pattern provided"
                        else:
                            try:
                                root = Path(directory).expanduser().resolve()
                                path_decision = self.sandbox_policy.check_path(root, operation="find")
                                if not path_decision.allowed:
                                    warning = f"find_files blocked by sandbox: {path_decision.reason}"
                                    self.event_logger.system_warning(warning)
                                    if self.sandbox_policy.enforce:
                                        action_error = warning
                                        state.last_action_summary = f"find_files FAILED: {path_decision.reason}"
                                    else:
                                        observe_warning = warning

                                if action_error is not None:
                                    raise RuntimeError(action_error)
                                glob_fn = root.rglob if recursive else root.glob
                                max_results = 50
                                found_matches: List[str] = []
                                for path in glob_fn(pattern):
                                    found_matches.append(str(path))
                                    if len(found_matches) > max_results:
                                        break

                                has_more = len(found_matches) > max_results
                                matches = sorted(found_matches[:max_results])

                                if matches:
                                    listing = "\n".join(matches)
                                    note = (
                                        f" (showing first {len(matches)}; more matches exist)"
                                        if has_more
                                        else f" ({len(matches)} found)"
                                    )
                                    state.last_action_summary = f"find_files: `{pattern}` in {root}{note}\n{listing}"
                                else:
                                    state.last_action_summary = f"find_files: `{pattern}` in {root} - no matches found"
                                if observe_warning:
                                    state.last_action_summary += f"\nsandbox(observe): {observe_warning}"
                                action_success = True
                            except Exception as e:
                                action_error = str(e)
                                state.last_action_summary = f"find_files FAILED: {action_error}"

                        self._record_controller_action(
                            action_type="find_files",
                            action_step=action_step,
                            success=action_success,
                            error_message=action_error,
                            action_params={
                                "pattern": pattern,
                                "directory": directory,
                                "recursive": recursive,
                            },
                            before_state=before_state,
                            after_state=self.memory_store._capture_current_state(),
                        )
                        _append_recent_action(f"find_files: {pattern} in {directory}")
                        state.actions_since_progress += 1
                        if action_success:
                            state.user_facing_actions_since_progress += 1
                        _set_checkpoint_pending(True)
                        self.event_logger.action_complete(
                            tool=function_name,
                            narrative=narrative,
                            success=action_success,
                            result_str="success" if action_success else "failed",
                            duration_ms=0.0,
                            iteration=self._current_iteration,
                        )
                        continue

                    if function_name == "read_clipboard":
                        import subprocess
                        import sys

                        before_state = self.memory_store._capture_current_state()
                        action_success = False
                        action_error: Optional[str] = None
                        content = ""
                        observe_warning: Optional[str] = None

                        clipboard_decision = self.sandbox_policy.check_clipboard_read()
                        if not clipboard_decision.allowed:
                            warning = f"read_clipboard blocked by sandbox: {clipboard_decision.reason}"
                            self.event_logger.system_warning(warning)
                            if self.sandbox_policy.enforce:
                                action_error = warning
                                state.last_action_summary = f"read_clipboard FAILED: {clipboard_decision.reason}"
                            else:
                                observe_warning = warning

                        try:
                            if action_error is not None:
                                raise RuntimeError(action_error)
                            if sys.platform == "darwin":
                                proc = subprocess.run(
                                    ["pbpaste"],
                                    capture_output=True,
                                    text=True,
                                    timeout=5,
                                )
                                if proc.returncode != 0:
                                    stderr = (proc.stderr or "").strip() or "unknown error"
                                    raise RuntimeError(f"pbpaste failed: {stderr}")
                                content = proc.stdout
                            elif sys.platform.startswith("linux"):
                                _any_tool_found = False
                                for cmd in (
                                    ["xclip", "-selection", "clipboard", "-o"],
                                    ["xsel", "--clipboard", "--output"],
                                ):
                                    try:
                                        proc = subprocess.run(
                                            cmd,
                                            capture_output=True,
                                            text=True,
                                            timeout=5,
                                        )
                                        _any_tool_found = True
                                        if proc.returncode == 0:
                                            content = proc.stdout
                                            break
                                    except FileNotFoundError:
                                        continue
                                else:
                                    _msg = (
                                        "Clipboard read failed (xclip/xsel returned non-zero)"
                                        if _any_tool_found
                                        else "No clipboard tool found (install xclip or xsel)"
                                    )
                                    raise RuntimeError(_msg)
                            elif sys.platform == "win32":
                                proc = subprocess.run(
                                    ["powershell", "-command", "Get-Clipboard"],
                                    capture_output=True,
                                    text=True,
                                    timeout=5,
                                )
                                if proc.returncode != 0:
                                    stderr = (proc.stderr or "").strip() or "unknown error"
                                    raise RuntimeError(f"Get-Clipboard failed: {stderr}")
                                content = proc.stdout
                            else:
                                raise RuntimeError(f"Unsupported platform: {sys.platform}")

                            content = content.rstrip()
                            if content:
                                preview = content if len(content) <= 1000 else f"{content[:1000]}\n... (truncated)"
                                state.last_action_summary = f"read_clipboard: {len(content)} chars\n{preview}"
                            else:
                                state.last_action_summary = "read_clipboard: clipboard is empty"
                            if observe_warning:
                                state.last_action_summary += f"\nsandbox(observe): {observe_warning}"
                            action_success = True
                        except Exception as e:
                            action_error = str(e)
                            state.last_action_summary = f"read_clipboard FAILED: {action_error}"

                        self._record_controller_action(
                            action_type="read_clipboard",
                            action_step=action_step,
                            success=action_success,
                            error_message=action_error,
                            action_params={"content_length": len(content)},
                            before_state=before_state,
                            after_state=self.memory_store._capture_current_state(),
                        )
                        _append_recent_action("read_clipboard")
                        state.actions_since_progress += 1
                        if action_success:
                            state.user_facing_actions_since_progress += 1
                        _set_checkpoint_pending(True)
                        self.event_logger.action_complete(
                            tool=function_name,
                            narrative=narrative,
                            success=action_success,
                            result_str="success" if action_success else "failed",
                            duration_ms=0.0,
                            iteration=self._current_iteration,
                        )
                        continue

                    if function_name == "open_url":
                        url = str(action_args.get("url", "")).strip()
                        url_decision = self.sandbox_policy.check_url(url)
                        if not url_decision.allowed:
                            warning = f"open_url blocked by sandbox: {url_decision.reason}"
                            self.event_logger.system_warning(warning)
                            if self.sandbox_policy.enforce:
                                before_state = self.memory_store._capture_current_state()
                                state.last_action_summary = f"open_url FAILED: {url_decision.reason}"
                                self._record_controller_action(
                                    action_type=InteractionType.NAVIGATION.value,
                                    action_step=action_step,
                                    success=False,
                                    error_message=warning,
                                    action_params={
                                        "operation": "open_url",
                                        "url": url,
                                        "sandbox_blocked": True,
                                    },
                                    before_state=before_state,
                                    after_state=self.memory_store._capture_current_state(),
                                )
                                _append_recent_action(state.last_action_summary)
                                state.actions_since_progress += 1
                                _set_checkpoint_pending(True)
                                self.event_logger.action_complete(
                                    tool=function_name,
                                    narrative=narrative,
                                    success=False,
                                    result_str="failed",
                                    duration_ms=0.0,
                                    iteration=self._current_iteration,
                                )
                                continue
                            policy_observe_warning = warning

                    result = self.action_executor.act(
                        action_step=action_step,
                        detected_elements=detected_elements,
                        page_info=page_info,
                        environment_state=environment_state,
                        base_knowledge=self.base_knowledge,
                        current_iteration=self._current_iteration,
                    )

                    post_nav_sensitive_functions = {
                        "click",
                        "press_key",
                        "open_url",
                        "go_back",
                        "go_forward",
                    }
                    if result.success and function_name in post_nav_sensitive_functions:
                        try:
                            current_url = self.browser.page.url if self.browser and self.browser.page else ""
                        except Exception:
                            current_url = ""
                        post_nav_decision = self.sandbox_policy.check_url(current_url)
                        if not post_nav_decision.allowed:
                            warning = f"Post-navigation URL blocked by sandbox: {post_nav_decision.reason}"
                            self.event_logger.system_warning(warning)
                            if self.sandbox_policy.enforce:
                                result.success = False
                                result.error = warning
                            else:
                                if policy_observe_warning:
                                    policy_observe_warning = f"{policy_observe_warning} | {warning}"
                                else:
                                    policy_observe_warning = warning

                    state.actions_since_progress += 1
                    result_str = "success" if result.success else "failed"
                    state.last_action_summary = (
                        f"{narrative} ({result_str})"
                        if narrative
                        else self._build_action_summary(action_step, result_str)
                    )
                    if policy_observe_warning:
                        state.last_action_summary += f" | sandbox(observe): {policy_observe_warning}"
                    if result.error:
                        state.last_action_summary += f" | {result.error}"
                    _append_recent_action(state.last_action_summary)
                    _set_checkpoint_pending(True)

                    duration_ms = float((result.metadata or {}).get("duration_ms", 0.0))
                    self.event_logger.action_complete(
                        tool=function_name,
                        narrative=narrative,
                        success=bool(result.success),
                        result_str=result_str,
                        duration_ms=duration_ms,
                        iteration=self._current_iteration,
                    )

                    user_facing_functions = {
                        "click",
                        "type_text",
                        "clear_text",
                        "select_option",
                        "upload_file",
                        "set_datetime",
                        "press_key",
                        "open_url",
                        "go_back",
                        "go_forward",
                        "scroll_page",
                        "extract_data",
                        "report_data",
                        "write_data",
                        "ask_user",
                    }
                    if result.success and function_name in user_facing_functions:
                        if function_name == "report_data":
                            reported = (
                                bool((result.data or {}).get("reported", result.success))
                                if isinstance(result.data, dict)
                                else bool(result.success)
                            )
                            if reported:
                                state.user_facing_actions_since_progress += 1
                        else:
                            state.user_facing_actions_since_progress += 1
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
                                timestamp=time.time(),
                            )
                            state.failed_elements.append(failed_action)
                        except Exception:
                            pass

                state.validation_failures = 0

            except Exception as e:
                return self._build_mission_result(
                    success=False,
                    reasoning=f"Error: {str(e)}",
                    narrative="Error",
                    state=state,
                )
            finally:
                iteration_duration_ms = (time.time() - iteration_started_at) * 1000.0
                self.event_logger.iteration_complete(
                    iteration=self._current_iteration,
                    duration_ms=iteration_duration_ms,
                )

        return self._build_mission_result(
            success=False,
            reasoning=f"Max actions ({max_actions}) reached without completion",
            narrative="Max actions reached without completion",
            state=state,
        )
