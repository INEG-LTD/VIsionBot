"""
Vision Bot - Clean modular version.
"""
from __future__ import annotations

import os
import re
import time
import traceback
import uuid
from typing import Any, Optional, List, Dict, Tuple, Union, Type, Callable

from playwright.sync_api import Browser, Page, Playwright
from playwright_stealth import stealth_sync

from models import VisionPlan, PageElements
from models.core_models import ActionStep, ActionType, PageInfo
from element_detection import ElementDetector
from element_detection.overlay_manager import OverlayManager
from action_executor import ActionExecutor
from utils import PageUtils
from action_queue import ActionQueue
from session_tracker import SessionTracker
# Removed goal imports - goals system no longer used
from planner.plan_generator import PlanGenerator
from utils.intent_parsers import (
    extract_click_target,
    extract_press_target,
    parse_action_intent,
    parse_keyword_command,
    parse_focus_command,
    parse_undo_command,
)
from focus_manager import FocusManager
from interaction_deduper import InteractionDeduper
from utils.bot_logger import get_logger, LogLevel, LogCategory
from utils.semantic_targets import SemanticTarget, build_semantic_target
from utils.event_logger import EventLogger, set_event_logger
from gif_recorder import GIFRecorder
from command_ledger import CommandLedger
from ai_utils import (
    ReasoningLevel,
    set_default_model,
    set_default_reasoning_level,
    set_default_agent_model,
    set_default_agent_reasoning_level,
)
from agent import AgentController
from agent.agent_result import AgentResult
from pydantic import BaseModel, Field
from bot_config import BotConfig
from browser_provider import BrowserProvider, create_browser_provider
from middleware import MiddlewareManager, ActionContext


class ExecutionTimer:
    """Tracks execution timings for tasks, iterations, and commands"""
    
    def __init__(self):
        self.task_start_time: Optional[float] = None
        self.task_end_time: Optional[float] = None
        self.iterations: List[Dict[str, float]] = []  # List of {start, end} dicts
        self.commands: List[Dict[str, Any]] = []  # List of {command_id, command, start, end} dicts
        self.current_iteration_start: Optional[float] = None
        self.current_command_id: Optional[str] = None
        self.current_command_start: Optional[float] = None
        self._current_command_text: str = ""
    
    def start_task(self) -> None:
        """Start tracking task execution"""
        self.task_start_time = time.time()
        self.iterations = []
        self.commands = []
    
    def end_task(self) -> None:
        """End task tracking"""
        self.task_end_time = time.time()
        # End any active iteration or command
        if self.current_iteration_start is not None:
            self.end_iteration()
        if self.current_command_start is not None:
            self.end_command()
    
    def start_iteration(self) -> None:
        """Start tracking an iteration"""
        # End previous iteration if still active
        if self.current_iteration_start is not None:
            self.end_iteration()
        self.current_iteration_start = time.time()
    
    def end_iteration(self) -> None:
        """End current iteration tracking"""
        if self.current_iteration_start is not None:
            self.iterations.append({
                "start": self.current_iteration_start,
                "end": time.time()
            })
            self.current_iteration_start = None
    
    def start_command(self, command_id: str, command: str) -> None:
        """Start tracking a command"""
        # End previous command if still active
        if self.current_command_start is not None:
            self.end_command()
        self.current_command_id = command_id
        self.current_command_start = time.time()
    
    def end_command(self) -> None:
        """End current command tracking"""
        if self.current_command_start is not None and self.current_command_id is not None:
            self.commands.append({
                "command_id": self.current_command_id,
                "command": getattr(self, "_current_command_text", ""),
                "start": self.current_command_start,
                "end": time.time()
            })
            self.current_command_id = None
            self.current_command_start = None
            self._current_command_text = ""
    
    def set_command_text(self, text: str) -> None:
        """Set the command text for the current command"""
        self._current_command_text = text
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all timings"""
        summary = {
            "task": {},
            "iterations": [],
            "commands": []
        }
        
        # Task timing
        if self.task_start_time and self.task_end_time:
            task_duration = self.task_end_time - self.task_start_time
            summary["task"] = {
                "duration_seconds": round(task_duration, 2),
                "duration_formatted": self._format_duration(task_duration)
            }
        
        # Iteration timings
        for i, iter_data in enumerate(self.iterations, 1):
            duration = iter_data["end"] - iter_data["start"]
            summary["iterations"].append({
                "iteration": i,
                "duration_seconds": round(duration, 2),
                "duration_formatted": self._format_duration(duration)
            })
        
        # Command timings
        for cmd_data in self.commands:
            duration = cmd_data["end"] - cmd_data["start"]
            summary["commands"].append({
                "command_id": cmd_data["command_id"],
                "command": cmd_data.get("command", ""),
                "duration_seconds": round(duration, 2),
                "duration_formatted": self._format_duration(duration)
            })
        
        return summary
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in a human-readable way"""
        if seconds < 1:
            return f"{int(seconds * 1000)}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.2f}s"
    
    def log_summary(self) -> None:
        """Log timing summary to console"""
        summary = self.get_summary()
        
        try:
            self.event_logger.system_info("\n" + "="*60)
            self.event_logger.system_info("⏱️  EXECUTION TIMING SUMMARY")
            self.event_logger.system_info("="*60)
            
            # Task timing
            if summary["task"]:
                self.event_logger.system_info(f"\n📋 Task Duration: {summary['task']['duration_formatted']} ({summary['task']['duration_seconds']}s)")
            
            # Iteration timings
            if summary["iterations"]:
                total_iter_time = sum(iter_data["duration_seconds"] for iter_data in summary["iterations"])
                avg_iter_time = total_iter_time / len(summary["iterations"])
                self.event_logger.system_info(f"\n🔄 Iterations: {len(summary['iterations'])}")
                self.event_logger.system_info(f"   Total iteration time: {self._format_duration(total_iter_time)}")
                self.event_logger.system_info(f"   Average per iteration: {self._format_duration(avg_iter_time)}")
                fastest_iter = min(summary['iterations'], key=lambda x: x['duration_seconds'])
                slowest_iter = max(summary['iterations'], key=lambda x: x['duration_seconds'])
                self.event_logger.system_info(f"   Fastest iteration: {fastest_iter['duration_formatted']}")
                self.event_logger.system_info(f"   Slowest iteration: {slowest_iter['duration_formatted']}")
            
            # Command timings
            if summary["commands"]:
                total_cmd_time = sum(cmd_data["duration_seconds"] for cmd_data in summary["commands"])
                avg_cmd_time = total_cmd_time / len(summary["commands"])
                self.event_logger.system_info(f"\n🎯 Commands: {len(summary['commands'])}")
                self.event_logger.system_info(f"   Total command time: {self._format_duration(total_cmd_time)}")
                self.event_logger.system_info(f"   Average per command: {self._format_duration(avg_cmd_time)}")
                
                # Show top 5 slowest commands
                sorted_commands = sorted(summary["commands"], key=lambda x: x["duration_seconds"], reverse=True)
                self.event_logger.system_info("\n   Top 5 slowest commands:")
                for i, cmd in enumerate(sorted_commands[:5], 1):
                    cmd_text = cmd["command"][:50] + "..." if len(cmd.get("command", "")) > 50 else cmd.get("command", "")
                    self.event_logger.system_info(f"   {i}. {cmd['command_id']}: {cmd['duration_formatted']} - {cmd_text}")
            
            self.event_logger.system_info("="*60 + "\n")
        except Exception:
            pass


class BrowserVisionBot:
    """Modular vision-based web automation bot"""


    def __init__(
        self,
        config: Optional[BotConfig] = None,
        browser_provider: Optional[BrowserProvider] = None,
        page: Page = None,  # Deprecated: use browser_provider instead
        event_logger: Optional[EventLogger] = None,
    ):
        """
        Initialize BrowserVisionBot.
        
        Args:
            config: BotConfig object with all settings. If not provided, uses defaults.
            browser_provider: BrowserProvider implementation. If not provided, creates from config.
            page: (Deprecated) Optional Playwright Page object. Use browser_provider instead.
            event_logger: Optional custom event logger. If not provided, creates default logger.
        """
        # Create default config if not provided
        if config is None:
            config = BotConfig()
        
        # Handle browser provider
        if browser_provider is None:
            # Create provider from config
            browser_provider = create_browser_provider(config.browser)
        
        self.browser_provider = browser_provider
        self.page = page  # Will be set in start() if None
        
        # Initialize middleware manager
        self.middleware = MiddlewareManager()

        # Extract model configuration
        self.command_model_name = config.model.command_model
        self.agent_model_name = config.model.agent_model
        self.model_name = self.command_model_name

        self.command_reasoning_level = ReasoningLevel.coerce(config.model.command_reasoning_level)
        self.agent_reasoning_level = ReasoningLevel.coerce(config.model.agent_reasoning_level)
        self.reasoning_level = self.command_reasoning_level

        # Set the centralized model configuration
        set_default_model(self.command_model_name)
        set_default_reasoning_level(self.command_reasoning_level)
        set_default_agent_model(self.agent_model_name)
        set_default_agent_reasoning_level(self.agent_reasoning_level)
        
        # Extract execution configuration
        self.max_attempts = config.execution.max_attempts
        self.parallel_completion_and_action = config.execution.parallel_completion_and_action
        self.dedup_mode = config.execution.dedup_mode
        
        self.started = False
        
        # Extract element configuration
        self.max_detailed_elements = config.elements.max_detailed_elements
        self.include_detailed_elements = config.elements.include_detailed_elements
        self.two_pass_planning = config.elements.two_pass_planning
        self.max_coordinate_overlays = config.elements.max_coordinate_overlays
        self.merge_overlay_selection = config.elements.merge_overlay_selection
        self.overlay_only_planning = config.elements.overlay_only_planning
        
        _overlay_max_samples = config.elements.overlay_selection_max_samples
        self.overlay_selection_max_samples = (
            None if _overlay_max_samples is not None and _overlay_max_samples <= 0 else _overlay_max_samples
        )
        
        # Removed goal evaluation overrides (no longer needed without goals)
        
        # Extract cache configuration
        self.plan_cache_enabled = config.cache.enabled
        self.plan_cache_ttl = max(config.cache.ttl, 0.0)
        
        try:
            self.plan_cache_max_reuse = int(config.cache.max_reuse)
        except (TypeError, ValueError):
            self.plan_cache_max_reuse = 1
        if self.plan_cache_max_reuse < -1:
            self.plan_cache_max_reuse = -1
        self._plan_cache_entry: Optional[Dict[str, Any]] = None
        
        # Element selection retry configuration
        self.element_selection_retry_attempts = max(1, int(config.elements.selection_retry_attempts))
        self.element_selection_fallback_model = config.elements.selection_fallback_model
        
        # Extract stuck detector configuration
        self.stuck_detector_enabled = config.stuck_detector.enabled
        self.stuck_detector_window_size = config.stuck_detector.window_size
        self.stuck_detector_threshold = config.stuck_detector.threshold
        self.stuck_detector_weight_repeated_action = config.stuck_detector.weight_repeated_action
        self.stuck_detector_weight_repetitive_action_no_change = config.stuck_detector.weight_repetitive_action_no_change
        self.stuck_detector_weight_no_state_change = config.stuck_detector.weight_no_state_change
        self.stuck_detector_weight_no_progress = config.stuck_detector.weight_no_progress
        self.stuck_detector_weight_error_spiral = config.stuck_detector.weight_error_spiral
        self.stuck_detector_weight_high_confidence_no_progress = config.stuck_detector.weight_high_confidence_no_progress
        
        # Auto-run actions on page load (opt-in)
        self._auto_on_load_enabled: bool = False
        self._auto_on_load_actions: List[str] = []
        self._auto_on_load_run_once_per_url: bool = True
        self._auto_on_load_urls_handled: set[str] = set()
        self._auto_on_load_handler = None  # keep reference for potential removal
        self._auto_on_load_running: bool = False
        self._auto_on_load_event_name: str | None = None
        self._auto_on_load_command_id: Optional[str] = None
        self._in_act: bool = False
        # Queue auto-on-load actions when a page load happens during an active act()
        self._pending_auto_on_load: bool = False
        self._pending_auto_on_load_url: Optional[str] = None
        
        # Command history for "do that again" functionality
        self.command_history: List[str] = []
        self.max_command_history: int = 10  # Keep last 10 commands
        
        # Multi-command reference storage
        self.command_refs: Dict[str, Dict[str, Any]] = {}  # refID -> metadata about stored prompts

        # Action queue system for deferred actions
        self.action_queue = ActionQueue()
        self._auto_process_queue = True  # Auto-process queue after each act()

        # Extract recording configuration
        self.save_gif = config.recording.save_gif
        self.gif_output_dir = config.recording.output_dir
        self.gif_recorder: Optional[GIFRecorder] = None

        # Bot termination state
        self.terminated = False

        # Initialize logger
        self.logger = get_logger()
        
        # Extract debug configuration
        _debug_mode = config.logging.debug_mode
        
        # Initialize event logger early (before any methods that might use it)
        # Create a safe logger that never fails
        class SafeLogger:
            """Fallback logger that accepts any method call and does nothing"""
            def __getattr__(self, name):
                def noop(*args, **kwargs):
                    pass
                return noop
        
        # Try to create real logger, fallback to safe logger
        # This MUST never fail - use SafeLogger as ultimate fallback
        try:
            if event_logger is None:
                event_logger = EventLogger(debug_mode=_debug_mode)
            self.event_logger = event_logger
            try:
                set_event_logger(event_logger)  # Set as global
            except Exception:
                pass  # Ignore global setter errors
        except Exception:
            # Fallback: create a minimal logger if initialization fails
            try:
                self.event_logger = EventLogger(debug_mode=True)
                try:
                    set_event_logger(self.event_logger)
                except Exception:
                    pass
            except Exception:
                # Ultimate fallback - safe logger that does nothing
                self.event_logger = SafeLogger()

        # Deduplication history settings (-1 = unlimited)
        self.dedup_history_quantity: int = config.execution.dedup_history_quantity

        # Interpretation / semantic resolution helpers
        self.default_interpretation_mode: str = "literal"
        self._interpretation_mode_stack: List[str] = []
        self._semantic_target_cache: Dict[str, Optional[SemanticTarget]] = {}
        
        # Deferred input handling
        self.defer_input_handler: Optional[Callable[[str, Any], str]] = None
        self._pending_defer_input: Optional[Dict[str, Any]] = None
        
        # Execution timer for tracking task, iteration, and command timings
        self.execution_timer = ExecutionTimer()
    
    def _safe_event_log(self, method_name: str, *args, **kwargs):
        """Safely call any event logger method - never raises exceptions"""
        try:
            method = getattr(self.event_logger, method_name, None)
            if method:
                method(*args, **kwargs)
        except Exception:
            pass  # Silently ignore all event logger errors
        
    def init_browser(self) -> tuple[Playwright, Browser, Page]:
        # Local import to avoid dependency when not running as script
        from playwright.sync_api import sync_playwright
        p = sync_playwright().start()
        browser = p.chromium.launch_persistent_context(
                    viewport={"width": 1280, "height": 800},
                    user_data_dir=os.path.expanduser(f"~/Library/Application Support/Google/Chrome/Automation_{str(uuid.uuid4())[:8]}"),
                    headless=False,
                    args=[
                        "--no-sandbox", 
                        "--disable-dev-shm-usage", 
                        "--disable-blink-features=AutomationControlled",
                        '--disable-features=VizDisplayCompositor',
                        f"--window-position={0},{0}",
                        f"--window-size={1280},{800}",
                        ],
                    channel="chrome"
                )
        
                    
        pages = browser.pages
        if pages:
            page = pages[0]
        else:
            page = browser.new_page()
        
        # Apply stealth to the page
        stealth_sync(page)
        
        return p, browser, page
    
    # ---------- Plan caching helpers ----------
    def _invalidate_plan_cache(self, reason: str = "") -> None:
        """Clear any cached plan."""
        if not self.plan_cache_enabled:
            return
        if not self._plan_cache_entry:
            return
        if reason:
            try:
                self.event_logger.plan_cleared(reason=reason)
            except Exception:
                pass
        self._plan_cache_entry = None

    def _store_plan_in_cache(
        self,
        *,
        goal_description: str,
        dom_signature: str,
        page_info: PageInfo,
        plan: VisionPlan,
        additional_context: str,
        interpretation_mode: str,
        agent_mode: bool,
        target_context_guard: Optional[str],
    ) -> None:
        """Store a deep copy of the plan so we can reuse it without another LLM call."""
        if not self.plan_cache_enabled:
            return
        if not plan or not getattr(plan, "action_steps", None):
            return
        description_key = (goal_description or "").strip()
        context_key = (additional_context or "").strip()
        guard_key = (target_context_guard or "").strip()
        short_sig = dom_signature[:8] if dom_signature else "none"
        self._plan_cache_entry = {
            "goal": description_key,
            "dom_signature": dom_signature,
            "page_url": page_info.url,
            "timestamp": time.time(),
            "plan": plan.copy(deep=True),
            "reuse_count": 0,
            "additional_context": context_key,
            "interpretation_mode": interpretation_mode,
            "agent_mode": agent_mode,
            "target_context_guard": guard_key,
        }
        try:
            self.event_logger.plan_cached(goal_description=description_key, signature=short_sig)
        except Exception:
            pass

    def _get_cached_plan(
        self,
        *,
        goal_description: str,
        dom_signature: str,
        page_info: PageInfo,
        additional_context: str,
        interpretation_mode: str,
        agent_mode: bool,
        target_context_guard: Optional[str],
    ) -> Optional[VisionPlan]:
        """Return a cached plan copy when all context still matches."""
        if not self.plan_cache_enabled:
            return None
        entry = self._plan_cache_entry
        if not entry:
            return None
        if self.plan_cache_ttl > 0 and (time.time() - entry["timestamp"]) > self.plan_cache_ttl:
            self._invalidate_plan_cache("expired")
            return None
        description_key = (goal_description or "").strip()
        context_key = (additional_context or "").strip()
        guard_key = (target_context_guard or "").strip()
        if entry["goal"] != description_key:
            return None
        if entry["dom_signature"] != dom_signature:
            return None
        if entry["page_url"] != page_info.url:
            return None
        if entry["additional_context"] != context_key:
            return None
        if entry["interpretation_mode"] != interpretation_mode:
            return None
        if entry["agent_mode"] != agent_mode:
            return None
        if entry["target_context_guard"] != guard_key:
            return None
        if self.plan_cache_max_reuse >= 0 and entry["reuse_count"] >= self.plan_cache_max_reuse:
            return None
        entry["reuse_count"] += 1
        entry["timestamp"] = time.time()
        try:
            self.event_logger.plan_reused()
        except Exception:
            pass
        return entry["plan"].copy(deep=True)
    
    def set_defer_input_handler(self, handler: Optional[Callable[[str, Any], str]]) -> None:
        """Register a custom handler for defer goals that request user input."""
        self.defer_input_handler = handler

    def _default_defer_input_handler(self, prompt: str, context: Any) -> str:
        try:
            message = prompt.strip() if prompt and prompt.strip() else "Please provide the requested input to continue."
            return input(f"{message}\n> ")
        except EOFError:
            try:
                self.event_logger.system_warning("[Defer] Warning: input stream unavailable; returning empty response.")
            except Exception:
                pass
            return ""

    def _request_defer_input(self, prompt: str, context: Any) -> str:
        handler = self.defer_input_handler or self._default_defer_input_handler
        response = handler(prompt, context)
        if response is None:
            response = ""
        response_str = str(response)
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "response": response_str,
            "timestamp": time.time(),
        }
        if context and getattr(context, "current_state", None):
            state = context.current_state
            payload["page_url"] = getattr(state, "url", None)
            payload["page_title"] = getattr(state, "title", None)
        self._pending_defer_input = payload
        return response_str

    def consume_last_defer_input(self) -> Optional[Dict[str, Any]]:
        """Return and clear the most recent defer input payload, if any."""
        payload = self._pending_defer_input
        self._pending_defer_input = None
        return payload
    
    def use(self, middleware) -> 'BrowserVisionBot':
        """
        Add middleware to the bot.
        
        Args:
            middleware: Middleware instance to add
            
        Returns:
            Self for method chaining
            
        Example:
            >>> bot.use(LoggingMiddleware()) \\
            ...    .use(CostTrackingMiddleware(max_cost=1.00))
        """
        self.middleware.use(middleware)
        return self
    
    def start(self) -> None:
        """Start the bot"""
        # Get page from browser provider if not already provided
        if self.page is None:
            self.page = self.browser_provider.get_page()
        
        # Try to get browser reference for compatibility
        try:
            self.browser = self.page.context.browser
        except Exception:
            self.browser = None
        
        # State tracking
        self.current_attempt = 0
        self.last_screenshot_hash = None
        self.last_dom_signature = None
        self._agent_mode = False  # Flag to track if we're in agent mode (disables DOM unchanged scroll check)
        
        # Screenshot and overlay caching for performance optimization
        self._cached_screenshot_with_overlays = None
        self._cached_clean_screenshot = None
        self._cached_overlay_data = None
        self._cached_dom_signature = None
        self._cached_focus_context = None
        self._pre_dedup_element_data = []
        
        # Use self.page (which may have been provided or just initialized)
        page = self.page
        
        # Initialize components
        self.session_tracker: SessionTracker = SessionTracker(page)
        self.overlay_manager = OverlayManager(page)
        
        # Initialize tab management (Phase 1)
        try:
            from tab_management import TabManager
            if page and hasattr(page, 'context'):
                self.tab_manager: Optional[TabManager] = TabManager(page.context)
                # Register the initial page
                self.tab_manager.register_tab(
                    page=page,
                    purpose="main",
                    agent_id=None,
                    metadata={"initial": True}
                )
            else:
                self.tab_manager: Optional[TabManager] = None
        except Exception as e:
            try:
                self.event_logger.system_error("Failed to initialize TabManager", error=e)
            except Exception:
                pass
            self.tab_manager: Optional[TabManager] = None
        self.element_detector = ElementDetector(model_name=self.model_name)
        self.page_utils = PageUtils(page)
        
        # Initialize deduplication system
        self.deduper = InteractionDeduper()
        try:
            self.deduper.set_interaction_history_limit(self.dedup_history_quantity)
        except Exception:
            pass
        
        # Initialize focus manager with deduper
        self.focus_manager: FocusManager = FocusManager(page, self.page_utils, self.deduper)
        
        # Initialize command ledger for tracking command execution
        self.command_ledger: CommandLedger = CommandLedger()
        
        # Initialize GIF recorder if enabled (BEFORE ActionExecutor)
        if self.save_gif:
            self.gif_recorder = GIFRecorder(page, self.gif_output_dir)
            self.gif_recorder.start_recording()
            try:
                self.event_logger.gif_start()
            except Exception:
                pass
        
        # Initialize action executor with deduper, GIF recorder, and command ledger
        self.action_executor: ActionExecutor = ActionExecutor(page, self.session_tracker, self.page_utils, self.deduper, self.gif_recorder, self.command_ledger)
        
        # Set action_executor on focus_manager for scroll tracking
        self.focus_manager.action_executor = self.action_executor
        
        # Plan generator for AI planning prompts
        self.plan_generator: PlanGenerator = PlanGenerator(
            include_detailed_elements=self.include_detailed_elements,
            max_detailed_elements=self.max_detailed_elements,
            merge_overlay_selection=self.merge_overlay_selection,
            return_overlay_only=self.overlay_only_planning,
            overlay_selection_max_samples=self.overlay_selection_max_samples,
        )
        
        # Auto-switch to new tabs/windows when they open (e.g., target=_blank)
        try:
            self._attach_new_page_listener()
        except Exception as e:
            try:
                self.event_logger.system_error("Failed to attach new page listener", error=e)
            except Exception:
                pass

        self.started = True

        # If auto-on-load was enabled before start, attach handler now
        try:
            if self._auto_on_load_enabled:
                self._attach_page_load_handler()
        except Exception:
            pass

    def stop_gif_recording(self) -> Optional[str]:
        """Stop GIF recording and return the path to the generated GIF"""
        if not self.save_gif or not self.gif_recorder:
            return None
            
        gif_path = self.gif_recorder.stop_recording()
        self.gif_recorder = None
        return gif_path

    def end(self) -> Optional[str]:
        """
        Terminate the bot, stop GIF recording, and prevent any subsequent operations.
        
        Returns:
            Optional[str]: Path to the generated GIF if recording was enabled, None otherwise
        """
        if self.terminated:
            try:
                self.event_logger.system_warning("Bot is already terminated")
            except Exception:
                pass
            return None
            
        try:
            self.event_logger.system_info("Terminating bot...")
        except Exception:
            pass
        
        # Stop GIF recording first
        gif_path = None
        if self.save_gif and self.gif_recorder:
            try:
                self.event_logger.system_info("Stopping GIF recording...")
            except Exception:
                pass
            gif_path = self.gif_recorder.stop_recording()
            self.gif_recorder = None
            if gif_path:
                try:
                    self.event_logger.gif_stop(gif_path=gif_path)
                except Exception:
                    pass
        
        # Close browser provider and cleanup
        try:
            if hasattr(self, 'browser_provider') and self.browser_provider:
                try:
                    self.event_logger.system_info("Closing browser...")
                except Exception:
                    pass
                self.browser_provider.close()
        except Exception as e:
            try:
                self.event_logger.system_error("Error closing browser", error=e)
            except Exception:
                pass
        
        # Mark as terminated
        self.terminated = True
        self.started = False
        
        try:
            self.event_logger.system_info("Bot terminated successfully")
        except Exception:
            pass
        
        if gif_path:
            try:
                self.event_logger.system_info(f"GIF recording available at: {gif_path}")
            except Exception:
                pass
        
        return gif_path

    def _check_termination(self) -> None:
        """Check if bot is terminated and raise error if so"""
        if self.terminated:
            raise RuntimeError("Bot has been terminated. No further operations are allowed.")

    def _attach_new_page_listener(self) -> None:
        """Attach a browser-context listener to detect new pages/tabs and switch context automatically."""
        if not self.page:
            return
        ctx = self.page.context

        def _on_new_page(new_page: Page) -> None:
            try:
                # Bring to front and wait for basic readiness
                try:
                    new_page.bring_to_front()
                except Exception:
                    pass
                try:
                    new_page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
                self.event_logger.system_info("New page/tab detected by context listener → switching…")
                
                # If TabManager is available, detect and register the new tab
                if self.tab_manager:
                    tab_id = self.tab_manager.detect_new_tab(new_page)
                    if tab_id:
                        self.event_logger.tab_new(tab_id=tab_id, url=new_page.url)
                
                self.switch_to_page(new_page)
            except Exception as e:
                self.event_logger.system_error("Error handling new page event", error=e)

        try:
            ctx.on("page", _on_new_page)
        except Exception as e:
            self.event_logger.system_error("Could not register context 'page' listener", error=e)

    def switch_to_page(self, new_page: Page) -> None:
        """Switch all components to a different active Page (new tab/window)."""
        if not new_page or new_page is self.page:
            return
        try:
            # Set a reasonable default timeout for snappy interactions on the new page
            try:
                new_page.set_default_timeout(2000)
            except Exception:
                pass

            # Update TabManager if available
            if self.tab_manager:
                # Find tab ID for this page
                tab_id = None
                for tid, tab_info in self.tab_manager.tabs.items():
                    if id(tab_info.page) == id(new_page):
                        tab_id = tid
                        break
                
                # If not found, detect and register
                if tab_id is None:
                    tab_id = self.tab_manager.detect_new_tab(new_page)
                
                # Switch to this tab in TabManager
                if tab_id:
                    self.tab_manager.switch_to_tab(tab_id)

            self.page = new_page
            # Update core components
            try:
                if self.page_utils:
                    if hasattr(self.page_utils, "set_page"):
                        self.page_utils.set_page(new_page)
                    else:
                        self.page_utils.page = new_page
            except Exception:
                pass
            try:
                if self.session_tracker:
                    self.session_tracker.switch_to_page(new_page)
            except Exception:
                pass
            try:
                if self.action_executor:
                    self.action_executor.set_page(new_page)
            except Exception:
                pass
            try:
                if hasattr(self, "focus_manager") and self.focus_manager:
                    if hasattr(self.focus_manager, "set_page"):
                        self.focus_manager.set_page(new_page)
                    else:
                        self.focus_manager.page = new_page
            except Exception:
                pass
            # Always refresh overlay manager for the new page context
            try:
                self.overlay_manager = OverlayManager(new_page)
            except Exception:
                pass

            # Clear cached overlay/screenshot data tied to previous page
            try:
                self._cached_screenshot_with_overlays = None
                self._cached_overlay_data = None
                self._cached_clean_screenshot = None
                self._cached_dom_signature = None
                self.last_dom_signature = None
            except Exception:
                pass

            self.event_logger.tab_switch(tab_id=str(id(new_page)), url=getattr(new_page, 'url', ''))
            # Re-attach auto-on-load handler for the new page if feature is enabled
            try:
                if self._auto_on_load_enabled:
                    self._attach_page_load_handler()
                    # Also run immediately for the new page so it happens before next act()
                    self._run_auto_actions_for_current_page()
            except Exception:
                pass
        except Exception as e:
            self.event_logger.system_error("Failed to switch to new page", error=e)

    # ---------- Auto actions on page load ----------
    def on_new_page_load(self, actions_to_take: List[str], run_once_per_url: bool = True, command_id: Optional[str] = None) -> None:
        """
        Register prompts to run via act() after each page load.

        Args:
            actions_to_take: List of commands to run on each page load
            run_once_per_url: If True, only run once per unique URL
            command_id: Optional command ID for tracking (auto-generated if not provided)

        Typical usage: on_new_page_load([
            "if a cookie banner is visible click the accept button",
            "close any newsletter modal if present",
        ], command_id="auto-page-load")
        """
        self._auto_on_load_actions = [a for a in (actions_to_take or []) if isinstance(a, str) and a.strip()]
        self._auto_on_load_run_once_per_url = bool(run_once_per_url)
        self._auto_on_load_enabled = True
        self._auto_on_load_urls_handled.clear()
        self._auto_on_load_command_id = command_id
        # Attach to current page if available
        try:
            self._attach_page_load_handler()
        except Exception:
            pass
        
        # If page is already loaded, run actions immediately before any next act()
        try:
            if self.page:
                state = self.page.evaluate("document.readyState").lower()
                if state in ("interactive", "complete"):
                    # If an act() is in progress, queue instead of running now
                    if self._in_act:
                        try:
                            self._pending_auto_on_load = True
                            self._pending_auto_on_load_url = self.page.url
                        except Exception:
                            pass
                    else:
                        self._run_auto_actions_for_current_page()
        except Exception:
            # If evaluate fails, best-effort attempt anyway
            try:
                if self._in_act:
                    self._pending_auto_on_load = True
                    try:
                        self._pending_auto_on_load_url = self.page.url
                    except Exception:
                        pass
                else:
                    self._run_auto_actions_for_current_page()
            except Exception:
                pass

    def _attach_page_load_handler(self) -> None:
        """Attach a 'load' event handler on the active page to auto-run actions."""
        if not self.page:
            return
        # Remove previous handler to avoid duplicates
        try:
            if self._auto_on_load_handler:
                evt = self._auto_on_load_event_name or "load"
                self.page.off(evt, self._auto_on_load_handler)
        except Exception:
            pass

        def _on_load(_: object = None) -> None:
            if not self._auto_on_load_enabled:
                return
            if not self._auto_on_load_actions:
                return
            if self._auto_on_load_running:
                # Avoid re-entrancy while actions themselves cause navigations
                return
            # If an act() is currently in progress, defer auto actions until act completes
            if self._in_act:
                try:
                    self._pending_auto_on_load = True
                    try:
                        self._pending_auto_on_load_url = self.page.url
                    except Exception:
                        self._pending_auto_on_load_url = None
                except Exception:
                    # Best-effort: if queueing fails, skip to avoid disrupting current act
                    pass
                return
            # Otherwise run now
            self._run_auto_actions_for_current_page()

        # Keep reference for removal and attach
        self._auto_on_load_handler = _on_load
        try:
            self.page.on("load", self._auto_on_load_handler)
            self._auto_on_load_event_name = "load"
        except Exception:
            # Fallback: try DOMContentLoaded if 'load' fails in some contexts
            try:
                self.page.on("domcontentloaded", self._auto_on_load_handler)
                self._auto_on_load_event_name = "domcontentloaded"
            except Exception:
                self._auto_on_load_event_name = None
                pass

    def _should_run_auto_actions_now(self, url: str) -> bool:
        if not self._auto_on_load_enabled:
            return False
        if not self._auto_on_load_actions:
            return False
        if self._auto_on_load_running:
            return False
        if self._auto_on_load_run_once_per_url and url and url in self._auto_on_load_urls_handled:
            return False
        return True

    def _run_auto_actions_for_current_page(self) -> None:
        """Run auto actions synchronously for the current page if due."""
        current_url = ""
        try:
            current_url = self.page.url or ""
        except Exception:
            pass

        if not self._should_run_auto_actions_now(current_url):
            return

        # Mark handled early to avoid duplicate runs
        if self._auto_on_load_run_once_per_url and current_url:
            self._auto_on_load_urls_handled.add(current_url)

        self._auto_on_load_running = True
        try:
            # Register parent command if we have a command_id
            parent_cmd_id = None
            if hasattr(self, '_auto_on_load_command_id') and self._auto_on_load_command_id:
                parent_cmd_id = self.command_ledger.register_command(
                    command=f"on_new_page_load: {len(self._auto_on_load_actions)} actions",
                    command_id=self._auto_on_load_command_id,
                    metadata={"source": "on_new_page_load", "url": current_url}
                )
                self.command_ledger.start_command(parent_cmd_id)
            
            for i, prompt in enumerate(self._auto_on_load_actions, 1):
                try:
                    try:
                        self.event_logger.system_info(f"Auto-on-load: act('{prompt}')")
                    except Exception:
                        pass
                    # Snapshot user prompt so auto-action does not disrupt ongoing task
                    saved_user_prompt = getattr(self.session_tracker, 'user_prompt', "") if hasattr(self, 'session_tracker') else ""
                    
                    # Generate child command ID if we have a parent
                    child_cmd_id = f"{parent_cmd_id}_action{i}" if parent_cmd_id else None
                    
                    try:
                        # If we're inside act(), we still call act() but our snapshot/restore prevents disruption
                        # and _auto_on_load_running avoids re-entrancy loops
                        self.act(prompt, command_id=child_cmd_id)
                    finally:
                        # Restore previous user prompt if it existed prior
                        try:
                            if hasattr(self, 'session_tracker') and saved_user_prompt:
                                self.session_tracker.set_user_prompt(saved_user_prompt)
                        except Exception:
                            pass
                except Exception as e:
                    try:
                        self.event_logger.system_error("Auto-on-load action failed", error=e)
                    except Exception:
                        pass
            
            # Complete parent command
            if parent_cmd_id:
                self.command_ledger.complete_command(parent_cmd_id, success=True)
        finally:
            self._auto_on_load_running = False

    def _flush_pending_auto_on_load(self) -> None:
        """Run any pending auto-on-load actions queued during act()."""
        if not self._pending_auto_on_load:
            return
        # Optional: only run if URL hasn't been handled yet (respects run-once-per-url)
        try:
            self._run_auto_actions_for_current_page()
        except Exception:
            pass
        finally:
            self._pending_auto_on_load = False
            self._pending_auto_on_load_url = None


    def _try_simple_goal_bypass(self, command_id: Optional[str] = None) -> Optional[bool]:
        """Fast path: if all active goals are simple (Press/Scroll), execute directly without LLM.

        Args:
            command_id: Optional command ID to track this execution
        
        Returns True/False if executed, or None to fall back to normal planning.
        """
        # Goal system removed - simple goal bypass no longer supported
        return None

    def _try_direct_click_bypass(self, prompt_text: str) -> Optional[bool]:
        """Direct DOM bypass removed; rely on vision planning."""
        return None

    def _try_direct_type_bypass(self, prompt_text: str) -> Optional[bool]:
        """Direct DOM bypass removed; rely on vision planning."""
        return None

    def _try_direct_select_bypass(self, prompt_text: str) -> Optional[bool]:
        """Direct DOM bypass removed; rely on vision planning."""
        return None

    def _try_direct_datetime_bypass(self, prompt_text: str) -> Optional[bool]:
        """Direct DOM bypass removed; rely on vision planning."""
        return None

    def _try_direct_upload_bypass(self, prompt_text: str) -> Optional[bool]:
        """Direct DOM bypass removed; rely on vision planning."""
        return None

    def act(
        self,
        goal_description: str,
        additional_context: str = "",
        interpretation_mode: Optional[str] = None,
        target_context_guard: Optional[str] = None,
        skip_post_guard_refinement: bool = True,
        confirm_before_interaction: bool = False,
        command_id: Optional[str] = None,
        modifier: Optional[List[str]] = None,
        max_attempts: Optional[int] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> bool:
        """
        Main method to achieve a goal using vision-based automation
        
        Args:
            goal_description: The goal to achieve
            additional_context: Extra context to help with planning
            interpretation_mode: Vision interpretation mode
            target_context_guard: Guard condition for actions
            skip_post_guard_refinement: Skip refinement after guard checks
            confirm_before_interaction: Require user confirmation before each action
            command_id: Optional command ID for tracking (auto-generated if not provided)
            modifier: Optional list of modifier strings to pass to goals
            max_attempts: Override bot's max_attempts for this command (None = use bot default)
            max_retries: Override goal's max_retries for this command (None = use goal default)
        
        Returns:
            True if goal was achieved, False otherwise
        """
        self._check_termination()
        
        if interpretation_mode is None:
            resolved_mode = self._get_current_interpretation_mode()
        else:
            resolved_mode = self._normalize_interpretation_mode(interpretation_mode)
        self._interpretation_mode_stack.append(resolved_mode)
        self._in_act = True
        start_time = time.time()
        
        try:
            if not self.started:
                self.logger.log_error("Bot not started", "act() called before bot.start()")
                self.event_logger.system_error("Bot not started")
                if self.execution_timer.current_command_start is not None:
                    self.execution_timer.end_command()
                return False
            
            if self.page.url.startswith("about:blank"):
                self.logger.log_error("Page is on initial blank page", "act() called before navigation")
                self.event_logger.system_error("Page is on the initial blank page")
                if self.execution_timer.current_command_start is not None:
                    self.execution_timer.end_command()
                return False
            
            # Register command in ledger
            command_id = self.command_ledger.register_command(
                command=goal_description,
                command_id=command_id,
                metadata={"source": "act", "mode": resolved_mode}
            )
            self.command_ledger.start_command(command_id)
            
            # Start command timer
            self.execution_timer.start_command(command_id, goal_description)
            self.execution_timer.set_command_text(goal_description)
            
            # Log goal start
            self.logger.log_goal_start(goal_description)
            self.event_logger.goal_start(goal_description, command_id=command_id)
            
            # Add command to history
            self._add_to_command_history(goal_description)
            
            # Check for focus commands first
            focus_result = self._handle_focus_commands(goal_description, self.overlay_manager)
            if focus_result is not None:
                self.command_ledger.complete_command(command_id, success=focus_result)
                self.execution_timer.end_command()
                return focus_result
            
            # Check for dedup commands
            dedup_result = self._handle_dedup_commands(goal_description)
            if dedup_result is not None:
                self.command_ledger.complete_command(command_id, success=dedup_result)
                self.execution_timer.end_command()
                return dedup_result
            
            # Check for ref commands
            ref_result = self._handle_ref_commands(goal_description)
            if ref_result is not None:
                self.command_ledger.complete_command(command_id, success=ref_result)
                self.execution_timer.end_command()
                return ref_result
            
            # Check for extract commands
            if goal_description.strip().lower().startswith("extract:"):
                extraction_prompt = goal_description.replace("extract:", "").strip()
                self.event_logger.extraction_start(extraction_prompt)
                
                try:
                    # Perform extraction
                    result = self.extract(
                        prompt=extraction_prompt,
                        output_format="json",
                        scope="viewport"
                    )
                    self.event_logger.extraction_success(extraction_prompt, result=result)
                    self.command_ledger.complete_command(command_id, success=True)
                    self.execution_timer.end_command()
                    return True
                except Exception as e:
                    self.event_logger.extraction_failure(extraction_prompt, error=str(e))
                    self.command_ledger.complete_command(command_id, success=False)
                    self.execution_timer.end_command()
                    return False

            # Reset DOM signature for new command - don't check against previous command's signature
            # This ensures the first attempt of a new goal doesn't get blocked by DOM signature checks
            self.last_dom_signature = None

            keyword_command_result = self._execute_keyword_command(
                goal_description=goal_description,
                additional_context=additional_context,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
                command_id=command_id,
                start_time=start_time,
            )
            if keyword_command_result is not None:
                return keyword_command_result

            # If keyword execution can't handle it, fail immediately
            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_goal_failure(goal_description, "Could not parse command as keyword action", duration_ms)
            print(f"❌ Could not parse command: {goal_description}")
            self.command_ledger.complete_command(command_id, success=False, error_message="Could not parse command as keyword action")
            self.execution_timer.end_command()
            return False
        finally:
            # End command timer if still active (safety net for any unhandled returns)
            if self.execution_timer.current_command_start is not None:
                self.execution_timer.end_command()
            
            # Mark act() as finished and flush any auto-on-load actions that arrived mid-act
            self._in_act = False
            if self._interpretation_mode_stack:
                self._interpretation_mode_stack.pop()
            try:
                self._flush_pending_auto_on_load()
            except Exception:
                pass
            
            # Process queued actions if auto-processing is enabled
            if self._auto_process_queue and not self.action_queue.is_empty():
                try:
                    executed_count = self.process_queue()
                    if executed_count > 0:
                        print(f"🔄 Auto-processed {executed_count} queued actions")
                except Exception as e:
                    print(f"⚠️ Error processing action queue: {e}")

    def agentic_mode(
        self,
        user_prompt: str,
        max_iterations: int = 50,
        track_ineffective_actions: bool = True,
        base_knowledge: Optional[List[str]] = None,
        allow_partial_completion: bool = False,
        check_ineffective_actions: Optional[bool] = None,
        show_completion_reasoning_every_iteration: bool = False,
        strict_mode: bool = False,
        clarification_callback: Optional[Callable[[str], str]] = None,
        max_clarification_rounds: int = 3
    ) -> AgentResult:
        """
        Run agentic mode (Step 1: Basic Reactive Agent).
        
        This is a basic implementation that:
        - Observes browser state
        - Checks simple completion criteria
        - Generates and executes actions reactively
        - Repeats until task is complete or max iterations reached
        - Automatically extracts data when extraction is needed
        
        Args:
            user_prompt: High-level user request (e.g., "click the submit button", "navigate to google.com", "extract product price")
            max_iterations: Maximum number of iterations before giving up (default: 50)
            track_ineffective_actions: If True, track and avoid repeating actions that didn't yield page changes.
            allow_partial_completion: If True, allow CompletionContract to mark tasks complete when major deliverables are satisfied, even if minor items remain.
            check_ineffective_actions: Optional override to enable/disable ineffective-action detection. If provided, supersedes track_ineffective_actions.
                                       Default: True (recommended for better performance)
            show_completion_reasoning_every_iteration: If True, show and log the completion reasoning on every iteration, not just when the task is complete.
                                                       Useful for debugging why the agent doesn't recognize completion.
            strict_mode: If True, the agent will follow instructions exactly without inferring extra requirements.
                        For example, if asked to "click the login button", it will consider the task complete after clicking,
                        even if login fails. Default: False.
            clarification_callback: Optional callback function(message: str) -> str for asking the user clarification questions
                                   when the agent wants to infer extra information. If provided and strict_mode is False,
                                   the agent will ask before inferring. Default: None.
            max_clarification_rounds: Maximum number of clarification rounds. Default: 3.
            base_knowledge: Optional list of knowledge rules/instructions that guide the agent's behavior.
                           These rules influence what actions the agent takes.
                           Example: ["just press enter after you've typed a search term into a search field"]
            
        Returns:
            AgentResult object containing:
            - success: Whether the task completed successfully
            - extracted_data: Dictionary of extracted data (key: extraction prompt, value: extracted result)
            - reasoning: Explanation of the result
            - confidence: Confidence score (0.0-1.0)
            
        Example:
            # Basic usage
            bot.start()
            bot.page.goto("https://example.com")
            result = bot.agentic_mode("search for python tutorials")
            if result.success:
                print("Task completed!")
            
            # With extraction
            result = bot.agentic_mode(
                "navigate to amazon.com, search for 'laptop', extract the first product name and price"
            )
            if result.success:
                print(f"Product: {result.extracted_data.get('first product name')}")
                print(f"Price: {result.extracted_data.get('price')}")
            
            # Access extracted data
            for prompt, data in result.extracted_data.items():
                print(f"{prompt}: {data}")
        """
        # Set agent mode flag
        self._agent_mode = True
        
        # Create middleware context
        context = ActionContext(
            action_type='agentic_mode',
            action_data={
                'user_prompt': user_prompt,
                'max_iterations': max_iterations,
                'strict_mode': strict_mode
            },
            bot=self,
            metadata={}
        )
        
        # Execute before hooks
        context = self.middleware.execute_before(context)
        
        # Check if middleware wants to skip execution
        if not context.should_continue:
            return context.cached_result
        
        try:
            if check_ineffective_actions is not None:
                track_ineffective_actions = check_ineffective_actions
            
            controller = AgentController(
                self,
                track_ineffective_actions=track_ineffective_actions,
                base_knowledge=base_knowledge,
                allow_partial_completion=allow_partial_completion,
                parallel_completion_and_action=self.parallel_completion_and_action,
                show_completion_reasoning_every_iteration=show_completion_reasoning_every_iteration,
                strict_mode=strict_mode,
                clarification_callback=clarification_callback,
                max_clarification_rounds=max_clarification_rounds,
                stuck_detector_enabled=self.stuck_detector_enabled,
                stuck_detector_window_size=self.stuck_detector_window_size,
                stuck_detector_threshold=self.stuck_detector_threshold,
                stuck_detector_weight_repeated_action=self.stuck_detector_weight_repeated_action,
                stuck_detector_weight_repetitive_action_no_change=self.stuck_detector_weight_repetitive_action_no_change,
                stuck_detector_weight_no_state_change=self.stuck_detector_weight_no_state_change,
                stuck_detector_weight_no_progress=self.stuck_detector_weight_no_progress,
                stuck_detector_weight_error_spiral=self.stuck_detector_weight_error_spiral,
                stuck_detector_weight_high_confidence_no_progress=self.stuck_detector_weight_high_confidence_no_progress,
            )
            controller.max_iterations = max_iterations
            
            task_result = controller.run_agentic_mode(user_prompt)
            
            # Create result
            result = AgentResult(task_result, controller.extracted_data)
            
            # Execute after hooks
            result = self.middleware.execute_after(context, result)
            
            return result
            
        except Exception as e:
            # Execute error hooks
            self.middleware.execute_on_error(context, e)
            raise
        finally:
            # Reset agent mode flag when done
            self._agent_mode = False

    def extract(
        self,
        prompt: str,
        output_format: str = "json",
        model_schema: Optional[Type[BaseModel]] = None,
        scope: str = "viewport",
        element_description: Optional[str] = None,
        max_retries: int = 2,
        confidence_threshold: float = 0.6,
    ) -> Union[str, Dict[str, Any], BaseModel]:
        """
        Extract data from the current page based on natural language description.
        
        Args:
            prompt: Natural language description of what to extract
                    (e.g., "product price", "article title", "form field values")
            output_format: "json" (default), "text", or "structured"
            model_schema: Optional Pydantic model for structured output
            scope: "viewport", "full_page", or "element"
            element_description: Required if scope="element"
            max_retries: Maximum retry attempts if extraction fails
            confidence_threshold: Minimum confidence to return result (0.0-1.0)
        
        Returns:
            - If output_format="text": str
            - If output_format="json": Dict[str, Any]
            - If output_format="structured" and model_schema provided: model instance
        """
        from ai_utils import generate_text, generate_model
        # Import InteractionType locally to avoid linter false positive
        from session_tracker import InteractionType as IT
        
        self._check_termination()
        
        if not self.started or not self.page:
            raise RuntimeError("Bot not started. Call bot.start() first.")
        
        # Validate scope
        if scope == "element" and not element_description:
            raise ValueError("element_description is required when scope='element'")
        
        # Capture screenshot based on scope
        try:
            if scope == "full_page":
                screenshot = self.page.screenshot(full_page=True)
                # Get full page text - use larger limit for full page
                visible_text = self.page.evaluate("document.body.innerText") or ""
            elif scope == "element":
                # For element scope, we'll use vision to find the element first
                screenshot = self.page.screenshot(full_page=False)
                visible_text = self.page.evaluate("document.body.innerText") or ""
            else:  # viewport
                screenshot = self.page.screenshot(full_page=False)
                visible_text = self.page.evaluate("document.body.innerText") or ""
        except Exception as e:
            raise RuntimeError(f"Failed to capture screenshot: {e}")
        
        # Clean and prepare the text (remove excessive whitespace but keep structure)
        if visible_text:
            # Remove excessive newlines but keep some structure
            import re
            visible_text = re.sub(r'\n{3,}', '\n\n', visible_text.strip())
        
        # Build extraction prompt for LLM with full page text for grounding
        extraction_prompt = f"""
Extract the following information from this webpage screenshot:
{prompt}

Current page context:
- URL: {self.page.url}
- Title: {self.page.title()}

FULL PAGE TEXT CONTENT (use this to verify your extraction):
{visible_text if visible_text else "(No text content found on page)"}

IMPORTANT INSTRUCTIONS:
1. Use the FULL PAGE TEXT CONTENT above to verify that the information you extract actually exists on the page
2. Only extract information that appears in BOTH the screenshot AND the text content provided above
3. Do NOT make up or infer data that is not present in the text content
4. If the requested information is not found in the text content, return an empty object {{}} or indicate "not available"
5. Cross-reference your visual extraction with the text content to ensure accuracy
"""
        
        # Try extraction with retries
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if output_format == "text":
                    # Simple text extraction using vision, grounded with page text
                    question = f"""What is the {prompt}? 

FULL PAGE TEXT CONTENT (use this to verify your extraction):
{visible_text if visible_text else "(No text content found on page)"}

Return only the extracted text that appears in the text content above. Do not make up text that isn't in the provided content."""
                    result_text = generate_text(
                        prompt=question,
                        system_prompt="You are extracting text from a webpage. The user has provided the FULL PAGE TEXT CONTENT. You must verify that the text you extract actually exists in the provided content. Return only the extracted text that appears in the content, no explanations or formatting.",
                        image=screenshot,
                        image_detail="high"
                    )
                    extracted_text = result_text.strip()
                    
                    # Record extraction in interaction history
                    self.session_tracker.record_interaction(
                        IT.EXTRACT,
                        extraction_prompt=prompt,
                        extracted_data={"text": extracted_text},
                        success=True
                    )
                    
                    return extracted_text
                
                elif output_format == "json":
                    # JSON extraction using structured output
                    # Use a string field for JSON to avoid schema validation issues with Dict[str, Any]
                    class ExtractionResult(BaseModel):
                        extracted_data: str = Field(description="The extracted data as a JSON string with key-value pairs")
                        confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the extraction")
                        reasoning: str = Field(description="Brief explanation of what was extracted")
                    
                    result = generate_model(
                        prompt=extraction_prompt,
                        model_object_type=ExtractionResult,
                        system_prompt="You are extracting structured data from a webpage screenshot. The user has provided the FULL PAGE TEXT CONTENT in the prompt. CRITICAL: You must verify every piece of extracted data against the provided text content. Only extract information that appears in BOTH the screenshot AND the text content. Do NOT make up, infer, or guess data. If information is not found in the text content, it does not exist on the page - return an empty object {} or mark it as 'not available'. The extracted_data field should be a valid JSON string containing only verified data that exists in the provided text content.",
                        image=screenshot,
                        image_detail="high"
                    )
                    
                    # Check if result is actually an ExtractionResult instance (parsing might have failed)
                    if not isinstance(result, ExtractionResult):
                        # Try manual parsing as fallback
                        from ai_utils import _manual_parse_structured_output
                        manual_result = _manual_parse_structured_output(str(result), ExtractionResult)
                        if manual_result and isinstance(manual_result, ExtractionResult):
                            print("✅ Successfully parsed extraction result using manual parser")
                            result = manual_result
                        else:
                            # Parsing failed completely
                            error_msg = str(result)[:200] if result else "Empty response"
                            raise ValueError(f"Failed to parse extraction result as ExtractionResult. Got raw text instead: {error_msg}")
                    
                    if result.confidence < confidence_threshold:
                        raise ValueError(f"Extraction confidence {result.confidence} below threshold {confidence_threshold}")
                    
                    # Warn if confidence is low (but above threshold)
                    if result.confidence < 0.8:
                        print(f"⚠️ Low extraction confidence ({result.confidence:.2f}). Verify extracted data matches what's actually on the page.")
                    
                    # Parse the JSON string
                    import json
                    try:
                        extracted_dict = json.loads(result.extracted_data)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Failed to parse extracted_data as JSON: {e}. Raw data: {result.extracted_data}")
                    
                    # Validate that extracted_dict is actually a dict (not a list or other type)
                    if not isinstance(extracted_dict, dict):
                        raise ValueError(f"Expected extracted_data to be a JSON object (dict), but got {type(extracted_dict).__name__}: {extracted_dict}")
                    
                    # Validate extracted values against page text to catch hallucinations
                    if visible_text:
                        page_text_lower = visible_text.lower()
                        validation_warnings = []
                        
                        def validate_value(value, key_path=""):
                            """Recursively validate that extracted values exist in page text"""
                            if isinstance(value, dict):
                                for k, v in value.items():
                                    validate_value(v, f"{key_path}.{k}" if key_path else k)
                            elif isinstance(value, list):
                                for i, item in enumerate(value):
                                    validate_value(item, f"{key_path}[{i}]" if key_path else f"[{i}]")
                            elif isinstance(value, str) and value.strip():
                                # Check if the value (or significant parts of it) appear in page text
                                value_clean = value.strip().lower()
                                # For longer strings, check if key parts appear
                                if len(value_clean) > 20:
                                    # Check if at least 50% of words appear
                                    words = value_clean.split()
                                    if words:
                                        words_found = sum(1 for word in words if len(word) > 3 and word in page_text_lower)
                                        if words_found < len(words) * 0.5:
                                            validation_warnings.append(f"Value at '{key_path}' may not exist on page: '{value[:50]}...'")
                                else:
                                    # For shorter strings, require exact or near-exact match
                                    if value_clean not in page_text_lower and len(value_clean) > 5:
                                        # Check if it's a number or percentage (these might be formatted differently)
                                        if not (value_clean.replace('.', '').replace('-', '').replace('%', '').replace('$', '').replace(',', '').isdigit()):
                                            validation_warnings.append(f"Value at '{key_path}' may not exist on page: '{value}'")
                        
                        # Validate all extracted values
                        for key, value in extracted_dict.items():
                            if not key.startswith('_'):  # Skip metadata fields
                                validate_value(value, key)
                        
                        if validation_warnings:
                            print("⚠️ Validation warnings - some extracted values may not exist on page:")
                            for warning in validation_warnings[:5]:  # Show first 5 warnings
                                print(f"   - {warning}")
                            if len(validation_warnings) > 5:
                                print(f"   ... and {len(validation_warnings) - 5} more warnings")
                    
                    # Add metadata
                    extracted_dict["_confidence"] = result.confidence
                    extracted_dict["_reasoning"] = result.reasoning
                    
                    # Record extraction in interaction history
                    self.session_tracker.record_interaction(
                        IT.EXTRACT,
                        extraction_prompt=prompt,
                        extracted_data=extracted_dict,
                        success=True
                    )
                    
                    return extracted_dict
                
                elif output_format == "structured":
                    if not model_schema:
                        raise ValueError("model_schema is required when output_format='structured'")
                    result = generate_model(
                        prompt=extraction_prompt,
                        model_object_type=model_schema,
                        system_prompt=f"You are extracting structured data from a webpage. Extract the requested information matching this schema: {model_schema.__name__}",
                        image=screenshot,
                        image_detail="high"
                    )
                    
                    # Record extraction in interaction history
                    # Convert Pydantic model to dict for storage
                    extracted_dict = result.model_dump() if hasattr(result, 'model_dump') else result.dict()
                    self.session_tracker.record_interaction(
                        IT.EXTRACT,
                        extraction_prompt=prompt,
                        extracted_data=extracted_dict,
                        success=True
                    )
                    
                    return result
                
                else:
                    raise ValueError(f"Invalid output_format: {output_format}")
            
            except Exception as e:
                last_error = e
                error_msg = str(e)
                # Provide more context for common errors
                if "list indices must be integers" in error_msg:
                    error_msg = f"Response parsing error (list indices): {e}. This may indicate an unexpected response format from the model."
                elif "not str" in error_msg or "must be" in error_msg:
                    error_msg = f"Type error during extraction: {e}. This may indicate a malformed response from the model."
                
                if attempt < max_retries:
                    print(f"[Extract] Attempt {attempt + 1} failed: {error_msg}, retrying...")
                    import time
                    time.sleep(0.5)
                else:
                    # Record failed extraction in interaction history
                    self.session_tracker.record_interaction(
                        IT.EXTRACT,
                        extraction_prompt=prompt,
                        extracted_data=None,
                        success=False,
                        error_message=error_msg
                    )
                    raise RuntimeError(f"Extraction failed after {max_retries + 1} attempts: {error_msg}")
        
            # Record failed extraction if we exhausted all retries
        if last_error:
            self.session_tracker.record_interaction(
                IT.EXTRACT,
                extraction_prompt=prompt,
                extracted_data=None,
                success=False,
                error_message=str(last_error)
            )
        raise RuntimeError(f"Extraction failed: {last_error}")
    
    def extract_batch(
        self,
        prompts: List[str],
        output_format: str = "json",
        model_schema: Optional[Type[BaseModel]] = None,
        scope: str = "viewport",
        **kwargs
    ) -> List[Union[str, Dict[str, Any], BaseModel]]:
        """
        Extract multiple fields from the current page in one operation.
        
        Args:
            prompts: List of extraction prompts
            output_format: "json" (default), "text", or "structured"
            model_schema: Optional Pydantic model for structured output
            scope: "viewport", "full_page", or "element"
            **kwargs: Additional arguments passed to extract()
        
        Returns:
            List of extraction results in the same order as prompts
        """
        results = []
        for prompt in prompts:
            try:
                result = self.extract(
                    prompt=prompt,
                    output_format=output_format,
                    model_schema=model_schema,
                    scope=scope,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                print(f"[ExtractBatch] Failed to extract '{prompt}': {e}")
                # Return None or empty dict for failed extractions
                if output_format == "text":
                    results.append("")
                elif output_format == "json":
                    results.append({"error": str(e)})
                else:
                    results.append(None)
        
        return results
    
    def extract_multi_field(
        self,
        prompt: str,
        fields: List[str],
        output_format: str = "json",
        scope: str = "viewport",
        **kwargs
    ) -> Dict[str, Union[str, Dict[str, Any], BaseModel]]:
        """
        Extract multiple related fields from a single prompt description.
        
        Example:
            bot.extract_multi_field(
                "product information",
                fields=["name", "price", "rating"]
            )
        
        Args:
            prompt: Overall description of what to extract (e.g., "product information")
            fields: List of specific field names to extract
            output_format: "json" (default), "text", or "structured"
            scope: "viewport", "full_page", or "element"
            **kwargs: Additional arguments passed to extract()
        
        Returns:
            Dictionary mapping field names to extracted values
        """
        # Build a combined extraction prompt
        field_prompts = [f"{prompt} - {field}" for field in fields]
        results = self.extract_batch(
            prompts=field_prompts,
            output_format=output_format,
            scope=scope,
            **kwargs
        )
        
        # Combine results into a dictionary
        extracted_data = {}
        for i, field in enumerate(fields):
            if i < len(results):
                extracted_data[field] = results[i]
            else:
                extracted_data[field] = None
        
        return extracted_data

    def goto(self, url: str, timeout: int = 2000) -> None:
        """Go to a URL"""
        self._check_termination()
        
        if not self.started:
            print("❌ Bot not started")
            return
        
        self.page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        self.url = url
        # Ensure SessionTracker history reflects the first real navigation instead of about:blank
        try:
            if hasattr(self, 'session_tracker') and self.session_tracker:
                hist = getattr(self.session_tracker, 'url_history', None)
                ptr = getattr(self.session_tracker, 'url_pointer', None)
                current = self.page.url
                # If we only have the initial about:blank entry, replace it with the real URL
                if isinstance(hist, list) and len(hist) == 1 and (hist[0] or '').startswith('about:blank'):
                    self.session_tracker.url_history = [current]
                    self.session_tracker.url_pointer = 0
                # If history exists but pointer is not at the end, truncate forward stack and append
                elif isinstance(hist, list) and isinstance(ptr, int) and 0 <= ptr < len(hist):
                    if hist[ptr] != current:
                        # Truncate any forward entries
                        if ptr < (len(hist) - 1):
                            self.session_tracker.url_history = hist[: ptr + 1]
                        # Append only if it's not already the last entry
                        if not self.session_tracker.url_history or self.session_tracker.url_history[-1] != current:
                            self.session_tracker.url_history.append(current)
                        self.session_tracker.url_pointer = len(self.session_tracker.url_history) - 1
        except Exception:
            # Non-fatal: history sync is best-effort
            pass
        # Run auto-on-load actions immediately on this freshly loaded page
        try:
            if self._auto_on_load_enabled:
                self._run_auto_actions_for_current_page()
        except Exception:
            pass

    def register_prompts(
        self,
        prompts: List[str],
        ref_id: str,
        all_must_be_true: bool = False,
        interpretation_mode: Optional[str] = None,
        additional_context: str = "",
        target_context_guard: Optional[str] = None,
        skip_post_guard_refinement: bool = True,
        confirm_before_interaction: bool = False,
        command_id: Optional[str] = None,
        max_attempts: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> bool:
        """
        Register multiple commands for later reference and execution.
        
        Args:
            prompts: List of command prompts to register
            ref_id: Unique identifier for referencing these commands
            all_must_be_true: Require every command to succeed when evaluating this ref
            interpretation_mode: Optional interpretation mode to apply when executing this ref
            additional_context: Extra context to include when executing each stored prompt
            target_context_guard: Guard condition description applied to each stored prompt
            skip_post_guard_refinement: Whether to skip post-plan guard refinement when executing stored prompts
            confirm_before_interaction: Whether to require confirmation before each stored interaction
            max_attempts: Override bot's max_attempts for each command in this ref (None = use bot default)
            max_retries: Override goal's max_retries for each command in this ref (None = use goal default)
            
        Returns:
            True if commands were registered successfully, False otherwise
        """
        self._check_termination()
        
        # Register the ref in command ledger
        command_id = self.command_ledger.register_command(
            command=f"register_prompts: {ref_id}",
            command_id=command_id,
            metadata={"source": "register_prompts", "ref_id": ref_id, "prompt_count": len(prompts)}
        )
        
        try:
            if not prompts:
                self.logger.log_error("No prompts provided for register_prompts", "register_prompts() called with empty prompts")
                print("❌ No prompts provided for register_prompts")
                return False
            
            # Store the commands for later reference
            normalized_mode = self._normalize_interpretation_mode(interpretation_mode) if interpretation_mode else None

            self.command_refs[ref_id] = {
                "prompts": prompts.copy(),
                "all_must_be_true": bool(all_must_be_true),
                "interpretation_mode": normalized_mode,
                "command_id": command_id,  # Store the original command ID
                "additional_context": additional_context or "",
                "target_context_guard": target_context_guard,
                "skip_post_guard_refinement": bool(skip_post_guard_refinement),
                "confirm_before_interaction": bool(confirm_before_interaction),
                "max_attempts": max_attempts,
                "max_retries": max_retries,
            }
            mode = "ALL" if all_must_be_true else "ANY"
            extra_parts = []
            if normalized_mode:
                extra_parts.append(f"interpretation={normalized_mode}")
            if additional_context:
                extra_parts.append("additional_context")
            if target_context_guard:
                extra_parts.append("target_context_guard")
            if not skip_post_guard_refinement:
                extra_parts.append("skip_post_guard_refinement=False")
            if confirm_before_interaction:
                extra_parts.append("confirm_before_interaction=True")
            extra_summary = f" ({', '.join(extra_parts)})" if extra_parts else ""
            self.logger.log(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                f"Registered {len(prompts)} commands with ref ID: {ref_id} (mode={mode}){extra_summary}",
            )
            print(f"📋 Registered {len(prompts)} commands with ref ID: {ref_id} (mode={mode}){extra_summary}")
            
            # Show what was registered
            for i, prompt in enumerate(prompts, 1):
                print(f"   {i}. {prompt}")
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"Error in register_prompts: {e}", "register_prompts() execution", {"ref_id": ref_id})
            print(f"❌ Error in register_prompts: {e}")
            return False

    def _normalize_interpretation_mode(self, mode: Optional[str]) -> str:
        if not mode:
            return self.default_interpretation_mode
        normalized = str(mode).lower().strip()
        if normalized not in {"literal", "semantic", "auto"}:
            return self.default_interpretation_mode
        if normalized == "auto":
            return "semantic"
        return normalized

    def _get_current_interpretation_mode(self) -> str:
        if self._interpretation_mode_stack:
            return self._interpretation_mode_stack[-1]
        return self.default_interpretation_mode

    def _get_semantic_target(self, description: str) -> Optional[SemanticTarget]:
        mode = self._get_current_interpretation_mode()
        if mode == "literal":
            return None
        key = (description or "").strip().lower()
        if not key:
            return None
        if key in self._semantic_target_cache:
            return self._semantic_target_cache[key]
        target = build_semantic_target(description)
        self._semantic_target_cache[key] = target
        return target

    def _determine_dedup_usage(self, goal_description: str) -> Tuple[bool, str]:
        """Decide whether duplicate interactions should be avoided for this prompt."""

        desc_lower = (goal_description or "").lower()
        keyword_triggers = [
            "no duplicate",
            "avoid duplicate",
            "avoid repeats",
            "no repeats",
            "unique",
            "fresh",
            "unseen",
            "next link",
            "different link",
            "dedup",
        ]
        explicit_request = any(trigger in desc_lower for trigger in keyword_triggers)

        if self.dedup_mode == "on":
            return True, "dedup_mode:on"
        if self.dedup_mode == "off":
            return False, "dedup_mode:off"
        if explicit_request:
            return True, "prompt_keywords"
        return False, "no_request"

    def _build_dedup_prompt_context(self, goal_description: str) -> Optional[Dict[str, Any]]:
        """Build deduplication context for plan prompts when duplicates must be avoided."""

        should_avoid, reason = self._determine_dedup_usage(goal_description)
        if not should_avoid:
            return None

        deduper = getattr(self, 'deduper', None)
        if not deduper or not hasattr(deduper, 'get_interacted_element_history'):
            return None

        limit = self.dedup_history_quantity
        history_limit = None if limit is None or limit < 0 else limit
        try:
            history = deduper.get_interacted_element_history(history_limit)
        except Exception:
            return None

        if not history:
            return None

        return {
            "avoid_duplicates": True,
            "quantity": limit if limit is not None else -1,
            "entries": history,
            "reason": reason,
        }

    def _build_navigation_plan(self, goal: Any) -> Optional[VisionPlan]:
        """Construct a single-step plan that opens the requested URL."""
        url = (goal.navigation_intent or "").strip()
        if not url:
            print("❌ Navigation goal missing URL; cannot create OPEN plan")
            return None

        print(f"[PlanGen] Building OPEN plan for navigation goal: {url}")
        return VisionPlan(
            detected_elements=PageElements(elements=[]),
            action_steps=[ActionStep(action=ActionType.OPEN, url=url)],
            reasoning=f"Open the URL '{url}' in the current tab.",
            confidence=1.0,
        )

    def _build_click_selection_plan(
        self,
        *,
        goal_description: str,
        element_data: List[Dict[str, Any]],
        screenshot_with_overlays: Optional[bytes],
        page_info: PageInfo,
        semantic_hint: Optional[SemanticTarget],
    ) -> Optional[VisionPlan]:
        """Use lightweight overlay selection to build a single-click plan."""
        if not element_data:
            return None

        overlay_idx = self.plan_generator.select_best_overlay(
            goal_description,
            element_data,
            semantic_hint=semantic_hint,
            screenshot=screenshot_with_overlays,
        )

        if overlay_idx is None:
            print("[PlanGen] ❌ Overlay selection failed to identify a target")
            return None

        matching_data = next((elem for elem in element_data if elem.get("index") == overlay_idx), None)
        if not matching_data:
            print(f"[PlanGen] ❌ Selected overlay #{overlay_idx} missing in element data")
            return None

        action_step = ActionStep(action=ActionType.CLICK, overlay_index=overlay_idx)
        detected_elements = self.plan_generator.convert_indices_to_elements([action_step], element_data)

        confidence = 0.0
        reasoning = f"Selected overlay #{overlay_idx} that best matches the goal."

        print(f"[PlanGen] 🎯 Click selection chose overlay #{overlay_idx}")

        return VisionPlan(
            detected_elements=detected_elements,
            action_steps=[action_step],
            reasoning=reasoning,
            confidence=confidence,
        )

    def _collect_overlay_data(
        self,
        goal_description: str,
        page_info: PageInfo,
    ) -> tuple[List[Dict[str, Any]], Optional[bytes], Optional[bytes]]:
        """Collect overlay metadata and screenshots, with caching and dedup filtering."""
        current_focus_context = self.focus_manager.get_current_focus_context()
        
        try:
            self.event_logger.system_debug("Numbering interactive elements...")
        except Exception:
            pass
        clean_screenshot = None
        try:
            clean_screenshot = self.page.screenshot(type="jpeg", quality=35, full_page=False)
        except Exception as e:
            try:
                self.event_logger.system_error("Failed to capture clean screenshot before overlays", error=e)
            except Exception:
                pass

        element_data = self.overlay_manager.create_numbered_overlays(page_info, mode="interactive") or []

        if current_focus_context:
            try:
                self.event_logger.system_debug("Filtering elements based on current focus context...")
            except Exception:
                pass
            element_data = self._filter_elements_by_focus(element_data)

        self._pre_dedup_element_data = element_data.copy() if element_data else []
        self._cached_focus_context = current_focus_context

        try:
            self.event_logger.system_debug("Capturing screenshot with overlays...")
        except Exception:
            pass
        screenshot_with_overlays = self.page.screenshot(type="jpeg", quality=35, full_page=False)
        self._cached_screenshot_with_overlays = screenshot_with_overlays
        self._cached_clean_screenshot = clean_screenshot
        self._cached_overlay_data = (
            self._pre_dedup_element_data.copy() if hasattr(self, "_pre_dedup_element_data") else element_data.copy()
        )
        self._cached_dom_signature = self.last_dom_signature
        try:
            self.event_logger.system_debug(f"Cached screenshot and {len(self._cached_overlay_data)} overlay elements")
        except Exception:
            pass

        if (
            self.deduper
            and self.deduper.dedup_enabled
            and element_data
        ):
            should_avoid, reason = self._determine_dedup_usage(goal_description)
            if should_avoid:
                print(f"🚫 Filtering out interacted elements (reason: {reason})...")
                elements_for_dedup: List[Dict[str, Any]] = []
                for elem in element_data:
                    elements_for_dedup.append(
                        {
                            "tagName": elem.get("tagName", ""),
                            "text": elem.get("text", ""),
                            "textContent": elem.get("text", ""),
                            "description": elem.get("description", ""),
                            "element_type": elem.get("tagName", ""),
                            "href": elem.get("href", ""),
                            "ariaLabel": elem.get("ariaLabel", ""),
                            "aria_label": elem.get("ariaLabel", ""),
                            "id": elem.get("id", ""),
                            "role": elem.get("role", ""),
                            "overlayIndex": elem.get("index"),
                            "box2d": elem.get("normalizedCoords"),
                            "normalizedCoords": elem.get("normalizedCoords"),
                        }
                    )

                filtered_elements = self.deduper.filter_interacted_elements(elements_for_dedup, "click")
                print(
                    f"🔢 Found {len(filtered_elements)} elements after deduplication "
                    f"(removed {len(elements_for_dedup) - len(filtered_elements)} duplicates)"
                )

                filtered_element_data: List[Dict[str, Any]] = []
                for elem in filtered_elements:
                    overlay_idx = elem.get("overlayIndex")
                    original_elem = next((e for e in element_data if e.get("index") == overlay_idx), None)
                    if original_elem:
                        filtered_element_data.append(original_elem.copy())
                element_data = filtered_element_data

        return element_data, screenshot_with_overlays, clean_screenshot
    
    def _execute_keyword_command(
        self,
        goal_description: str,
        additional_context: str,
        target_context_guard: Optional[str],
        confirm_before_interaction: bool,
        command_id: Optional[str],
        start_time: float,
    ) -> Optional[bool]:
        """Attempt to execute the command using keyword-based execution. Returns None to fall back."""
        parsed = parse_keyword_command(goal_description)
        if not parsed:
            return None
        keyword, payload, helper = parsed
        keyword = (keyword or "").strip().lower()

        if keyword == "click":
            result = self._keyword_click(
                goal_description=goal_description,
                payload=payload,
                helper=helper,
                additional_context=additional_context,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
            )
        elif keyword == "type":
            result = self._keyword_type(
                goal_description=goal_description,
                payload=payload,
                helper=helper,
                additional_context=additional_context,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
            )
        elif keyword == "select":
            result = self._keyword_select(
                goal_description=goal_description,
                payload=payload,
                helper=helper,
                additional_context=additional_context,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
            )
        elif keyword == "upload":
            result = self._keyword_upload(
                goal_description=goal_description,
                payload=payload,
                helper=helper,
                additional_context=additional_context,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
            )
        elif keyword == "datetime":
            result = self._keyword_datetime(
                goal_description=goal_description,
                payload=payload,
                helper=helper,
                additional_context=additional_context,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
            )
        elif keyword == "scroll":
            result = self._keyword_scroll(
                goal_description=goal_description,
                payload=payload,
                helper=helper,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
            )
        elif keyword == "wait":
            result = self._keyword_wait(
                goal_description=goal_description,
                payload=payload,
                helper=helper,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
            )
        elif keyword == "press":
            result = self._keyword_press(
                goal_description=goal_description,
                payload=payload,
                helper=helper,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
            )
        elif keyword == "stop":
            result = self._keyword_stop(
                goal_description=goal_description,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
            )
        elif keyword == "back":
            result = self._keyword_back(
                goal_description=goal_description,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
            )
        elif keyword == "forward":
            result = self._keyword_forward(
                goal_description=goal_description,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
            )
        elif keyword == "navigate":
            result = self._keyword_open(
                goal_description=goal_description,
                payload=payload,
                helper=helper,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
            )
        else:
            # Unsupported keyword – fall back
            return None

        if result is None:
            # Keyword command could not confidently execute – allow normal flow
            return None

        duration_ms = (time.time() - start_time) * 1000
        if result:
            self.logger.log_goal_success(goal_description, duration_ms)
            try:
                self.event_logger.command_execution_complete(goal_description=goal_description, success=True)
            except Exception:
                pass
            self.command_ledger.complete_command(command_id, success=True)
        else:
            self.logger.log_goal_failure(goal_description, "Keyword command execution failed", duration_ms)
            self.command_ledger.complete_command(
                command_id,
                success=False,
                error_message="Keyword command execution failed",
            )
        self._invalidate_plan_cache("keyword command execution")
        return result

    def _normalize_hint(self, text: Optional[str]) -> str:
        if not text:
            return ""
        stripped = re.sub(r"\b(click|press|tap|button|link|the|a|type|select|choose|set)\b", " ", text, flags=re.IGNORECASE)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        return stripped or text.strip()

    def _compose_selection_instruction(
        self,
        *,
        request_instruction: str,
        target_hint: str,
        detail: Optional[str] = None,
    ) -> str:
        parts: List[str] = []
        if request_instruction:
            parts.append(request_instruction)
        if target_hint:
            parts.append(f"(target: {target_hint})")
        if detail:
            parts.append(f"(detail: {detail})")
        if parts:
            return " ".join(parts)
        return request_instruction or target_hint or detail or ""

    def _execute_keyword_plan(
        self,
        *,
        action_steps: List[ActionStep],
        reasoning: str,
        target_context_guard: Optional[str],
        confirm_before_interaction: bool,
        detected_elements: Optional[PageElements] = None,
        page_info: Optional[PageInfo] = None,
        disable_vision_tag_hint: bool = False,
    ) -> bool:
        plan = VisionPlan(
            detected_elements=detected_elements or PageElements(elements=[]),
            action_steps=action_steps,
            reasoning=reasoning,
            confidence=0.0,
        )

        page_info = page_info or self.page_utils.get_page_info()

        original_tag_hint_state: Optional[bool] = None
        if disable_vision_tag_hint and hasattr(self.action_executor, "enable_vision_tag_hint"):
            original_tag_hint_state = getattr(self.action_executor, "enable_vision_tag_hint")
            self.action_executor.enable_vision_tag_hint = False

        try:
            print("[KeywordCommand] Executing plan via action_executor")
            success = self.action_executor.execute_plan(
                plan,
                page_info,
                target_context_guard=target_context_guard,
                skip_post_guard_refinement=True,
                confirm_before_interaction=confirm_before_interaction,
            )
        finally:
            if disable_vision_tag_hint and hasattr(self.action_executor, "enable_vision_tag_hint"):
                self.action_executor.enable_vision_tag_hint = original_tag_hint_state

        try:
            self.event_logger.system_debug(f"[KeywordCommand] Execution result: {success}")
        except Exception:
            pass
        return success

    def _keyword_overlay_action(
        self,
        *,
        goal_description: str,
        selection_instruction: str,
        target_hint: str,
        action_type: ActionType,
        action_kwargs: Dict[str, Any],
        target_context_guard: Optional[str],
        confirm_before_interaction: bool,
    ) -> Optional[bool]:
        max_attempts = self.element_selection_retry_attempts
        instruction = selection_instruction or goal_description
        
        page_info = None
        element_data = None
        overlay_index = None
        
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                print(f"[KeywordCommand] Retry attempt {attempt}/{max_attempts} for element selection")
                # Small delay before retry to allow page to stabilize
                time.sleep(0.5)
            
            page_info = self.page_utils.get_page_info()
            element_data, screenshot_with_overlays, clean_screenshot = self._collect_overlay_data(goal_description, page_info)
            if not element_data:
                if attempt < max_attempts:
                    print(f"⚠️ No interactive elements detected (attempt {attempt}/{max_attempts}), retrying...")
                    continue
                print("❌ No interactive elements detected after all retries")
                return False

            # Use fallback model for retry attempts if configured
            selection_model = None
            if attempt > 1 and self.element_selection_fallback_model:
                selection_model = self.element_selection_fallback_model
                print(f"[KeywordCommand] Using fallback model: {selection_model}")

            try:
                self.event_logger.overlay_selection(f"Requesting overlay selection from LLM (attempt {attempt}/{max_attempts})")
            except Exception:
                pass
            selection = self.plan_generator.select_best_overlay(
                instruction=instruction,
                element_data=element_data,
                semantic_hint=None,
                screenshot=clean_screenshot or screenshot_with_overlays,
                model=selection_model,
            )
            try:
                self.event_logger.overlay_selection(f"Overlay selection response: {selection}")
            except Exception:
                pass

            if selection is None:
                if attempt < max_attempts:
                    print(f"⚠️ Overlay selection failed (attempt {attempt}/{max_attempts}), retrying...")
                    try:
                        self.overlay_manager.remove_overlays()
                    except Exception:
                        pass
                    continue
                print("❌ Overlay selection failed after all retries")
                try:
                    self.overlay_manager.remove_overlays()
                except Exception:
                    pass
                return False

            overlay_index = selection
            try:
                self.event_logger.overlay_selection(f"LLM chose overlay #{overlay_index}")
            except Exception:
                pass

            matching_data = next((elem for elem in element_data if elem.get("index") == overlay_index), None)
            if not matching_data:
                if attempt < max_attempts:
                    print(f"⚠️ Selected overlay #{overlay_index} missing in element data (attempt {attempt}/{max_attempts}), retrying...")
                    try:
                        self.overlay_manager.remove_overlays()
                    except Exception:
                        pass
                    continue
                print(f"[KeywordCommand] ❌ Selected overlay #{overlay_index} missing in element data after all retries")
                try:
                    self.overlay_manager.remove_overlays()
                except Exception:
                    pass
                return False
            
            # Success - break out of retry loop
            break

        try:
            self.overlay_manager.remove_overlays()
        except Exception:
            pass

        filtered_kwargs = {k: v for k, v in action_kwargs.items() if v is not None}
        action_step = ActionStep(action=action_type, overlay_index=overlay_index, **filtered_kwargs)
        detected_elements = self.plan_generator.convert_indices_to_elements([action_step], element_data)

        try:
            return self._execute_keyword_plan(
                action_steps=[action_step],
                reasoning=f"Selected overlay #{overlay_index} for '{target_hint or goal_description}'.",
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
                detected_elements=detected_elements,
                page_info=page_info,
                disable_vision_tag_hint=True,
            )
        finally:
            try:
                self.overlay_manager.remove_overlays()
            except Exception:
                pass

    def _keyword_click(
        self,
        *,
        goal_description: str,
        payload: str,
        helper: Optional[str],
        additional_context: str,
        target_context_guard: Optional[str],
        confirm_before_interaction: bool,
    ) -> Optional[bool]:
        target_hint_raw = (payload or "").strip()
        if helper:
            helper = helper.strip()
            if helper:
                target_hint_raw = f"{target_hint_raw} {helper}".strip()
        if not target_hint_raw:
            extracted = extract_click_target(goal_description)
            if extracted:
                target_hint_raw = extracted

        target_hint = self._normalize_hint(target_hint_raw)
        request_instruction = self._normalize_hint(goal_description)
        try:
            self.event_logger.command_execution_start(instruction=goal_description, target_hint=target_hint)
        except Exception:
            pass

        selection_instruction = self._compose_selection_instruction(
            request_instruction=request_instruction,
            target_hint=target_hint,
        )

        return self._keyword_overlay_action(
            goal_description=goal_description,
            selection_instruction=selection_instruction,
            target_hint=target_hint,
            action_type=ActionType.CLICK,
            action_kwargs={},
            target_context_guard=target_context_guard,
            confirm_before_interaction=confirm_before_interaction,
        )

    def _keyword_type(
        self,
        *,
        goal_description: str,
        payload: str,
        helper: Optional[str],
        additional_context: str,
        target_context_guard: Optional[str],
        confirm_before_interaction: bool,
    ) -> Optional[bool]:
        intent = parse_action_intent(goal_description)
        if not intent or intent.action != "type" or not intent.value:
            return None

        target_hint_raw = intent.target_text or helper or intent.helper_text
        target_hint = self._normalize_hint(target_hint_raw)
        if not target_hint:
            print("ℹ️ TYPE command missing target hint – falling back")
            return None

        request_instruction = self._normalize_hint(goal_description)
        selection_instruction = self._compose_selection_instruction(
            request_instruction=request_instruction,
            target_hint=target_hint,
        )

        return self._keyword_overlay_action(
            goal_description=goal_description,
            selection_instruction=selection_instruction,
            target_hint=target_hint,
            action_type=ActionType.TYPE,
            action_kwargs={"text_to_type": intent.value},
            target_context_guard=target_context_guard,
            confirm_before_interaction=confirm_before_interaction,
        )

    def _keyword_select(
        self,
        *,
        goal_description: str,
        payload: str,
        helper: Optional[str],
        additional_context: str,
        target_context_guard: Optional[str],
        confirm_before_interaction: bool,
    ) -> Optional[bool]:
        intent = parse_action_intent(goal_description)
        if not intent or intent.action != "select" or not intent.value:
            return None

        target_hint_raw = intent.target_text or helper or intent.helper_text
        target_hint = self._normalize_hint(target_hint_raw)
        if not target_hint:
            print("ℹ️ SELECT command missing target hint – falling back")
            return None

        request_instruction = self._normalize_hint(goal_description)
        option_detail = intent.value[:40] + ("…" if len(intent.value or "") > 40 else "")
        selection_instruction = self._compose_selection_instruction(
            request_instruction=request_instruction,
            target_hint=target_hint,
            detail=f"option: {option_detail}" if option_detail else None,
        )

        return self._keyword_overlay_action(
            goal_description=goal_description,
            selection_instruction=selection_instruction,
            target_hint=target_hint,
            action_type=ActionType.HANDLE_SELECT,
            action_kwargs={"select_option_text": intent.value},
            target_context_guard=target_context_guard,
            confirm_before_interaction=confirm_before_interaction,
        )

    def _keyword_upload(
        self,
        *,
        goal_description: str,
        payload: str,
        helper: Optional[str],
        additional_context: str,
        target_context_guard: Optional[str],
        confirm_before_interaction: bool,
    ) -> Optional[bool]:
        intent = parse_action_intent(goal_description)
        if not intent or intent.action != "upload" or not intent.value:
            return None

        target_hint_raw = intent.target_text or helper or intent.helper_text
        target_hint = self._normalize_hint(target_hint_raw)
        if not target_hint:
            print("ℹ️ UPLOAD command missing target hint – falling back")
            return None

        request_instruction = self._normalize_hint(goal_description)
        selection_instruction = self._compose_selection_instruction(
            request_instruction=request_instruction,
            target_hint=target_hint,
        )

        return self._keyword_overlay_action(
            goal_description=goal_description,
            selection_instruction=selection_instruction,
            target_hint=target_hint,
            action_type=ActionType.HANDLE_UPLOAD,
            action_kwargs={"upload_file_path": intent.value},
            target_context_guard=target_context_guard,
            confirm_before_interaction=confirm_before_interaction,
        )

    def _keyword_datetime(
        self,
        *,
        goal_description: str,
        payload: str,
        helper: Optional[str],
        additional_context: str,
        target_context_guard: Optional[str],
        confirm_before_interaction: bool,
    ) -> Optional[bool]:
        intent = parse_action_intent(goal_description)
        if not intent or intent.action != "datetime" or not intent.value:
            return None

        target_hint_raw = intent.target_text or helper or intent.helper_text
        target_hint = self._normalize_hint(target_hint_raw)
        if not target_hint:
            print("ℹ️ DATETIME command missing target hint – falling back")
            return None

        request_instruction = self._normalize_hint(goal_description)
        selection_instruction = self._compose_selection_instruction(
            request_instruction=request_instruction,
            target_hint=target_hint,
        )

        return self._keyword_overlay_action(
            goal_description=goal_description,
            selection_instruction=selection_instruction,
            target_hint=target_hint,
            action_type=ActionType.HANDLE_DATETIME,
            action_kwargs={"datetime_value": intent.value},
            target_context_guard=target_context_guard,
            confirm_before_interaction=confirm_before_interaction,
        )

    def _keyword_scroll(
        self,
        *,
        goal_description: str,
        payload: str,
        helper: Optional[str],
        target_context_guard: Optional[str],
        confirm_before_interaction: bool,
    ) -> Optional[bool]:
        text = " ".join(filter(None, [payload, helper, goal_description])).lower()
        direction = "down"
        if any(term in text for term in ["up", "top", "page up"]):
            direction = "up"
        elif any(term in text for term in ["left"]):
            direction = "left"
        elif any(term in text for term in ["right"]):
            direction = "right"
        elif "bottom" in text:
            direction = "down"

        return self._execute_keyword_plan(
            action_steps=[ActionStep(action=ActionType.SCROLL, scroll_direction=direction)],
            reasoning=f"Scroll {direction} command.",
            target_context_guard=target_context_guard,
            confirm_before_interaction=confirm_before_interaction,
        )

    def _keyword_wait(
        self,
        *,
        goal_description: str,
        payload: str,
        helper: Optional[str],
        target_context_guard: Optional[str],
        confirm_before_interaction: bool,
    ) -> Optional[bool]:
        text = " ".join(filter(None, [payload, helper, goal_description]))
        duration_ms = self._parse_duration_ms(text)
        return self._execute_keyword_plan(
            action_steps=[ActionStep(action=ActionType.WAIT, wait_time_ms=duration_ms)],
            reasoning=f"Wait for {duration_ms} ms.",
            target_context_guard=target_context_guard,
            confirm_before_interaction=confirm_before_interaction,
        )

    def _keyword_press(
        self,
        *,
        goal_description: str,
        payload: str,
        helper: Optional[str],
        target_context_guard: Optional[str],
        confirm_before_interaction: bool,
    ) -> Optional[bool]:
        key = extract_press_target(goal_description)
        if not key:
            key = (payload or helper or "").strip()
        if not key:
            print("ℹ️ PRESS command missing key – falling back")
            return None

        return self._execute_keyword_plan(
            action_steps=[ActionStep(action=ActionType.PRESS, keys_to_press=key)],
            reasoning=f"Press '{key}' command.",
            target_context_guard=target_context_guard,
            confirm_before_interaction=confirm_before_interaction,
        )

    def _keyword_stop(
        self,
        *,
        goal_description: str,
        target_context_guard: Optional[str],
        confirm_before_interaction: bool,
    ) -> Optional[bool]:
        return self._execute_keyword_plan(
            action_steps=[ActionStep(action=ActionType.STOP)],
            reasoning="Stop command.",
            target_context_guard=target_context_guard,
            confirm_before_interaction=confirm_before_interaction,
        )

    def _keyword_back(
        self,
        *,
        goal_description: str,
        target_context_guard: Optional[str],
        confirm_before_interaction: bool,
    ) -> Optional[bool]:
        return self._execute_keyword_plan(
            action_steps=[ActionStep(action=ActionType.BACK)],
            reasoning="Back navigation command.",
            target_context_guard=target_context_guard,
            confirm_before_interaction=confirm_before_interaction,
        )

    def _keyword_forward(
        self,
        *,
        goal_description: str,
        target_context_guard: Optional[str],
        confirm_before_interaction: bool,
    ) -> Optional[bool]:
        return self._execute_keyword_plan(
            action_steps=[ActionStep(action=ActionType.FORWARD)],
            reasoning="Forward navigation command.",
            target_context_guard=target_context_guard,
            confirm_before_interaction=confirm_before_interaction,
        )

    def _keyword_open(
        self,
        *,
        goal_description: str,
        payload: str,
        helper: Optional[str],
        target_context_guard: Optional[str],
        confirm_before_interaction: bool,
    ) -> Optional[bool]:
        url = self._extract_url(payload) or self._extract_url(helper) or self._extract_url(goal_description)
        if not url:
            print("ℹ️ NAVIGATE command missing URL – falling back")
            return None

        return self._execute_keyword_plan(
            action_steps=[ActionStep(action=ActionType.OPEN, url=url)],
            reasoning=f"Navigate to {url}.",
            target_context_guard=target_context_guard,
            confirm_before_interaction=confirm_before_interaction,
        )

    def _parse_duration_ms(self, text: Optional[str]) -> int:
        default_ms = 1000
        if not text:
            return default_ms
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if not match:
            return default_ms
        value = float(match.group(1))
        lowered = text.lower()
        if any(unit in lowered for unit in ["hour", "hours", "hr", "hrs"]):
            ms = int(value * 3_600_000)
        elif any(unit in lowered for unit in ["minute", "minutes", "min", "mins"]):
            ms = int(value * 60_000)
        elif any(unit in lowered for unit in ["ms", "millisecond", "milliseconds"]):
            ms = int(value)
        else:
            ms = int(value * 1_000)
        return max(ms, 0)

    def _extract_url(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        text = text.strip()
        if not text:
            return None
        url_match = re.search(r"(https?://[^\s]+)", text, flags=re.IGNORECASE)
        if url_match:
            candidate = url_match.group(1).rstrip(").,;\"'")
            return candidate
        domain_match = re.search(r"\b([a-z0-9][a-z0-9\.-]*\.[a-z]{2,})(/[^\s]*)?\b", text, flags=re.IGNORECASE)
        if domain_match:
            domain = domain_match.group(1).rstrip(").,;\"'")
            path = (domain_match.group(2) or "").rstrip(").,;\"'")
            if not domain.startswith(("http://", "https://")):
                return f"https://{domain}{path}"
            return f"{domain}{path}"
        return None

    def _enable_goal_evaluation_override(self) -> List:
        """Override goal evaluation with automatic success."""
        # Goal system removed - return empty list
        return []

    def _restore_goal_evaluations(self) -> None:
        """Restore original goal evaluation behavior after keyword command completes."""
        # Goal system removed - no-op
        return

    def _generate_plan(
        self,
        goal_description: str,
        additional_context: str,
        screenshot: bytes,
        page_info: PageInfo,
        target_context_guard: Optional[str] = None,
    ) -> Optional[VisionPlan]:
        """Generate an action plan using numbered element detection"""

        # Disable context guards while running in agent mode to avoid repeated guard blocks
        if self._agent_mode:
            target_context_guard = None

        # Goal system removed - always do detection
        needs_detection = True

        current_mode = self._get_current_interpretation_mode()
        semantic_hint = self._get_semantic_target(goal_description)
        print(f"[PlanGen] interpretation_mode={current_mode} semantic_hint={'yes' if semantic_hint else 'no'} for '{goal_description}'")

        dedup_context = self._build_dedup_prompt_context(goal_description)

        if not needs_detection:
            print("🚫 Skipping element detection - goal doesn't require it")
            # Create action plan without element detection
            # In agent mode, limit to 1 step for strict action execution
            max_steps = 1 if self._agent_mode else None
            
            plan = self.plan_generator.create_plan(
                goal_description=goal_description,
                additional_context=additional_context,
                detected_elements=PageElements(elements=[]),
                page_info=page_info,
                screenshot=screenshot,
                active_goal=None,
                retry_goal=None,
                page=self.page,
                command_history=self.command_history,
                dedup_context=dedup_context,
                target_context_guard=target_context_guard,
                max_steps=max_steps,
            )
            return plan

        element_data, screenshot_with_overlays, clean_screenshot = self._collect_overlay_data(goal_description, page_info)

        if not element_data:
            print("❌ No interactive elements found for overlays")
            return None

        # Step 3: Generate plan with element indices (and optional filtered overlay list)
        # In agent mode, limit to 1 step for strict action execution
        max_steps = 1 if self._agent_mode else None
        
        plan = self.plan_generator.create_plan_with_element_indices(
            goal_description=goal_description,
            additional_context=additional_context,
            element_data=element_data,
            screenshot_with_overlays=screenshot_with_overlays,
            page_info=page_info,
            active_goal=None,
            retry_goal=None,
            page=self.page,
            command_history=self.command_history,
            interpretation_mode=current_mode,
            semantic_hint=semantic_hint,
            dedup_context=dedup_context,
            target_context_guard=target_context_guard,
            max_steps=max_steps,
        )

        if plan:
            try:
                plan_payload = plan.model_dump()
            except AttributeError:
                plan_payload = plan.dict()
            except Exception as e:
                plan_payload = f"<unable to serialize plan: {e}>"
            print(f"[PlanGen] Plan response: {plan_payload}")

        # Step 4: Clean up overlays after plan generation
        self.overlay_manager.remove_overlays()

        return plan

    # -------------------- Public memory helpers (general) --------------------
    # Memory store methods removed - deduplication now handled by focus manager

    def _create_goal_from_description(self, goal_description: str, modifier: Optional[List[str]] = None) -> tuple[None, str]:
        """Goal system removed - returns None, description"""
        return None, goal_description

    def _create_goal_from_keyword(self, keyword: str, payload: str) -> None:
        """Goal system removed - returns None"""
        return None

    def _create_while_goal_from_parts(self, goal_description: str, cond_text: str, body_text: str, route: Optional[str] = None, modifier: Optional[List[str]] = None, fail_on_body_failure: Optional[bool] = None) -> Optional[Any]:
        """Create WhileGoal from explicit condition/body parts with route determination."""
        try:
            # Determine route: modifier first, then parsed route, then fail
            determined_route = None
            
            # 1. Check modifier parameter first
            if modifier:
                for mod in modifier:
                    if mod.lower() in ["see", "page"]:
                        determined_route = mod.lower()
                        break
            
            # 2. Use parsed route if no modifier route
            if not determined_route and route:
                determined_route = route.lower()
            
            # 3. Fail if no route specified
            if not determined_route:
                raise ValueError("While goal requires route specification. Use modifier=['see'] or modifier=['page'] or specify in command like 'while see: condition do: action'")
            
            if determined_route not in ["see", "page"]:
                raise ValueError(f"Invalid route '{determined_route}'. Must be 'see' or 'page'")
            
            # Get max_retries and fail_on_body_failure from temp kwargs if available
            _max_retries = getattr(self, '_temp_goal_kwargs', {}).get('max_retries', 3)  # Unused - goal system removed
            if fail_on_body_failure is None:
                fail_on_body_failure = getattr(self, '_temp_goal_kwargs', {}).get('fail_on_body_failure', True)
            
            # Goal system removed - WhileGoal no longer exists
            return None
        except Exception as e:
            print(f"⚠️ Error creating structured WhileGoal: {e}")
            return None

    def _create_for_goal_from_parts(self, iteration_mode: str, iteration_target: str, loop_body: str, break_conditions: Optional[str] = None) -> Optional[Any]:
        """Create ForGoal from parsed iteration parts."""
        try:
            # Parse iteration target based on mode (unused - goal system removed)
            if iteration_mode == "count":
                _target_count = int(iteration_target)
                _target = _target_count
            elif iteration_mode == "elements":
                _target = iteration_target
            elif iteration_mode == "items":
                # Parse "item|list" format
                if "|" in iteration_target:
                    item, item_list = iteration_target.split("|", 1)
                    # For now, create a simple list - in real implementation, this would parse the actual list
                    _target = [item.strip(), "item2", "item3"]  # Placeholder
                else:
                    _target = [iteration_target]
            else:
                print(f"⚠️ Unsupported iteration mode: {iteration_mode}")
                return None
            
            # Parse break conditions (unused - goal system removed)
            _break_conds = [break_conditions] if break_conditions else []
            
            # Get max_retries from temp kwargs if available (unused - goal system removed)
            _max_retries = getattr(self, '_temp_goal_kwargs', {}).get('max_retries', 3)
            
            # Goal system removed - ForGoal no longer exists
            return None
            
        except Exception as e:
            print(f"⚠️ Error creating structured ForGoal: {e}")
            return None

    def _execute_for_loop(self, goal: Any, start_time: float, parent_command_id: Optional[str] = None) -> bool:
        """Goal system removed - for loops no longer supported"""
        print("❌ For loops are no longer supported (goal system removed)")
        return False

    def _execute_while_loop(self, goal: Any, start_time: float) -> bool:
        """Goal system removed - while loops no longer supported"""
        print("❌ While loops are no longer supported (goal system removed)")
        return False

    def _create_if_goal_from_parts(self, condition_text: str, success_text: str, fail_text: Optional[str], route: Optional[str] = None, modifier: Optional[List[str]] = None) -> Optional[Any]:
        """Create IfGoal from explicit parts with route determination."""
        try:
            # Determine route: modifier first, then parsed route, then fail
            determined_route = None
            
            # 1. Check modifier parameter first
            if modifier:
                for mod in modifier:
                    if mod.lower() in ["see", "page"]:
                        determined_route = mod.lower()
                        break
            
            # 2. Use parsed route if no modifier route
            if not determined_route and route:
                determined_route = route.lower()
            
            # 3. Default to page route if none specified
            if not determined_route:
                determined_route = "page"
                print("ℹ️ No route specified for IfGoal; defaulting to 'page' evaluation.")
            
            if determined_route not in ["see", "page"]:
                raise ValueError(f"Invalid route '{determined_route}'. Must be 'see' or 'page'")

            print(f"Success text: '{success_text}'")
            
            # Handle reference commands differently - don't create goals for them
            normalized_success = success_text.strip()
            if normalized_success.lower().startswith("ref:"):
                # Create a simple placeholder goal for reference commands
                # Goal system removed - RefPlaceholderGoal removed
                class RefPlaceholderGoal:
                    def __init__(self, description: str):
                        super().__init__(description)
                    def evaluate(self, context):
                        return None  # Will be handled by _evaluate_if_goal
                    def get_description(self, context):
                        return self.description
                success_goal = RefPlaceholderGoal(success_text)
            else:
                success_goal, _ = self._create_goal_from_description(success_text)
                if not success_goal:
                    return None
            if success_goal:
                setattr(success_goal, "fast_command_text", success_text)
            
            print(f"🔀 Created success goal: '{success_goal.description}'")
            
            if fail_text and fail_text.strip():
                normalized_fail = fail_text.strip()
                if normalized_fail.lower().startswith("ref:"):
                    # Create a simple placeholder goal for reference commands
                    # Goal system removed - BaseGoal removed
                    class RefPlaceholderGoal:
                        def __init__(self, description: str):
                            self.description = description
                        def evaluate(self, context):
                            return None  # Will be handled by _evaluate_if_goal
                        def get_description(self, context):
                            return self.description
                    fail_goal = RefPlaceholderGoal(fail_text)
                else:
                    fail_goal, _ = self._create_goal_from_description(fail_text)
                if fail_goal:
                    setattr(fail_goal, "fast_command_text", fail_text)
            else:
                fail_goal = None

            # Unused variables (goal system removed)
            _fail_desc = fail_goal.description if fail_goal else "(no fail action)"
            _max_retries = getattr(self, '_temp_goal_kwargs', {}).get('max_retries', 3)
            
            # Goal system removed - IfGoal no longer exists
            return None
        except Exception as e:
            print(f"⚠️ Error creating structured IfGoal: {e}")
            return None


    def _evaluate_if_goal(self, if_goal: Any) -> tuple[Optional[Any], str]:
        """Goal system removed - if goals no longer supported"""
        print("❌ If goals are no longer supported (goal system removed)")
        return None, ""

    def _print_goal_summary(self) -> None:
        """Print a summary of all goal statuses - deprecated, kept for compatibility"""
        # Goal summaries no longer available without goal system
        pass

    def _add_to_command_history(self, command: str) -> None:
        """Add a command to the history, maintaining max size"""
        if command and command.strip():
            self.command_history.append(command.strip())
            # Keep only the last max_command_history commands
            if len(self.command_history) > self.max_command_history:
                self.command_history = self.command_history[-self.max_command_history:]
            try:
                self.event_logger.command_history(command.strip())
            except Exception:
                pass
    
    def _handle_focus_commands(self, goal_description: str, overlay_manager: OverlayManager) -> Optional[bool]:
        """
        Handle focus, subfocus, and undo commands.
        
        Args:
            goal_description: The goal description to check for focus commands
        
        Returns:
            bool: True if command was handled successfully, False if failed, None if not a focus command
        """
        try:
            # Check for focus commands
            focus_parsed = parse_focus_command(goal_description)
            if focus_parsed:
                command_type, payload = focus_parsed
                if command_type == "focus":
                    # Get current page info for AI-first approach
                    page_info = self.page_utils.get_page_info()
                    success = self.focus_manager.focus_on_elements(payload, page_info, overlay_manager)
                    if success:
                        self.logger.log_focus_operation("focus", payload, True)
                        print("🎯 AI-first focus command executed successfully")
                        return True
                    else:
                        self.logger.log_focus_operation("focus", payload, False)
                        print("❌ AI-first focus command failed")
                        return False
            
            # Subfocus commands not needed in AI-first approach
            
            # Check for undo commands
            undo_parsed = parse_undo_command(goal_description)
            if undo_parsed:
                command_type, payload = undo_parsed
                if command_type == "undo":
                    success = self.focus_manager.undo_focus()
                    if success:
                        self.logger.log_focus_operation("undo", "focus", True)
                        print("↩️ Undo command executed successfully")
                        return True
                    else:
                        self.logger.log_focus_operation("undo", "focus", False)
                        print("❌ Undo command failed")
                        return False
                elif command_type == "undofocus":
                    success = self.focus_manager.undo_focus()
                    if success:
                        self.logger.log_focus_operation("undofocus", "focus", True)
                        print("↩️ Undofocus command executed successfully")
                        return True
                    else:
                        self.logger.log_focus_operation("undofocus", "focus", False)
                        print("❌ Undofocus command failed")
                        return False
            
            # Check for keyword commands that might be focus-related
            kw_parsed = parse_keyword_command(goal_description)
            if kw_parsed:
                keyword, payload, helper = kw_parsed
                if keyword == "focus":
                    success = self.focus_manager.focus_on_elements(payload, page_info, overlay_manager)
                    if success:
                        print("🎯 Focus command executed successfully")
                        return True
                    else:
                        print("❌ Focus command failed")
                        return False
                elif keyword == "subfocus":
                    success = self.focus_manager.subfocus_on_elements(payload, page_info, overlay_manager)
                    if success:
                        print("🎯 Subfocus command executed successfully")
                        return True
                    else:
                        print("❌ Subfocus command failed")
                        return False
                elif keyword == "undo":
                    success = self.focus_manager.undo_focus()
                    if success:
                        print("↩️ Undo command executed successfully")
                        return True
                    else:
                        print("❌ Undo command failed")
                        return False
                elif keyword == "undofocus":
                    success = self.focus_manager.undo_focus()
                    if success:
                        print("↩️ Undofocus command executed successfully")
                        return True
                    else:
                        print("❌ Undofocus command failed")
                        return False
            
            return None  # Not a focus command
            
        except Exception as e:
            print(f"⚠️ Error handling focus commands: {e}")
            return False
    
    def _handle_dedup_commands(self, goal_description: str) -> Optional[bool]:
        """
        Handle dedup enable/disable commands.
        
        Args:
            goal_description: The goal description to check for dedup commands
        
        Returns:
            bool: True if command was handled successfully, False if failed, None if not a dedup command
        """
        try:
            goal_lower = goal_description.lower().strip()
            
            # Check for dedup: enable
            if goal_lower in ["dedup: enable", "dedup:enabled"]:
                self.dedup_mode = "on"
                if hasattr(self, 'deduper') and self.deduper:
                    self.deduper.set_dedup_enabled(True)
                self.logger.log(LogLevel.INFO, LogCategory.SYSTEM, "Deduplication enabled")
                print("🧹 Deduplication enabled")
                return True
            
            # Check for dedup: disable
            elif goal_lower in ["dedup: disable", "dedup:disabled"]:
                self.dedup_mode = "off"
                if hasattr(self, 'deduper') and self.deduper:
                    self.deduper.set_dedup_enabled(False)
                self.logger.log(LogLevel.INFO, LogCategory.SYSTEM, "Deduplication disabled")
                print("🧹 Deduplication disabled")
                return True
            
            return None  # Not a dedup command
            
        except Exception as e:
            print(f"⚠️ Error handling dedup commands: {e}")
            return False
    
    def _handle_ref_commands(self, goal_description: str) -> Optional[bool]:
        """
        Handle ref commands that execute stored multi-commands.
        
        Args:
            goal_description: The goal description to check for ref commands
        
        Returns:
            bool: True if command was handled successfully, False if failed, None if not a ref command
        """
        try:
            goal_lower = goal_description.lower().strip()
            
            # Check for ref: refID pattern
            if goal_lower.startswith("ref:"):
                ref_id = goal_description[4:].strip()
                
                if not ref_id:
                    print("❌ No ref ID provided after 'ref:'")
                    return False
                
                if ref_id not in self.command_refs:
                    print(f"❌ Ref ID '{ref_id}' not found in stored commands")
                    return False
                
                ref_entry = self.command_refs[ref_id]
                stored_prompts = ref_entry.get("prompts", [])
                all_must_be_true = bool(ref_entry.get("all_must_be_true", False))
                ref_mode = ref_entry.get("interpretation_mode")
                ref_additional_context = ref_entry.get("additional_context", "")
                ref_target_context_guard = ref_entry.get("target_context_guard")
                ref_skip_post_guard_refinement = ref_entry.get("skip_post_guard_refinement", True)
                ref_confirm_before_interaction = ref_entry.get("confirm_before_interaction", False)
                ref_max_attempts = ref_entry.get("max_attempts")
                ref_max_retries = ref_entry.get("max_retries")
                stored_command_id = ref_entry.get("command_id")  # Get the original command ID

                if not stored_prompts:
                    print(f"⚠️ Ref ID '{ref_id}' has no stored commands")
                    return True

                summary_mode = 'ALL' if all_must_be_true else 'ANY'
                mode_suffix = f", interpretation={ref_mode}" if ref_mode else ""
                print(f"🔄 Executing {len(stored_prompts)} stored commands for ref ID: {ref_id} (mode={summary_mode}{mode_suffix})")
                results: List[bool] = []

                # Use the stored command ID as the parent, fallback to current if not available
                ref_command_id = stored_command_id or self.command_ledger.get_current_command_id()
                
                for i, prompt in enumerate(stored_prompts, 1):
                    print(f"▶️ Executing stored command {i}/{len(stored_prompts)}: {prompt}")
                    
                    # Generate a child command ID
                    child_cmd_id = f"{ref_command_id}_cmd{i}" if ref_command_id else None
                    
                    success = bool(
                        self.act(
                            prompt,
                            additional_context=ref_additional_context,
                            interpretation_mode=ref_mode,
                            target_context_guard=ref_target_context_guard,
                            skip_post_guard_refinement=ref_skip_post_guard_refinement,
                            confirm_before_interaction=ref_confirm_before_interaction,
                            command_id=child_cmd_id,  # Pass child ID
                            max_attempts=ref_max_attempts,  # Pass custom max_attempts
                            max_retries=ref_max_retries,    # Pass custom max_retries
                        )
                    )
                    results.append(success)
                    if success:
                        print(f"   ✅ Stored command {i} succeeded")
                    else:
                        print(f"   ❌ Stored command {i} failed")
                        
                        # If all_must_be_true is True and this command failed, abort immediately
                        if all_must_be_true:
                            print("🛑 Aborting execution due to failure (all_must_be_true=True)")
                            summary_mode = "ALL must succeed"
                            print(f"📊 Ref '{ref_id}' evaluation ({summary_mode}) → ❌ False (aborted after command {i})")
                            return False

                # If we get here, either all_must_be_true=False or all commands succeeded
                final_result = all(results) if all_must_be_true else any(results)
                summary_mode = "ALL must succeed" if all_must_be_true else "ANY success suffices"
                print(f"📊 Ref '{ref_id}' evaluation ({summary_mode}) → {'✅ True' if final_result else '❌ False'}")
                return final_result
            
            return None  # Not a ref command
            
        except Exception as e:
            traceback.print_exc()
            print(f"⚠️ Error handling ref commands: {e}")
            return False
    
    def _filter_elements_by_focus(self, element_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter element data based on current focus context.
        
        Args:
            element_data: List of element data dictionaries
        
        Returns:
            Filtered list of element data
        """
        try:
            current_focus = self.focus_manager.get_current_focus_context()
            if not current_focus:
                return element_data  # No focus means all elements are available
            
            
            # Filter elements that are within the focus context
            focused_elements = []
            for elem in element_data:
                element_id = str(elem.get('index', ''))
                is_in_focus = self.focus_manager.is_element_in_focus(element_id, elem)
                if is_in_focus:
                    focused_elements.append(elem)
                    print(f"✅ Element {element_id} is in focus: {elem.get('description', 'Unknown')}")
                else:
                    print(f"🚫 Filtered out element {element_id} (not in focus): {elem.get('description', 'Unknown')}")
            
            print(f"🎯 Focused on {len(focused_elements)} out of {len(element_data)} elements")
            return focused_elements
            
        except Exception as e:
            print(f"⚠️ Error filtering elements by focus: {e}")
            return element_data

    def queue_action(self, action: str, command_id: Optional[str] = None, 
                    priority: int = 0, metadata: Dict[str, Any] = None) -> None:
        """
        Queue an action for later execution.
        
        Args:
            action: The action to execute (e.g., "click: button")
            command_id: Optional command ID for tracking
            priority: Priority level (higher = executed first)
            metadata: Optional metadata dict
        """
        self.action_queue.enqueue(action, command_id, priority, metadata)
        print(f"📋 Queued action: {action} [Priority: {priority}]")
    
    def process_queue(self) -> int:
        """
        Process all queued actions, returns count of executed actions.
        
        Returns:
            int: Number of actions successfully executed
        """
        executed = 0
        failed = 0
        
        while not self.action_queue.is_empty():
            queued_action = self.action_queue.dequeue()
            if queued_action:
                print(f"🔄 Processing queued action: {queued_action.action}")
                try:
                    success = self.act(
                        queued_action.action,
                        command_id=queued_action.command_id
                    )
                    if success:
                        executed += 1
                        print(f"   ✅ Queued action succeeded: {queued_action.action}")
                    else:
                        failed += 1
                        print(f"   ❌ Queued action failed: {queued_action.action}")
                except Exception as e:
                    failed += 1
                    print(f"   ❌ Queued action error: {e}")
        
        if failed > 0:
            print(f"⚠️ {failed} queued actions failed")
        
        return executed
    
