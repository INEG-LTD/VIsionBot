"""
Vision Bot - Clean modular version.
"""
from __future__ import annotations

import os
import re
import threading
import time
import traceback
import uuid
from typing import Any, Optional, List, Dict, Tuple, Union, Type, Callable

from playwright.sync_api import Browser, Page, Playwright
# Compatible import for playwright_stealth across versions
try:
    from playwright_stealth import Stealth
    stealth_instance = Stealth()

    def stealth_sync(page):
        return stealth_instance.apply_stealth_sync(page)

except ImportError:
    # Fallback: disable stealth if not available
    def stealth_sync(page):
        pass  # No-op function

from agent.mini_goal_manager import MiniGoalMode, MiniGoalScriptContext, MiniGoalTrigger
from models import VisionPlan, PageElements
from models.core_models import ActionStep, ActionType, PageInfo
from element_detection import ElementDetector
from element_detection.overlay_manager import OverlayManager
from action_executor import ActionExecutor
from utils import PageUtils
from action_queue import ActionQueue
from session_tracker import SessionTracker, InteractionType
# Removed goal imports - goals system no longer used
from planner.plan_generator import PlanGenerator
from utils.intent_parsers import (
    extract_click_target,
    extract_press_target,
    parse_action_intent,
    parse_keyword_command,
)
from interaction_deduper import InteractionDeduper
from utils.bot_logger import get_logger, LogLevel, LogCategory
from utils.semantic_targets import SemanticTarget, build_semantic_target
from utils.event_logger import EventLogger, set_event_logger
from action_ledger import ActionLedger
from ai_utils import (
    ReasoningLevel,
    set_default_model,
    set_default_reasoning_level,
    set_default_agent_model,
    set_default_agent_reasoning_level,
)
from agent import AgentController
from agent.agent_result import AgentResult
from action_result import ActionResult
from pydantic import BaseModel, Field
from bot_config import BotConfig
from browser_provider import BrowserProvider, create_browser_provider
from middleware import MiddlewareManager, ActionContext
from error_handling import (
    BotNotStartedError,
    BotTerminatedError,
    ActionFailedError,
    ExtractionError,
    ValidationError,
    ErrorContext,
)


class ExecutionTimer:
    """Tracks execution timings for tasks, iterations, and actions"""
    
    def __init__(self):
        self.task_start_time: Optional[float] = None
        self.task_end_time: Optional[float] = None
        self.iterations: List[Dict[str, float]] = []  # List of {start, end} dicts
        self.actions: List[Dict[str, Any]] = []  # List of {action_id, goal, start, end} dicts
        self.current_iteration_start: Optional[float] = None
        self.current_action_id: Optional[str] = None
        self.current_action_start: Optional[float] = None
        self._current_goal_text: str = ""
    
    def start_task(self) -> None:
        """Start tracking task execution"""
        self.task_start_time = time.time()
        self.iterations = []
        self.actions = []
    
    def end_task(self) -> None:
        """End task tracking"""
        self.task_end_time = time.time()
        # End any active iteration or action
        if self.current_iteration_start is not None:
            self.end_iteration()
        if self.current_action_start is not None:
            self.end_action()
    
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
    
    def start_action(self, action_id: str, goal: str) -> None:
        """Start tracking an action"""
        # End previous action if still active
        if self.current_action_start is not None:
            self.end_action()
        self.current_action_id = action_id
        self.current_action_start = time.time()
        self._current_goal_text = goal
    
    def end_action(self) -> None:
        """End current action tracking"""
        if self.current_action_start is not None and self.current_action_id is not None:
            self.actions.append({
                "action_id": self.current_action_id,
                "goal": getattr(self, "_current_goal_text", ""),
                "start": self.current_action_start,
                "end": time.time()
            })
            self.current_action_id = None
            self.current_action_start = None
            self._current_goal_text = ""
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all timings"""
        summary = {
            "task": {},
            "iterations": [],
            "actions": []
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
        
        # Action timings
        for action_data in self.actions:
            duration = action_data["end"] - action_data["start"]
            summary["actions"].append({
                "action_id": action_data["action_id"],
                "goal": action_data.get("goal", ""),
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
    
    def log_summary(self, event_logger=None) -> None:
        """Log timing summary to console"""
        summary = self.get_summary()
        
        try:
            # Use event_logger if provided, otherwise use print
            log_func = event_logger.system_info if event_logger else print
            
            log_func("\n" + "="*60)
            log_func("⏱️  EXECUTION TIMING SUMMARY")
            log_func("="*60)
            
            # Task timing
            if summary["task"]:
                log_func(f"\n📋 Task Duration: {summary['task']['duration_formatted']} ({summary['task']['duration_seconds']}s)")
            
            # Iteration timings
            if summary["iterations"]:
                total_iter_time = sum(iter_data["duration_seconds"] for iter_data in summary["iterations"])
                avg_iter_time = total_iter_time / len(summary["iterations"])
                log_func(f"\n🔄 Iterations: {len(summary['iterations'])}")
                log_func(f"   Total iteration time: {self._format_duration(total_iter_time)}")
                log_func(f"   Average per iteration: {self._format_duration(avg_iter_time)}")
                fastest_iter = min(summary['iterations'], key=lambda x: x['duration_seconds'])
                slowest_iter = max(summary['iterations'], key=lambda x: x['duration_seconds'])
                log_func(f"   Fastest iteration: {fastest_iter['duration_formatted']}")
                log_func(f"   Slowest iteration: {slowest_iter['duration_formatted']}")
            
            # Action timings
            if summary["actions"]:
                total_action_time = sum(action_data["duration_seconds"] for action_data in summary["actions"])
                avg_action_time = total_action_time / len(summary["actions"])
                log_func(f"\n🎯 Actions: {len(summary['actions'])}")
                log_func(f"   Total action time: {self._format_duration(total_action_time)}")
                log_func(f"   Average per action: {self._format_duration(avg_action_time)}")
                
                # Show top 5 slowest actions
                sorted_actions = sorted(summary["actions"], key=lambda x: x["duration_seconds"], reverse=True)
                log_func("\n   Top 5 slowest actions:")
                for i, action in enumerate(sorted_actions[:5], 1):
                    goal_text = action["goal"][:50] + "..." if len(action.get("goal", "")) > 50 else action.get("goal", "")
                    log_func(f"   {i}. {action['action_id']}: {action['duration_formatted']} - {goal_text}")
            
            log_func("="*60 + "\n")
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
        
        # Store config for later use
        self.config = config
        
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
        self.completion_mode = config.execution.completion_mode
        self.enable_sub_agents = config.execution.enable_sub_agents
        self.dedup_mode = config.execution.dedup_mode

        self.started = False
        
        # Extract element configuration
        self.max_detailed_elements = config.elements.max_detailed_elements
        self.include_detailed_elements = config.elements.include_detailed_elements
        self.two_pass_planning = config.elements.two_pass_planning
        self.max_coordinate_overlays = config.elements.max_coordinate_overlays
        self.merge_overlay_selection = config.elements.merge_overlay_selection
        self.overlay_only_planning = config.elements.overlay_only_planning
        self.overlay_mode = getattr(config.elements, "overlay_mode", "interactive")
        self.include_textless_overlays = getattr(config.elements, "include_textless_overlays", False)
        
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
        self.include_overlays_in_agent_context = config.elements.include_overlays_in_agent_context
        self.include_visible_text_in_agent_context = config.elements.include_visible_text_in_agent_context

        # Extract act function configuration
        self.act_enable_target_context_guard = config.act_function.enable_target_context_guard
        self.act_enable_modifier = config.act_function.enable_modifier
        self.act_enable_additional_context = config.act_function.enable_additional_context
        
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
        
        # Action history for "do that again" functionality
        self.command_history: List[str] = []
        self.max_command_history: int = 10  # Keep last 10 actions
        
        # Multi-command reference storage
        self.command_refs: Dict[str, Dict[str, Any]] = {}  # refID -> metadata about stored prompts

        # Mini goals registry (will be passed to AgentController)
        self.mini_goals: List[Dict[str, Any]] = []

        # Action queue system for deferred actions
        self.action_queue = ActionQueue()
        self._auto_process_queue = True  # Auto-process queue after each act()


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

        # Semantic resolution helpers (used internally for element matching)
        self._semantic_target_cache: Dict[str, Optional[SemanticTarget]] = {}
        
        # Deferred input handling
        self.defer_input_handler: Optional[Callable[[str, Any], str]] = None
        self._pending_defer_input: Optional[Dict[str, Any]] = None
        
        # Execution timer for tracking task, iteration, and action timings
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
        self.agent_controller = None  # Current agent controller instance (set during execute_task)
        
        # Screenshot and overlay caching for performance optimization
        self._cached_screenshot_with_overlays = None
        self._cached_clean_screenshot = None
        self._cached_overlay_data = None
        self._cached_dom_signature = None
        # Focus system removed - no longer caching focus context
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
        
        # Initialize action ledger for tracking action execution
        self.action_ledger: ActionLedger = ActionLedger()
        
        # Initialize action executor with deduper and action ledger
        # Pass a callback so action executor can execute actions through bot infrastructure
        self.action_executor: ActionExecutor = ActionExecutor(
            page, 
            self.session_tracker, 
            self.page_utils, 
            self.deduper, 
            self.action_ledger,
            execute_action_callback=self._execute_action_via_bot,
            user_messages_config=self.config.user_messages if self.config else None
        )
        
        # Plan generator for AI planning prompts
        self.plan_generator: PlanGenerator = PlanGenerator(
            include_detailed_elements=self.include_detailed_elements,
            max_detailed_elements=self.max_detailed_elements,
            merge_overlay_selection=self.merge_overlay_selection,
            return_overlay_only=self.overlay_only_planning,
            overlay_selection_max_samples=self.overlay_selection_max_samples,
            include_textless_overlays=self.include_textless_overlays,
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
    
    def _execute_action_via_bot(self, action_command: str) -> bool:
        """
        Execute an action command through the bot's infrastructure.
        This allows the action executor to execute actions (like click) using
        the full bot infrastructure (overlay detection, element finding, etc.)
        
        Args:
            action_command: The action command to execute (e.g., "click: Basle-Country, CHE")
            
        Returns:
            bool: True if action succeeded, False otherwise
        """
        try:
            result = self.act(
                goal_description=action_command,
                additional_context="",
                target_context_guard=None,
                max_attempts=3,  # Fewer attempts for auto-converted actions
                allow_non_clickable_clicks=True,
            )
            return result.success if result else False
        except Exception as e:
            print(f"    ⚠️ Error executing auto-converted action '{action_command}': {e}")
            return False

    def end(self) -> None:
        """
        Terminate the bot and prevent any subsequent operations.
        """
        if self.terminated:
            try:
                self.event_logger.system_warning("Bot is already terminated")
            except Exception:
                pass
            return
            
        try:
            self.event_logger.system_info("Terminating bot...")
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

    def __enter__(self) -> 'BrowserVisionBot':
        """
        Context manager entry point. Automatically calls start().
        
        Example:
            >>> with BrowserVisionBot(config=config) as bot:
            ...     bot.page.goto("https://example.com")
            ...     bot.act("Click the button")
            # Automatically calls end() on exit
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Context manager exit point. Automatically calls end() to cleanup resources.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self.end()

    def _check_termination(self) -> None:
        """
        Check if bot is terminated and raise error if so.
        
        Raises:
            BotTerminatedError: If bot has been terminated
        """
        if self.terminated:
            raise BotTerminatedError(
                "Bot has been terminated. No further operations are allowed.",
                context=ErrorContext(
                    error_type="BotTerminatedError",
                    message="Bot has been terminated. No further operations are allowed.",
                    metadata={"terminated": True}
                )
            )

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
            # Focus system removed - no longer needed
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
    def on_new_page_load(self, actions_to_take: List[str], run_once_per_url: bool = True, action_id: Optional[str] = None) -> None:
        """
        Register prompts to run via act() after each page load.

        Args:
            actions_to_take: List of commands to run on each page load
            run_once_per_url: If True, only run once per unique URL
            action_id: Optional action ID for tracking (auto-generated if not provided)

        Typical usage: on_new_page_load([
            "if a cookie banner is visible click the accept button",
            "close any newsletter modal if present",
        ], action_id="auto-page-load")
        """
        self._auto_on_load_actions = [a for a in (actions_to_take or []) if isinstance(a, str) and a.strip()]
        self._auto_on_load_run_once_per_url = bool(run_once_per_url)
        self._auto_on_load_enabled = True
        self._auto_on_load_urls_handled.clear()
        self._auto_on_load_action_id = action_id
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
            # Register parent action if we have an action_id
            parent_cmd_id = None
            if hasattr(self, '_auto_on_load_action_id') and self._auto_on_load_action_id:
                parent_action_id = self.action_ledger.register_action(
                    goal=f"on_new_page_load: {len(self._auto_on_load_actions)} actions",
                    action_id=self._auto_on_load_action_id,
                    metadata={"source": "on_new_page_load", "url": current_url}
                )
                self.action_ledger.start_action(parent_action_id)
            
            for i, prompt in enumerate(self._auto_on_load_actions, 1):
                try:
                    try:
                        self.event_logger.system_info(f"Auto-on-load: act('{prompt}')")
                    except Exception:
                        pass
                    # Snapshot user prompt so auto-action does not disrupt ongoing task
                    saved_user_prompt = getattr(self.session_tracker, 'user_prompt', "") if hasattr(self, 'session_tracker') else ""
                    
                    # Generate child action ID if we have a parent
                    child_action_id = f"{parent_action_id}_action{i}" if parent_action_id else None
                    
                    try:
                        # If we're inside act(), we still call act() but our snapshot/restore prevents disruption
                        # and _auto_on_load_running avoids re-entrancy loops
                        self.act(prompt, action_id=child_action_id)
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
            
            # Complete parent action
            if parent_cmd_id:
                self.action_ledger.complete_action(parent_action_id, success=True)
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

    def act(
        self,
        goal_description: str,
        additional_context: str = "",
        target_context_guard: Optional[str] = None,
        skip_post_guard_refinement: bool = True,
        confirm_before_interaction: bool = False,
        action_id: Optional[str] = None,
        max_attempts: Optional[int] = None,
        base_knowledge: Optional[List[str]] = None,
        **kwargs
    ) -> ActionResult:
        """
        Main method to achieve a goal using vision-based automation
        
        This method executes a single action based on natural language description.
        Validates inputs and bot state before execution.
        
        Args:
            goal_description: The goal to achieve (must use keyword format: "click: button", "type: text", etc.)
            additional_context: Extra context to help with planning
            target_context_guard: Guard condition for actions
            skip_post_guard_refinement: Skip refinement after guard checks
            confirm_before_interaction: Require user confirmation before each action
            action_id: Optional action ID for tracking (auto-generated if not provided)
            max_attempts: Override bot's max_attempts for this command (None = use bot default)
            base_knowledge: Optional list of knowledge rules/instructions that guide planning and element selection
        
        Returns:
            ActionResult - Structured result with success, message, confidence, and metadata
            
        Raises:
            BotTerminatedError: If bot has been terminated
            BotNotStartedError: If bot is not started
            ValidationError: If goal_description is empty or invalid
            
        Example:
            >>> bot.start()
            >>> bot.page.goto("https://example.com")
            >>> result = bot.act("click: login button")
            >>> if result.success:
            ...     print(f"Success: {result.message}")
            ...     print(f"Confidence: {result.confidence}")
            ...     print(f"Attempts: {result.metadata.get('attempts')}")
        """
        # Parameter validation
        if not goal_description or not goal_description.strip():
            raise ValidationError(
                "goal_description cannot be empty or whitespace",
                context=ErrorContext(
                    error_type="ValidationError",
                    message="goal_description cannot be empty or whitespace",
                    action_data={"goal_description": goal_description}
                )
            )
        
        if max_attempts is not None and max_attempts < 1:
            raise ValidationError(
                "max_attempts must be >= 1",
                context=ErrorContext(
                    error_type="ValidationError",
                    message="max_attempts must be >= 1",
                    action_data={"max_attempts": max_attempts}
                )
            )
        
        self._check_termination()
        
        if not self.started or not self.page:
            page_url = self.page.url if self.page else None
            raise BotNotStartedError(
                "Bot not started. Call bot.start() first.",
                context=ErrorContext(
                    error_type="BotNotStartedError",
                    message="Bot not started. Call bot.start() first.",
                    page_url=page_url,
                    action_data={"goal_description": goal_description}
                )
            )
        
        # Helper function to create ActionResult
        def _create_result(success: bool, message: str = "", error: Optional[str] = None, 
                          action_id: Optional[str] = None, duration: Optional[float] = None,
                          additional_metadata: Optional[Dict[str, Any]] = None,
                          data: Optional[Any] = None) -> ActionResult:
            """Create ActionResult with metadata"""
            metadata = {
                "goal_description": goal_description,
                "action_id": action_id,
            }
            if duration is not None:
                metadata["duration_ms"] = duration * 1000
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Calculate confidence based on success
            confidence = 0.9 if success else 0.1
            
            return ActionResult(
                success=success,
                message=message or ("Action completed successfully" if success else "Action failed"),
                confidence=confidence,
                metadata=metadata,
                error=error,
                data=data
            )
        
        # Track act() execution state
        self._in_act = True
        start_time = time.time()
        
        try:
            if not self.started:
                self.logger.log_error("Bot not started", "act() called before bot.start()")
                self.event_logger.system_error("Bot not started")
                if self.execution_timer.current_action_start is not None:
                    self.execution_timer.end_action()
                duration = (time.time() - start_time) if 'start_time' in locals() else None
                return _create_result(
                    False,
                    "Bot not started. Call bot.start() first.",
                    error="Bot not started",
                    action_id=action_id,
                    duration=duration
                )
            
            if self.page.url.startswith("about:blank"):
                self.logger.log_error("Page is on initial blank page", "act() called before navigation")
                self.event_logger.system_error("Page is on the initial blank page")
                if self.execution_timer.current_action_start is not None:
                    self.execution_timer.end_action()
                duration = (time.time() - start_time) if 'start_time' in locals() else None
                return _create_result(
                    False,
                    "Page is on initial blank page. Navigate to a page first.",
                    error="Page not navigated",
                    action_id=action_id,
                    duration=duration
                )
            
            # Register action in ledger
            action_id = self.action_ledger.register_action(
                goal=goal_description,
                action_id=action_id,
                metadata={"source": "act", "mode": "keyword"}
            )
            self.action_ledger.start_action(action_id)
            
            # Start action timer
            self.execution_timer.start_action(action_id, goal_description)
            
            # Log goal start
            self.logger.log_goal_start(goal_description)
            self.event_logger.goal_start(goal_description, action_id=action_id)
            
            # Add action to history
            self._add_to_command_history(goal_description)
            
            # Check for ref actions
            ref_result = self._handle_ref_commands(goal_description)
            if ref_result is not None:
                self.action_ledger.complete_action(action_id, success=ref_result)
                self.execution_timer.end_action()
                duration = time.time() - start_time
                return _create_result(
                    ref_result,
                    "Reference command executed successfully" if ref_result else "Reference command failed",
                    action_id=action_id,
                    duration=duration,
                    additional_metadata={"command_type": "ref"}
                )
            
            # Check for extract actions
            if goal_description.strip().lower().startswith("extract:"):
                extraction_prompt = goal_description.replace("extract:", "").strip()
                self.event_logger.extraction_start(extraction_prompt)
                
                # Perform extraction (now always returns ActionResult)
                extract_result = self.extract(
                    prompt=extraction_prompt,
                    output_format="json",
                    scope="viewport"
                )
                
                if extract_result.success:
                    # Extract the actual data from ActionResult for logging
                    extracted_data = extract_result.data
                    self.event_logger.extraction_success(extraction_prompt, result=extracted_data)
                    self.action_ledger.complete_action(action_id, success=True)
                    self.execution_timer.end_action()
                    duration = time.time() - start_time
                    return _create_result(
                        True,
                        "Extraction completed successfully",
                        action_id=action_id,
                        duration=duration,
                        additional_metadata={"command_type": "extract", "extraction_prompt": extraction_prompt},
                        data=extracted_data  # Store extracted data in ActionResult
                    )
                else:
                    self.event_logger.extraction_failure(extraction_prompt, error=extract_result.error or extract_result.message)
                    self.action_ledger.complete_action(action_id, success=False)
                    self.execution_timer.end_action()
                    duration = time.time() - start_time
                    return _create_result(
                        False,
                        f"Extraction failed: {extract_result.message}",
                        error=extract_result.error or extract_result.message,
                        action_id=action_id,
                        duration=duration,
                        additional_metadata={"command_type": "extract", "extraction_prompt": extraction_prompt}
                    )

            # Reset DOM signature for new action - don't check against previous action's signature
            # This ensures the first attempt of a new goal doesn't get blocked by DOM signature checks
            self.last_dom_signature = None

            # Only keyword goals are supported (click:, type:, etc.)
            keyword_command_result = self._execute_keyword_command(
                goal_description=goal_description,
                additional_context=additional_context,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
                action_id=action_id,
                start_time=start_time,
                base_knowledge=base_knowledge,
            )
            if keyword_command_result is not None:
                # keyword_action_result is now an ActionResult or bool (temporary compatibility)
                if isinstance(keyword_command_result, ActionResult):
                    return keyword_command_result
                else:
                    # Convert bool to ActionResult (for temporary compatibility)
                    duration = time.time() - start_time
                    return _create_result(
                        keyword_command_result,
                        "Action executed successfully" if keyword_command_result else "Action failed",
                        action_id=action_id,
                        duration=duration,
                        additional_metadata={"command_type": "keyword"}
                    )

            # If keyword execution can't handle it, fail with clear error message
            duration_ms = (time.time() - start_time) * 1000
            duration = time.time() - start_time
            self.logger.log_goal_failure(goal_description, "Could not parse command as keyword action. Use format: 'click: button', 'type: text', etc.", duration_ms)
            print(f"❌ Could not parse command: {goal_description}")
            print("   Hint: Use keyword format like 'click: button name', 'type: text in field', 'scroll: down', etc.")
            self.action_ledger.complete_action(action_id, success=False, error_message="Could not parse goal as keyword action. Must use keyword format (click:, type:, etc.)")
            self.execution_timer.end_action()
            return _create_result(
                False,
                "Could not parse command as keyword action. Use format: 'click: button', 'type: text', etc.",
                error="Could not parse command as keyword action. Must use keyword format (click:, type:, etc.)",
                action_id=action_id,
                duration=duration
            )
        finally:
            # End action timer if still active (safety net for any unhandled returns)
            if self.execution_timer.current_action_start is not None:
                self.execution_timer.end_action()
            
            # Mark act() as finished and flush any auto-on-load actions that arrived mid-act
            self._in_act = False
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

    def execute_task(
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
        max_clarification_rounds: int = 3,
        interaction_summary_limit_completion: Optional[int] = None,
        interaction_summary_limit_action: Optional[int] = None,
        user_question_callback: Optional[Callable[[str, dict], Optional[str]]] = None,
    ) -> AgentResult:
        """
        Execute a task autonomously (Step 1: Basic Reactive Agent).
        
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
            interaction_summary_limit_completion: Max interactions to feed into completion evaluation prompts.
                                                  None means include all interactions. Default: None.
            interaction_summary_limit_action: Max interactions to feed into next-action prompts.
                                              None means include all interactions. Default: None.
            user_question_callback: Optional callback for asking user questions when the agent gets stuck.
                                   Signature: (question: str, context: dict) -> Optional[str]
                                   Return user's answer or None to skip. Default: None.
            
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
            result = bot.execute_task("search for python tutorials")
            if result.success:
                print("Task completed!")
            
            # With extraction
            result = bot.execute_task(
                "navigate to amazon.com, search for 'laptop', extract the first product name and price"
            )
            if result.success:
                print(f"Product: {result.extracted_data.get('first product name')}")
                print(f"Price: {result.extracted_data.get('price')}")
            
            # Access extracted data
            for prompt, data in result.extracted_data.items():
                print(f"{prompt}: {data}")
        """
        # Parameter validation
        if not user_prompt or not user_prompt.strip():
            raise ValidationError(
                "user_prompt cannot be empty or whitespace",
                context=ErrorContext(
                    error_type="ValidationError",
                    message="user_prompt cannot be empty or whitespace",
                    action_data={"user_prompt": user_prompt}
                )
            )
        
        if max_iterations < 1:
            raise ValidationError(
                "max_iterations must be >= 1",
                context=ErrorContext(
                    error_type="ValidationError",
                    message="max_iterations must be >= 1",
                    action_data={"max_iterations": max_iterations}
                )
            )
        
        if max_clarification_rounds < 0:
            raise ValidationError(
                "max_clarification_rounds must be >= 0",
                context=ErrorContext(
                    error_type="ValidationError",
                    message="max_clarification_rounds must be >= 0",
                    action_data={"max_clarification_rounds": max_clarification_rounds}
                )
            )
        
        if clarification_callback is not None and not callable(clarification_callback):
            raise ValidationError(
                "clarification_callback must be callable",
                context=ErrorContext(
                    error_type="ValidationError",
                    message="clarification_callback must be callable",
                    action_data={"clarification_callback_type": type(clarification_callback).__name__}
                )
            )
        
        if base_knowledge is not None and not isinstance(base_knowledge, list):
            raise ValidationError(
                "base_knowledge must be a list",
                context=ErrorContext(
                    error_type="ValidationError",
                    message="base_knowledge must be a list",
                    action_data={"base_knowledge_type": type(base_knowledge).__name__}
                )
            )
        
        self._check_termination()
        
        if not self.started or not self.page:
            page_url = self.page.url if self.page else None
            raise BotNotStartedError(
                "Bot not started. Call bot.start() first.",
                context=ErrorContext(
                    error_type="BotNotStartedError",
                    message="Bot not started. Call bot.start() first.",
                    page_url=page_url,
                    action_data={"user_prompt": user_prompt}
                )
            )
        
        # Set agent mode flag
        self._agent_mode = True
        
        # Create middleware context
        context = ActionContext(
            action_type='execute_task',
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
                completion_mode=self.completion_mode,
                enable_sub_agents=self.enable_sub_agents,
                show_completion_reasoning_every_iteration=show_completion_reasoning_every_iteration,
                strict_mode=strict_mode,
                clarification_callback=clarification_callback,
                max_clarification_rounds=max_clarification_rounds,
                user_question_callback=user_question_callback,
                act_enable_target_context_guard=self.act_enable_target_context_guard,
                act_enable_modifier=self.act_enable_modifier,
                act_enable_additional_context=self.act_enable_additional_context,
                include_overlays_in_agent_context=self.include_overlays_in_agent_context,
                include_visible_text_in_agent_context=self.include_visible_text_in_agent_context,
                interaction_summary_limit_completion=interaction_summary_limit_completion,
                interaction_summary_limit_action=interaction_summary_limit_action,
            )
            controller.max_iterations = max_iterations
            
            # Register all pre-registered mini goals with the new controller
            for goal_data in self.mini_goals:
                controller.register_mini_goal(
                    trigger=goal_data["trigger"],
                    mode=goal_data["mode"],
                    handler=goal_data["handler"],
                    instruction_override=goal_data["instruction_override"]
                )
            
            # Store controller for pause/resume access
            # Why: Allows external code to pause/resume the agent during execution
            self.agent_controller = controller
            
            # Connect ActionExecutor pause callback to AgentController
            # Why: Enables pausing between individual action steps within plans,
            # providing granular control beyond just pausing between agent-determined actions
            if hasattr(self, 'action_executor') and self.action_executor:
                def pause_check():
                    """Callback to check pause state between action steps."""
                    if controller.is_paused():
                        controller._check_pause("action step")
                
                self.action_executor.set_pause_callback(pause_check)
            
            task_result = controller.run_execute_task(user_prompt)
            
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
            # Clear controller reference when task completes
            self.agent_controller = None
    
    def pause_agent(self, message: str = "Paused") -> None:
        """
        Pause the currently running agent between actions.
        
        When paused, the agent will wait before executing the next action, allowing for:
        - Manual inspection of page state after each action
        - Debugging specific action sequences
        - User intervention when needed
        - Verification of intermediate results
        
        The pause occurs between actions (not between iterations), providing fine-grained control.
        This means you can pause after a specific action completes and inspect the result before
        the agent decides what to do next.
        
        **Why pause between actions?**
        - Granular control: Inspect state after each individual action, not just at iteration boundaries
        - Better debugging: See immediate results of each action before the agent continues
        - User intervention: Handle edge cases that require human judgment
        - State inspection: Verify page state after each action completes
        
        **Thread-safe**: Can be called from any thread while the agent is running.
        
        Args:
            message: Optional message to display when paused (default: "Paused")
        
        Raises:
            RuntimeError: If no agent is currently running
        
        Example:
            >>> import threading
            >>> import time
            >>> 
            >>> bot.start()
            >>> bot.page.goto("https://example.com")
            >>> 
            >>> # Pause after 3 seconds in a background thread
            >>> def pause_after_delay():
            ...     time.sleep(3)
            ...     bot.pause_agent("Manual inspection needed")
            ...     time.sleep(10)  # Keep paused for 10 seconds
            ...     bot.resume_agent()
            >>> 
            >>> thread = threading.Thread(target=pause_after_delay)
            >>> thread.start()
            >>> 
            >>> result = bot.execute_task("search for jobs")
        """
        if not self.agent_controller:
            raise RuntimeError("No agent is currently running. Call execute_task() first.")
        self.agent_controller.pause(message)
    
    def resume_agent(self) -> None:
        """
        Resume the paused agent execution.
        
        Unblocks the agent to continue executing actions. If the agent is not paused,
        this method has no effect.
        
        **Thread-safe**: Can be called from any thread.
        
        Raises:
            RuntimeError: If no agent is currently running
        
        Example:
            >>> bot.pause_agent("Checking results")
            >>> # ... inspect page state ...
            >>> bot.resume_agent()  # Continue execution
        """
        if not self.agent_controller:
            raise RuntimeError("No agent is currently running. Call execute_task() first.")
        self.agent_controller.resume()
    
    def is_agent_paused(self) -> bool:
        """
        Check if the agent is currently paused.
        
        Returns:
            True if the agent is paused, False otherwise
        
        Raises:
            RuntimeError: If no agent is currently running
        
        Example:
            >>> if bot.is_agent_paused():
            ...     print("Agent is paused, waiting for resume...")
            ...     bot.resume_agent()
        """
        if not self.agent_controller:
            raise RuntimeError("No agent is currently running. Call execute_task() first.")
        return self.agent_controller.is_paused()

    def extract(
        self,
        prompt: str,
        output_format: str = "json",
        model_schema: Optional[Type[BaseModel]] = None,
        scope: str = "viewport",
        element_description: Optional[str] = None,
        max_retries: int = 2,
        confidence_threshold: float = 0.6,
    ) -> ActionResult:
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
            
        Returns:
            - If return_result=False (default): Raw extracted data (str, Dict, or BaseModel depending on output_format)
            - If return_result=True: ActionResult with extracted data in .data field
        
        Raises:
            BotTerminatedError: If bot has been terminated
            BotNotStartedError: If bot is not started
            ValidationError: If prompt is empty, invalid output_format/scope, or missing element_description
            ExtractionError: If extraction fails after retries
            
        Example:
            # Old way (still works)
            >>> bot.start()
            >>> bot.page.goto("https://example.com")
            >>> title = bot.extract("What is the page title?", output_format="text")
            >>> data = bot.extract("Extract product information", output_format="json")
            
            # New way (with structured result)
            >>> result = bot.extract("Get page title", output_format="text", return_result=True)
            >>> if result.success:
            ...     title = result.data  # The extracted text
            ...     print(f"Confidence: {result.confidence}")
        """
        # Parameter validation
        if not prompt or not prompt.strip():
            raise ValidationError(
                "prompt cannot be empty or whitespace",
                context=ErrorContext(
                    error_type="ValidationError",
                    message="prompt cannot be empty or whitespace",
                    action_data={"prompt": prompt}
                )
            )
        
        valid_output_formats = ["json", "text", "structured"]
        if output_format not in valid_output_formats:
            raise ValidationError(
                f"Invalid output_format: {output_format}. Must be one of {valid_output_formats}",
                context=ErrorContext(
                    error_type="ValidationError",
                    message=f"Invalid output_format: {output_format}",
                    action_data={"output_format": output_format, "valid_formats": valid_output_formats}
                )
            )
        
        valid_scopes = ["viewport", "full_page", "element"]
        if scope not in valid_scopes:
            raise ValidationError(
                f"Invalid scope: {scope}. Must be one of {valid_scopes}",
                context=ErrorContext(
                    error_type="ValidationError",
                    message=f"Invalid scope: {scope}",
                    action_data={"scope": scope, "valid_scopes": valid_scopes}
                )
            )
        
        if scope == "element" and not element_description:
            raise ValidationError(
                "element_description is required when scope='element'",
                context=ErrorContext(
                    error_type="ValidationError",
                    message="element_description is required when scope='element'",
                    action_data={"scope": scope, "element_description": element_description}
                )
            )
        
        if output_format == "structured" and model_schema is None:
            raise ValidationError(
                "model_schema is required when output_format='structured'",
                context=ErrorContext(
                    error_type="ValidationError",
                    message="model_schema is required when output_format='structured'",
                    action_data={"output_format": output_format, "model_schema": model_schema}
                )
            )
        
        if max_retries < 0:
            raise ValidationError(
                "max_retries must be >= 0",
                context=ErrorContext(
                    error_type="ValidationError",
                    message="max_retries must be >= 0",
                    action_data={"max_retries": max_retries}
                )
            )
        
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValidationError(
                "confidence_threshold must be between 0.0 and 1.0",
                context=ErrorContext(
                    error_type="ValidationError",
                    message="confidence_threshold must be between 0.0 and 1.0",
                    action_data={"confidence_threshold": confidence_threshold}
                )
            )
        
        from ai_utils import generate_text, generate_model
        # Import InteractionType locally to avoid linter false positive
        from session_tracker import InteractionType as IT
        
        self._check_termination()
        
        if not self.started or not self.page:
            page_url = self.page.url if self.page else None
            raise BotNotStartedError(
                "Bot not started. Call bot.start() first.",
                context=ErrorContext(
                    error_type="BotNotStartedError",
                    message="Bot not started. Call bot.start() first.",
                    page_url=page_url,
                    action_data={"extraction_prompt": prompt}
                )
            )
        
        # Helper function to create ActionResult for extraction
        def _create_extraction_result(data: Any, confidence: float = 0.9, 
                                     message: str = "Extraction completed successfully",
                                     error: Optional[str] = None,
                                     extraction_result: Optional[Any] = None) -> ActionResult:
            """Create ActionResult for extraction result"""
            metadata = {
                "prompt": prompt,
                "output_format": output_format,
                "scope": scope,
            }
            if extraction_result and hasattr(extraction_result, 'confidence'):
                metadata["extraction_confidence"] = extraction_result.confidence
            if extraction_result and hasattr(extraction_result, 'reasoning'):
                metadata["extraction_reasoning"] = extraction_result.reasoning
            
            return ActionResult(
                success=True,
                message=message,
                data=data,  # The actual extracted data
                confidence=confidence,
                metadata=metadata,
                error=error
            )
        
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
            raise ExtractionError(
                f"Failed to capture screenshot: {e}",
                context=ErrorContext(
                    error_type="ExtractionError",
                    message="Failed to capture screenshot for extraction",
                    page_url=self.page.url if self.page else None,
                    action_data={
                        "extraction_prompt": prompt,
                        "scope": scope,
                        "original_error": str(e)
                    }
                )
            ) from e
        
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
                    
                    return _create_extraction_result(
                        data=extracted_text,
                        confidence=0.9,
                        message="Text extraction completed successfully"
                    )
                
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
                    
                    return _create_extraction_result(
                        data=extracted_dict,
                        confidence=result.confidence,
                        message="JSON extraction completed successfully",
                        extraction_result=result
                    )
                
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
                    
                    return _create_extraction_result(
                        data=result,
                        confidence=0.95,  # Structured models are generally reliable
                        message="Structured extraction completed successfully"
                    )
                
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
                    
                    # Always return ActionResult on failure
                    return ActionResult(
                        success=False,
                        message=f"Extraction failed after {max_retries + 1} attempts: {error_msg}",
                        data=None,
                        confidence=0.0,
                        metadata={
                            "prompt": prompt,
                            "output_format": output_format,
                            "scope": scope,
                            "attempts": max_retries + 1,
                            "confidence_threshold": confidence_threshold
                        },
                        error=error_msg
                    )
        
        # This should not be reached, but handle it just in case
        if last_error:
            self.session_tracker.record_interaction(
                IT.EXTRACT,
                extraction_prompt=prompt,
                extracted_data=None,
                success=False,
                error_message=str(last_error)
            )
            
            # Always return ActionResult on failure
            return ActionResult(
                success=False,
                message=f"Extraction failed: {last_error}",
                data=None,
                confidence=0.0,
                metadata={
                    "prompt": prompt,
                    "output_format": output_format,
                    "scope": scope
                },
                error=str(last_error)
            )
    
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
        additional_context: str = "",
        target_context_guard: Optional[str] = None,
        skip_post_guard_refinement: bool = True,
        confirm_before_interaction: bool = False,
        action_id: Optional[str] = None,
        max_attempts: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> bool:
        """
        Register multiple commands for later reference and execution.
        
        Args:
            prompts: List of command prompts to register (must use keyword format: "click:", "type:", etc.)
            ref_id: Unique identifier for referencing these commands
            all_must_be_true: Require every command to succeed when evaluating this ref
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
        
        # Register the ref in action ledger
        action_id = self.action_ledger.register_action(
            goal=f"register_prompts: {ref_id}",
            action_id=action_id,
            metadata={"source": "register_prompts", "ref_id": ref_id, "prompt_count": len(prompts)}
        )
        
        try:
            if not prompts:
                self.logger.log_error("No prompts provided for register_prompts", "register_prompts() called with empty prompts")
                print("❌ No prompts provided for register_prompts")
                return False
            
            # Store the goals for later reference
            self.command_refs[ref_id] = {
                "prompts": prompts.copy(),
                "all_must_be_true": bool(all_must_be_true),
                "action_id": action_id,  # Store the original action ID
                "additional_context": additional_context or "",
                "target_context_guard": target_context_guard,
                "skip_post_guard_refinement": bool(skip_post_guard_refinement),
                "confirm_before_interaction": bool(confirm_before_interaction),
                "max_attempts": max_attempts,
                "max_retries": max_retries,
            }
            mode = "ALL" if all_must_be_true else "ANY"
            extra_parts = []
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

    def register_mini_goal(
        self,
        trigger: 'MiniGoalTrigger',
        mode: 'MiniGoalMode',
        handler: Optional[Callable[['MiniGoalScriptContext'], None]] = None,
        instruction_override: Optional[str] = None
    ) -> None:
        """
        Register a mini goal trigger and handler.
        
        Mini goals allow the agent to temporarily switch focus to a sub-objective
        when specific conditions (observations or actions) are met.
        
        Args:
            trigger: The condition that activates this mini-goal
            mode: Either AUTONOMY (agent solves it) or SCRIPTED (handler executes)
            handler: For SCRIPTED mode, the Python function to execute
            instruction_override: Custom instruction for the agent in AUTONOMY mode
        """
        goal_data = {
            "trigger": trigger,
            "mode": mode,
            "handler": handler,
            "instruction_override": instruction_override
        }
        self.mini_goals.append(goal_data)

        # If an agent is already running, register it there too
        if hasattr(self, 'agent_controller') and self.agent_controller:
            self.agent_controller.register_mini_goal(
                trigger=trigger,
                mode=mode,
                handler=handler,
                instruction_override=instruction_override
            )

    def _get_semantic_target(self, description: str) -> Optional[SemanticTarget]:
        """Build semantic target for better element matching."""
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

    def _collect_overlay_data(
        self,
        goal_description: str,
        page_info: PageInfo,
    ) -> tuple[List[Dict[str, Any]], Optional[bytes], Optional[bytes]]:
        """Collect overlay metadata and screenshots, with caching and dedup filtering."""
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

        # Create overlays (visibility controlled by config)
        show_overlays = self.config.elements.show_overlays
        element_data = self.overlay_manager.create_numbered_overlays(
            page_info,
            mode=self.overlay_mode,
            visible=show_overlays,
        ) or []

        # Focus system removed - no longer filtering by focus context
        self._pre_dedup_element_data = element_data.copy() if element_data else []

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
        action_id: Optional[str],
        start_time: float,
        base_knowledge: Optional[List[str]] = None,
    ) -> Optional[bool]:
        """Attempt to execute the action using keyword-based execution. Returns None to fall back."""
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
                base_knowledge=base_knowledge,
            )
        elif keyword == "type":
            result = self._keyword_type(
                goal_description=goal_description,
                payload=payload,
                helper=helper,
                additional_context=additional_context,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
                base_knowledge=base_knowledge,
            )
        elif keyword == "select":
            result = self._keyword_select(
                goal_description=goal_description,
                payload=payload,
                helper=helper,
                additional_context=additional_context,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
                base_knowledge=base_knowledge,
            )
        elif keyword == "upload":
            result = self._keyword_upload(
                goal_description=goal_description,
                payload=payload,
                helper=helper,
                additional_context=additional_context,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
                base_knowledge=base_knowledge,
            )
        elif keyword == "datetime":
            result = self._keyword_datetime(
                goal_description=goal_description,
                payload=payload,
                helper=helper,
                additional_context=additional_context,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
                base_knowledge=base_knowledge,
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
        elif keyword == "defer":
            result = self._keyword_defer(
                goal_description=goal_description,
                payload=payload,
                target_context_guard=target_context_guard,
                confirm_before_interaction=confirm_before_interaction,
            )
        else:
            # Unsupported keyword – fall back
            return None

        if result is None:
            # Keyword action could not confidently execute – allow normal flow
            return None

        duration_ms = (time.time() - start_time) * 1000
        if result:
            self.logger.log_goal_success(goal_description, duration_ms)
            try:
                self.event_logger.command_execution_complete(goal_description=goal_description, success=True)
            except Exception:
                pass
            self.action_ledger.complete_action(action_id, success=True)
        else:
            self.logger.log_goal_failure(goal_description, "Keyword command execution failed", duration_ms)
            self.action_ledger.complete_action(
                action_id,
                success=False,
                error_message="Keyword action execution failed",
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
            # Only show in debug mode
            if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
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
        base_knowledge: Optional[List[str]] = None,
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
                base_knowledge=base_knowledge,
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
        base_knowledge: Optional[List[str]] = None,
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
            base_knowledge=base_knowledge,
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
        base_knowledge: Optional[List[str]] = None,
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
            base_knowledge=base_knowledge,
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
        base_knowledge: Optional[List[str]] = None,
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
            base_knowledge=base_knowledge,
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
        base_knowledge: Optional[List[str]] = None,
    ) -> Optional[bool]:
        intent = parse_action_intent(goal_description)
        if not intent or intent.action != "upload":
            return None

        # Use target_text, helper, helper_text, or value (in that order) as target hint
        # When format is "upload: <target>" (no file), value contains the target
        target_hint_raw = intent.target_text or helper or intent.helper_text or intent.value
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
            action_kwargs={"upload_file_path": intent.value or None},
            target_context_guard=target_context_guard,
            confirm_before_interaction=confirm_before_interaction,
            base_knowledge=base_knowledge,
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
        base_knowledge: Optional[List[str]] = None,
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
            base_knowledge=base_knowledge,
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

    def _keyword_defer(
        self,
        *,
        goal_description: str,
        payload: str,
        target_context_guard: Optional[str],
        confirm_before_interaction: bool,
    ) -> Optional[bool]:
        """
        Handle defer command to pause agent execution.
        
        Supports:
        - "defer" → pause indefinitely until user resumes
        - "defer: 10" → pause for 10 seconds, then auto-resume
        - "defer: message" → pause indefinitely with custom message
        
        This method blocks execution until resume is called or auto-resume timer expires.
        """
        # Determine message and whether to auto-resume
        message = payload.strip() if payload else "Paused - waiting for user"
        auto_resume_seconds = None
        
        # Check if payload is a number (seconds to wait)
        try:
            auto_resume_seconds = float(payload.strip())
            if auto_resume_seconds > 0:
                message = f"Paused for {auto_resume_seconds} seconds"
            else:
                auto_resume_seconds = None
        except (ValueError, AttributeError):
            # Not a number, treat as message
            pass
        
        # If no agent is running, just log and return success
        if not self.agent_controller:
            print(f"ℹ️  Defer command received: {message}")
            print("   (No agent is currently running, so nothing to pause)")
            return True
        
        # Record defer action in interaction history (before pausing)
        # Extract reasoning from goal_description if available
        reasoning = goal_description if goal_description != f"defer: {payload}" else message
        
        # Record the defer action start
        if hasattr(self, 'session_tracker') and self.session_tracker:
            self.session_tracker.record_interaction(
                interaction_type=InteractionType.DEFER,
                reasoning=reasoning,
                text_input=message,  # Store the defer message
                success=True
            )
        
        # Pause the agent and block execution
        try:
            self.pause_agent(message)
            
            # Track how the defer was resumed (for recording in main thread)
            resume_reason = None
            
            # If auto-resume is requested, set up a timer
            if auto_resume_seconds is not None:
                def auto_resume():
                    nonlocal resume_reason
                    try:
                        if self.agent_controller and self.agent_controller.is_paused():
                            self.resume_agent()
                            print(f"▶️  Auto-resumed after {auto_resume_seconds} seconds")
                            # Set resume reason - will be recorded in main thread
                            resume_reason = f"Defer auto-resumed after {auto_resume_seconds} seconds"
                    except Exception as e:
                        print(f"⚠️  Error during auto-resume: {e}")
                
                timer = threading.Timer(auto_resume_seconds, auto_resume)
                timer.daemon = True
                timer.start()
                print(f"⏸️  Paused: {message} (will auto-resume in {auto_resume_seconds} seconds)")
                
                # For timed pauses, wait on the pause event (timer will resume)
                if self.agent_controller:
                    self.agent_controller._pause_event.wait()
            else:
                # For indefinite pauses, wait for user to press Enter
                print(f"⏸️  Paused: {message}")
                print("   (Press Enter or call bot.resume_agent() to continue)")
                
                # Start a thread to listen for Enter key press
                def wait_for_enter():
                    nonlocal resume_reason
                    try:
                        input()  # Wait for Enter key
                        # Resume the agent when Enter is pressed
                        if self.agent_controller and self.agent_controller.is_paused():
                            self.resume_agent()
                            print("▶️  Resumed by user")
                            # Set resume reason - will be recorded in main thread
                            resume_reason = "Defer resumed by user - control returned to agent"
                    except (EOFError, KeyboardInterrupt):
                        # If input stream is unavailable, just resume
                        if self.agent_controller and self.agent_controller.is_paused():
                            self.resume_agent()
                            print("▶️  Resumed (input unavailable)")
                            # Set resume reason - will be recorded in main thread
                            resume_reason = "Defer resumed (input unavailable)"
                
                input_thread = threading.Thread(target=wait_for_enter, daemon=True)
                input_thread.start()
                
                # Block execution by waiting on the pause event
                # This will be released when user presses Enter (via input_thread) or resume_agent() is called
                if self.agent_controller:
                    self.agent_controller._pause_event.wait()
            
            # Record resume in main thread (after pause event is released)
            # This avoids greenlet errors from calling Playwright APIs from background threads
            if resume_reason and hasattr(self, 'session_tracker') and self.session_tracker:
                try:
                    self.session_tracker.record_interaction(
                        interaction_type=InteractionType.DEFER,
                        reasoning=resume_reason,
                        text_input="resumed",
                        success=True
                    )
                except Exception as e:
                    # If recording fails, log but don't break the defer flow
                    print(f"⚠️  Could not record defer resume: {e}")
            
            return True
        except Exception as e:
            print(f"⚠️  Error pausing agent: {e}")
            return False

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

 
    # -------------------- Public memory helpers (general) --------------------
    # Memory store methods removed - deduplication now handled by focus manager

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
                ref_additional_context = ref_entry.get("additional_context", "")
                ref_target_context_guard = ref_entry.get("target_context_guard")
                ref_skip_post_guard_refinement = ref_entry.get("skip_post_guard_refinement", True)
                ref_confirm_before_interaction = ref_entry.get("confirm_before_interaction", False)
                ref_max_attempts = ref_entry.get("max_attempts")
                stored_action_id = ref_entry.get("action_id")  # Get the original action ID

                if not stored_prompts:
                    print(f"⚠️ Ref ID '{ref_id}' has no stored commands")
                    return True

                summary_mode = 'ALL' if all_must_be_true else 'ANY'
                print(f"🔄 Executing {len(stored_prompts)} stored commands for ref ID: {ref_id} (mode={summary_mode})")
                results: List[bool] = []

                # Use the stored command ID as the parent, fallback to current if not available
                ref_action_id = stored_action_id or self.action_ledger.get_current_action_id()
                
                for i, prompt in enumerate(stored_prompts, 1):
                    print(f"▶️ Executing stored command {i}/{len(stored_prompts)}: {prompt}")
                    
                    # Generate a child action ID
                    child_action_id = f"{ref_action_id}_action{i}" if ref_action_id else None
                    
                    success = bool(
                        self.act(
                            prompt,
                            additional_context=ref_additional_context,
                            target_context_guard=ref_target_context_guard,
                            skip_post_guard_refinement=ref_skip_post_guard_refinement,
                            confirm_before_interaction=ref_confirm_before_interaction,
                            action_id=child_action_id,  # Pass child ID
                            max_attempts=ref_max_attempts,  # Pass custom max_attempts
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
        # Focus system removed - returns all elements
        return element_data

    def queue_action(self, action: str, action_id: Optional[str] = None, 
                    priority: int = 0, metadata: Dict[str, Any] = None) -> None:
        """
        Queue an action for later execution.
        
        Args:
            action: The action to execute (e.g., "click: button")
            action_id: Optional action ID for tracking
            priority: Priority level (higher = executed first)
            metadata: Optional metadata dict
        """
        self.action_queue.enqueue(action, action_id, priority, metadata)
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
                    action_result = self.act(
                        queued_action.action,
                        action_id=queued_action.action_id
                    )
                    success = action_result.success
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

    # ==================== Convenience Methods ====================

    def get_url(self) -> str:
        """
        Get the current page URL.
        
        Returns:
            str: Current page URL, or empty string if page not available
            
        Raises:
            BotTerminatedError: If bot has been terminated
            BotNotStartedError: If bot is not started
        """
        self._check_termination()
        if not self.started or not self.page:
            raise BotNotStartedError(
                "Bot not started. Call bot.start() first.",
                context=ErrorContext(
                    error_type="BotNotStartedError",
                    message="Bot not started. Call bot.start() first."
                )
            )
        return self.page.url

    def get_title(self) -> str:
        """
        Get the current page title.
        
        Returns:
            str: Current page title, or empty string if page not available
            
        Raises:
            BotTerminatedError: If bot has been terminated
            BotNotStartedError: If bot is not started
        """
        self._check_termination()
        if not self.started or not self.page:
            raise BotNotStartedError(
                "Bot not started. Call bot.start() first.",
                context=ErrorContext(
                    error_type="BotNotStartedError",
                    message="Bot not started. Call bot.start() first."
                )
            )
        try:
            return self.page.title()
        except Exception:
            return ""

    def wait_for_load(self, timeout: int = 30000, state: str = "networkidle") -> None:
        """
        Wait for the page to finish loading.
        
        Args:
            timeout: Maximum time to wait in milliseconds (default: 30000)
            state: Load state to wait for: "load", "domcontentloaded", "networkidle" (default: "networkidle")
            
        Raises:
            BotTerminatedError: If bot has been terminated
            BotNotStartedError: If bot is not started
            ValidationError: If invalid state is provided
        """
        self._check_termination()
        if not self.started or not self.page:
            raise BotNotStartedError(
                "Bot not started. Call bot.start() first.",
                context=ErrorContext(
                    error_type="BotNotStartedError",
                    message="Bot not started. Call bot.start() first."
                )
            )
        
        valid_states = ["load", "domcontentloaded", "networkidle"]
        if state not in valid_states:
            raise ValidationError(
                f"Invalid state: {state}. Must be one of {valid_states}",
                context=ErrorContext(
                    error_type="ValidationError",
                    message=f"Invalid load state: {state}",
                    action_data={"state": state, "valid_states": valid_states, "timeout": timeout}
                )
            )
        
        try:
            self.page.wait_for_load_state(state, timeout=timeout)
        except Exception as e:
            # Don't raise, just log - sometimes pages don't fully load
            try:
                self.event_logger.system_warning(f"Page load wait timeout or error: {e}")
            except Exception:
                pass

    def screenshot(self, path: Optional[str] = None, full_page: bool = False) -> bytes:
        """
        Take a screenshot of the current page.
        
        Args:
            path: Optional file path to save the screenshot. If None, returns bytes.
            full_page: If True, capture full page. If False, capture viewport only (default: False)
            
        Returns:
            bytes: Screenshot image data (if path is None)
            
        Raises:
            BotTerminatedError: If bot has been terminated
            BotNotStartedError: If bot is not started
            ActionFailedError: If screenshot capture fails
        """
        self._check_termination()
        if not self.started or not self.page:
            raise BotNotStartedError(
                "Bot not started. Call bot.start() first.",
                context=ErrorContext(
                    error_type="BotNotStartedError",
                    message="Bot not started. Call bot.start() first."
                )
            )
        
        try:
            return self.page.screenshot(path=path, full_page=full_page, type="png")
        except Exception as e:
            raise ActionFailedError(
                f"Failed to capture screenshot: {e}",
                context=ErrorContext(
                    error_type="ActionFailedError",
                    message="Failed to capture screenshot",
                    page_url=self.page.url if self.page else None,
                    action_data={
                        "path": path,
                        "full_page": full_page,
                        "original_error": str(e)
                    }
                )
            ) from e

    # ==================== Property Accessors ====================

    @property
    def is_started(self) -> bool:
        """
        Check if the bot has been started.
        
        Returns:
            bool: True if bot is started, False otherwise
        """
        return self.started

    @property
    def is_terminated(self) -> bool:
        """
        Check if the bot has been terminated.
        
        Returns:
            bool: True if bot is terminated, False otherwise
        """
        return self.terminated

    @property
    def current_url(self) -> str:
        """
        Get the current page URL.
        
        Returns:
            str: Current page URL, or empty string if not available
            
        Raises:
            RuntimeError: If bot is not started
        """
        return self.get_url()

    @property
    def session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - interaction_count: Number of interactions recorded
                - url_history: Number of URLs visited
                - commands_executed: Number of commands in history
                - current_url: Current page URL
                
        Raises:
            BotTerminatedError: If bot has been terminated
            BotNotStartedError: If bot is not started
        """
        self._check_termination()
        if not self.started:
            raise BotNotStartedError(
                "Bot not started. Call bot.start() first.",
                context=ErrorContext(
                    error_type="BotNotStartedError",
                    message="Bot not started. Call bot.start() first."
                )
            )
        
        stats = {
            "interaction_count": len(self.session_tracker.interaction_history) if hasattr(self, 'session_tracker') and self.session_tracker else 0,
            "url_history": len(self.session_tracker.url_history) if hasattr(self, 'session_tracker') and self.session_tracker else 0,
            "commands_executed": len(self.command_history) if hasattr(self, 'command_history') else 0,
            "current_url": self.page.url if self.page else "",
        }
        return stats
    
