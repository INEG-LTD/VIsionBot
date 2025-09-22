"""
Vision Bot - Clean modular version.
"""
from __future__ import annotations

import hashlib
import os
import time
import traceback
import uuid
import re
from typing import Any, Optional, List, Dict

from playwright.sync_api import Browser, Page, Playwright

from goals.base import GoalResult
from models import VisionPlan, PageElements
from models.core_models import ActionStep, ActionType
from element_detection import ElementDetector
from element_detection.overlay_manager import OverlayManager
from action_executor import ActionExecutor
from models.core_models import PageInfo
from utils import PageUtils
from goals import GoalMonitor, ClickGoal, GoalStatus, FormFillGoal, NavigationGoal, IfGoal, WhileGoal, PressGoal, ScrollGoal, Condition, BaseGoal, BackGoal, ForwardGoal
# from goals.condition_engine import create_predicate_condition as create_predicate
from planner.plan_generator import PlanGenerator
from utils.intent_parsers import (
    parse_while_statement,
    extract_click_target,
    extract_press_target,
    extract_navigation_intent,
    parse_structured_if,
    parse_structured_while,
    parse_keyword_command,
    parse_focus_command,
    parse_undo_command,
)
from goals.history_utils import (
    resolve_back_target,
    resolve_forward_target,
)
from focus_manager_ai_first import AIFirstFocusManager as FocusManager
from utils.bot_logger import get_logger, LogLevel, LogCategory
from utils.semantic_targets import SemanticTarget, build_semantic_target


class BrowserVisionBot:
    """Modular vision-based web automation bot"""

    def __init__(
        self,
        page: Page = None,
        model_name: str = "gemini-2.0-flash",
        max_attempts: int = 10,
        max_detailed_elements: int = 400,
        include_detailed_elements: bool = True,
        two_pass_planning: bool = True,
        max_coordinate_overlays: int = 600,
    ):
        self.page = page
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.started = False
        # Controls for how much element detail to include in planning prompts
        self.max_detailed_elements = max_detailed_elements
        self.include_detailed_elements = include_detailed_elements
        # Two-pass planning: pre-select relevant overlays to shrink prompt
        self.two_pass_planning = two_pass_planning
        # Hard cap for how many overlay coordinates to include in prompts
        self.max_coordinate_overlays = max_coordinate_overlays
        
        # Auto-run actions on page load (opt-in)
        self._auto_on_load_enabled: bool = False
        self._auto_on_load_actions: List[str] = []
        self._auto_on_load_run_once_per_url: bool = True
        self._auto_on_load_urls_handled: set[str] = set()
        self._auto_on_load_handler = None  # keep reference for potential removal
        self._auto_on_load_running: bool = False
        self._auto_on_load_event_name: str | None = None
        self._in_act: bool = False
        # Queue auto-on-load actions when a page load happens during an active act()
        self._pending_auto_on_load: bool = False
        self._pending_auto_on_load_url: Optional[str] = None
        
        # Dedup policy: detect from prompt by default ('auto').
        # 'off' = never dedup, 'on' = always dedup, 'auto' = only when prompt asks.
        self.dedup_mode: str = "auto"
        
        # Command history for "do that again" functionality
        self.command_history: List[str] = []
        self.max_command_history: int = 10  # Keep last 10 commands
        
        # Multi-command reference storage
        self.command_refs: Dict[str, Dict[str, Any]] = {}  # refID -> metadata about stored prompts
        
        # Initialize logger
        self.logger = get_logger()

        # Interpretation / semantic resolution helpers
        self.default_interpretation_mode: str = "literal"
        self._interpretation_mode_stack: List[str] = []
        self._semantic_target_cache: Dict[str, Optional[SemanticTarget]] = {}
        
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
        
        return p, browser, page
    
    def start(self) -> None:
        """Start the bot"""
        playwright, browser, page = self.init_browser()
        self.playwright = playwright
        self.browser = browser
        self.page = page
        # Keep Playwright waits snappy while avoiding long stalls
        try:
            self.page.set_default_timeout(2000)
        except Exception:
            pass
        
        # State tracking
        self.current_attempt = 0
        self.last_screenshot_hash = None
        self.last_dom_signature = None
        
        # Initialize components
        self.goal_monitor = GoalMonitor(page)
        self.overlay_manager = OverlayManager(page)
        self.element_detector = ElementDetector(model_name=self.model_name)
        self.page_utils = PageUtils(page)
        # AI-first focus manager for element focusing
        self.focus_manager = FocusManager(page, self.page_utils)
        self.action_executor = ActionExecutor(page, self.goal_monitor, self.page_utils, self.focus_manager)
        # Plan generator for AI planning prompts
        self.plan_generator = PlanGenerator(
            include_detailed_elements=self.include_detailed_elements,
            max_detailed_elements=self.max_detailed_elements,
        )
        # Attach shared memory to the monitor so clicks are remembered generically
        try:
            self.goal_monitor.set_memory_store(self.memory_store)
        except Exception:
            pass
        
        # Auto-switch to new tabs/windows when they open (e.g., target=_blank)
        try:
            self._attach_new_page_listener()
        except Exception as e:
            print(f"⚠️ Failed to attach new page listener: {e}")

        self.started = True

        # If auto-on-load was enabled before start, attach handler now
        try:
            if self._auto_on_load_enabled:
                self._attach_page_load_handler()
        except Exception:
            pass

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
                print("🆕 New page/tab detected by context listener → switching…")
                self.switch_to_page(new_page)
            except Exception as e:
                print(f"⚠️ Error handling new page event: {e}")

        try:
            ctx.on("page", _on_new_page)
        except Exception as e:
            print(f"⚠️ Could not register context 'page' listener: {e}")

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

            self.page = new_page
            # Update core components
            try:
                if self.goal_monitor:
                    self.goal_monitor.switch_to_page(new_page)
            except Exception:
                pass
            try:
                if self.action_executor:
                    self.action_executor.set_page(new_page)
            except Exception:
                pass
            try:
                if self.page_utils:
                    self.page_utils.page = new_page
            except Exception:
                pass
            try:
                if self.overlay_manager:
                    self.overlay_manager.page = new_page
            except Exception:
                pass

            print(f"🔀 Switched active context to new tab: {getattr(new_page, 'url', '')}")
            # Re-attach auto-on-load handler for the new page if feature is enabled
            try:
                if self._auto_on_load_enabled:
                    self._attach_page_load_handler()
                    # Also run immediately for the new page so it happens before next act()
                    self._run_auto_actions_for_current_page()
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ Failed to switch to new page: {e}")

    # ---------- Auto actions on page load ----------
    def on_new_page_load(self, actions_to_take: List[str], run_once_per_url: bool = True) -> None:
        """Register prompts to run via act() after each page load.

        Typical usage: on_new_page_load([
            "if a cookie banner is visible click the accept button",
            "close any newsletter modal if present",
        ])
        """
        self._auto_on_load_actions = [a for a in (actions_to_take or []) if isinstance(a, str) and a.strip()]
        self._auto_on_load_run_once_per_url = bool(run_once_per_url)
        self._auto_on_load_enabled = True
        self._auto_on_load_urls_handled.clear()
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
            for prompt in self._auto_on_load_actions:
                try:
                    print(f"⚡ Auto-on-load: act('{prompt}')")
                    # Snapshot current goals and user prompt so auto-action does not disrupt ongoing task
                    saved_goal = getattr(self.goal_monitor, 'active_goal', None) if hasattr(self, 'goal_monitor') else None
                    saved_user_prompt = getattr(self.goal_monitor, 'user_prompt', "") if hasattr(self, 'goal_monitor') else ""
                    try:
                        # If we're inside act(), we still call act() but our snapshot/restore prevents disruption
                        # and _auto_on_load_running avoids re-entrancy loops
                        self.act(prompt)
                    finally:
                        # Restore previous goals/user prompt if they existed prior
                        try:
                            if hasattr(self, 'goal_monitor'):
                                self.goal_monitor.clear_all_goals()
                                for g in saved_goal:
                                    try:
                                        self.goal_monitor.add_goal(g)
                                    except Exception:
                                        pass
                                try:
                                    self.goal_monitor.set_user_prompt(saved_user_prompt)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception as e:
                    print(f"⚠️ Auto-on-load action failed: {e}")
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


    def _try_simple_goal_bypass(self) -> Optional[bool]:
        """Fast path: if all active goals are simple (Press/Scroll), execute directly without LLM.

        Returns True/False if executed, or None to fall back to normal planning.
        """
        try:
            if not self.goal_monitor or not self.goal_monitor.active_goal:
                return None
            from goals import PressGoal, ScrollGoal
            goal = self.goal_monitor.active_goal
            # Only bypass for Press/Scroll; Back/Forward use history + AI selection
            if not isinstance(goal, (PressGoal, ScrollGoal)):
                return None

            # Build minimal plan
            steps: List[ActionStep] = []
            if isinstance(goal, PressGoal):
                if getattr(goal, 'target_keys', None):
                    steps.append(ActionStep(action=ActionType.PRESS, keys_to_press=goal.target_keys))
            elif isinstance(goal, ScrollGoal):
                # Heuristic direction from request; executor will refine using ScrollGoal
                req = (getattr(goal, 'user_request', '') or '').lower()
                direction = 'down'
                if 'up' in req:
                    direction = 'up'
                elif 'left' in req:
                    direction = 'left'
                elif 'right' in req:
                    direction = 'right'
                steps.append(ActionStep(action=ActionType.SCROLL, scroll_direction=direction))
                # Intentionally skip Back/Forward here to avoid duplicate/conflicting navigation

            # Always add STOP to return quickly
            steps.append(ActionStep(action=ActionType.STOP))

            fast_plan = VisionPlan(
                detected_elements=PageElements(elements=[]),
                action_steps=steps,
                reasoning="Simple-goal bypass: executing press/scroll without planning",
                confidence=0.99,
            )

            page_info = self.page_utils.get_page_info()
            ok = self.action_executor.execute_plan(fast_plan, page_info)
            if not ok:
                return False

            # Evaluate goals after execution
            goal_result = self.goal_monitor.evaluate_goal()
            if goal_result.status == GoalStatus.ACHIEVED:
                self._print_goal_summary()
                return True

            # If not achieved, allow normal planning to proceed
            return None
        except Exception as e:
            print(f"⚠️ Simple-goal bypass failed: {e}")
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
    ) -> bool:
        """Main method to achieve a goal using vision-based automation"""
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
                print("❌ Bot not started")
                return False
            
            if self.page.url.startswith("about:blank"):
                self.logger.log_error("Page is on initial blank page", "act() called before navigation")
                print("❌ Page is on the initial blank page")
                return False
            
            # Log goal start
            self.logger.log_goal_start(goal_description)
            print(f"🎯 Starting goal: {goal_description}")
            
            # Add command to history
            self._add_to_command_history(goal_description)
            
            # Check for focus commands first
            focus_result = self._handle_focus_commands(goal_description, self.overlay_manager)
            if focus_result is not None:
                return focus_result
            
            # Check for dedup commands
            dedup_result = self._handle_dedup_commands(goal_description)
            if dedup_result is not None:
                return dedup_result
            
            # Check for ref commands
            ref_result = self._handle_ref_commands(goal_description)
            if ref_result is not None:
                return ref_result

            # Clear any existing goals before starting a new goal
            if self.goal_monitor.active_goal:
                print(f"🧹 Clearing {self.goal_monitor.active_goal} previous goals")
                self.goal_monitor.clear_all_goals()
            
            # Reset goal monitor state for fresh start
            self.goal_monitor.reset_retry_request()
            
            self.goal_monitor.set_user_prompt(goal_description)
            
            # Set up smart goal monitoring if enabled
            goal, transformed_goal_description = self._create_goal_from_description(goal_description)

            if isinstance(goal, WhileGoal):
                return self._execute_while_loop(goal, start_time)

            if goal:
                goal_description = transformed_goal_description
                self.goal_monitor.add_goal(goal)
            else:
                print("ℹ️ No smart goal setup")
                return True
            
            # If conditional evaluation resulted in a deliberate no-op (no fail action and condition false)
            if not goal_description.strip():
                duration_ms = (time.time() - start_time) * 1000
                self.logger.log(LogLevel.INFO, LogCategory.GOAL, "No actionable goal after condition evaluation (no-op)", duration_ms=duration_ms)
                print("ℹ️ No actionable goal after condition evaluation (no-op). Skipping.")
                return True
            
            print(f"🔍 Smart goals setup: {goal}\n")

            # Simple goal bypass (no LLM): handle press/scroll-only flows directly
            simple_result = self._try_simple_goal_bypass()
            if simple_result is not None:
                return simple_result

            for attempt in range(self.max_attempts):
                self.current_attempt = attempt + 1
                print(f"\n--- Attempt {self.current_attempt}/{self.max_attempts} ---")
                
                # Show retry context at the start of each new attempt (but don't reset yet)
                if attempt > 0:  # Don't reset on first attempt
                    retry_goal = self.goal_monitor.check_for_retry_request()
                    if retry_goal:
                        print(f"🔄 Starting attempt {self.current_attempt} with retry state from previous attempt")
                        print(f"   🔄 {retry_goal}: Retry attempt {retry_goal.retry_count}/{retry_goal.max_retries}")
                        # Don't reset retry state here - let it persist until after plan generation
                    else:
                        print("ℹ️ No retry state from previous attempt")
                
                # Check if goal is already achieved
                goal_result = self.goal_monitor.evaluate_goal()
                if goal_result.status == GoalStatus.ACHIEVED:
                    duration_ms = (time.time() - start_time) * 1000
                    self.logger.log_goal_success(goal_description, duration_ms)
                    print("✅ Smart goal achieved!")
                    self._print_goal_summary()
                    return True
                
                # Get current page state
                page_info = self.page_utils.get_page_info()
                
                # Fast DOM signature: avoid screenshot if the page hasn't changed
                try:
                    sig_src = self.page.evaluate(
                        """
                        () => {
                            const url = location.href || '';
                            const y = (window.scrollY||0) + ':' + (window.scrollX||0);
                            const cnt = document.body ? document.body.getElementsByTagName('*').length : 0;
                            const txt = (document.body && document.body.innerText) ? document.body.innerText.slice(0, 4000) : '';
                            return `${url}|${y}|${cnt}|${txt}`;
                        }
                        """
                    )
                except Exception:
                    sig_src = f"{self.page.url}|{page_info.scroll_y}|{page_info.scroll_x}"
                sig_hash = hashlib.md5(sig_src.encode("utf-8")).hexdigest()
                
                # Skip if DOM signature hasn't changed and no retry requested
                retry_goal = self.goal_monitor.check_for_retry_request()
                if sig_hash == self.last_dom_signature and not retry_goal:
                    print("⚠️ Same DOM signature as last attempt, scrolling to break loop")
                    self.page_utils.scroll_page()
                    continue
                elif sig_hash == self.last_dom_signature and retry_goal:
                    print("🔄 Same DOM but retry requested - proceeding with retry attempt")
                self.last_dom_signature = sig_hash
                
                # Decide whether detection will be needed to avoid an extra screenshot
                needs_detection = self.goal_monitor.active_goal.needs_detection if self.goal_monitor.active_goal else True
                # Only take a screenshot now when not running detection
                screenshot = None
                if not needs_detection:
                    # Lower quality for model-bound screenshot to reduce payload
                    screenshot = self.page.screenshot(type="jpeg", quality=35, full_page=False)

                if goal.needs_plan:
                # Generate plan using vision model (conditional goals are already resolved to their sub-goals)
                    plan = self._generate_plan(goal_description, additional_context, screenshot, page_info)
                else:
                    plan = None
                    
                    goal_eval_result = self.goal_monitor.evaluate_goal()
                    print(f"🔍 Smart goal {goal_description} for goal {self.goal_monitor.active_goal.__class__.__name__} evaluation result: {goal_eval_result}")
                    if goal_eval_result.status == GoalStatus.ACHIEVED:
                        duration_ms = (time.time() - start_time) * 1000
                        self.logger.log_goal_success(goal_description, duration_ms)
                        print(f"✅ Smart goal {goal_description} achieved during plan execution!")
                        self._print_goal_summary()
                        return True
                    else:
                        print(f"⏳ Smart goal {goal_description} pending further evaluation")

                if not plan or not plan.action_steps:
                    print(f"❌ No valid plan generated for goal: {goal_description}")
                    continue
                
                # Reset retry state after plan generation (retry context has been used)
                retry_goal = self.goal_monitor.check_for_retry_request()
                if retry_goal:
                    print("🔄 Retry context used in plan generation, resetting retry state")
                    self.goal_monitor.reset_retry_request()
                
                print(f"📋 Generated plan with {len(plan.action_steps)} steps")
                print(f"🤔 Reasoning: {plan.reasoning}")
                
                # Execute the plan
                success = self.action_executor.execute_plan(plan, page_info)
                if success:
                    
                    # Check if goals were achieved during execution
                    goal_result = self.goal_monitor.evaluate_goal()
                    
                    # Check for retry requests after goal evaluation
                    retry_goal = self.goal_monitor.check_for_retry_request()
                    if retry_goal:
                        print("🔄 Goal requested retry after plan execution - regenerating plan")
                        print(f"   🔄 {retry_goal}: Retry requested (attempt {retry_goal.retry_count}/{retry_goal.max_retries})")
                        # Don't reset retry requests here - let them persist for the next iteration
                        continue
                    
                    if goal_result.status == GoalStatus.ACHIEVED:
                        duration_ms = (time.time() - start_time) * 1000
                        self.logger.log_goal_success(goal_description, duration_ms)
                        print(f"✅ Smart goal {goal_description} achieved during plan execution!")
                        self._print_goal_summary()
                        return True
                    
                    # If plan executed successfully but no goals achieved, scroll down one viewport height
                    # to explore more of the page instead of waiting for duplicate screenshots
                    print("📜 Plan executed successfully, scrolling down to explore more content")
                    self.page_utils.scroll_page()
                    continue
                else:
                    # Plan execution failed - check if it was due to retry request
                    retry_goal = self.goal_monitor.check_for_retry_request()
                    if retry_goal:
                        print("🔄 Plan execution aborted due to retry request - regenerating plan")
                        # Don't reset retry requests here - let them persist for the next iteration
                        # The retry state will be used to inform the next plan generation
                        continue
                    else:
                        print("❌ Plan execution failed for unknown reason")
                        continue
            
            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_goal_failure(goal_description, f"Failed after {self.max_attempts} attempts", duration_ms)
            print(f"❌ Failed to achieve goal after {self.max_attempts} attempts")
            self._print_goal_summary()
            return False
        finally:
            # Mark act() as finished and flush any auto-on-load actions that arrived mid-act
            self._in_act = False
            if self._interpretation_mode_stack:
                self._interpretation_mode_stack.pop()
            try:
                self._flush_pending_auto_on_load()
            except Exception:
                pass

    def goto(self, url: str, timeout: int = 2000) -> None:
        """Go to a URL"""
        if not self.started:
            print("❌ Bot not started")
            return
        
        self.page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        self.url = url
        # Ensure GoalMonitor history reflects the first real navigation instead of about:blank
        try:
            if hasattr(self, 'goal_monitor') and self.goal_monitor:
                hist = getattr(self.goal_monitor, 'url_history', None)
                ptr = getattr(self.goal_monitor, 'url_pointer', None)
                current = self.page.url
                # If we only have the initial about:blank entry, replace it with the real URL
                if isinstance(hist, list) and len(hist) == 1 and (hist[0] or '').startswith('about:blank'):
                    self.goal_monitor.url_history = [current]
                    self.goal_monitor.url_pointer = 0
                # If history exists but pointer is not at the end, truncate forward stack and append
                elif isinstance(hist, list) and isinstance(ptr, int) and 0 <= ptr < len(hist):
                    if hist[ptr] != current:
                        # Truncate any forward entries
                        if ptr < (len(hist) - 1):
                            self.goal_monitor.url_history = hist[: ptr + 1]
                        # Append only if it's not already the last entry
                        if not self.goal_monitor.url_history or self.goal_monitor.url_history[-1] != current:
                            self.goal_monitor.url_history.append(current)
                        self.goal_monitor.url_pointer = len(self.goal_monitor.url_history) - 1
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
    ) -> bool:
        """
        Register multiple commands for later reference and execution.
        
        Args:
            prompts: List of command prompts to register
            ref_id: Unique identifier for referencing these commands
            all_must_be_true: Require every command to succeed when evaluating this ref
            interpretation_mode: Optional interpretation mode to apply when executing this ref
            
        Returns:
            True if commands were registered successfully, False otherwise
        """
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
            }
            mode = "ALL" if all_must_be_true else "ANY"
            extra = f", interpretation={normalized_mode}" if normalized_mode else ""
            self.logger.log(LogLevel.INFO, LogCategory.SYSTEM, f"Registered {len(prompts)} commands with ref ID: {ref_id} (mode={mode}{extra})")
            print(f"📋 Registered {len(prompts)} commands with ref ID: {ref_id} (mode={mode}{extra})")
            
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

    def _generate_plan(self, goal_description: str, additional_context: str, screenshot: bytes, page_info: PageInfo) -> Optional[VisionPlan]:
        """Generate an action plan using numbered element detection"""

        # Check if any active goal needs element detection
        needs_detection = self.goal_monitor.active_goal.needs_detection if self.goal_monitor.active_goal else True

        current_mode = self._get_current_interpretation_mode()
        semantic_hint = self._get_semantic_target(goal_description)
        print(f"[PlanGen] interpretation_mode={current_mode} semantic_hint={'yes' if semantic_hint else 'no'} for '{goal_description}'")

        if not needs_detection:
            print("🚫 Skipping element detection - goal doesn't require it")
            # Create action plan without element detection
            plan = self.plan_generator.create_plan(
                goal_description=goal_description,
                additional_context=additional_context,
                detected_elements=PageElements(elements=[]),
                page_info=page_info,
                screenshot=screenshot,
                active_goal=self.goal_monitor.active_goal if self.goal_monitor else None,
                retry_goal=self.goal_monitor.check_for_retry_request() if self.goal_monitor else None,
                page=self.page,
                command_history=self.command_history,
            )
            return plan

        # Step 1: Number interactive elements (respecting focus context)
        print("🔢 Numbering interactive elements...")
        element_data = self.overlay_manager.create_numbered_overlays(page_info, mode="interactive")
        
        # Filter elements based on current focus context
        if self.focus_manager.get_current_focus_context():
            print("🎯 Filtering elements based on current focus context...")
            element_data = self._filter_elements_by_focus(element_data)

        if not element_data:
            print("❌ No interactive elements found for overlays")
            return None

        # Step 2: Take screenshot with numbered overlays visible (JPEG for speed)
        # Lower quality for model-bound overlay screenshot
        screenshot_with_overlays = self.page.screenshot(type="jpeg", quality=35, full_page=False)

        # Step 3: Generate plan with element indices (and optional filtered overlay list)
        plan = self.plan_generator.create_plan_with_element_indices(
            goal_description=goal_description,
            additional_context=additional_context,
            element_data=element_data,
            screenshot_with_overlays=screenshot_with_overlays,
            page_info=page_info,
                active_goal=self.goal_monitor.active_goal if self.goal_monitor else None,
                retry_goal=self.goal_monitor.check_for_retry_request() if self.goal_monitor else None,
                page=self.page,
                command_history=self.command_history,
                interpretation_mode=current_mode,
                semantic_hint=semantic_hint,
            )

        # Step 4: Clean up overlays after plan generation
        self.overlay_manager.remove_overlays()

        return plan

    # -------------------- Public memory helpers (general) --------------------
    # Memory store methods removed - deduplication now handled by focus manager

    def _create_goal_from_description(self, goal_description: str) -> tuple[BaseGoal, str]:
        """Set up smart goals based on the goal description and return the updated description"""
        # 1) Structured syntax (explicit) takes precedence
        try:
            sw = parse_structured_while(goal_description)
            if sw:
                print(f"🔁 WhileGoal (structured): '{sw}'")
                cond_text, body_text = sw
                wg = self._create_while_goal_from_parts(goal_description, cond_text, body_text)
                if wg:
                    print(f"🔁 Created WhileGoal (structured): '{wg.description}'")
                    return wg, body_text
                else:
                    print(f"ℹ️ No WhileGoal created from structured while parse for goal description: '{goal_description}'")
            else:
                print(f"ℹ️ No structured while parse for goal description: '{goal_description}'")
        except Exception as e:
            print(f"⚠️ Structured while parse failed: {e}")

        try:
            parsed_if = parse_structured_if(goal_description)
            if parsed_if:
                print(f"🔀 Structured IF parse: {parsed_if}")
                condition_text, success_action, fail_action = parsed_if
                if_goal = self._create_if_goal_from_parts(condition_text, success_action, fail_action)
                print(f"🔀 Created IfGoal (structured): '{if_goal.description}'")
                if if_goal:
                    result_goal, result_description = self._evaluate_if_goal(if_goal)
                    print(f"🔀 IfGoal (structured) evaluation result: {result_goal}")
                    if result_goal:
                        print(f"🔀 Added IfGoal (structured) active sub-goal: {result_goal.__class__.__name__} - '{result_goal.description}'")
                        return result_goal, result_description
                    elif result_description == "":
                        print("ℹ️ Structured IF evaluated false with no fail action → no-op")
                        return None, ""
                else:
                    print(f"ℹ️ No IfGoal created from structured if parse for goal description: '{goal_description}'")
            else:
                print(f"ℹ️ No structured if parse for goal description: '{goal_description}'")
            
        except Exception as e:
            print(f"⚠️ Structured IF parse failed: {e}")

        # 2) Single-keyword commands (fast path)
        try:
            kw = parse_keyword_command(goal_description)
            if kw:
                keyword, payload = kw
                
                goal = self._create_goal_from_keyword(keyword, payload)
                if goal:
                    print(f"✅ Created {goal.__class__.__name__} via keyword '{keyword}': '{goal.description}'")
                    # Focus plan generation on the payload/action text
                    return goal, payload or goal_description
                else:
                    print(f"ℹ️ No goal created from keyword command for goal description: '{goal_description}'")
                    return None, goal_description
            else:
                print(f"ℹ️ No keyword command parse for goal description: '{goal_description}'")
                return None, goal_description
        except Exception as e:
            print(f"⚠️ Keyword command parse failed: {e}")


        print(f"ℹ️ No goal created from goal description: '{goal_description}'")
        return None, goal_description

    def _create_goal_from_keyword(self, keyword: str, payload: str) -> Optional[BaseGoal]:
        """Create a specific goal from a single keyword and payload."""
        k = (keyword or "").lower().strip()
        p = (payload or "").strip()
        if k == "press":
            keys = p or extract_press_target(p)
            if keys:
                return PressGoal(description=f"Press action: {keys}", target_keys=keys)
        if k == "scroll":
            if p:
                return ScrollGoal(description=f"Scroll action: {p}", user_request=p)
            else:
                return ScrollGoal(description="Scroll action: down", user_request="scroll down")
        if k == "click":
            target = p or extract_click_target(p)
            if target:
                return ClickGoal(description=f"Click action: {target}", target_description=target)
        if k == "navigate":
            target = p
            if target:
                return NavigationGoal(description=f"Navigation action: {target}", navigation_intent=target)
        if k == "form":
            desc = p or "Fill the form"
            return FormFillGoal(description=f"Form fill action: {desc}", trigger_on_submit=False, trigger_on_field_input=True)
        if k == "back":
            import re as _re
            steps = 1
            m = _re.match(r"^(\d+)$", p)
            if m:
                steps = max(1, int(m.group(1)))
            try:
                pointer = getattr(self.goal_monitor, 'url_pointer', None)
                start_index = pointer if pointer is not None else (len(self.goal_monitor.url_history) - 1 if self.goal_monitor and self.goal_monitor.url_history else 0)
                start_url = self.goal_monitor.url_history[start_index] if self.goal_monitor and self.goal_monitor.url_history and 0 <= start_index < len(self.goal_monitor.url_history) else (self.page.url if self.page else "")
            except Exception:
                start_index, start_url = 0, (self.page.url if self.page else "")
            return BackGoal(description=f"Back action: {steps}", steps_back=steps, start_index=start_index, start_url=start_url, needs_detection=False)
        if k == "forward":
            import re as _re
            steps = 1
            m = _re.match(r"^(\d+)$", p)
            if m:
                steps = max(1, int(m.group(1)))
            try:
                pointer = getattr(self.goal_monitor, 'url_pointer', None)
                start_index = pointer if pointer is not None else (len(self.goal_monitor.url_history) - 1 if self.goal_monitor and self.goal_monitor.url_history else 0)
                start_url = self.goal_monitor.url_history[start_index] if self.goal_monitor and self.goal_monitor.url_history and 0 <= start_index < len(self.goal_monitor.url_history) else (self.page.url if self.page else "")
            except Exception:
                start_index, start_url = 0, (self.page.url if self.page else "")
            return ForwardGoal(description=f"Forward action: {steps}", steps_forward=steps, start_index=start_index, start_url=start_url, needs_detection=False)
        # if k == "ref":
        #     goal_description = keyword + ": " + payload
        #     print(f"🔄 Handling ref command: '{goal_description}'")
        #     ref_result = self._handle_ref_commands(goal_description)
        #     if ref_result is not None:
        #         print("✅ Created RefGoal (structured)")
        #         return BaseGoal.make_ref_goal(goal_description, payload, ref_result)
        #     else:
        #         print(f"ℹ️ No ref result for ref command: '{goal_description}'")
        #         return BaseGoal.make_ref_goal(goal_description, payload, False)
        return None

    def _create_while_goal_from_parts(self, goal_description: str, cond_text: str, body_text: str) -> Optional[WhileGoal]:
        """Create WhileGoal from explicit condition/body parts."""
        try:
            from goals.condition_engine import compile_nl_to_expr, create_predicate_condition as _create_predicate
            expr = compile_nl_to_expr(cond_text)
            if not expr:
                return None
            condition = _create_predicate(expr, f"Predicate: {cond_text}")
            wg = WhileGoal(
                description=goal_description,
                condition=condition,
                loop_prompt=body_text,
            )
            return wg
        except Exception as e:
            print(f"⚠️ Error creating structured WhileGoal: {e}")
            return None

    def _execute_while_loop(self, goal: WhileGoal, start_time: float) -> bool:
        """Execute a WhileGoal using standard while-loop semantics."""
        loop_description = goal.description or f"While loop: {goal.loop_prompt}"
        print(f"🔁 Starting while loop: {loop_description}")

        iterations = 0
        # Ensure we evaluate a clean condition on the first pass
        if self.goal_monitor.active_goal:
            self.goal_monitor.clear_all_goals()
        self.goal_monitor.reset_retry_request()

        while True:
            try:
                context = self.goal_monitor._build_goal_context()
            except Exception as e:
                error_msg = f"Unable to build context for while loop evaluation: {e}"
                print(f"❌ {error_msg}")
                self.logger.log_goal_failure(loop_description, error_msg, (time.time() - start_time) * 1000)
                return False

            try:
                condition_result = bool(goal.condition.evaluator(context))
            except Exception as e:
                error_msg = f"While condition error: {e}"
                print(f"❌ {error_msg}")
                self.logger.log_goal_failure(loop_description, error_msg, (time.time() - start_time) * 1000)
                return False

            goal.progress.last_condition_result = condition_result
            self.logger.log_condition_evaluation(goal.condition.description, condition_result)
            print(f"🔍 While condition '{goal.condition.description}' → {condition_result}")

            if not condition_result:
                duration_ms = (time.time() - start_time) * 1000
                print(f"✅ While loop condition false after {iterations} iteration{'s' if iterations != 1 else ''}")
                if goal.else_prompt and goal.else_prompt.strip():
                    print(f"➡️ Executing while-else prompt: {goal.else_prompt}")
                    else_result = bool(self.act(goal.else_prompt))
                    if else_result:
                        self.logger.log_goal_success(loop_description, duration_ms, {"iterations": iterations, "outcome": "else"})
                    else:
                        self.logger.log_goal_failure(loop_description, f"Else prompt failed: {goal.else_prompt}", duration_ms)
                    return else_result

                self.logger.log_goal_success(loop_description, duration_ms, {"iterations": iterations, "outcome": "completed"})
                return True

            if iterations >= goal.max_iterations:
                duration_ms = (time.time() - start_time) * 1000
                reason = f"Loop exceeded max iterations ({goal.max_iterations})"
                print(f"❌ {reason}")
                self.logger.log_goal_failure(loop_description, reason, duration_ms)
                return False

            if not goal.loop_prompt or not goal.loop_prompt.strip():
                duration_ms = (time.time() - start_time) * 1000
                reason = "While loop body is empty"
                print(f"❌ {reason}")
                self.logger.log_goal_failure(loop_description, reason, duration_ms)
                return False

            iterations += 1
            goal.progress.iterations = iterations
            print(f"🔄 While loop iteration {iterations}: executing body '{goal.loop_prompt}'")

            body_result = bool(self.act(goal.loop_prompt))
            if not body_result:
                duration_ms = (time.time() - start_time) * 1000
                reason = f"Loop body failed on iteration {iterations}"
                print(f"❌ {reason}")
                self.logger.log_goal_failure(loop_description, reason, duration_ms)
                return False

            # Reset goal monitor state before the next condition check
            if self.goal_monitor.active_goal:
                self.goal_monitor.clear_all_goals()
            self.goal_monitor.reset_retry_request()

    def _create_if_goal_from_parts(self, condition_text: str, success_text: str, fail_text: Optional[str]) -> Optional[IfGoal]:
        """Create IfGoal from explicit parts."""
        try:
            from goals.condition_engine import compile_nl_to_expr, create_predicate_condition as _create_predicate
            expr = compile_nl_to_expr(condition_text)
            if not expr:
                return None
            condition = _create_predicate(expr, f"Predicate: {condition_text}")

            print(f"Success text: '{success_text}'")
            success_goal, _ = self._create_goal_from_description(success_text)
            if not success_goal:
                return None
            print(f"🔀 Created success goal: '{success_goal.description}'")
            if fail_text and fail_text.strip():
                fail_goal, _ = self._create_goal_from_description(fail_text)
                if not fail_goal:
                    fail_goal = BaseGoal.make_noop_goal("No-op fail branch")
            else:
                fail_goal = BaseGoal.make_noop_goal("No-op fail branch")

            if_goal = IfGoal(condition, success_goal, fail_goal, description=f"If {condition_text} then {success_goal.description} else {fail_goal.description}")
            return if_goal
        except Exception as e:
            print(f"⚠️ Error creating structured IfGoal: {e}")
            return None

    def _evaluate_if_goal(self, if_goal: IfGoal) -> tuple[Optional[BaseGoal], str]:
        """Evaluate a conditional goal immediately and return the active sub-goal and updated description"""
        try:
            # Create a basic context for evaluation
            from goals.base import GoalContext, BrowserState
            page_info = self.page_utils.get_page_info()
            
            basic_context = GoalContext(
                initial_state=BrowserState(
                    timestamp=0, url=page_info.url, title=page_info.title,
                    page_width=page_info.width, page_height=page_info.height,
                    scroll_x=0, scroll_y=0
                ),
                current_state=BrowserState(
                    timestamp=0, url=page_info.url, title=page_info.title,
                    page_width=page_info.width, page_height=page_info.height,
                    scroll_x=0, scroll_y=0
                ),
                page_reference=self.page
            )
            
            # Evaluate the conditional goal to get the active sub-goal
            eval_result = if_goal.evaluate(basic_context)
            
            if eval_result.status == GoalStatus.ACHIEVED:
                return if_goal.success_goal, if_goal.success_goal.description
            else:
                return if_goal.fail_goal, if_goal.fail_goal.description
                
        except Exception as e:
            print(f"⚠️ Error evaluating IfGoal: {e}")
            return None, ""

    def _print_goal_summary(self) -> None:
        """Print a summary of all goal statuses"""
        summary = self.goal_monitor.get_status_summary()
        
        print("\n📊 Goal Summary:")
        print(f"   ✅ Achieved: {summary['achieved']}")
        print(f"   ⏳ Pending: {summary['pending']}")
        print(f"   ❌ Failed: {summary['failed']}")

    def _add_to_command_history(self, command: str) -> None:
        """Add a command to the history, maintaining max size"""
        if command and command.strip():
            self.command_history.append(command.strip())
            # Keep only the last max_command_history commands
            if len(self.command_history) > self.max_command_history:
                self.command_history = self.command_history[-self.max_command_history:]
            print(f"📝 Added to command history: '{command.strip()}'")
    
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
                keyword, payload = kw_parsed
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
                if hasattr(self, 'focus_manager') and self.focus_manager:
                    self.focus_manager.dedup_enabled = True
                self.logger.log(LogLevel.INFO, LogCategory.SYSTEM, "Deduplication enabled")
                print("🧹 Deduplication enabled")
                return True
            
            # Check for dedup: disable
            elif goal_lower in ["dedup: disable", "dedup:disabled"]:
                self.dedup_mode = "off"
                if hasattr(self, 'focus_manager') and self.focus_manager:
                    self.focus_manager.dedup_enabled = False
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

                if not stored_prompts:
                    print(f"⚠️ Ref ID '{ref_id}' has no stored commands")
                    return True

                summary_mode = 'ALL' if all_must_be_true else 'ANY'
                mode_suffix = f", interpretation={ref_mode}" if ref_mode else ""
                print(f"🔄 Executing {len(stored_prompts)} stored commands for ref ID: {ref_id} (mode={summary_mode}{mode_suffix})")
                results: List[bool] = []

                for i, prompt in enumerate(stored_prompts, 1):
                    print(f"▶️ Executing stored command {i}/{len(stored_prompts)}: {prompt}")
                    success = bool(self.act(prompt, interpretation_mode=ref_mode))
                    results.append(success)
                    if success:
                        print(f"   ✅ Stored command {i} succeeded")
                    else:
                        print(f"   ❌ Stored command {i} failed")

                final_result = all(results) if all_must_be_true else any(results)
                summary_mode = "ALL must succeed" if all_must_be_true else "ANY success suffices"
                print(f"📊 Ref '{ref_id}' evaluation ({summary_mode}) → {'✅ True' if final_result else '❌ False'}")
                return final_result
            
            return None  # Not a ref command
            
        except Exception as e:
            traceback.print_exc()
            print(f"⚠️ Error handling ref commands: {e}")
            return False
    
    def write_session_log(self):
        """Write the session summary to the log file"""
        if hasattr(self, 'logger') and self.logger:
            self.logger.write_session_summary()
            print(f"📝 Session log written to: {self.logger.log_file}")
    
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


# Example usage
if __name__ == "__main__":
    bot = BrowserVisionBot()
    bot.start()
    bot.goto("https://www.reed.co.uk/jobs/ios-developer-jobs")
    bot.on_new_page_load(["if: cookie banner visible then: click: the button to accept cookies"])
    # bot.act("dedup: enable")
    # bot.register_prompts([
    #     "click: an ios job button", 
    #     "press: escape"
    # ], "job_loop_commands")
    # bot.act("while: not at the bottom of the page do: ref: job_loop_commands")
    
    bot.act("click: an ios job listing button", interpretation_mode="semantic")
    bot.act("click: the apply button", interpretation_mode="semantic")
    
    
    while True:
        user_input = input('\n👤 New task or "q" to quit: ')
        if user_input.lower() == "q":
            break
        bot.act(user_input)
    
    # Write session log when done
    bot.write_session_log()
