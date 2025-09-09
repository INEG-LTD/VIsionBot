"""
Vision Bot - Clean modular version.
"""
from __future__ import annotations

import hashlib
import os
import uuid
import re
from typing import Any, Optional, List, Dict

from playwright.sync_api import Browser, Page, Playwright

from goals.base import GoalResult
from models import VisionPlan, PageElements
from models.core_models import ActionStep, ActionType
from element_detection import OverlayManager, ElementDetector
from action_executor import ActionExecutor
from models.core_models import PageInfo
from utils import PageUtils
from utils.memory_store import MemoryStore
from goals import GoalMonitor, ClickGoal, GoalStatus, FormFillGoal, NavigationGoal, IfGoal, WhileGoal, PressGoal, ScrollGoal, Condition, BaseGoal, BackGoal, ForwardGoal
from goals.condition_engine import compile_nl_to_expr, create_predicate_condition as create_predicate
from planner.plan_generator import PlanGenerator
from utils.intent_parsers import (
    parse_while_statement,
    extract_click_target,
    extract_press_target,
    extract_navigation_intent,
    parse_structured_if,
    parse_structured_while,
    parse_keyword_command,
)
from goals.history_utils import (
    resolve_back_target,
    resolve_forward_target,
)


class BrowserVisionBot:
    """Modular vision-based web automation bot"""

    def __init__(
        self,
        page: Page = None,
        model_name: str = "gemini-2.0-flash",
        max_attempts: int = 10,
        max_detailed_elements: int = 20,
        include_detailed_elements: bool = True,
        two_pass_planning: bool = True,
        max_coordinate_overlays: int = 60,
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
        
        # General-purpose memory (dedup, context). Session-lifetime by default.
        self.memory_store = MemoryStore()
        # Dedup policy: detect from prompt by default ('auto').
        # 'off' = never dedup, 'on' = always dedup, 'auto' = only when prompt asks.
        self.dedup_mode: str = "auto"
        
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
        self.action_executor = ActionExecutor(page, self.goal_monitor, self.page_utils)
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

    def disable_auto_act_on_page_load(self) -> None:
        """Disable the auto-on-load behavior and detach the handler if possible."""
        self._auto_on_load_enabled = False
        self._auto_on_load_actions = []
        self._auto_on_load_urls_handled.clear()
        try:
            if self.page and self._auto_on_load_handler:
                # Playwright's sync API exposes .off on EventEmitter-like objects
                if self._auto_on_load_event_name:
                    self.page.off(self._auto_on_load_event_name, self._auto_on_load_handler)
                else:
                    self.page.off("load", self._auto_on_load_handler)
        except Exception:
            pass
        finally:
            self._auto_on_load_handler = None
            self._auto_on_load_event_name = None

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
                    saved_goals = list(getattr(self.goal_monitor, 'active_goals', []) or []) if hasattr(self, 'goal_monitor') else []
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
                                for g in saved_goals:
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
            if not self.goal_monitor or not self.goal_monitor.active_goals:
                return None
            from goals import PressGoal, ScrollGoal
            goals = self.goal_monitor.active_goals
            # Only bypass for Press/Scroll; Back/Forward use history + AI selection
            if not all(isinstance(g, (PressGoal, ScrollGoal)) for g in goals):
                return None

            # Build minimal plan
            steps: List[ActionStep] = []
            for g in goals:
                if isinstance(g, PressGoal):
                    if getattr(g, 'target_keys', None):
                        steps.append(ActionStep(action=ActionType.PRESS, keys_to_press=g.target_keys))
                elif isinstance(g, ScrollGoal):
                    # Heuristic direction from request; executor will refine using ScrollGoal
                    req = (getattr(g, 'user_request', '') or '').lower()
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
            results = self.goal_monitor.evaluate_goals()
            if any(r.status == GoalStatus.ACHIEVED for r in results.values()):
                self._print_goal_summary()
                return True

            # If not achieved, allow normal planning to proceed
            return None
        except Exception as e:
            print(f"⚠️ Simple-goal bypass failed: {e}")
            return None

    def act(self, goal_description: str, additional_context: str = "", smart_goals: bool = True) -> bool:
        """Main method to achieve a goal using vision-based automation"""
        self._in_act = True
        try:
            if not self.started:
                print("❌ Bot not started")
                return False
            
            if self.page.url.startswith("about:blank"):
                print("❌ Page is on the initial blank page")
                return False
            
            print(f"🎯 Starting goal: {goal_description}")
            
            # Clear any existing goals before starting a new goal
            if self.goal_monitor.active_goals:
                print(f"🧹 Clearing {len(self.goal_monitor.active_goals)} previous goals")
                self.goal_monitor.clear_all_goals()
            
            # Reset goal monitor state for fresh start
            self.goal_monitor.reset_retry_requests()
            
            self.goal_monitor.set_user_prompt(goal_description)
            
            # Set up smart goal monitoring if enabled
            if smart_goals:
                goal_description = self._setup_smart_goals(goal_description, additional_context)
                # If conditional evaluation resulted in a deliberate no-op (no fail action and condition false)
                if not goal_description.strip():
                    print("ℹ️ No actionable goal after condition evaluation (no-op). Skipping.")
                    return True
            
            print(f"🔍 Smart goals setup: {self.goal_monitor.active_goals}\n")

            # Simple goal bypass (no LLM): handle press/scroll-only flows directly
            simple_result = self._try_simple_goal_bypass()
            if simple_result is not None:
                return simple_result
            
            for attempt in range(self.max_attempts):
                self.current_attempt = attempt + 1
                print(f"\n--- Attempt {self.current_attempt}/{self.max_attempts} ---")
                
                # Show retry context at the start of each new attempt (but don't reset yet)
                if attempt > 0:  # Don't reset on first attempt
                    retry_goals = self.goal_monitor.check_for_retry_requests()
                    if retry_goals:
                        print(f"🔄 Starting attempt {self.current_attempt} with retry state from previous attempt")
                        for goal in retry_goals:
                            print(f"   🔄 {goal}: Retry attempt {goal.retry_count}/{goal.max_retries}")
                        # Don't reset retry state here - let it persist until after plan generation
                
                # Check if goal is already achieved
                if smart_goals:
                    goal_results = self.goal_monitor.evaluate_goals()
                    if any(result.status == GoalStatus.ACHIEVED for result in goal_results.values()):
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
                retry_goals = self.goal_monitor.check_for_retry_requests()
                if sig_hash == self.last_dom_signature and not retry_goals:
                    print("⚠️ Same DOM signature as last attempt, scrolling to break loop")
                    self.page_utils.scroll_page()
                    continue
                elif sig_hash == self.last_dom_signature and retry_goals:
                    print("🔄 Same DOM but retry requested - proceeding with retry attempt")
                self.last_dom_signature = sig_hash
                
                # Decide whether detection will be needed to avoid an extra screenshot
                needs_detection = any(g.needs_detection for g in self.goal_monitor.active_goals) if self.goal_monitor.active_goals else True
                # Only take a screenshot now when not running detection
                screenshot = None
                if not needs_detection:
                    # Lower quality for model-bound screenshot to reduce payload
                    screenshot = self.page.screenshot(type="jpeg", quality=35, full_page=False)

                # Generate plan using vision model (conditional goals are already resolved to their sub-goals)
                plan = self._generate_plan(goal_description, additional_context, screenshot, page_info)
                
                if not plan or not plan.action_steps:
                    print("❌ No valid plan generated")
                    continue
                
                # Reset retry state after plan generation (retry context has been used)
                retry_goals = self.goal_monitor.check_for_retry_requests()
                if retry_goals:
                    print("🔄 Retry context used in plan generation, resetting retry state")
                    self.goal_monitor.reset_retry_requests()
                
                print(f"📋 Generated plan with {len(plan.action_steps)} steps")
                print(f"🤔 Reasoning: {plan.reasoning}")
                
                # Execute the plan
                success = self.action_executor.execute_plan(plan, page_info)
                if success:
                    if smart_goals:
                        # Check if goals were achieved during execution
                        goal_results = self.goal_monitor.evaluate_goals()
                        
                        # Check for retry requests after goal evaluation
                        retry_goals = self.goal_monitor.check_for_retry_requests()
                        if retry_goals:
                            print("🔄 Goals requested retry after plan execution - regenerating plan")
                            for goal in retry_goals:
                                print(f"   🔄 {goal}: Retry requested (attempt {goal.retry_count}/{goal.max_retries})")
                            # Don't reset retry requests here - let them persist for the next iteration
                            continue
                        
                        if any(result.status == GoalStatus.ACHIEVED for result in goal_results.values()):
                            print("✅ Smart goal achieved during plan execution!")
                            self._print_goal_summary()
                            return True
                    
                    # If plan executed successfully but no goals achieved, scroll down one viewport height
                    # to explore more of the page instead of waiting for duplicate screenshots
                    print("📜 Plan executed successfully, scrolling down to explore more content")
                    self.page_utils.scroll_page()
                    continue
                else:
                    # Plan execution failed - check if it was due to retry request
                    retry_goals = self.goal_monitor.check_for_retry_requests()
                    if retry_goals:
                        print("🔄 Plan execution aborted due to retry request - regenerating plan")
                        # Don't reset retry requests here - let them persist for the next iteration
                        # The retry state will be used to inform the next plan generation
                        continue
                    else:
                        print("❌ Plan execution failed for unknown reason")
                        continue
            
            print(f"❌ Failed to achieve goal after {self.max_attempts} attempts")
            if smart_goals:
                self._print_goal_summary()
            return False
        finally:
            # Mark act() as finished and flush any auto-on-load actions that arrived mid-act
            self._in_act = False
            try:
                self._flush_pending_auto_on_load()
            except Exception:
                pass

    def goto(self, url: str) -> None:
        """Go to a URL"""
        if not self.started:
            print("❌ Bot not started")
            return
        
        self.page.goto(url, wait_until="domcontentloaded")
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


    def _generate_plan(self, goal_description: str, additional_context: str, screenshot: bytes, page_info: PageInfo) -> Optional[VisionPlan]:
        """Generate an action plan using numbered element detection"""

        # Check if any active goal needs element detection
        needs_detection = any(goal.needs_detection for goal in self.goal_monitor.active_goals) if self.goal_monitor.active_goals else True

        if not needs_detection:
            print("🚫 Skipping element detection - goal doesn't require it")
            # Create action plan without element detection
            plan = self.plan_generator.create_plan(
                goal_description=goal_description,
                additional_context=additional_context,
                detected_elements=PageElements(elements=[]),
                page_info=page_info,
                screenshot=screenshot,
                active_goals=list(self.goal_monitor.active_goals or []) if self.goal_monitor else [],
                retry_goals=list(self.goal_monitor.check_for_retry_requests() or []) if self.goal_monitor else [],
                page=self.page,
            )
            return plan

        # Step 1: Number every element
        print("🔢 Numbering all interactive elements...")
        element_data = self.overlay_manager.create_numbered_overlays(page_info)

        if not element_data:
            print("❌ No interactive elements found for overlays")
            return None

        # Step 2: Take screenshot with numbered overlays visible (JPEG for speed)
        # Lower quality for model-bound overlay screenshot
        screenshot_with_overlays = self.page.screenshot(type="jpeg", quality=35, full_page=False)

        # Optional pass 2a: pre-select relevant overlays using the detector to reduce prompt size
        relevant_overlay_indices = None
        # Run two-pass detection only when there are many overlays (saves a model call)
        if self.two_pass_planning and len(element_data) > self.max_coordinate_overlays:
            try:
                detected = self.element_detector.detect_elements_with_overlays(
                    goal_description=goal_description,
                    additional_context=additional_context,
                    screenshot=screenshot_with_overlays,
                    element_data=element_data,
                    page_info=page_info,
                )
                if detected and getattr(detected, 'elements', None):
                    relevant_overlay_indices = [
                        e.overlay_number for e in detected.elements if getattr(e, 'overlay_number', None) is not None
                    ]
                    # Heuristic refinement for link/navigation intents (e.g., prefer hrefs matching goal)
                    refined = self.plan_generator.refine_overlays_by_goal(element_data, relevant_overlay_indices, goal_description)
                    if refined:
                        relevant_overlay_indices = refined
                    print(f"🎯 Two-pass: {len(relevant_overlay_indices)} recommended overlays from detector: {relevant_overlay_indices[:10]}{'...' if len(relevant_overlay_indices)>10 else ''}")
            except Exception as e:
                print(f"⚠️ Two-pass detection failed, continuing with single-pass: {e}")

        # Step 3: Generate plan with element indices (and optional filtered overlay list)
        plan = self.plan_generator.create_plan_with_element_indices(
            goal_description=goal_description,
            additional_context=additional_context,
            element_data=element_data,
            screenshot_with_overlays=screenshot_with_overlays,
            page_info=page_info,
            relevant_overlay_indices=relevant_overlay_indices,
            active_goals=list(self.goal_monitor.active_goals or []) if self.goal_monitor else [],
            retry_goals=list(self.goal_monitor.check_for_retry_requests() or []) if self.goal_monitor else [],
            page=self.page,
            dedup_mode=self.dedup_mode,
        )

        # Step 4: Clean up overlays after plan generation
        self.overlay_manager.remove_overlays()

        return plan

    # -------------------- Dedup policy helpers --------------------
    def set_dedup_mode(self, mode: str) -> None:
        """Set dedup mode: 'off' (default), 'on', or 'auto'."""
        mode = (mode or "off").strip().lower()
        if mode not in ("off", "on", "auto"):
            mode = "off"
        self.dedup_mode = mode


    # -------------------- Public memory helpers (general) --------------------
    def remember(self, kind: str, signature: str, *, scope: str = "domain", ttl_seconds: float | None = 3600.0, data: Dict | None = None) -> None:
        """Add an item to the bot's general memory.

        - kind: logical namespace (e.g., 'clicked', 'dismissed_banner', 'filled_field')
        - signature: any stable identifier for the target (use stable_sig for hashing)
        - scope: 'global' | 'domain' | 'page'
        - ttl_seconds: expire after this many seconds (None = no expiry)
        - data: optional metadata
        """
        try:
            url = self.page.url if self.page else ""
            self.memory_store.put(kind, signature, scope=scope, url=url, ttl_seconds=ttl_seconds, data=data)
        except Exception:
            pass

    def remember_has(self, kind: str, signature: str, *, scope: str = "domain") -> bool:
        try:
            url = self.page.url if self.page else ""
            return self.memory_store.has(kind, signature, scope=scope, url=url)
        except Exception:
            return False

    def remember_remove(self, kind: str, signature: str, *, scope: str = "domain") -> None:
        try:
            url = self.page.url if self.page else ""
            self.memory_store.remove(kind, signature, scope=scope, url=url)
        except Exception:
            pass

    def remember_list(self, kind: str | None = None, *, scope: str = "domain") -> Dict[str, Any]:
        try:
            url = self.page.url if self.page else ""
            return self.memory_store.list(kind=kind, scope=scope, url=url)
        except Exception:
            return {}

    def remember_clear_scope(self, scope: str = "domain") -> None:
        try:
            url = self.page.url if self.page else ""
            self.memory_store.clear_scope(scope=scope, url=url)
        except Exception:
            pass

    def _setup_smart_goals(self, goal_description: str, additional_context: str = "") -> str:
        """Set up smart goals based on the goal description and return the updated description"""
        # 1) Structured syntax (explicit) takes precedence
        try:
            sw = parse_structured_while(goal_description)
            if sw:
                cond_text, body_text = sw
                wg = self._create_while_goal_from_parts(cond_text, body_text)
                if wg:
                    self.goal_monitor.add_goal(wg)
                    print(f"🔁 Added WhileGoal (structured): '{wg.description}'")
                    return body_text
        except Exception as e:
            print(f"⚠️ Structured while parse failed: {e}")

        try:
            si = parse_structured_if(goal_description)
            if si:
                cond_text, success_desc, fail_desc = si
                ig = self._create_if_goal_from_parts(cond_text, success_desc, fail_desc)
                if ig:
                    active_sub_goal, updated_description = self._evaluate_conditional_goal_immediately(ig)
                    if active_sub_goal:
                        self.goal_monitor.add_goal(active_sub_goal)
                        print(f"🔀 Added IfGoal (structured) active sub-goal: {active_sub_goal.__class__.__name__} - '{active_sub_goal.description}'")
                        return updated_description
                    elif updated_description == "":
                        print("ℹ️ Structured IF evaluated false with no fail action → no-op")
                        return ""
        except Exception as e:
            print(f"⚠️ Structured IF parse failed: {e}")

        # 2) Single-keyword commands (fast path)
        try:
            kw = parse_keyword_command(goal_description)
            if kw:
                keyword, payload = kw
                goal = self._create_goal_from_keyword(keyword, payload)
                if goal:
                    self.goal_monitor.add_goal(goal)
                    print(f"✅ Added {goal.__class__.__name__} via keyword '{keyword}': '{goal.description}'")
                    # Focus plan generation on the payload/action text
                    return payload or goal_description
        except Exception as e:
            print(f"⚠️ Keyword command parse failed: {e}")
            
        return goal_description

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
        return None

    def _create_while_goal(self, goal_description: str) -> tuple[Optional[WhileGoal], str]:
        """Create a WhileGoal from natural language loop text. Returns (goal, loop_body_description)."""
        try:
            cond_text, body_text = parse_while_statement(goal_description)
            if not cond_text or not body_text:
                return None, ""

            print("🔍 DEBUG: Parsed while/until statement:")
            print(f"   📋 Stop condition: '{cond_text}'")
            print(f"   🔁 Loop body: '{body_text}'")

            # Build condition via predicate engine only (no legacy screenshot methods)
            from goals.condition_engine import compile_nl_to_expr, create_predicate_condition as _create_predicate

            expr = compile_nl_to_expr(cond_text)
            condition = _create_predicate(expr, f"Predicate: {cond_text}") if expr else None

            # Fallbacks for common loop stops
            if not condition:
                low = cond_text.lower()
                # time patterns like "5pm", "17:00" → use engine system.hour/minute
                if not condition:
                    import re
                    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", cond_text, flags=re.IGNORECASE)
                    if m:
                        hour = int(m.group(1))
                        minute = int(m.group(2) or 0)
                        mer = (m.group(3) or "").lower()
                        if mer == "pm" and hour < 12:
                            hour += 12
                        if mer == "am" and hour == 12:
                            hour = 0
                        expr = {
                            "or": [
                                {">": [{"call": {"name": "system.hour"}}, hour]},
                                {"and": [
                                    {"==": [{"call": {"name": "system.hour"}}, hour]},
                                    {">=": [{"call": {"name": "system.minute"}}, minute]}
                                ]}
                            ]
                        }
                        condition = _create_predicate(expr, f"Time >= {hour:02d}:{minute:02d}")
                # "see X" → viewport text contains X
                if not condition and ("see" in low or "visible" in low):
                    # Try to extract quoted target first
                    import re
                    qm = re.search(r"['\"]([^'\"]+)['\"]", cond_text)
                    target = qm.group(1) if qm else cond_text
                    expr = {"contains": [{"call": {"name": "env.page.viewport_text"}}, target]}
                    condition = _create_predicate(expr, f"Viewport shows '{target}'")

            if not condition:
                # Heuristic fallback: handle "reach X page" style conditions by building a predicate
                low = cond_text.lower()
                import re as _re
                # Try to extract quoted phrase or the word(s) after 'reach'/'back to'/'return to'
                qm = _re.search(r"['\"]([^'\"]+)['\"]", cond_text)
                target_phrase = (qm.group(1) if qm else cond_text).strip()
                # Common specialization: google search page
                expr = None
                if ("google" in low) and ("search" in low or "results" in low):
                    expr = {
                        "and": [
                            {"contains": [{"call": {"name": "env.page.url"}}, "google"]},
                            {"or": [
                                {"contains": [{"call": {"name": "env.page.url"}}, "search"]},
                                {"contains": [{"call": {"name": "env.page.url"}}, "q="]}
                            ]}
                        ]
                    }
                else:
                    # Generic: URL or title contains target phrase tokens
                    expr = {
                        "or": [
                            {"contains": [{"call": {"name": "env.page.url"}}, target_phrase]},
                            {"contains": [{"call": {"name": "env.page.title"}}, target_phrase]}
                        ]
                    }
                from goals.condition_engine import create_predicate_condition as _create_predicate
                try:
                    condition = _create_predicate(expr, f"Reach '{target_phrase}'")
                except Exception:
                    condition = None

            if not condition:
                print(f"⚠️ Could not create condition for while-statement: {cond_text}")
                return None, ""

            # Create the loop body as a sub-goal using shared logic
            body_goal = self._create_sub_goal(body_text)

            # Build WhileGoal; align detection with body goal
            wg = WhileGoal(
                condition=condition,
                loop_goal=body_goal,
                description=goal_description,
                max_iterations=30,
                max_duration_s=240.0,
                needs_detection=getattr(body_goal, 'needs_detection', True),
            )

            # For planning, we want the body text as the main requested action
            updated_description = body_goal.description
            return wg, updated_description

        except Exception as e:
            print(f"❌ Error creating WhileGoal: {e}")
            return None, ""

    def _create_conditional_goal(self, goal_description: str) -> Optional[IfGoal]:
        """Create a conditional goal from a goal description"""
        try:
            # Parse the conditional statement
            condition_text, success_action, fail_action = self._parse_conditional_statement(goal_description)
            
            print("🔍 DEBUG: Parsed conditional statement:")
            print(f"   📋 Condition: '{condition_text}'")
            print(f"   ✅ Success action: '{success_action}'")
            print(f"   ❌ Fail action: '{fail_action}'")
            
            if not condition_text:
                print(f"⚠️ Could not parse condition from: {goal_description}")
                return None
            
            # Build the predicate using the condition engine (which can use AI-assisted helpers internally)
            from goals.condition_engine import compile_nl_to_expr, create_predicate_condition as _create_predicate
            expr = compile_nl_to_expr(condition_text)
            condition = _create_predicate(expr, f"Predicate: {condition_text}") if expr else None
            
            if not condition:
                # Fallback: create a simple condition based on common patterns
                condition = self._create_fallback_condition(condition_text)
            
            if not condition:
                print(f"⚠️ Could not create condition for: {condition_text}")
                return None
            
            print("🔍 DEBUG: Created condition:")
            print(f"   📋 Condition type: {condition.condition_type}")
            print(f"   📋 Condition description: {condition.description}")
            
            # Create sub-goals for success and (optional) fail actions
            # A success action is required - if none provided, the goal should fail
            if success_action is None:
                print("⚠️ No success action provided for conditional goal. Goal will fail.")
                return None

            success_goal = self._create_sub_goal(success_action)

            if fail_action is None:
                # Use a no-op fail goal so we can skip adding any fail action later
                from goals.base import BaseGoal, GoalResult, GoalStatus
                class _NoOpFailGoal(BaseGoal):
                    def __init__(self):
                        super().__init__("No-op (no fail action)")
                        self._is_noop = True
                    def evaluate(self, context):
                        return GoalResult(status=GoalStatus.ACHIEVED, confidence=1.0, reasoning="Condition false → no-op fail branch")
                    def get_description(self, context):
                        return self.description
                fail_goal = _NoOpFailGoal()
            else:
                fail_goal = self._create_sub_goal(fail_action)
            
            print("🔍 DEBUG: Created sub-goals:")
            print(f"   ✅ Success goal: {success_goal.__class__.__name__} - '{success_goal.description}'")
            print(f"   ❌ Fail goal: {fail_goal.__class__.__name__} - '{fail_goal.description}'")
            
            # Create the IfGoal
            if_goal = IfGoal(
                condition=condition,
                success_goal=success_goal,
                fail_goal=fail_goal,
                description=goal_description
            )
            
            return if_goal
            
        except Exception as e:
            print(f"❌ Error creating conditional goal: {e}")
            return None

    def _parse_conditional_statement(self, goal_description: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse a conditional statement into condition, success action, and fail action using AI"""
        try:
            from ai_utils import generate_text
            
            # Create a prompt for the AI to parse the conditional statement
            prompt = f"""
                        Parse the following conditional statement and extract the condition, success action, and fail action.

                        Conditional statement: "{goal_description}"

                        Rules:
                        - Extract the condition that needs to be evaluated
                        - Extract the success action to take if the condition is true
                        - Extract the fail action to take if the condition is false (if present)
                        - A success action is REQUIRED - if none is found, the statement is invalid
                        - A fail action is OPTIONAL - if none is found, use null

                        Common conditional patterns:
                        - "if X then Y else Z" → condition: X, success: Y, fail: Z
                        - "when X do Y otherwise Z" → condition: X, success: Y, fail: Z
                        - "if X, Y, else Z" → condition: X, success: Y, fail: Z
                        - "should X, then Y, otherwise Z" → condition: X, success: Y, fail: Z
                        - "provided that X, do Y, else Z" → condition: X, success: Y, fail: Z
                        - "in case X, Y, otherwise Z" → condition: X, success: Y, fail: Z
                        - "unless X, do Y, else Z" → condition: X, success: Y, fail: Z
                        - "if X then Y" → condition: X, success: Y, fail: null
                        - "when X do Y" → condition: X, success: Y, fail: null

                        Response format (JSON only):
                        {{
                            "condition": "the condition to evaluate",
                            "success_action": "action to take if condition is true",
                            "fail_action": "action to take if condition is false (or null if not specified)"
                        }}

                        If the statement cannot be parsed or has no success action, return:
                        {{
                            "condition": null,
                            "success_action": null,
                            "fail_action": null
                        }}
                        """
            
            # Call AI to parse the conditional statement
            response = generate_text(
                prompt=prompt,
                reasoning_level="minimal",
                system_prompt="You are a conditional statement parser. Extract condition, success action, and fail action from natural language conditional statements. Return only JSON.",
                model="gpt-5-nano"
            )
            
            if not response:
                print(f"⚠️ No AI response for conditional statement: '{goal_description}'")
                return None, None, None
            
            # Parse the JSON response
            import json
            try:
                result: dict = json.loads(response.strip())
            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON response for conditional statement '{goal_description}': {response}")
                return None, None, None
            
            condition = result.get('condition')
            success_action = result.get('success_action')
            fail_action = result.get('fail_action')
            
            # Validate that we have at least a condition and success action
            if not condition or not success_action:
                print(f"⚠️ Invalid conditional statement - missing condition or success action: '{goal_description}'")
                return None, None, None
            
            # Convert null fail_action to None
            if fail_action is None or fail_action == "null":
                fail_action = None
            
            return condition, success_action, fail_action
                
        except Exception as e:
            print(f"⚠️ Error parsing conditional statement '{goal_description}': {e}")
            return None, None, None

    def _create_while_goal_from_parts(self, cond_text: str, body_text: str) -> Optional[WhileGoal]:
        """Create WhileGoal from explicit condition/body parts."""
        try:
            from goals.condition_engine import compile_nl_to_expr, create_predicate_condition as _create_predicate
            expr = compile_nl_to_expr(cond_text)
            if not expr:
                return None
            condition = _create_predicate(expr, f"Predicate: {cond_text}")
            loop_goal = self._create_goal_from_description(body_text, is_sub_goal=True)
            if not loop_goal:
                return None
            wg = WhileGoal(
                condition=condition,
                loop_goal=loop_goal,
                description=f"While ({cond_text}) do: {loop_goal.description}",
                needs_detection=getattr(loop_goal, 'needs_detection', True),
            )
            return wg
        except Exception as e:
            print(f"⚠️ Error creating structured WhileGoal: {e}")
            return None

    def _create_if_goal_from_parts(self, condition_text: str, success_text: str, fail_text: Optional[str]) -> Optional[IfGoal]:
        """Create IfGoal from explicit parts."""
        try:
            from goals.condition_engine import compile_nl_to_expr, create_predicate_condition as _create_predicate
            expr = compile_nl_to_expr(condition_text)
            if not expr:
                return None
            condition = _create_predicate(expr, f"Predicate: {condition_text}")

            success_goal = self._create_goal_from_description(success_text, is_sub_goal=True)
            if not success_goal:
                return None

            if fail_text and fail_text.strip():
                fail_goal = self._create_goal_from_description(fail_text, is_sub_goal=True)
                if not fail_goal:
                    fail_goal = self._make_noop_goal("No-op fail branch")
            else:
                fail_goal = self._make_noop_goal("No-op fail branch")

            if_goal = IfGoal(condition, success_goal, fail_goal, description=f"If {condition_text} then {success_goal.description} else {fail_goal.description}")
            return if_goal
        except Exception as e:
            print(f"⚠️ Error creating structured IfGoal: {e}")
            return None

    def _make_noop_goal(self, reason: str):
        class _NoOpGoal(BaseGoal):
            def __init__(self, description: str):
                super().__init__(description, needs_detection=False)
                self._is_noop = True
            def evaluate(self, context):
                return GoalResult(status=GoalStatus.ACHIEVED, confidence=1.0, reasoning="No-op branch")
            def get_description(self, context):
                return self.description
        return _NoOpGoal(reason)

    def _create_fallback_condition(self, condition_text: str) -> Optional[Condition]:
        """Create a fallback condition when AI parser fails"""
        try:
            # Simple fallback conditions based on common patterns
            condition_lower = condition_text.lower().strip()
            
            # Check for cookie banner patterns
            if "cookie" in condition_lower and ("banner" in condition_lower or "consent" in condition_lower):
                # Use common cookie banner selectors
                cookie_selectors = [
                    "[data-testid*='cookie']",
                    ".cookie-banner",
                    "#cookie-banner", 
                    "[class*='cookie']",
                    "[id*='cookie']",
                    "[data-testid*='consent']",
                    ".consent-banner",
                    "[class*='consent']"
                ]
                selector = ", ".join(cookie_selectors)
                from goals.condition_engine import create_predicate_condition as _create_predicate
                expr = {"call": {"name": "dom.exists", "args": {"selector": selector, "within": "page"}}}
                return _create_predicate(expr, "Cookie banner exists")
            
            # Check for element existence patterns
            if "element" in condition_lower and ("exists" in condition_lower or "visible" in condition_lower):
                # Extract selector if possible
                selector_match = re.search(r"element\s+['\"]([^'\"]+)['\"]", condition_text)
                if selector_match:
                    selector = selector_match.group(1)
                    from goals.condition_engine import create_predicate_condition as _create_predicate
                    expr = {"call": {"name": "dom.exists", "args": {"selector": selector, "within": "page"}}}
                    return _create_predicate(expr, condition_text)
            
            # Check for text content patterns
            if "contains" in condition_lower and ("text" in condition_lower or "page" in condition_lower):
                text_match = re.search(r"['\"]([^'\"]+)['\"]", condition_text)
                if text_match:
                    text = text_match.group(1)
                    from goals.condition_engine import create_predicate_condition as _create_predicate
                    expr = {"contains": [{"call": {"name": "env.page.viewport_text"}}, text]}
                    return _create_predicate(expr, condition_text)
            
            # Check for URL patterns
            if "url" in condition_lower and "contains" in condition_lower:
                url_match = re.search(r"['\"]([^'\"]+)['\"]", condition_text)
                if url_match:
                    url_fragment = url_match.group(1)
                    from goals.condition_engine import create_predicate_condition as _create_predicate
                    expr = {"contains": [{"call": {"name": "env.page.url"}}, url_fragment]}
                    return _create_predicate(expr, condition_text)
            
            # Check for weekday patterns
            if "weekday" in condition_lower or "weekend" in condition_lower:
                from goals.condition_engine import create_predicate_condition as _create_predicate
                if "weekday" in condition_lower:
                    expr = {"call": {"name": "system.weekday"}}
                    return _create_predicate(expr, condition_text)
                else:
                    expr = {"call": {"name": "system.weekend"}}
                    return _create_predicate(expr, condition_text)
            
            # Default: no reliable fallback
            return None
            
        except Exception as e:
            print(f"⚠️ Error creating fallback condition: {e}")
            return None

    def _create_goal_from_description(self, description: str, is_sub_goal: bool = False) -> Optional[BaseGoal]:
        """Shared method to create goals from descriptions (used by both _setup_smart_goals and _create_sub_goal)"""
        description_lower = description.lower().strip()
        prefix = "action" if is_sub_goal else "goal"
        
        # Detect press goals first (keyboard key presses)
        press_patterns = ["press enter", "press tab", "press escape", "press ctrl", "press alt", "press shift", "press f1", "press f2", "press f3", "press f4", "press f5", "press f6", "press f7", "press f8", "press f9", "press f10", "press f11", "press f12", "press space", "press backspace", "press delete", "press home", "press end", "press pageup", "press pagedown", "press up", "press down", "press left", "press right"]
        if any(pattern in description_lower for pattern in press_patterns):
            target_keys = extract_press_target(description)
            if target_keys:
                press_goal = PressGoal(
                    description=f"Press {prefix}: {description}",
                    target_keys=target_keys
                )
                return press_goal
        
        # Detect scroll goals
        scroll_patterns = [
            # Vertical scroll patterns
            "scroll down", "scroll up", "scroll to", "scroll by", "scroll a bit", "scroll slightly", 
            "scroll a page", "scroll one page", "scroll half", "scroll a third", "scroll a quarter",
            "scroll two thirds", "scroll three quarters", "scroll to top", "scroll to bottom",
            "scroll to middle", "scroll to center", "scroll down a bit", "scroll up a bit",
            "scroll down slightly", "scroll up slightly", "scroll down a page", "scroll up a page",
            # Horizontal scroll patterns
            "scroll left", "scroll right", "scroll left a bit", "scroll right a bit",
            "scroll left slightly", "scroll right slightly", "scroll left a page", "scroll right a page",
            "scroll left half", "scroll right half", "scroll left a third", "scroll right a third",
            "scroll left a quarter", "scroll right a quarter", "scroll left two thirds", "scroll right two thirds",
            "scroll left three quarters", "scroll right three quarters", "scroll to left edge", "scroll to right edge",
            "scroll to left middle", "scroll to right middle", "scroll to left center", "scroll to right center"
        ]
        if any(pattern in description_lower for pattern in scroll_patterns):
            scroll_goal = ScrollGoal(
                description=f"Scroll {prefix}: {description}",
                user_request=description
            )
            return scroll_goal

        # Detect back navigation goals (before general navigation detection)
        back_patterns = [
            "go back", "back", "previous page", "prev page", "back to", "go back to", "return to"
        ]
        if any(pattern in description_lower for pattern in back_patterns):
            url_history = list(self.goal_monitor.url_history or []) if self.goal_monitor else []
            pointer = getattr(self.goal_monitor, 'url_pointer', None)
            state_history = getattr(self.goal_monitor, 'state_history', None)
            target = resolve_back_target(description, url_history, pointer, state_history, self.page.url if self.page else None)
            steps_back = target.get("steps_back") or 1
            expected_url = target.get("expected_url")
            expected_title_substr = target.get("expected_title_substr")
            # Capture baseline index/url at goal creation for reliable multi-step validation
            try:
                pointer = getattr(self.goal_monitor, 'url_pointer', None)
                start_index = pointer if pointer is not None else (len(self.goal_monitor.url_history) - 1 if self.goal_monitor and self.goal_monitor.url_history else 0)
                start_url = self.goal_monitor.url_history[start_index] if self.goal_monitor and self.goal_monitor.url_history and 0 <= start_index < len(self.goal_monitor.url_history) else (self.page.url if self.page else "")
            except Exception:
                start_index, start_url = 0, (self.page.url if self.page else "")

            back_goal = BackGoal(
                description=f"Back {prefix}: {description}",
                expected_url=expected_url,
                steps_back=steps_back,
                expected_title_substr=expected_title_substr,
                start_index=start_index,
                start_url=start_url,
                needs_detection=False
            )
            return back_goal

        # Detect forward navigation goals
        forward_patterns = [
            "go forward", "forward", "next page", "go forward to"
        ]
        if any(pattern in description_lower for pattern in forward_patterns):
            url_history = list(self.goal_monitor.url_history or []) if self.goal_monitor else []
            pointer = getattr(self.goal_monitor, 'url_pointer', None)
            state_history = getattr(self.goal_monitor, 'state_history', None)
            target = resolve_forward_target(description, url_history, pointer, state_history, self.page.url if self.page else None)
            steps_forward = target.get("steps_forward") or 1
            expected_url = target.get("expected_url")

            # Baseline should be the current pointer
            try:
                pointer = getattr(self.goal_monitor, 'url_pointer', None)
                start_index = pointer if pointer is not None else (len(self.goal_monitor.url_history) - 1 if self.goal_monitor and self.goal_monitor.url_history else 0)
                start_url = self.goal_monitor.url_history[start_index] if self.goal_monitor and self.goal_monitor.url_history and 0 <= start_index < len(self.goal_monitor.url_history) else (self.page.url if self.page else "")
            except Exception:
                start_index, start_url = 0, (self.page.url if self.page else "")

            forward_goal = ForwardGoal(
                description=f"Forward {prefix}: {description}",
                expected_url=expected_url,
                steps_forward=steps_forward,
                start_index=start_index,
                start_url=start_url,
                needs_detection=False
            )
            return forward_goal
        
        # Detect click goals (more specific patterns, excluding press)
        click_patterns = ["click", "tap", "select", "choose", "close"]
        if any(pattern in description_lower for pattern in click_patterns):
            target_description = extract_click_target(description)
            if target_description:
                click_goal = ClickGoal(
                    description=f"Click {prefix}: {description}",
                    target_description=target_description
                )
                return click_goal
        
        # Detect form filling goals (after click goals to avoid conflicts)
        form_patterns = [
            "fill", "complete", "submit", "form", "enter", "input", "set the", "set only", "type", "search"
        ]
        
        if any(pattern in description_lower for pattern in form_patterns):
            form_goal = FormFillGoal(
                description=f"Form fill {prefix}: {description}",
                trigger_on_submit=False,
                trigger_on_field_input=True
            )
            return form_goal
        
        # Detect navigation goals
        navigation_patterns = ["go to", "navigate to", "open", "visit"]
        if any(pattern in description_lower for pattern in navigation_patterns):
            navigation_intent = extract_navigation_intent(description)
            if navigation_intent:
                nav_goal = NavigationGoal(
                    description=f"Navigation {prefix}: {description}",
                    navigation_intent=navigation_intent
                )
                return nav_goal
        
        # If no specific goal type is detected, create a generic action goal
        class ActionGoal(BaseGoal):
            def __init__(self, description: str):
                super().__init__(description)
            
            def evaluate(self, context):
                from goals.base import GoalResult
                return GoalResult(
                    status=GoalStatus.ACHIEVED,
                    confidence=0.8,
                    reasoning=f"Action completed: {self.description}"
                )
            
            def get_description(self, context):
                return f"Action: {self.description}"
        
        return ActionGoal(description)

    def _create_sub_goal(self, action_description: str):
        """Create a sub-goal from an action description using shared goal creation logic"""
        return self._create_goal_from_description(action_description, is_sub_goal=True)

    def _evaluate_conditional_goal_immediately(self, conditional_goal) -> tuple[Optional[BaseGoal], str]:
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
            conditional_goal.evaluate(basic_context)
            active_sub_goal = getattr(conditional_goal, '_current_sub_goal', None)
            condition_result = getattr(conditional_goal, '_last_condition_result', None)
            
            if active_sub_goal:
                print("🔍 DEBUG: Conditional goal evaluated immediately:")
                print(f"   📋 Condition result: {condition_result}")
                print(f"   🎯 Active sub-goal: {active_sub_goal.__class__.__name__} - '{active_sub_goal.description}'")
                # If this was a no-op fail branch (no explicit fail action), skip activating any goal
                if getattr(active_sub_goal, '_is_noop', False):
                    print("   📝 No-op fail branch (no fail action specified) → skipping goal activation")
                    return None, ""

                # Extract the specific action from the sub-goal description
                # Remove the prefix like "Click action:" or "Form fill action:" to get the clean action
                sub_goal_desc = active_sub_goal.description
                updated_description = sub_goal_desc
                
                print(f"   📝 Updated goal description: '{updated_description}'")
                return active_sub_goal, updated_description
            else:
                print("⚠️ Conditional goal evaluation failed, no active sub-goal")
                return None, ""
                
        except Exception as e:
            print(f"⚠️ Error evaluating conditional goal immediately: {e}")
            return None, ""

    def _get_detected_elements_for_goal(self, goal: BaseGoal, page_info) -> str:
        """Get detected elements for a specific goal by running element detection"""
        try:
            # Check if this goal needs element detection
            if not goal.needs_detection:
                return "No element detection needed for this goal type"
            
            # Take a screenshot for element detection (lower quality for speed)
            screenshot = self.page.screenshot(type="jpeg", quality=35, full_page=False)
            
            # Create numbered overlays and get element data
            element_data = self.overlay_manager.create_numbered_overlays(page_info)
            
            if not element_data:
                return "No interactive elements found"
            
            # Use AI to identify relevant elements for this specific goal
            detected_elements = self.element_detector.detect_elements_with_overlays(
                goal.description, "", screenshot, element_data, page_info
            )
            
            # Clean up overlays after detection
            self.overlay_manager.remove_overlays()
            
            if not detected_elements or not detected_elements.elements:
                return "No elements detected for this goal"
            
            # Format the detected elements
            formatted_elements = []
            for i, element in enumerate(detected_elements.elements):
                element_info = f"Element {i}: {element.description} ({element.element_type})"
                if hasattr(element, 'box_2d') and element.box_2d:
                    element_info += f" at {element.box_2d}"
                formatted_elements.append(element_info)
            
            return "\n".join(formatted_elements)
            
        except Exception as e:
            print(f"⚠️ Error detecting elements for goal: {e}")
            return "Error detecting elements"

    def _format_elements_for_prompt(self, elements) -> str:
        """Format detected elements for AI prompt"""
        if not elements or not hasattr(elements, 'elements'):
            return "No elements detected"
        
        formatted_elements = []
        for i, element in enumerate(elements.elements):
            element_info = f"Element {i}: {element.description} ({element.element_type})"
            if hasattr(element, 'box_2d') and element.box_2d:
                element_info += f" at {element.box_2d}"
            formatted_elements.append(element_info)
        
        return "\n".join(formatted_elements) if formatted_elements else "No elements detected"

    def _print_goal_summary(self) -> None:
        """Print a summary of all goal statuses"""
        summary = self.goal_monitor.get_status_summary()
        
        print("\n📊 Goal Summary:")
        print(f"   Total Goals: {summary['total_goals']}")
        print(f"   ✅ Achieved: {summary['achieved']}")
        print(f"   ⏳ Pending: {summary['pending']}")
        print(f"   ❌ Failed: {summary['failed']}")

    # Convenience methods
    def click_element(self, description: str) -> bool:
        return self.act(f"Click the {description}")

    def fill_form_field(self, field_name: str, value: str) -> bool:
        return self.act(f"Fill the {field_name} field with '{value}'")


# Example usage
if __name__ == "__main__":
    bot = BrowserVisionBot()
    bot.start()
    bot.goto("https://google.com/")
    bot.on_new_page_load(["if: cookie banner visible then: click accept cookies button"])
    
    bot.act("form: fill the search bar with 'nuro ai'")
    bot.act("press: enter")
    bot.act("click: the nuro ai link")
    bot.act("while: we can't see the phrase 'Proven L4 Autonomy' do: scroll down a bit")
    # bot.act("scroll down")
    # bot.act("click the 'company' link")
    
    
    while True:
        user_input = input('\n👤 New task or "q" to quit: ')
        if user_input.lower() == "q":
            break
        bot.act(user_input)
