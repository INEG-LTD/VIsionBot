"""
Core executor initialization and plan execution logic.
"""
import time
import random
from dataclasses import dataclass
from typing import Tuple, Optional, List, Callable
from enum import Enum

from playwright.sync_api import Page
from models import ActionStep, ActionType, PageElements, PageInfo
from execution.handlers import UploadHandler
from utils import SelectorUtils
from unittest.mock import Mock
from utils.page_utils import PageUtils
from utils.context_guard import ContextGuard, GuardDecision
from core.session import SessionTracker, InteractionType
from execution.ledger import ActionLedger
from utils.debug_print import dprint, PrintMode


class ScrollReason(Enum):
    """Enum for different scroll reasons"""
    USER_ACTION = "user_action"  # User explicitly requested a scroll action
    DUPLICATE_REJECTION = "duplicate_rejection"  # Scrolled due to repeated duplicate element detection
    DOM_UNCHANGED = "dom_unchanged"  # Scrolled because DOM signature hasn't changed
    EXPLORE_CONTENT = "explore_content"  # Scrolled to explore more content after successful plan execution
    MANUAL = "manual"  # Default for programmatic scrolls


@dataclass
class PreActionContext:
    """Complete context information passed to pre-action callbacks"""
    action_type: ActionType
    step: ActionStep
    page_info: PageInfo
    elements: PageElements
    coordinates: Optional[Tuple[int, int]] = None
    page: Optional[Page] = None  # Access to the page for custom actions
    action_id: Optional[str] = None  # ID of the action that triggered this action
    action_lineage: Optional[List[str]] = None  # Full lineage of action IDs


@dataclass
class PostActionContext:
    """Complete context information passed to post-action callbacks"""
    action_type: ActionType
    success: bool
    step: ActionStep
    page_info: PageInfo
    elements: PageElements
    coordinates: Optional[Tuple[int, int]] = None
    error_message: Optional[str] = None
    page: Optional[Page] = None  # Access to the page for custom actions
    action_id: Optional[str] = None  # ID of the action that triggered this action
    action_lineage: Optional[List[str]] = None  # Full lineage of action IDs
    scroll_occurred: bool = False  # Whether a scroll happened
    scroll_reason: Optional[ScrollReason] = None  # Reason for scroll (see ScrollReason enum)


class Executor:
    """Executes automation actions"""

    def __init__(self, page: Page, session_tracker: SessionTracker, page_utils:PageUtils=None, action_ledger: ActionLedger=None, preferred_click_method: str = "programmatic", execute_action_callback: Optional[Callable[[str], bool]] = None, user_messages_config=None):
        self.page = page
        self.session_tracker = session_tracker
        self.page_utils = page_utils
        self.action_ledger = action_ledger or ActionLedger()
        self.execute_action_callback = execute_action_callback  # Callback to execute actions through bot infrastructure
        self.last_failure_reason: Optional[str] = None
        self.user_messages_config = user_messages_config  # Store user messages config

        # Click method configuration
        if preferred_click_method not in ["programmatic", "mouse"]:
            raise ValueError(f"preferred_click_method must be 'programmatic' or 'mouse', got '{preferred_click_method}'")
        self.preferred_click_method = preferred_click_method  # "programmatic" or "mouse"

        # Scroll tracking
        self.last_scroll_occurred: bool = False
        self.last_scroll_reason: Optional[ScrollReason] = None

        # Click retry tracking: track previous clicks to detect when page hasn't changed
        self.last_click_selector: Optional[str] = None
        self.last_click_url: Optional[str] = None
        self.last_click_dom_signature: Optional[str] = None
        self.last_click_method: str = preferred_click_method

        # Initialize specialized handlers
        # DateTimeHandler may still need goal_monitor for compatibility, but we'll update it
        # For now, pass session_tracker instead
        try:
            from execution.handlers.datetime import DateTimeHandler
            self.datetime_handler = DateTimeHandler(page, session_tracker)
        except Exception:
            # Fallback if DateTimeHandler hasn't been updated yet
            self.datetime_handler = None
        self.upload_handler = UploadHandler(page, user_messages_config=user_messages_config)
        self.selector_utils = SelectorUtils(page)
        # ContextGuard doesn't need element_analyzer anymore
        self.context_guard = ContextGuard(page, None)

        # Vision-assisted refinements
        self.enable_vision_tag_hint: bool = True

        # Get event logger directly
        from utils.event_logger import get_event_logger
        self.event_logger = get_event_logger()

        # Ensure event_logger is never None - create a dummy one if needed
        if self.event_logger is None:
            from utils.event_logger import EventLogger
            self.event_logger = EventLogger(debug_mode=True, show_overlay_candidates=False)

        # Pre-action callback system
        self.pre_action_callbacks: List[Callable[[PreActionContext], None]] = []
        # Post-action callback system
        self.post_action_callbacks: List[Callable[[PostActionContext], None]] = []

        # Pause callback: Called between action steps within a plan
        # Why: Allows pausing between individual steps (e.g., click, type, scroll) within
        # a single plan execution. This provides even more granular control than pausing
        # between agent-determined actions. Useful for debugging complex multi-step plans.
        # The callback should handle pause checking and blocking internally.
        self._pause_callback: Optional[Callable[[], None]] = None

    def set_pause_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """
        Set a callback function to check for pause state between action steps.

        This callback is invoked between each step in a plan execution (e.g., between
        a click and a type action within the same plan). The callback should handle
        pause checking and blocking internally.

        Why between steps: Some plans contain multiple steps (e.g., "click field, then type").
        Pausing between steps allows inspection after each individual step completes,
        providing the most granular control possible.

        Args:
            callback: Function to call between action steps. Should handle pause checking
                    and blocking. Pass None to disable pause checking between steps.

        Example:
            >>> def check_pause():
            ...     if agent_controller.is_paused():
            ...         agent_controller._check_pause("action step")
            >>> action_executor.set_pause_callback(check_pause)
        """
        self._pause_callback = callback

    def _human_mouse_move(self, target_x: int, target_y: int, steps: int = 25):
        """
        Move mouse to target coordinates in a human-like curve (Bezier).
        """
        # Get current position
        # Playwright doesn't expose current mouse position directly easily without tracking,
        # but we can assume start from 0,0 or just jump if we don't know.
        # Better: just move from a random point near the last known position or just curve from "somewhere".
        # Since we can't easily get current pos, we'll just simulate the curve approach
        # by assuming we are at a random offset or just doing a direct curve.

        # Actually, we can just use steps to smooth it out even if we jump start.
        # But to be real, let's try to get current position or just start from a reasonable place.
        # For now, we'll just implement a simple smoothing function.

        # A simple way to simulate human movement is to use steps with some jitter.
        # But Playwright's mouse.move has steps!
        # self.page.mouse.move(x, y, steps=25) is already linear smoothing.
        # We want non-linear (Bezier).

        # Let's assume we are at (0,0) if we don't know, or we can track it.
        # For this implementation, we will rely on Playwright's internal tracking if we chain moves.

        # To make it look real, we need a control point for the Bezier curve.
        # We can't get start_x/y easily from Playwright API (it's stateless in that regard unless we track).
        # So we will just use Playwright's built-in steps with variable speed for now,
        # OR we can implement a "wind mouse" algorithm if we track state.

        # Let's stick to Playwright's steps but randomize the steps count and add jitter.

        # Randomize steps based on distance (if we knew it) or just random.
        actual_steps = random.randint(15, 35)
        self.page.mouse.move(target_x, target_y, steps=actual_steps)

        # Add a small random overshot or jitter at the end?
        # Maybe just a small random delay after move
        time.sleep(random.uniform(0.05, 0.15))


    def set_page(self, page: Page) -> None:
        """Update internal references when the active page changes."""
        if not page or page is self.page:
            return
        self.page = page
        if self.page_utils:
            if hasattr(self.page_utils, "set_page"):
                self.page_utils.set_page(page)
            else:
                self.page_utils.page = page
        if self.datetime_handler:
            if hasattr(self.datetime_handler, "set_page"):
                self.datetime_handler.set_page(page)
            else:
                self.datetime_handler.page = page
        if self.upload_handler:
            if hasattr(self.upload_handler, "set_page"):
                self.upload_handler.set_page(page)
            else:
                self.upload_handler.page = page
        if self.selector_utils:
            if hasattr(self.selector_utils, "set_page"):
                self.selector_utils.set_page(page)
            else:
                self.selector_utils.page = page
        if self.context_guard:
            self.context_guard.set_page(page, None)
        # Reset click tracking state for new page
        self.last_click_selector = None
        self.last_click_url = None
        self.last_click_dom_signature = None

    def execute_plan(
        self,
        plan,
        page_info: PageInfo,
        *,
        target_context_guard: Optional[str] = None,
        skip_post_guard_refinement: bool = True,
        confirm_before_interaction: bool = False,
        action_id: Optional[str] = None,
    ) -> bool:
        """
        Execute the generated plan

        Args:
            plan: The plan to execute
            page_info: Current page information
            target_context_guard: Guard condition for actions
            skip_post_guard_refinement: Skip refinement after guard checks
            confirm_before_interaction: Require user confirmation before actions
            action_id: Optional action ID for tracking (will use execution stack if not provided)

        Returns:
            True if plan executed successfully, False otherwise
        """
        # Import action implementations
        from .actions import (
            execute_click, execute_type, execute_scroll, execute_wait,
            execute_press, execute_open, execute_back, execute_forward,
            execute_stop, execute_upload, execute_datetime
        )

        self.event_logger.system_info(f"Executing plan with {len(plan.action_steps)} steps")
        try:
            self.event_logger.plan_execute_start(len(plan.action_steps))
        except Exception:
            pass

        # Note: Only evaluate goals AFTER actions execute, not before

        guard_text = (target_context_guard or "").strip()
        if guard_text:
            self.context_guard.reset_cache()

        self.last_failure_reason = None

        # Store action_id for use in action hooks
        self._current_plan_action_id = action_id

        for i, step in enumerate(plan.action_steps):
            # Check for pause between action steps
            # Why: Allows pausing between individual steps within a plan (e.g., between
            # click and type actions). This provides fine-grained control for debugging
            # complex multi-step plans.
            if self._pause_callback:
                try:
                    self._pause_callback()
                except Exception:
                    pass  # Don't let pause callback errors break execution

            try:
                self.event_logger.action_step(step_number=i+1, action_type=str(step.action))
            except Exception:
                pass
            try:
                self.event_logger.action_start(
                    action_type=str(step.action),
                    step_number=i + 1,
                    overlay_index=step.overlay_index,
                )
            except Exception:
                pass

            try:
                step_success = True
                allow_refinement = True
                if guard_text and ContextGuard.is_guarded_action(step.action):
                    decision = self.context_guard.validate(
                        step=step,
                        plan=plan,
                        page_info=page_info,
                        guard_text=guard_text,
                    )
                    if not decision.passed:
                        self._handle_context_guard_failure(i, step, guard_text, decision)
                        return False
                    elif skip_post_guard_refinement:
                        allow_refinement = False
                if step.action == ActionType.CLICK:
                    step_success = execute_click(
                        self, step, plan.detected_elements, page_info,
                        allow_refinement=allow_refinement,
                        confirm_before_interaction=confirm_before_interaction,
                    )
                elif step.action == ActionType.TYPE:
                    step_success = execute_type(
                        self, step, plan.detected_elements, page_info,
                        allow_refinement=allow_refinement,
                        confirm_before_interaction=confirm_before_interaction,
                    )
                elif step.action == ActionType.SCROLL:
                    step_success = execute_scroll(self, step)
                elif step.action == ActionType.WAIT:
                    step_success = execute_wait(self, step)
                elif step.action == ActionType.PRESS:
                    step_success = execute_press(self, step)
                elif step.action == ActionType.HANDLE_UPLOAD:
                    step_success = execute_upload(
                        self, step, plan.detected_elements, page_info,
                        confirm_before_interaction=confirm_before_interaction,
                    )
                elif step.action == ActionType.HANDLE_DATETIME:
                    step_success = execute_datetime(
                        self, step, plan.detected_elements, page_info,
                        confirm_before_interaction=confirm_before_interaction,
                    )
                elif step.action == ActionType.OPEN:
                    step_success = execute_open(
                        self, step,
                        confirm_before_interaction=confirm_before_interaction,
                    )
                elif step.action == ActionType.BACK:
                    step_success = execute_back(self)
                elif step.action == ActionType.FORWARD:
                    step_success = execute_forward(self)
                elif step.action == ActionType.STOP:
                    step_success = execute_stop(self, step)
                else:
                    dprint(f"⚠️ Unknown action type: {step.action}")
                    continue

                if step_success:
                    try:
                        self.event_logger.action_success(
                            action_type=str(step.action),
                            step_number=i + 1,
                            overlay_index=step.overlay_index,
                        )
                    except Exception:
                        pass
                else:
                    try:
                        self.event_logger.action_failure(
                            action_type=str(step.action),
                            error=self.last_failure_reason,
                            step_number=i + 1,
                            overlay_index=step.overlay_index,
                        )
                    except Exception:
                        pass
                # Check if step failed (e.g., due to retry request)
                if not step_success:
                    dprint(f"❌ Step {i+1} failed - aborting plan execution")
                    try:
                        self.event_logger.plan_execute_fail(self.last_failure_reason or "step_failed")
                    except Exception:
                        pass
                    return False

                # Goal checking removed - keyword goals handle completion directly

                # Small delay between actions
                time.sleep(0.5)

            except Exception as e:
                dprint(f"❌ Error executing step {i+1}: {e}")
                self.last_failure_reason = f"Error executing step {i+1}: {e}"
                try:
                    self.event_logger.plan_execute_fail(self.last_failure_reason)
                except Exception:
                    pass
                return False

        self.event_logger.system_info("Plan execution completed")
        try:
            self.event_logger.plan_execute_complete()
        except Exception:
            pass
        return True

    def _handle_context_guard_failure(
        self,
        step_index: int,
        step: ActionStep,
        guard_text: str,
        decision: GuardDecision,
    ) -> None:
        reason = decision.reason or "Context guard validation failed"
        self.last_failure_reason = reason
        dprint(
            f"🛑 Context guard blocked step {step_index + 1} ({step.action}). Reason: {reason}"
        )
        try:
            overlay_index = step.overlay_index
            self.session_tracker.record_interaction(
                InteractionType.CONTEXT_GUARD,
                target_element_info={
                    "overlay_index": overlay_index,
                    "action": step.action.value,
                    "guard": guard_text,
                    "cached": decision.cached,
                },
                success=False,
                error_message=reason,
            )
        except Exception:
            pass
