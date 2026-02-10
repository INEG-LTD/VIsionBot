"""
Core executor initialization and plan execution logic.
"""
import hashlib
import math
import re
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List, Callable, Type, Union
from enum import Enum

from playwright.sync_api import Page
from pydantic import BaseModel, Field, create_model
from agent.agent_context import EnvironmentState
from agent.notebook import Notebook
from core.browser import Browser
from core.config import Config
from core.executor.ui_feedback import highlight_click_location
from models import ActionStep, ActionType, PageElements, PageInfo
from models.models import FailedAction
from utils import SelectorUtils
from utils.intent_parsers import parse_keyword_command
from utils.page_utils import PageUtils
from core.session import SessionTracker, InteractionType
from utils.debug_print import dprint
from execution.result import ActionResult
from lib.ai import generate_text
from lib.ai import generate_text, generate_model
from core.session import InteractionType as IT

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

    def __init__(self, 
                 browser: Browser, 
                 session_tracker: SessionTracker, 
                 notebook: Notebook,
                 page_utils:PageUtils=None, 
                 preferred_click_method: str = "programmatic", 
                 user_question_callback: Optional[Callable[[str, dict], str]] = None, 
                 agent_talk_callback: Optional[Callable[[str], None]] = None, 
                 user_messages_config=None):
        self.browser = browser
        self.session_tracker = session_tracker
        self.page_utils = page_utils
        self.last_failure_reason: Optional[str] = None
        self.user_messages_config = user_messages_config  # Store user messages config
        self.user_question_callback = user_question_callback
        self.agent_talk_callback = agent_talk_callback
        self.notebook = notebook  # Optional notebook for storing extraction results
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

        self.selector_utils = SelectorUtils(browser.page)

        # Vision-assisted refinements
        self.enable_vision_tag_hint: bool = True

        # Get event logger directly
        from utils.event_logger import get_event_logger
        self.event_logger = get_event_logger()

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
        self.command_history: List[str] = []

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
        # self.browser.page.mouse.move(x, y, steps=25) is already linear smoothing.
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
        self.browser.page.mouse.move(target_x, target_y, steps=actual_steps)

        # Add a small random overshot or jitter at the end?
        # Maybe just a small random delay after move
        time.sleep(random.uniform(0.05, 0.15))

    def set_page(self, page: Page) -> None:
        """Update internal references when the active page changes."""
        if not page or page is self.browser.page:
            return
        self.browser.page = page
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
        # Reset click tracking state for new page
        self.last_click_selector = None
        self.last_click_url = None
        self.last_click_dom_signature = None
     
    def execute_forward(self, step: ActionStep) -> bool:
        """Navigate forward in browser history and record interaction."""
        # Capture state BEFORE navigation for accurate before_state
        before_url = ""
        before_state = None
        try:
            before_url = self.browser.page.url
            before_state = self.session_tracker._capture_current_state()
        except Exception:
            pass

        num_forward = step.action.split(":", 1)[1].strip()
        if num_forward.isdigit():
            num_forward = int(num_forward)
        else:
            num_forward = 1

        for _ in range(num_forward):
            try:
                self.browser.page.go_forward()
                success = True
                error_msg = None
            except Exception as e:
                success = False
                error_msg = str(e)
                dprint(f"  ❌ Forward navigation failed: {e}")

        # Get URL after navigation
        after_url = before_url  # Default to before_url if navigation failed
        if success:
            try:
                after_url = self.browser.page.url
                after_state = self.session_tracker._capture_current_state()
            except Exception:
                pass
        
        self.event_logger.command_execution_complete("FORWARD", success=success, target_description=after_url, reasoning=None)
        
        # Record navigation interaction with explicit before_state
        try:
            self.session_tracker.record_interaction(
                InteractionType.NAVIGATION,
                before_state=before_state,  # Pass explicit before_state since navigation already happened
                after_state=after_state,
                target_element_info={"direction": "forward", "from": before_url, "to": after_url},
                success=success,
                error_message=error_msg,
            )
        except Exception:
            pass
        return success

    def execute_click(
        self,
        step: ActionStep,
        elements: PageElements,
        failed_actions: List[str],
        failed_elements: List[FailedAction],
        page_info: PageInfo,
        base_knowledge: Optional[List[str]] = None,
    ) -> bool:
        """Execute a click action"""
        # Figure out the overlay to use for the click
        # Capture state BEFORE performing the click (critical for accurate before_state)
        before_state = self.session_tracker._capture_current_state()
        
        current_screenshot = before_state.screenshot
        overlay_index = self.select_best_overlay(step.action, elements, failed_elements, screenshot=current_screenshot, base_knowledge=self.session_tracker.base_knowledge)
        
        if overlay_index is None:
            self.event_logger.command_failure(step.action, error="Could not determine best overlay")
            return False
        x, y = self.get_click_coordinates(overlay_index, elements, page_info)

        if x is None or y is None:
            self.event_logger.command_failure(step.action, error="Could not determine click coordinates")
            return False
        
        try:
            # Small delay to show the highlight
            time.sleep(0.3)
        except Exception as e:
            # Continue even if highlight fails
            try:
                self.event_logger.system_debug(f"Could not highlight click location: {e}")
            except Exception:
                pass
    
        try:
            self.event_logger.system_info(f"Clicking at ({x}, {y}) using mouse click")
        except Exception:
            pass

        click_executed = False
        success = False

        try:
            highlight_click_location(self, x, y)
            # self._human_mouse_move(x, y)
            self.browser.page.mouse.click(x, y)
            click_executed = True
            print(f"Click executed: {click_executed}")
            
            # Capture state after click
            after_state = self.session_tracker._capture_current_state()

        except Exception as e:
            self.event_logger.command_failure(step.action, error=f"An error occured while clicking: {e}")
            return False

        success = True
        # if not click_executed:
        #     self.event_logger.command_failure(step.action, error="An error occured while clicking")
        #     success = False
        # else:
        #     # Detect if something meaningful changed
        #     state_changed = self.session_tracker.detect_state_change(before_state, after_state)

        #     if state_changed:
        #         success = True
        #     else:
        #         success = False
        #         try:
        #             self.event_logger.command_failure(command=step.action, error=f"Click at ({x}, {y}) executed but caused no visible page change")
        #         except Exception:
        #             pass

        # Build target description from step information
        target_description = None
        if step.action:
            # Extract target from action string (e.g., "click: Submit button" -> "Submit button")
            action_str = step.action
            if ":" in action_str:
                target_description = action_str.split(":", 1)[1].strip()
        print(f"Target description: {target_description}")
        if not target_description and overlay_index is not None:
            target_description = f"element #{overlay_index}"
        print(f"Target description: {target_description}")
        # Get reasoning if available; fall back to session_tracker's current action reasoning
        step_reasoning = step.reasoning
        if not step_reasoning:
            try:
                step_reasoning = self.session_tracker.get_current_action_reasoning()
            except Exception:
                step_reasoning = None
        print(f"Step reasoning: {step_reasoning}")
        # Record actual interaction with goal monitor (pass explicit before_state since click already happened)
        self.session_tracker.record_interaction(
            InteractionType.CLICK,
            before_state=before_state,  # Pass explicit before_state captured before the click
            after_state=after_state,
            coordinates=(x, y),
            target_element_info={
                "description": target_description,
                "overlay_index": overlay_index,
                "action": step.action,
            } if target_description or overlay_index else None,
            reasoning=step_reasoning,  # Include WHY this action was taken
            success=success,
        )
        
        return success

    def execute_ask(
        self,
        step: ActionStep,
        environment_state: EnvironmentState,
        base_knowledge: Optional[List[str]] = None,
    ) -> bool:
        def _handle_ask_command(
            question: str,
            environment_state: EnvironmentState
        ) -> Optional[str]:
            if not self.user_question_callback:
                self.event_logger.system_warning("Agent wants to ask a question but no callback configured")
                return None
            
            # Build context for the callback
            context = {
                "current_url": environment_state.current_url,
                "page_title": environment_state.page_title,
            }

            try:
                # Disable page blocking while asking user question
                if hasattr(self.browser, '_thinking_border_manager'):
                    self.browser._thinking_border_manager.disable_blocking()

                try:
                    answer = self.user_question_callback(question, context)
                finally:
                    # Re-enable page blocking after user responds
                    if hasattr(self.browser, '_thinking_border_manager'):
                        self.browser._thinking_border_manager.enable_blocking()
                
                if answer:
                    self.event_logger.ask_command_answered(
                        question=question,
                        response=answer,
                    )
                    return answer
                else:
                    self.event_logger.ask_command_skipped(question=question)
                    return None
            except Exception as e:
                self.event_logger.ask_command_failure(question=question, error=str(e), details=context)
                return None

        question = step.action.split(":", 1)[1].strip() if ":" in step.action else "Need assistance"
        try:
            self.event_logger.ask_requested(question)
        except Exception:
            pass

        # For now, return failure to indicate human intervention needed
        # In the future, this could pause and wait for user input
        try:
            before_state = self.session_tracker._capture_current_state()
            ask_result = _handle_ask_command(question, environment_state)
            after_state = self.session_tracker._capture_current_state()
        
            self.session_tracker.record_interaction(
                InteractionType.ASK,
                before_state=before_state,
                after_state=after_state,
                coordinates=None,
                target_element_info={
                    "question": question,
                },
                success=ask_result,
            )
            
            # Store question/answer pair for agent context if answer was received
            if ask_result:
                self.session_tracker.add_question_answer(question, ask_result)
            
        except Exception:
            pass
        
        return ask_result

    def execute_clear_text(
        self,
        step: ActionStep,
    ) -> bool:
        """Execute a clear text action"""
        before_state = self.session_tracker._capture_current_state()
        
        current_screenshot = before_state.screenshot
        elements = before_state.elements
        page_info = before_state.page_info
        failed_elements = before_state.failed_elements

        def clear_input_field(x: Optional[int], y: Optional[int]) -> str:
            """
            Clear an input field before typing to ensure previous text is removed.
            Tries multiple methods: JavaScript first, then keyboard select-all+delete.

            Returns:
                Status string: "js" if JS cleared, "keyboard" if keyboard fallback, "failed" if both failed
            """
            if x is None or y is None:
                return "failed"

            try:
                # Try to clear using JavaScript first (most reliable)
                element_js = f"""
                (function() {{
                    const element = document.elementFromPoint({x}, {y});
                    if (element && (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA')) {{
                        element.focus();
                        element.value = '';
                        element.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        element.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        return true;
                    }}
                    return false;
                }})();
                """
                cleared = self.browser.page.evaluate(element_js)
                if cleared:
                    # Only show in debug mode
                    if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
                        dprint(f"  ✅ Cleared field using JavaScript")
                    time.sleep(0.1)
                    return "js"
            except Exception as e:
                # Only show in debug mode
                if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
                    dprint(f"  ⚠️ JavaScript clear failed, using keyboard: {e}")

            # Fallback: click, select all, delete
            try:
                self.browser.page.mouse.click(x, y)
                time.sleep(0.2)
                self.browser.page.keyboard.press('Control+a')
                time.sleep(0.1)
                self.browser.page.keyboard.press('Delete')
                time.sleep(0.1)
                # Only show in debug mode
                if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
                    dprint(f"  ✅ Cleared field using keyboard (Ctrl+A, Delete)")
                return "keyboard"
            except Exception as e:
                # Only show in debug mode
                if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
                    dprint(f"  ⚠️ Keyboard clear failed: {e}")
                return "failed"

        overlay_index = self.select_best_overlay(
            step.action,
            elements,
            failed_elements,
            screenshot=current_screenshot,
            base_knowledge=self.session_tracker.base_knowledge
        )

        # Get coordinates for the element to type into
        x, y = self.get_click_coordinates(overlay_index, elements, page_info)
        
        # Click first to focus the element
        if x is not None and y is not None:
            self._human_mouse_move(x, y)
            self.browser.page.mouse.click(x, y)
            time.sleep(random.uniform(0.1, 0.3))
            
        try:
            # Always clear the field before typing to ensure previous text is removed
            clear_input_field(x, y)
        except Exception as e:
            success = False
            error_msg = str(e)
            
        # Build target description from step information
        target_description = None
        if step.action:
            # Extract target from action string (e.g., "type: text : field" -> "field")
            action_str = step.action
            if ":" in action_str:
                parts = action_str.split(":")
                if len(parts) >= 3:  # "type: text : field" format
                    target_description = parts[2].strip()
                elif len(parts) == 2:
                    target_description = parts[1].strip()

        if not target_description and overlay_index is not None:
            target_description = f"element #{overlay_index}"
            
        after_state = self.session_tracker._capture_current_state()
        # Record type interaction with goal monitor (pass explicit before_state since typing already happened)
        self.session_tracker.record_interaction(
            InteractionType.CLEAR_TEXT,
            before_state=before_state,  # Pass explicit before_state captured before the typing
            after_state=after_state,
            coordinates=(x, y) if x is not None and y is not None else None,
            target_element_info={
                "description": target_description,
                "overlay_index": overlay_index,
                "action": step.action,
            } if target_description or overlay_index else None,
            reasoning=None,
            success=success,
            error_message=error_msg,
        )
        
    def execute_type(
        self,
        step: ActionStep,
        elements: PageElements,
        failed_actions: List[str],
        failed_elements: List[FailedAction],
        page_info: PageInfo,
        *,
        base_knowledge: Optional[List[str]] = None,
    ) -> bool:
        """Execute a type action"""
        before_state = self.session_tracker._capture_current_state()
        
        current_screenshot = before_state.screenshot

        def clear_input_field(x: Optional[int], y: Optional[int]) -> str:
            """
            Clear an input field before typing to ensure previous text is removed.
            Tries multiple methods: JavaScript first, then keyboard select-all+delete.

            Returns:
                Status string: "js" if JS cleared, "keyboard" if keyboard fallback, "failed" if both failed
            """
            if x is None or y is None:
                return "failed"

            try:
                # Try to clear using JavaScript first (most reliable)
                element_js = f"""
                (function() {{
                    const element = document.elementFromPoint({x}, {y});
                    if (element && (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA')) {{
                        element.focus();
                        element.value = '';
                        element.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        element.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        return true;
                    }}
                    return false;
                }})();
                """
                cleared = self.browser.page.evaluate(element_js)
                if cleared:
                    # Only show in debug mode
                    if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
                        dprint(f"  ✅ Cleared field using JavaScript")
                    time.sleep(0.1)
                    return "js"
            except Exception as e:
                # Only show in debug mode
                if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
                    dprint(f"  ⚠️ JavaScript clear failed, using keyboard: {e}")

            # Fallback: click, select all, delete
            try:
                self.browser.page.mouse.click(x, y)
                time.sleep(0.2)
                self.browser.page.keyboard.press('Control+a')
                time.sleep(0.1)
                self.browser.page.keyboard.press('Delete')
                time.sleep(0.1)
                # Only show in debug mode
                if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
                    dprint(f"  ✅ Cleared field using keyboard (Ctrl+A, Delete)")
                return "keyboard"
            except Exception as e:
                # Only show in debug mode
                if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
                    dprint(f"  ⚠️ Keyboard clear failed: {e}")
                return "failed"

        # Extract the text from the type command. eg type: text : field
        split_action = step.action.split(":")
        text_to_type = split_action[1].strip()

        if not text_to_type:
            dprint("⚠️ No text specified for TYPE action")
            return False
        overlay_index = self.select_best_overlay(
            step.action,
            elements,
            failed_elements,
            screenshot=current_screenshot,
            base_knowledge=self.session_tracker.base_knowledge
        )

        # Get coordinates for the element to type into
        x, y = self.get_click_coordinates(overlay_index, elements, page_info)
        
        # Click first to focus the element
        if x is not None and y is not None:
            self._human_mouse_move(x, y)
            self.browser.page.mouse.click(x, y)
            time.sleep(random.uniform(0.1, 0.3))

        try:
            self.event_logger.system_debug(f"Typing: {text_to_type}")
        except Exception:
            pass

        clear_status = "skipped"
        try:
            # Always clear the field before typing to ensure previous text is removed
            clear_status = clear_input_field(x, y)

            # Try to get element selector and use fill() or press_sequentially
            element_selector = None
            if x is not None and y is not None:
                try:
                    element_selector = self.selector_utils.get_element_selector_from_coordinates(x, y)
                    if element_selector:
                        try:
                            self.event_logger.system_debug(f"Using fill() method with selector: {element_selector}")
                        except Exception:
                            pass
                        # Use locator for more reliable filling
                        locator = self.browser.page.locator(element_selector).first
                        # Type with random delay between keystrokes if text is short, otherwise fill
                        # Note: fill() automatically clears, but we already cleared above for consistency
                        if len(text_to_type) < 50:
                            locator.press_sequentially(text_to_type, delay=random.randint(50, 150))
                        else:
                            locator.fill(text_to_type)
                        success = True
                        error_msg = None
                    else:
                        raise ValueError("Could not get element selector")
                except Exception as e:
                    # Only show in debug mode
                    if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
                        self.event_logger.system_debug(f"fill() method failed, falling back to keyboard: {e}")
                    element_selector = None

            # Fallback to keyboard method if fill() didn't work
            if not element_selector:
                used_keyboard_fallback = True
                # Ensure element is focused before keyboard typing
                if x is not None and y is not None:
                    try:
                        self.browser.page.mouse.click(x, y)
                        time.sleep(0.1)
                    except Exception:
                        pass
                # Field was already cleared above, so just type the new text
                self.browser.page.keyboard.type(text_to_type, delay=50)
                success = True
                error_msg = None
            else:
                used_keyboard_fallback = False
        except Exception as e:
            used_keyboard_fallback = False
            success = False
            error_msg = str(e)
            self.event_logger.command_failure(command=step.action, error=f"Typing failed: {e}")

        # Build executor feedback notes
        type_notes_parts = []
        if clear_status == "failed":
            type_notes_parts.append("clear failed — field may still have old text")
        if used_keyboard_fallback:
            type_notes_parts.append("fill() failed, used keyboard fallback")
        if not success:
            type_notes_parts.append(f"typing failed: {error_msg}" if error_msg else "typing failed")
        type_notes = "; ".join(type_notes_parts) if type_notes_parts else None

        # Build target description from step information
        target_description = None
        if step.action:
            # Extract target from action string (e.g., "type: text : field" -> "field")
            action_str = step.action
            if ":" in action_str:
                parts = action_str.split(":")
                if len(parts) >= 3:  # "type: text : field" format
                    target_description = parts[2].strip()
                elif len(parts) == 2:
                    target_description = parts[1].strip()

        if not target_description and overlay_index is not None:
            target_description = f"element #{overlay_index}"

        # Get reasoning if available; fall back to session_tracker's current action reasoning
        step_reasoning = getattr(step, 'reasoning', None)
        if not step_reasoning:
            try:
                step_reasoning = self.session_tracker.get_current_action_reasoning()
            except Exception:
                step_reasoning = None
        after_state = self.session_tracker._capture_current_state()
        # Record type interaction with goal monitor (pass explicit before_state since typing already happened)
        self.session_tracker.record_interaction(
            InteractionType.TYPE,
            before_state=before_state,  # Pass explicit before_state captured before the typing
            after_state=after_state,
            coordinates=(x, y) if x is not None and y is not None else None,
            text_input=text_to_type,
            target_element_info={
                "description": target_description,
                "overlay_index": overlay_index,
                "action": step.action,
            } if target_description or overlay_index else None,
            reasoning=step_reasoning,  # Include WHY this action was taken
            success=success,
            error_message=error_msg,
            notes=type_notes,
        )

        try:
            self.event_logger.system_debug(f"Success: {success}")
        except Exception:
            pass

        return success

    def execute_scroll(self, step: ActionStep) -> bool:
        """
        Execute a scroll action via a straightforward Playwright scroll call.

        Args:
            step: ActionStep containing scroll direction and parameters

        Returns:
            bool: True if scroll succeeded, False otherwise
        """
        direction = step.action.split(":", 1)[1].strip() or "down"
        axis = "vertical"
        current_scroll_x = int(self.browser.page.evaluate("window.pageXOffset || window.scrollX") or 0)
        current_scroll_y = int(self.browser.page.evaluate("window.pageYOffset || window.scrollY") or 0)

        if direction == "down":
            target_x = current_scroll_x
            target_y = min(current_scroll_y + 300, 9999)
        elif direction == "up":
            target_x = current_scroll_x
            target_y = max(current_scroll_y - 300, 0)
        elif direction == "right":
            target_x = min(current_scroll_x + 300, 9999)
            target_y = current_scroll_y
            axis = "horizontal"
        else:
            target_x = max(current_scroll_x - 300, 0)
            target_y = current_scroll_y
            axis = "horizontal"
        if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
            self.event_logger.system_debug(f"[Executor] Using default scroll: target position ({target_x}, {target_y}) {direction} ({axis})")
        target_x = int(target_x)
        target_y = int(target_y)

        if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
            self.event_logger.system_debug(f"  Scrolling to position ({target_x}, {target_y}) {direction} ({axis})")

        before_state = self.session_tracker._capture_current_state()
        success = False
        error_msg = None
        try:
            scroll_amount_y = target_y - current_scroll_y
            scroll_amount_x = target_x - current_scroll_x
            if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
                self.event_logger.system_debug(f"🔍 [Executor] Attempting to scroll {direction} by ({scroll_amount_x}, {scroll_amount_y})px")
            self.browser.page.evaluate(f"window.scrollBy({scroll_amount_x}, {scroll_amount_y})")
            if self.page_utils:
                actual_scroll_y = int(self.browser.page.evaluate("window.pageYOffset || window.scrollY") or 0)
                actual_scroll_x = int(self.browser.page.evaluate("window.pageXOffset || window.scrollX") or 0)
                self.page_utils.last_scroll_y = actual_scroll_y
                self.page_utils.last_scroll_x = actual_scroll_x
            success = True
        except Exception as exc:
            error_msg = str(exc)
            success = False
            self.event_logger.system_error(f"Scroll failed: {exc}")
        after_state = self.session_tracker._capture_current_state()

        self.session_tracker.record_interaction(
            InteractionType.SCROLL,
            before_state=before_state,
            after_state=after_state,
            target_x=target_x,
            target_y=target_y,
            scroll_direction=direction,
            scroll_axis=axis,
            success=success,
            error_message=error_msg,
        )

        return success

    def execute_press(self, step: ActionStep) -> bool:
        """Execute a key press action"""
        step_keys: List[str] = step.keys_to_press
        if not step_keys:
            # Extract the keys from the action string
            step_keys: List[str] = step.action.split(":", 1)[1].strip()
            if not step_keys:
                self.event_logger.command_failure(command=step.action, error="No keys specified for PRESS action")
                return False
            step_keys: List[str] = step_keys.split(",")
            step_keys = [k.strip() for k in step_keys]

        try:
            self.event_logger.system_debug(f"Pressing keys: {step_keys}")
        except Exception:
            pass

        # Capture state BEFORE performing the press action (critical for accurate before_state)
        before_state = self.session_tracker._capture_current_state()
        for key in step_keys:
            try:
                # Parse and execute the key combination
                self.parse_and_press_keys(key)
            except Exception as e:
                after_state = self.session_tracker._capture_current_state()
                self.event_logger.command_failure("PRESS", error=f"Key press failed: {e}")
                self.session_tracker.record_interaction(
                    InteractionType.PRESS,
                    before_state=before_state,  # Pass explicit before_state captured before the press
                    after_state=after_state,
                    keys_pressed=step_keys,
                    success=False,
                    error_message=str(e),
                    )
                return False
        after_state = self.session_tracker._capture_current_state()

        self.event_logger.command_execution_complete("PRESS", success=True, keys_pressed=step.keys_to_press)
        self.session_tracker.record_interaction(
            InteractionType.PRESS,
            before_state=before_state,  # Pass explicit before_state captured before the press
            after_state=after_state,
            keys_pressed=step_keys,
            success=True,
            error_message=None,
        )
        
        return True


    def parse_and_press_keys(self, keys_string: str) -> None:
        """Parse a key string and execute the key press(es)"""
        # Normalize the key string
        keys_string = keys_string.lower().strip()

        # Handle comma-separated keys (multiple keys to press sequentially)
        if ',' in keys_string:
            keys_list = [k.strip() for k in keys_string.split(',')]
            for key in keys_list:
                self.parse_and_press_keys(key)  # Recursively handle each key
            return

        # Normalize common key name variations
        # Handle "arrow_down", "arrowdown", "arrow_down" -> "down"
        if keys_string.startswith('arrow'):
            # Remove "arrow" prefix and normalize
            key_part = keys_string.replace('arrow', '').replace('_', '').strip()
            if key_part in ['up', 'down', 'left', 'right']:
                keys_string = key_part

        # Handle common key combinations
        if '+' in keys_string:
            # Handle key combinations like "ctrl+c", "cmd+enter", etc.
            parts = keys_string.split('+')
            modifiers = parts[:-1]
            main_key = parts[-1]

            # Map common modifier names
            modifier_map = {
                'ctrl': 'Control',
                'control': 'Control',
                'cmd': 'Meta',
                'command': 'Meta',
                'meta': 'Meta',
                'alt': 'Alt',
                'option': 'Alt',
                'shift': 'Shift'
            }

            # Map common key names
            key_map = {
                'enter': 'Enter',
                'return': 'Enter',
                'tab': 'Tab',
                'space': ' ',
                'esc': 'Escape',
                'escape': 'Escape',
                'backspace': 'Backspace',
                'delete': 'Delete',
                'del': 'Delete',
                'up': 'ArrowUp',
                'down': 'ArrowDown',
                'left': 'ArrowLeft',
                'right': 'ArrowRight',
                'home': 'Home',
                'end': 'End',
                'pageup': 'PageUp',
                'pagedown': 'PageDown',
                'f1': 'F1',
                'f2': 'F2',
                'f3': 'F3',
                'f4': 'F4',
                'f5': 'F5',
                'f6': 'F6',
                'f7': 'F7',
                'f8': 'F8',
                'f9': 'F9',
                'f10': 'F10',
                'f11': 'F11',
                'f12': 'F12'
            }

            # Convert modifiers
            playwright_modifiers = []
            for mod in modifiers:
                playwright_mod = modifier_map.get(mod.strip())
                if playwright_mod:
                    playwright_modifiers.append(playwright_mod)

            # Convert main key
            playwright_key = key_map.get(main_key.strip(), main_key.strip())

            # Execute the key combination
            if playwright_modifiers:
                self.browser.page.keyboard.press(f"{'+'.join(playwright_modifiers)}+{playwright_key}")
            else:
                self.browser.page.keyboard.press(playwright_key)
        else:
            # Handle single keys
            key_map = {
                'enter': 'Enter',
                'return': 'Enter',
                'tab': 'Tab',
                'space': ' ',
                'esc': 'Escape',
                'escape': 'Escape',
                'backspace': 'Backspace',
                'delete': 'Delete',
                'del': 'Delete',
                'up': 'ArrowUp',
                'down': 'ArrowDown',
                'left': 'ArrowLeft',
                'right': 'ArrowRight',
                'home': 'Home',
                'end': 'End',
                'pageup': 'PageUp',
                'pagedown': 'PageDown',
                'f1': 'F1',
                'f2': 'F2',
                'f3': 'F3',
                'f4': 'F4',
                'f5': 'F5',
                'f6': 'F6',
                'f7': 'F7',
                'f8': 'F8',
                'f9': 'F9',
                'f10': 'F10',
                'f11': 'F11',
                'f12': 'F12'
            }

            playwright_key = key_map.get(keys_string, keys_string)
            self.browser.page.keyboard.press(playwright_key)

    def get_click_coordinates(self, overlay_index: int, elements: PageElements, page_info: PageInfo) -> Tuple[Optional[int], Optional[int]]:
        if overlay_index is None or not getattr(elements, "elements", None):
            return None, None

        w, h = page_info.width, page_info.height

        for el in elements.elements:
            box = getattr(el, "box_2d", None)
            if getattr(el, "overlay_number", None) != overlay_index or not box or len(box) != 4:
                continue

            y0, x0, y1, x1 = box
            m = max(abs(y0), abs(x0), abs(y1), abs(x1))

            # If 0..1 normalized, scale to 0..1000
            if m <= 1.0:
                y0, x0, y1, x1 = (v * 1000 for v in (y0, x0, y1, x1))
                m = 1000

            normalized = m <= 2000  # allow slack for oversized normalized values

            if normalized:
                if not (0 <= y0 <= 1000 and 0 <= x0 <= 1000 and 0 <= y1 <= 1000 and 0 <= x1 <= 1000 and y0 < y1 and x0 < x1):
                    continue
                cx, cy = int(((x0 + x1) / 2) / 1000 * w), int(((y0 + y1) / 2) / 1000 * h)
            else:
                if not (y0 < y1 and x0 < x1):
                    continue
                cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)

            if 0 <= cx < w and 0 <= cy < h:
                return cx, cy

        return None, None
    
    def execute_open(
        self,
        step: ActionStep,
    ) -> bool:
        """Open a URL directly in the current tab and record navigation."""
        # Seperate URL from action instruction
        # Robustly extract the URL from the action string (e.g., handle extra spaces and trailing quotes)
        try:
            before_state = self.session_tracker._capture_current_state()
            url = step.action.split(":", 1)[1].strip()
            # Remove trailing or leading single/double quotes if present
            if (url.startswith("'") and url.endswith("'")) or (url.startswith('"') and url.endswith('"')):
                url = url[1:-1].strip()
        except Exception:
            url = ""

        success = True
        error_message = None
        try:
            self.browser.page.goto(url, wait_until="domcontentloaded")
        except Exception as e:
            success = False
            error_message = str(e)
            self.event_logger.command_failure(command=step.action, error=f"Open navigation failed: {e}")

        after_state = self.session_tracker._capture_current_state()
        self.session_tracker.record_interaction(
            InteractionType.NAVIGATION,
            before_state=before_state,
            after_state=after_state,
            target_element_info={"url": url},
            navigation_url=url if success else None,
            success=success,
            error_message=error_message,
        )

        return success


    def execute_back(self, step: ActionStep) -> bool:
        """Navigate back in browser history and record interaction."""
        # Capture state BEFORE navigation for accurate before_state
        before_url = ""
        before_state = None
        try:
            before_url = self.browser.page.url
            before_state = self.session_tracker._capture_current_state()
        except Exception:
            pass

        num_back = step.action.split(":", 1)[1].strip()
        if num_back.isdigit():
            num_back = int(num_back)
        else:
            num_back = 1

        for _ in range(num_back):
            try:
                self.browser.page.go_back()
                success = True
                error_msg = None
            except Exception as e:
                success = False
                error_msg = str(e)
                dprint(f"  ❌ Back navigation failed: {e}")

        # Get URL after navigation
        after_url = before_url  # Default to before_url if navigation failed
        if success:
            try:
                after_url = self.browser.page.url
            except Exception:
                pass
        
        after_state = self.session_tracker._capture_current_state()
        # Record navigation interaction with explicit before_state
        try:
            self.session_tracker.record_interaction(
                InteractionType.NAVIGATION,
                before_state=before_state,  # Pass explicit before_state since navigation already happened
                after_state=after_state,
                target_element_info={"direction": "back", "from": before_url, "to": after_url},
                success=success,
                error_message=error_msg,
            )
        except Exception:
            pass
        return success


    def execute_upload(
        self,
        step: ActionStep,
        elements: PageElements,
        page_info: PageInfo,
        *,
        confirm_before_interaction: bool = False,
    ) -> bool:
        """Execute a file upload action"""
        # Get coordinates for the upload element first
        x, y = self.get_click_coordinates(self, step, elements, page_info)

        # Record planned interaction with goal monitor and get pre-interaction evaluations
        # Record interaction (planned -> actual)
        self.session_tracker.record_interaction(
            InteractionType.UPLOAD,
            coordinates=(x, y) if x is not None and y is not None else None,
            target_description=step.upload_file_path,
            upload_file_path=step.upload_file_path,
        )
            
        self.upload_handler.handle_upload_field(step, elements, page_info)
        step_success = True  # Assume success for handlers that don't return values yet
        return step_success


    def execute_datetime(
        self,
        step: ActionStep,
        elements: PageElements,
        page_info: PageInfo,
        *,
        confirm_before_interaction: bool = False,
    ) -> bool:
        """Execute a datetime field action"""
        # Get coordinates for the datetime element first
        x, y = self.get_click_coordinates(self, step, elements, page_info)

        # Record planned interaction with goal monitor and get pre-interaction evaluations
        # Record interaction (planned -> actual)
        self.session_tracker.record_interaction(
            InteractionType.DATETIME,
            coordinates=(x, y) if x is not None and y is not None else None,
            target_description=step.datetime_value or "date field",
            datetime_value=step.datetime_value,
        )

        self.datetime_handler.handle_datetime_field(step, elements, page_info)
        step_success = True  # Assume success for handlers that don't return values yet

        # Record the datetime interaction
        self.session_tracker.record_interaction(
            InteractionType.DATETIME,
            coordinates=(step.x, step.y) if step.x and step.y else None,
            target_element_info={
                "overlay_index": step.overlay_index,
                "datetime_value": step.datetime_value,
            },
            text_input=step.datetime_value,
            success=step_success,
        )
        return step_success
    
    def _execute_keyword_command(
        self,
        action_step: ActionStep,
        detected_elements: PageElements,
        failed_actions: List[str],
        failed_elements: List[FailedAction],
        page_info: PageInfo,
        confirm_before_interaction: bool,
        environment_state: EnvironmentState,
        start_time: float,
        extraction_schema: Optional[Dict[str, Any]] = None,
        base_knowledge: Optional[List[str]] = None,
    ) -> Optional[bool]:
        """Attempt to execute the action using keyword-based execution. Returns None to fall back."""
        parsed = parse_keyword_command(action_step.action)
        if not parsed:
            return None
        keyword, payload, helper = parsed
        keyword = (keyword or "").strip().lower()

        self.event_logger.command_start(command=action_step.action)
        if keyword == "complete_sequence":
            try:
                result = self.execute_complete_sequence(
                    step=action_step,
                )
            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing complete sequence command: {e}")
                return None
        elif keyword == "click":
            try:
                result = self.execute_click(
                    step=action_step,
                    elements=detected_elements,
                    failed_actions=failed_actions,
                    failed_elements=failed_elements,
                    page_info=page_info,
                    base_knowledge=base_knowledge,
                )
            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing click command: {e}")
                return None
        elif keyword == "type":
            try:
                result = self.execute_type(
                    step=action_step,
                    elements=detected_elements,
                    failed_actions=failed_actions,
                    failed_elements=failed_elements,
                    page_info=page_info,
                    base_knowledge=base_knowledge,
                )
            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing type command: {e}")
                return None
        elif keyword == "clear_text":
            try:
                result = self.execute_clear_text(
                    step=action_step,
                )
            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing clear text command: {e}")
                return None
        elif keyword == "ask":
            try:
                result = self.execute_ask(
                    step=action_step,
                    environment_state=environment_state,
                    base_knowledge=base_knowledge,
                )
            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing ask command: {e}")
                return None
        elif keyword == "select":
            try:
                result = self._keyword_select(
                    action_step=action_step,
                    payload=payload,
                    helper=helper,
                    confirm_before_interaction=confirm_before_interaction,
                    base_knowledge=base_knowledge,
                )
            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing select command: {e}")
                return None
        elif keyword == "upload":
            try:
                result = self._keyword_upload(
                    action_step=action_step,
                    payload=payload,
                    helper=helper,
                    confirm_before_interaction=confirm_before_interaction,
                    base_knowledge=base_knowledge,
                )
            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing upload command: {e}")
                return None
        elif keyword == "datetime":
            try:
                result = self._keyword_datetime(
                action_step=action_step,
                payload=payload,
                helper=helper,
                confirm_before_interaction=confirm_before_interaction,
                base_knowledge=base_knowledge,
            )
            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing datetime command: {e}")
                return None
        elif keyword == "scroll":
            try:
                result = self.execute_scroll(
                step=action_step,
            )
            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing scroll command: {e}")
                return None
        elif keyword == "press":
            try:
                result = self.execute_press(
                step=action_step
            )
            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing press command: {e}")
                return None
        elif keyword == "back":
            try:
                result = self.execute_back(
                    step=action_step,
            )
            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing back command: {e}")
                return None
        elif keyword == "forward":
            try:
                result = self.execute_forward(
                    step=action_step,
            )
            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing forward command: {e}")
                return None
        elif keyword == "open":
            try:
                result = self.execute_open(
                    step=action_step,
                )
            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing navigate command: {e}")
                return None
        elif keyword == "extract":
            try:
                result, err = self.extract(
                    step=action_step,
                    extraction_schema=extraction_schema,
                )

                if not result:
                    self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing extract command: {err}")

            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing extract command: {e}")
                return None
        elif keyword == "think":
            try:
                result = self.execute_think(step=action_step)
            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing think command: {e}")
                return None
        elif keyword == "assert":
            try:
                result = self.execute_assert(step=action_step)
            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing assert command: {e}")
                return None
        elif keyword == "flag":
            try:
                result = self.execute_flag(step=action_step)
            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing flag command: {e}")
                return None
        elif keyword == "wait_for":
            try:
                result = self.execute_wait_for(step=action_step)
            except Exception as e:
                self.event_logger.command_execution_failure(command=action_step.action, error=f"Error executing wait_for command: {e}")
                return None
        else:
            # Unsupported keyword
            self.event_logger.command_execution_failure(command=action_step.action, error=f"Unsupported keyword: {keyword}")
            return None

        if result is None:
            # Keyword action could not confidently execute – allow normal flow
            return None

        duration_ms = (time.time() - start_time) * 1000
        if result:
            try:
                self.event_logger.command_success(action_step.action)
            except Exception:
                pass
        else:
            print(f"Keyword command execution failed: {result}")
            self.event_logger.command_failure(command=action_step.action, error="Keyword command execution failed", duration_ms=duration_ms)
        return result
    
    def extract(
        self,
        step: ActionStep,
        extraction_schema: Optional[Dict[str, Any]] = None,
    ) -> Union[bool, str]:

        extraction_prompt = step.action.replace("extract:", "").strip()
        self.event_logger.extraction_start(extraction_prompt)
        self.event_logger.extraction_detected(extraction_prompt)

        # Capture before state with screenshot
        before_state = self.session_tracker._capture_current_state()

        # Capture screenshot based on scope
        try:
            screenshot = self.browser.page.screenshot(full_page=False)
            # Get full page text - use larger limit for full page
            visible_text = self.browser.page.evaluate("document.body.innerText") or ""
        except Exception as e:
            self.event_logger.extraction_failure(extraction_prompt, error=str(e))
            return None
        
        # Clean and prepare the text (remove excessive whitespace but keep structure)
        if visible_text:
            # Remove excessive newlines but keep some structure
            import re
            visible_text = re.sub(r'\n{3,}', '\n\n', visible_text.strip())
        
        # Extract urls from the page's html content and store it with the text content of the element that had the url
        try:
            # JavaScript to extract all links with their text content and title
            url_extraction_script = """
            () => {
                const links = [];
                const linkElements = document.querySelectorAll('a[href]');
                
                linkElements.forEach((link) => {
                    const href = link.href || link.getAttribute('href') || '';
                    if (!href || href === '#' || href.startsWith('javascript:')) {
                        return; // Skip empty, anchor-only, or javascript links
                    }
                    
                    // Get text content - prefer innerText, fallback to textContent
                    let textContent = (link.innerText || link.textContent || '').trim();
                    
                    // If no text content, try to get from child elements (like images with alt text)
                    if (!textContent) {
                        const img = link.querySelector('img');
                        if (img) {
                            textContent = img.getAttribute('alt') || img.getAttribute('title') || '';
                        }
                    }
                    
                    // Get title attribute if available
                    const title = link.getAttribute('title') || link.getAttribute('aria-label') || '';
                    
                    // Use title as text if no text content available
                    if (!textContent && title) {
                        textContent = title;
                    }
                    
                    // Skip if still no meaningful content
                    if (!textContent) {
                        textContent = '(no text)';
                    }
                    
                    links.push({
                        url: href,
                        text: textContent,
                        title: title
                    });
                });
                
                return links;
            }
            """
            extracted_urls = self.browser.page.evaluate(url_extraction_script) or []
            
            # Format URLs for the prompt
            if extracted_urls:
                url_lines = ["\n            Links found on page:"]
                for i, link_info in enumerate(extracted_urls[:50], 1):  # Limit to 50 links to avoid overwhelming the prompt
                    url = link_info.get('url', '')
                    text = link_info.get('text', '(no text)')
                    title = link_info.get('title', '')
                    
                    # Format: [text/title] - url
                    display_text = text
                    if title and title != text:
                        display_text = f"{text} (title: {title})"
                    
                    url_lines.append(f"            {i}. [{display_text}] - {url}")
                
                if len(extracted_urls) > 50:
                    url_lines.append(f"            ... and {len(extracted_urls) - 50} more links")
                
                url_hint = "\n".join(url_lines)
            else:
                url_hint = "\n            No links found on page."
        except Exception as e:
            # If URL extraction fails, continue without it
            url_hint = f"\n            (Could not extract links: {str(e)})"
        
        # Build extraction prompt for LLM with full page text for grounding
        extraction_system_prompt = f"""
            You are given a webpage's screenshots and its page content. Your task is to extract the information requested by the user.

            Current page context:
            - URL: {self.browser.page.url}
            - Title: {self.browser.page.title()}{url_hint}

            FULL PAGE TEXT CONTENT (use this for extra context):
            {visible_text if visible_text else "(No text content found on page)"}

            You also have access to the urls from the page. You can use them to get the url of any element on the page whose URL you need to extract.
            {url_hint}
            
            Notebook
                Data you extract is stored in the notebook and you have access to it
                Before you extract data, check the notebook to see if the data you are extracting is already in the notebook.
                    If it is then you should leave a note in the notebook that you have already extracted the data.
                    If it is not then you should extract the data as usual.
            
            Notebook entries:
            {self.notebook.to_list()}
            
            IMPORTANT INSTRUCTIONS:
            1. Only extract information that appears in BOTH the screenshot AND the text content provided above
            2. Do NOT make up or infer data that is not present
            3. If the requested information is not found, return an empty object {{}} or indicate "not available"
            """
        
        # Try extraction with retries
        try:
            if not extraction_schema:
                # Simple text extraction using vision, grounded with page text
                extraction_user_prompt = f"""
                            Extract the following information from this webpage screenshot:
                            {step.action}

                            Your task is to extract the following information from the webpage screenshot: {step.action} 
                            Do not make up text that isn't in the provided content."""
                result_text = generate_text(
                    prompt=extraction_user_prompt,
                    system_prompt=extraction_system_prompt,
                    image=screenshot,
                    image_detail="high"
                )
                extracted_text = result_text.strip()

                # Capture after state with screenshot
                after_state = self.session_tracker._capture_current_state()

                # Record extraction in interaction history
                self.session_tracker.record_interaction(
                    IT.EXTRACT,
                    before_state=before_state,
                    after_state=after_state,
                    extraction_prompt=extraction_system_prompt,
                    extracted_data=extracted_text,
                    success=True,
                    )
                self.event_logger.extraction_success(extraction_prompt, result={"text": extracted_text})
                
                # Add extraction to notebook if available
                try:
                    current_url = self.browser.page.url if self.browser.page else None
                    self.notebook.add_extraction(
                        task=extraction_prompt,
                        data=extracted_text,
                        url=current_url
                    )
                except Exception as e:
                    # Don't fail extraction if notebook addition fails
                    self.event_logger.system_warning(f"Failed to add extraction to notebook: {e}")
                
                return True, ""
            
            else:
                # JSON extraction using structured output
                # Use a string field for JSON to avoid schema validation issues with Dict[str, Any]
                result = generate_model(
                    prompt=extraction_user_prompt,
                    model_object_type=extraction_schema,
                    system_prompt=extraction_system_prompt,
                    image=screenshot,
                    image_detail="high"
                )               
                if not result.success:
                    self.event_logger.extraction_failure(extraction_prompt, error="Extraction failed")
                    return False, "Extraction failed"
                
                # Parse the JSON string
                import json
                try:
                    extracted_dict = json.loads(result.extracted_data)
                except json.JSONDecodeError as e:
                    self.event_logger.extraction_failure(extraction_prompt, error=f"Failed to parse extracted_data as JSON: {e}. Raw data: {result.extracted_data}")
                    return False, f"Failed to parse extracted_data as JSON: {e}. Raw data: {result.extracted_data}"
                
                # Accept both dicts and lists as valid extracted data
                if not isinstance(extracted_dict, (dict, list)):
                    self.event_logger.extraction_failure(extraction_prompt, error=f"Expected JSON object or array, got {type(extracted_dict).__name__}")
                    return False, f"Expected JSON object or array, got {type(extracted_dict).__name__}"
                if (isinstance(extracted_dict, dict) and not extracted_dict) or (isinstance(extracted_dict, list) and len(extracted_dict) == 0):
                    self.event_logger.extraction_empty(extraction_prompt, error="Extracted data is empty")
                    return False, "Extracted data is empty"

                # Record extraction in interaction history
                self.session_tracker.record_interaction(
                    IT.EXTRACT,
                    extraction_prompt=extraction_prompt,
                    extracted_data=extracted_dict,
                    success=True,
                    )
                self.event_logger.extraction_success(extraction_prompt, result=extracted_dict)
                
                # Add extraction to notebook if available
                try:
                    current_url = self.browser.page.url if self.browser.page else None
                    self.notebook.add_extraction(
                        task=extraction_prompt,
                        data=extracted_dict,
                        url=current_url
                    )
                except Exception as e:
                    # Don't fail extraction if notebook addition fails
                    self.event_logger.system_warning(f"Failed to add extraction to notebook: {e}")
                
                return True, ""
        
        except Exception as e:
            error_msg = str(e)
            # Provide more context for common errors
            if "list indices must be integers" in error_msg:
                error_msg = f"Response parsing error (list indices): {e}. This may indicate an unexpected response format from the model."
            elif "not str" in error_msg or "must be" in error_msg:
                error_msg = f"Type error during extraction: {e}. This may indicate a malformed response from the model."
            
            # Record failed extraction in interaction history
            self.session_tracker.record_interaction(
                IT.EXTRACT,
                extraction_prompt=extraction_prompt,
                extracted_data=None,
                success=False,
                error_message=error_msg,
            )
            self.event_logger.extraction_failure(extraction_prompt, error=error_msg)
            
            return False, error_msg

    def execute_think(self, step: ActionStep) -> bool:
        """Execute a think action - pure reasoning with no browser action."""
        reasoning = step.action.split(":", 1)[1].strip() if ":" in step.action else ""

        if not reasoning:
            self.event_logger.system_warning("No reasoning provided for think action")
            return False

        # Capture state for record keeping
        before_state = self.session_tracker._capture_current_state()
        after_state = before_state  # No change for think

        # Record the think interaction
        self.session_tracker.record_interaction(
            IT.THINK,
            before_state=before_state,
            after_state=after_state,
            reasoning=reasoning,
            success=True,
        )

        self.event_logger.system_info(f"🤔 Agent thinking: {reasoning}")
        return True

    def execute_assert(self, step: ActionStep) -> bool:
        """Execute an assert action - check a condition from the screenshot."""
        condition = step.action.split(":", 1)[1].strip() if ":" in step.action else ""

        if not condition:
            self.event_logger.system_warning("No condition provided for assert action")
            return False

        # Capture state for record keeping
        before_state = self.session_tracker._capture_current_state()
        after_state = before_state  # No change for assert

        # Record the assert interaction
        self.session_tracker.record_interaction(
            IT.ASSERT,
            before_state=before_state,
            after_state=after_state,
            target_element_info={"condition": condition},
            success=True,
        )

        self.event_logger.system_info(f"✓ Agent checking: {condition}")
        return True

    def execute_flag(self, step: ActionStep) -> bool:
        """Execute a flag action - non-blocking user notification."""
        message = step.action.split(":", 1)[1].strip() if ":" in step.action else ""

        if not message:
            self.event_logger.system_warning("No message provided for flag action")
            return False

        # Capture state for record keeping
        before_state = self.session_tracker._capture_current_state()
        after_state = before_state  # No change for flag

        # Record the flag interaction
        self.session_tracker.record_interaction(
            IT.FLAG,
            before_state=before_state,
            after_state=after_state,
            target_element_info={"message": message},
            success=True,
        )

        self.event_logger.system_warning(f"🚩 {message}")
        return True

    def execute_wait_for(self, step: ActionStep) -> bool:
        """Execute a wait_for action - conditional wait with timeout."""
        # Parse: "wait_for: condition | timeout=10"
        parts = step.action.split(":", 1)[1].strip() if ":" in step.action else ""

        if not parts:
            self.event_logger.system_warning("No condition provided for wait_for action")
            return False

        # Split condition and timeout
        condition = parts
        timeout_seconds = 10  # default

        if "|" in parts:
            condition_part, timeout_part = parts.split("|", 1)
            condition = condition_part.strip()
            # Parse timeout=N
            timeout_match = re.search(r'timeout=(\d+)', timeout_part)
            if timeout_match:
                timeout_seconds = int(timeout_match.group(1))

        # Capture before state
        before_state = self.session_tracker._capture_current_state()

        # Perform the wait - use a simple sleep for now
        # TODO: Could be enhanced with actual page.wait_for_selector or similar
        import time
        self.event_logger.system_info(f"⏳ Waiting for: {condition} (timeout: {timeout_seconds}s)")
        time.sleep(min(timeout_seconds, 5))  # Cap at 5 seconds for safety

        # Capture after state
        after_state = self.session_tracker._capture_current_state()

        # Record the wait interaction
        self.session_tracker.record_interaction(
            IT.WAIT_FOR,
            before_state=before_state,
            after_state=after_state,
            target_element_info={"condition": condition, "timeout": timeout_seconds},
            success=True,
        )

        return True

    def act(
        self,
        action_step: ActionStep,
        detected_elements: PageElements,
        page_info: PageInfo = None,
        failed_actions: List[str] = [],
        failed_elements: List[str] = [],
        extraction_schema: Optional[Dict[str, Any]] = None,
        confirm_before_interaction: bool = False,
        action_id: Optional[str] = None,
        environment_state: EnvironmentState = None,
        max_attempts: Optional[int] = None,
        base_knowledge: Optional[List[str]] = None,
        current_iteration: Optional[int] = None,
        **kwargs
    ) -> ActionResult:
        command = action_step.action

        # Store iteration counter as instance variable for access throughout execution
        self.current_iteration = current_iteration
        
        # Helper function to create ActionResult
        def _create_result(success: bool, message: str = "", error: Optional[str] = None, 
                          action_id: Optional[str] = None, duration: Optional[float] = None,
                          additional_metadata: Optional[Dict[str, Any]] = None,
                          data: Optional[Any] = None) -> ActionResult:
            """Create ActionResult with metadata"""
            metadata = {
                "command": command,
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

        start_time = time.time()
        
        # Generate action ID if not provided
        if action_id is None:
            import uuid
            action_id = str(uuid.uuid4())[:8]
            
        try:
            # Start action timer
            # self.execution_timer.start_action(action_id, command)

            # Add action to history
            self._add_to_command_history(command)

            # Only keyword commands are supported (click:, type:, etc.)
            keyword_command_result = self._execute_keyword_command(
                action_step=action_step,
                detected_elements=detected_elements,
                failed_actions=failed_actions,
                failed_elements=failed_elements,
                page_info=page_info,
                confirm_before_interaction=confirm_before_interaction,
                environment_state=environment_state,
                start_time=start_time,
                base_knowledge=base_knowledge,
            )
            if keyword_command_result is not None:
                # Convert bool to ActionResult (for temporary compatibility)
                duration = time.time() - start_time
                # Get overlay_index from the last interaction (if available)
                overlay_index = self.session_tracker.get_last_interaction_overlay_index()
                metadata = {"command_type": "keyword"}
                if overlay_index is not None:
                    metadata["overlay_index"] = overlay_index
                return _create_result(
                    keyword_command_result,
                    "Action executed successfully" if keyword_command_result else "Action failed",
                    action_id=action_id,
                    duration=duration,
                    additional_metadata=metadata
                )

            # If keyword execution can't handle it, fail with clear error message
            duration_ms = (time.time() - start_time) * 1000
            duration = time.time() - start_time
            self.event_logger.command_failure(command=command, error="Could not parse command as keyword action. Use format: 'click: button', 'type: text', etc.", duration_ms=duration_ms)
            
            # self.execution_timer.end_action()
            return _create_result(
                False,
                "Could not parse command as keyword action. Use format: 'click: button', 'type: text', etc.",
                error="Could not parse command as keyword action. Must use keyword format (click:, type:, etc.)",
                action_id=action_id,
                duration=duration
            )
        except Exception as e:
            self.event_logger.command_execution_failure(command=command, error=f"An error occured while executing command: {e}")
            return _create_result(
                False,
                f"Error executing command: {command}",
                error=str(e),
                action_id=action_id,
                duration=duration
            )
        # finally:
            # End action timer if still active (safety net for any unhandled returns)
            # if self.execution_timer.current_action_start is not None:
            #     self.execution_timer.end_action()
        
        
    def _add_to_command_history(self, command: str) -> None:
        """Add a command to the history, maintaining max size"""
        if command and command.strip():
            self.command_history.append(command.strip())
                
            try:
                self.event_logger.command_history(command.strip())
            except Exception:
                pass
            
    def select_best_overlay(
        self,
        action: str,
        element_data: PageElements,
        failed_elements: List[FailedAction],
        screenshot: bytes,
        *,
        base_knowledge: Optional[List[str]] = None,
    ) -> Optional[int]:
        candidate_lines: list[str] = []
            
        for elem in element_data.elements:
            idx = elem.overlay_number
            # role = elem.role_hint
            tag = elem.element_type
            text = elem.element_label
            element_label = elem.element_label
            is_focused = elem.is_focused
            tag_str = tag or "unknown"
            # Build comprehensive description
            parts = []
            parts.append(f"Overlay {idx} tag={tag_str} text={text} placeholder={element_label} is-focused={is_focused}")
            candidate_lines.append("\n".join(parts))
        
        if candidate_lines:
            try:
                self.event_logger.plan_overlay_candidates(candidates=candidate_lines)
            except Exception:
                pass

        # Build base knowledge section if provided
        base_knowledge_section = ""
        if base_knowledge:
            base_knowledge_section = "\n\nBASE KNOWLEDGE (Custom Rules):\n"
            for i, knowledge in enumerate(base_knowledge, 1):
                base_knowledge_section += f"{i}. {knowledge}\n"

        self.event_logger.system_debug(f"Looking for overlay for action: {action}")

        prompt = f"""
            You are given a page screenshot and a list of numbered overlay summaries
            Your job is to pick the overlay number that best satisfies the browsing instruction
            If NONE of the overlays clearly match the instruction (their text/aria labels do not correspond to the requested control), respond with 0 to indicate that there is no suitable element
            Respond with ONLY the overlay number as an integer (e.g., 5), or 0 if there is no match. No explanation, no JSON.
            
            These are the overlays you can choose from:
            {candidate_lines}
            
            This is how each overlay is structured:
            Overlay <index> tag=<tag> text=<text> type=<type>
            - index is the overlay number/index
            - tag is the HTML tag of the element
            - text is the text of the element
            - type is the type of the element
            Use all of these together to decide which overlay index is the best match.

            {base_knowledge_section}
        """

        # Add failed elements to the prompt
        if failed_elements:
            # Format failed actions with context
            failed_actions_lines = []
            for failed_action in failed_elements:
                overlay_idx = failed_action.overlay_index
                action = failed_action.action
                url = failed_action.url
                page_title = failed_action.page_title
                
                # Build context string
                context_parts = []
                action_display = action if action != "unknown action" else f"overlay #{overlay_idx}" if overlay_idx is not None and overlay_idx != -1 else "an action"
                if overlay_idx is not None and overlay_idx != -1:
                    context_parts.append(f"overlay #{overlay_idx}")
                if page_title:
                    context_parts.append(f"on page '{page_title}'")
                elif url:
                    # Use URL if no title available
                    context_parts.append(f"at {url}")
                
                context_str = "\n".join(context_parts)
                failed_actions_lines.append(f"  • {action_display}{context_str}")
            
            prompt += f"""
            
            Failed actions:
            - You tried these and they didn't work:
            {failed_actions_lines}
            
            - You should try selecting a different overlay index next time.
            """

        try:
            raw_response = generate_text(
                prompt=f"This is the action you are focusing on: {action}",
                system_prompt=prompt,
                image=screenshot,
                image_detail="high",
            )
        except Exception as e:
            self.event_logger.system_error(f"Error selecting best overlay: {e}")
            import traceback
            traceback.print_exc()
            return None

        if not raw_response:
            return None

        match = re.search(r"\d+", str(raw_response))
        if not match:
            return None

        overlay_index = int(match.group())
        if overlay_index == 0:
            try:
                self.event_logger.plan_overlay_chosen(overlay_index=0, raw_response=raw_response)
            except Exception:
                pass
            return None

        try:
            self.event_logger.plan_overlay_chosen(overlay_index=overlay_index, raw_response=raw_response)
        except Exception:
            pass
        return overlay_index
 