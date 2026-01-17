"""
Action execution implementations for all supported action types.
"""
import time
import random
from typing import Tuple, Optional
from unittest.mock import Mock

from models import ActionStep, ActionType, PageElements, PageInfo
from core.session import InteractionType
from vision.utils import get_gemini_box_2d_center_pixels
from utils.debug_print import dprint
from .callbacks import trigger_pre_action_hooks, trigger_post_action_hooks
from .ui_feedback import confirm_interaction_visual


def execute_forward(executor) -> bool:
    """Navigate forward in browser history and record interaction."""
    # Capture state BEFORE navigation for accurate before_state
    before_url = ""
    before_state = None
    try:
        before_url = executor.page.url
        before_state = executor.session_tracker._capture_current_state()
    except Exception:
        pass

    try:
        executor.page.go_forward()
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
            after_url = executor.page.url
        except Exception:
            pass

    # Record navigation interaction with explicit before_state
    try:
        executor.session_tracker.record_interaction(
            InteractionType.NAVIGATION,
            before_state=before_state,  # Pass explicit before_state since navigation already happened
            target_element_info={"direction": "forward", "from": before_url, "to": after_url},
            success=success,
            error_message=error_msg,
        )
    except Exception:
        pass
    return success


def execute_click(
    executor,
    step: ActionStep,
    elements: PageElements,
    page_info: PageInfo,
    *,
    allow_refinement: bool = True,
    confirm_before_interaction: bool = False,
) -> bool:
    """Execute a click action"""
    # Check if target element is in the filtered elements (focus context)
    if step.overlay_index is not None:
        target_found = False
        for element in elements.elements:
            if getattr(element, 'overlay_number', None) == step.overlay_index:
                target_found = True
                break

        if not target_found:
            dprint(f"❌ Target element {step.overlay_index} is not in focus context - goal failed")
            return False

    x, y = get_click_coordinates(executor, step, elements, page_info)
    if x is None or y is None:
        raise ValueError("Could not determine click coordinates")

    box = None
    if step.overlay_index is not None and elements and getattr(elements, 'elements', None):
        for el in elements.elements:
            if getattr(el, 'overlay_number', None) == step.overlay_index and getattr(el, 'box_2d', None):
                box = el.box_2d
                break

    # Trigger pre-action hooks early, before any confirmations or refinements
    trigger_pre_action_hooks(
        executor,
        action_type=ActionType.CLICK,
        step=step,
        page_info=page_info,
        elements=elements,
        coordinates=(x, y),
    )

    if confirm_before_interaction:
        confirm_interaction_visual(
            executor,
            action_label="click",
            overlay_index=step.overlay_index,
            selector=None,
            coordinates=(x, y),
            box=box,
            page_info=page_info,
        )

    # Removed planned interaction tracking (was for goal evaluation)
    # Check for retry requests - disabled as retries are handled elsewhere
    retry_goal = None
    if False:  # Disabled retry check
        try:
            executor.event_logger.system_info(f"Goals have requested retry - aborting current plan execution")
            executor.event_logger.system_info(f"   {retry_goal}: Retry requested (attempt {retry_goal.retry_count}/{retry_goal.max_retries})")
        except Exception:
            pass
        # Return False to indicate plan should be regenerated
        return False

    # Goal system removed - pre-evaluation checks removed

    try:
        executor.event_logger.system_debug(f"Clicking at ({x}, {y}) using mouse click")
    except Exception:
        pass

    # Capture state BEFORE performing the click (critical for accurate before_state)
    before_state = executor.session_tracker._capture_current_state()

    success = False
    error_msg = None

    try:
        executor._human_mouse_move(x, y)
        executor.page.mouse.click(x, y)
        success = True
    except Exception as e:
        error_msg = str(e)
        if hasattr(executor.event_logger, 'debug_mode') and executor.event_logger.debug_mode:
            dprint(f"  ⚠️ Mouse click failed ({e})")

    if not success:
        dprint(f"  ❌ Click failed: {error_msg}")

    # Record actual interaction with goal monitor (pass explicit before_state since click already happened)
    executor.session_tracker.record_interaction(
        InteractionType.CLICK,
        before_state=before_state,  # Pass explicit before_state captured before the click
        coordinates=(x, y),
        success=success,
        error_message=error_msg
    )

    # Trigger post-action hooks
    trigger_post_action_hooks(
        executor,
        action_type=ActionType.CLICK,
        success=success,
        step=step,
        page_info=page_info,
        elements=elements,
        coordinates=(x, y),
        error_message=error_msg,
        action_id=getattr(executor, '_current_plan_action_id', None),
    )

    return success


def clear_input_field(executor, x: Optional[int], y: Optional[int]) -> None:
    """
    Clear an input field before typing to ensure previous text is removed.
    Tries multiple methods: JavaScript first, then keyboard select-all+delete.

    Args:
        x: X coordinate of the input field
        y: Y coordinate of the input field
    """
    if x is None or y is None:
        return

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
        cleared = executor.page.evaluate(element_js)
        if cleared:
            # Only show in debug mode
            if hasattr(executor.event_logger, 'debug_mode') and executor.event_logger.debug_mode:
                dprint(f"  ✅ Cleared field using JavaScript")
            time.sleep(0.1)
            return
    except Exception as e:
        # Only show in debug mode
        if hasattr(executor.event_logger, 'debug_mode') and executor.event_logger.debug_mode:
            dprint(f"  ⚠️ JavaScript clear failed, using keyboard: {e}")

    # Fallback: click, select all, delete
    try:
        executor.page.mouse.click(x, y)
        time.sleep(0.2)
        executor.page.keyboard.press('Control+a')
        time.sleep(0.1)
        executor.page.keyboard.press('Delete')
        time.sleep(0.1)
        # Only show in debug mode
        if hasattr(executor.event_logger, 'debug_mode') and executor.event_logger.debug_mode:
            dprint(f"  ✅ Cleared field using keyboard (Ctrl+A, Delete)")
    except Exception as e:
        # Only show in debug mode
        if hasattr(executor.event_logger, 'debug_mode') and executor.event_logger.debug_mode:
            dprint(f"  ⚠️ Keyboard clear failed: {e}")


def execute_type(
    executor,
    step: ActionStep,
    elements: PageElements,
    page_info: PageInfo,
    *,
    allow_refinement: bool = True,
    confirm_before_interaction: bool = False,
) -> bool:
    """Execute a type action"""
    if not step.text_to_type:
        dprint("⚠️ No text specified for TYPE action")
        return False

    # Get coordinates for the element to type into
    x, y = get_click_coordinates(executor, step, elements, page_info)
    box = None
    if step.overlay_index is not None and elements and getattr(elements, 'elements', None):
        for el in elements.elements:
            if getattr(el, 'overlay_number', None) == step.overlay_index and getattr(el, 'box_2d', None):
                box = el.box_2d
                break

    # Record planned interaction - removed (was for goal evaluation)

    # Check for retry requests from goals immediately after evaluation
    # Removed retry goal check - retries are handled elsewhere
    retry_goal = None
    # retry_goal = self.goal_monitor.check_for_retry_request()
    if retry_goal:
        try:
            executor.event_logger.system_info(f"Goals have requested retry - aborting current plan execution")
            executor.event_logger.system_info(f"   {retry_goal}: Retry requested (attempt {retry_goal.retry_count}/{retry_goal.max_retries})")
        except Exception:
            pass
        return False

    # Trigger pre-action hooks early, before any interactions
    trigger_pre_action_hooks(
        executor,
        action_type=ActionType.TYPE,
        step=step,
        page_info=page_info,
        elements=elements,
        coordinates=(x, y) if x is not None and y is not None else None,
    )

    # Click first to focus the element
    if x is not None and y is not None:
        # Validate and clamp coordinates to viewport
        if confirm_before_interaction:
            confirm_interaction_visual(
                executor,
                action_label="type",
                overlay_index=step.overlay_index,
                selector=None,
                coordinates=(x, y),
                box=box,
                page_info=page_info,
            )
        executor._human_mouse_move(x, y)
        executor.page.mouse.click(x, y)
        time.sleep(random.uniform(0.1, 0.3))

    # Capture state BEFORE performing the type action (critical for accurate before_state)
    before_state = executor.session_tracker._capture_current_state()

    try:
        executor.event_logger.system_debug(f"Typing: {step.text_to_type}")
    except Exception:
        pass

    try:
        # Always clear the field before typing to ensure previous text is removed
        clear_input_field(executor, x, y)

        # Try to get element selector and use fill() or press_sequentially
        element_selector = None
        if x is not None and y is not None:
            try:
                element_selector = executor.selector_utils.get_element_selector_from_coordinates(x, y)
                if element_selector:
                    try:
                        executor.event_logger.system_debug(f"Using fill() method with selector: {element_selector}")
                    except Exception:
                        pass
                    # Use locator for more reliable filling
                    locator = executor.page.locator(element_selector).first
                    # Type with random delay between keystrokes if text is short, otherwise fill
                    # Note: fill() automatically clears, but we already cleared above for consistency
                    if len(step.text_to_type) < 50:
                        locator.press_sequentially(step.text_to_type, delay=random.randint(50, 150))
                    else:
                        locator.fill(step.text_to_type)
                    success = True
                    error_msg = None
                else:
                    raise ValueError("Could not get element selector")
            except Exception as e:
                # Only show in debug mode
                if hasattr(executor.event_logger, 'debug_mode') and executor.event_logger.debug_mode:
                    dprint(f"  ⚠️ fill() method failed, falling back to keyboard: {e}")
                element_selector = None

        # Fallback to keyboard method if fill() didn't work
        if not element_selector:
            # Ensure element is focused before keyboard typing
            if x is not None and y is not None:
                try:
                    executor.page.mouse.click(x, y)
                    time.sleep(0.1)
                except Exception:
                    pass
            # Field was already cleared above, so just type the new text
            executor.page.keyboard.type(step.text_to_type, delay=50)
            success = True
            error_msg = None
    except Exception as e:
        success = False
        error_msg = str(e)
        dprint(f"  ❌ Typing failed: {e}")

    # Record type interaction with goal monitor (pass explicit before_state since typing already happened)
    executor.session_tracker.record_interaction(
        InteractionType.TYPE,
        before_state=before_state,  # Pass explicit before_state captured before the typing
        coordinates=(x, y) if x is not None and y is not None else None,
        text_input=step.text_to_type,
        success=success,
        error_message=error_msg
    )

    try:
        executor.event_logger.system_debug(f"Success: {success}")
    except Exception:
        pass

    # Trigger post-action hooks
    trigger_post_action_hooks(
        executor,
        action_type=ActionType.TYPE,
        success=success,
        step=step,
        page_info=page_info,
        elements=elements,
        coordinates=(x, y) if x is not None and y is not None else None,
        error_message=error_msg,
        action_id=getattr(executor, '_current_plan_action_id', None),
    )

    return success


def get_interpreted_scroll_position(executor, direction: str):
    """Get the interpreted scroll target position from active ScrollGoal"""
    try:
        # Find active ScrollGoal
        # Goal system removed - ScrollGoal removed
        scroll_goal = None

        if not scroll_goal:
            return None

        # Goal system removed - BrowserState moved to session_tracker
        if executor.page_utils:
            page_info = executor.page_utils.get_page_info()
        else:
            # Fallback if page_utils is not available
            page_info = Mock()
            page_info.doc_height = 2000
            page_info.doc_width = 1200
            page_info.height = 800
            page_info.width = 1200
            page_info.scroll_x = 0
            page_info.scroll_y = 0

        # Only show in debug mode
        if hasattr(executor.event_logger, 'debug_mode') and executor.event_logger.debug_mode:
            dprint(f"Page height: {page_info.doc_height}, Page width: {page_info.doc_width}")

        # Goal system removed - ScrollGoal interpretation removed
        # Scroll interpretation no longer available
        interpretation = None
        if interpretation:
            # Only show in debug mode
            if hasattr(executor.event_logger, 'debug_mode') and executor.event_logger.debug_mode:
                dprint(f"[Executor] ScrollGoal interpreted '{scroll_goal.user_request}' as target position ({interpretation.target_x}, {interpretation.target_y}) {interpretation.direction} ({interpretation.axis})")
            return interpretation

        return None

    except Exception as e:
        # Only show in debug mode
        if hasattr(executor.event_logger, 'debug_mode') and executor.event_logger.debug_mode:
            dprint(f"[Executor] Error getting interpreted scroll position: {e}")
        return None


def execute_scroll(executor, step: ActionStep) -> bool:
    """
    Execute a scroll action via a straightforward Playwright scroll call.

    Args:
        step: ActionStep containing scroll direction and parameters

    Returns:
        bool: True if scroll succeeded, False otherwise
    """
    direction = step.scroll_direction or "down"
    axis = "vertical"
    current_scroll_x = int(executor.page.evaluate("window.pageXOffset || window.scrollX") or 0)
    current_scroll_y = int(executor.page.evaluate("window.pageYOffset || window.scrollY") or 0)

    interpreted_scroll = get_interpreted_scroll_position(executor, direction)
    if interpreted_scroll:
        target_x = interpreted_scroll.target_x
        target_y = interpreted_scroll.target_y
        axis = interpreted_scroll.axis
        direction = interpreted_scroll.direction
        if hasattr(executor.event_logger, 'debug_mode') and executor.event_logger.debug_mode:
            dprint(f"[Executor] Using interpreted scroll: target position ({target_x}, {target_y}) {direction} ({axis})")
    else:
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
        if hasattr(executor.event_logger, 'debug_mode') and executor.event_logger.debug_mode:
            dprint(f"[Executor] Using default scroll: target position ({target_x}, {target_y}) {direction} ({axis})")
    target_x = int(target_x)
    target_y = int(target_y)

    page_info = PageInfo(
        url=executor.page.url, width=1200, height=800,
        scroll_x=current_scroll_x, scroll_y=current_scroll_y,
        title=executor.page.title() or "",
        dpr=1.0,
        ss_pixel_w=1200, ss_pixel_h=800, css_scale=1.0,
        doc_width=1200, doc_height=2000
    )
    elements = PageElements(elements=[])
    trigger_pre_action_hooks(
        executor,
        action_type=ActionType.SCROLL,
        step=step,
        page_info=page_info,
        elements=elements,
        coordinates=None,
    )

    retry_goal = None
    if retry_goal:
        try:
            executor.event_logger.system_info("Goals have requested retry - aborting current plan execution")
            executor.event_logger.system_info(f"   {retry_goal}: Retry requested (attempt {retry_goal.retry_count}/{retry_goal.max_retries})")
        except Exception:
            pass
        return False

    if hasattr(executor.event_logger, 'debug_mode') and executor.event_logger.debug_mode:
        dprint(f"  Scrolling to position ({target_x}, {target_y}) {direction} ({axis})")

    before_state = executor.session_tracker._capture_current_state()
    success = False
    error_msg = None
    try:
        scroll_amount_y = target_y - current_scroll_y
        scroll_amount_x = target_x - current_scroll_x
        if hasattr(executor.event_logger, 'debug_mode') and executor.event_logger.debug_mode:
            dprint(f"🔍 [Executor] Attempting to scroll {direction} by ({scroll_amount_x}, {scroll_amount_y})px")
        executor.page.evaluate(f"window.scrollBy({scroll_amount_x}, {scroll_amount_y})")
        if executor.page_utils:
            actual_scroll_y = int(executor.page.evaluate("window.pageYOffset || window.scrollY") or 0)
            actual_scroll_x = int(executor.page.evaluate("window.pageXOffset || window.scrollX") or 0)
            executor.page_utils.last_scroll_y = actual_scroll_y
            executor.page_utils.last_scroll_x = actual_scroll_x
        success = True
    except Exception as exc:
        error_msg = str(exc)
        success = False
        dprint(f"  ❌ Scroll failed: {exc}")

    executor.session_tracker.record_interaction(
        InteractionType.SCROLL,
        before_state=before_state,
        target_x=target_x,
        target_y=target_y,
        scroll_direction=direction,
        scroll_axis=axis,
        success=success,
        error_message=error_msg
    )

    from models import PageElements as PE, PageInfo as PI
    current_page_info = executor.page_utils.get_page_info() if executor.page_utils else PI(
        width=1200, height=800, scroll_x=target_x, scroll_y=target_y,
        url=executor.page.url, title="", dpr=1.0,
        ss_pixel_w=1200, ss_pixel_h=800, css_scale=1.0,
        doc_width=1200, doc_height=2000
    )
    trigger_post_action_hooks(
        executor,
        action_type=ActionType.SCROLL,
        success=success,
        step=step,
        page_info=current_page_info,
        elements=PE(elements=[]),
        coordinates=(target_x, target_y),
        error_message=error_msg,
        action_id=getattr(executor, '_current_plan_action_id', None),
    )

    return success


def execute_wait(executor, step: ActionStep) -> bool:
    """Execute a wait action"""
    wait_time = step.wait_time_ms or 500
    # Only show in debug mode
    if hasattr(executor.event_logger, 'debug_mode') and executor.event_logger.debug_mode:
        dprint(f"  Waiting {wait_time}ms")
    time.sleep(wait_time / 1000)
    return True


def execute_press(executor, step: ActionStep) -> bool:
    """Execute a key press action"""
    if not step.keys_to_press:
        dprint("⚠️ No keys specified for PRESS action")
        return False

    try:
        executor.event_logger.system_debug(f"Pressing keys: {step.keys_to_press}")
    except Exception:
        pass

    # Trigger pre-action hooks early, before any interactions
    from models import PageElements as PE, PageInfo as PI
    current_page_info = executor.page_utils.get_page_info() if executor.page_utils else PI(
        width=1200, height=800, scroll_x=0, scroll_y=0,
        url=executor.page.url, title="", dpr=1.0,
        ss_pixel_w=1200, ss_pixel_h=800, css_scale=1.0,
        doc_width=1200, doc_height=800
    )
    trigger_pre_action_hooks(
        executor,
        action_type=ActionType.PRESS,
        step=step,
        page_info=current_page_info,
        elements=PE(elements=[]),
        coordinates=None,
    )

    # Capture state BEFORE performing the press action (critical for accurate before_state)
    before_state = executor.session_tracker._capture_current_state()

    # Record planned interaction with goal monitor
    # Removed planned interaction tracking (was for goal evaluation)

    # Check for retry requests from goals immediately after evaluation
    # Removed retry goal check - retries are handled elsewhere
    retry_goal = None
    # retry_goal = self.goal_monitor.check_for_retry_request()
    if retry_goal:
        try:
            executor.event_logger.system_info(f"Goals have requested retry - aborting current plan execution")
            executor.event_logger.system_info(f"   {retry_goal}: Retry requested (attempt {retry_goal.retry_count}/{retry_goal.max_retries})")
        except Exception:
            pass
        return False

    # Goal system removed - pre-evaluation checks removed

    try:
        # Parse and execute the key combination
        parse_and_press_keys(executor, step.keys_to_press)
        success = True
        error_msg = None
    except Exception as e:
        success = False
        error_msg = str(e)
        dprint(f"  ❌ Key press failed: {e}")

    # Record actual interaction with goal monitor (pass explicit before_state since press already happened)
    executor.session_tracker.record_interaction(
        InteractionType.PRESS,
        before_state=before_state,  # Pass explicit before_state captured before the press
        keys_pressed=step.keys_to_press,
        success=success,
        error_message=error_msg
    )

    # Trigger post-action hooks
    from models import PageElements as PE, PageInfo as PI
    # Get current page info for press actions
    current_page_info = executor.page_utils.get_page_info() if executor.page_utils else PI(
        width=1200, height=800, scroll_x=0, scroll_y=0,
        url=executor.page.url, title="", dpr=1.0,
        ss_pixel_w=1200, ss_pixel_h=800, css_scale=1.0,
        doc_width=1200, doc_height=800
    )
    trigger_post_action_hooks(
        executor,
        action_type=ActionType.PRESS,
        success=success,
        step=step,
        page_info=current_page_info,
        elements=PE(elements=[]),
        coordinates=None,
        error_message=error_msg,
        action_id=getattr(executor, '_current_plan_action_id', None),
    )

    return success


def parse_and_press_keys(executor, keys_string: str) -> None:
    """Parse a key string and execute the key press(es)"""
    # Normalize the key string
    keys_string = keys_string.lower().strip()

    # Handle comma-separated keys (multiple keys to press sequentially)
    if ',' in keys_string:
        keys_list = [k.strip() for k in keys_string.split(',')]
        for key in keys_list:
            parse_and_press_keys(executor, key)  # Recursively handle each key
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
            executor.page.keyboard.press(f"{'+'.join(playwright_modifiers)}+{playwright_key}")
        else:
            executor.page.keyboard.press(playwright_key)
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
        executor.page.keyboard.press(playwright_key)


def get_click_coordinates(executor, step: ActionStep, elements: PageElements, page_info: PageInfo) -> Tuple[Optional[int], Optional[int]]:
    """Get the coordinates to click based on the step"""
    # Prefer explicit coordinates when provided
    if step.x is not None and step.y is not None:
        try:
            executor.event_logger.action_coordinates(f"Using explicit coordinates=({int(step.x)},{int(step.y)})")
        except Exception:
            pass
        return int(step.x), int(step.y)
    # Prefer overlay index → convert to pixel center from normalized box
    if step.overlay_index is not None:
        # First try to find by overlay number (preferred)
        for element in elements.elements:
            if element.overlay_number == step.overlay_index and element.box_2d:
                # ADD THIS: Validate coordinates before using them
                if len(element.box_2d) == 4:
                    y_min, x_min, y_max, x_max = element.box_2d
                    if (0 <= y_min <= 1000 and 0 <= x_min <= 1000 and
                        0 <= y_max <= 1000 and 0 <= x_max <= 1000 and
                        y_min < y_max and x_min < x_max):
                        center_x, center_y = get_gemini_box_2d_center_pixels(
                            element.box_2d, page_info.width, page_info.height
                        )
                        if center_x > 0 or center_y > 0:
                            try:
                                executor.event_logger.action_coordinates(f"Using overlay #{step.overlay_index} center=({center_x},{center_y}) from box={element.box_2d}")
                            except Exception:
                                pass
                            return center_x, center_y
                    else:
                        try:
                            executor.event_logger.action_coordinates(f"Skipping overlay #{step.overlay_index} with invalid coordinates: {element.box_2d}")
                        except Exception:
                            pass
                        continue
                else:
                    try:
                        executor.event_logger.action_coordinates(f"Skipping overlay #{step.overlay_index} with malformed coordinates: {element.box_2d}")
                    except Exception:
                        pass
                    continue
        # Fallback to array index (legacy support)
        if 0 <= step.overlay_index < len(elements.elements):
            element = elements.elements[step.overlay_index]
            if element.box_2d:
                center_x, center_y = get_gemini_box_2d_center_pixels(
                    element.box_2d, page_info.width, page_info.height
                )
                if center_x > 0 or center_y > 0:
                    try:
                        executor.event_logger.action_coordinates(f"Using elements[{step.overlay_index}] center=({center_x},{center_y}) from box={element.box_2d}")
                    except Exception:
                        pass
                    return center_x, center_y

    return None, None


def get_simple_dom_signature(executor) -> str:
    """Get a simple DOM signature for change detection (URL + element count)"""
    try:
        import hashlib
        url = executor.page.url
        # Get element count as a simple DOM change indicator
        element_count = executor.page.evaluate("() => document.querySelectorAll('*').length")
        sig_src = f"{url}|{element_count}"
        return hashlib.md5(sig_src.encode("utf-8")).hexdigest()
    except Exception:
        # Fallback to just URL
        try:
            import hashlib
            return hashlib.md5(executor.page.url.encode("utf-8")).hexdigest()
        except Exception:
            return "unknown"


def execute_stop(executor, step: ActionStep) -> bool:
    """Execute a stop action - returns True to indicate successful stop"""
    dprint("🛑 STOP action executed - terminating automation")
    return True


def execute_open(
    executor,
    step: ActionStep,
    *,
    confirm_before_interaction: bool = False,
) -> bool:
    """Open a URL directly in the current tab and record navigation."""
    url = (step.url or "").strip()
    if not url:
        dprint("❌ OPEN action missing URL")
        executor.session_tracker.record_interaction(
            InteractionType.NAVIGATION,
            navigation_url="",
            success=False,
            error_message="OPEN action missing URL",
        )
        return False

    # Removed planned interaction tracking (was for goal evaluation)

    # Removed retry goal check - retries are handled elsewhere
    retry_goal = None
    # retry_goal = self.goal_monitor.check_for_retry_request()
    if retry_goal:
        try:
            executor.event_logger.system_info(f"Goals have requested retry - aborting current plan execution")
            executor.event_logger.system_info(f"   {retry_goal}: Retry requested (attempt {retry_goal.retry_count}/{retry_goal.max_retries})")
        except Exception:
            pass
        return False

    if confirm_before_interaction:
        dprint("ℹ️ Confirmation requested for OPEN action, skipping visual confirmation (no overlay)")

    success = True
    error_message = None
    try:
        executor.page.goto(url, wait_until="domcontentloaded")
    except Exception as e:
        success = False
        error_message = str(e)
        dprint(f"  ❌ Open navigation failed: {e}")

    executor.session_tracker.record_interaction(
        InteractionType.NAVIGATION,
        target_element_info={"url": url},
        navigation_url=url if success else None,
        success=success,
        error_message=error_message,
    )

    return success


def execute_back(executor) -> bool:
    """Navigate back in browser history and record interaction."""
    # Capture state BEFORE navigation for accurate before_state
    before_url = ""
    before_state = None
    try:
        before_url = executor.page.url
        before_state = executor.session_tracker._capture_current_state()
    except Exception:
        pass

    try:
        executor.page.go_back()
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
            after_url = executor.page.url
        except Exception:
            pass

    # Record navigation interaction with explicit before_state
    try:
        executor.session_tracker.record_interaction(
            InteractionType.NAVIGATION,
            before_state=before_state,  # Pass explicit before_state since navigation already happened
            target_element_info={"direction": "back", "from": before_url, "to": after_url},
            success=success,
            error_message=error_msg,
        )
    except Exception:
        pass
    return success


def execute_upload(
    executor,
    step: ActionStep,
    elements: PageElements,
    page_info: PageInfo,
    *,
    confirm_before_interaction: bool = False,
) -> bool:
    """Execute a file upload action"""
    # Get coordinates for the upload element first
    x, y = get_click_coordinates(executor, step, elements, page_info)

    # Record planned interaction with goal monitor and get pre-interaction evaluations
    # Record interaction (planned -> actual)
    executor.session_tracker.record_interaction(
        InteractionType.UPLOAD,
        coordinates=(x, y) if x is not None and y is not None else None,
        target_description=step.upload_file_path,
        upload_file_path=step.upload_file_path,
    )

    # Check for retry requests from goals immediately after evaluation
    # Removed retry goal check - retries are handled elsewhere
    retry_goal = None
    if False:  # Disabled retry check
        try:
            executor.event_logger.system_info(f"Goals have requested retry - aborting current plan execution")
            executor.event_logger.system_info(f"   {retry_goal}: Retry requested (attempt {retry_goal.retry_count}/{retry_goal.max_retries})")
        except Exception:
            pass
        return False

    if confirm_before_interaction:
        confirm_interaction_visual(
            executor,
            action_label="upload",
            overlay_index=step.overlay_index,
            selector=None,
            coordinates=None,
            box=None,
            page_info=page_info,
        )
    executor.upload_handler.handle_upload_field(step, elements, page_info)
    step_success = True  # Assume success for handlers that don't return values yet
    return step_success


def execute_datetime(
    executor,
    step: ActionStep,
    elements: PageElements,
    page_info: PageInfo,
    *,
    confirm_before_interaction: bool = False,
) -> bool:
    """Execute a datetime field action"""
    # Get coordinates for the datetime element first
    x, y = get_click_coordinates(executor, step, elements, page_info)

    # Record planned interaction with goal monitor and get pre-interaction evaluations
    # Record interaction (planned -> actual)
    executor.session_tracker.record_interaction(
        InteractionType.DATETIME,
        coordinates=(x, y) if x is not None and y is not None else None,
        target_description=step.datetime_value or "date field",
        datetime_value=step.datetime_value,
    )

    # Check for retry requests from goals immediately after evaluation
    # Removed retry goal check - retries are handled elsewhere
    retry_goal = None
    if False:  # Disabled retry check
        try:
            executor.event_logger.system_info(f"Goals have requested retry - aborting current plan execution")
            executor.event_logger.system_info(f"   {retry_goal}: Retry requested (attempt {retry_goal.retry_count}/{retry_goal.max_retries})")
        except Exception:
            pass
        return False

    if confirm_before_interaction:
        confirm_interaction_visual(
            executor,
            action_label="datetime",
            overlay_index=step.overlay_index,
            selector=None,
            coordinates=None,
            box=None,
            page_info=page_info,
        )
    executor.datetime_handler.handle_datetime_field(step, elements, page_info)
    step_success = True  # Assume success for handlers that don't return values yet

    # Record the datetime interaction
    executor.session_tracker.record_interaction(
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
