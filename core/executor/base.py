"""
Core executor initialization and plan execution logic.
"""
import hashlib
import math
import re
import time
import random
import uuid
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List, Callable, Type, Union
from enum import Enum
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Page
from pydantic import BaseModel, Field, create_model
from agent.agent_context import EnvironmentState
from agent.notebook import Notebook
from core.browser import Browser
from core.config import Config
from core.executor.ui_feedback import highlight_click_location
from models import ActionStep, ActionType, PageElements, PageInfo
from models.models import FailedAction, ChosenOption
from utils import SelectorUtils
from utils.page_utils import PageUtils
from agent.memory import NarrativeMemory, InteractionType
from utils.debug_print import dprint
from execution.result import ActionResult
from lib.ai import generate_text, generate_model
from agent.memory import InteractionType as IT

DOM_SMART_CLICK_POINT_SCRIPT = """
({ overlayIndex, preferTextDescendant }) => {
    if (!overlayIndex) {
        return null;
    }

    const findByShadow = (idx, root) => {
        const found = root.querySelector(`[data-dom-index="${idx}"]`);
        if (found) return found;
        for (const host of root.querySelectorAll('*')) {
            if (host.shadowRoot) {
                const deep = findByShadow(idx, host.shadowRoot);
                if (deep) return deep;
            }
        }
        return null;
    };
    const el = findByShadow(overlayIndex, document);
    if (!el) {
        return null;
    }

    const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 0;
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;

    const inViewportRect = (rect) => {
        if (!rect) return false;
        if (rect.width <= 0 || rect.height <= 0) return false;
        return rect.right > 0 &&
               rect.bottom > 0 &&
               rect.left < viewportWidth &&
               rect.top < viewportHeight;
    };

    const isVisible = (node) => {
        if (!node) return false;
        const style = window.getComputedStyle(node);
        if (!style) return false;
        if (style.display === "none" || style.visibility === "hidden" || style.pointerEvents === "none") return false;
        const opacity = Number.parseFloat(style.opacity || "1");
        if (Number.isFinite(opacity) && opacity === 0) return false;
        return inViewportRect(node.getBoundingClientRect());
    };

    const isHittableAt = (x, y, target) => {
        if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
        if (x < 0 || y < 0 || x >= viewportWidth || y >= viewportHeight) return false;
        const hit = document.elementFromPoint(x, y);
        if (!hit || !target) return false;
        return hit === target || target.contains(hit) || hit.contains(target);
    };

    const candidatePointsFromRect = (rect) => {
        if (!inViewportRect(rect)) {
            return [];
        }
        const points = [
            { x: rect.left + rect.width * 0.5, y: rect.top + rect.height * 0.5 },
            { x: rect.left + rect.width * 0.5, y: rect.top + Math.min(rect.height * 0.3, rect.height - 1) },
            { x: rect.left + rect.width * 0.35, y: rect.top + rect.height * 0.5 },
            { x: rect.left + rect.width * 0.65, y: rect.top + rect.height * 0.5 },
            { x: rect.left + rect.width * 0.2, y: rect.top + rect.height * 0.5 },
            { x: rect.left + rect.width * 0.8, y: rect.top + rect.height * 0.5 },
        ];

        const dedup = new Set();
        const normalized = [];
        for (const p of points) {
            const px = Math.round(p.x);
            const py = Math.round(p.y);
            const key = `${px}:${py}`;
            if (dedup.has(key)) continue;
            dedup.add(key);
            normalized.push({ x: px, y: py });
        }
        return normalized;
    };

    // If the element has pointer-events:none, elementFromPoint will never
    // return it — hit-testing is pointless.  Just return its center directly.
    // The click at this location passes through to the interactive element
    // behind it (e.g. chess board square behind a hint dot).
    const elStyle = window.getComputedStyle(el);
    if (elStyle && elStyle.pointerEvents === "none") {
        const elRect = el.getBoundingClientRect();
        if (inViewportRect(elRect)) {
            return {
                x: Math.round(elRect.left + elRect.width * 0.5),
                y: Math.round(elRect.top + elRect.height * 0.5),
                source: "pointer-events-none",
                targetTag: (el.tagName || "").toLowerCase(),
                baseTag: (el.tagName || "").toLowerCase(),
            };
        }
    }

    const descendantTargets = [];
    if (preferTextDescendant && el.tagName === "A") {
        const selectors = "h1,h2,h3,h4,h5,h6,[role='heading'],span,strong,b,p,div";
        const children = Array.from(el.querySelectorAll(selectors)).filter(isVisible);
        children.sort((a, b) => {
            const ar = a.getBoundingClientRect();
            const br = b.getBoundingClientRect();
            return (br.width * br.height) - (ar.width * ar.height);
        });
        for (const child of children) {
            descendantTargets.push(child);
        }
    }

    const orderedTargets = [...descendantTargets, el];
    for (const target of orderedTargets) {
        const rect = target.getBoundingClientRect();
        const points = candidatePointsFromRect(rect);
        for (const point of points) {
            if (isHittableAt(point.x, point.y, target)) {
                return {
                    x: point.x,
                    y: point.y,
                    source: target === el ? "target" : "descendant",
                    targetTag: (target.tagName || "").toLowerCase(),
                    baseTag: (el.tagName || "").toLowerCase(),
                };
            }
            // For link containers, accept points that hit any part of the anchor subtree.
            if (target === el && isHittableAt(point.x, point.y, el)) {
                return {
                    x: point.x,
                    y: point.y,
                    source: "anchor-subtree",
                    targetTag: (target.tagName || "").toLowerCase(),
                    baseTag: (el.tagName || "").toLowerCase(),
                };
            }
        }
    }

    // All hit-test points failed — return sentinel for JS direct click.
    const elRect = el.getBoundingClientRect();
    return {
        x: Math.round(elRect.left + elRect.width * 0.5),
        y: Math.round(elRect.top + elRect.height * 0.5),
        source: "js-click-fallback",
        targetTag: (el.tagName || "").toLowerCase(),
        baseTag: (el.tagName || "").toLowerCase(),
    };
}
"""

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
                 memory_store: NarrativeMemory, 
                 notebook: Notebook,
                 page_utils:PageUtils=None, 
                 preferred_click_method: str = "programmatic", 
                 user_question_callback: Optional[Callable[[str, dict, List[str], bool, bool], str]] = None, 
                 agent_talk_callback: Optional[Callable[[str], None]] = None, 
                 data_report_callback: Optional[Callable[[str, dict], None]] = None,
                 workspace_paths: Optional[Dict[str, str]] = None,
                 force_workspace_write_data: bool = False,
                 upload_mode: str = "auto",
                 sandbox_policy: Optional[Any] = None):
        self.browser = browser
        self.memory_store = memory_store
        self.page_utils = page_utils
        self.last_failure_reason: Optional[str] = None
        self.user_question_callback = user_question_callback
        self.agent_talk_callback = agent_talk_callback
        self.data_report_callback = data_report_callback
        self.notebook = notebook  # Optional notebook for storing extraction results
        self.workspace_paths = workspace_paths or {}
        self.force_workspace_write_data = bool(force_workspace_write_data)
        self.upload_mode = upload_mode if upload_mode in ("auto", "workspace_only", "user_only") else "auto"
        self.sandbox_policy = sandbox_policy
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
        self._write_data_session_id: str = f"session_{uuid.uuid4().hex[:8]}"

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

    @staticmethod
    def _get_agent_element_id(step: ActionStep) -> Optional[int]:
        """Extract element_id from function_arguments (element_index mode)."""
        args = getattr(step, 'function_arguments', None)
        if args and isinstance(args, dict):
            eid = args.get('element_id')
            if eid is not None:
                try:
                    return int(eid)
                except (ValueError, TypeError):
                    pass
        return None

    @staticmethod
    def _resolve_element_id_to_overlay(
        step: "ActionStep",
        elements: "PageElements",
    ) -> Optional[int]:
        """If step has element_id, resolve to overlay_number. Returns None if not found."""
        eid = Executor._get_agent_element_id(step)
        if eid is None:
            return None
        for elem in elements.elements:
            if getattr(elem, 'overlay_number', None) == eid:
                return elem.overlay_number
        return None

    @staticmethod
    def _get_action_args(step: ActionStep) -> Dict[str, Any]:
        """Return normalized function arguments for a step."""
        args = getattr(step, "function_arguments", None)
        return args if isinstance(args, dict) else {}

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        """Coerce a value to a positive integer; return None for missing/invalid values."""
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _get_action_command(self, step: ActionStep) -> str:
        """Human-readable command string for logging/history."""
        command = getattr(step, "action", None)
        if isinstance(command, str) and command.strip():
            return command.strip()
        function_name = getattr(step, "function_name", None) or "unknown_action"
        return f"{function_name}: {self._get_action_args(step)}"

    def mark_element_done(self, overlay_index: int) -> None:
        """Mark a DOM element as done by setting data-bvb-done attribute."""
        try:
            self.browser.page.evaluate(
                """(overlayIndex) => {
                    const el = document.querySelector(`[data-dom-index="${overlayIndex}"]`);
                    if (el) el.setAttribute('data-bvb-done', 'true');
                }""",
                overlay_index,
            )
        except Exception:
            pass

    def clear_done_markers(self) -> None:
        """Remove all data-bvb-done attributes from the DOM."""
        try:
            self.browser.page.evaluate(
                """() => {
                    document.querySelectorAll('[data-bvb-done]').forEach(
                        el => el.removeAttribute('data-bvb-done')
                    );
                }"""
            )
        except Exception:
            pass

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
            before_state = self.memory_store._capture_current_state()
        except Exception:
            pass

        args = self._get_action_args(step)
        raw_steps = args.get("steps", 1)
        try:
            num_forward = int(raw_steps)
        except Exception:
            num_forward = 1
        if num_forward < 1:
            num_forward = 1

        error_msg: Optional[str] = None
        for _ in range(num_forward):
            try:
                self.browser.page.go_forward(wait_until="domcontentloaded")
            except Exception as e:
                error_msg = str(e)
                dprint(f"  ❌ Forward navigation failed: {e}")
                break

        after_state = self.memory_store._capture_current_state()
        after_url = before_url
        try:
            after_url = self.browser.page.url
        except Exception:
            if after_state and getattr(after_state, "url", ""):
                after_url = after_state.url

        state_changed = False
        try:
            state_changed = bool(
                self.memory_store._has_meaningful_change(before_state, after_state)
            )
        except Exception:
            if before_state and after_state:
                state_changed = (
                    before_state.url != after_state.url
                    or before_state.title != after_state.title
                )
            elif before_url and after_url:
                state_changed = before_url != after_url

        success = error_msg is None and state_changed
        if error_msg is None and not state_changed:
            error_msg = "Forward navigation did not change the page state."

        self.event_logger.command_execution_complete("FORWARD", success=success, target_description=after_url, reasoning=None)

        # Record navigation interaction with explicit before_state
        try:
            self.memory_store.record_interaction(
                InteractionType.NAVIGATION,
                before_state=before_state,  # Pass explicit before_state since navigation already happened
                after_state=after_state,
                target_element_info={
                    "direction": "forward",
                    "steps": num_forward,
                    "from": before_url,
                    "to": after_url,
                },
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
        prior_failures: List[str],
        failed_elements: List[FailedAction],
        page_info: PageInfo,
        base_knowledge: Optional[List[str]] = None,
    ) -> bool:
        """Execute a click action"""
        # Figure out the overlay to use for the click
        # Capture state BEFORE performing the click (critical for accurate before_state)
        before_state = self.memory_store._capture_current_state()

        current_screenshot = before_state.screenshot

        args = self._get_action_args(step)
        x, y = None, None
        use_js_click = False
        overlay_index = None

        # Resolve element_id from the INTERACTIVE ELEMENTS index
        element_id = self._get_agent_element_id(step)
        if element_id is not None:
            self.event_logger.system_debug(
                f"Element index click: element_id={element_id}"
            )
            matched = None
            for elem in elements.elements:
                if getattr(elem, 'overlay_number', None) == element_id:
                    matched = elem
                    break
            if matched is not None:
                overlay_index = matched.overlay_number
                x, y, use_js_click = self.get_click_coordinates(overlay_index, elements, page_info)
                self.event_logger.system_debug(
                    f"Matched element [id={element_id}] → overlay={overlay_index} "
                    f"(type={getattr(matched, 'element_type', '?')}, "
                    f"class={getattr(matched, 'css_class', '')})"
                )
            else:
                desc = str(args.get("description", "")).strip()
                self.event_logger.system_debug(
                    f"Element [id={element_id}] not found in elements, "
                    f"falling back to description match: {desc!r}"
                )

        # LLM fallback if element_id didn't resolve
        if x is None:
            click_intent = str(args.get("description", "")).strip() or step.action
            overlay_index = self.select_best_overlay(
                click_intent,
                elements,
                failed_elements,
                screenshot=current_screenshot,
                base_knowledge=self.memory_store.base_knowledge,
            )
            if overlay_index is None:
                self.event_logger.command_failure(step.action, error="Could not determine best overlay")
                return False
            x, y, use_js_click = self.get_click_coordinates(overlay_index, elements, page_info)

        if x is None or y is None:
            self.event_logger.command_failure(step.action, error="Could not determine click coordinates")
            return False
        
        try:
            # Configurable pre-click delay to show the highlight overlay.
            # Defaults to 0 (no delay); set click_pre_highlight_ms in ExecutionConfig for visual feedback.
            _highlight_ms = int(
                getattr(
                    getattr(getattr(self.browser, "config", None), "execution", None),
                    "click_pre_highlight_ms",
                    0,
                ) or 0
            )
            if _highlight_ms > 0:
                time.sleep(_highlight_ms / 1000.0)
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

        success = False

        try:
            highlight_click_location(self, x, y)
            if use_js_click:
                self.browser.page.evaluate(
                    """(overlayIndex) => {
                        const findByShadow = (idx, root) => {
                            const found = root.querySelector(`[data-dom-index="${idx}"]`);
                            if (found) return found;
                            for (const host of root.querySelectorAll('*')) {
                                if (host.shadowRoot) {
                                    const deep = findByShadow(idx, host.shadowRoot);
                                    if (deep) return deep;
                                }
                            }
                            return null;
                        };
                        const el = findByShadow(overlayIndex, document);
                        if (el && typeof el.click === 'function') { el.click(); }
                    }""",
                    overlay_index,
                )
                try:
                    self.event_logger.system_info(
                        f"JS direct click used for overlay {overlay_index} "
                        f"(elementFromPoint blocked by overlapping element)"
                    )
                except Exception:
                    pass
            else:
                # self._human_mouse_move(x, y)
                self.browser.page.mouse.click(x, y)

            # Capture state after click
            after_state = self.memory_store._capture_current_state()

        except Exception as e:
            self.event_logger.command_failure(step.action, error=f"An error occured while clicking: {e}")
            return False

        success = True

        # Build target description from step information
        target_description = str(args.get("description", "")).strip() or None
        if not target_description and overlay_index is not None:
            target_description = f"element #{overlay_index}"
        # Get reasoning if available; fall back to memory_store's current action reasoning
        step_reasoning = step.reasoning
        if not step_reasoning:
            try:
                step_reasoning = self.memory_store.get_current_action_reasoning()
            except Exception:
                step_reasoning = None
        # Record actual interaction with goal monitor (pass explicit before_state since click already happened)
        self.memory_store.record_interaction(
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
    ) -> Dict[str, Any]:
        _ = base_knowledge
        skipped_answer = "(user skipped)"

        def _normalize_options(raw_options: Any) -> List[str]:
            if not isinstance(raw_options, list):
                return []
            normalized: List[str] = []
            for item in raw_options:
                text = str(item or "").strip()
                if text:
                    normalized.append(text)
            return normalized

        def _handle_ask_command(
            question: str,
            environment_state: EnvironmentState,
            options: List[str],
            multi_select: bool,
            yes_no: bool,
        ) -> Dict[str, Any]:
            if not self.user_question_callback:
                self.event_logger.system_warning("Agent wants to ask a question but no callback configured")
                return {"status": "unavailable", "answer": "", "answered": False, "success": False}
            
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
                    answer = self.user_question_callback(
                        question,
                        context,
                        options,
                        multi_select,
                        yes_no,
                    )
                finally:
                    # Re-enable page blocking after user responds
                    if hasattr(self.browser, '_thinking_border_manager'):
                        self.browser._thinking_border_manager.enable_blocking()

                answer_text = str(answer or "").strip()
                if answer_text:
                    self.event_logger.ask_command_answered(
                        question=question,
                        response=answer_text,
                    )
                    return {"status": "answered", "answer": answer_text, "answered": True, "success": True}
                self.event_logger.ask_command_skipped(question=question)
                return {"status": "skipped", "answer": skipped_answer, "answered": False, "success": True}
            except Exception as e:
                self.event_logger.ask_command_failure(question=question, error=str(e), details=context)
                return {"status": "failed", "answer": "", "answered": False, "success": False}

        args = self._get_action_args(step)
        question = str(args.get("question", "")).strip()
        yes_no = bool(args.get("yes_no", False))
        options = _normalize_options(args.get("options", []))
        if yes_no:
            options = ["Yes", "No"]
        multi_select = bool(args.get("multi_select", False)) and not yes_no
        if not question:
            question = step.action.split(":", 1)[1].strip() if ":" in step.action else "Need assistance"
        try:
            self.event_logger.ask_requested(
                question,
                options=options,
                multi_select=multi_select,
                yes_no=yes_no,
            )
        except Exception:
            pass

        # For now, return failure to indicate human intervention needed
        # In the future, this could pause and wait for user input
        ask_result: Dict[str, Any] = {
            "status": "failed",
            "answer": "",
            "answered": False,
            "success": False,
        }
        try:
            before_state = self.memory_store._capture_current_state()
            ask_result = _handle_ask_command(
                question,
                environment_state,
                options,
                multi_select,
                yes_no,
            )
            after_state = self.memory_store._capture_current_state()
        
            self.memory_store.record_interaction(
                InteractionType.ASK,
                before_state=before_state,
                after_state=after_state,
                coordinates=None,
                target_element_info={
                    "question": question,
                    "answer": ask_result.get("answer", ""),
                    "answered": bool(ask_result.get("answered")),
                    "status": ask_result.get("status", "failed"),
                    "options": options,
                    "multi_select": multi_select,
                    "yes_no": yes_no,
                },
                success=bool(ask_result.get("success")),
            )
            
            # Store question/answer pair for agent context if answer was received
            if ask_result.get("answered"):
                self.memory_store.add_question_answer(question, ask_result.get("answer", ""))
            
        except Exception:
            pass
        
        return ask_result

    def execute_report(
        self,
        step: ActionStep,
        environment_state: Optional[EnvironmentState],
        current_iteration: Optional[int] = None,
    ) -> tuple[bool, Dict[str, Any]]:
        """Execute report_data action - non-blocking textual payload callback."""
        args = self._get_action_args(step)
        payload = str(args.get("payload", "")).strip()
        if not payload:
            payload = step.action.split(":", 1)[1].strip() if ":" in step.action else ""

        if not payload:
            self.event_logger.system_warning("No payload provided for report_data action")
            return False, {"delivered": False}

        context = {
            "current_url": (
                getattr(environment_state, "current_url", None)
                if environment_state is not None
                else (self.browser.page.url if self.browser.page else "")
            ),
            "page_title": (
                getattr(environment_state, "page_title", None)
                if environment_state is not None
                else (self.browser.page.title() if self.browser.page else "")
            ),
            "iteration": current_iteration,
        }

        delivered = False
        callback_error: Optional[str] = None
        if not self.data_report_callback:
            self.event_logger.system_warning(
                "report_data called but no callback configured; continuing with delivered=false"
            )
        else:
            try:
                self.data_report_callback(payload, context)
                delivered = True
            except Exception as e:
                callback_error = str(e)
                self.event_logger.system_warning(
                    f"report_data callback failed; continuing with delivered=false: {callback_error}"
                )

        try:
            before_state = self.memory_store._capture_current_state()
            after_state = self.memory_store._capture_current_state()
            self.memory_store.record_interaction(
                IT.REPORT,
                before_state=before_state,
                after_state=after_state,
                target_element_info={
                    "payload": payload,
                    "delivered": delivered,
                    "context": context,
                },
                success=True,
                error_message=callback_error,
            )
        except Exception:
            pass

        return True, {
            "payload": payload,
            "reported": True,
            "delivered": delivered,
            "context": context,
            "error": callback_error,
        }

    @staticmethod
    def _sanitize_session_segment(raw: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(raw or "").strip())
        cleaned = cleaned.strip("._-")
        if not cleaned:
            return "session"
        return cleaned[:80]

    def _resolve_non_local_session_segment(self, provider_type: str) -> str:
        provider_type = str(provider_type or "").strip().lower()
        browser_cfg = getattr(getattr(self.browser, "config", None), "browser", None)

        if provider_type == "remote" and browser_cfg is not None:
            remote_cdp_url = str(getattr(browser_cfg, "remote_cdp_url", "") or "").strip()
            if remote_cdp_url:
                try:
                    parsed = urlparse(remote_cdp_url)
                    query = parse_qs(parsed.query or "")
                    for key in ("session_id", "sessionId", "browser_session_id", "browserSessionId"):
                        values = query.get(key) or []
                        if values and str(values[0]).strip():
                            return self._sanitize_session_segment(values[0])

                    parts = [part for part in parsed.path.split("/") if part]
                    for idx in range(len(parts) - 1):
                        if parts[idx].lower() in {"session", "sessions"}:
                            return self._sanitize_session_segment(parts[idx + 1])
                except Exception:
                    pass

        if provider_type == "persistent" and browser_cfg is not None:
            user_data_dir = str(getattr(browser_cfg, "user_data_dir", "") or "").strip()
            if user_data_dir:
                basename = Path(user_data_dir).expanduser().name
                if basename:
                    return self._sanitize_session_segment(basename)

        return self._write_data_session_id

    def execute_write_data(self, step: ActionStep) -> tuple[bool, Dict[str, Any]]:
        """Execute write_data action - write textual data to local disk."""
        args = self._get_action_args(step)
        data = str(args.get("data", ""))
        if data == "":
            self.event_logger.system_warning("No data provided for write_data action")
            return False, {"error": "No data provided"}

        raw_path = args.get("path", "")
        if raw_path is None:
            path_arg = ""
        else:
            path_arg = str(raw_path).strip()
        if path_arg.casefold() in {"none", "null"}:
            path_arg = ""

        raw_file_name = args.get("file_name", "")
        if raw_file_name is None:
            file_name = ""
        else:
            file_name = str(raw_file_name).strip()
        if file_name.casefold() in {"none", "null"}:
            file_name = ""

        mode = str(args.get("mode", "overwrite")).strip().lower() or "overwrite"
        if mode not in {"overwrite", "append"}:
            mode = "overwrite"
        format_hint = str(args.get("format_hint", "text")).strip().lower() or "text"

        ext_map = {
            "text": ".txt",
            "markdown": ".md",
            "json": ".json",
            "csv": ".csv",
        }
        default_ext = ext_map.get(format_hint, ".txt")
        generated_name = (
            f"data_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            f"_{uuid.uuid4().hex[:6]}{default_ext}"
        )

        provider_type = "local"
        try:
            provider_type = str(self.browser.config.browser.provider_type).strip().lower()
        except Exception:
            provider_type = "local"

        session_segment = (
            "temp"
            if provider_type == "local"
            else self._resolve_non_local_session_segment(provider_type)
        )
        used_default_location = False
        explicit_dir_hint = False

        if self.force_workspace_write_data:
            used_default_location = True
            workspace_default = str(self.workspace_paths.get("outputs_root", "")).strip()
            if workspace_default:
                target = Path(workspace_default).expanduser().resolve()
            else:
                error_message = "write_data default workspace path is not configured"
                self.event_logger.system_warning(error_message)
                return False, {"error": error_message}
            if path_arg:
                self.event_logger.system_info(
                    "write_data ignored explicit path because force_workspace_write_data is enabled",
                    requested_path=path_arg,
                    outputs_root=str(target),
                )
        elif path_arg:
            explicit_dir_hint = path_arg.endswith("/") or path_arg.endswith("\\")
            target = Path(path_arg).expanduser()
            if not target.is_absolute():
                target = (Path.cwd() / target).resolve()
        else:
            used_default_location = True
            workspace_default = str(self.workspace_paths.get("outputs_root", "")).strip()
            if workspace_default:
                target = Path(workspace_default).expanduser().resolve()
            else:
                error_message = "write_data default workspace path is not configured"
                self.event_logger.system_warning(error_message)
                return False, {"error": error_message}

        if explicit_dir_hint:
            target_is_directory = True
        elif target.exists():
            target_is_directory = target.is_dir()
        elif used_default_location:
            target_is_directory = True
        elif file_name:
            target_is_directory = True
        elif target.suffix:
            target_is_directory = False
        else:
            # Extensionless user path defaults to file path (e.g. "./notes").
            target_is_directory = False

        if target_is_directory:
            target_file = target / (file_name or generated_name)
        else:
            target_file = target

        observe_warning: Optional[str] = None
        if self.sandbox_policy is not None:
            try:
                path_decision = self.sandbox_policy.check_path(target_file, operation="write")
            except Exception as e:
                path_decision = type("Decision", (), {"allowed": False, "reason": f"Policy engine error: {e}"})()
            if not path_decision.allowed:
                warning = f"write_data blocked by sandbox: {path_decision.reason}"
                self.event_logger.system_warning(warning)
                if bool(getattr(self.sandbox_policy, "enforce", True)):
                    return False, {"error": warning}
                observe_warning = warning

        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            write_mode = "a" if mode == "append" else "w"
            with target_file.open(write_mode, encoding="utf-8") as f:
                f.write(data)
        except Exception as e:
            error_message = f"Failed to write data: {e}"
            self.event_logger.system_warning(error_message)
            return False, {"error": error_message}

        bytes_written = len(data.encode("utf-8"))
        resolved_path = str(target_file)

        data_preview = data if len(data) <= 240 else f"{data[:237]}..."
        try:
            before_state = self.memory_store._capture_current_state()
            after_state = self.memory_store._capture_current_state()
            self.memory_store.record_interaction(
                IT.WRITE_DATA,
                before_state=before_state,
                after_state=after_state,
                target_element_info={
                    "resolved_path": resolved_path,
                    "path": path_arg or None,
                    "file_name": file_name or target_file.name,
                    "mode": mode,
                    "format_hint": format_hint,
                    "bytes_written": bytes_written,
                    "provider_type": provider_type,
                    "session_segment": session_segment,
                    "used_default_location": used_default_location,
                    "force_workspace_write_data": self.force_workspace_write_data,
                },
                text_input=data_preview,
                success=True,
            )
        except Exception:
            pass

        return True, {
            "resolved_path": resolved_path,
            "bytes_written": bytes_written,
            "mode": mode,
            "format_hint": format_hint,
            "used_default_location": used_default_location,
            "force_workspace_write_data": self.force_workspace_write_data,
            "provider_type": provider_type,
            "session_segment": session_segment,
            "sandbox_warning": observe_warning,
        }

    def execute_clear_text(
        self,
        step: ActionStep,
        elements: PageElements,
        page_info: PageInfo,
        failed_elements: List[FailedAction],
    ) -> bool:
        """Execute a clear text action."""
        before_state = self.memory_store._capture_current_state()
        current_screenshot = before_state.screenshot

        success = False
        error_msg: Optional[str] = None

        def clear_input_field(x: Optional[int], y: Optional[int]) -> bool:
            if x is None or y is None:
                return False

            try:
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
                if bool(self.browser.page.evaluate(element_js)):
                    return True
            except Exception:
                pass

            try:
                self.browser.page.mouse.click(x, y)
                time.sleep(0.2)
                self.browser.page.keyboard.press("Control+a")
                time.sleep(0.1)
                self.browser.page.keyboard.press("Delete")
                time.sleep(0.1)
                return True
            except Exception:
                return False

        args = self._get_action_args(step)
        # Resolve target: element_id → LLM fallback
        overlay_index = self._resolve_element_id_to_overlay(step, elements)
        if overlay_index is None:
            overlay_index = self.select_best_overlay(
                str(args.get("field_description", "")).strip() or step.action,
                elements,
                failed_elements,
                screenshot=current_screenshot,
                base_knowledge=self.memory_store.base_knowledge,
            )

        x, y, _ = self.get_click_coordinates(overlay_index, elements, page_info)
        if x is None or y is None:
            error_msg = "Could not determine coordinates for clear_text"
        else:
            try:
                self._human_mouse_move(x, y)
                self.browser.page.mouse.click(x, y)
                time.sleep(random.uniform(0.1, 0.3))
                success = clear_input_field(x, y)
                if not success:
                    error_msg = "Unable to clear field with JS or keyboard fallback"
            except Exception as e:
                success = False
                error_msg = str(e)

        target_description = str(args.get("field_description", "")).strip() or None
        if not target_description and overlay_index is not None:
            target_description = f"element #{overlay_index}"

        after_state = self.memory_store._capture_current_state()
        self.memory_store.record_interaction(
            InteractionType.CLEAR_TEXT,
            before_state=before_state,
            after_state=after_state,
            coordinates=(x, y) if x is not None and y is not None else None,
            target_element_info={
                "description": target_description,
                "overlay_index": overlay_index,
                "action": step.action,
            } if target_description or overlay_index is not None else None,
            reasoning=None,
            success=success,
            error_message=error_msg,
        )
        return success
        
    def execute_type(
        self,
        step: ActionStep,
        elements: PageElements,
        prior_failures: List[str],
        failed_elements: List[FailedAction],
        page_info: PageInfo,
        *,
        base_knowledge: Optional[List[str]] = None,
    ) -> bool:
        """Execute a type action"""
        before_state = self.memory_store._capture_current_state()
        
        current_screenshot = before_state.screenshot

        args = self._get_action_args(step)
        text_to_type = str(args.get("text", "")).strip()
        field_description = str(args.get("field_description", "")).strip()
        if not text_to_type:
            # Fallback parser for non-function inputs
            split_action = step.action.split(":")
            text_to_type = split_action[1].strip() if len(split_action) > 1 else ""

        if not text_to_type:
            dprint("⚠️ No text specified for TYPE action")
            return False

        # Resolve target: element_id → LLM fallback
        overlay_index = self._resolve_element_id_to_overlay(step, elements)
        if overlay_index is None:
            overlay_index = self.select_best_overlay(
                field_description or step.action,
                elements,
                failed_elements,
                screenshot=current_screenshot,
                base_knowledge=self.memory_store.base_knowledge
            )

        # Get coordinates for the element to type into
        x, y, _ = self.get_click_coordinates(overlay_index, elements, page_info)
        
        # Click first to focus the element
        if x is not None and y is not None:
            self._human_mouse_move(x, y)
            self.browser.page.mouse.click(x, y)
            time.sleep(random.uniform(0.1, 0.3))

        try:
            self.event_logger.system_debug(f"Typing: {text_to_type}")
        except Exception:
            pass

        try:
            # Type without clearing so append and in-place edits remain possible.
            element_selector = None
            if x is not None and y is not None:
                try:
                    element_selector = self.selector_utils.get_element_selector_from_coordinates(x, y)
                    if element_selector:
                        try:
                            self.event_logger.system_debug(
                                f"Using locator typing with selector: {element_selector}"
                            )
                        except Exception:
                            pass
                        locator = self.browser.page.locator(element_selector).first
                        locator.focus()
                        delay = random.randint(20, 60) if len(text_to_type) < 100 else random.randint(5, 20)
                        locator.press_sequentially(text_to_type, delay=delay)
                        success = True
                        error_msg = None
                    else:
                        raise ValueError("Could not get element selector")
                except Exception as e:
                    # Only show in debug mode
                    if hasattr(self.event_logger, 'debug_mode') and self.event_logger.debug_mode:
                        self.event_logger.system_debug(
                            f"Locator typing failed, falling back to keyboard: {e}"
                        )
                    element_selector = None

            # Fallback to keyboard typing if selector-based typing didn't work
            if not element_selector:
                used_keyboard_fallback = True
                # Ensure element is focused before keyboard typing
                if x is not None and y is not None:
                    try:
                        self.browser.page.mouse.click(x, y)
                        time.sleep(0.1)
                    except Exception:
                        pass
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
        if used_keyboard_fallback:
            type_notes_parts.append("selector typing failed, used keyboard fallback")
        if not success:
            type_notes_parts.append(f"typing failed: {error_msg}" if error_msg else "typing failed")
        type_notes = "; ".join(type_notes_parts) if type_notes_parts else None

        # Build target description from step information
        target_description = field_description or None

        if not target_description and overlay_index is not None:
            target_description = f"element #{overlay_index}"

        # Get reasoning if available; fall back to memory_store's current action reasoning
        step_reasoning = getattr(step, 'reasoning', None)
        if not step_reasoning:
            try:
                step_reasoning = self.memory_store.get_current_action_reasoning()
            except Exception:
                step_reasoning = None
        after_state = self.memory_store._capture_current_state()
        # Record type interaction with goal monitor (pass explicit before_state since typing already happened)
        self.memory_store.record_interaction(
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
        Execute a scroll action. Supports three modes:
        1. Window scroll: no element_id/scroll_to_element_id — scrolls the main page.
        2. Container scroll: element_id provided — walks DOM to find the nearest
           scrollable ancestor of that element and scrolls it (modals, sidebars, etc.).
        3. Scroll-to: scroll_to_element_id provided — brings that element into view.

        Returns True if scroll moved content, False if at boundary or element not found.
        """
        args = self._get_action_args(step)
        direction = (str(args.get("direction", "")).strip() or "down").lower()
        amount_label = (str(args.get("amount", "")).strip() or "medium").lower()
        raw_element_id = args.get("element_id")
        raw_scroll_to_element_id = args.get("scroll_to_element_id")
        element_id = self._coerce_positive_int(raw_element_id)
        scroll_to_element_id = self._coerce_positive_int(raw_scroll_to_element_id)
        allow_scroll_to_fallback = bool(args.get("_allow_scroll_to_fallback", True))
        allow_container_fallback = bool(args.get("_allow_container_fallback", True))

        if raw_scroll_to_element_id is not None and scroll_to_element_id is None:
            self.event_logger.system_debug(
                f"[Scroll] Ignoring invalid scroll_to_element_id={raw_scroll_to_element_id!r}; expected positive integer"
            )
        if raw_element_id is not None and element_id is None:
            self.event_logger.system_debug(
                f"[Scroll] Ignoring invalid element_id={raw_element_id!r}; expected positive integer"
            )

        amount_px = {"small": 150, "medium": 400, "large": 800}.get(amount_label, 400)
        axis = "horizontal" if direction in ("left", "right") else "vertical"
        dx = {"right": amount_px, "left": -amount_px}.get(direction, 0)
        dy = {"down": amount_px, "up": -amount_px}.get(direction, 0)

        before_state = self.memory_store._capture_current_state()
        success = False
        error_msg = None
        target_x, target_y = 0, 0

        try:
            if scroll_to_element_id is not None:
                # Mode 3: bring element into view
                before_x = int(self.browser.page.evaluate("window.pageXOffset || window.scrollX") or 0)
                before_y = int(self.browser.page.evaluate("window.pageYOffset || window.scrollY") or 0)
                js = """
                    (idx) => {
                        const el = document.querySelector('[data-dom-index="' + idx + '"]');
                        if (!el) return false;
                        el.scrollIntoView({behavior: 'instant', block: 'nearest', inline: 'nearest'});
                        return true;
                    }
                """
                found = self.browser.page.evaluate(js, int(scroll_to_element_id))
                if not found:
                    error_msg = f"scroll_to_element_id={scroll_to_element_id} not found in DOM"
                    success = False
                else:
                    after_x = int(self.browser.page.evaluate("window.pageXOffset || window.scrollX") or 0)
                    after_y = int(self.browser.page.evaluate("window.pageYOffset || window.scrollY") or 0)
                    if self.page_utils:
                        self.page_utils.last_scroll_y = after_y
                        self.page_utils.last_scroll_x = after_x

                    if after_y != before_y or after_x != before_x:
                        success = True
                    else:
                        # Element is already in view (or scrollIntoView had no effect).
                        if not allow_scroll_to_fallback:
                            self.event_logger.system_debug(
                                f"[Scroll] scroll_to_element_id={scroll_to_element_id} caused no movement; "
                                "treated as success (no fallback requested)"
                            )
                            success = True
                        else:
                            # Fall back to directional window scroll so "scroll down" still tries to move.
                            target_x = before_x + dx
                            target_y = before_y + dy
                            self.browser.page.evaluate(f"window.scrollBy({dx}, {dy})")
                            fallback_x = int(self.browser.page.evaluate("window.pageXOffset || window.scrollX") or 0)
                            fallback_y = int(self.browser.page.evaluate("window.pageYOffset || window.scrollY") or 0)
                            if self.page_utils:
                                self.page_utils.last_scroll_y = fallback_y
                                self.page_utils.last_scroll_x = fallback_x

                            if fallback_y == before_y and fallback_x == before_x:
                                boundary = "bottom" if dy > 0 else "top" if dy < 0 else "right" if dx > 0 else "left"
                                error_msg = (
                                    f"scroll_to_element_id={scroll_to_element_id} already in view and page already at {boundary} boundary"
                                )
                                success = False
                            else:
                                self.event_logger.system_debug(
                                    f"[Scroll] scroll_to_element_id={scroll_to_element_id} caused no movement; "
                                    "fell back to directional window scroll"
                                )
                                success = True

            elif element_id is not None:
                # Mode 2: scroll the nearest scrollable ancestor of this element
                js = """
                    ([idx, dx, dy]) => {
                        const el = document.querySelector('[data-dom-index="' + idx + '"]');
                        if (!el) return {found: false};
                        let node = el.parentElement;
                        while (node && node !== document.body && node !== document.documentElement) {
                            const style = getComputedStyle(node);
                            const oy = style.overflowY;
                            const ox = style.overflowX;
                            const canY = (oy === 'auto' || oy === 'scroll') && node.scrollHeight > node.clientHeight;
                            const canX = (ox === 'auto' || ox === 'scroll') && node.scrollWidth > node.clientWidth;
                            if (canY || canX) {
                                const beforeY = node.scrollTop;
                                const beforeX = node.scrollLeft;
                                node.scrollBy(dx, dy);
                                return {found: true, no_ancestor: false, delta_y: node.scrollTop - beforeY, delta_x: node.scrollLeft - beforeX};
                            }
                            node = node.parentElement;
                        }
                        return {found: true, no_ancestor: true};
                    }
                """
                result = self.browser.page.evaluate(js, [int(element_id), dx, dy])
                if not isinstance(result, dict) or not result.get("found"):
                    error_msg = f"element_id={element_id} not found in DOM"
                    success = False
                elif result.get("no_ancestor"):
                    if not allow_container_fallback:
                        error_msg = f"No scrollable ancestor for element {element_id}"
                        success = False
                    else:
                        # No scrollable ancestor — fall back to window scroll
                        before_x = int(self.browser.page.evaluate("window.pageXOffset || window.scrollX") or 0)
                        before_y = int(self.browser.page.evaluate("window.pageYOffset || window.scrollY") or 0)
                        target_x = before_x + dx
                        target_y = before_y + dy
                        self.browser.page.evaluate(f"window.scrollBy({dx}, {dy})")
                        after_x = int(self.browser.page.evaluate("window.pageXOffset || window.scrollX") or 0)
                        after_y = int(self.browser.page.evaluate("window.pageYOffset || window.scrollY") or 0)
                        if self.page_utils:
                            self.page_utils.last_scroll_y = after_y
                            self.page_utils.last_scroll_x = after_x
                        if after_y == before_y and after_x == before_x:
                            boundary = "bottom" if dy > 0 else "top" if dy < 0 else "right" if dx > 0 else "left"
                            error_msg = f"No scrollable ancestor for element {element_id}; page already at {boundary} boundary"
                            success = False
                        else:
                            self.event_logger.system_debug(
                                f"[Scroll] No scrollable ancestor for element {element_id}, fell back to window scroll"
                            )
                            success = True
                else:
                    delta_y = result.get("delta_y", 0)
                    delta_x = result.get("delta_x", 0)
                    if delta_y == 0 and delta_x == 0:
                        boundary = "bottom" if dy > 0 else "top" if dy < 0 else "right" if dx > 0 else "left"
                        error_msg = f"Container already at {boundary} boundary"
                        success = False
                    else:
                        success = True

            else:
                # Mode 1: window scroll
                before_x = int(self.browser.page.evaluate("window.pageXOffset || window.scrollX") or 0)
                before_y = int(self.browser.page.evaluate("window.pageYOffset || window.scrollY") or 0)
                target_x = before_x + dx
                target_y = before_y + dy
                self.browser.page.evaluate(f"window.scrollBy({dx}, {dy})")
                after_x = int(self.browser.page.evaluate("window.pageXOffset || window.scrollX") or 0)
                after_y = int(self.browser.page.evaluate("window.pageYOffset || window.scrollY") or 0)
                if self.page_utils:
                    self.page_utils.last_scroll_y = after_y
                    self.page_utils.last_scroll_x = after_x
                if after_y == before_y and after_x == before_x:
                    boundary = "bottom" if dy > 0 else "top" if dy < 0 else "right" if dx > 0 else "left"
                    error_msg = f"Page already at {boundary} boundary"
                    success = False
                else:
                    success = True

        except Exception as exc:
            error_msg = str(exc)
            success = False
            self.event_logger.system_error(f"Scroll failed: {exc}")

        after_state = self.memory_store._capture_current_state()
        self.memory_store.record_interaction(
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
        self.last_failure_reason = error_msg if not success else None
        return success

    def execute_scroll_down(self, step: ActionStep) -> bool:
        """Scroll the main page down."""
        args = self._get_action_args(step)
        amount_label = (str(args.get("amount", "")).strip() or "medium").lower()
        synthetic_step = ActionStep(
            action="scroll: down",
            function_name="scroll_down",
            function_arguments={
                "direction": "down",
                "amount": amount_label,
            },
        )
        return self.execute_scroll(step=synthetic_step)

    def execute_scroll_up(self, step: ActionStep) -> bool:
        """Scroll the main page up."""
        args = self._get_action_args(step)
        amount_label = (str(args.get("amount", "")).strip() or "medium").lower()
        synthetic_step = ActionStep(
            action="scroll: up",
            function_name="scroll_up",
            function_arguments={
                "direction": "up",
                "amount": amount_label,
            },
        )
        return self.execute_scroll(step=synthetic_step)

    def execute_scroll_container(self, step: ActionStep) -> bool:
        """Scroll a specific container (requires element_id)."""
        args = self._get_action_args(step)
        raw_element_id = args.get("element_id")
        element_id = self._coerce_positive_int(raw_element_id)
        if element_id is None:
            self.last_failure_reason = (
                f"scroll_container requires a valid element_id; got {raw_element_id!r}"
            )
            return False

        direction = (str(args.get("direction", "")).strip() or "down").lower()
        if direction not in ("up", "down"):
            direction = "down"
        amount_label = (str(args.get("amount", "")).strip() or "medium").lower()
        synthetic_step = ActionStep(
            action=f"scroll_container: {direction} [id={element_id}]",
            function_name="scroll_container",
            function_arguments={
                "direction": direction,
                "amount": amount_label,
                "element_id": element_id,
                "_allow_container_fallback": False,
            },
        )
        return self.execute_scroll(step=synthetic_step)

    def execute_scroll_to_element(self, step: ActionStep) -> bool:
        """Bring a specific element into view without directional fallback."""
        args = self._get_action_args(step)
        raw_element_id = args.get("element_id")
        element_id = self._coerce_positive_int(raw_element_id)
        if element_id is None:
            self.last_failure_reason = (
                f"scroll_to_element requires a valid element_id; got {raw_element_id!r}"
            )
            return False

        synthetic_step = ActionStep(
            action=f"scroll_to: [id={element_id}]",
            function_name="scroll_to_element",
            function_arguments={
                "direction": "to_element",
                "scroll_to_element_id": element_id,
                "_allow_scroll_to_fallback": False,
            },
        )
        return self.execute_scroll(step=synthetic_step)

    def execute_press(self, step: ActionStep) -> bool:
        """Execute a key press action"""
        args = self._get_action_args(step)
        step_keys: List[str] = step.keys_to_press
        if not step_keys:
            arg_key = args.get("key")
            if isinstance(arg_key, str) and arg_key.strip():
                step_keys = [arg_key.strip()]
            else:
                # Extract the keys from the action string
                step_keys = step.action.split(":", 1)[1].strip() if ":" in step.action else ""
                if not step_keys:
                    self.event_logger.command_failure(command=step.action, error="No keys specified for PRESS action")
                    return False
                step_keys = step_keys.split(",")
            step_keys = [k.strip() for k in step_keys]

        try:
            self.event_logger.system_debug(f"Pressing keys: {step_keys}")
        except Exception:
            pass

        # Capture state BEFORE performing the press action (critical for accurate before_state)
        before_state = self.memory_store._capture_current_state()
        for key in step_keys:
            try:
                # Parse and execute the key combination
                self.parse_and_press_keys(key)
            except Exception as e:
                after_state = self.memory_store._capture_current_state()
                self.event_logger.command_failure("PRESS", error=f"Key press failed: {e}")
                self.memory_store.record_interaction(
                    InteractionType.PRESS,
                    before_state=before_state,  # Pass explicit before_state captured before the press
                    after_state=after_state,
                    keys_pressed=step_keys,
                    success=False,
                    error_message=str(e),
                    )
                return False
        after_state = self.memory_store._capture_current_state()

        self.event_logger.command_execution_complete("PRESS", success=True, keys_pressed=step.keys_to_press)
        self.memory_store.record_interaction(
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

    def get_click_coordinates(self, overlay_index: Optional[int], elements: PageElements, page_info: PageInfo) -> Tuple[Optional[int], Optional[int], bool]:
        if overlay_index is None or not getattr(elements, "elements", None):
            return None, None, False

        w, h = page_info.width, page_info.height
        selected_element = None

        for el in elements.elements:
            if getattr(el, "overlay_number", None) == overlay_index:
                selected_element = el
                break

        # Prefer a DOM-backed, hittable click point over pure box center math.
        if selected_element is not None and self.browser.page is not None:
            try:
                prefer_text_descendant = (getattr(selected_element, "element_type", "") or "").lower() == "a"
                smart_point = self.browser.page.evaluate(
                    DOM_SMART_CLICK_POINT_SCRIPT,
                    {
                        "overlayIndex": int(overlay_index),
                        "preferTextDescendant": prefer_text_descendant,
                    },
                )
                if isinstance(smart_point, dict):
                    sx = smart_point.get("x")
                    sy = smart_point.get("y")
                    if isinstance(sx, (int, float)) and isinstance(sy, (int, float)):
                        cx, cy = int(round(sx)), int(round(sy))
                        if 0 <= cx < w and 0 <= cy < h:
                            source = smart_point.get("source", "smart")
                            use_js = source == "js-click-fallback"
                            target_tag = smart_point.get("targetTag", "")
                            try:
                                self.event_logger.system_debug(
                                    f"Smart click point for overlay {overlay_index}: ({cx}, {cy}) "
                                    f"source={source} target_tag={target_tag}"
                                    + (" [JS-CLICK FALLBACK]" if use_js else "")
                                )
                            except Exception:
                                pass
                            return cx, cy, use_js
            except Exception as e:
                try:
                    self.event_logger.system_debug(
                        f"Smart click point resolver failed for overlay {overlay_index}: {e}"
                    )
                except Exception:
                    pass

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
                return cx, cy, False

        return None, None, False
    
    def execute_open(
        self,
        step: ActionStep,
    ) -> bool:
        """Open a URL directly in the current tab and record navigation."""
        # Seperate URL from action instruction
        # Robustly extract the URL from the action string (e.g., handle extra spaces and trailing quotes)
        try:
            before_state = self.memory_store._capture_current_state()
            args = self._get_action_args(step)
            url = str(args.get("url", "")).strip()
            if not url:
                url = step.action.split(":", 1)[1].strip() if ":" in step.action else ""
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

        after_state = self.memory_store._capture_current_state()
        self.memory_store.record_interaction(
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
            before_state = self.memory_store._capture_current_state()
        except Exception:
            pass

        args = self._get_action_args(step)
        raw_steps = args.get("steps", 1)
        try:
            num_back = int(raw_steps)
        except Exception:
            num_back = 1
        if num_back < 1:
            num_back = 1

        error_msg: Optional[str] = None
        for _ in range(num_back):
            try:
                self.browser.page.go_back(wait_until="domcontentloaded")
            except Exception as e:
                error_msg = str(e)
                dprint(f"  ❌ Back navigation failed: {e}")
                break

        after_state = self.memory_store._capture_current_state()
        after_url = before_url
        try:
            after_url = self.browser.page.url
        except Exception:
            if after_state and getattr(after_state, "url", ""):
                after_url = after_state.url

        state_changed = False
        try:
            state_changed = bool(
                self.memory_store._has_meaningful_change(before_state, after_state)
            )
        except Exception:
            if before_state and after_state:
                state_changed = (
                    before_state.url != after_state.url
                    or before_state.title != after_state.title
                )
            elif before_url and after_url:
                state_changed = before_url != after_url

        success = error_msg is None and state_changed
        if error_msg is None and not state_changed:
            error_msg = "Back navigation did not change the page state."

        self.event_logger.command_execution_complete("BACK", success=success, target_description=after_url, reasoning=None)

        # Record navigation interaction with explicit before_state
        try:
            self.memory_store.record_interaction(
                InteractionType.NAVIGATION,
                before_state=before_state,  # Pass explicit before_state since navigation already happened
                after_state=after_state,
                target_element_info={
                    "direction": "back",
                    "steps": num_back,
                    "from": before_url,
                    "to": after_url,
                },
                success=success,
                error_message=error_msg,
            )
        except Exception:
            pass
        return success

    # ------------------------------------------------------------------
    # Dropdown helpers
    # ------------------------------------------------------------------

    def _extract_dropdown_options(self) -> List[str]:
        """Extract visible option texts from any open dropdown in the DOM."""
        js = """
        () => {
            const texts = new Set();

            // Native <select> options (from the focused or last select)
            const sel = document.querySelector('select:focus') || document.querySelector('select');
            if (sel) {
                for (const opt of sel.options) {
                    const t = opt.textContent.trim();
                    if (t) texts.add(t);
                }
            }

            // ARIA role-based options
            for (const el of document.querySelectorAll('[role="option"], [role="listbox"] > *')) {
                const t = el.textContent.trim();
                if (t && el.offsetParent !== null) texts.add(t);
            }

            // Common custom dropdown patterns
            const selectors = [
                '.dropdown-menu li', '.dropdown-menu a',
                '[class*="option"]', '[class*="menu-item"]',
                '[class*="listbox"] > *', '[class*="select"] [class*="option"]',
                'ul[class*="dropdown"] li', 'div[class*="dropdown"] div[class*="item"]',
                '[data-value]',
            ];
            for (const s of selectors) {
                try {
                    for (const el of document.querySelectorAll(s)) {
                        if (el.offsetParent === null) continue;
                        const t = el.textContent.trim();
                        if (t && t.length < 200) texts.add(t);
                    }
                } catch {}
            }

            return [...texts].slice(0, 200);
        }
        """
        try:
            return self.browser.page.evaluate(js) or []
        except Exception:
            return []

    def _poll_for_dropdown_options(self, max_wait_ms: int = 2000, poll_interval_ms: int = 300) -> List[str]:
        """Poll _extract_dropdown_options until results stabilize (2 consecutive same-count polls)."""
        elapsed = 0
        prev_count = -1
        options: List[str] = []
        while elapsed < max_wait_ms:
            time.sleep(poll_interval_ms / 1000)
            elapsed += poll_interval_ms
            options = self._extract_dropdown_options()
            if len(options) > 0 and len(options) == prev_count:
                return options
            prev_count = len(options)
        return options

    def _is_dropdown_still_open(self) -> bool:
        """Check if a dropdown/listbox is still visibly open."""
        js = """
        () => {
            // ARIA listbox / options
            for (const el of document.querySelectorAll('[role="listbox"], [role="option"]')) {
                if (el.offsetParent !== null) return true;
            }
            // Common dropdown classes
            const selectors = ['.dropdown-menu', '[class*="menu"][class*="open"]',
                               '[class*="listbox"]', '[class*="select"][class*="open"]',
                               '[class*="dropdown"][class*="show"]'];
            for (const s of selectors) {
                try {
                    for (const el of document.querySelectorAll(s)) {
                        if (el.offsetParent !== null) return true;
                    }
                } catch {}
            }
            // Native <select> with :focus
            const sel = document.querySelector('select:focus');
            if (sel) return true;
            return false;
        }
        """
        try:
            return bool(self.browser.page.evaluate(js))
        except Exception:
            return False

    def _click_option_in_open_dropdown(self, text: str) -> bool:
        """Find an option element by text in an open dropdown and click it."""
        js = """
        (targetText) => {
            const candidates = [
                ...document.querySelectorAll('[role="option"]'),
                ...document.querySelectorAll('[role="listbox"] > *'),
                ...document.querySelectorAll('.dropdown-menu li, .dropdown-menu a'),
                ...document.querySelectorAll('[class*="option"]'),
                ...document.querySelectorAll('[class*="menu-item"]'),
                ...document.querySelectorAll('[data-value]'),
            ];
            const lower = targetText.toLowerCase();
            let exact = null;
            let partial = null;
            for (const el of candidates) {
                if (el.offsetParent === null) continue;
                const t = el.textContent.trim();
                if (t.toLowerCase() === lower) { exact = el; break; }
                if (!partial && t.toLowerCase().includes(lower)) { partial = el; }
            }
            const match = exact || partial;
            if (!match) return false;
            match.scrollIntoView({ block: 'nearest' });
            match.click();
            return true;
        }
        """
        try:
            return bool(self.browser.page.evaluate(js, text))
        except Exception:
            return False

    def _normalize_dropdown_value(self, value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()

    def _dropdown_values_match(self, expected_value: str, observed_value: Optional[str]) -> bool:
        expected_normalized = self._normalize_dropdown_value(expected_value)
        observed_normalized = self._normalize_dropdown_value(observed_value)
        if not expected_normalized or not observed_normalized:
            return False
        return (
            expected_normalized == observed_normalized
            or expected_normalized in observed_normalized
            or observed_normalized in expected_normalized
        )

    def _read_selected_dropdown_value(self, overlay_index: Optional[int]) -> Optional[str]:
        js = """
        (overlayIndex) => {
            const findByShadow = (idx, root) => {
                if (!idx) return null;
                const found = root.querySelector(`[data-dom-index="${idx}"]`);
                if (found) return found;
                for (const host of root.querySelectorAll('*')) {
                    if (host.shadowRoot) {
                        const deep = findByShadow(idx, host.shadowRoot);
                        if (deep) return deep;
                    }
                }
                return null;
            };

            const isVisible = (node) => {
                if (!node || !(node instanceof Element)) return false;
                const style = window.getComputedStyle(node);
                if (!style) return false;
                if (style.display === 'none' || style.visibility === 'hidden') return false;
                const rect = node.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0;
            };

            const textOf = (node) => {
                if (!node) return '';
                const raw = (
                    node.innerText ||
                    node.textContent ||
                    node.getAttribute?.('aria-label') ||
                    ''
                );
                return String(raw || '').trim();
            };

            const selectedDescendantText = (root) => {
                if (!root || !root.querySelectorAll) return '';
                const selectors = [
                    '[role="option"][aria-selected="true"]',
                    '[role="treeitem"][aria-selected="true"]',
                    '[role="option"][data-selected="true"]',
                    '[aria-checked="true"]',
                    'option:checked',
                    '[selected]',
                ];
                for (const selector of selectors) {
                    const candidates = Array.from(root.querySelectorAll(selector));
                    for (const candidate of candidates) {
                        if (!isVisible(candidate)) continue;
                        const text = textOf(candidate);
                        if (text) return text;
                    }
                }
                return '';
            };

            const valuesFor = (node) => {
                if (!node) return [];
                const values = [];
                if (node.tagName === 'SELECT') {
                    const selectedOption = node.selectedOptions && node.selectedOptions[0];
                    if (selectedOption) {
                        values.push(textOf(selectedOption));
                        values.push(String(selectedOption.value || '').trim());
                    }
                    values.push(String(node.value || '').trim());
                }
                if (typeof node.value === 'string') {
                    values.push(String(node.value || '').trim());
                }
                const activeDescendantId = node.getAttribute?.('aria-activedescendant');
                if (activeDescendantId) {
                    const activeDescendant = document.getElementById(activeDescendantId);
                    if (activeDescendant) {
                        values.push(textOf(activeDescendant));
                    }
                }
                values.push(selectedDescendantText(node));
                values.push(textOf(node));
                return values.map((value) => String(value || '').trim()).filter(Boolean);
            };

            const target = findByShadow(overlayIndex, document);
            const seen = new Set();
            const orderedNodes = [target, document.activeElement];
            for (const node of orderedNodes) {
                if (!node || seen.has(node)) continue;
                seen.add(node);
                const values = valuesFor(node);
                for (const value of values) {
                    if (value) return value;
                }
            }
            const globalSelected = selectedDescendantText(document.body);
            if (globalSelected) return globalSelected;
            return '';
        }
        """
        try:
            observed_value = self.browser.page.evaluate(js, overlay_index)
        except Exception:
            return None
        observed_text = str(observed_value or "").strip()
        return observed_text or None

    def _llm_choose_option(
        self,
        intent: str,
        dropdown_description: str,
        available_options: Optional[List[str]],
    ) -> Optional[ChosenOption]:
        """Use LLM to choose the best dropdown option using mission context.

        When *available_options* is provided, the LLM picks the best match and
        returns ``exact_text``.  When it is ``None`` (type-to-search dropdown
        with no initial options), the LLM generates a ``search_term`` instead.
        """
        # -- Build context from memory_store --
        mission = getattr(self.memory_store, "current_mission", "") or ""
        base_knowledge = getattr(self.memory_store, "base_knowledge", []) or []
        qa_pairs = getattr(self.memory_store, "question_answer_pairs", []) or []

        context_parts: List[str] = []
        if mission:
            context_parts.append(f"Current mission: {mission}")
        if base_knowledge:
            context_parts.append("Known facts about the user:\n" + "\n".join(f"- {k}" for k in base_knowledge))
        if qa_pairs:
            qa_lines = [f"Q: {p.get('question','')} → A: {p.get('answer','')}" for p in qa_pairs[-5:]]
            context_parts.append("Recent Q&A:\n" + "\n".join(qa_lines))

        context_block = "\n\n".join(context_parts) if context_parts else "No additional context."

        if available_options is not None:
            truncated = available_options[:100]
            numbered = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(truncated))
            prompt = (
                f"Dropdown field: \"{dropdown_description}\"\n"
                f"Intent: \"{intent}\"\n\n"
                f"Context:\n{context_block}\n\n"
                f"Available options:\n{numbered}\n\n"
                f"Pick the best option. Return its EXACT text in exact_text. "
                f"Leave search_term null."
            )
        else:
            prompt = (
                f"Dropdown field: \"{dropdown_description}\"\n"
                f"Intent: \"{intent}\"\n\n"
                f"Context:\n{context_block}\n\n"
                f"This is a type-to-search dropdown with no visible options yet. "
                f"Generate a short search_term to type that will surface the right option. "
                f"Leave exact_text null."
            )

        try:
            result = generate_model(
                prompt=prompt,
                model_object_type=ChosenOption,
                system_prompt=(
                    "You choose dropdown options using context about the user and their mission. "
                    "Return the best choice with your reasoning."
                ),
                model="gpt-4o-mini",
                temperature=0.0,
            )
            if isinstance(result, ChosenOption):
                return result
        except Exception as exc:
            dprint(f"[select] LLM choose failed: {exc}")
        return None

    # ------------------------------------------------------------------
    # Main select handler
    # ------------------------------------------------------------------

    def _is_native_select(self, overlay_index: Optional[int], elements: PageElements) -> bool:
        """Check if the resolved overlay element is a native <select>."""
        if overlay_index is None:
            return False
        for elem in elements.elements:
            if getattr(elem, "overlay_number", None) == overlay_index:
                etype = getattr(elem, "element_type", "")
                fsubtype = getattr(elem, "field_subtype", "") or ""
                return etype == "select" or fsubtype == "select"
        return False

    def _try_playwright_select(self, chosen_text: str) -> bool:
        """Attempt Playwright's native select_option on the focused <select>."""
        js = """
        () => {
            const sel = document.querySelector('select:focus');
            if (!sel) return null;
            // Build a unique CSS selector for Playwright
            if (sel.id) return '#' + CSS.escape(sel.id);
            if (sel.name) return 'select[name=' + JSON.stringify(sel.name) + ']';
            return null;
        }
        """
        try:
            selector = self.browser.page.evaluate(js)
            if selector:
                self.browser.page.select_option(selector, label=chosen_text)
                return True
        except Exception as exc:
            dprint(f"[select] Playwright select_option failed: {exc}")
        return False

    def execute_select_option(
        self,
        step: ActionStep,
        elements: PageElements,
        failed_elements: List[FailedAction],
        page_info: PageInfo,
    ) -> bool:
        """Context-aware autonomous dropdown selection.

        Flow:
        1. Resolve element_id (required, fail fast)
        2. Click to open dropdown
        3. Extract options
        4. Branch A (options found): LLM picks best match → exact_text
           - Native <select>: use Playwright select_option API
           - Custom dropdown: click the option directly, fall back to type+Enter
        5. Branch B (no options / type-to-search): LLM generates search_term →
           type it → poll for options → LLM picks from results
        6. If no chosen_text resolved, fail (never type raw intent)
        """
        args = self._get_action_args(step)
        intent = str(args.get("intent", "")).strip()
        dropdown_description = str(args.get("dropdown_description", "")).strip()
        before_state = self.memory_store._capture_current_state()

        # 1. Resolve element_id — required, fail fast
        overlay_index = self._resolve_element_id_to_overlay(step, elements)
        if overlay_index is None:
            after_state = self.memory_store._capture_current_state()
            self.memory_store.record_interaction(
                InteractionType.SELECT,
                before_state=before_state,
                after_state=after_state,
                coordinates=None,
                target_element_info={
                    "description": dropdown_description or "dropdown",
                    "overlay_index": None,
                    "intent": intent,
                    "action": step.action,
                },
                success=False,
                error_message="element_id could not be resolved — required for select_option",
            )
            return False

        native_select = self._is_native_select(overlay_index, elements)
        x, y, _ = self.get_click_coordinates(overlay_index, elements, page_info)
        success = False
        error_msg: Optional[str] = None
        chosen_text: Optional[str] = None
        observed_value: Optional[str] = None
        llm_reasoning: Optional[str] = None

        if not intent:
            error_msg = "No intent provided for select_option"
        elif x is None or y is None:
            error_msg = "Could not determine coordinates for select_option"
        else:
            try:
                # 2. Click to open dropdown
                self.browser.page.mouse.click(x, y)
                time.sleep(0.3)

                # 3. Extract available options
                options = self._extract_dropdown_options()

                typed_to_search = False

                if options:
                    # --- Flow A: options visible ---
                    choice = self._llm_choose_option(intent, dropdown_description, options)
                    if choice and choice.exact_text:
                        chosen_text = choice.exact_text
                        llm_reasoning = choice.reasoning
                else:
                    # --- Flow B: type-to-search ---
                    choice = self._llm_choose_option(intent, dropdown_description, None)
                    if choice and choice.search_term:
                        self.browser.page.keyboard.type(choice.search_term, delay=30)
                        typed_to_search = True
                        llm_reasoning = choice.reasoning

                        # Poll for async options
                        polled = self._poll_for_dropdown_options(max_wait_ms=2000, poll_interval_ms=300)
                        if polled:
                            # Re-run LLM with actual options
                            choice2 = self._llm_choose_option(intent, dropdown_description, polled)
                            if choice2 and choice2.exact_text:
                                chosen_text = choice2.exact_text
                                llm_reasoning = choice2.reasoning

                # --- Guard: never type raw intent into a dropdown ---
                if not chosen_text:
                    base_knowledge = getattr(self.memory_store, "base_knowledge", []) or []
                    if not base_knowledge:
                        error_msg = (
                            f"Could not determine a value for \"{dropdown_description}\" "
                            f"(intent: \"{intent}\"). No user profile data available — "
                            f"consider using ask_user to get the value first."
                        )
                    else:
                        error_msg = (
                            f"Could not determine a value for \"{dropdown_description}\" "
                            f"(intent: \"{intent}\"). LLM did not return a match."
                        )
                    # Close the dropdown so the page isn't left in a broken state
                    self.browser.page.keyboard.press("Escape")
                    time.sleep(0.1)
                else:
                    # --- Apply chosen_text ---

                    # Native <select>: prefer Playwright API
                    selection_applied = native_select and self._try_playwright_select(chosen_text)
                    if not selection_applied:
                        selection_applied = self._click_option_in_open_dropdown(chosen_text)
                    if selection_applied:
                        time.sleep(0.3)
                    else:
                        # Clear search text if we typed to search
                        if typed_to_search:
                            self.browser.page.keyboard.press("Control+a")
                            time.sleep(0.05)

                        # Type chosen text + Enter
                        self.browser.page.keyboard.type(chosen_text, delay=30)
                        time.sleep(0.15)
                        self.browser.page.keyboard.press("Enter")
                        time.sleep(0.3)

                    observed_value = self._read_selected_dropdown_value(overlay_index)
                    if self._dropdown_values_match(chosen_text, observed_value):
                        success = True
                    else:
                        error_msg = (
                            f"select_option did not set the intended value "
                            f"'{chosen_text}' (observed: '{observed_value or 'none'}')"
                        )
                        success = False
            except Exception as exc:
                success = False
                error_msg = str(exc)

        after_state = self.memory_store._capture_current_state()
        self.memory_store.record_interaction(
            InteractionType.SELECT,
            before_state=before_state,
            after_state=after_state,
            coordinates=(x, y) if x is not None and y is not None else None,
            target_element_info={
                "description": dropdown_description or "dropdown",
                "overlay_index": overlay_index,
                "intent": intent,
                "chosen_text": chosen_text,
                "observed_value": observed_value,
                "llm_reasoning": llm_reasoning,
                "action": step.action,
            },
            success=success,
            error_message=error_msg,
        )
        return success


    def _resolve_workspace_file(self, file_path: str) -> Optional[Path]:
        """Resolve a file path relative to the workspace root.

        Returns the resolved Path if the file exists under workspace_root,
        None otherwise.
        """
        workspace_root_str = str(self.workspace_paths.get("workspace_root", "")).strip()
        if not workspace_root_str:
            return None
        workspace_root = Path(workspace_root_str).resolve()
        if not workspace_root.is_dir():
            return None

        candidate = Path(file_path)
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
        candidate = candidate.resolve()

        # Must be under workspace_root
        try:
            candidate.relative_to(workspace_root)
        except ValueError:
            return None

        if candidate.is_file():
            return candidate
        return None

    def _resolve_upload_target_selector(
        self,
        *,
        selector: Optional[str],
        x: Optional[float],
        y: Optional[float],
    ) -> Optional[str]:
        """Resolve the actual file input associated with the clicked upload control."""
        token = f"codex-upload-target-{int(time.time() * 1000)}"
        try:
            resolved = self.browser.page.evaluate(
                """
                ({ selector, x, y, token }) => {
                    const markerAttr = "data-codex-upload-target";

                    const removeMarkers = () => {
                        for (const node of document.querySelectorAll(`[${markerAttr}]`)) {
                            node.removeAttribute(markerAttr);
                        }
                    };

                    const isVisible = (element) => {
                        if (!element) return false;
                        const style = window.getComputedStyle(element);
                        if (style.display === "none" || style.visibility === "hidden") return false;
                        const rect = element.getBoundingClientRect();
                        return rect.width > 0 && rect.height > 0;
                    };

                    const resolveFromElement = (element) => {
                        if (!element) return null;
                        if (element.matches?.('input[type="file"]')) return element;

                        const closestFileInput = element.closest?.('input[type="file"]');
                        if (closestFileInput) return closestFileInput;

                        const closestLabel = element.closest?.("label");
                        if (closestLabel) {
                            const htmlFor = closestLabel.getAttribute("for");
                            if (htmlFor) {
                                const linked = document.getElementById(htmlFor);
                                if (linked?.matches?.('input[type="file"]')) return linked;
                            }
                            const nestedInput = closestLabel.querySelector?.('input[type="file"]');
                            if (nestedInput) return nestedInput;
                        }

                        let current = element;
                        while (current) {
                            const nestedInput = current.querySelector?.('input[type="file"]');
                            if (nestedInput) return nestedInput;
                            current = current.parentElement;
                        }
                        return null;
                    };

                    const findNearestVisibleFileInput = (cx, cy) => {
                        let best = null;
                        let bestDistance = Number.POSITIVE_INFINITY;
                        for (const input of document.querySelectorAll('input[type="file"]')) {
                            if (!isVisible(input)) continue;
                            const rect = input.getBoundingClientRect();
                            const centerX = rect.left + rect.width / 2;
                            const centerY = rect.top + rect.height / 2;
                            const distance = Math.hypot(centerX - cx, centerY - cy);
                            if (distance < bestDistance) {
                                bestDistance = distance;
                                best = input;
                            }
                        }
                        if (bestDistance <= 240) return best;
                        return null;
                    };

                    removeMarkers();

                    let target = null;
                    if (selector) {
                        try {
                            target = document.querySelector(selector);
                        } catch (error) {
                            target = null;
                        }
                    }

                    let cx = Number.isFinite(x) ? x : null;
                    let cy = Number.isFinite(y) ? y : null;
                    if (cx !== null && cy !== null) {
                        if (cx < 0 || cx > window.innerWidth || cy < 0 || cy > window.innerHeight) {
                            cx = cx - window.scrollX;
                            cy = cy - window.scrollY;
                        }
                    }

                    if (!target && cx !== null && cy !== null) {
                        target = document.elementFromPoint(cx, cy);
                    }

                    let input = resolveFromElement(target);
                    if (!input && cx !== null && cy !== null) {
                        input = findNearestVisibleFileInput(cx, cy);
                    }

                    if (!input) {
                        const allInputs = Array.from(document.querySelectorAll('input[type="file"]'));
                        if (allInputs.length === 1) {
                            input = allInputs[0];
                        }
                    }

                    if (!input) return null;
                    input.setAttribute(markerAttr, token);
                    return `input[type="file"][${markerAttr}="${token}"]`;
                }
                """,
                {
                    "selector": selector,
                    "x": x,
                    "y": y,
                    "token": token,
                },
            )
        except Exception:
            return None

        if isinstance(resolved, str) and resolved.strip():
            return resolved.strip()
        return None

    def _clear_upload_target_selector(self) -> None:
        try:
            self.browser.page.evaluate(
                """
                () => {
                    for (const node of document.querySelectorAll('[data-codex-upload-target]')) {
                        node.removeAttribute('data-codex-upload-target');
                    }
                }
                """
            )
        except Exception:
            pass

    def execute_upload(
        self,
        step: ActionStep,
        elements: PageElements,
        page_info: PageInfo,
        failed_elements: Optional[List[FailedAction]] = None,
        *,
        confirm_before_interaction: bool = False,
    ) -> bool:
        """Execute a file upload action respecting upload_mode config."""
        args = self._get_action_args(step)
        failed_elements = failed_elements or []
        before_state = self.memory_store._capture_current_state()
        file_path = str(args.get("file_path", "")).strip()
        target_description = str(args.get("target_description", "")).strip()
        current_screenshot = before_state.screenshot

        overlay_index = self._resolve_element_id_to_overlay(step, elements)
        if overlay_index is None:
            intent = f"upload file {file_path} in {target_description}".strip()
            overlay_index = self.select_best_overlay(
                intent,
                elements,
                failed_elements,
                screenshot=current_screenshot,
                base_knowledge=self.memory_store.base_knowledge,
            )

        x, y, _ = self.get_click_coordinates(overlay_index, elements, page_info)

        success = False
        error_msg: Optional[str] = None
        selector: Optional[str] = None
        resolved_path: Optional[Path] = None
        effective_mode = self.upload_mode

        if effective_mode == "workspace_only":
            resolved_path = self._resolve_workspace_file(file_path) if file_path else None
            if resolved_path:
                success, error_msg, selector = self._upload_set_input_files(
                    str(resolved_path), x, y
                )
            else:
                error_msg = (
                    f"File not found in workspace: {file_path}"
                    if file_path
                    else "No file path provided for upload"
                )

        elif effective_mode == "user_only":
            if x is None or y is None:
                error_msg = "Could not determine upload target coordinates"
            else:
                try:
                    self.browser.page.mouse.click(x, y)
                    success = True
                except Exception as exc:
                    error_msg = str(exc)
                if success and self.agent_talk_callback:
                    msg = f"Please select a file to upload for: {target_description}" if target_description else "Please select a file to upload."
                    try:
                        self.agent_talk_callback(msg)
                    except Exception:
                        pass

        else:  # auto
            if file_path:
                resolved_path = self._resolve_workspace_file(file_path)
            if resolved_path:
                success, error_msg, selector = self._upload_set_input_files(
                    str(resolved_path), x, y
                )
            else:
                # Fall back to user_only behavior
                if x is None or y is None:
                    error_msg = "Could not determine upload target coordinates"
                else:
                    try:
                        self.browser.page.mouse.click(x, y)
                        success = True
                    except Exception as exc:
                        error_msg = str(exc)
                    if success and self.agent_talk_callback:
                        msg = f"Please select a file to upload for: {target_description}" if target_description else "Please select a file to upload."
                        try:
                            self.agent_talk_callback(msg)
                        except Exception:
                            pass

        after_state = self.memory_store._capture_current_state()
        self.memory_store.record_interaction(
            InteractionType.UPLOAD,
            before_state=before_state,
            after_state=after_state,
            coordinates=(x, y) if x is not None and y is not None else None,
            target_element_info={
                "overlay_index": overlay_index,
                "description": target_description,
                "selector": selector,
                "upload_mode": effective_mode,
                "resolved_path": str(resolved_path) if resolved_path else None,
            },
            notes=file_path,
            success=success,
            error_message=error_msg,
        )
        return success

    def _upload_set_input_files(
        self, file_path: str, x: Optional[float], y: Optional[float]
    ) -> tuple:
        """Programmatically set input files. Returns (success, error_msg, selector)."""
        if x is None or y is None:
            return False, "Could not determine upload target coordinates", None
        selector: Optional[str] = None
        try:
            selector = self.selector_utils.get_element_selector_from_coordinates(x, y)
        except Exception:
            selector = None
        upload_target_selector = self._resolve_upload_target_selector(
            selector=selector,
            x=x,
            y=y,
        )
        attempted_selectors: list[str] = []
        try:
            for candidate_selector in [upload_target_selector, selector]:
                if not candidate_selector or candidate_selector in attempted_selectors:
                    continue
                attempted_selectors.append(candidate_selector)
                try:
                    self.browser.page.locator(candidate_selector).first.set_input_files(file_path)
                    return True, None, candidate_selector
                except Exception:
                    continue

            if upload_target_selector:
                return False, f"Failed to set files on resolved upload target: {upload_target_selector}", upload_target_selector
            if selector:
                return False, f"Resolved selector is not a usable file input: {selector}", selector
            return False, "Could not resolve a file input near the chosen upload target", None
        except Exception as exc:
            return False, str(exc), selector
        finally:
            self._clear_upload_target_selector()


    def execute_datetime(
        self,
        step: ActionStep,
        elements: PageElements,
        page_info: PageInfo,
        failed_elements: Optional[List[FailedAction]] = None,
        *,
        confirm_before_interaction: bool = False,
    ) -> bool:
        """Execute a datetime field action."""
        args = self._get_action_args(step)
        failed_elements = failed_elements or []
        before_state = self.memory_store._capture_current_state()
        datetime_value = str(args.get("value", "")).strip()
        picker_description = str(args.get("picker_description", "")).strip()
        current_screenshot = before_state.screenshot

        overlay_index = self._resolve_element_id_to_overlay(step, elements)
        if overlay_index is None:
            intent = f"set datetime {datetime_value} in {picker_description}".strip()
            overlay_index = self.select_best_overlay(
                intent,
                elements,
                failed_elements,
                screenshot=current_screenshot,
                base_knowledge=self.memory_store.base_knowledge,
            )
        x, y, _ = self.get_click_coordinates(overlay_index, elements, page_info)

        success = False
        error_msg: Optional[str] = None
        selector: Optional[str] = None
        if not datetime_value:
            error_msg = "No datetime value provided"
        elif x is None or y is None:
            error_msg = "Could not determine datetime target coordinates"
        else:
            try:
                selector = self.selector_utils.get_element_selector_from_coordinates(x, y)
            except Exception:
                selector = None

            try:
                if selector:
                    locator = self.browser.page.locator(selector).first
                    locator.fill(datetime_value)
                else:
                    self.browser.page.mouse.click(x, y)
                    self.browser.page.keyboard.type(datetime_value, delay=30)
                success = True
            except Exception as exc:
                success = False
                error_msg = str(exc)

        after_state = self.memory_store._capture_current_state()
        self.memory_store.record_interaction(
            InteractionType.DATETIME,
            before_state=before_state,
            after_state=after_state,
            coordinates=(x, y) if x is not None and y is not None else None,
            target_element_info={
                "overlay_index": overlay_index,
                "datetime_value": datetime_value,
                "description": picker_description,
                "selector": selector,
            },
            text_input=datetime_value,
            success=success,
            error_message=error_msg,
        )
        return success
    
    def extract(
        self,
        step: ActionStep,
        extraction_schema: Optional[Dict[str, Any]] = None,
    ) -> Union[bool, str]:
        args = self._get_action_args(step)
        extraction_prompt = str(args.get("data_description", "")).strip()
        if not extraction_prompt:
            extraction_prompt = step.action.replace("extract:", "").strip()
        self.event_logger.extraction_start(extraction_prompt)
        self.event_logger.extraction_detected(extraction_prompt)

        # Capture before state with screenshot
        before_state = self.memory_store._capture_current_state()

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
            You are given a webpage's screenshots and its page content. Your goal is to extract the information requested by the user.

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
            extraction_user_prompt = f"""
                        Extract the following information from this webpage screenshot:
                        {extraction_prompt}

                        Your goal is to extract the following information from the webpage screenshot: {extraction_prompt}
                        Do not make up text that isn't in the provided content."""
            if not extraction_schema:
                # Simple text extraction using vision, grounded with page text
                result_text = generate_text(
                    prompt=extraction_user_prompt,
                    system_prompt=extraction_system_prompt,
                    image=screenshot,
                    image_detail="high"
                )
                extracted_text = result_text.strip()

                # Capture after state with screenshot
                after_state = self.memory_store._capture_current_state()

                # Record extraction in interaction history
                self.memory_store.record_interaction(
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
                        description=extraction_prompt,
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
                self.memory_store.record_interaction(
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
                        description=extraction_prompt,
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
            self.memory_store.record_interaction(
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
        args = self._get_action_args(step)
        reasoning = str(args.get("reasoning", "")).strip()
        next_action = str(args.get("next_action", "continue")).strip().lower() or "continue"
        recommended_next_step = str(args.get("recommended_next_step", "")).strip()
        stuck_pattern = str(args.get("stuck_pattern", "")).strip()
        if not reasoning:
            reasoning = step.action.split(":", 1)[1].strip() if ":" in step.action else ""

        if not reasoning:
            self.event_logger.system_warning("No reasoning provided for think action")
            return False

        # Capture state for record keeping
        before_state = self.memory_store._capture_current_state()
        after_state = before_state  # No change for think

        # Record the think interaction
        self.memory_store.record_interaction(
            IT.THINK,
            before_state=before_state,
            after_state=after_state,
            reasoning=reasoning,
            next_action=next_action,
            recommended_next_step=recommended_next_step or None,
            stuck_pattern=stuck_pattern or None,
            success=True,
        )

        self.event_logger.system_info(f"🤔 Agent thinking: {reasoning}")
        return True

    def execute_assert(self, step: ActionStep) -> bool:
        """Execute an assert action - check a condition from the screenshot."""
        args = self._get_action_args(step)
        condition = str(args.get("condition", "")).strip()
        if not condition:
            condition = step.action.split(":", 1)[1].strip() if ":" in step.action else ""

        if not condition:
            self.event_logger.system_warning("No condition provided for assert action")
            return False

        # Capture state for record keeping
        before_state = self.memory_store._capture_current_state()
        after_state = before_state  # No change for assert

        # Record the assert interaction
        self.memory_store.record_interaction(
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
        args = self._get_action_args(step)
        message = str(args.get("message", "")).strip()
        if not message:
            message = step.action.split(":", 1)[1].strip() if ":" in step.action else ""

        if not message:
            self.event_logger.system_warning("No message provided for flag action")
            return False

        # Capture state for record keeping
        before_state = self.memory_store._capture_current_state()
        after_state = before_state  # No change for flag

        # Record the flag interaction
        self.memory_store.record_interaction(
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
        args = self._get_action_args(step)
        condition = str(args.get("condition", "")).strip()
        timeout_seconds = args.get("timeout_seconds", 10)
        try:
            timeout_seconds = int(timeout_seconds)
        except Exception:
            timeout_seconds = 10

        # Fallback parser for non-function calls.
        parts = ""
        if not condition:
            parts = step.action.split(":", 1)[1].strip() if ":" in step.action else ""
            condition = parts
            if "|" in parts:
                condition_part, timeout_part = parts.split("|", 1)
                condition = condition_part.strip()
                timeout_match = re.search(r'timeout=(\d+)', timeout_part)
                if timeout_match:
                    timeout_seconds = int(timeout_match.group(1))

        if not condition:
            self.event_logger.system_warning("No condition provided for wait_for action")
            return False

        # Capture before state
        before_state = self.memory_store._capture_current_state()

        # Perform the wait - use a simple sleep for now
        # TODO: Could be enhanced with actual page.wait_for_selector or similar
        import time
        self.event_logger.system_info(f"⏳ Waiting for: {condition} (timeout: {timeout_seconds}s)")
        time.sleep(min(timeout_seconds, 5))  # Cap at 5 seconds for safety

        # Capture after state
        after_state = self.memory_store._capture_current_state()

        # Record the wait interaction
        self.memory_store.record_interaction(
            IT.WAIT_FOR,
            before_state=before_state,
            after_state=after_state,
            target_element_info={"condition": condition, "timeout": timeout_seconds},
            success=True,
        )

        return True

    def execute_via_adapter(
        self,
        *,
        function_name: str,
        function_arguments: Dict[str, Any],
        detected_elements: PageElements,
        page_info: PageInfo,
        environment_state: Optional[EnvironmentState] = None,
        base_knowledge: Optional[List[str]] = None,
        current_iteration: Optional[int] = None,
    ) -> ActionResult:
        """Adapter entrypoint used by the declarative tool runtime for executor-backed tools."""
        step = ActionStep.from_function_call(function_name, dict(function_arguments or {}))
        return self.act(
            action_step=step,
            detected_elements=detected_elements,
            page_info=page_info,
            environment_state=environment_state,
            base_knowledge=base_knowledge,
            current_iteration=current_iteration,
        )

    def act(
        self,
        action_step: ActionStep,
        detected_elements: PageElements,
        page_info: PageInfo = None,
        prior_failures: Optional[List[str]] = None,
        failed_elements: Optional[List[FailedAction]] = None,
        extraction_schema: Optional[Dict[str, Any]] = None,
        confirm_before_interaction: bool = False,
        action_id: Optional[str] = None,
        environment_state: EnvironmentState = None,
        max_attempts: Optional[int] = None,
        base_knowledge: Optional[List[str]] = None,
        current_iteration: Optional[int] = None,
        **kwargs
    ) -> ActionResult:
        command = self._get_action_command(action_step)
        function_name = (getattr(action_step, "function_name", None) or "").strip()
        prior_failures = prior_failures or []
        failed_elements = failed_elements or []

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

        function_args = self._get_action_args(action_step)
        self.last_failure_reason = None
        try:
            self.memory_store.set_current_action_context(
                reasoning=getattr(action_step, "reasoning", None) or function_args.get("reasoning"),
                memory_evidence_ids=function_args.get("memory_evidence_ids"),
                stuck_pattern=function_args.get("stuck_pattern"),
            )
        except Exception:
            pass
            
        try:
            # Start action timer
            # self.execution_timer.start_action(action_id, command)

            # Add action to history
            self._add_to_command_history(command)

            if not function_name:
                duration = time.time() - start_time
                error_message = "Missing function_name on action step"
                self.event_logger.command_failure(
                    command=command,
                    error=error_message,
                    duration_ms=duration * 1000,
                )
                return _create_result(
                    False,
                    error_message,
                    error=error_message,
                    action_id=action_id,
                    duration=duration,
                )

            # Safety gate: block browser actions when a dialog is pending.
            if self.browser.tab_manager and self.browser.tab_manager.has_pending_dialog_on_active():
                browser_functions = {
                    "click",
                    "type_text",
                    "clear_text",
                    "select_option",
                    "scroll_down",
                    "scroll_up",
                    "scroll_container",
                    "scroll_to_element",
                    "press_key",
                    "open_url",
                    "go_back",
                    "go_forward",
                    "upload_file",
                    "extract_data",
                }
                if function_name in browser_functions:
                    duration = time.time() - start_time
                    error_message = "Cannot interact with page — a dialog is blocking. Use dismiss_dialog first."
                    self.event_logger.command_failure(
                        command=command,
                        error=error_message,
                        duration_ms=duration * 1000,
                    )
                    return _create_result(
                        False,
                        error_message,
                        error=error_message,
                        action_id=action_id,
                        duration=duration,
                    )

            self.event_logger.command_start(command=command)

            controller_only_functions = {
                "switch_tab",
                "close_tab",
                "open_tab",
                "dismiss_dialog",
                "send_email",
            }
            if function_name in controller_only_functions:
                duration = time.time() - start_time
                error_message = f"{function_name} should be handled by agent controller, not executor"
                self.event_logger.command_failure(
                    command=command,
                    error=error_message,
                    duration_ms=duration * 1000,
                )
                return _create_result(
                    False,
                    error_message,
                    error=error_message,
                    action_id=action_id,
                    duration=duration,
                )

            result_data: Optional[Any] = None
            if function_name == "click":
                executed = self.execute_click(
                    step=action_step,
                    elements=detected_elements,
                    prior_failures=prior_failures,
                    failed_elements=failed_elements,
                    page_info=page_info,
                    base_knowledge=base_knowledge,
                )
            elif function_name == "type_text":
                executed = self.execute_type(
                    step=action_step,
                    elements=detected_elements,
                    prior_failures=prior_failures,
                    failed_elements=failed_elements,
                    page_info=page_info,
                    base_knowledge=base_knowledge,
                )
            elif function_name == "clear_text":
                executed = self.execute_clear_text(
                    step=action_step,
                    elements=detected_elements,
                    page_info=page_info,
                    failed_elements=failed_elements,
                )
            elif function_name == "select_option":
                executed = self.execute_select_option(
                    step=action_step,
                    elements=detected_elements,
                    failed_elements=failed_elements,
                    page_info=page_info,
                )
            elif function_name == "upload_file":
                executed = self.execute_upload(
                    step=action_step,
                    elements=detected_elements,
                    page_info=page_info,
                    failed_elements=failed_elements,
                    confirm_before_interaction=confirm_before_interaction,
                )
            elif function_name == "press_key":
                executed = self.execute_press(step=action_step)
            elif function_name == "open_url":
                executed = self.execute_open(step=action_step)
            elif function_name == "go_back":
                executed = self.execute_back(step=action_step)
            elif function_name == "go_forward":
                executed = self.execute_forward(step=action_step)
            elif function_name == "scroll_down":
                executed = self.execute_scroll_down(step=action_step)
            elif function_name == "scroll_up":
                executed = self.execute_scroll_up(step=action_step)
            elif function_name == "scroll_container":
                executed = self.execute_scroll_container(step=action_step)
            elif function_name == "scroll_to_element":
                executed = self.execute_scroll_to_element(step=action_step)
            elif function_name == "extract_data":
                extract_result = self.extract(
                    step=action_step,
                    extraction_schema=extraction_schema,
                )
                if isinstance(extract_result, tuple):
                    executed, extraction_error = extract_result
                else:
                    executed, extraction_error = False, "Extraction failed"
                if extraction_error:
                    result_data = {"error": extraction_error}
            elif function_name == "think":
                executed = self.execute_think(step=action_step)
            elif function_name == "assert_condition":
                executed = self.execute_assert(step=action_step)
            elif function_name == "flag":
                executed = self.execute_flag(step=action_step)
            elif function_name == "wait_for":
                executed = self.execute_wait_for(step=action_step)
            elif function_name == "ask_user":
                ask_result = self.execute_ask(
                    step=action_step,
                    environment_state=environment_state,
                    base_knowledge=base_knowledge,
                )
                executed = bool(ask_result.get("success"))
                ask_question = str(function_args.get("question", "")).strip()
                if not ask_question:
                    ask_question = (
                        action_step.action.split(":", 1)[1].strip()
                        if ":" in action_step.action
                        else ""
                    )
                result_data = {
                    "question": ask_question,
                    "answer": ask_result.get("answer", ""),
                    "answered": bool(ask_result.get("answered")),
                    "status": ask_result.get("status", "failed"),
                }
            elif function_name == "report_data":
                executed, report_data = self.execute_report(
                    step=action_step,
                    environment_state=environment_state,
                    current_iteration=current_iteration,
                )
                result_data = report_data
            elif function_name == "write_data":
                executed, write_data_result = self.execute_write_data(step=action_step)
                result_data = write_data_result
            else:
                duration = time.time() - start_time
                error_message = f"Unsupported function: {function_name}"
                self.event_logger.command_failure(
                    command=command,
                    error=error_message,
                    duration_ms=duration * 1000,
                )
                return _create_result(
                    False,
                    error_message,
                    error=error_message,
                    action_id=action_id,
                    duration=duration,
                )

            executed = bool(executed)
            duration = time.time() - start_time
            failure_detail: Optional[str] = None
            if isinstance(result_data, dict):
                result_error = result_data.get("error")
                if isinstance(result_error, str) and result_error.strip():
                    failure_detail = result_error.strip()
            if not failure_detail:
                last_reason = (self.last_failure_reason or "").strip()
                if last_reason:
                    failure_detail = last_reason
            if executed:
                try:
                    self.event_logger.command_success(command)
                except Exception:
                    pass
            else:
                error_message = f"{function_name} execution failed"
                if failure_detail:
                    error_message = f"{error_message}: {failure_detail}"
                self.event_logger.command_failure(
                    command=command,
                    error=error_message,
                    duration_ms=duration * 1000,
                )

            overlay_index = self.memory_store.get_last_interaction_overlay_index()
            metadata = {"command_type": "function_call", "function_name": function_name}
            if overlay_index is not None:
                metadata["overlay_index"] = overlay_index

            return _create_result(
                executed,
                "Action executed successfully" if executed else "Action failed",
                error=None if executed else (failure_detail or f"{function_name} execution failed"),
                action_id=action_id,
                duration=duration,
                additional_metadata=metadata,
                data=result_data,
            )
        except Exception as e:
            duration = time.time() - start_time
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

        sorted_elements = sorted(
            element_data.elements,
            key=lambda e: (
                -int(getattr(e, "text_presence_score", 0) or 0),
                int(getattr(e, "overlay_number", 10**9) or 10**9),
            ),
        )

        for elem in sorted_elements:
            idx = elem.overlay_number
            # role = elem.role_hint
            tag = elem.element_type
            text = elem.element_label
            element_label = elem.element_label
            is_focused = elem.is_focused
            text_score = int(getattr(elem, "text_presence_score", 0) or 0)
            has_visible_text = bool(getattr(elem, "has_visible_text", False))
            tag_str = tag or "unknown"
            # Build comprehensive description
            parts = []
            parts.append(
                f"Overlay {idx} tag={tag_str} text={text} placeholder={element_label} "
                f"text_score={text_score} has_text={str(has_visible_text).lower()} is-focused={is_focused}"
            )
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
            Overlay <index> tag=<tag> text=<text> type=<type> text_score=<0-3> has_text=<true|false>
            - index is the overlay number/index
            - tag is the HTML tag of the element
            - text is the text of the element
            - type is the type of the element
            - text_score is a heuristic of visible text signal strength
            - has_text indicates visible on-screen text presence
            Use all of these together to decide which overlay index is the best match.
            When matching a named control by label/text, prefer higher text_score/has_text=true candidates.

            {base_knowledge_section}
        """

        # Add failed elements to the prompt
        if failed_elements:
            # Format failed actions with context
            prior_failure_lines = []
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
                prior_failure_lines.append(f"  • {action_display}{context_str}")
            
            prompt += f"""
            
            Failed actions:
            - You tried these and they didn't work:
            {prior_failure_lines}
            
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
 
