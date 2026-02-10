"""
Session Tracker - Lightweight interaction and state tracking for agent mode.

This module provides tracking capabilities without goal evaluation logic.
It maintains interaction history, browser state snapshots, and URL history
for use by agent mode and other components.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import time

from playwright.sync_api import Page

from core.browser import Browser


class InteractionType(str, Enum):
    """Types of browser interactions"""
    CLICK = "click"
    TYPE = "type"
    CLEAR_TEXT = "clear_text"
    SCROLL = "scroll"
    PRESS = "press"
    SELECT = "select"
    UPLOAD = "upload"
    DATETIME = "datetime"
    NAVIGATION = "navigation"
    PAGE_LOAD = "page_load"
    ELEMENT_APPEAR = "element_appear"
    ELEMENT_DISAPPEAR = "element_disappear"
    CONTEXT_GUARD = "context_guard"
    EXTRACT = "extract"
    THINK = "think"
    ASSERT = "assert"
    FLAG = "flag"
    WAIT_FOR = "wait_for"
    MARK_PROGRESS = "mark_progress"


@dataclass
class BrowserState:
    """Snapshot of browser state at a point in time"""
    timestamp: float
    url: str
    title: str
    page_width: int
    page_height: int
    scroll_x: int
    scroll_y: int
    screenshot: Optional[bytes] = None
    dom_snapshot: Optional[str] = None
    visible_text: Optional[str] = None
    page_source: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class OverlayCapture:
    timestamp: float
    url: str
    command: str
    element_data: List[Dict[str, Any]]
    screenshot: Optional[bytes]

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class Interaction:
    """Record of a browser interaction"""
    timestamp: float
    interaction_type: InteractionType
    before_state: BrowserState
    after_state: BrowserState
    coordinates: Optional[tuple[int, int]] = None
    target_element_info: Optional[Dict[str, Any]] = None
    text_input: Optional[str] = None
    keys_pressed: Optional[str] = None
    scroll_direction: Optional[str] = None
    scroll_axis: Optional[str] = None
    target_x: Optional[int] = None
    target_y: Optional[int] = None
    navigation_url: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    extraction_prompt: Optional[str] = None
    reasoning: Optional[str] = None  # Why this action was taken
    notes: Optional[str] = None  # Executor feedback (e.g., "fill failed, used keyboard")

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

    def summary_line(self, step_number: int) -> str:
        """Generate a human-readable summary line for this interaction."""
        
        # Map action types to past tense verbs
        action_verb_map = {
            "click": "clicked",
            "type": "type",
            "clear_text": "clear text",
            "scroll": "scroll",
            "press": "press",
            "select": "select",
            "upload": "upload",
            "datetime": "set datetime",
            "navigation": "navigate",
            "page_load": "load page",
            "element_appear": "wait for element to appear",
            "element_disappear": "wait for element to disappear",
            "context_guard": "guard context",
            "extract": "extract",
            "think": "think",
            "assert": "check",
            "flag": "flag",
            "wait_for": "wait for",
            "mark_progress": "mark progress",
        }
        
        # Get past tense verb for the action
        action_type = self.interaction_type.value
        past_tense_verb = action_verb_map.get(action_type, action_type)

        # Add details based on type
        details = []
        if self.text_input:
            details.append(f"'{self.text_input}'")

        if self.target_element_info:
            elem_desc = (self.target_element_info.get('description', '') or '')
            if elem_desc:
                details.append(f"on {elem_desc}")

        if self.extracted_data:
            data_count = len(self.extracted_data.get('items', [])) if isinstance(self.extracted_data, dict) and 'items' in self.extracted_data else len(self.extracted_data) if isinstance(self.extracted_data, (list, dict)) else 1
            details.append(f"extracted {data_count} item(s)")

        if self.error_message:
            details.append(f"ERROR: {self.error_message}")

        # Build the action description
        action_description = past_tense_verb
        if details:
            action_description += " " + " ".join(details)

        # Page context (from after_state if available)
        page_info = ""
        if self.after_state:
            page_title = (self.after_state.title or 'page')
            page_url = self.after_state.url if self.after_state.url else ""
            page_info = f" @ {page_title} ({page_url})"

        # Reasoning
        reasoning = (self.reasoning or "").replace("\n", " ").strip()
        reasoning_text = f" — {reasoning}" if reasoning else ""

        # Executor notes (feedback about how the action was executed)
        notes_text = f" [{self.notes}]" if self.notes else ""

        # Format: "1. You successfully clicked ..." or "2. You failed to type ..."
        success_prefix = "successfully" if self.success else "failed to"

        return f"{step_number}. You {success_prefix} {action_description}{notes_text}{page_info}{reasoning_text}"


class SessionTracker:
    """
    Lightweight session tracker for agent mode.
    
    Tracks interactions, browser state, and URL history without goal evaluation.
    Provides all the data structures that agent mode needs without the complexity
    of goal evaluation and management.
    """
    
    def __init__(self, browser: Browser):
        self.page = browser.page
        self.interaction_history: List[Interaction] = []
        self.url_history: List[str] = []
        self.url_pointer: int = -1
        self.session_start_time = time.time()
        self.base_knowledge: List[str] = []
        self.user_prompt: str = ""
        self._current_action_reasoning: Optional[str] = None  # Store reasoning for next interaction
        self._last_overlay_capture: Optional[OverlayCapture] = None
        # Store question/answer pairs from ask: commands for agent context
        self.question_answer_pairs: List[Dict[str, str]] = []

        # Capture initial state
        self._capture_initial_state()
    
    def set_base_knowledge(self, knowledge: List[str]) -> None:
        """Set base knowledge rules"""
        self.base_knowledge = knowledge or []
    
    def set_user_prompt(self, prompt: str) -> None:
        """Set the user prompt"""
        self.user_prompt = prompt
    
    def add_question_answer(self, question: str, answer: str) -> None:
        """Add a question/answer pair from an ask: command for agent context."""
        self.question_answer_pairs.append({
            "question": question,
            "answer": answer
        })
    
    def set_current_action_reasoning(self, reasoning: Optional[str]) -> None:
        """Set the reasoning for the next action to be executed"""
        self._current_action_reasoning = reasoning

    def get_current_action_reasoning(self) -> Optional[str]:
        """Retrieve and clear the stored reasoning for the next action."""
        reasoning = self._current_action_reasoning
        self._current_action_reasoning = None
        return reasoning

    def set_current_action_overlay_index(self, overlay_index: Optional[int]) -> None:
        """Set the overlay index for the next action to be executed"""
        self._current_action_overlay_index = overlay_index

    def get_current_action_overlay_index(self) -> Optional[int]:
        """Get the overlay index for the current action"""
        return getattr(self, '_current_action_overlay_index', None)

    def clear_current_action_overlay_index(self) -> None:
        """Clear the overlay index after action execution"""
        self._current_action_overlay_index = None

    def get_last_interaction_overlay_index(self) -> Optional[int]:
        """Get the overlay index from the most recent interaction"""
        if not self.interaction_history:
            return None
        
        last_interaction = self.interaction_history[-1]
        if last_interaction.target_element_info and isinstance(last_interaction.target_element_info, dict):
            return last_interaction.target_element_info.get("overlay_index")
        return None

    def store_overlay_capture(
        self,
        *,
        command: str,
        element_data: List[Dict[str, Any]],
        screenshot: Optional[bytes],
    ) -> None:
        try:
            url = self.page.url if self.page else ""
        except Exception:
            url = ""
        self._last_overlay_capture = OverlayCapture(
            timestamp=time.time(),
            url=url,
            command=command,
            element_data=element_data or [],
            screenshot=screenshot,
        )

    def get_overlay_capture(self) -> Optional[OverlayCapture]:
        capture = self._last_overlay_capture
        if not capture:
            return None
        try:
            current_url = self.page.url if self.page else ""
        except Exception:
            current_url = ""
        if current_url and capture.url and (capture.url != current_url):
            return None
        return capture

    def clear_overlay_capture(self) -> None:
        self._last_overlay_capture = None

    def _capture_current_state(self) -> BrowserState:
        """Capture current browser state"""
        try:
            url = self.page.url if self.page else ""
            title = ""
            try:
                title = self.page.title() if self.page else ""
            except Exception:
                pass
            
            # Get viewport size
            viewport_size = self.page.viewport_size if self.page else {"width": 0, "height": 0}
            page_width = viewport_size.get("width", 0) if viewport_size else 0
            page_height = viewport_size.get("height", 0) if viewport_size else 0
            
            # Get scroll position
            scroll_x = 0
            scroll_y = 0
            try:
                if self.page:
                    scroll_x = self.page.evaluate("window.scrollX || 0") or 0
                    scroll_y = self.page.evaluate("window.scrollY || 0") or 0
            except Exception:
                pass
            
            # Get visible text (first 2000 chars) - VIEWPORT ONLY
            visible_text = ""
            try:
                if self.page:
                    # Only capture text from elements visible in the current viewport
                    # This matches the viewport-only screenshot
                    viewport_text = self.page.evaluate("""
                        () => {
                            const viewportHeight = window.innerHeight;
                            const viewportWidth = window.innerWidth;
                            const textParts = [];

                            // Get all text nodes that are visible in viewport
                            const walker = document.createTreeWalker(
                                document.body,
                                NodeFilter.SHOW_TEXT,
                                {
                                    acceptNode: function(node) {
                                        // Skip empty text nodes
                                        if (!node.textContent.trim()) {
                                            return NodeFilter.FILTER_REJECT;
                                        }

                                        // Check if parent element is visible in viewport
                                        const parent = node.parentElement;
                                        if (!parent) return NodeFilter.FILTER_REJECT;

                                        // Skip hidden elements
                                        const style = window.getComputedStyle(parent);
                                        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
                                            return NodeFilter.FILTER_REJECT;
                                        }

                                        // Check if element is in viewport
                                        const rect = parent.getBoundingClientRect();
                                        const isInViewport = (
                                            rect.top < viewportHeight &&
                                            rect.bottom > 0 &&
                                            rect.left < viewportWidth &&
                                            rect.right > 0
                                        );

                                        return isInViewport ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
                                    }
                                }
                            );

                            let node;
                            while (node = walker.nextNode()) {
                                const text = node.textContent.trim();
                                if (text) {
                                    textParts.push(text);
                                }
                            }

                            return textParts.join(' ');
                        }
                    """)
                    visible_text = viewport_text[:2000] if viewport_text else ""
            except Exception:
                pass
            
            # Optionally capture screenshot (expensive, so only when needed)
            screenshot = None
            try:
                if self.page:
                    screenshot = self.page.screenshot(full_page=False)
            except Exception:
                pass
            
            return BrowserState(
                timestamp=time.time(),
                url=url,
                title=title,
                page_width=page_width,
                page_height=page_height,
                scroll_x=scroll_x,
                scroll_y=scroll_y,
                visible_text=visible_text,
                screenshot=screenshot
            )
        except Exception as e:
            # Fallback minimal state
            return BrowserState(
                timestamp=time.time(),
                url=self.page.url if self.page else "",
                title="",
                page_width=0,
                page_height=0,
                scroll_x=0,
                scroll_y=0
            )
    
    def _capture_initial_state(self) -> None:
        """Capture initial browser state"""
        try:
            initial_state = self._capture_current_state()
            if initial_state.url:
                self.url_history.append(initial_state.url)
                self.url_pointer = 0
        except Exception:
            pass
    
    def record_interaction(self, interaction_type: InteractionType, before_state: BrowserState, after_state: BrowserState, **kwargs) -> None:
        """
        Record an interaction that has occurred.
        Simple tracking without goal evaluation.
        """

        interaction = Interaction(
            timestamp=time.time(),
            interaction_type=interaction_type,
            coordinates=kwargs.get('coordinates'),
            target_element_info=kwargs.get('target_element_info'),
            text_input=kwargs.get('text_input'),
            keys_pressed=kwargs.get('keys_pressed'),
            scroll_direction=kwargs.get('scroll_direction'),
            scroll_axis=kwargs.get('scroll_axis'),
            target_x=kwargs.get('target_x'),
            target_y=kwargs.get('target_y'),
            navigation_url=kwargs.get('navigation_url'),
            before_state=before_state,
            after_state=after_state,
            success=kwargs.get('success', True),
            error_message=kwargs.get('error_message'),
            extracted_data=kwargs.get('extracted_data'),
            extraction_prompt=kwargs.get('extraction_prompt'),
            reasoning=kwargs.get('reasoning') or self._current_action_reasoning,  # Why this action was taken
            notes=kwargs.get('notes'),  # Executor feedback
        )
        
        # Clear reasoning after using it (it's only for the next interaction)
        self._current_action_reasoning = None
        
        self.interaction_history.append(interaction)
        
        # Update URL history
        current_url = interaction.after_state.url if interaction.after_state else (self.page.url if self.page else "")
        if current_url and (not self.url_history or self.url_history[-1] != current_url):
            self.url_history.append(current_url)
            self.url_pointer = len(self.url_history) - 1
        
        # Emit event
        try:
            from utils.event_logger import get_event_logger
            event_details = {}
            if interaction.reasoning:
                event_details['reasoning'] = interaction.reasoning
            get_event_logger().interaction_recorded(
                interaction_type=interaction_type.value if hasattr(interaction_type, 'value') else str(interaction_type),
                **event_details
            )
        except Exception:
            pass

    def history_block(self, limit: Optional[int] = 20, just_data: bool = False) -> str:
        """
        Generate a formatted history block for LLM prompts.

        Args:
            limit: Maximum number of recent interactions to include (None = all)
            just_data: If True, return only the interaction lines without the full template

        Returns:
            Formatted string suitable for inclusion in LLM prompts
        """
        interactions = self.interaction_history

        if limit is not None and limit > 0:
            interactions = interactions[-limit:]

        if not interactions:
            return "HISTORY: <none yet>"

        lines_str = ""
        for i, interaction in enumerate(interactions, start=1):
            lines_str += f"{interaction.summary_line(step_number=i)}\n"
        # Build question/answer pairs section if available
        qa_section = ""
        if self.question_answer_pairs:
            qa_section = ""
            for i, qa in enumerate(self.question_answer_pairs, 1):
                qa_section += f"{i}. Q: {qa['question']}\n   A: {qa['answer']}\n"
        else:
            qa_section = "No conversation history recorded yet."
            
        if just_data:
            return lines_str
        else:
            return f"""
        
        Here's what you've done so far. Use this as a reference when deciding what to do next:
        
        When planning your next move, look at both what you've already tried (listed below) and what's currently 
        on the page. If something isn't working—like you've tried clicking the same button twice and nothing's 
        happening, or the page isn't changing, or you're not making progress—try a different approach. There's 
        usually an obvious alternative action you can take, so try that before asking for help.
        
        Here's what you've tried and what happened:
        {lines_str}
        
        Conversation history:
        Below you'll also see the back-and-forth between you and the user—questions you asked and their answers.
        
        When what the user told you in conversation doesn't match what you already know, go with what they said 
        in the conversation. For example, if you asked where they want to travel and they said "London", use 
        "London" when filling out location fields, not whatever you might have known before.
        
        Sometimes what the user says will directly go against what you already know. Like they might say they 
        want to go to "London" in the summer, but you previously knew they wanted "Egypt" in the summer. When 
        that happens, just ask them to clarify—they know what they actually want better than what you knew before.
        
        Pay attention when the user gives you instructions about how to work. If they say things like "Don't do X", 
        "From now on, always Y", or "Only use Z", follow those instructions completely. The only time you can 
        ignore a direct instruction is if there's another instruction that conflicts with it or information that 
        doesn't match. One thing though—asking questions is core to how you work, so you can't stop doing that 
        even if asked.
        
        For example, if a user answers a question with "... and just use African countries for the location", 
        that means whenever you need to pick a country, stick to African ones.
        
        Here's what you've discussed with the user:
        {qa_section}
        """

    def detect_state_change(self, before: BrowserState, after: BrowserState) -> bool:
        """
        Detect if a meaningful change occurred between two browser states.

        Returns:
            True if something changed, False if nothing happened
        """
        if not before or not after:
            return False

        # Check URL change (navigation)
        if before.url != after.url:
            return True

        # Check title change
        if before.title != after.title:
            return True

        # Check scroll position change (user scrolled or page auto-scrolled)
        if abs(before.scroll_y - after.scroll_y) > 10 or abs(before.scroll_x - after.scroll_x) > 10:
            return True

        # Check visible text change (DOM changed)
        if before.visible_text and after.visible_text:
            # Normalize whitespace for comparison
            before_text = ' '.join(before.visible_text.split())
            after_text = ' '.join(after.visible_text.split())

            # If text is significantly different (>5% change), something happened
            if before_text != after_text:
                # Calculate simple difference ratio
                max_len = max(len(before_text), len(after_text))
                if max_len > 0:
                    # Use a simple character-level comparison
                    common_length = len(set(before_text) & set(after_text))
                    diff_ratio = 1.0 - (common_length / max_len)
                    if diff_ratio > 0.05:  # 5% threshold
                        return True

        # If we get here, nothing meaningful changed
        return False
