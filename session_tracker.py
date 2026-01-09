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


class InteractionType(str, Enum):
    """Types of browser interactions"""
    CLICK = "click"
    TYPE = "type"
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
    DEFER = "defer"


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
class Interaction:
    """Record of a browser interaction"""
    timestamp: float
    interaction_type: InteractionType
    coordinates: Optional[tuple[int, int]] = None
    target_element_info: Optional[Dict[str, Any]] = None
    text_input: Optional[str] = None
    keys_pressed: Optional[str] = None
    scroll_direction: Optional[str] = None
    scroll_axis: Optional[str] = None
    target_x: Optional[int] = None
    target_y: Optional[int] = None
    navigation_url: Optional[str] = None
    before_state: Optional[BrowserState] = None
    after_state: Optional[BrowserState] = None
    success: bool = True
    error_message: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    extraction_prompt: Optional[str] = None
    reasoning: Optional[str] = None  # Why this action was taken
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


class SessionTracker:
    """
    Lightweight session tracker for agent mode.
    
    Tracks interactions, browser state, and URL history without goal evaluation.
    Provides all the data structures that agent mode needs without the complexity
    of goal evaluation and management.
    """
    
    def __init__(self, page: Page):
        self.page = page
        self.interaction_history: List[Interaction] = []
        self.state_history: List[BrowserState] = []
        self.url_history: List[str] = []
        self.url_pointer: int = -1
        self.session_start_time = time.time()
        self.base_knowledge: List[str] = []
        self.user_prompt: str = ""
        self._current_action_reasoning: Optional[str] = None  # Store reasoning for next interaction
        
        # Capture initial state
        self._capture_initial_state()
    
    def switch_to_page(self, page: Page) -> None:
        """Update page reference when active page changes"""
        if page and page != self.page:
            self.page = page
    
    def set_base_knowledge(self, knowledge: List[str]) -> None:
        """Set base knowledge rules"""
        self.base_knowledge = knowledge or []
    
    def set_user_prompt(self, prompt: str) -> None:
        """Set the user prompt"""
        self.user_prompt = prompt
    
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

    def _capture_current_state(self, include_screenshot: bool = False) -> BrowserState:
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
            if include_screenshot:
                try:
                    if self.page:
                        screenshot = self.page.screenshot(type="jpeg", quality=50, full_page=False)
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
            self.state_history.append(initial_state)
            if initial_state.url:
                self.url_history.append(initial_state.url)
                self.url_pointer = 0
        except Exception:
            pass
    
    def record_interaction(self, interaction_type: InteractionType, **kwargs) -> None:
        """
        Record an interaction that has occurred.
        Simple tracking without goal evaluation.
        """
        # For navigation interactions, use provided before_state if available (since navigation already happened)
        # Otherwise capture current state as before_state
        before_state = kwargs.get('before_state') or self._capture_current_state()
        
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
            success=kwargs.get('success', True),
            error_message=kwargs.get('error_message'),
            extracted_data=kwargs.get('extracted_data'),
            extraction_prompt=kwargs.get('extraction_prompt'),
            reasoning=kwargs.get('reasoning') or self._current_action_reasoning  # Why this action was taken
        )
        
        # Clear reasoning after using it (it's only for the next interaction)
        self._current_action_reasoning = None
        
        # Capture state after interaction (small delay for page updates)
        time.sleep(0.1)
        interaction.after_state = self._capture_current_state()
        
        self.interaction_history.append(interaction)
        if interaction.after_state:
            self.state_history.append(interaction.after_state)
        
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
        # Keep url_history as it's useful for navigation context
        # self.url_history.clear()
        # self.url_pointer = -1
