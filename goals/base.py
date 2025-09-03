"""
Base classes and types for the goal framework.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from playwright.sync_api import Page
from pydantic import BaseModel, Field


# =============================================================================
# Core Framework Types
# =============================================================================

class GoalStatus(str, Enum):
    """Status of goal evaluation"""
    PENDING = "pending"        # Goal not yet achieved
    ACHIEVED = "achieved"      # Goal successfully completed
    FAILED = "failed"          # Goal cannot be achieved
    UNKNOWN = "unknown"        # Cannot determine status


class InteractionType(str, Enum):
    """Types of browser interactions"""
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    NAVIGATION = "navigation"
    PAGE_LOAD = "page_load"
    ELEMENT_APPEAR = "element_appear"
    ELEMENT_DISAPPEAR = "element_disappear"


class EvaluationTiming(str, Enum):
    """When a goal should be evaluated relative to interactions"""
    BEFORE = "before"           # Evaluate before interaction occurs
    AFTER = "after"             # Evaluate after interaction completes
    BOTH = "both"               # Evaluate both before and after
    CONTINUOUS = "continuous"   # Evaluate continuously (polling-based)


@dataclass
class BrowserState:
    """Comprehensive snapshot of browser state at a point in time"""
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
    before_state: Optional[BrowserState] = None
    after_state: Optional[BrowserState] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class GoalContext:
    """Complete context provided to goals for evaluation"""
    initial_state: BrowserState
    current_state: BrowserState
    all_interactions: List[Interaction] = field(default_factory=list)
    url_history: List[str] = field(default_factory=list)
    page_changes: List[BrowserState] = field(default_factory=list)
    detected_elements: List[Any] = field(default_factory=list)  # Current page elements
    session_duration: float = 0.0
    # For pre-interaction evaluation
    planned_interaction: Optional[Dict[str, Any]] = None
    # Browser access for advanced goals (like navigation preview)
    page_reference: Optional[Page] = None  # Playwright Page object


class GoalResult(BaseModel):
    """Result of goal evaluation"""
    status: GoalStatus
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the assessment")
    reasoning: str = Field(description="Explanation of why this status was determined")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Supporting evidence")
    next_actions: List[str] = Field(default_factory=list, description="Suggested next steps")


# =============================================================================
# Abstract Base Goal Class
# =============================================================================

class BaseGoal(ABC):
    """
    Abstract base class for all goal types.
    
    Goals monitor browser interactions and state changes to determine
    when a user's objective has been achieved.
    """
    
    # Subclasses should override this to specify when they want to be evaluated
    EVALUATION_TIMING: EvaluationTiming = EvaluationTiming.AFTER
    
    def __init__(self, description: str, max_retries: int = 3, **kwargs):
        self.description = description
        self.created_at = time.time()
        self.kwargs = kwargs
        self._monitoring_active = False
        self._last_evaluation: Optional[GoalResult] = None
        self.max_retries = max_retries
        self.retry_count = 0
        self.retry_requested = False
    
    @abstractmethod
    def evaluate(self, context: GoalContext) -> GoalResult:
        """
        Evaluate whether this goal has been achieved given the current context.
        
        Args:
            context: Complete browser state and interaction history
            
        Returns:
            GoalResult with status and reasoning
        """
        pass
    
    @abstractmethod
    def get_description(self, context: GoalContext) -> str:
        """
        Generate a detailed description of what this goal is looking for.
        This description should be used to help align plan generation with goal requirements.
        
        The description should be structured and informative, providing:
        1. Clear goal statement (what needs to be accomplished)
        2. Current state analysis (what's already done vs what's needed)
        3. Specific requirements (exact fields, targets, or criteria)
        4. Status indicators (progress, errors, availability)
        5. Actionable guidance (what the plan should focus on)
        
        Format guidelines:
        - Use clear section headers and bullet points
        - Be specific about quantities and names
        - Highlight what's completed vs what's pending
        - Mention any errors or issues that need addressing
        - Keep it concise but comprehensive
        
        Example structure:
        ```
        Goal Type: [goal description]
        Target/Intent: [specific target]
        Current Status: [what's been done]
        Requirements: [what still needs to be done]
        Issues: [any problems to address]
        Progress: [completion metrics]
        ```
        
        Args:
            context: Complete browser state and interaction history
            
        Returns:
            String description formatted for AI plan generation consumption
        """
        pass
    
    def start_monitoring(self) -> None:
        """Called when goal monitoring begins"""
        self._monitoring_active = True
        self.on_monitoring_start()
    
    def stop_monitoring(self) -> None:
        """Called when goal monitoring ends"""
        self._monitoring_active = False
        self.on_monitoring_stop()
    
    def on_monitoring_start(self) -> None:
        """Override to perform setup when monitoring starts"""
        pass
    
    def on_monitoring_stop(self) -> None:
        """Override to perform cleanup when monitoring stops"""
        pass
    
    def on_interaction(self, interaction: Interaction) -> None:
        """
        Called after each browser interaction.
        Override to perform goal-specific tracking.
        """
        pass
    
    def on_state_change(self, old_state: BrowserState, new_state: BrowserState) -> None:
        """
        Called when browser state changes significantly.
        Override to track state-dependent goals.
        """
        pass
    
    def request_retry(self, reason: str = "Goal requested retry") -> bool:
        """
        Request a retry of the current plan/action.
        
        Returns:
            True if retry is allowed (under max_retries limit), False otherwise
        """
        if self.retry_count >= self.max_retries:
            print(f"[{self.__class__.__name__}] Max retries ({self.max_retries}) exceeded, cannot retry")
            return False
        
        self.retry_count += 1
        self.retry_requested = True
        print(f"[{self.__class__.__name__}] Retry requested (attempt {self.retry_count}/{self.max_retries}): {reason}")
        return True
    
    def reset_retry_state(self) -> None:
        """Reset retry state (called after successful retry)"""
        self.retry_requested = False
        print(f"[{self.__class__.__name__}] Retry state reset")
    
    def can_retry(self) -> bool:
        """Check if retry is still allowed"""
        return self.retry_count < self.max_retries
    
    @property
    def is_monitoring(self) -> bool:
        return self._monitoring_active
    
    @property
    def last_evaluation(self) -> Optional[GoalResult]:
        return self._last_evaluation
    
    def detect_elements_on_page(self, page: Page, goal_description: str = "") -> List[Dict[str, Any]]:
        """
        Allow goals to detect elements on the current page for their own evaluation.
        This gives goals the ability to "see" what's on the screen.
        """
        try:
            # Simple element detection for goals - gets basic info about form elements
            js_code = """
            (function() {
                const elements = [];
                
                // Get all form-related elements
                const formElements = document.querySelectorAll('input, select, textarea, button, [role="button"]');
                
                formElements.forEach((element, index) => {
                    const rect = element.getBoundingClientRect();
                    
                    // Skip hidden elements
                    if (rect.width === 0 || rect.height === 0 || 
                        getComputedStyle(element).display === 'none' ||
                        getComputedStyle(element).visibility === 'hidden') {
                        return;
                    }
                    
                    const elementInfo = {
                        index: index,
                        tagName: element.tagName.toLowerCase(),
                        type: element.type || '',
                        id: element.id || '',
                        name: element.name || '',
                        className: element.className || '',
                        value: element.value || '',
                        textContent: element.textContent?.trim() || '',
                        placeholder: element.placeholder || '',
                        required: element.required || false,
                        disabled: element.disabled || false,
                        coordinates: {
                            x: Math.round(rect.left + rect.width / 2),
                            y: Math.round(rect.top + rect.height / 2)
                        },
                        rect: {
                            left: Math.round(rect.left),
                            top: Math.round(rect.top),
                            width: Math.round(rect.width),
                            height: Math.round(rect.height)
                        },
                        ariaLabel: element.getAttribute('aria-label') || '',
                        role: element.getAttribute('role') || '',
                        isVisible: true,
                        isFilled: !!(element.value && element.value.trim()),
                        isSubmitButton: (
                            element.type === 'submit' ||
                            element.textContent?.toLowerCase().includes('submit') ||
                            element.textContent?.toLowerCase().includes('send') ||
                            element.className.toLowerCase().includes('submit')
                        ),
                        isNextButton: (
                            element.textContent?.toLowerCase().includes('next') ||
                            element.textContent?.toLowerCase().includes('continue') ||
                            element.className.toLowerCase().includes('next')
                        )
                    };
                    
                    elements.push(elementInfo);
                });
                
                return elements;
            })();
            """
            
            elements = page.evaluate(js_code)
            return elements or []
            
        except Exception as e:
            print(f"⚠️ Error detecting elements for goal: {e}")
            return []
    
    def get_field_value_at_coordinates(self, page: Page, x: int, y: int) -> str:
        """Get the value of a form field at specific coordinates"""
        try:
            js_code = f"""
            (function() {{
                const element = document.elementFromPoint({x}, {y});
                if (!element) return '';
                
                // For form elements, return the value
                if (element.value !== undefined) {{
                    return element.value;
                }}
                
                // For other elements, return text content
                return element.textContent?.trim() || '';
            }})();
            """
            
            value = page.evaluate(js_code)
            return str(value).strip()
        except Exception as e:
            print(f"⚠️ Error getting field value at ({x}, {y}): {e}")
            return ""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}('{self.description}')"
