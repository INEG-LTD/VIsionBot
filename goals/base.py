"""
Base classes and types for the goal framework.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

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
    PRESS = "press"
    SELECT = "select"
    UPLOAD = "upload"
    DATETIME = "datetime"
    NAVIGATION = "navigation"
    PAGE_LOAD = "page_load"
    ELEMENT_APPEAR = "element_appear"
    ELEMENT_DISAPPEAR = "element_disappear"
    CONTEXT_GUARD = "context_guard"


class EvaluationTiming(str, Enum):
    """
    When a goal should be evaluated relative to interactions.
    
    This enum defines the different timing strategies for goal evaluation,
    each suited for different types of goals and use cases.
    """
    BEFORE = "before"
    """
    Evaluate before interaction occurs.
    
    Use for:
    - Goals that need to check conditions before taking action
    - Validation goals that prevent actions if conditions aren't met
    - Pre-flight checks that determine if an action should proceed
    
    Example: "Check if user is logged in before clicking the profile button"
    """
    
    AFTER = "after"
    """
    Evaluate after interaction completes.
    
    Use for:
    - Goals that verify the result of an action
    - One-time actions that complete immediately
    - Goals that check if an action was successful
    
    Example: "Verify the page loaded correctly after navigation"
    """
    
    BOTH = "both"
    """
    Evaluate both before and after interactions.
    
    Use for:
    - Goals that need to track state changes
    - Goals that monitor the full lifecycle of an action
    - Complex goals that need pre and post validation
    
    Example: "Monitor form submission - check form is valid before submit, verify success after"
    """
    
    CONTINUOUS = "continuous"
    """
    Evaluate continuously (polling-based).
    
    Use for:
    - Goals that monitor ongoing conditions
    - Goals that wait for specific states to change
    - Goals that need to respond to dynamic changes
    - Goals that may take time to complete
    
    Important: Must use _completed flag to prevent re-execution of blocking operations.
    
    Example: "Wait until element appears", "Monitor for error messages", "Defer until user input"
    """


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
    keys_pressed: Optional[str] = None
    scroll_direction: Optional[str] = None
    scroll_axis: Optional[str] = None
    target_x: Optional[int] = None
    target_y: Optional[int] = None
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
    
    Evaluation Timing:
        Subclasses should override EVALUATION_TIMING to specify when they want
        to be evaluated. See EvaluationTiming enum for detailed documentation
        on when to use each timing type (BEFORE, AFTER, BOTH, CONTINUOUS).
    """
    
    # Subclasses should override this to specify when they want to be evaluated
    # See EvaluationTiming enum for detailed documentation on timing options
    EVALUATION_TIMING: EvaluationTiming = EvaluationTiming.AFTER
    
    def __init__(self, description: str, max_retries: int = 3, needs_detection: bool = True, needs_plan: bool = True, **kwargs):
        self.description = description
        self.created_at = time.time()
        self.kwargs = kwargs
        self._monitoring_active = False
        self._last_evaluation: Optional[GoalResult] = None
        self.max_retries = max_retries
        self.retry_count = 0
        self.retry_requested = False
        self.retry_reason = ""
        self.needs_detection = needs_detection
        self.needs_plan = needs_plan
        
    @classmethod
    def make_noop_goal(cls, description: str) -> BaseGoal:
        return _NoOpGoal(description)
      
    @classmethod
    def make_ref_goal(cls, description: str, ref_id: str, run_result: bool) -> BaseGoal:
        return RefGoal(description, ref_id, run_result)
        
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
        self.retry_reason = reason
        print(f"[{self.__class__.__name__}] Retry requested (attempt {self.retry_count}/{self.max_retries}): {reason}")
        return True
    
    def reset_retry_state(self) -> None:
        """Reset retry state (called after successful retry)"""
        self.retry_requested = False
        self.retry_reason = ""
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


# =============================================================================
# Conditional Goal System
# =============================================================================

class ConditionType(str, Enum):
    """Types of conditions that can be evaluated"""
    ENVIRONMENT_STATE = "environment_state"    # Test page state, elements, etc.
    COMPUTATIONAL = "computational"            # Mathematical, date/time, logic
    USER_DEFINED = "user_defined"             # Custom conditions defined by user


@dataclass
class Condition:
    """Represents a condition to be evaluated"""
    condition_type: ConditionType
    description: str
    evaluator: Callable[[GoalContext], bool]
    confidence_threshold: float = 0.8

class _NoOpGoal(BaseGoal):
    def __init__(self, description: str):
        super().__init__(description, needs_detection=False)
    def evaluate(self, context):
        return GoalResult(status=GoalStatus.ACHIEVED, confidence=1.0, reasoning="No-op goal")
    def get_description(self, context):
        return self.description
    
class RefGoal(BaseGoal):
    def __init__(self, description: str, ref_id: str, run_result: bool):
        super().__init__(description)
        self.ref_id = ref_id
        self.run_result = run_result
        self.needs_plan = False
        self.needs_detection = False
        
    def evaluate(self, context):
        return GoalResult(status=GoalStatus.ACHIEVED if self.run_result else GoalStatus.FAILED, confidence=1.0, reasoning="Ref goal")
    
    def get_description(self, context):
        return self.description

class ConditionalGoal(BaseGoal):
    """
    Abstract base class for goals that evaluate conditions and run different sub-goals.
    
    ConditionalGoals can evaluate various types of conditions and execute different
    sub-goals based on the result. This allows for complex branching logic in goal execution.
    """
    
    def __init__(self, description: str, condition: Condition,
                 success_goal: Optional[BaseGoal], fail_goal: Optional[BaseGoal],
                 max_retries: int = 3, **kwargs):
        super().__init__(description, max_retries, **kwargs)
        self.condition = condition
        self.success_goal = success_goal
        self.fail_goal = fail_goal
        self._last_condition_result: Optional[bool] = None
        self._current_sub_goal: Optional[BaseGoal] = None
    
    def evaluate(self, context: GoalContext) -> GoalResult:
        """
        Evaluate the condition once and then delegate to the appropriate sub-goal.
        """
        try:
            # Evaluate the condition once
            condition_result = self.condition.evaluator(context)
            self._last_condition_result = condition_result
            
            print("🔍 DEBUG: ConditionalGoal evaluation:")
            print(f"   📋 Condition: {self.condition.description}")
            print(f"   🎯 Condition result: {condition_result}")
            
            # Determine which sub-goal to use
            active_goal = self.success_goal if condition_result else self.fail_goal
            self._current_sub_goal = active_goal

            if active_goal:
                print(f"   🎯 Selected sub-goal: {active_goal.__class__.__name__} - '{active_goal.description}'")
            else:
                print("   🎯 No sub-goal to execute for this branch")

            if active_goal is None:
                status = GoalStatus.ACHIEVED
                reasoning = (
                    "Condition TRUE but no success goal provided"
                    if condition_result
                    else "Condition FALSE and no fail goal provided"
                )
            else:
                status = GoalStatus.ACHIEVED if condition_result else GoalStatus.FAILED
                reasoning = (
                    f"Condition TRUE: executing success goal '{active_goal.description}'"
                    if condition_result
                    else f"Condition FALSE: executing fail goal '{active_goal.description}'"
                )

            return GoalResult(
                status=status,
                confidence=1,
                reasoning=reasoning,
                evidence={
                    "condition_type": self.condition.condition_type.value,
                    "condition_result": condition_result,
                    "active_goal": getattr(active_goal, 'description', None),
                }
            )
            
        except Exception as e:
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.0,
                reasoning=f"Error evaluating conditional goal: {str(e)}",
                evidence={"error": str(e), "condition_type": self.condition.condition_type.value}
            )
    
    def get_description(self, context: GoalContext) -> str:
        """Generate description including condition and sub-goal status"""
        condition_status = "TRUE" if self._last_condition_result else "FALSE" if self._last_condition_result is not None else "UNEVALUATED"
        
        base_desc = f"""
Goal Type: Conditional Goal
Condition: {self.condition.description}
Condition Type: {self.condition.condition_type.value}
Condition Status: {condition_status}
"""
        
        if self._current_sub_goal:
            sub_desc = self._current_sub_goal.get_description(context)
            base_desc += f"\nActive Sub-Goal:\n{sub_desc}"
        else:
            base_desc += "\nSub-Goals:\n"
            base_desc += f"Success Goal: {self.success_goal.description}\n"
            base_desc += f"Fail Goal: {self.fail_goal.description}"
        
        return base_desc.strip()
    
    def start_monitoring(self) -> None:
        """Start monitoring both sub-goals"""
        super().start_monitoring()
        self.success_goal.start_monitoring()
        self.fail_goal.start_monitoring()
    
    def stop_monitoring(self) -> None:
        """Stop monitoring both sub-goals"""
        super().stop_monitoring()
        self.success_goal.stop_monitoring()
        self.fail_goal.stop_monitoring()
    
    def on_interaction(self, interaction: Interaction) -> None:
        """Forward interactions to both sub-goals"""
        super().on_interaction(interaction)
        self.success_goal.on_interaction(interaction)
        self.fail_goal.on_interaction(interaction)
    
    def on_state_change(self, old_state: BrowserState, new_state: BrowserState) -> None:
        """Forward state changes to both sub-goals"""
        super().on_state_change(old_state, new_state)
        self.success_goal.on_state_change(old_state, new_state)
        self.fail_goal.on_state_change(old_state, new_state)
    
    @property
    def current_sub_goal(self) -> Optional[BaseGoal]:
        """Get the currently active sub-goal"""
        return self._current_sub_goal
    
    @property
    def last_condition_result(self) -> Optional[bool]:
        """Get the result of the last condition evaluation"""
        return self._last_condition_result





# =============================================================================
# Condition Factory Functions
# =============================================================================

def create_environment_condition(description: str, evaluator: Callable[[GoalContext], bool], 
                                confidence_threshold: float = 0.8) -> Condition:
    """Create an environment state condition"""
    return Condition(
        condition_type=ConditionType.ENVIRONMENT_STATE,
        description=description,
        evaluator=evaluator,
        confidence_threshold=confidence_threshold
    )


def create_computational_condition(description: str, evaluator: Callable[[GoalContext], bool], 
                                  confidence_threshold: float = 0.8) -> Condition:
    """Create a computational condition"""
    return Condition(
        condition_type=ConditionType.COMPUTATIONAL,
        description=description,
        evaluator=evaluator,
        confidence_threshold=confidence_threshold
    )


def create_user_defined_condition(description: str, evaluator: Callable[[GoalContext], bool], 
                                 confidence_threshold: float = 0.8) -> Condition:
    """Create a user-defined condition"""
    return Condition(
        condition_type=ConditionType.USER_DEFINED,
        description=description,
        evaluator=evaluator,
        confidence_threshold=confidence_threshold
    )
