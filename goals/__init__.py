"""
Goal Framework - Extensible system for monitoring and evaluating goal completion.

This package provides a comprehensive way to track browser interactions and 
deterministically assess whether user goals have been achieved.
"""

from .base import (
    GoalStatus,
    InteractionType,
    EvaluationTiming,
    BrowserState,
    Interaction,
    GoalContext,
    GoalResult,
    BaseGoal,
    ConditionalGoal,
    Condition,
    ConditionType
)

from .monitor import GoalMonitor
from .element_analyzer import ElementAnalyzer

# Import concrete goal implementations
from .click_goal import ClickGoal
from .navigation_goal import NavigationGoal
from .form_goal import FormFillGoal
from .if_goal import IfGoal
from .press_goal import PressGoal
from .scroll_goal import ScrollGoal

__all__ = [
    # Base types
    "GoalStatus",
    "InteractionType", 
    "EvaluationTiming",
    "BrowserState",
    "Interaction",
    "GoalContext",
    "GoalResult",
    "BaseGoal",
    "ConditionalGoal",
    "Condition",
    "ConditionType",
    
    # Framework components
    "GoalMonitor",
    "ElementAnalyzer",
    
    # Concrete goals
    "ClickGoal",
    "NavigationGoal", 
    "FormFillGoal",
    "IfGoal",
    "PressGoal",
    "ScrollGoal",
]
