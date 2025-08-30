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
    BaseGoal
)

from .monitor import GoalMonitor
from .element_analyzer import ElementAnalyzer

# Import concrete goal implementations
from .click_goal import ClickGoal
from .navigation_goal import NavigationGoal
from .form_goal import FormFillGoal

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
    
    # Framework components
    "GoalMonitor",
    "ElementAnalyzer",
    
    # Concrete goals
    "ClickGoal",
    "NavigationGoal", 
    "FormFillGoal",
]
