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
from .type_goal import TypeGoal
from .date_goal import DateGoal
from .select_goal import SelectGoal
from .if_goal import IfGoal
from .press_goal import PressGoal
from .scroll_goal import ScrollGoal
from .while_goal import WhileGoal
from .for_goal import ForGoal
from .for_models import IterationTargetsResponse, ElementContextResponse, TargetValidationResponse
from .back_goal import BackGoal
from .forward_goal import ForwardGoal
from .defer_goal import DeferGoal

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
    "TypeGoal",
    "DateGoal",
    "SelectGoal",
    "IfGoal",
    "PressGoal",
    "ScrollGoal",
    "WhileGoal",
    "ForGoal",
    "IterationTargetsResponse",
    "ElementContextResponse", 
    "TargetValidationResponse",
    "BackGoal",
    "ForwardGoal",
    "DeferGoal",
]
