"""
IfGoal implementation for conditional goal execution.
"""
from __future__ import annotations

from typing import Optional
from .base import ConditionalGoal, Condition, BaseGoal, GoalResult


class IfGoal(ConditionalGoal):
    """
    A conditional goal that works like an if statement.
    
    If the condition is true, runs the success goal.
    If the condition is false, runs the fail goal.
    
    This is the most basic form of conditional goal execution,
    providing simple branching logic based on condition evaluation.
    
    Example:
        ```python
        # Create a condition
        condition = is_weekday_condition()
        
        # Create sub-goals
        success_goal = SomeGoal("Do weekday action")
        fail_goal = SomeGoal("Do weekend action")
        
        # Create the if goal
        if_goal = IfGoal(condition, success_goal, fail_goal)
        
        # Use like any other goal
        result = if_goal.evaluate(context)
        ```
    """
    
    def __init__(self, condition: Condition, success_goal: BaseGoal, 
                 fail_goal: BaseGoal, description: Optional[str] = None, 
                 max_retries: int = 3, **kwargs):
        """
        Initialize an IfGoal.
        
        Args:
            condition: The condition to evaluate
            success_goal: Goal to run if condition is True
            fail_goal: Goal to run if condition is False
            description: Optional custom description. If None, auto-generates from condition and goals
            max_retries: Maximum number of retries allowed
            **kwargs: Additional arguments passed to BaseGoal
        """
        if description is None:
            description = f"If {condition.description} then {success_goal.description} else {fail_goal.description}"
        
        super().__init__(description, condition, success_goal, fail_goal, max_retries, **kwargs)
    
    def evaluate(self, context) -> GoalResult:
        """
        Evaluate the condition and run the appropriate sub-goal.
        
        This method:
        1. Evaluates the condition using the provided evaluator function
        2. Selects either the success_goal or fail_goal based on the result
        3. Evaluates the selected sub-goal and returns its result
        
        Args:
            context: GoalContext containing browser state and interaction history
            
        Returns:
            GoalResult from the active sub-goal
        """
        return super().evaluate(context)
    
    def get_condition_summary(self) -> str:
        """
        Get a summary of the condition and its current state.
        
        Returns:
            String summary of the condition
        """
        status = "TRUE" if self.last_condition_result else "FALSE" if self.last_condition_result is not None else "UNEVALUATED"
        return f"Condition: {self.condition.description} (Status: {status})"
    
    def get_active_goal_info(self) -> str:
        """
        Get information about the currently active sub-goal.
        
        Returns:
            String description of the active sub-goal
        """
        if self.current_sub_goal:
            return f"Active Goal: {self.current_sub_goal.description}"
        return "No active sub-goal"
