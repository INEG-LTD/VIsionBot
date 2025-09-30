"""
WhileGoal implementation for loop-like goal execution.

Runs a sub-goal repeatedly until a condition is achieved, similar to:
  - while not CONDITION: do ACTION
  - ACTION until CONDITION

This composes the existing Conditional/Condition infra and integrates with
the retry mechanism so the planner keeps iterating the body until done.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .base import BaseGoal, GoalResult, GoalStatus, EvaluationTiming, Condition, GoalContext


@dataclass
class LoopProgress:
    iterations: int = 0
    last_condition_result: Optional[bool] = None
    started_at: float = 0.0
    loop_retries: int = 0  # Track custom loop retries separately from goal retries


class WhileGoal(BaseGoal):
    """
    Goal that keeps requesting retries (i.e., new plans) until the condition
    evaluates to True. The provided `loop_goal` describes the body (what to do
    on each iteration), and its description is surfaced to planning prompts.
    
    Enhanced to support multi-command loop bodies through simple parsing.
    """

    # Evaluate after interactions to decide whether to continue
    EVALUATION_TIMING: EvaluationTiming = EvaluationTiming.AFTER

    def __init__(
        self,
        condition: Condition,
        loop_prompt: str,
        else_prompt: Optional[str] = None,
        description: Optional[str] = None,
        max_iterations: int = 30,
        **kwargs,
    ) -> None:
        # Update description to include else goal if provided
        # if else_prompt:
        #     desc = description or f"While ({condition.description}) do: {loop_prompt} else: {else_prompt}"
        # else:
        #     desc = description or f"While ({condition.description}) do: {loop_prompt}"
        
        # Align detection need with the body by default
        super().__init__(description, needs_detection=False, needs_plan=False, **kwargs)

        self.condition = condition
        self.loop_prompt = loop_prompt
        self.else_prompt = else_prompt
        self.progress = LoopProgress(iterations=0, last_condition_result=None, started_at=time.time(), loop_retries=0)
        self.max_iterations = max(1, int(max_iterations))
        

    def set_goal_monitor(self, goal_monitor) -> None:
        """Set the goal monitor."""
        self.goal_monitor = goal_monitor

    def get_description(self, context: GoalContext) -> str:
        """Generate a detailed description of what this goal is looking for."""
        if self.else_prompt:
            return f"While ({self.condition.description}) do: {self.loop_prompt} else: {self.else_prompt}"
        else:
            return f"While ({self.condition.description}) do: {self.loop_prompt}"

    def evaluate(self, context: GoalContext) -> GoalResult:
        """Evaluate loop condition; request retries while condition is true."""
        try:
            cond = bool(self.condition.evaluator(context))
        except Exception:
            cond = False
        self.progress.last_condition_result = cond

        # If condition is false -> loop terminates, execute else goal if provided
        if not cond:
            # Execute else goal if provided
            if self.else_prompt:
                print(f"🔄 Executing else goal: {self.else_prompt}")
                # The else goal will be executed by the main bot loop
                return GoalResult(
                    status=GoalStatus.ACHIEVED,
                    confidence=1.0,
                    reasoning=f"Loop condition false, executing else goal: {self.else_prompt}",
                    evidence={"iterations": self.progress.iterations},
                    next_actions=[f"Execute else: {self.else_prompt}"]
                )
            else:
                # No else goal, loop is complete
                return GoalResult(
                    status=GoalStatus.ACHIEVED,
                    confidence=1.0,
                    reasoning=f"Loop condition false after {self.progress.iterations} iterations",
                    evidence={"iterations": self.progress.iterations},
                    next_actions=[]
                )

        # Condition is true -> continue loop
        self.progress.iterations += 1
        
        # Check iteration limits
        if self.progress.iterations > self.max_iterations:
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.8,
                reasoning=f"Loop exceeded max iterations ({self.max_iterations})",
                evidence={"iterations": self.progress.iterations},
                next_actions=[]
            )

        # Execute the loop body via act() - this will be handled by the main bot loop
        return GoalResult(
            status=GoalStatus.PENDING,
            confidence=0.9,
            reasoning=f"Loop condition true (iteration {self.progress.iterations}), executing loop body",
            evidence={"iterations": self.progress.iterations, "condition_result": cond},
            next_actions=[f"Execute: {self.loop_prompt}"],
        )
