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


class WhileGoal(BaseGoal):
    """
    Goal that keeps requesting retries (i.e., new plans) until the condition
    evaluates to True. The provided `loop_goal` describes the body (what to do
    on each iteration), and its description is surfaced to planning prompts.
    """

    # Evaluate after interactions to decide whether to continue
    EVALUATION_TIMING: EvaluationTiming = EvaluationTiming.AFTER

    def __init__(
        self,
        condition: Condition,
        loop_goal: BaseGoal,
        description: Optional[str] = None,
        max_iterations: int = 30,
        max_duration_s: Optional[float] = 180.0,
        **kwargs,
    ) -> None:
        desc = description or f"While ({condition.description}) do: {loop_goal.description}"
        # Align detection need with the body by default
        needs_detection = kwargs.pop("needs_detection", getattr(loop_goal, "needs_detection", True))
        super().__init__(desc, needs_detection=needs_detection, **kwargs)

        self.condition = condition
        self.loop_goal = loop_goal
        self.progress = LoopProgress(iterations=0, last_condition_result=None, started_at=time.time())
        self.max_iterations = max(1, int(max_iterations))
        self.max_duration_s = max_duration_s if (max_duration_s is None or max_duration_s > 0) else None

    def evaluate(self, context: GoalContext) -> GoalResult:
        """Evaluate loop condition; request retries while condition is true."""
        try:
            cond = bool(self.condition.evaluator(context))
        except Exception:
            cond = False
        self.progress.last_condition_result = cond

        # If condition is false -> loop terminates successfully
        if not cond:
            return GoalResult(
                status=GoalStatus.ACHIEVED,
                confidence=1.0,
                reasoning=f"Loop complete. Condition no longer satisfied: {self.condition.description}",
                evidence={
                    "iterations": self.progress.iterations,
                    "duration_s": round(time.time() - self.progress.started_at, 2),
                    "condition": self.condition.description,
                },
                next_actions=[],
            )

        # Check iteration/time safety limits
        self.progress.iterations += 1
        duration = time.time() - self.progress.started_at
        if self.progress.iterations >= self.max_iterations:
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.3,
                reasoning=(
                    f"Loop aborted after {self.progress.iterations} iterations: "
                    f"condition still true: {self.condition.description}"
                ),
                evidence={"iterations": self.progress.iterations, "duration_s": round(duration, 2)},
                next_actions=["Consider refining the loop action or condition"],
            )

        if self.max_duration_s is not None and duration > self.max_duration_s:
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.3,
                reasoning=(
                    f"Loop timed out after {round(duration, 1)}s: "
                    f"condition still true: {self.condition.description}"
                ),
                evidence={"iterations": self.progress.iterations, "duration_s": round(duration, 2)},
                next_actions=["Narrow the condition or adjust max duration"],
            )

        # Condition is still true: request another planning/execution pass
        self.request_retry(
            reason=f"WhileGoal continuing: condition still true → {self.condition.description}"
        )
        return GoalResult(
            status=GoalStatus.PENDING,
            confidence=0.6,
            reasoning=(
                f"Condition still true; continue loop body: {self.loop_goal.description}"
            ),
            evidence={
                "iterations": self.progress.iterations,
                "condition": self.condition.description,
                "condition_result": True,
            },
            next_actions=[f"Repeat: {self.loop_goal.description}"],
        )

    def get_description(self, context: GoalContext) -> str:
        """Structured description used to guide planning."""
        status = (
            "TRUE" if self.progress.last_condition_result
            else "FALSE" if self.progress.last_condition_result is not None
            else "UNEVALUATED"
        )
        return (
            f"Goal Type: While Loop\n"
            f"Continue Condition: {self.condition.description}\n"
            f"Condition Status: {status}\n"
            f"Iterations: {self.progress.iterations} / {self.max_iterations}\n"
            f"Max Duration (s): {self.max_duration_s if self.max_duration_s is not None else 'unlimited'}\n"
            f"Loop Body: {self.loop_goal.description}\n"
            f"Planning Guidance: Focus on executing the loop body once per pass while the condition remains true."
        )

    # Propagate lifecycle events to the loop body for consistency
    def start_monitoring(self) -> None:
        super().start_monitoring()
        try:
            self.loop_goal.start_monitoring()
        except Exception:
            pass

    def stop_monitoring(self) -> None:
        super().stop_monitoring()
        try:
            self.loop_goal.stop_monitoring()
        except Exception:
            pass

    def on_interaction(self, interaction) -> None:
        try:
            self.loop_goal.on_interaction(interaction)
        except Exception:
            pass

    def on_state_change(self, old_state, new_state) -> None:
        try:
            self.loop_goal.on_state_change(old_state, new_state)
        except Exception:
            pass

