"""
Back Goal - Validates backward navigation in browser history.
"""
from __future__ import annotations

from typing import Optional, Dict, Any

from .base import BaseGoal, GoalResult, GoalStatus, EvaluationTiming, GoalContext
from .history_utils import evaluate_history_move_goal


class BackGoal(BaseGoal):
    """
    Goal for navigating back in browser history.

    Success criteria:
    - If expected_url is provided: current URL equals expected_url.
    - Else if steps_back is provided: current URL equals the URL that was N steps before
      at the time of goal creation (we capture expected_url then).
    - Else (default 1 step): current URL equals the previous URL from pre-navigation state.
    """

    # Evaluate after navigation occurs
    EVALUATION_TIMING = EvaluationTiming.AFTER

    def __init__(
        self,
        description: str,
        expected_url: Optional[str] = None,
        steps_back: Optional[int] = None,
        expected_title_substr: Optional[str] = None,
        start_index: Optional[int] = None,
        start_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(description, **kwargs)
        self.expected_url = expected_url
        self.steps_back = steps_back if (steps_back or 0) > 0 else 1
        self.expected_title_substr = (expected_title_substr or "").strip() or None
        # Baseline snapshot at goal creation to reason about steps_back movement
        self.start_index = start_index
        self.start_url = start_url or ""

    def evaluate(self, context: GoalContext) -> GoalResult:
        try:
            return evaluate_history_move_goal(
                direction='back',
                steps=self.steps_back or 1,
                expected_url=self.expected_url,
                expected_title_substr=self.expected_title_substr,
                start_index=self.start_index,
                start_url=self.start_url,
                context=context,
            )
        except Exception as e:
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.0,
                reasoning=f"Error evaluating back goal: {e}",
                evidence={"error": str(e)}
            )

    def get_description(self, context: GoalContext) -> str:
        parts = [
            f"Back navigation goal: {self.description}",
        ]
        if self.expected_url:
            parts.append(f"Expected URL: {self.expected_url}")
        if self.steps_back:
            parts.append(f"Requested steps back: {self.steps_back}")
        if self.expected_title_substr:
            parts.append(f"Expected title contains: '{self.expected_title_substr}'")
        if self.start_index is not None:
            parts.append(f"Start index: {self.start_index}")
        if context and context.url_history:
            parts.append(f"History length: {len(context.url_history)}")
            parts.append(f"Current URL: {context.current_state.url}")
        parts.append("Guidance: Use BACK action. Prefer either steps_back or target_history_index/target_url.")
        return "\n".join(parts)
