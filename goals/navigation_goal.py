"""Navigation goal ensuring a direct URL is provided for open actions."""

from __future__ import annotations

from typing import Optional

from .base import (
    BaseGoal,
    GoalContext,
    GoalResult,
    GoalStatus,
    EvaluationTiming,
    InteractionType,
)


class NavigationGoal(BaseGoal):
    """Validate that navigation commands include a URL and record it."""

    EVALUATION_TIMING = EvaluationTiming.BOTH

    def __init__(
        self,
        description: str,
        navigation_intent: str,
        max_retries: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(
            description,
            max_retries=max_retries,
            needs_detection=False,
            **kwargs,
        )
        self.navigation_intent = (navigation_intent or "").strip()

    def evaluate(self, context: GoalContext) -> GoalResult:
        planned = getattr(context, "planned_interaction", None) or {}
        if planned.get("interaction_type") == InteractionType.NAVIGATION:
            url = (planned.get("url") or "").strip()
            return self._result_for_url(url, timing="pre_interaction")

        url = self._latest_navigation_url(context)
        if url:
            return GoalResult(
                status=GoalStatus.ACHIEVED,
                confidence=1.0,
                reasoning=f"Navigation opened URL '{url}'.",
                evidence={
                    "url": url,
                    "timing": "post_interaction",
                },
            )

        return GoalResult(
            status=GoalStatus.FAILED,
            confidence=1.0 if self.navigation_intent else 0.2,
            reasoning="Navigation action did not include a URL.",
            evidence={"timing": "post_interaction"},
        )

    def get_description(self, context: GoalContext) -> str:
        lines = [
            f"Navigation goal: {self.description}",
            f"Required URL: {self.navigation_intent or '(none provided)'}",
        ]
        latest_url = self._latest_navigation_url(context)
        if latest_url:
            lines.append(f"Last navigation URL: {latest_url}")
        else:
            lines.append("Last navigation URL: (none)")
        return "\n".join(lines)

    def _result_for_url(self, url: str, *, timing: str) -> GoalResult:
        if url:
            return GoalResult(
                status=GoalStatus.ACHIEVED,
                confidence=1.0,
                reasoning=f"Navigation will open URL '{url}'.",
                evidence={
                    "url": url,
                    "timing": timing,
                },
            )
        return GoalResult(
            status=GoalStatus.FAILED,
            confidence=1.0,
            reasoning="Navigation commands must include a URL.",
            evidence={"timing": timing},
        )

    def _latest_navigation_url(self, context: GoalContext) -> Optional[str]:
        for interaction in reversed(context.all_interactions or []):
            if interaction.interaction_type == InteractionType.NAVIGATION:
                if interaction.navigation_url:
                    return interaction.navigation_url.strip()
                info = interaction.target_element_info or {}
                url = info.get("url")
                if url:
                    return str(url).strip()
        return None