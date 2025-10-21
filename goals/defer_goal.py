"""Defer goal pauses automation until the user resumes control."""
from __future__ import annotations

import time
from typing import Optional

from utils.bot_logger import get_logger, LogLevel, LogCategory

from .base import BaseGoal, GoalResult, GoalStatus, EvaluationTiming, GoalContext


class DeferGoal(BaseGoal):
    """
    Goal that hands control back to the user until they press Enter.
    
    Uses CONTINUOUS evaluation timing because:
    - It needs to be evaluated multiple times during the goal monitoring cycle
    - The _completed flag prevents re-execution of the input() operation
    - It's a blocking operation that waits for user input
    """

    EVALUATION_TIMING = EvaluationTiming.CONTINUOUS

    def __init__(self, description: str, prompt: Optional[str] = None, max_retries: int = 0) -> None:
        super().__init__(
            description=description,
            max_retries=max_retries,
            needs_detection=False,
            needs_plan=False,
        )
        self.prompt_message = (prompt or "").strip() or "Manual control active. Press Enter when you're ready to resume."
        self._started_at: Optional[float] = None
        self._completed: bool = False
        self._displayed_prompt: bool = False
        self._logged_start: bool = False
        self._logged_end: bool = False
        self._logger = get_logger()

    def on_monitoring_start(self) -> None:
        self._started_at = time.time()

    def evaluate(self, context: GoalContext) -> GoalResult:
        if not self._completed:
            summary_details = self._build_context_summary(context)
            summary_line = self._format_summary_line(summary_details)
            if not self._displayed_prompt:
                print("[Defer] Manual pause engaged.")
                if self.prompt_message:
                    print(f"[Defer] {self.prompt_message}")
                if summary_line:
                    print(f"[Defer] {summary_line}")
                self._displayed_prompt = True

            if not self._logged_start:
                log_details = {"prompt": self.prompt_message or "", **summary_details}
                self._logger.log(
                    LogLevel.INFO,
                    LogCategory.SYSTEM,
                    "Defer pause started",
                    details=log_details,
                )
                self._logged_start = True
            try:
                input("[Defer] Press Enter to continue automation...")
            except EOFError:
                print("[Defer] Warning: input stream unavailable; resuming automatically.")
            self._completed = True

        evidence = {"defer_prompt": self.prompt_message}
        if self._started_at is not None:
            evidence["defer_duration_seconds"] = round(time.time() - self._started_at, 3)
        if not self._logged_end:
            duration_ms = None
            if self._started_at is not None:
                duration_ms = (time.time() - self._started_at) * 1000
            self._logger.log(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                "Defer pause ended",
                details=evidence,
                duration_ms=duration_ms,
                success=True,
            )
            self._logged_end = True

        return GoalResult(
            status=GoalStatus.ACHIEVED,
            confidence=1.0,
            reasoning="User signaled readiness to resume after defer pause.",
            evidence=evidence,
        )

    def get_description(self, context: GoalContext) -> str:
        return self.prompt_message

    def _build_context_summary(self, context: GoalContext) -> dict:
        """Create a small context summary for display/logging."""
        summary: dict = {}
        if not context:
            return summary
        state = getattr(context, "current_state", None)
        if state:
            if getattr(state, "title", None):
                summary["page_title"] = state.title
            if getattr(state, "url", None):
                summary["page_url"] = state.url
        if getattr(context, "planned_interaction", None):
            summary["pending_interaction"] = str(context.planned_interaction)
        return summary

    def _format_summary_line(self, summary: dict) -> str:
        if not summary:
            return ""
        parts = []
        for key, value in summary.items():
            if not value:
                continue
            label = key.replace("_", " ").title()
            parts.append(f"{label}: {value}")
        return " | ".join(parts)


class TimedSleepGoal(BaseGoal):
    """
    Goal that pauses automation for a specific number of seconds using time.sleep().
    
    Uses CONTINUOUS evaluation timing because:
    - It needs to be evaluated multiple times during the goal monitoring cycle
    - The _completed flag prevents re-execution of the sleep operation
    - Follows the same pattern as DeferGoal for consistency
    """

    EVALUATION_TIMING = EvaluationTiming.CONTINUOUS  # Evaluate continuously like DeferGoal

    def __init__(self, description: str, delay_seconds: int, prompt: Optional[str] = None, max_retries: int = 0) -> None:
        super().__init__(
            description=description,
            max_retries=max_retries,
            needs_detection=False,
            needs_plan=False,
        )
        self.delay_seconds = delay_seconds
        self.prompt_message = (prompt or "").strip() or f"Pausing for {delay_seconds} seconds..."
        self._logger = get_logger()
        self._completed = False

    def evaluate(self, context: GoalContext) -> GoalResult:
        """Execute the sleep and return immediately."""
        
        if not self._completed:
            print(f"[TimedSleep] Pausing for {self.delay_seconds} seconds...")
            
            # Log the start
            self._logger.log(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                "Timed sleep started",
                details={"delay_seconds": self.delay_seconds, "prompt": self.prompt_message or ""},
            )
            
            # Show countdown while sleeping
            for remaining in range(self.delay_seconds, 0, -1):
                print(f"[TimedSleep] {remaining} seconds remaining...", end='\r')
                time.sleep(1)
            
            print("\n[TimedSleep] Sleep completed!")
            
            # Mark as completed
            self._completed = True
            
            # Log the end
            self._logger.log(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                "Timed sleep ended",
                details={"delay_seconds": self.delay_seconds},
                duration_ms=self.delay_seconds * 1000,
                success=True,
            )

        # Always return ACHIEVED after completion (like DeferGoal)
        evidence = {"delay_seconds": self.delay_seconds}
        return GoalResult(
            status=GoalStatus.ACHIEVED,
            confidence=1.0,
            reasoning=f"Timed sleep completed after {self.delay_seconds} seconds.",
            evidence=evidence,
        )

    def get_description(self, context: GoalContext) -> str:
        return f"Sleep for {self.delay_seconds} seconds"
