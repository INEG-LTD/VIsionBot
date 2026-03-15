"""
Mission progress policies for the job application app.
"""

from __future__ import annotations

from typing import Any

from agent.mission_progress import (
    ActionResultPayload,
    FinishAttemptPayload,
    FinishDecision,
)

APPLICATION_TERMINAL_EVENT_NAMES = {
    "application_submitted",
    "application_cancelled",
    "application_manual_followup_required",
    "application_failed",
}


class ApplicationMissionProgressPolicy:
    """Progress policy for single job application missions."""

    def on_mission_start(
        self,
        *,
        mission: str,
        starting_url: str,
        base_knowledge: list[str],
    ) -> dict[str, Any]:
        return {
            "terminal_event_seen": False,
            "terminal_event_name": "",
            "auth_interruptions": 0,
            "recoverable_errors": 0,
        }

    def on_action_result(
        self,
        *,
        progress_state: dict[str, Any],
        action_result: ActionResultPayload,
    ) -> tuple[dict[str, Any], list[str]]:
        state = self._coerce_state(progress_state)
        hints: list[str] = []
        accepted_names = set(action_result.accepted_event_names or ())

        terminal_name = next(
            (name for name in APPLICATION_TERMINAL_EVENT_NAMES if name in accepted_names),
            "",
        )
        if terminal_name:
            state["terminal_event_seen"] = True
            state["terminal_event_name"] = terminal_name

        if "application_auth_required" in accepted_names:
            state["auth_interruptions"] = int(state.get("auth_interruptions", 0) or 0) + 1
            hints.append(
                "Wait for the user to complete the required sign-in, verification, or anti-bot step, then re-detect the page before continuing."
            )

        if "application_error" in accepted_names:
            state["recoverable_errors"] = int(state.get("recoverable_errors", 0) or 0) + 1
            hints.append(
                "Recover only if the browser returns to the same application flow. Otherwise end with application_failed or application_manual_followup_required."
            )

        return state, hints

    def get_progress_context(
        self,
        *,
        progress_state: dict[str, Any],
    ) -> str:
        state = self._coerce_state(progress_state)
        return "\n".join(
            [
                f"- terminal_event_seen: {str(bool(state['terminal_event_seen'])).lower()}",
                f"- terminal_event_name: {state['terminal_event_name'] or 'none'}",
                f"- auth_interruptions: {state['auth_interruptions']}",
                f"- recoverable_errors: {state['recoverable_errors']}",
                "- Mission completion is blocked until one terminal application event is accepted.",
            ]
        )

    def on_finish_attempt(
        self,
        *,
        progress_state: dict[str, Any],
        finish_attempt: FinishAttemptPayload,
    ) -> FinishDecision:
        state = self._coerce_state(progress_state)
        accepted_names = set(finish_attempt.accepted_event_names or ())

        if finish_attempt.kind == "terminal_event":
            if finish_attempt.event_name in APPLICATION_TERMINAL_EVENT_NAMES:
                return FinishDecision(allow=True)
            return FinishDecision(
                allow=False,
                reason=f"unknown application terminal event '{finish_attempt.event_name or 'missing'}'",
            )

        if finish_attempt.kind == "done":
            if state["terminal_event_seen"] or accepted_names.intersection(APPLICATION_TERMINAL_EVENT_NAMES):
                return FinishDecision(allow=True)
            return FinishDecision(
                allow=False,
                reason="application mission cannot end before a terminal application event is accepted",
                hint="Continue until the flow reaches submission, cancellation, manual follow-up, or failure.",
            )

        return FinishDecision(allow=True)

    @staticmethod
    def _coerce_state(progress_state: dict[str, Any]) -> dict[str, Any]:
        source = dict(progress_state or {})
        return {
            "terminal_event_seen": bool(source.get("terminal_event_seen", False)),
            "terminal_event_name": str(source.get("terminal_event_name", "") or "").strip(),
            "auth_interruptions": int(source.get("auth_interruptions", 0) or 0),
            "recoverable_errors": int(source.get("recoverable_errors", 0) or 0),
        }


class GoogleJobsMissionProgressPolicy:
    """Progress policy for Google Jobs collection missions."""

    def __init__(self, *, target_jobs: int):
        self.target_jobs = max(1, int(target_jobs or 1))

    def on_mission_start(
        self,
        *,
        mission: str,
        starting_url: str,
        base_knowledge: list[str],
    ) -> dict[str, Any]:
        return {
            "saved_jobs": 0,
            "target_jobs": self.target_jobs,
            "consecutive_ineligible_jobs": 0,
            "results_exhausted": False,
            "last_job_outcome": "",
            "pending_rejection_reason_code": "",
        }

    def on_action_result(
        self,
        *,
        progress_state: dict[str, Any],
        action_result: ActionResultPayload,
    ) -> tuple[dict[str, Any], list[str]]:
        state = self._coerce_state(progress_state)
        hints: list[str] = []
        handled_process_outcome = False

        if action_result.success and action_result.action_name == "process_focused_job":
            data = dict(action_result.result_data or {})
            if data.get("saved") is True:
                state["saved_jobs"] = int(state.get("saved_jobs", 0) or 0) + 1
                state["consecutive_ineligible_jobs"] = 0
                state["results_exhausted"] = False
                state["last_job_outcome"] = "saved"
                state["pending_rejection_reason_code"] = ""
                handled_process_outcome = True
            elif data.get("duplicate") is True:
                state["consecutive_ineligible_jobs"] = int(
                    state.get("consecutive_ineligible_jobs", 0) or 0
                ) + 1
                state["last_job_outcome"] = "duplicate"
                state["pending_rejection_reason_code"] = "duplicate"
                handled_process_outcome = True
                hints.append(
                    "Duplicate or ineligible jobs do not count toward the target. Continue to another unseen job card."
                )
            elif data.get("matches_profile") is False:
                state["consecutive_ineligible_jobs"] = int(
                    state.get("consecutive_ineligible_jobs", 0) or 0
                ) + 1
                state["last_job_outcome"] = "profile_mismatch"
                state["pending_rejection_reason_code"] = "profile_mismatch"
                handled_process_outcome = True
                hints.append(
                    "This job did not match the target profile. Continue to another unseen job card."
                )
            elif data.get("reason") == "search_context_mismatch":
                state["last_job_outcome"] = "search_context_mismatch"
                state["pending_rejection_reason_code"] = ""
                handled_process_outcome = True
                hints.append(
                    "The Google Jobs page drifted away from the original search. Check recent navigation history, use go_back if the previous page is the correct Google Jobs results page, otherwise reopen the original Google Jobs search URL with &udm=8."
                )
            elif data.get("processable") is False:
                state["consecutive_ineligible_jobs"] = int(
                    state.get("consecutive_ineligible_jobs", 0) or 0
                ) + 1
                state["last_job_outcome"] = "unprocessable_missing_required_fields"
                state["pending_rejection_reason_code"] = "unprocessable_missing_required_fields"
                handled_process_outcome = True
                hints.append(
                    "This job could not be saved because required fields were missing. Continue to another unseen job card."
                )

        accepted_names = set(action_result.accepted_event_names or ())
        accepted_events = list(action_result.accepted_events or ())

        if "job_rejected" in accepted_names and not handled_process_outcome:
            reason_code = self._event_reason_code(
                accepted_events,
                event_name="job_rejected",
                fallback="rejected",
            )
            pending_reason_code = str(state.get("pending_rejection_reason_code", "") or "").strip()
            if pending_reason_code and (not reason_code or reason_code == pending_reason_code):
                state["pending_rejection_reason_code"] = ""
            else:
                state["consecutive_ineligible_jobs"] = int(
                    state.get("consecutive_ineligible_jobs", 0) or 0
                ) + 1
                state["last_job_outcome"] = reason_code
                state["pending_rejection_reason_code"] = ""
                hints.append(
                    "Duplicate or ineligible jobs do not count toward the target. Continue to another unseen job card."
                )

        if "job_results_exhausted" in accepted_names:
            state["results_exhausted"] = True
            hints.append(
                "Results are explicitly exhausted. Partial completion is now allowed if the target count is still unmet."
            )

        return state, hints

    def get_progress_context(
        self,
        *,
        progress_state: dict[str, Any],
    ) -> str:
        state = self._coerce_state(progress_state)
        return "\n".join(
            [
                f"- saved_jobs: {state['saved_jobs']}/{state['target_jobs']}",
                f"- consecutive_ineligible_jobs: {state['consecutive_ineligible_jobs']}",
                f"- results_exhausted: {str(bool(state['results_exhausted'])).lower()}",
                f"- last_job_outcome: {state['last_job_outcome'] or 'none'}",
                "- Duplicate or ineligible jobs do not count toward the target.",
                "- Partial completion is blocked until results are explicitly exhausted.",
            ]
        )

    def on_finish_attempt(
        self,
        *,
        progress_state: dict[str, Any],
        finish_attempt: FinishAttemptPayload,
    ) -> FinishDecision:
        state = self._coerce_state(progress_state)
        saved_jobs = int(state.get("saved_jobs", 0) or 0)
        target_jobs = int(state.get("target_jobs", self.target_jobs) or self.target_jobs)
        results_exhausted = bool(state.get("results_exhausted", False))
        accepted_events = set(finish_attempt.accepted_event_names or ())

        if finish_attempt.kind == "terminal_event" and finish_attempt.event_name == "job_collection_done":
            outcome = str((finish_attempt.event_data or {}).get("outcome", "") or "").strip().lower()
            if outcome == "complete":
                if saved_jobs >= target_jobs:
                    return FinishDecision(allow=True)
                return FinishDecision(
                    allow=False,
                    reason="complete collection is blocked until the saved job target is reached",
                    hint="Continue to another unseen job card.",
                )
            if outcome == "partial":
                if saved_jobs >= target_jobs:
                    return FinishDecision(
                        allow=False,
                        reason="partial collection is incorrect because the saved job target is already met",
                    )
                if results_exhausted:
                    return FinishDecision(allow=True)
                return FinishDecision(
                    allow=False,
                    reason="partial collection is blocked until results are explicitly exhausted",
                    hint="Scroll the left jobs panel and confirm results are exhausted before ending partial.",
                )
            if outcome == "empty":
                if saved_jobs > 0:
                    return FinishDecision(
                        allow=False,
                        reason="empty collection is incorrect because jobs have already been saved",
                    )
                if results_exhausted:
                    return FinishDecision(allow=True)
                return FinishDecision(
                    allow=False,
                    reason="empty collection is blocked until results are explicitly exhausted",
                    hint="Confirm there are no viable Google Jobs results left before ending empty.",
                )
            return FinishDecision(
                allow=False,
                reason=f"unknown job_collection_done outcome '{outcome or 'missing'}'",
            )

        if finish_attempt.kind == "done":
            if saved_jobs >= target_jobs:
                return FinishDecision(allow=True)
            if accepted_events.intersection({"job_collection_done", "job_collection_failed"}):
                return FinishDecision(allow=True)
            return FinishDecision(
                allow=False,
                reason="mission cannot end before the target is reached or a terminal collection event is accepted",
                hint="Continue to another unseen job card, or explicitly confirm results are exhausted first.",
            )

        return FinishDecision(allow=True)

    def _coerce_state(self, progress_state: dict[str, Any]) -> dict[str, Any]:
        source = dict(progress_state or {})
        return {
            "saved_jobs": int(source.get("saved_jobs", 0) or 0),
            "target_jobs": max(1, int(source.get("target_jobs", self.target_jobs) or self.target_jobs)),
            "consecutive_ineligible_jobs": int(source.get("consecutive_ineligible_jobs", 0) or 0),
            "results_exhausted": bool(source.get("results_exhausted", False)),
            "last_job_outcome": str(source.get("last_job_outcome", "") or "").strip(),
            "pending_rejection_reason_code": str(
                source.get("pending_rejection_reason_code", "") or ""
            ).strip(),
        }

    @staticmethod
    def _event_reason_code(
        accepted_events: list[dict[str, Any]],
        *,
        event_name: str,
        fallback: str,
    ) -> str:
        for event in accepted_events:
            if str(event.get("name", "") or "").strip() != event_name:
                continue
            data = event.get("data")
            if isinstance(data, dict):
                reason_code = str(data.get("reason_code", "") or "").strip()
                if reason_code:
                    return reason_code
        return fallback
