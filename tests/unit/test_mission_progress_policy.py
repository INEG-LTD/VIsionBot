from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_progress_modules():
    agent_package = types.ModuleType("agent")
    agent_package.__path__ = [str(PROJECT_ROOT / "agent")]
    sys.modules["agent"] = agent_package

    mission_progress = _load_module("agent.mission_progress", "agent/mission_progress.py")
    setattr(agent_package, "mission_progress", mission_progress)

    progress_policies = _load_module(
        "job_application_app.progress_policies_under_test",
        "job_application_app/progress_policies.py",
    )
    return mission_progress, progress_policies


def test_google_jobs_policy_duplicate_blocks_partial_until_exhausted() -> None:
    mission_progress, progress_policies = _load_progress_modules()
    policy = progress_policies.GoogleJobsMissionProgressPolicy(target_jobs=1)
    progress_state = policy.on_mission_start(
        mission="Collect one iOS job",
        starting_url="https://google.com",
        base_knowledge=[],
    )

    progress_state, hints = policy.on_action_result(
        progress_state=progress_state,
        action_result=mission_progress.ActionResultPayload(
            action_name="process_focused_job",
            success=True,
            result_data={"saved": False, "duplicate": True},
        ),
    )

    assert progress_state["saved_jobs"] == 0
    assert progress_state["consecutive_ineligible_jobs"] == 1
    assert progress_state["last_job_outcome"] == "duplicate"
    assert any("do not count toward the target" in hint.lower() for hint in hints)

    progress_state, _ = policy.on_action_result(
        progress_state=progress_state,
        action_result=mission_progress.ActionResultPayload(
            action_name="think",
            success=True,
            accepted_event_names=("job_rejected",),
            accepted_events=(
                {
                    "name": "job_rejected",
                    "data": {"reason_code": "duplicate", "reason": "already saved"},
                },
            ),
        ),
    )

    assert progress_state["consecutive_ineligible_jobs"] == 1

    decision = policy.on_finish_attempt(
        progress_state=progress_state,
        finish_attempt=mission_progress.FinishAttemptPayload(
            kind="terminal_event",
            event_name="job_collection_done",
            event_data={"outcome": "partial"},
        ),
    )

    assert decision.allow is False
    assert "explicitly exhausted" in decision.reason


def test_google_jobs_policy_save_allows_complete() -> None:
    mission_progress, progress_policies = _load_progress_modules()
    policy = progress_policies.GoogleJobsMissionProgressPolicy(target_jobs=1)
    progress_state = policy.on_mission_start(
        mission="Collect one iOS job",
        starting_url="https://google.com",
        base_knowledge=[],
    )

    progress_state, _ = policy.on_action_result(
        progress_state=progress_state,
        action_result=mission_progress.ActionResultPayload(
            action_name="process_focused_job",
            success=True,
            result_data={"saved": True},
        ),
    )

    decision = policy.on_finish_attempt(
        progress_state=progress_state,
        finish_attempt=mission_progress.FinishAttemptPayload(
            kind="terminal_event",
            event_name="job_collection_done",
            event_data={"outcome": "complete"},
        ),
    )

    assert progress_state["saved_jobs"] == 1
    assert decision.allow is True


def test_google_jobs_policy_results_exhausted_allows_partial_and_blocks_done_without_terminal_event() -> None:
    mission_progress, progress_policies = _load_progress_modules()
    policy = progress_policies.GoogleJobsMissionProgressPolicy(target_jobs=2)
    progress_state = policy.on_mission_start(
        mission="Collect two iOS jobs",
        starting_url="https://google.com",
        base_knowledge=[],
    )

    progress_state, hints = policy.on_action_result(
        progress_state=progress_state,
        action_result=mission_progress.ActionResultPayload(
            action_name="think",
            success=True,
            accepted_event_names=("job_results_exhausted",),
            accepted_events=({"name": "job_results_exhausted", "data": {"confirmed": True}},),
        ),
    )

    assert progress_state["results_exhausted"] is True
    assert any("partial completion is now allowed" in hint.lower() for hint in hints)

    partial_decision = policy.on_finish_attempt(
        progress_state=progress_state,
        finish_attempt=mission_progress.FinishAttemptPayload(
            kind="terminal_event",
            event_name="job_collection_done",
            event_data={"outcome": "partial"},
        ),
    )
    done_decision = policy.on_finish_attempt(
        progress_state=progress_state,
        finish_attempt=mission_progress.FinishAttemptPayload(kind="done"),
    )

    assert partial_decision.allow is True
    assert done_decision.allow is False
    assert "cannot end" in done_decision.reason


def test_google_jobs_policy_search_context_mismatch_is_recoverable_not_ineligible() -> None:
    mission_progress, progress_policies = _load_progress_modules()
    policy = progress_policies.GoogleJobsMissionProgressPolicy(target_jobs=1)
    progress_state = policy.on_mission_start(
        mission="Collect one junior developer job",
        starting_url="https://google.com",
        base_knowledge=[],
    )

    progress_state, hints = policy.on_action_result(
        progress_state=progress_state,
        action_result=mission_progress.ActionResultPayload(
            action_name="process_focused_job",
            success=True,
            result_data={
                "saved": False,
                "processable": False,
                "reason": "search_context_mismatch",
                "search_context_valid": False,
                "recoverable": True,
            },
        ),
    )

    assert progress_state["saved_jobs"] == 0
    assert progress_state["consecutive_ineligible_jobs"] == 0
    assert progress_state["last_job_outcome"] == "search_context_mismatch"
    assert any("drifted away from the original search" in hint.lower() for hint in hints)


def test_mission_progress_payload_types_are_plain_python_data() -> None:
    mission_progress, _ = _load_progress_modules()

    action_result = mission_progress.ActionResultPayload(
        action_name="process_focused_job",
        action_args={"file_name": "google-jobs-list.jsonl"},
        success=True,
        result_data={"saved": True},
        accepted_event_names=("job_saved",),
        accepted_events=({"name": "job_saved", "data": {"job_title": "Lead iOS Developer"}},),
    )
    finish_attempt = mission_progress.FinishAttemptPayload(
        kind="terminal_event",
        event_name="job_collection_done",
        event_data={"outcome": "complete"},
    )
    decision = mission_progress.FinishDecision(allow=True, reason="", hint="")

    assert action_result.action_name == "process_focused_job"
    assert action_result.accepted_event_names == ("job_saved",)
    assert finish_attempt.event_data["outcome"] == "complete"
    assert decision.allow is True
