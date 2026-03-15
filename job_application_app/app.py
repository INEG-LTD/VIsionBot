import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import chime
from agent import AgentEvent
from agent.agent_controller import Agent
from agent.events import EventDefinition
from core.config import Config, DebugConfig, ExecutionConfig, ModelConfig, SandboxConfig, StorageConfig
from job_application_app.document_generation import generate_cover_letter, generate_cv
from job_application_app.process_focused_job_tool import process_focused_job
from job_application_app.progress_policies import GoogleJobsMissionProgressPolicy
from job_application_app.profile_store import (
    build_application_preferences,
    build_cv_professional_summary,
    build_user_data,
    ensure_workspace_source_cv_alias,
    resolve_app_paths,
)
from lib.ai import ReasoningLevel

STABLE_AGENT_ID = "job_application_profile"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SKILLS_ROOT = PROJECT_ROOT / "skills" / "job-finder-skills"
WORKSPACE_SKILLS_DIRNAME = "agent_skills"
APPLICATIONS_DIRNAME = "applications"
APPLICATIONS_LOG_FILENAME = "applications-log.jsonl"
ACTIVE_RESUME_FILENAME = "active-application-resume.pdf"
ACTIVE_COVER_LETTER_FILENAME = "active-application-cover-letter.pdf"
ACTIVE_COVER_LETTER_MARKDOWN_FILENAME = "active-application-cover-letter.md"
ACTIVE_COVER_LETTER_TEXT_FILENAME = "active-application-cover-letter.txt"
ACTIVE_APPLICATION_CONTEXT_FILENAME = "active-application-context.json"


JOB_COLLECTION_EVENTS = [
    EventDefinition(
        name="jobs_page_ready",
        description="Google Jobs is open and the Jobs tab is highlighted.",
        when="the Google Jobs page is visible and the Jobs tab is already selected/highlighted",
        schema={"jobs_tab_highlighted": bool},
    ),
    EventDefinition(
        name="job_saved",
        description="A focused job was saved to the jobs JSONL file.",
        when="after reading a successful process_focused_job result and taking the next successful action",
        schema={
            "job_title": str,
            "file_name": str,
            "dedupe_key": str,
        },
        dedupe_by_payload=True,
    ),
    EventDefinition(
        name="job_rejected",
        description="A focused job was examined and not saved.",
        when="after reading a non-saved process_focused_job result and taking the next successful action",
        schema={
            "reason_code": str,
            "reason": str,
        },
    ),
    EventDefinition(
        name="job_results_exhausted",
        description="The agent explicitly confirmed there are no more viable unseen Google Jobs results to inspect.",
        when="after checking for more unseen cards, attempting recovery or scrolling if needed, and confirming results are exhausted",
        schema={"confirmed": bool},
    ),
    EventDefinition(
        name="job_collection_done",
        description="The job collection mission finished with a non-failure outcome.",
        when="reporting terminal complete, partial, or empty results",
        schema={
            "outcome": str,
            "saved_count": int,
            "target_count": int,
        },
        once_per_mission=True,
        terminal=True,
        allowed_tools=["report_data", "think"],
    ),
    EventDefinition(
        name="job_collection_failed",
        description="The job collection mission ended because it could not continue.",
        when="ending the mission due to an unrecoverable blocker",
        schema={
            "failure_code": str,
            "reason": str,
        },
        once_per_mission=True,
        terminal=True,
        allowed_tools=["flag", "think"],
    ),
    EventDefinition(
        name="job_collection_error",
        description="A recoverable workflow error or anomaly occurred.",
        when="a recoverable problem happened but the workflow may continue",
        schema={
            "error_code": str,
            "message": str,
            "recoverable": bool,
        },
    ),
]


APPLICATION_EVENTS = [
    EventDefinition(
        name="application_form_detected",
        description="The first real application form is visible.",
        when="the application form page is ready to start filling",
        schema={
            "form_url": str,
            "ats_type": str,
        },
    ),
    EventDefinition(
        name="application_auth_required",
        description="The workflow hit a login, registration, MFA, captcha, or verification gate.",
        when="the user must complete an auth step before the mission can continue",
        schema={
            "auth_type": str,
            "message": str,
        },
    ),
    EventDefinition(
        name="application_page_filled",
        description="A form page or major step was completed.",
        when="after filling the visible fields for a page and advancing or confirming the step",
        schema={
            "page_number": int,
            "fields_filled": int,
        },
    ),
    EventDefinition(
        name="application_file_uploaded",
        description="An application document upload completed.",
        when="after a resume or cover letter file upload succeeds",
        schema={
            "file_type": str,
            "file_name": str,
        },
    ),
    EventDefinition(
        name="application_review_reached",
        description="The final review step is visible.",
        when="the workflow reaches the review page or final confirmation screen before submit",
        schema={"review_visible": bool},
    ),
    EventDefinition(
        name="application_submit_attempted",
        description="The final submit button was clicked.",
        when="after the agent clicks the final submit button",
        schema={"submit_button_label": str},
    ),
    EventDefinition(
        name="application_submitted",
        description="The application was submitted and a confirmation is visible.",
        when="a confirmation page or clear success message is visible after submit",
        schema={
            "confirmation_detected": bool,
            "confirmation_text": str,
        },
        once_per_mission=True,
        terminal=True,
    ),
    EventDefinition(
        name="application_cancelled",
        description="The user chose not to submit the application.",
        when="the workflow ends because the user declined submission or asked to stop",
        schema={"reason": str},
        once_per_mission=True,
        terminal=True,
    ),
    EventDefinition(
        name="application_manual_followup_required",
        description="The application needs a human follow-up step outside the automated flow.",
        when="the workflow cannot finish directly but a clear next step is available for the user",
        schema={
            "followup_type": str,
            "instructions": str,
        },
        once_per_mission=True,
        terminal=True,
    ),
    EventDefinition(
        name="application_failed",
        description="The application mission ended because it could not continue.",
        when="an unrecoverable blocker stopped the workflow",
        schema={
            "failure_code": str,
            "reason": str,
        },
        once_per_mission=True,
        terminal=True,
    ),
    EventDefinition(
        name="application_error",
        description="A recoverable application problem occurred.",
        when="the workflow hit a problem but can still continue or recover",
        schema={
            "error_code": str,
            "message": str,
            "recoverable": bool,
        },
    ),
]


def make_event_callback(recorded_events: list[AgentEvent] | None = None):
    def _callback(event: AgentEvent):
        if recorded_events is not None:
            recorded_events.append(event)

        payload = {
            "event_id": event.event_id,
            "action_id": event.action_id,
            "name": event.name,
            "data": event.data,
            "context": event.context,
        }
        print("Agent milestone event:")
        print(json.dumps(payload, indent=2, sort_keys=True))

        if event.name in {"job_collection_done", "application_submitted"}:
            chime.success()
        elif event.name in {"application_failed", "application_manual_followup_required"}:
            chime.warning()
        return {"ack": True}

    return _callback


def build_agent_config(*, debug: bool = False) -> Config:
    return Config(
        sandbox=SandboxConfig(enabled=False),
        debug=DebugConfig(
            debug_mode=debug,
            suppress_policy_debug_logs=True,
            suppress_live_telemetry_terminal_logs=True,
        ),
        model=ModelConfig(image_detail="low", agent_reasoning_level=ReasoningLevel.HIGH),
        execution=ExecutionConfig(use_previous_response_id=False),
        storage=StorageConfig(
            base_dir="job-application-data/agents",
            default_persistence_mode="persistent",
        ),
    )


def sync_skills_to_agent_workspace(
    agent: Agent,
    *,
    source_root: Path = CANONICAL_SKILLS_ROOT,
    workspace_dir_name: str = WORKSPACE_SKILLS_DIRNAME,
) -> list[Path]:
    started = time.perf_counter()
    workspace_root = agent.agent_workspace.workspace_root
    target_root = (workspace_root / workspace_dir_name).resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    resolved_source = source_root.expanduser().resolve()
    if not resolved_source.is_dir():
        print(f"Skills source not found: {resolved_source}")
        agent.event_logger.skills_sync_completed(
            source_root=str(resolved_source),
            target_root=str(target_root),
            synced_count=0,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            error="source_not_found",
        )
        return []

    synced: list[Path] = []
    for skill_dir in sorted(resolved_source.iterdir()):
        if not skill_dir.is_dir():
            continue
        if not (skill_dir / "SKILL.md").is_file():
            continue
        destination = target_root / skill_dir.name
        shutil.copytree(skill_dir, destination, dirs_exist_ok=True)
        synced.append(destination)

    if synced:
        print(f"Synced {len(synced)} skills into {target_root}")
    else:
        print(f"No skills found under {resolved_source}")
    agent.event_logger.skills_sync_completed(
        source_root=str(resolved_source),
        target_root=str(target_root),
        synced_count=len(synced),
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )
    return synced


def _build_common_candidate_knowledge(
    user_details: dict,
    *,
    cv_professional_summary: str = "",
    desired_salary: str = "",
    source_cv_upload_path: str = "",
) -> list[str]:
    first_name = str(user_details.get("first_name", "")).strip()
    last_name = str(user_details.get("last_name", "")).strip()
    email = str(user_details.get("email", "")).strip()
    phone = str(user_details.get("phone", "")).strip()
    address = str(user_details.get("address", "")).strip()
    city = str(user_details.get("city", "")).strip()
    state = str(user_details.get("state", "")).strip()
    post_code = str(user_details.get("post_code", "")).strip()
    country = str(user_details.get("country", "")).strip()
    linkedin_url = str(user_details.get("linkedin_url", "")).strip()
    github_url = str(user_details.get("github_url", "")).strip()
    cv_path = str(user_details.get("cv_path", "")).strip()
    effective_source_cv_path = str(source_cv_upload_path or "").strip() or cv_path

    knowledge = ["Use these candidate facts when filling forms and writing answers."]
    full_name = f"{first_name} {last_name}".strip()
    if full_name:
        knowledge.append(f"Candidate name: {full_name}")
    if email:
        knowledge.append(f"Candidate email: {email}")
    if phone:
        knowledge.append(f"Candidate phone: {phone}")
    if address:
        knowledge.append(f"Candidate address: {address}")
    if city or state or post_code or country:
        location_parts = [part for part in [city, state, post_code, country] if part]
        knowledge.append(f"Candidate location: {', '.join(location_parts)}")
    if linkedin_url:
        knowledge.append(f"Candidate LinkedIn: {linkedin_url}")
    if github_url:
        knowledge.append(f"Candidate GitHub: {github_url}")
    if effective_source_cv_path:
        knowledge.append(f"Candidate source CV path: {effective_source_cv_path}")
    if desired_salary:
        knowledge.append(f"Desired salary: {desired_salary}")
    if cv_professional_summary:
        knowledge.append(
            "Professional summary from candidate CV (100 words): "
            f"{cv_professional_summary}"
        )
    return knowledge


def build_search_base_knowledge(
    user_details: dict,
    *,
    job_title: str,
    job_location: str,
    target_job_count: int,
    years_of_experience: str,
    is_remote: str,
    desired_salary: str,
    cv_professional_summary: str,
    source_cv_upload_path: str = "",
) -> list[str]:
    remote_preference = "Yes" if is_remote.lower() in {"y", "yes", "true"} else "No"
    base_knowledge = _build_common_candidate_knowledge(
        user_details,
        cv_professional_summary=cv_professional_summary,
        desired_salary=desired_salary,
        source_cv_upload_path=source_cv_upload_path,
    )
    if years_of_experience:
        base_knowledge.append(f"Candidate years of experience: {years_of_experience}")
    if job_title:
        base_knowledge.append(f"Target job title: {job_title}")
    if job_location:
        base_knowledge.append(f"Target job location: {job_location}")
    base_knowledge.append(f"Target job count: {target_job_count}")
    base_knowledge.append(f"Remote preference: {remote_preference}")
    return base_knowledge


def build_application_base_knowledge(
    user_details: dict,
    *,
    job: dict,
    application_url: str,
    cv_professional_summary: str,
    application_preferences: dict,
    staged_artifacts: dict[str, str],
    source_cv_upload_path: str = "",
) -> list[str]:
    desired_salary = str(application_preferences.get("desired_salary", "")).strip()
    base_knowledge = _build_common_candidate_knowledge(
        user_details,
        cv_professional_summary=cv_professional_summary,
        desired_salary=desired_salary,
        source_cv_upload_path=source_cv_upload_path,
    )

    job_title = str(job.get("job_title", "")).strip()
    company_name = str(job.get("company_name", "")).strip()
    location = str(job.get("location", "")).strip()
    job_summary = str(job.get("job_summary", "")).strip()
    work_authorization = str(application_preferences.get("work_authorization", "")).strip()
    sponsorship_needed = str(application_preferences.get("sponsorship_needed", "")).strip()
    notice_period = str(application_preferences.get("notice_period", "")).strip()
    earliest_start_date = str(application_preferences.get("earliest_start_date", "")).strip()
    free_text_mode = str(application_preferences.get("free_text_mode", "ask")).strip() or "ask"
    prefill_review_mode = (
        str(application_preferences.get("prefill_review_mode", "smart")).strip() or "smart"
    )
    marketing_policy = (
        str(application_preferences.get("marketing_opt_in_policy", "auto_deny")).strip() or "auto_deny"
    )
    sms_policy = (
        str(application_preferences.get("sms_consent_policy", "auto_deny")).strip() or "auto_deny"
    )
    talent_pool_policy = (
        str(application_preferences.get("talent_pool_policy", "auto_deny")).strip() or "auto_deny"
    )
    eeo_preferences = application_preferences.get("eeo_preferences", {})

    if company_name:
        base_knowledge.append(f"Target company: {company_name}")
    if job_title:
        base_knowledge.append(f"Target job title: {job_title}")
    if location:
        base_knowledge.append(f"Target job location: {location}")
    if job_summary:
        base_knowledge.append(f"Target job summary: {job_summary}")
    base_knowledge.append(f"Application URL: {application_url}")
    if work_authorization:
        base_knowledge.append(f"Work authorization: {work_authorization}")
    if sponsorship_needed:
        base_knowledge.append(f"Visa sponsorship needed: {sponsorship_needed}")
    if notice_period:
        base_knowledge.append(f"Notice period: {notice_period}")
    if earliest_start_date:
        base_knowledge.append(f"Earliest start date: {earliest_start_date}")
    base_knowledge.append(f"Free text response mode: {free_text_mode}")
    base_knowledge.append(f"Prefill review mode: {prefill_review_mode}")
    base_knowledge.append(f"Marketing opt-in policy: {marketing_policy}")
    base_knowledge.append(f"SMS consent policy: {sms_policy}")
    base_knowledge.append(f"Talent pool consent policy: {talent_pool_policy}")
    base_knowledge.append(
        "Required terms and conditions policy: auto accept required legal terms, privacy acknowledgements, "
        "and 'I certify this is accurate' confirmations."
    )
    if isinstance(eeo_preferences, dict) and eeo_preferences:
        base_knowledge.append(
            "EEO preferences: "
            f"{json.dumps(eeo_preferences, sort_keys=True)}"
        )

    base_knowledge.append(
        "Use the exact staged application file paths from these rules when calling upload_file or read_file."
    )
    base_knowledge.append(f"Resume upload file path: {staged_artifacts['resume_upload_path']}")
    base_knowledge.append(
        f"Cover letter upload file path: {staged_artifacts['cover_letter_upload_path']}"
    )
    base_knowledge.append(
        f"Cover letter markdown file path: {staged_artifacts['cover_letter_markdown_path']}"
    )
    base_knowledge.append(
        f"Cover letter text file path: {staged_artifacts['cover_letter_text_path']}"
    )
    base_knowledge.append(
        f"Application context file path: {staged_artifacts['application_context_path']}"
    )
    base_knowledge.append(
        "Use the staged resume and staged cover letter for uploads. Use the cover letter text file for any "
        "cover letter text area or essay box that clearly requests the cover letter body."
    )
    base_knowledge.append(
        "This mission covers exactly one job application. Do not search for other jobs or leave the target flow unless recovering."
    )
    base_knowledge.append(
        "Always ask the user before the final submit click, even if all form fields are complete."
    )
    return base_knowledge


def on_user_question(question: str, context: dict, options: list[str], multi_select: bool, yes_no: bool) -> str:
    print(f"User question: {question}")
    print(f"Context: {context}")
    print(f"Options: {options}")
    print(f"Multi-select: {multi_select}")
    print(f"Yes/No: {yes_no}")
    return input("Enter your answer: ").strip()


def _prompt_yes_no(prompt: str, *, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    answer = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def _job_label(job: dict) -> str:
    title = str(job.get("job_title", "")).strip() or "Unknown role"
    company = str(job.get("company_name", "")).strip() or "Unknown company"
    location = str(job.get("location", "")).strip()
    if location:
        return f"{title} at {company} in {location}"
    return f"{title} at {company}"


def _safe_slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower())
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "job-application"


def _cover_letter_text_from_markdown(markdown: str) -> str:
    text = str(markdown or "").replace("<br><br>", "\n\n").replace("<br>", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text + "\n" if text else ""


def _copy_file(source_path: str | Path, destination_path: Path) -> Path:
    source = Path(source_path).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination_path)
    return destination_path


def _write_text_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content or ""), encoding="utf-8")
    return path


def _write_json_file(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def stage_application_artifacts(
    agent: Agent,
    *,
    job: dict,
    application_url: str,
    cv_result: dict,
    cover_letter_result: dict,
) -> dict[str, str]:
    workspace_root = agent.agent_workspace.workspace_root
    application_slug = _safe_slug(
        f"{job.get('company_name', '')}-{job.get('job_title', '')}"
    )
    application_dir = workspace_root / APPLICATIONS_DIRNAME / application_slug
    application_dir.mkdir(parents=True, exist_ok=True)

    canonical_resume_path = _copy_file(
        cv_result["output_pdf"],
        application_dir / "resume.pdf",
    )
    canonical_cover_letter_path = _copy_file(
        cover_letter_result["output_pdf"],
        application_dir / "cover-letter.pdf",
    )
    canonical_cover_letter_markdown_path = _copy_file(
        cover_letter_result["output_markdown"],
        application_dir / "cover-letter.md",
    )
    cover_letter_text = _cover_letter_text_from_markdown(
        str(cover_letter_result.get("cover_letter_markdown", "")).strip()
        or canonical_cover_letter_markdown_path.read_text(encoding="utf-8", errors="replace")
    )
    canonical_cover_letter_text_path = _write_text_file(
        application_dir / "cover-letter.txt",
        cover_letter_text,
    )

    active_resume_path = _copy_file(
        canonical_resume_path,
        workspace_root / ACTIVE_RESUME_FILENAME,
    )
    active_cover_letter_path = _copy_file(
        canonical_cover_letter_path,
        workspace_root / ACTIVE_COVER_LETTER_FILENAME,
    )
    active_cover_letter_markdown_path = _copy_file(
        canonical_cover_letter_markdown_path,
        workspace_root / ACTIVE_COVER_LETTER_MARKDOWN_FILENAME,
    )
    active_cover_letter_text_path = _write_text_file(
        workspace_root / ACTIVE_COVER_LETTER_TEXT_FILENAME,
        cover_letter_text,
    )

    application_context = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job": {
            "job_title": str(job.get("job_title", "")).strip(),
            "company_name": str(job.get("company_name", "")).strip(),
            "location": str(job.get("location", "")).strip(),
            "job_summary": str(job.get("job_summary", "")).strip(),
        },
        "application_url": application_url,
        "artifacts": {
            "canonical_resume_path": str(canonical_resume_path),
            "canonical_cover_letter_path": str(canonical_cover_letter_path),
            "canonical_cover_letter_markdown_path": str(canonical_cover_letter_markdown_path),
            "canonical_cover_letter_text_path": str(canonical_cover_letter_text_path),
            "active_resume_path": str(active_resume_path),
            "active_cover_letter_path": str(active_cover_letter_path),
            "active_cover_letter_markdown_path": str(active_cover_letter_markdown_path),
            "active_cover_letter_text_path": str(active_cover_letter_text_path),
        },
    }
    canonical_context_path = _write_json_file(
        application_dir / "application-context.json",
        application_context,
    )
    active_context_path = _write_json_file(
        workspace_root / ACTIVE_APPLICATION_CONTEXT_FILENAME,
        application_context,
    )

    return {
        "application_dir": str(application_dir),
        "resume_upload_path": str(active_resume_path),
        "cover_letter_upload_path": str(active_cover_letter_path),
        "cover_letter_markdown_path": str(active_cover_letter_markdown_path),
        "cover_letter_text_path": str(active_cover_letter_text_path),
        "application_context_path": str(active_context_path),
        "canonical_resume_path": str(canonical_resume_path),
        "canonical_cover_letter_path": str(canonical_cover_letter_path),
        "canonical_cover_letter_text_path": str(canonical_cover_letter_text_path),
        "canonical_application_context_path": str(canonical_context_path),
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def log_application_outcome(
    agent: Agent,
    *,
    job: dict,
    application_url: str,
    mission_result,
    recorded_events: list[AgentEvent],
    staged_artifacts: dict[str, str],
) -> dict[str, Any]:
    terminal_names = {
        "application_submitted",
        "application_cancelled",
        "application_manual_followup_required",
        "application_failed",
    }
    terminal_event = next((event for event in reversed(recorded_events) if event.name in terminal_names), None)

    outcome = terminal_event.name if terminal_event else ("application_submitted" if mission_result.success else "application_failed")
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_title": str(job.get("job_title", "")).strip(),
        "company_name": str(job.get("company_name", "")).strip(),
        "location": str(job.get("location", "")).strip(),
        "application_url": application_url,
        "outcome": outcome,
        "terminal_event": {
            "name": terminal_event.name,
            "data": terminal_event.data,
        } if terminal_event else None,
        "mission": {
            "success": bool(mission_result.success),
            "partial": bool(getattr(mission_result, "partial", False)),
            "reasoning": str(getattr(mission_result, "reasoning", "")).strip(),
            "final_answer_draft": str(getattr(mission_result, "final_answer_draft", "")).strip(),
            "final_url": str(getattr(mission_result, "final_url", "")).strip(),
            "failure_code": str(getattr(mission_result, "failure_code", "")).strip(),
            "failure_stage": str(getattr(mission_result, "failure_stage", "")).strip(),
            "total_iterations": int(getattr(mission_result, "total_iterations", 0) or 0),
            "total_actions": int(getattr(mission_result, "total_actions", 0) or 0),
        },
        "artifacts": staged_artifacts,
        "events": [
            {
                "name": event.name,
                "data": event.data,
            }
            for event in recorded_events
        ],
    }
    _append_jsonl(agent.agent_workspace.outputs_root / APPLICATIONS_LOG_FILENAME, payload)
    return payload


def _print_application_outcome_summary(application_log: dict[str, Any]) -> None:
    print("Application run recorded.")
    print(f"Outcome: {application_log.get('outcome', 'unknown')}")

    terminal_event = application_log.get("terminal_event") or {}
    terminal_name = str(terminal_event.get("name", "") or "").strip()
    terminal_data = terminal_event.get("data") or {}

    if not terminal_name:
        return

    print(f"Terminal event: {terminal_name}")

    if terminal_name == "application_submitted":
        confirmation_text = str(terminal_data.get("confirmation_text", "") or "").strip()
        if confirmation_text:
            print(f"Confirmation: {confirmation_text}")
        return

    if terminal_name == "application_manual_followup_required":
        followup_type = str(terminal_data.get("followup_type", "") or "").strip()
        instructions = str(terminal_data.get("instructions", "") or "").strip()
        if followup_type:
            print(f"Follow-up type: {followup_type}")
        if instructions:
            print(f"Next step: {instructions}")
        return

    if terminal_name == "application_failed":
        failure_code = str(terminal_data.get("failure_code", "") or "").strip()
        reason = str(terminal_data.get("reason", "") or "").strip()
        if failure_code:
            print(f"Failure code: {failure_code}")
        if reason:
            print(f"Reason: {reason}")
        return

    if terminal_name == "application_cancelled":
        reason = str(terminal_data.get("reason", "") or "").strip()
        if reason:
            print(f"Reason: {reason}")


def _normalize_search_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def _tokenize_search_text(value: str) -> set[str]:
    normalized = _normalize_search_text(value)
    if not normalized:
        return set()
    return {token for token in normalized.split() if token}


def _search_queries_match(expected_query: str, observed_query: str) -> bool:
    expected_normalized = _normalize_search_text(expected_query)
    observed_normalized = _normalize_search_text(observed_query)
    if not expected_normalized or not observed_normalized:
        return False
    if expected_normalized == observed_normalized:
        return True
    if expected_normalized in observed_normalized or observed_normalized in expected_normalized:
        return True

    expected_tokens = _tokenize_search_text(expected_query)
    observed_tokens = _tokenize_search_text(observed_query)
    if not expected_tokens or not observed_tokens:
        return False
    overlap_ratio = len(expected_tokens.intersection(observed_tokens)) / float(len(expected_tokens))
    return overlap_ratio >= 0.75


def _build_google_jobs_search_query(*, job_title: str, job_location: str, is_remote: str) -> str:
    parts: list[str] = []
    if job_title:
        parts.append(job_title.strip())
    if job_location:
        location = job_location.strip()
        if parts:
            parts.append(f"in {location}")
        else:
            parts.append(location)
    if is_remote.lower() in {"y", "yes", "true"}:
        parts.append("remote")
    return " ".join(part for part in parts if part).strip()


def _load_matching_saved_google_jobs(
    path: Path,
    *,
    requested_search_query: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    matches: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            saved_search_query = str(payload.get("search_query", "") or "").strip()
            if not saved_search_query:
                continue
            if _search_queries_match(requested_search_query, saved_search_query):
                matches.append(payload)
    return matches


def find_jobs(debug: bool = False) -> list[dict]:
    job_title = input("Enter the job title you are applying for: ").strip()
    job_location = input("Enter the job location you are applying to: ").strip()
    is_remote = input("Are you looking for a remote job? (y/n): ").strip()
    years_of_experience = input("Enter your years of experience: ").strip()
    desired_salary = input("Enter your desired salary: ").strip()
    target_job_count_raw = input("How many matching jobs should be saved? [10]: ").strip()
    try:
        target_job_count = int(target_job_count_raw) if target_job_count_raw else 10
    except ValueError:
        target_job_count = 10
    normalized_target_job_count = (
        0 if target_job_count == 0 else max(1, min(target_job_count, 50))
    )
    app_paths = resolve_app_paths(project_root=PROJECT_ROOT, agent_id=STABLE_AGENT_ID)
    google_jobs_list_path = app_paths.outputs_root / "google-jobs-list.jsonl"
    requested_search_query = _build_google_jobs_search_query(
        job_title=job_title,
        job_location=job_location,
        is_remote=is_remote,
    )

    if normalized_target_job_count == 0:
        return _load_matching_saved_google_jobs(
            google_jobs_list_path,
            requested_search_query=requested_search_query,
        )

    with Agent(
        config=build_agent_config(debug=debug),
        agent_id=STABLE_AGENT_ID,
        event_definitions=JOB_COLLECTION_EVENTS,
        event_callback=make_event_callback(),
        mission_progress_policy=GoogleJobsMissionProgressPolicy(target_jobs=normalized_target_job_count),
        user_question_callback=on_user_question,
    ) as agent:
        agent.register_tool(process_focused_job)
        sync_skills_to_agent_workspace(agent)
        user_details = build_user_data(agent)
        cv_professional_summary = build_cv_professional_summary(agent, user_details)
        source_cv_upload_path = ensure_workspace_source_cv_alias(
            agent.agent_workspace.workspace_root,
            user_details,
        )

        base_knowledge = build_search_base_knowledge(
            user_details,
            job_title=job_title,
            job_location=job_location,
            target_job_count=normalized_target_job_count,
            years_of_experience=years_of_experience,
            is_remote=is_remote,
            desired_salary=desired_salary,
            cv_professional_summary=cv_professional_summary,
            source_cv_upload_path=source_cv_upload_path,
        )

        mission_result = agent.execute_mission(
            f"""First action: call activate_skill with skill_name="google-job-finder".
                Then follow that skill to find Google Jobs for "{job_title} {'in' if job_location else ''} {job_location}{' and is remote' if is_remote.lower() in {'y', 'yes', 'true'} else ''}",
                save exactly {normalized_target_job_count} new matching jobs unless results run out, and report completion.
                """,
            starting_url="https://google.com",
            base_knowledge=base_knowledge,
        )

        if mission_result.success:
            if google_jobs_list_path.exists():
                with google_jobs_list_path.open("r", encoding="utf-8") as handle:
                    return [json.loads(line) for line in handle if line.strip()]
        return []


def apply_to_job(
    *,
    job: dict,
    application_url: str,
    cv_result: dict,
    cover_letter_result: dict,
    debug: bool = False,
) -> dict[str, Any]:
    recorded_events: list[AgentEvent] = []
    with Agent(
        config=build_agent_config(debug=debug),
        agent_id=STABLE_AGENT_ID,
        event_definitions=APPLICATION_EVENTS,
        event_callback=make_event_callback(recorded_events),
        user_question_callback=on_user_question,
    ) as agent:
        sync_skills_to_agent_workspace(agent)
        user_details = build_user_data(agent)
        application_preferences = build_application_preferences(agent, user_details)
        cv_professional_summary = build_cv_professional_summary(agent, user_details)
        source_cv_upload_path = ensure_workspace_source_cv_alias(
            agent.agent_workspace.workspace_root,
            user_details,
        )
        staged_artifacts = stage_application_artifacts(
            agent,
            job=job,
            application_url=application_url,
            cv_result=cv_result,
            cover_letter_result=cover_letter_result,
        )
        base_knowledge = build_application_base_knowledge(
            user_details,
            job=job,
            application_url=application_url,
            cv_professional_summary=cv_professional_summary,
            application_preferences=application_preferences,
            staged_artifacts=staged_artifacts,
            source_cv_upload_path=source_cv_upload_path,
        )

        mission_result = agent.execute_mission(
            """First action: call activate_skill with skill_name="job-application-filler".
            Then follow that skill to complete this single job application end to end.
            Use the staged application files and candidate facts from CUSTOM RULES.
            If the site requires login, registration, verification, MFA, or captcha solving, pause and ask the user.
            Ask the user again before the final submit click.
            If the flow needs a human-only follow-up step outside the browser, explain it clearly and report the outcome.""",
            starting_url=application_url,
            base_knowledge=base_knowledge,
        )

        return log_application_outcome(
            agent,
            job=job,
            application_url=application_url,
            mission_result=mission_result,
            recorded_events=recorded_events,
            staged_artifacts=staged_artifacts,
        )


def _print_cv_changes(changes: list[str], *, label: str) -> None:
    if not changes:
        return
    print(label)
    for change in changes:
        print(f"- {change}")


def _select_job(jobs: list[dict]) -> dict:
    while True:
        raw_value = input("Enter the number of the job you want to work on: ").strip()
        try:
            selected_job_index = int(raw_value) - 1
        except ValueError:
            print("Please enter a valid number.")
            continue
        if 0 <= selected_job_index < len(jobs):
            return jobs[selected_job_index]
        print(f"Please choose a number between 1 and {len(jobs)}.")


def _choose_application_url(job: dict) -> str:
    preferred_direct_link = str(job.get("apply_directly_link", "") or "").strip()
    if preferred_direct_link:
        print(f"Using preferred direct-apply link: {preferred_direct_link}")
        return preferred_direct_link

    apply_links = []
    for item in job.get("apply_links", []) or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        apply_links.append(
            {
                "label": str(item.get("label", "Apply")).strip() or "Apply",
                "url": url,
            }
        )

    if apply_links:
        print("Available application links:")
        for index, item in enumerate(apply_links, start=1):
            print(f"{index}. {item['label']}: {item['url']}")
        while True:
            choice = input(
                "Choose an application link number, press Enter for the first link, or type M to paste a manual URL: "
            ).strip()
            if not choice:
                return apply_links[0]["url"]
            if choice.lower() == "m":
                manual_url = input("Enter the application URL: ").strip()
                if manual_url:
                    return manual_url
                print("Please enter a valid URL.")
                continue
            try:
                selected_index = int(choice) - 1
            except ValueError:
                print("Please enter a valid number or M.")
                continue
            if 0 <= selected_index < len(apply_links):
                return apply_links[selected_index]["url"]
            print(f"Please choose a number between 1 and {len(apply_links)}.")

    while True:
        manual_url = input("Enter the application URL for this job: ").strip()
        if manual_url:
            return manual_url
        print("An application URL is required to continue.")


def _generate_cover_letter_for_job(job: dict) -> dict:
    cover_letter_result = generate_cover_letter(
        job=job,
        project_root=PROJECT_ROOT,
        agent_id=STABLE_AGENT_ID,
    )
    print(f"Cover letter generated successfully: {cover_letter_result['output_pdf']}")
    while _prompt_yes_no("Do you want to revise the generated cover letter?", default=False):
        revision_request = input("Describe the cover letter changes you want: ").strip()
        cover_letter_result = generate_cover_letter(
            job=job,
            project_root=PROJECT_ROOT,
            agent_id=STABLE_AGENT_ID,
            revision_request=revision_request,
            existing_markdown_path=cover_letter_result["output_markdown"],
        )
        print(f"Updated cover letter saved to: {cover_letter_result['output_pdf']}")
    return cover_letter_result


def main() -> None:
    jobs = find_jobs()
    if not jobs:
        print("No jobs were collected.")
        return

    print("Extracted jobs:")
    for index, job in enumerate(jobs, start=1):
        print(f"{index}. {_job_label(job)}")

    while True:
        selected_job = _select_job(jobs)

        print(f"Generating CV for {_job_label(selected_job)}")
        cv_result = generate_cv(
            job=selected_job,
            project_root=PROJECT_ROOT,
            agent_id=STABLE_AGENT_ID,
        )
        print(f"CV generated successfully: {cv_result['output_pdf']}")
        _print_cv_changes(cv_result["changes"], label="CV changes:")

        while _prompt_yes_no("Do you want to revise the generated CV?", default=False):
            revision_request = input("Describe the CV changes you want: ").strip()
            cv_result = generate_cv(
                job=selected_job,
                project_root=PROJECT_ROOT,
                agent_id=STABLE_AGENT_ID,
                revision_request=revision_request,
                existing_markdown_path=cv_result["output_markdown"],
            )
            print(f"Updated CV saved to: {cv_result['output_pdf']}")
            _print_cv_changes(cv_result["changes"], label="Updated CV changes:")

        cover_letter_result: dict | None = None
        if _prompt_yes_no("Do you want to generate a cover letter for this job?", default=True):
            cover_letter_result = _generate_cover_letter_for_job(selected_job)

        if _prompt_yes_no("Do you want to apply to this job now?", default=False):
            if cover_letter_result is None:
                print("A tailored cover letter is required before applying.")
                if _prompt_yes_no("Generate the cover letter now?", default=True):
                    cover_letter_result = _generate_cover_letter_for_job(selected_job)
                else:
                    print("Skipping the application because the cover letter was not generated.")
            if cover_letter_result is not None:
                application_url = _choose_application_url(selected_job)
                application_log = apply_to_job(
                    job=selected_job,
                    application_url=application_url,
                    cv_result=cv_result,
                    cover_letter_result=cover_letter_result,
                    debug=True,
                )
                _print_application_outcome_summary(application_log)

        if not _prompt_yes_no("Do you want to work on another job?", default=False):
            break


if __name__ == "__main__":
    main()
