import json
import shutil
from pathlib import Path
import time

import chime
from agent import AgentEvent
from agent.agent_controller import Agent
from agent.events import EventDefinition
from core.config import Config, DebugConfig, ExecutionConfig, ModelConfig, SandboxConfig, StorageConfig
from job_application_app.document_generation import generate_cover_letter, generate_cv
from job_application_app.process_focused_job_tool import process_focused_job
from job_application_app.profile_store import build_cv_professional_summary, build_user_data

STABLE_AGENT_ID = "job_application_profile"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SKILLS_ROOT = PROJECT_ROOT / "skills" / "job-finder-skills"
WORKSPACE_SKILLS_DIRNAME = "agent_skills"


events = [
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
        },
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
        name="job_collection_done",
        description="The job collection mission finished with a non-failure outcome.",
        when="reporting terminal complete, partial, or empty results",
        schema={
            "outcome": str,
            "saved_count": int,
            "target_count": int,
        },
        once_per_mission=True,
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


def on_event(event: AgentEvent):
    payload = {
        "event_id": event.event_id,
        "action_id": event.action_id,
        "name": event.name,
        "data": event.data,
        "context": event.context,
    }
    print("Agent milestone event:")
    print(json.dumps(payload, indent=2, sort_keys=True))
    
    if event.name == "job_collection_done":
        chime.success()
    return {"ack": True}


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


def build_base_knowledge(
    user_details: dict,
    *,
    job_title: str,
    job_location: str,
    target_job_count: int,
    years_of_experience: str,
    is_remote: str,
    desired_salary: str,
    cv_professional_summary: str,
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

    remote_preference = "Yes" if is_remote.lower() in {"y", "yes", "true"} else "No"
    base_knowledge = ["Use these candidate facts when filling forms and writing answers."]
    full_name = f"{first_name} {last_name}".strip()
    if full_name:
        base_knowledge.append(f"Candidate name: {full_name}")
    if email:
        base_knowledge.append(f"Candidate email: {email}")
    if phone:
        base_knowledge.append(f"Candidate phone: {phone}")
    if address:
        base_knowledge.append(f"Candidate address: {address}")
    if city or state or post_code or country:
        location_parts = [part for part in [city, state, post_code, country] if part]
        base_knowledge.append(f"Candidate location: {', '.join(location_parts)}")
    if linkedin_url:
        base_knowledge.append(f"Candidate LinkedIn: {linkedin_url}")
    if github_url:
        base_knowledge.append(f"Candidate GitHub: {github_url}")
    if cv_path:
        base_knowledge.append(f"Candidate CV file path: {cv_path}")
    if years_of_experience:
        base_knowledge.append(f"Candidate years of experience: {years_of_experience}")
    if job_title:
        base_knowledge.append(f"Target job title: {job_title}")
    if job_location:
        base_knowledge.append(f"Target job location: {job_location}")
    base_knowledge.append(f"Target job count: {target_job_count}")
    base_knowledge.append(f"Remote preference: {remote_preference}")
    if desired_salary:
        base_knowledge.append(f"Desired salary: {desired_salary}")
    if cv_professional_summary:
        base_knowledge.append(
            "Professional summary from candidate CV (100 words): "
            f"{cv_professional_summary}"
        )

    return base_knowledge


def on_user_question(question: str, context: dict, options: list[str], multi_select: bool, yes_no: bool) -> str:
    print(f"User question: {question}")
    print(f"Context: {context}")
    print(f"Options: {options}")
    print(f"Multi-select: {multi_select}")
    print(f"Yes/No: {yes_no}")
    return input("Enter your answer: ").strip()


def find_jobs(debug: bool = False) -> list[dict]:
    config = Config(
        sandbox=SandboxConfig(enabled=False),
        debug=DebugConfig(
            debug_mode=debug,
            suppress_policy_debug_logs=True,
            suppress_live_telemetry_terminal_logs=True,
        ),
        model=ModelConfig(image_detail="low"),
        execution=ExecutionConfig(use_previous_response_id=False),
        storage=StorageConfig(
            base_dir="job-application-data/agents",
            default_persistence_mode="persistent",
        ),
    )

    with Agent(
        config=config,
        agent_id=STABLE_AGENT_ID,
        event_definitions=events,
        event_callback=on_event,
        user_question_callback=on_user_question,
    ) as agent:
        agent.register_tool(process_focused_job)
        sync_skills_to_agent_workspace(agent)
        user_details = build_user_data(agent)
        cv_professional_summary = build_cv_professional_summary(agent, user_details)

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
        target_job_count = max(1, min(target_job_count, 50))

        base_knowledge = build_base_knowledge(
            user_details,
            job_title=job_title,
            job_location=job_location,
            target_job_count=target_job_count,
            years_of_experience=years_of_experience,
            is_remote=is_remote,
            desired_salary=desired_salary,
            cv_professional_summary=cv_professional_summary,
        )

        agent_job_finder_result = agent.execute_mission(
            f"""First action: call activate_skill with skill_name="google-job-finder".
                Then follow that skill to find Google Jobs for "{job_title} {'in' if job_location else ''} {job_location}{' and is remote' if is_remote.lower() in {'y', 'yes', 'true'} else ''}",
                save exactly {target_job_count} new matching jobs unless results run out, and report completion.
                """,
            starting_url="https://google.com",
            base_knowledge=base_knowledge,
        )

        if agent_job_finder_result.success:
            google_jobs_list_path = agent.agent_workspace.written_data_dir / "google-jobs-list.jsonl"
            if google_jobs_list_path.exists():
                with open(google_jobs_list_path, "r", encoding="utf-8") as f:
                    return [json.loads(line) for line in f.readlines()]
        return []


def _print_cv_changes(changes: list[str], *, label: str) -> None:
    if not changes:
        return
    print(label)
    for change in changes:
        print(f"- {change}")


def main() -> None:
    jobs = find_jobs()

    print("Extracted jobs:")
    for i, job in enumerate(jobs):
        print(f"{i+1}. {job['job_title']} at {job['company_name']} in {job['location']}")

    while True:
        selected_job_index = int(input("Enter the number of the job you want to generate a CV for: ")) - 1
        selected_job = jobs[selected_job_index]

        print(
            f"Generating CV for {selected_job['job_title']} "
            f"at {selected_job['company_name']} in {selected_job['location']}"
        )

        cv_result = generate_cv(
            job=selected_job,
            project_root=PROJECT_ROOT,
            agent_id=STABLE_AGENT_ID,
        )
        print(f"CV generated successfully: {cv_result['output_pdf']}")
        _print_cv_changes(cv_result["changes"], label="CV changes:")

        while input("Do you want to revise the generated CV? (y/n): ").strip().lower() in {"y", "yes"}:
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

        if input("Do you want to generate a cover letter for this job? (y/n): ").strip().lower() in {"y", "yes"}:
            cover_letter_result = generate_cover_letter(
                job=selected_job,
                project_root=PROJECT_ROOT,
                agent_id=STABLE_AGENT_ID,
            )
            print(f"Cover letter generated successfully: {cover_letter_result['output_pdf']}")
            while (
                input("Do you want to revise the generated cover letter? (y/n): ").strip().lower()
                in {"y", "yes"}
            ):
                revision_request = input("Describe the cover letter changes you want: ").strip()
                cover_letter_result = generate_cover_letter(
                    job=selected_job,
                    project_root=PROJECT_ROOT,
                    agent_id=STABLE_AGENT_ID,
                    revision_request=revision_request,
                    existing_markdown_path=cover_letter_result["output_markdown"],
                )
                print(f"Updated cover letter saved to: {cover_letter_result['output_pdf']}")

        print("Do you want to generate documents for another job? (y/n)")
        if input().strip().lower() == "n":
            break


if __name__ == "__main__":
    main()
