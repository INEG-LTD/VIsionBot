import json
import shutil
import tkinter as tk
from hashlib import sha256
from pathlib import Path
import re
from tkinter import filedialog

from agent import AgentEvent
from agent.agent_controller import Agent
from agent.events import EventDefinition
from core.config import Config, DebugConfig, ModelConfig, SandboxConfig, StorageConfig
from job_application_app.modify_cv_tool import _file_to_markdown
from lib.ai import generate_text

STABLE_AGENT_ID = "job_application_profile"
USER_DETAILS_FILENAME = "user_details.json"
CV_MARKDOWN_FILENAME = "cv_markdown.md"
CV_SUMMARY_CACHE_FILENAME = "cv_summary_cache.json"
CV_SUMMARY_WORD_COUNT = 100

config = Config(
    sandbox=SandboxConfig(
        enabled=False
    ),
    debug=DebugConfig(
        debug_mode=True,
        suppress_policy_debug_logs=True,
        suppress_live_telemetry_terminal_logs=True
    ),
    model=ModelConfig(
        image_detail="low"
    ),
    storage=StorageConfig(
        base_dir="job-application-data/agents",
        default_persistence_mode="persistent",
    )
)

events = [
    EventDefinition(
        name="google_visited",
        description="Google was visited",
        schema={}
    )
]

def on_event(event: AgentEvent):
    print(event)


def _pick_file() -> str:
    root = tk.Tk()
    root.withdraw()  # hide the main window
    try:
        file_path = filedialog.askopenfilename()
    finally:
        root.destroy()
    return str(file_path or "").strip()


def _request_user_profile(workspace_root: Path) -> dict:
    print("No saved user profile found. Please enter your details.")
    first_name = input("Enter your name: ").strip()
    last_name = input("Enter your last name: ").strip()
    email = input("Enter your email: ").strip()
    phone = input("Enter your phone number: ").strip()
    address = input("Enter your address: ").strip()
    city = input("Enter your city: ").strip()
    state = input("Enter your state: ").strip()
    post_code = input("Enter your post code: ").strip()
    country = input("Enter your country: ").strip()
    linkedin_url = input("Enter your LinkedIn URL: ").strip()
    github_url = input("Enter your GitHub URL: ").strip()

    target_cv = _copy_cv_to_workspace(
        workspace_root,
        prompt_message="Please select your CV file.",
    )

    return {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone": phone,
        "address": address,
        "city": city,
        "state": state,
        "post_code": post_code,
        "country": country,
        "linkedin_url": linkedin_url,
        "github_url": github_url,
        "cv_path": str(target_cv),
    }


def _copy_cv_to_workspace(workspace_root: Path, prompt_message: str) -> Path:
    print(prompt_message)
    selected_cv = _pick_file()
    if not selected_cv:
        raise RuntimeError("No CV file selected.")
    source_cv = Path(selected_cv).expanduser().resolve()
    if not source_cv.exists():
        raise FileNotFoundError(f"Selected CV file does not exist: {source_cv}")

    target_cv = workspace_root / f"user_cv{source_cv.suffix.lower()}"
    shutil.copy2(source_cv, target_cv)
    return target_cv


def build_user_data(agent: Agent) -> dict:
    workspace_root = agent.agent_workspace.workspace_root
    user_details_path = workspace_root / USER_DETAILS_FILENAME
    user_details: dict = {}

    if user_details_path.exists():
        try:
            loaded = json.loads(user_details_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                user_details = loaded
            else:
                print(f"Saved profile at {user_details_path} is invalid. Recreating it.")
        except Exception:
            print(f"Failed to read saved profile at {user_details_path}. Recreating it.")

    if not user_details:
        user_details = _request_user_profile(workspace_root)
        user_details_path.write_text(
            json.dumps(user_details, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Saved user profile to {user_details_path}")
        return user_details

    cv_path_raw = str(user_details.get("cv_path", "")).strip()
    cv_path = Path(cv_path_raw).expanduser().resolve() if cv_path_raw else None
    if cv_path is None or not cv_path.exists():
        target_cv = _copy_cv_to_workspace(
            workspace_root,
            prompt_message="Saved profile CV is missing or unavailable. Please select your CV file again.",
        )
        user_details["cv_path"] = str(target_cv)
        user_details_path.write_text(
            json.dumps(user_details, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(f"Loaded user profile from {user_details_path}")
    return user_details


def _count_words(text: str) -> int:
    return len([word for word in str(text or "").split() if word.strip()])


def _truncate_to_words(text: str, max_words: int) -> str:
    words = [word for word in str(text or "").split() if word.strip()]
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _summarize_cv_markdown(cv_markdown: str) -> str:
    system_prompt = (
        "You are an expert resume writer. Generate a professional candidate summary "
        "strictly from the provided CV markdown."
    )
    prompt = (
        f"Write a professional summary of exactly {CV_SUMMARY_WORD_COUNT} words.\n"
        "Use only details present in the CV.\n"
        "No bullet points. No title. No extra commentary.\n\n"
        "CV markdown:\n"
        "```markdown\n"
        f"{cv_markdown}\n"
        "```"
    )

    first_attempt = generate_text(
        prompt=prompt,
        system_prompt=system_prompt,
    ).strip()
    first_attempt = re.sub(r"\s+", " ", first_attempt).strip()
    if _count_words(first_attempt) == CV_SUMMARY_WORD_COUNT:
        return first_attempt

    second_prompt = (
        f"Rewrite this text to exactly {CV_SUMMARY_WORD_COUNT} words while preserving meaning:\n\n"
        f"{first_attempt}"
    )
    second_attempt = generate_text(
        prompt=second_prompt,
        system_prompt=system_prompt,
    ).strip()
    second_attempt = re.sub(r"\s+", " ", second_attempt).strip()
    if _count_words(second_attempt) >= CV_SUMMARY_WORD_COUNT:
        return _truncate_to_words(second_attempt, CV_SUMMARY_WORD_COUNT)
    if second_attempt:
        return second_attempt
    return _truncate_to_words(first_attempt, CV_SUMMARY_WORD_COUNT)


def build_cv_professional_summary(agent: Agent, user_details: dict) -> str:
    workspace_root = agent.agent_workspace.workspace_root
    cv_path_raw = str(user_details.get("cv_path", "")).strip()
    if not cv_path_raw:
        return ""

    cv_path = Path(cv_path_raw).expanduser().resolve()
    if not cv_path.exists():
        return ""

    try:
        cv_markdown = _file_to_markdown(cv_path)
    except Exception as exc:
        print(f"Failed to convert CV to markdown for summary: {exc}")
        return ""

    if not cv_markdown.strip():
        return ""

    markdown_path = workspace_root / CV_MARKDOWN_FILENAME
    markdown_path.write_text(cv_markdown.strip() + "\n", encoding="utf-8")

    markdown_hash = sha256(cv_markdown.encode("utf-8")).hexdigest()
    cache_path = workspace_root / CV_SUMMARY_CACHE_FILENAME
    if cache_path.exists():
        try:
            cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                isinstance(cache_payload, dict)
                and str(cache_payload.get("cv_markdown_sha256", "")).strip() == markdown_hash
            ):
                cached_summary = str(cache_payload.get("summary", "")).strip()
                if cached_summary:
                    return cached_summary
        except Exception:
            pass

    try:
        summary = _summarize_cv_markdown(cv_markdown)
    except Exception as exc:
        print(f"Failed to generate CV summary with LLM: {exc}")
        return ""

    summary = re.sub(r"\s+", " ", str(summary or "")).strip()
    if not summary:
        return ""

    cache_payload = {
        "cv_path": str(cv_path),
        "cv_markdown_sha256": markdown_hash,
        "word_count": _count_words(summary),
        "summary": summary,
    }
    cache_path.write_text(
        json.dumps(cache_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def build_base_knowledge(
    user_details: dict,
    *,
    job_title: str,
    job_location: str,
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
    base_knowledge.append(f"Remote preference: {remote_preference}")
    if desired_salary:
        base_knowledge.append(f"Desired salary: {desired_salary}")
    if cv_professional_summary:
        base_knowledge.append(
            "Professional summary from candidate CV (100 words): "
            f"{cv_professional_summary}"
        )

    return base_knowledge

with Agent(
    config=config,
    agent_id=STABLE_AGENT_ID,
    event_definitions=events,
    event_callback=on_event
) as agent:
    user_details = build_user_data(agent)
    cv_professional_summary = build_cv_professional_summary(agent, user_details)

    # Request job preferences
    job_title = input("Enter the job title you are applying for: ").strip()
    job_location = input("Enter the job location you are applying for: ").strip()
    is_remote = input("Are you looking for a remote job? (y/n): ").strip()
    years_of_experience = input("Enter your years of experience: ").strip()
    desired_salary = input("Enter your desired salary: ").strip()

    base_knowledge = build_base_knowledge(
        user_details,
        job_title=job_title,
        job_location=job_location,
        years_of_experience=years_of_experience,
        is_remote=is_remote,
        desired_salary=desired_salary,
        cv_professional_summary=cv_professional_summary,
    )

    agent.execute_mission(
        "go to google. emit google_visited when you reach there",
        starting_url="https://google.com",
        base_knowledge=base_knowledge,
    )
