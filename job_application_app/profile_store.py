from __future__ import annotations

import json
import shutil
import tkinter as tk
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from tkinter import filedialog

from agent.agent_controller import Agent
from core.agent_workspace import resolve_agent_workspace
from job_application_app.modify_cv_tool import _file_to_markdown
from lib.ai import generate_text

USER_DETAILS_FILENAME = "user_details.json"
CV_MARKDOWN_FILENAME = "cv_markdown.md"
CV_SUMMARY_CACHE_FILENAME = "cv_summary_cache.json"
CV_SUMMARY_WORD_COUNT = 100
APPLICATION_PREFERENCES_KEY = "application_preferences"
EEO_PREFERENCES_KEY = "eeo_preferences"
ACTIVE_SOURCE_CV_STEM = "active-source-cv"
SUPPORTING_DOCUMENTS_DIRNAME = "supporting_documents"
COMMON_SUPPORTING_DOCUMENTS = (
    ("portfolio", "portfolio file"),
    ("transcript", "transcript"),
    ("certification", "certification"),
    ("work_sample", "work sample"),
    ("visa_document", "visa document"),
)


@dataclass(frozen=True)
class AppPaths:
    agent_root: Path
    profile_root: Path
    workspace_root: Path
    outputs_root: Path


def resolve_app_paths(*, project_root: Path, agent_id: str) -> AppPaths:
    workspace = resolve_agent_workspace(
        base_dir=project_root / "job-application-data" / "agents",
        agent_id=agent_id,
        persistence_mode="persistent",
    )
    _migrate_profile_files(workspace.workspace_root, workspace.profile_root)
    return AppPaths(
        agent_root=workspace.agent_root,
        profile_root=workspace.profile_root,
        workspace_root=workspace.workspace_root,
        outputs_root=workspace.outputs_root,
    )


def _pick_file() -> str:
    root = tk.Tk()
    root.withdraw()
    try:
        file_path = filedialog.askopenfilename()
    finally:
        root.destroy()
    return str(file_path or "").strip()


def _copy_cv_to_profile(profile_root: Path, prompt_message: str) -> Path:
    print(prompt_message)
    selected_cv = _pick_file()
    if not selected_cv:
        raise RuntimeError("No CV file selected.")
    source_cv = Path(selected_cv).expanduser().resolve()
    if not source_cv.exists():
        raise FileNotFoundError(f"Selected CV file does not exist: {source_cv}")

    profile_root.mkdir(parents=True, exist_ok=True)
    target_cv = profile_root / f"user_cv{source_cv.suffix.lower()}"
    shutil.copy2(source_cv, target_cv)
    return target_cv


def _copy_optional_file_to_profile(profile_root: Path, *, prompt_message: str, target_stem: str) -> str:
    print(prompt_message)
    selected_path = _pick_file()
    if not selected_path:
        return ""

    source_path = Path(selected_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Selected file does not exist: {source_path}")

    documents_dir = profile_root / SUPPORTING_DOCUMENTS_DIRNAME
    documents_dir.mkdir(parents=True, exist_ok=True)
    destination = documents_dir / f"{target_stem}{source_path.suffix.lower()}"
    shutil.copy2(source_path, destination)
    return str(destination)


def ensure_workspace_source_cv_alias(workspace_root: Path, user_details: dict) -> str:
    cv_path_raw = str(user_details.get("cv_path", "")).strip()
    if not cv_path_raw:
        return ""
    try:
        source_cv = Path(cv_path_raw).expanduser().resolve()
    except Exception:
        return ""
    if not source_cv.exists() or not source_cv.is_file():
        return ""

    try:
        workspace_root.mkdir(parents=True, exist_ok=True)
        alias_path = workspace_root / f"{ACTIVE_SOURCE_CV_STEM}{source_cv.suffix.lower()}"
        for stale_alias in workspace_root.glob(f"{ACTIVE_SOURCE_CV_STEM}.*"):
            if stale_alias == alias_path or not stale_alias.is_file():
                continue
            stale_alias.unlink(missing_ok=True)
        shutil.copy2(source_cv, alias_path)
        return str(alias_path)
    except Exception:
        return ""


def _move_profile_path(old_path: Path, new_path: Path) -> bool:
    if not old_path.exists() or new_path.exists():
        return False
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_path), str(new_path))
    return True


def _migrate_profile_files(workspace_root: Path, profile_root: Path) -> None:
    profile_root.mkdir(parents=True, exist_ok=True)
    for filename in (USER_DETAILS_FILENAME, CV_MARKDOWN_FILENAME, CV_SUMMARY_CACHE_FILENAME):
        _move_profile_path(workspace_root / filename, profile_root / filename)

    for legacy_cv in workspace_root.glob("user_cv.*"):
        _move_profile_path(legacy_cv, profile_root / legacy_cv.name)

    user_details_path = profile_root / USER_DETAILS_FILENAME
    if not user_details_path.exists():
        return
    try:
        payload = json.loads(user_details_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return

    cv_path_raw = str(payload.get("cv_path", "")).strip()
    if not cv_path_raw:
        return
    try:
        cv_path = Path(cv_path_raw).expanduser().resolve()
    except Exception:
        return
    if cv_path.parent != workspace_root or not cv_path.name.startswith("user_cv"):
        return

    migrated_cv = profile_root / cv_path.name
    if not migrated_cv.exists():
        if not cv_path.exists():
            return
        _move_profile_path(cv_path, migrated_cv)
    if migrated_cv.exists():
        payload["cv_path"] = str(migrated_cv)
        user_details_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _request_user_profile(profile_root: Path) -> dict:
    print("No saved user profile found. Please enter your details.")
    first_name = input("Enter your first name: ").strip()
    last_name = input("Enter your last name: ").strip()
    email = input("Enter your email: ").strip()
    phone = input("Enter your phone number: ").strip()
    address = input("Enter your address: ").strip()
    city = input("Enter your city: ").strip()
    state = input("Enter your state: ").strip()
    post_code = input("Enter your post code: ").strip()
    country = input("Enter your country: ").strip()
    website_url = input("Enter your website URL (leave blank if none): ").strip()
    portfolio_url = input("Enter your portfolio URL (leave blank if none): ").strip()
    linkedin_url = input("Enter your LinkedIn URL: ").strip()
    github_url = input("Enter your GitHub URL: ").strip()

    target_cv = _copy_cv_to_profile(
        profile_root,
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
        "website_url": website_url,
        "portfolio_url": portfolio_url,
        "linkedin_url": linkedin_url,
        "github_url": github_url,
        "cv_path": str(target_cv),
        "supporting_documents": _prompt_supporting_documents(profile_root),
    }


def _save_user_details(profile_root: Path, user_details: dict) -> None:
    profile_root.mkdir(parents=True, exist_ok=True)
    user_details_path = profile_root / USER_DETAILS_FILENAME
    user_details_path.write_text(
        json.dumps(user_details, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _prompt_with_default(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    return input(f"{prompt}{suffix}: ").strip() or default


def _prompt_yes_no(prompt: str, *, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    answer = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def _prompt_choice(prompt: str, choices: list[str], *, default: str) -> str:
    normalized = {choice.lower(): choice for choice in choices}
    while True:
        answer = input(f"{prompt} [{'/'.join(choices)}] [{default}]: ").strip().lower()
        if not answer:
            return default
        if answer in normalized:
            return normalized[answer]
        print(f"Please choose one of: {', '.join(choices)}")


def _normalize_supporting_documents(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, str] = {}
    for raw_key, raw_path in value.items():
        key = str(raw_key or "").strip()
        path_text = str(raw_path or "").strip()
        if not key or not path_text:
            continue
        try:
            resolved_path = Path(path_text).expanduser().resolve()
        except Exception:
            continue
        if resolved_path.exists() and resolved_path.is_file():
            normalized[key] = str(resolved_path)
    return normalized


def _prompt_supporting_documents(profile_root: Path) -> dict[str, str]:
    print("You can optionally stage supporting documents for job applications.")
    if not _prompt_yes_no("Do you want to add optional supporting documents now?", default=False):
        return {}

    documents: dict[str, str] = {}
    for key, label in COMMON_SUPPORTING_DOCUMENTS:
        if not _prompt_yes_no(f"Do you want to add a {label}?", default=False):
            continue
        selected_path = _copy_optional_file_to_profile(
            profile_root,
            prompt_message=f"Please select your {label}. Cancel to skip it.",
            target_stem=key.replace("_", "-"),
        )
        if selected_path:
            documents[key] = selected_path
    return documents


def build_eeo_preferences(agent: Agent, user_details: dict) -> dict[str, str]:
    profile_root = agent.agent_workspace.profile_root
    existing = user_details.get(EEO_PREFERENCES_KEY)
    if isinstance(existing, dict) and existing:
        return {
            "gender": str(existing.get("gender", "Prefer not to say")).strip() or "Prefer not to say",
            "ethnicity": str(existing.get("ethnicity", "Prefer not to say")).strip() or "Prefer not to say",
            "veteran_status": (
                str(existing.get("veteran_status", "Prefer not to say")).strip() or "Prefer not to say"
            ),
            "disability_status": (
                str(existing.get("disability_status", "Prefer not to say")).strip() or "Prefer not to say"
            ),
        }

    print("Configure saved EEO responses for job applications.")
    if not _prompt_yes_no(
        "Do you want to configure EEO answers now? Leave this off to default to 'Prefer not to say'.",
        default=False,
    ):
        eeo_preferences = {
            "gender": "Prefer not to say",
            "ethnicity": "Prefer not to say",
            "veteran_status": "Prefer not to say",
            "disability_status": "Prefer not to say",
        }
    else:
        eeo_preferences = {
            "gender": _prompt_with_default("Preferred gender response", "Prefer not to say"),
            "ethnicity": _prompt_with_default("Preferred ethnicity response", "Prefer not to say"),
            "veteran_status": _prompt_with_default("Preferred veteran status response", "Prefer not to say"),
            "disability_status": _prompt_with_default(
                "Preferred disability status response",
                "Prefer not to say",
            ),
        }

    user_details[EEO_PREFERENCES_KEY] = eeo_preferences
    _save_user_details(profile_root, user_details)
    return eeo_preferences


def build_application_preferences(agent: Agent, user_details: dict) -> dict:
    profile_root = agent.agent_workspace.profile_root
    preferences = user_details.get(APPLICATION_PREFERENCES_KEY)
    if not isinstance(preferences, dict):
        preferences = {}

    changed = False

    if "desired_salary" not in preferences:
        preferences["desired_salary"] = input(
            "Enter your desired salary for application forms (leave blank to answer case by case): "
        ).strip()
        changed = True

    if "work_authorization" not in preferences:
        preferences["work_authorization"] = "Yes" if _prompt_yes_no(
            "Are you currently authorized to work without additional approval?",
            default=True,
        ) else "No"
        changed = True

    if "sponsorship_needed" not in preferences:
        preferences["sponsorship_needed"] = "Yes" if _prompt_yes_no(
            "Will you require visa sponsorship for these applications?",
            default=False,
        ) else "No"
        changed = True

    if "notice_period" not in preferences:
        preferences["notice_period"] = input(
            "Enter your notice period (leave blank if you want to answer later): "
        ).strip()
        changed = True

    if "earliest_start_date" not in preferences:
        preferences["earliest_start_date"] = input(
            "Enter your earliest start date (leave blank if you want to answer later): "
        ).strip()
        changed = True

    if "relocation_willingness" not in preferences:
        preferences["relocation_willingness"] = input(
            "Enter your relocation preference (for example: Yes, No, or Case by case): "
        ).strip()
        changed = True

    if "travel_willingness" not in preferences:
        preferences["travel_willingness"] = input(
            "Enter your travel preference (for example: Yes, No, or Up to 25%): "
        ).strip()
        changed = True

    if "free_text_mode" not in preferences:
        preferences["free_text_mode"] = _prompt_choice(
            "How should open-ended application questions be handled?",
            ["ask", "best_effort"],
            default="ask",
        )
        changed = True

    if "prefill_review_mode" not in preferences:
        preferences["prefill_review_mode"] = _prompt_choice(
            "How thoroughly should parsed resume prefills be reviewed?",
            ["off", "smart", "full"],
            default="smart",
        )
        changed = True

    for key, prompt in [
        ("marketing_opt_in_policy", "How should marketing opt-in fields be handled?"),
        ("sms_consent_policy", "How should SMS consent fields be handled?"),
        ("talent_pool_policy", "How should talent pool consent fields be handled?"),
    ]:
        if key not in preferences:
            preferences[key] = _prompt_choice(
                prompt,
                ["auto_deny", "ask_user", "auto_allow"],
                default="auto_deny",
            )
            changed = True

    eeo_preferences = build_eeo_preferences(agent, user_details)
    if preferences.get(EEO_PREFERENCES_KEY) != eeo_preferences:
        preferences[EEO_PREFERENCES_KEY] = eeo_preferences
        changed = True

    if changed:
        user_details[APPLICATION_PREFERENCES_KEY] = preferences
        _save_user_details(profile_root, user_details)

    return preferences


def build_user_data(agent: Agent) -> dict:
    workspace_root = agent.agent_workspace.workspace_root
    profile_root = agent.agent_workspace.profile_root
    _migrate_profile_files(workspace_root, profile_root)
    user_details_path = profile_root / USER_DETAILS_FILENAME
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
        user_details = _request_user_profile(profile_root)
        _save_user_details(profile_root, user_details)
        print(f"Saved user profile to {user_details_path}")
        return user_details

    cv_path_raw = str(user_details.get("cv_path", "")).strip()
    cv_path = Path(cv_path_raw).expanduser().resolve() if cv_path_raw else None
    changed = False
    if cv_path is None or not cv_path.exists():
        target_cv = _copy_cv_to_profile(
            profile_root,
            prompt_message="Saved profile CV is missing or unavailable. Please select your CV file again.",
        )
        user_details["cv_path"] = str(target_cv)
        changed = True

    if "website_url" not in user_details:
        user_details["website_url"] = input("Enter your website URL (leave blank if none): ").strip()
        changed = True

    if "portfolio_url" not in user_details:
        user_details["portfolio_url"] = input("Enter your portfolio URL (leave blank if none): ").strip()
        changed = True

    normalized_documents = _normalize_supporting_documents(user_details.get("supporting_documents", {}))
    if "supporting_documents" not in user_details:
        user_details["supporting_documents"] = _prompt_supporting_documents(profile_root)
        changed = True
    elif normalized_documents != user_details.get("supporting_documents"):
        user_details["supporting_documents"] = normalized_documents
        changed = True

    if changed:
        _save_user_details(profile_root, user_details)
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
    profile_root = agent.agent_workspace.profile_root
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

    profile_root.mkdir(parents=True, exist_ok=True)
    markdown_path = profile_root / CV_MARKDOWN_FILENAME
    markdown_path.write_text(cv_markdown.strip() + "\n", encoding="utf-8")

    markdown_hash = sha256(cv_markdown.encode("utf-8")).hexdigest()
    cache_path = profile_root / CV_SUMMARY_CACHE_FILENAME
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


def load_saved_user_details(profile_root: Path) -> dict:
    user_details_path = profile_root / USER_DETAILS_FILENAME
    if not user_details_path.exists():
        raise FileNotFoundError(
            f"User details not found at {user_details_path}. Run the job finder first to create the profile."
        )
    loaded = json.loads(user_details_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Saved profile at {user_details_path} is invalid.")
    return loaded


def load_cv_markdown_for_generation(profile_root: Path, user_details: dict) -> str:
    markdown_path = profile_root / CV_MARKDOWN_FILENAME
    if markdown_path.exists():
        markdown = markdown_path.read_text(encoding="utf-8", errors="replace").strip()
        if markdown:
            return markdown

    cv_path_raw = str(user_details.get("cv_path", "")).strip()
    if not cv_path_raw:
        raise RuntimeError("Saved profile does not contain a CV path.")
    cv_path = Path(cv_path_raw).expanduser().resolve()
    if not cv_path.exists():
        raise FileNotFoundError(f"Saved CV file does not exist: {cv_path}")

    markdown = _file_to_markdown(cv_path).strip()
    if not markdown:
        raise RuntimeError("CV markdown is empty after conversion.")
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    return markdown
