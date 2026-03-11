from __future__ import annotations

import json
from pathlib import Path
import re

from pydantic import BaseModel, Field

from job_application_app.modify_cv_tool import (
    CV_PDF_TITLE_TEMPLATE,
    CvEditResult,
    _CV_SYSTEM_PROMPT,
    _coerce_edit_result,
    _render_pdf_via_subprocess,
)
from job_application_app.profile_store import (
    CV_MARKDOWN_FILENAME,
    load_cv_markdown_for_generation,
    load_saved_user_details,
    resolve_app_paths,
)
from lib.ai import generate_model

_MAX_COVER_LETTER_WORDS = 250
_MIN_COVER_LETTER_WORDS = 150


class CoverLetterResult(BaseModel):
    cover_letter_markdown: str = Field(min_length=1)
    highlights: list[str] = Field(default_factory=list)


_COVER_LETTER_SYSTEM_PROMPT = """
You write tailored, interview-winning cover letters.

Rules:
- Use only information grounded in the provided candidate profile, CV markdown, and job details.
- Never invent employers, products, achievements, technologies, or metrics that are not supported by the CV or user details.
- Keep the letter concise and human: aim for 150-250 words.
- Use the underlying shape of hook, proof, alignment, close, but do not make the structure feel templated or visibly formulaic.
- Make the opening immediately relevant to the role.
- Show evidence instead of vague claims.
- Explicitly connect the candidate's experience to the employer's needs.
- Explain why this role is appealing in a specific way.
- Use achievement-focused verbs such as built, delivered, improved, implemented, and designed.
- Avoid generic phrases like 'hardworking', 'passionate', 'responsible for', or 'assisted with'.
- If it fits naturally, add a brief confidence signal such as being happy to discuss architecture or recent Swift work.
- Write with natural flow and transitions so the letter reads like a thoughtful human making a case for an interview.
- Do not mechanically restate the CV or mirror the job description line-by-line.
- Avoid formulaic sentences such as 'Your focus on X aligns with my experience in Y' unless rewritten into natural prose.
- Use confident, plain English. Avoid robotic or generic phrasing.
- Avoid ellipses and filler.
- Return the final letter body ready to save, using `<br>` for line breaks.
""".strip()


def _safe_job_stem(job_name: str) -> str:
    text = re.sub(r"[^\w\s-]+", "", str(job_name or "").strip())
    text = re.sub(r"\s+", "-", text).strip("-")
    return text or "job"


def _job_name(job: dict) -> str:
    title = str(job.get("job_title", "") or "").strip()
    company = str(job.get("company_name", "") or "").strip()
    return title or company or "job"


def _job_context(job: dict) -> str:
    ordered_fields = [
        ("job_title", job.get("job_title")),
        ("company_name", job.get("company_name")),
        ("location", job.get("location")),
        ("posted_date", job.get("posted_date")),
        ("salary", job.get("salary")),
        ("employment_type", job.get("employment_type")),
        ("job_summary", job.get("job_summary")),
        ("apply_links", job.get("apply_links")),
        ("apply_labels", job.get("apply_labels")),
        ("source", job.get("source")),
    ]
    filtered = {
        key: value
        for key, value in ordered_fields
        if value not in (None, "", [], {})
    }
    return json.dumps(filtered, indent=2, sort_keys=True)


def _build_cv_generation_prompt(
    *,
    job: dict,
    source_cv_markdown: str,
    revision_request: str | None,
    existing_markdown: str | None,
) -> str:
    prompt = (
        "Tailor the CV to the target job using the source CV as the source of truth.\n\n"
        f"Target job details:\n```json\n{_job_context(job)}\n```\n\n"
    )
    if existing_markdown:
        prompt += (
            "Current tailored CV draft to revise:\n"
            "```markdown\n"
            f"{existing_markdown.strip()}\n"
            "```\n\n"
        )
    if revision_request:
        prompt += f"Requested revisions:\n{revision_request.strip()}\n\n"
    prompt += (
        "Original CV markdown (source of truth):\n"
        "```markdown\n"
        f"{source_cv_markdown}\n"
        "```"
    )
    return prompt


def _build_cover_letter_prompt(
    *,
    user_details: dict,
    cv_markdown: str,
    job: dict,
    revision_request: str | None,
    existing_markdown: str | None,
) -> str:
    candidate_profile = {
        "name": " ".join(
            part for part in [user_details.get("first_name", ""), user_details.get("last_name", "")] if str(part).strip()
        ).strip(),
        "email": str(user_details.get("email", "") or "").strip() or None,
        "phone": str(user_details.get("phone", "") or "").strip() or None,
        "city": str(user_details.get("city", "") or "").strip() or None,
        "state": str(user_details.get("state", "") or "").strip() or None,
        "country": str(user_details.get("country", "") or "").strip() or None,
        "linkedin_url": str(user_details.get("linkedin_url", "") or "").strip() or None,
        "github_url": str(user_details.get("github_url", "") or "").strip() or None,
    }
    prompt = (
        "Write a tailored cover letter for this candidate and job.\n\n"
        f"Candidate details:\n```json\n{json.dumps(candidate_profile, indent=2, sort_keys=True)}\n```\n\n"
        f"Target job details:\n```json\n{_job_context(job)}\n```\n\n"
        "Use the CV markdown below as the factual source of truth for achievements, projects, products, and technologies.\n"
        "The cover letter should feel like an A* cover letter with a strong opening, evidence, alignment, and a confident close, but it should read like a flowing human letter rather than a visible template.\n"
        f"Keep it between {_MIN_COVER_LETTER_WORDS} and {_MAX_COVER_LETTER_WORDS} words.\n"
        "If the job does not specify a hiring manager, start with 'Dear Hiring Manager,'.\n"
        "Write with natural transitions and persuasive flow.\n"
        "Do not make it read like an AI summary of the CV.\n"
        "Use `<br>` for line breaks and `<br><br>` between paragraphs.\n\n"
    )
    if existing_markdown:
        prompt += (
            "Current cover letter draft to revise:\n"
            "```markdown\n"
            f"{existing_markdown.strip()}\n"
            "```\n\n"
        )
    if revision_request:
        prompt += f"Requested revisions:\n{revision_request.strip()}\n\n"
    prompt += (
        "Candidate CV markdown:\n"
        "```markdown\n"
        f"{cv_markdown}\n"
        "```"
    )
    return prompt


def _write_text_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content or "").strip() + "\n", encoding="utf-8")
    return path


def _format_with_br_line_breaks(text: str) -> str:
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    formatted: list[str] = []
    blank_run = 0

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            blank_run += 1
            continue

        if blank_run > 0:
            formatted.append("<br><br>\n")
        suffix = "" if line.endswith("<br>") else "<br>"
        formatted.append(f"{line}{suffix}\n")
        blank_run = 0

    return "".join(formatted).strip()


def _coerce_cover_letter_result(value: object) -> CoverLetterResult:
    if isinstance(value, CoverLetterResult):
        return value
    if isinstance(value, dict):
        return CoverLetterResult.model_validate(value)
    if isinstance(value, str):
        return CoverLetterResult.model_validate_json(value)
    raise RuntimeError(f"unexpected model output type: {type(value).__name__}")


def generate_cv(
    *,
    job: dict,
    project_root: Path,
    agent_id: str,
    revision_request: str | None = None,
    existing_markdown_path: str | None = None,
) -> dict:
    app_paths = resolve_app_paths(project_root=project_root, agent_id=agent_id)
    user_details = load_saved_user_details(app_paths.workspace_root)
    source_cv_markdown = load_cv_markdown_for_generation(app_paths.workspace_root, user_details)

    existing_markdown = None
    job_stem = _safe_job_stem(_job_name(job))
    default_markdown_path = app_paths.written_data_dir / f"{job_stem}-cv.md"
    if existing_markdown_path:
        candidate_path = Path(existing_markdown_path).expanduser().resolve()
        if candidate_path.exists():
            existing_markdown = candidate_path.read_text(encoding="utf-8", errors="replace").strip()
    elif revision_request and default_markdown_path.exists():
        existing_markdown = default_markdown_path.read_text(encoding="utf-8", errors="replace").strip()

    prompt = _build_cv_generation_prompt(
        job=job,
        source_cv_markdown=source_cv_markdown,
        revision_request=revision_request,
        existing_markdown=existing_markdown,
    )
    raw_result = generate_model(
        prompt=prompt,
        model_object_type=CvEditResult,
        system_prompt=_CV_SYSTEM_PROMPT,
    )
    edit_result = _coerce_edit_result(raw_result)
    modified_cv = str(edit_result.modified_cv or "").strip()
    if not modified_cv:
        raise RuntimeError("CV generation returned empty markdown.")
    changes = [str(item).strip() for item in edit_result.changes if str(item).strip()]

    markdown_path = _write_text_file(default_markdown_path, modified_cv)
    pdf_path = app_paths.written_data_dir / f"{job_stem}-cv.pdf"
    pdf_pages = _render_pdf_via_subprocess(
        markdown_path=markdown_path,
        pdf_path=pdf_path,
        title=CV_PDF_TITLE_TEMPLATE.format(job_name=_job_name(job)),
    )
    return {
        "job_name": _job_name(job),
        "output_markdown": str(markdown_path),
        "output_pdf": str(pdf_path),
        "changes": changes,
        "pdf_pages": int(pdf_pages),
        "modified_cv": modified_cv,
        "source_of_truth_markdown": str(app_paths.workspace_root / CV_MARKDOWN_FILENAME),
    }


def generate_cover_letter(
    *,
    job: dict,
    project_root: Path,
    agent_id: str,
    revision_request: str | None = None,
    existing_markdown_path: str | None = None,
) -> dict:
    app_paths = resolve_app_paths(project_root=project_root, agent_id=agent_id)
    user_details = load_saved_user_details(app_paths.workspace_root)
    cv_markdown = load_cv_markdown_for_generation(app_paths.workspace_root, user_details)

    existing_markdown = None
    job_stem = _safe_job_stem(_job_name(job))
    default_markdown_path = app_paths.written_data_dir / f"{job_stem}-cover-letter.md"
    if existing_markdown_path:
        candidate_path = Path(existing_markdown_path).expanduser().resolve()
        if candidate_path.exists():
            existing_markdown = candidate_path.read_text(encoding="utf-8", errors="replace").strip()
    elif revision_request and default_markdown_path.exists():
        existing_markdown = default_markdown_path.read_text(encoding="utf-8", errors="replace").strip()

    prompt = _build_cover_letter_prompt(
        user_details=user_details,
        cv_markdown=cv_markdown,
        job=job,
        revision_request=revision_request,
        existing_markdown=existing_markdown,
    )
    raw_result = generate_model(
        prompt=prompt,
        model_object_type=CoverLetterResult,
        system_prompt=_COVER_LETTER_SYSTEM_PROMPT,
    )
    result = _coerce_cover_letter_result(raw_result)
    cover_letter_markdown = str(result.cover_letter_markdown or "").strip()
    if not cover_letter_markdown:
        raise RuntimeError("Cover letter generation returned empty markdown.")
    cover_letter_markdown = _format_with_br_line_breaks(cover_letter_markdown)

    markdown_path = _write_text_file(default_markdown_path, cover_letter_markdown)
    pdf_path = app_paths.written_data_dir / f"{job_stem}-cover-letter.pdf"
    pdf_pages = _render_pdf_via_subprocess(
        markdown_path=markdown_path,
        pdf_path=pdf_path,
        title=f"Cover Letter - {_job_name(job)}",
    )
    return {
        "job_name": _job_name(job),
        "output_markdown": str(markdown_path),
        "output_pdf": str(pdf_path),
        "pdf_pages": int(pdf_pages),
        "cover_letter_markdown": cover_letter_markdown,
        "highlights": [str(item).strip() for item in result.highlights if str(item).strip()],
    }
