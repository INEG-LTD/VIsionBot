from __future__ import annotations

import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any, Optional

from pydantic import BaseModel, Field

from agent.tooling import (
    DialogPolicy,
    Effect,
    ProgressPolicy,
    ToolManifest,
    tool,
)
from agent.tooling.context import ToolContext
from agent.tooling.types import ToolOutcome, ToolOutput
from lib.ai import generate_model

CV_MAX_PAGES = 1
CV_WRITE_BEST_EFFORT = True
CV_PDF_TITLE_TEMPLATE = "CV - {job_name}"
CV_PDF_TIMEOUT_SECONDS = 180

_CV_EXTENSIONS = {".pdf", ".md", ".txt"}
_CV_NAME_TOKENS = ("cv", "resume")
_GENERATED_PREFIX = "tailored_cv_"
_MAX_CHANGE_SUMMARY_ITEMS = 10

_CV_SYSTEM_PROMPT = """
You tailor CVs for specific jobs while preserving factual integrity.

Hard constraints:
- Use the original CV content as the source of truth.
- Never invent new employers, job titles, date ranges, or education credentials.
- You may rewrite, reorder, and sharpen existing bullet points to align with the target role.
- You may add new bullet points only when they are reasonable elaborations of existing role/project evidence already present in the CV.
- Do not invent brand-new projects, products, or certifications that are not grounded in the source CV.
- Align the professional summary/intro to the target job.
- Preserve the source document's heading structure using markdown heading markers (#, ##, ###) instead of plain text labels.
- Keep the markdown in a clear, recruiter-friendly CV layout with obvious section breaks, scannable role headers, and concise bullet points.
- Keep output as a complete, clean markdown CV.

Output requirements:
- Return full tailored markdown in modified_cv.
- Return concise, user-facing change notes in changes.
- Ensure the markdown reads like a finished one-page CV document, not notes or free-form prose.
""".strip()

_PDF_WORKER_SCRIPT = r"""
import json
import sys
from pathlib import Path

from md_to_pdf import markdown_to_pdf_playwright

markdown_path = Path(sys.argv[1])
pdf_path = Path(sys.argv[2])
title = sys.argv[3]
max_pages = int(sys.argv[4])
write_best_effort = sys.argv[5] == "1"

try:
    out_path, fit = markdown_to_pdf_playwright(
        markdown_path,
        pdf_path,
        is_text=False,
        title=title,
        max_pages=max_pages,
        write_best_effort=write_best_effort,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output_pdf": str(out_path),
                "pages": int(getattr(fit, "pages", 0) or 0),
            }
        )
    )
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc)}))
    raise
""".strip()


class CvEditResult(BaseModel):
    modified_cv: str
    changes: list[str] = Field(default_factory=list)


class ModifyCvArgs(BaseModel):
    job_name: str = Field(min_length=1)
    job_description: str = Field(min_length=1)
    reasoning: str


def _log_warning(ctx: ToolContext, message: str) -> None:
    logger = getattr(ctx, "event_logger", None)
    if logger is None:
        return
    try:
        logger.system_warning(message)
    except Exception:
        pass


def _check_path_policy(
    ctx: ToolContext,
    path: Path,
    *,
    operation: str,
    warnings: list[str],
) -> Optional[str]:
    policy = getattr(ctx, "sandbox_policy", None)
    if policy is None or not hasattr(policy, "check_path"):
        return None

    try:
        decision = policy.check_path(path, operation=operation)
    except Exception as exc:
        return f"sandbox policy check failed for {path}: {exc}"

    if bool(getattr(decision, "allowed", False)):
        return None

    reason = str(getattr(decision, "reason", "") or "blocked by sandbox policy")
    message = f"{operation} blocked by sandbox for {path}: {reason}"
    if bool(getattr(policy, "enforce", True)):
        return message

    warnings.append(message)
    _log_warning(ctx, message)
    return None


def _slug(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower())
    normalized = normalized.strip("_")
    if not normalized:
        normalized = "job"
    return normalized[:40].strip("_") or "job"


def _find_cv_file(
    *,
    search_dirs: list[Path],
    ctx: ToolContext,
    warnings: list[str],
) -> tuple[Optional[Path], list[str], list[str]]:
    searched_dirs: list[str] = []
    blocked_errors: list[str] = []
    seen_dirs: set[Path] = set()
    candidates: list[tuple[float, Path]] = []

    for directory in search_dirs:
        try:
            resolved_dir = directory.expanduser().resolve()
        except Exception:
            continue
        if resolved_dir in seen_dirs:
            continue
        seen_dirs.add(resolved_dir)
        searched_dirs.append(str(resolved_dir))
        if not resolved_dir.exists() or not resolved_dir.is_dir():
            continue

        find_error = _check_path_policy(
            ctx,
            resolved_dir,
            operation="find",
            warnings=warnings,
        )
        if find_error:
            blocked_errors.append(find_error)
            continue

        try:
            iter_paths = resolved_dir.rglob("*")
        except Exception as exc:
            warnings.append(f"failed to walk directory {resolved_dir}: {exc}")
            continue

        for path in iter_paths:
            if not path.is_file():
                continue
            name_lower = path.name.lower()
            if name_lower.startswith(_GENERATED_PREFIX):
                continue
            if path.suffix.lower() not in _CV_EXTENSIONS:
                continue
            if not any(token in name_lower for token in _CV_NAME_TOKENS):
                continue

            read_error = _check_path_policy(
                ctx,
                path,
                operation="read",
                warnings=warnings,
            )
            if read_error:
                blocked_errors.append(read_error)
                continue

            try:
                mtime = float(path.stat().st_mtime)
            except Exception:
                mtime = 0.0
            candidates.append((mtime, path))

    if not candidates:
        return None, searched_dirs, blocked_errors

    # Auto-pick most recently modified candidate.
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return candidates[0][1], searched_dirs, blocked_errors


def _file_to_markdown(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="replace").strip()

    if suffix == ".pdf":
        markdown = _pdf_to_markdown(path)
        if not markdown:
            raise RuntimeError(f"no readable text extracted from PDF: {path}")
        return markdown

    raise RuntimeError(f"unsupported CV file extension: {suffix}")


def _normalize_pdf_line(raw: str) -> str:
    text = str(raw or "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize_for_size_lookup(line: str) -> list[str]:
    return [tok for tok in re.findall(r"[A-Za-z0-9\+#]+", str(line or "").upper()) if tok]


def _is_bullet_line(line: str) -> bool:
    text = str(line or "").lstrip()
    return text.startswith(("•", "●", "-", "*", "–", "—"))


def _headingish(line: str) -> bool:
    text = str(line or "").strip()
    if not text or _is_bullet_line(text):
        return False
    if len(text) > 80:
        return False
    if text.endswith("."):
        return False
    words = text.split()
    if len(words) > 8:
        return False

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / float(len(letters))
    if upper_ratio >= 0.75:
        return True

    title_case_words = 0
    alpha_words = 0
    for word in words:
        alpha = "".join(ch for ch in word if ch.isalpha())
        if not alpha:
            continue
        alpha_words += 1
        if alpha[:1].isupper() and alpha[1:].islower():
            title_case_words += 1
    return alpha_words > 0 and title_case_words >= max(1, int(alpha_words * 0.7))


def _looks_like_section_heading(line: str) -> bool:
    text = str(line or "").strip()
    if not text or _is_bullet_line(text):
        return False
    if len(text) > 60:
        return False
    if re.search(r"(https?://|www\.|@)", text, flags=re.IGNORECASE):
        return False

    words = text.split()
    if len(words) > 7:
        return False

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / float(len(letters))
    if upper_ratio >= 0.8:
        return True

    normalized = re.sub(r"[^A-Z ]+", "", text.upper()).strip()
    common_sections = {
        "PROFESSIONAL SUMMARY",
        "WORK EXPERIENCE",
        "EXPERIENCE",
        "SKILLS",
        "TECHNICAL SKILLS",
        "COLLABORATION WORKFLOW",
        "APP STORE EXPERTISE",
        "ADDITIONAL EXPERTISE",
        "NOTABLE PROJECTS",
        "PROJECTS",
        "EDUCATION",
    }
    return normalized in common_sections


def _looks_like_role_heading(line: str) -> bool:
    text = str(line or "").strip()
    if not text or _is_bullet_line(text):
        return False
    if len(text) > 110:
        return False
    if text.endswith("."):
        return False
    if re.search(r"(https?://|www\.|@)", text, flags=re.IGNORECASE):
        return False

    words = text.split()
    if len(words) > 16:
        return False

    if not re.search(r"\b(?:19|20)\d{2}\b", text):
        return False

    lower = text.lower()
    has_role_separators = any(token in text for token in ("/", "-", "|")) or " at " in lower
    return has_role_separators


def _pdf_token_size_map(path: Path) -> dict[str, float]:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError(f"pypdf is required to read PDF CV files: {exc}") from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise RuntimeError(f"failed to open PDF {path}: {exc}") from exc

    token_sizes: dict[str, float] = {}
    for page in reader.pages:
        def _visitor_text(text: Any, _cm: Any, _tm: Any, _font_dict: Any, font_size: Any) -> None:
            size = float(font_size or 0.0)
            if size <= 0.0:
                return
            cleaned = _normalize_pdf_line(str(text or ""))
            if not cleaned:
                return
            for token in _tokenize_for_size_lookup(cleaned):
                prev = token_sizes.get(token, 0.0)
                if size > prev:
                    token_sizes[token] = size

        try:
            page.extract_text(visitor_text=_visitor_text, extraction_mode="plain")
        except Exception:
            continue
    return token_sizes


def _pdf_layout_lines(path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError(f"pypdf is required to read PDF CV files: {exc}") from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise RuntimeError(f"failed to open PDF {path}: {exc}") from exc

    lines: list[str] = []
    for page in reader.pages:
        text = ""
        try:
            text = str(page.extract_text(extraction_mode="layout") or "")
        except Exception:
            text = str(page.extract_text() or "")
        page_lines = [_normalize_pdf_line(raw) for raw in text.splitlines()]
        page_lines = [line for line in page_lines if line]
        if page_lines and lines:
            lines.append("")
        lines.extend(page_lines)
    return lines


def _line_estimated_size(line: str, token_sizes: dict[str, float]) -> float:
    sizes: list[float] = []
    for token in _tokenize_for_size_lookup(line):
        size = token_sizes.get(token)
        if size:
            sizes.append(float(size))
    if not sizes:
        return 0.0
    return float(statistics.median(sizes))


def _pdf_lines_to_markdown(lines: list[str], token_sizes: dict[str, float]) -> str:
    sized_lines: list[tuple[str, float]] = []
    for line in lines:
        if not line:
            sized_lines.append(("", 0.0))
            continue
        sized_lines.append((line, _line_estimated_size(line, token_sizes)))

    non_zero = [size for line, size in sized_lines if line and size > 0]
    body_size = float(statistics.median(non_zero)) if non_zero else 12.0
    max_size = max(non_zero) if non_zero else body_size
    h1_threshold = body_size + max(3.0, (max_size - body_size) * 0.5)
    h2_threshold = body_size + 1.2

    output: list[str] = []
    h1_used = False

    for idx, (line, size) in enumerate(sized_lines):
        text = line.strip()
        if not text:
            if output and output[-1] != "":
                output.append("")
            continue

        prev_raw = sized_lines[idx - 1][0] if idx > 0 else ""
        next_raw = sized_lines[idx + 1][0] if idx + 1 < len(sized_lines) else ""
        prev_is_blank = not prev_raw.strip()
        next_is_blank = not next_raw.strip()
        next_is_bullet = _is_bullet_line(next_raw.strip()) if next_raw else False

        if _is_bullet_line(text):
            bullet_text = text.lstrip("•●-*–—").strip()
            output.append(f"- {bullet_text}" if bullet_text else "-")
            continue

        if (not h1_used) and size >= h1_threshold and len(text.split()) <= 12:
            output.append(f"# {text}")
            output.append("")
            h1_used = True
            continue

        if _looks_like_section_heading(text) and (
            size >= h2_threshold or (prev_is_blank and next_is_blank)
        ):
            output.append(f"## {text}")
            output.append("")
            continue

        if _looks_like_role_heading(text) and (next_is_bullet or size >= body_size):
            output.append(f"### {text}")
            output.append("")
            continue

        if _headingish(text) and (
            size >= h2_threshold or (prev_is_blank and (next_is_blank or next_is_bullet))
        ):
            output.append(f"## {text}")
            output.append("")
            continue

        output.append(text)

    while output and output[-1] == "":
        output.pop()
    return "\n".join(output).strip()


def _pdf_to_markdown(path: Path) -> str:
    token_sizes = _pdf_token_size_map(path)
    lines = _pdf_layout_lines(path)
    markdown = _pdf_lines_to_markdown(lines, token_sizes)
    if markdown:
        return markdown

    # Last-resort fallback.
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError(f"pypdf is required to read PDF CV files: {exc}") from exc
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise RuntimeError(f"failed to open PDF {path}: {exc}") from exc

    pages: list[str] = []
    for page in reader.pages:
        try:
            text = str(page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            pages.append(text)
    return "\n\n".join(pages).strip()


def _build_prompt(*, args: ModifyCvArgs, cv_markdown: str) -> str:
    return (
        "Tailor the CV to the target role using the rules in the system prompt.\n\n"
        f"Target job name:\n{args.job_name.strip()}\n\n"
        f"Target job description:\n{args.job_description.strip()}\n\n"
        f"Planner reasoning/context:\n{args.reasoning.strip()}\n\n"
        "Original CV markdown (source of truth):\n"
        "```markdown\n"
        f"{cv_markdown}\n"
        "```"
    )


def _coerce_edit_result(value: Any) -> CvEditResult:
    if isinstance(value, CvEditResult):
        return value
    if isinstance(value, dict):
        return CvEditResult.model_validate(value)
    if isinstance(value, str):
        try:
            return CvEditResult.model_validate_json(value)
        except Exception as exc:
            raise RuntimeError(f"model output was not valid CvEditResult JSON: {exc}") from exc
    raise RuntimeError(f"unexpected model output type: {type(value).__name__}")


def _fail(message: str, *, data: Optional[dict[str, Any]] = None) -> ToolOutcome:
    return ToolOutcome(
        output=ToolOutput(
            success=False,
            summary=f"modify_cv FAILED: {message}",
            error=message,
            data=data if isinstance(data, dict) else None,
        )
    )


def _render_pdf_via_subprocess(
    *,
    markdown_path: Path,
    pdf_path: Path,
    title: str,
) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-c",
        _PDF_WORKER_SCRIPT,
        str(markdown_path),
        str(pdf_path),
        str(title or "Document"),
        str(int(CV_MAX_PAGES)),
        "1" if CV_WRITE_BEST_EFFORT else "0",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=CV_PDF_TIMEOUT_SECONDS,
    )

    payload = None
    stdout_lines = [line.strip() for line in str(proc.stdout or "").splitlines() if line.strip()]
    for line in reversed(stdout_lines):
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                break
            except Exception:
                continue

    if not isinstance(payload, dict):
        stderr_preview = str(proc.stderr or "").strip()
        raise RuntimeError(
            f"PDF worker did not return parseable JSON. exit={proc.returncode}. stderr={stderr_preview[:500]}"
        )

    if not bool(payload.get("ok", False)) or proc.returncode != 0:
        error_text = str(payload.get("error", "") or "").strip()
        stderr_preview = str(proc.stderr or "").strip()
        if not error_text:
            error_text = stderr_preview or "unknown PDF worker failure"
        raise RuntimeError(error_text)

    try:
        return int(payload.get("pages", 0) or 0)
    except Exception:
        return 0


@tool(
    manifest=ToolManifest(
        name="modify_cv",
        description=(
            "Find a CV in the agent workspace, tailor it to a target job without "
            "fabricating history, and save markdown/PDF outputs."
        ),
        effects=frozenset({Effect.READ_HOST, Effect.WRITE_HOST}),
        dialog_policy=DialogPolicy.ALLOW_WHEN_DIALOG,
        progress_policy=ProgressPolicy.USER_FACING,
        tags=frozenset({"cv", "host", "data"}),
    ),
    args_model=ModifyCvArgs,
)
def modify_cv(ctx: ToolContext, args: ModifyCvArgs) -> ToolOutcome:
    runtime_state = getattr(ctx, "runtime_state", None)
    agent = getattr(runtime_state, "agent", None)
    workspace = getattr(agent, "agent_workspace", None)
    if workspace is None:
        return _fail("agent workspace is unavailable on tool runtime state")

    warnings: list[str] = []
    search_dirs = [
        Path(workspace.workspace_root),
        Path(workspace.written_data_dir),
        Path(workspace.browser_downloads_dir),
    ]
    cv_path, searched_dirs, blocked_errors = _find_cv_file(
        search_dirs=search_dirs,
        ctx=ctx,
        warnings=warnings,
    )
    if cv_path is None:
        details = {
            "searched_dirs": searched_dirs,
            "blocked_errors": blocked_errors,
        }
        return _fail(
            "no matching CV file found. Looked for names containing 'cv' or "
            "'resume' with extensions .pdf/.md/.txt.",
            data=details,
        )

    read_error = _check_path_policy(ctx, cv_path, operation="read", warnings=warnings)
    if read_error:
        return _fail(read_error, data={"source_cv": str(cv_path)})

    try:
        source_markdown = _file_to_markdown(cv_path)
    except Exception as exc:
        return _fail(f"failed to convert CV to markdown: {exc}", data={"source_cv": str(cv_path)})
    if not source_markdown.strip():
        return _fail("CV content is empty after conversion", data={"source_cv": str(cv_path)})

    prompt = _build_prompt(args=args, cv_markdown=source_markdown)
    try:
        raw_result = generate_model(
            prompt=prompt,
            model_object_type=CvEditResult,
            system_prompt=_CV_SYSTEM_PROMPT,
        )
        edit_result = _coerce_edit_result(raw_result)
    except Exception as exc:
        return _fail(f"CV tailoring model call failed: {exc}", data={"source_cv": str(cv_path)})

    modified_cv = str(edit_result.modified_cv or "").strip()
    if not modified_cv:
        return _fail("model returned empty modified_cv", data={"source_cv": str(cv_path)})
    changes = [
        str(item).strip()
        for item in (edit_result.changes or [])
        if str(item).strip()
    ]

    try:
        output_dir = Path(workspace.written_data_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return _fail(f"failed to prepare output directory: {exc}")

    slug = _slug(args.job_name)
    markdown_path = output_dir / f"{_GENERATED_PREFIX}{slug}.md"
    pdf_path = output_dir / f"{_GENERATED_PREFIX}{slug}.pdf"

    write_md_error = _check_path_policy(ctx, markdown_path, operation="write", warnings=warnings)
    if write_md_error:
        return _fail(write_md_error, data={"output_markdown": str(markdown_path)})
    write_pdf_error = _check_path_policy(ctx, pdf_path, operation="write", warnings=warnings)
    if write_pdf_error:
        return _fail(write_pdf_error, data={"output_pdf": str(pdf_path)})

    try:
        markdown_path.write_text(modified_cv + "\n", encoding="utf-8")
    except Exception as exc:
        return _fail(
            f"failed to write tailored markdown file: {exc}",
            data={"output_markdown": str(markdown_path)},
        )

    try:
        pdf_pages = _render_pdf_via_subprocess(
            markdown_path=markdown_path,
            pdf_path=pdf_path,
            title=CV_PDF_TITLE_TEMPLATE.format(job_name=args.job_name.strip()),
        )
    except Exception as exc:
        return _fail(
            f"failed to generate tailored PDF: {exc}",
            data={
                "output_markdown": str(markdown_path),
                "output_pdf": str(pdf_path),
            },
        )

    if changes:
        change_lines = [f"- {line}" for line in changes[:_MAX_CHANGE_SUMMARY_ITEMS]]
        changes_text = "\n".join(change_lines)
    else:
        changes_text = "- (model did not provide explicit change notes)"

    warning_text = ""
    if warnings:
        warning_text = "\nWarnings:\n" + "\n".join(f"- {item}" for item in warnings[:5])

    summary = (
        "Tailored CV created successfully.\n"
        f"Source CV: {cv_path}\n"
        f"Tailored markdown: {markdown_path}\n"
        f"Tailored PDF: {pdf_path}\n"
        f"PDF page count: {pdf_pages}\n"
        "Key changes:\n"
        f"{changes_text}"
        f"{warning_text}\n"
        "Now use ask_user to present these changes and ask whether the user is satisfied or wants adjustments."
    )

    return ToolOutcome(
        output=ToolOutput(
            success=True,
            summary=summary,
            data={
                "source_cv": str(cv_path),
                "job_name": args.job_name.strip(),
                "output_markdown": str(markdown_path),
                "output_pdf": str(pdf_path),
                "changes": changes,
                "warnings": warnings,
                "pdf_pages": int(pdf_pages),
            },
        )
    )
