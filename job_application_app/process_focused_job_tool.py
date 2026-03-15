from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

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

_MAX_VISIBLE_TEXT_CHARS = 18000
_RETRY_WAIT_MS = 750
_GOOGLE_SEARCH_TITLE_SUFFIX = " - Google Search"
_SEARCH_CONTEXT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "for",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)

_PROCESS_FOCUSED_JOB_SYSTEM_PROMPT = """
You extract and evaluate the single currently focused Google Jobs detail panel.

Rules:
- Use only information visible in the current screenshot and provided page text.
- Never invent missing values.
    - Required fields are: job_title and location.
- If a required field is not clearly visible, return an empty string for that field.
- Optional fields may be omitted or returned as empty values when unavailable.
- job_summary should be a concise, complete summary of the most important visible parts of the role.
- job_summary should prioritize role purpose, key responsibilities, important skills/technologies, work arrangement, and compensation details if visible.
- Do not quote the full job description.
- Do not end job_summary with an ellipsis.
- matches_profile should be true only when the focused job clearly fits the provided job profile.
    - For profile fit, prioritize job title and location. Use posted date as optional supporting context when relevant.
""".strip()


class ProcessFocusedJobArgs(BaseModel):
    job_profile: str = Field(min_length=1)
    file_name: str = "google-jobs-list.jsonl"
    search_query: Optional[str] = None
    reasoning: str


class ApplyLink(BaseModel):
    label: str
    url: str


class FocusedJobExtraction(BaseModel):
    job_title: str = ""
    location: str = ""
    posted_date: str = ""
    company_name: Optional[str] = None
    salary: Optional[str] = None
    employment_type: Optional[str] = None
    job_summary: Optional[str] = None
    matches_profile: bool = False
    match_reason: str = ""


def _log_info(ctx: ToolContext, message: str, **details: Any) -> None:
    logger = getattr(ctx, "event_logger", None)
    if logger is None:
        return
    try:
        logger.system_info(message, **details)
    except Exception:
        pass


def _log_warning(ctx: ToolContext, message: str, **details: Any) -> None:
    logger = getattr(ctx, "event_logger", None)
    if logger is None:
        return
    try:
        logger.system_warning(message, **details)
    except Exception:
        pass


def _check_path_policy(
    ctx: ToolContext,
    path: Path,
    *,
    operation: str,
) -> Optional[str]:
    policy = getattr(ctx, "sandbox_policy", None)
    if policy is None or not hasattr(policy, "check_path"):
        return None
    try:
        decision = policy.check_path(path, operation=operation)
    except Exception as exc:
        return f"path policy error for {path}: {exc}"
    if bool(getattr(decision, "allowed", True)):
        return None
    reason = str(getattr(decision, "reason", "") or "blocked")
    if bool(getattr(policy, "enforce", True)):
        return f"{operation} blocked by sandbox: {reason}"
    _log_warning(ctx, f"{operation} blocked by sandbox (observe): {reason}", path=str(path))
    return None


def _coerce_extraction(value: Any) -> FocusedJobExtraction:
    if isinstance(value, FocusedJobExtraction):
        return value
    if isinstance(value, dict):
        return FocusedJobExtraction.model_validate(value)
    if isinstance(value, str):
        try:
            return FocusedJobExtraction.model_validate_json(value)
        except Exception as exc:
            raise RuntimeError(
                f"model output was not valid FocusedJobExtraction JSON: {exc}"
            ) from exc
    raise RuntimeError(f"unexpected model output type: {type(value).__name__}")


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text


def _get_page_url(page: Any) -> str:
    try:
        return str(page.url or "")
    except Exception:
        return ""


def _get_page_title(page: Any) -> str:
    try:
        return str(page.title() or "")
    except Exception:
        return ""


def _extract_google_query_from_url(current_url: str) -> str:
    if not current_url:
        return ""
    try:
        parsed = urlparse(current_url)
        query_values = parse_qs(parsed.query).get("q", ())
    except Exception:
        return ""
    if not query_values:
        return ""
    return str(query_values[0] or "").strip()


def _extract_google_query_from_title(page_title: str) -> str:
    title = str(page_title or "").strip()
    if not title:
        return ""
    if title.casefold().endswith(_GOOGLE_SEARCH_TITLE_SUFFIX.casefold()):
        return title[: -len(_GOOGLE_SEARCH_TITLE_SUFFIX)].strip()
    return ""


def _read_google_search_box_value(page: Any) -> str:
    script = """
    () => {
        const node = document.querySelector('input[name="q"], textarea[name="q"]');
        if (!node) return "";
        return String(node.value || node.textContent || "").trim();
    }
    """
    try:
        value = page.evaluate(script)
    except Exception:
        return ""
    return str(value or "").strip()


def _tokenize_search_context(value: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", str(value or "").casefold()))
    return {token for token in tokens if token and token not in _SEARCH_CONTEXT_STOPWORDS}


def _queries_semantically_match(expected_query: str, observed_query: str) -> bool:
    expected_normalized = _normalize_text(expected_query)
    observed_normalized = _normalize_text(observed_query)
    if not expected_normalized or not observed_normalized:
        return False
    if expected_normalized == observed_normalized:
        return True
    if expected_normalized in observed_normalized or observed_normalized in expected_normalized:
        return True

    expected_tokens = _tokenize_search_context(expected_query)
    observed_tokens = _tokenize_search_context(observed_query)
    if not expected_tokens or not observed_tokens:
        return False

    overlap_ratio = len(expected_tokens.intersection(observed_tokens)) / float(len(expected_tokens))
    return overlap_ratio >= 0.5


def _detect_search_context_mismatch(
    ctx: ToolContext,
    args: ProcessFocusedJobArgs,
) -> Optional[dict[str, Any]]:
    expected_query = str(args.search_query or "").strip()
    if not expected_query:
        return None

    page = getattr(ctx, "page", None)
    if page is None:
        return None

    current_url = _get_page_url(page)
    page_title = _get_page_title(page)
    observed_candidates = [
        ("url_q", _extract_google_query_from_url(current_url)),
        ("search_box", _read_google_search_box_value(page)),
        ("page_title", _extract_google_query_from_title(page_title)),
    ]
    non_empty_candidates = [
        (source, str(value or "").strip())
        for source, value in observed_candidates
        if str(value or "").strip()
    ]
    if not non_empty_candidates:
        return None

    for _, observed_query in non_empty_candidates:
        if _queries_semantically_match(expected_query, observed_query):
            return None

    observed_source, observed_query = non_empty_candidates[0]
    return {
        "processable": False,
        "saved": False,
        "duplicate": False,
        "matches_profile": None,
        "reason": "search_context_mismatch",
        "recoverable": True,
        "search_context_valid": False,
        "missing_fields": [],
        "dedupe_key": "",
        "expected_search_query": expected_query,
        "observed_search_query": observed_query,
        "observed_query_source": observed_source,
        "current_url": current_url,
        "page_title": page_title,
    }


def _build_dedupe_key(*, job_title: str, location: str, company_name: Optional[str]) -> str:
    title = _normalize_text(job_title)
    loc = _normalize_text(location)
    company = _normalize_text(company_name)
    if company:
        return "|".join(part for part in (title, company, loc) if part)
    return "|".join(part for part in (title, loc) if part)


def _extract_google_docid_from_url(current_url: str) -> str:
    decoded_url = unquote(str(current_url or "")).strip()
    if not decoded_url:
        return ""
    match = re.search(r"(?:^|[/?#&])(?:docid|htidocid)=([^&#/]+)", decoded_url)
    if not match:
        return ""
    return str(match.group(1) or "").strip()


def _build_job_record(
    *,
    extracted: FocusedJobExtraction,
    apply_links: list[dict[str, str]],
    apply_directly_link: Optional[dict[str, str]],
    search_query: Optional[str],
    dedupe_key: str,
    source_job_url: str,
) -> dict[str, Any]:
    return {
        "job_title": extracted.job_title,
        "location": extracted.location,
        "posted_date": extracted.posted_date,
        "company_name": extracted.company_name,
        "salary": extracted.salary,
        "employment_type": extracted.employment_type,
        "job_summary": extracted.job_summary,
        "apply_links": apply_links,
        "apply_directly_link": apply_directly_link,
        "search_query": str(search_query or "").strip() or None,
        "source": "Google Jobs",
        "source_docid": _extract_google_docid_from_url(source_job_url),
        "source_job_url": str(source_job_url or "").strip(),
        "dedupe_key": dedupe_key,
        "match_reason": extracted.match_reason,
    }


def _build_job_saved_event_data(
    *,
    file_name: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "job_title": str(record.get("job_title", "") or "").strip(),
        "company_name": record.get("company_name"),
        "location": str(record.get("location", "") or "").strip(),
        "file_name": str(file_name or "").strip(),
        "dedupe_key": str(record.get("dedupe_key", "") or "").strip(),
        "source": str(record.get("source", "") or "").strip(),
        "source_docid": str(record.get("source_docid", "") or "").strip(),
        "source_job_url": str(record.get("source_job_url", "") or "").strip(),
        "job": dict(record),
    }


def _missing_required_fields(extracted: FocusedJobExtraction) -> list[str]:
    missing: list[str] = []
    if not str(extracted.job_title or "").strip():
        missing.append("job_title")
    if not str(extracted.location or "").strip():
        missing.append("location")
    return missing


def _clean_job_summary(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?:\.\.\.|…)\s*$", "", text).strip()
    return text or None


def _trim_visible_text(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if len(cleaned) <= _MAX_VISIBLE_TEXT_CHARS:
        return cleaned
    return cleaned[:_MAX_VISIBLE_TEXT_CHARS] + "\n... (truncated)"


def _collect_page_text(ctx: ToolContext) -> str:
    page = getattr(ctx, "page", None)
    if page is None:
        return ""
    try:
        visible_text = page.evaluate("document.body.innerText") or ""
    except Exception:
        return ""
    return _trim_visible_text(str(visible_text or ""))


def _extract_focused_job(ctx: ToolContext, args: ProcessFocusedJobArgs) -> FocusedJobExtraction:
    page = getattr(ctx, "page", None)
    if page is None:
        raise RuntimeError("page is unavailable")

    screenshot = page.screenshot(full_page=False)
    visible_text = _collect_page_text(ctx)
    page_url = _get_page_url(page)
    page_title = _get_page_title(page)

    prompt = (
        "Extract the currently focused Google job from the open Google Jobs page.\n\n"
        f"Search query:\n{str(args.search_query or '').strip() or '(not provided)'}\n\n"
        f"Job profile to match:\n{args.job_profile.strip()}\n\n"
        f"Planner reasoning/context:\n{args.reasoning.strip()}\n\n"
        "Return these fields:\n"
        "- job_title\n"
        "- location\n"
        "- posted_date\n"
        "- company_name\n"
        "- salary\n"
        "- employment_type\n"
        "- job_summary\n"
        "- matches_profile\n"
        "- match_reason\n\n"
        "Required fields for processability are job_title and location.\n"
        "If any required field is not visible, return an empty string for it.\n"
        "job_summary must be a concise complete summary of the most important visible parts of the role.\n"
        "Do not return a raw copied block of description text, and do not end job_summary with an ellipsis.\n"
        "Focus on the currently open job details, not the general search page.\n\n"
        f"Current page URL: {page_url}\n"
        f"Current page title: {page_title}\n\n"
        "Visible page text:\n"
        f"{visible_text or '(no visible text found)'}\n"
    )

    raw_result = generate_model(
        prompt=prompt,
        model_object_type=FocusedJobExtraction,
        system_prompt=_PROCESS_FOCUSED_JOB_SYSTEM_PROMPT,
        image=screenshot,
        image_detail="high",
    )
    return _coerce_extraction(raw_result)


def _collect_apply_links(ctx: ToolContext) -> list[ApplyLink]:
    page = getattr(ctx, "page", None)
    if page is None:
        return []

    script = r"""
    () => {
        const marker = "__codex_extract_apply_links__";
        void marker;

        const normalize = (value) => String(value || "").replace(/\s+/g, " ").trim();
        const looksInteresting = (label) => {
            const text = normalize(label).toLowerCase();
            if (!text) return false;
            return text.startsWith("apply");
        };
        const isVisible = (element) => {
            if (!element) return false;
            const style = window.getComputedStyle(element);
            if (!style || style.visibility === "hidden" || style.display === "none") return false;
            const rect = element.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.top < window.innerHeight;
        };
        const rightPaneThreshold = window.innerWidth * 0.45;
        const nodes = Array.from(document.querySelectorAll("a[href], button, [role='button']"));
        const seen = new Set();
        const links = [];

        for (const node of nodes) {
            if (!isVisible(node)) continue;
            const rect = node.getBoundingClientRect();
            const centerX = rect.left + (rect.width / 2);
            if (centerX < rightPaneThreshold) continue;

            const anchor =
                (node.matches && node.matches("a[href]") ? node : null) ||
                (node.closest ? node.closest("a[href]") : null) ||
                (node.querySelector ? node.querySelector("a[href]") : null);
            if (!anchor) continue;

            const href = normalize(anchor.href || anchor.getAttribute("href"));
            if (!href || href === "#" || href.startsWith("javascript:")) continue;

            const label = normalize(
                node.innerText ||
                node.textContent ||
                node.getAttribute?.("aria-label") ||
                node.getAttribute?.("title") ||
                anchor.innerText ||
                anchor.textContent ||
                anchor.getAttribute?.("aria-label") ||
                anchor.getAttribute?.("title")
            );
            if (!looksInteresting(label)) continue;

            const key = `${label}|${href}`;
            if (seen.has(key)) continue;
            seen.add(key);
            links.push({
                label,
                url: href,
                top: Number.isFinite(rect.top) ? rect.top : 0,
                left: Number.isFinite(rect.left) ? rect.left : 0,
            });
        }

        links.sort((a, b) => {
            if (a.top !== b.top) return a.top - b.top;
            return a.left - b.left;
        });
        return links.slice(0, 10).map(({label, url}) => ({label, url}));
    }
    """

    try:
        raw_links = page.evaluate(script) or []
    except Exception:
        return []

    collected: list[ApplyLink] = []
    for item in raw_links:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "") or "").strip()
        url = str(item.get("url", "") or "").strip()
        if not label or not url or not _normalize_text(label).startswith("apply"):
            continue
        try:
            collected.append(ApplyLink(label=label, url=url))
        except Exception:
            continue
    return collected


def _normalize_apply_links(dom_links: list[ApplyLink]) -> list[dict[str, str]]:
    preferred: list[dict[str, str]] = []
    seen_pairs: set[str] = set()
    for item in dom_links:
        label = str(item.label or "").strip()
        url = str(item.url or "").strip()
        pair_key = f"{_normalize_text(label)}|{url}"
        if not label or not url or pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        preferred.append({"label": label, "url": url})

    return preferred


def _select_apply_directly_link(apply_links: list[dict[str, str]]) -> Optional[str]:
    for item in apply_links:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "") or "").strip()
        url = str(item.get("url", "") or "").strip()
        if not label or not url:
            continue
        if _normalize_text(label).startswith("apply directly on"):
            return url
    return None


def _resolve_output_path(ctx: ToolContext, file_name: str) -> Path:
    runtime_state = getattr(ctx, "runtime_state", None)
    agent = getattr(runtime_state, "agent", None)
    workspace = getattr(agent, "agent_workspace", None)
    if workspace is None:
        raise RuntimeError("agent workspace is unavailable on tool runtime state")

    safe_name = Path(str(file_name or "").strip() or "google-jobs-list.jsonl").name
    safe_name = safe_name or "google-jobs-list.jsonl"
    return Path(workspace.outputs_root).expanduser().resolve() / safe_name


def _load_existing_entries(ctx: ToolContext, target_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    entries: list[dict[str, Any]] = []
    dedupe_keys: set[str] = set()

    if not target_path.exists():
        return entries, dedupe_keys

    read_error = _check_path_policy(ctx, target_path, operation="read")
    if read_error:
        raise RuntimeError(read_error)

    raw_text = target_path.read_text(encoding="utf-8", errors="replace")
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        entries.append(payload)
        dedupe_key = _build_dedupe_key(
            job_title=str(payload.get("job_title", "") or ""),
            location=str(payload.get("location", "") or ""),
            company_name=payload.get("company_name"),
        )
        if dedupe_key:
            dedupe_keys.add(dedupe_key)
    return entries, dedupe_keys


def _record_tool_action(
    ctx: ToolContext,
    *,
    args: ProcessFocusedJobArgs,
    success: bool,
    error_message: Optional[str],
    result_data: dict[str, Any],
) -> None:
    memory_store = getattr(ctx, "memory_store", None)
    if memory_store is None:
        return
    try:
        before_state = memory_store._capture_current_state()
        after_state = memory_store._capture_current_state()
        action_params = {
            "job_profile": args.job_profile,
            "file_name": args.file_name,
            "search_query": args.search_query,
        }
        action_params.update(result_data)
        memory_store.record_action(
            action_type="process_focused_job",
            action_params=action_params,
            reasoning=args.reasoning,
            before_state=before_state,
            after_state=after_state,
            success=success,
            error_message=error_message,
            mission=memory_store.current_mission,
        )
    except Exception:
        pass


def _fail(
    ctx: ToolContext,
    args: ProcessFocusedJobArgs,
    message: str,
    *,
    data: Optional[dict[str, Any]] = None,
) -> ToolOutcome:
    result_data = data if isinstance(data, dict) else {}
    _record_tool_action(
        ctx,
        args=args,
        success=False,
        error_message=message,
        result_data=result_data,
    )
    return ToolOutcome(
        output=ToolOutput(
            success=False,
            summary=f"process_focused_job FAILED: {message}",
            error=message,
            data=result_data or None,
        )
    )


@tool(
    manifest=ToolManifest(
        name="process_focused_job",
        description=(
            "Process the currently focused Google job: validate required fields, "
            "match against a job profile, dedupe against saved jobs, and append "
            "a new JSONL record when appropriate."
        ),
        effects=frozenset({Effect.READ_PAGE, Effect.READ_HOST, Effect.WRITE_HOST}),
        dialog_policy=DialogPolicy.BLOCK_WHEN_DIALOG,
        progress_policy=ProgressPolicy.NON_USER_FACING,
        tags=frozenset({"jobs", "data", "host"}),
    ),
    args_model=ProcessFocusedJobArgs,
)
def process_focused_job(ctx: ToolContext, args: ProcessFocusedJobArgs) -> ToolOutcome:
    page = getattr(ctx, "page", None)
    if page is None:
        return _fail(ctx, args, "page is unavailable")

    search_context_mismatch = _detect_search_context_mismatch(ctx, args)
    if search_context_mismatch:
        _record_tool_action(
            ctx,
            args=args,
            success=True,
            error_message=None,
            result_data=search_context_mismatch,
        )
        observed_search_query = str(
            search_context_mismatch.get("observed_search_query", "") or ""
        ).strip()
        expected_search_query = str(
            search_context_mismatch.get("expected_search_query", "") or ""
        ).strip()
        return ToolOutcome(
            output=ToolOutput(
                success=True,
                summary=(
                    "process_focused_job: search context mismatch "
                    f"(expected '{expected_search_query}', observed '{observed_search_query or 'unknown'}')"
                ),
                data=search_context_mismatch,
            )
        )

    try:
        target_path = _resolve_output_path(ctx, args.file_name)
    except Exception as exc:
        return _fail(ctx, args, str(exc))

    extracted: Optional[FocusedJobExtraction] = None
    missing_fields: list[str] = []
    for attempt in range(2):
        try:
            extracted = _extract_focused_job(ctx, args)
        except Exception as exc:
            return _fail(ctx, args, f"focused job extraction failed: {exc}")
        extracted.job_summary = _clean_job_summary(extracted.job_summary)
        missing_fields = _missing_required_fields(extracted)
        if not missing_fields:
            break
        if attempt == 0:
            try:
                page.wait_for_timeout(_RETRY_WAIT_MS)
            except Exception:
                pass

    if extracted is None:
        return _fail(ctx, args, "focused job extraction returned no data")

    apply_links = _normalize_apply_links(_collect_apply_links(ctx))
    apply_directly_link = _select_apply_directly_link(apply_links)
    dedupe_key = _build_dedupe_key(
        job_title=extracted.job_title,
        location=extracted.location,
        company_name=extracted.company_name,
    )
    source_job_url = _get_page_url(page)
    job_record = _build_job_record(
        extracted=extracted,
        apply_links=apply_links,
        apply_directly_link=apply_directly_link,
        search_query=args.search_query,
        dedupe_key=dedupe_key,
        source_job_url=source_job_url,
    )

    if missing_fields:
        result_data = {
            "processable": False,
            "saved": False,
            "duplicate": False,
            "matches_profile": None,
            "reason": "missing_required_fields",
            "missing_fields": missing_fields,
            "dedupe_key": dedupe_key,
            "job": job_record,
        }
        _record_tool_action(
            ctx,
            args=args,
            success=True,
            error_message=None,
            result_data=result_data,
        )
        return ToolOutcome(
            output=ToolOutput(
                success=True,
                summary=(
                    "process_focused_job: focused job missing required fields: "
                    + ", ".join(missing_fields)
                ),
                data=result_data,
            )
        )

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return _fail(ctx, args, f"failed to prepare output directory: {exc}")

    write_error = _check_path_policy(ctx, target_path, operation="write")
    if write_error:
        return _fail(ctx, args, write_error, data={"resolved_path": str(target_path)})

    try:
        _, existing_keys = _load_existing_entries(ctx, target_path)
    except Exception as exc:
        return _fail(ctx, args, f"failed to load existing saved jobs: {exc}")

    if dedupe_key and dedupe_key in existing_keys:
        result_data = {
            "processable": True,
            "saved": False,
            "duplicate": True,
            "matches_profile": bool(extracted.matches_profile),
            "reason": "duplicate",
            "missing_fields": [],
            "dedupe_key": dedupe_key,
            "resolved_path": str(target_path),
            "job": job_record,
            "match_reason": extracted.match_reason,
        }
        _record_tool_action(
            ctx,
            args=args,
            success=True,
            error_message=None,
            result_data=result_data,
        )
        return ToolOutcome(
            output=ToolOutput(
                success=True,
                summary=f"process_focused_job: duplicate job '{extracted.job_title}'",
                data=result_data,
            )
        )

    if not bool(extracted.matches_profile):
        result_data = {
            "processable": True,
            "saved": False,
            "duplicate": False,
            "matches_profile": False,
            "reason": "profile_mismatch",
            "missing_fields": [],
            "dedupe_key": dedupe_key,
            "resolved_path": str(target_path),
            "job": job_record,
            "match_reason": extracted.match_reason,
        }
        _record_tool_action(
            ctx,
            args=args,
            success=True,
            error_message=None,
            result_data=result_data,
        )
        return ToolOutcome(
            output=ToolOutput(
                success=True,
                summary=f"process_focused_job: skipped non-match '{extracted.job_title}'",
                data=result_data,
            )
        )

    record = job_record
    record_json = json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"

    try:
        with target_path.open("a", encoding="utf-8") as handle:
            handle.write(record_json)
    except Exception as exc:
        return _fail(
            ctx,
            args,
            f"failed to append focused job record: {exc}",
            data={"resolved_path": str(target_path)},
        )

    result_data = {
        "processable": True,
        "saved": True,
        "duplicate": False,
        "matches_profile": True,
        "reason": "saved",
        "missing_fields": [],
        "dedupe_key": dedupe_key,
        "resolved_path": str(target_path),
        "job": record,
        "job_saved_event": _build_job_saved_event_data(file_name=args.file_name, record=record),
        "match_reason": extracted.match_reason,
    }
    _log_info(
        ctx,
        "Focused Google job processed and saved",
        dedupe_key=dedupe_key,
        resolved_path=str(target_path),
    )
    _record_tool_action(
        ctx,
        args=args,
        success=True,
        error_message=None,
        result_data=result_data,
    )
    return ToolOutcome(
        output=ToolOutput(
            success=True,
            summary=f"process_focused_job: saved '{extracted.job_title}' to {target_path.name}",
            data=result_data,
        )
    )
