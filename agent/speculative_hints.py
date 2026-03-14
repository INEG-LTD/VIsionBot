"""Speculative hint runtime helpers.

These helpers keep deterministic filtering and validation separate from execution
so safety checks remain testable.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

from lib.ai import ReasoningLevel, generate_model_with_cost
from models.models import ActionStep, PageElements


class _StrictModel(BaseModel):
    """Base model with strict JSON-schema compatibility for Responses.parse()."""

    model_config = ConfigDict(extra="forbid")


class HintPreconditions(_StrictModel):
    """Optional page-state constraints for using a speculative candidate."""

    url_contains: Optional[str] = None
    title_contains: Optional[str] = None
    dialog_pending: Optional[bool] = None
    tab_id: Optional[str] = None


class HintTargetSignature(_StrictModel):
    """Expected target characteristics for deterministic re-validation."""

    overlay_index: Optional[int] = None
    element_type: Optional[str] = None
    text_contains: Optional[str] = None
    description_contains: Optional[str] = None


class HintCandidate(_StrictModel):
    """One speculative next action candidate."""

    candidate_id: str = Field(min_length=1)
    function_name: str = Field(min_length=1)
    function_arguments: Dict[str, Any] = Field(default_factory=dict)
    target_signature: Optional[HintTargetSignature] = None
    preconditions: Optional[HintPreconditions] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class HintBundle(_StrictModel):
    """Prediction payload produced after an iteration for the next iteration."""

    source_iteration: int = 0
    source_url: str = ""
    source_title: str = ""
    candidates: List[HintCandidate] = Field(default_factory=list)


RejectReason = Literal[
    "target_missing",
    "precondition_failed",
    "state_conflict",
    "low_confidence",
    "invalid_candidate_id",
    "validator_error",
    "abstain",
]


class HintValidationResult(_StrictModel):
    """Validator decision for speculative hints."""

    decision: Literal["accept", "reject", "abstain"] = "abstain"
    candidate_id: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    reject_reason: Optional[RejectReason] = None


_DEFAULT_TOOLS_REQUIRING_ELEMENT = {
    "click",
    "type_text",
    "clear_text",
    "select_option",
    "scroll_container",
    "scroll_to_element",
}
_TOOLS_REQUIRING_ELEMENT = set(_DEFAULT_TOOLS_REQUIRING_ELEMENT)


def set_tools_requiring_element(tool_names: Sequence[str]) -> None:
    """Override the element-targeting requirement set from tool registry metadata."""
    global _TOOLS_REQUIRING_ELEMENT
    normalized = {
        _safe_lower(name)
        for name in (tool_names or [])
        if str(name or "").strip()
    }
    if not normalized:
        _TOOLS_REQUIRING_ELEMENT = set(_DEFAULT_TOOLS_REQUIRING_ELEMENT)
        return
    _TOOLS_REQUIRING_ELEMENT = normalized


def _safe_lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _candidate_overlay_index(candidate: HintCandidate) -> Optional[int]:
    if candidate.target_signature and candidate.target_signature.overlay_index is not None:
        return int(candidate.target_signature.overlay_index)
    args = candidate.function_arguments or {}
    if args.get("element_id") is not None:
        try:
            return int(args.get("element_id"))
        except Exception:
            return None
    if args.get("overlay_index") is not None:
        try:
            return int(args.get("overlay_index"))
        except Exception:
            return None
    return None


def _tool_needs_element(tool_name: str) -> bool:
    return _safe_lower(tool_name) in _TOOLS_REQUIRING_ELEMENT


def _candidate_action_preview(candidate: HintCandidate) -> str:
    args = candidate.function_arguments or {}
    element_id = args.get("element_id")
    if element_id is None:
        element_id = _candidate_overlay_index(candidate)
    suffix = f" [id={element_id}]" if element_id is not None else ""
    return f"{candidate.function_name}{suffix}"


def _compact_text(value: Any, max_chars: int = 90) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return f"{text[: max(0, max_chars - 3)].rstrip()}..."


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _build_fallback_narrative(function_name: str, args: Dict[str, Any]) -> str:
    tool = _safe_lower(function_name)
    element_id = _safe_int(args.get("element_id"))
    if element_id is None:
        element_id = _safe_int(args.get("overlay_index"))

    if tool == "click":
        description = _compact_text(args.get("description"), 110)
        if description:
            return f"I'm clicking {description}."
        if element_id is not None:
            return f"I'm clicking element [id={element_id}]."
        return "I'm clicking an element."

    if tool == "type_text":
        field = _compact_text(args.get("field_description"), 100)
        if field:
            return f"I'm entering text into the {field}."
        return "I'm entering text into an input field."

    if tool == "clear_text":
        field = _compact_text(args.get("field_description"), 100)
        if field:
            return f"I'm clearing the text in the {field}."
        return "I'm clearing an input field."

    if tool == "select_option":
        option = _compact_text(args.get("intent"), 60)
        dropdown = _compact_text(args.get("dropdown_description"), 90)
        if option and dropdown:
            return f"I'm selecting '{option}' in the {dropdown}."
        if option:
            return f"I'm selecting '{option}'."
        if dropdown:
            return f"I'm choosing an option in the {dropdown}."
        return "I'm selecting an option."

    if tool == "upload_file":
        file_path = _compact_text(args.get("file_path"), 100)
        target = _compact_text(args.get("target_description"), 90)
        if file_path and target:
            return f"I'm uploading {file_path} to the {target}."
        if file_path:
            return f"I'm uploading {file_path}."
        if target:
            return f"I'm uploading a file to the {target}."
        return "I'm uploading a file."

    if tool == "press_key":
        key = _compact_text(args.get("key"), 40)
        if key:
            return f"I'm pressing {key}."
        return "I'm pressing a key."

    if tool == "open_url":
        url = _compact_text(args.get("url"), 140)
        if url:
            return f"I'm opening {url}."
        return "I'm opening the target page."

    if tool == "go_back":
        steps = _safe_int(args.get("steps")) or 1
        if steps > 1:
            return f"I'm going back {steps} pages."
        return "I'm going back to the previous page."

    if tool == "go_forward":
        steps = _safe_int(args.get("steps")) or 1
        if steps > 1:
            return f"I'm going forward {steps} pages."
        return "I'm going forward to the next page."

    if tool == "scroll_down":
        return "I'm scrolling down the page."

    if tool == "scroll_up":
        return "I'm scrolling up the page."

    if tool == "scroll_container":
        direction = _safe_lower(args.get("direction")) or "down"
        if element_id is not None:
            return f"I'm scrolling {direction} inside element [id={element_id}]."
        return f"I'm scrolling {direction} inside a container."

    if tool == "scroll_to_element":
        if element_id is not None:
            return f"I'm scrolling to element [id={element_id}]."
        return "I'm scrolling to the target element."

    if tool == "extract_data":
        description = _compact_text(args.get("data_description"), 110)
        if description:
            return f"I'm extracting {description}."
        return "I'm extracting the requested data."

    if tool == "think":
        next_action = _safe_lower(args.get("next_action"))
        if next_action and next_action != "continue":
            return f"I'm thinking through the next step ({next_action})."
        return "I'm thinking through the next step."

    if tool == "assert_condition":
        condition = _compact_text(args.get("condition"), 120)
        if condition:
            return f"I'm checking whether {condition}."
        return "I'm checking that the expected condition is true."

    if tool == "flag":
        message = _compact_text(args.get("message"), 120)
        if message:
            return f"I'm flagging this to the user: {message}"
        return "I'm flagging an issue for the user."

    if tool == "wait_for":
        condition = _compact_text(args.get("condition"), 120)
        timeout = _safe_int(args.get("timeout_seconds"))
        if condition and timeout and timeout > 0:
            return f"I'm waiting for {condition} (up to {timeout} seconds)."
        if condition:
            return f"I'm waiting for {condition}."
        return "I'm waiting for the page to update."

    if tool == "ask_user":
        question = _compact_text(args.get("question"), 120)
        if question:
            return f"I'm asking the user: {question}"
        return "I'm asking the user for clarification."

    if tool == "report_data":
        return "I'm sharing the requested data with the user."

    if tool == "write_data":
        target = _compact_text(args.get("path"), 100) or _compact_text(args.get("file_name"), 80)
        if target:
            return f"I'm writing data to {target}."
        return "I'm writing the data to a file."

    if tool == "switch_tab":
        tab_id = _compact_text(args.get("tab_id"), 20)
        if tab_id:
            return f"I'm switching to tab {tab_id}."
        return "I'm switching browser tabs."

    if tool == "close_tab":
        tab_id = _compact_text(args.get("tab_id"), 20)
        if tab_id:
            return f"I'm closing tab {tab_id}."
        return "I'm closing the current tab."

    if tool == "open_tab":
        url = _compact_text(args.get("url"), 140)
        if url:
            return f"I'm opening a new tab at {url}."
        return "I'm opening a new browser tab."

    if tool == "dismiss_dialog":
        accept = args.get("accept")
        if accept is True:
            return "I'm accepting the browser dialog."
        if accept is False:
            return "I'm dismissing the browser dialog."
        return "I'm handling the browser dialog."

    if tool == "send_email":
        recipient = _compact_text(args.get("to"), 80)
        subject = _compact_text(args.get("subject"), 90)
        if recipient and subject:
            return f"I'm sending an email to {recipient} with subject '{subject}'."
        if recipient:
            return f"I'm sending an email to {recipient}."
        return "I'm sending an email."

    if tool == "bash":
        command = _compact_text(args.get("command"), 120)
        if command:
            return f"I'm running this shell command: {command}"
        return "I'm running a shell command."

    if tool == "read_file":
        path = _compact_text(args.get("path"), 120)
        if path:
            return f"I'm reading {path}."
        return "I'm reading a file."

    if tool == "find_files":
        pattern = _compact_text(args.get("pattern"), 80)
        directory = _compact_text(args.get("directory"), 80)
        if pattern and directory:
            return f"I'm searching for files matching '{pattern}' in {directory}."
        if pattern:
            return f"I'm searching for files matching '{pattern}'."
        return "I'm searching for matching files."

    if tool == "read_clipboard":
        return "I'm reading the clipboard contents."

    return "I'm taking the next step to keep the task moving."


def filter_candidates_deterministic(
    *,
    hint_bundle: HintBundle,
    current_url: str,
    current_title: str,
    detected_elements: PageElements,
    policy_visible_tool_names: Optional[Sequence[str]] = None,
    dialog_pending: bool = False,
    in_loop: bool = False,
    min_confidence: float = 0.0,
) -> List[HintCandidate]:
    """Apply deterministic guardrails before any validator/model decision."""
    allowed = {
        _safe_lower(name)
        for name in (policy_visible_tool_names or [])
        if str(name or "").strip()
    }
    use_policy_filter = bool(allowed)

    overlay_map: Dict[int, Any] = {}
    for element in (getattr(detected_elements, "elements", None) or []):
        idx = getattr(element, "overlay_number", None)
        if idx is None:
            continue
        try:
            overlay_map[int(idx)] = element
        except Exception:
            continue

    filtered: List[HintCandidate] = []
    for candidate in hint_bundle.candidates:
        tool_name = _safe_lower(candidate.function_name)
        if not tool_name:
            continue
        if use_policy_filter and tool_name not in allowed:
            continue
        if float(candidate.confidence or 0.0) < float(min_confidence or 0.0):
            continue

        pre = candidate.preconditions
        if pre:
            expected_url = str(pre.url_contains or "").strip()
            expected_title = str(pre.title_contains or "").strip()
            if expected_url and expected_url not in str(current_url or ""):
                continue
            if expected_title and expected_title not in str(current_title or ""):
                continue
            if pre.dialog_pending is not None and bool(pre.dialog_pending) != bool(dialog_pending):
                continue

        overlay_index = _candidate_overlay_index(candidate)
        if _tool_needs_element(tool_name) and overlay_index is None:
            continue
        if overlay_index is not None:
            element = overlay_map.get(int(overlay_index))
            if element is None:
                continue
            if in_loop and bool(getattr(element, "is_done", False)):
                continue
            signature = candidate.target_signature
            if signature:
                expected_type = _safe_lower(signature.element_type)
                actual_type = _safe_lower(getattr(element, "element_type", ""))
                if expected_type and actual_type and expected_type not in actual_type and actual_type not in expected_type:
                    continue

                text_expected = str(
                    signature.text_contains
                    or signature.description_contains
                    or ""
                ).strip()
                if text_expected:
                    haystack = " ".join(
                        [
                            str(getattr(element, "element_label", "") or ""),
                            str(getattr(element, "css_id", "") or ""),
                            str(getattr(element, "css_class", "") or ""),
                        ]
                    ).lower()
                    if text_expected.lower() not in haystack:
                        continue

        filtered.append(candidate)
    return filtered


def validate_hints(
    *,
    mission: str,
    current_url: str,
    current_title: str,
    hint_bundle: HintBundle,
    filtered_candidates: Sequence[HintCandidate],
    screenshot: bytes,
    element_index_text: str,
    model: str,
    reasoning_level: Union[ReasoningLevel, str, None],
    image_detail: str = "low",
) -> HintValidationResult:
    """Run a fast validator decision over pre-filtered speculative candidates."""
    candidates = list(filtered_candidates or [])
    if not candidates:
        return HintValidationResult(
            decision="reject",
            confidence=1.0,
            reason="No deterministic-valid candidates available.",
            reject_reason="target_missing",
        )

    serialized_candidates: List[Dict[str, Any]] = []
    for candidate in candidates:
        serialized_candidates.append(
            {
                "candidate_id": candidate.candidate_id,
                "confidence": candidate.confidence,
                "action": _candidate_action_preview(candidate),
                "function_name": candidate.function_name,
                "function_arguments": candidate.function_arguments,
                "target_signature": (
                    candidate.target_signature.model_dump(mode="python")
                    if candidate.target_signature is not None
                    else None
                ),
                "preconditions": (
                    candidate.preconditions.model_dump(mode="python")
                    if candidate.preconditions is not None
                    else None
                ),
                "reason": candidate.reason,
            }
        )

    user_prompt = (
        "Decide whether to accept one speculative candidate for immediate execution.\n"
        "Return HintValidationResult JSON only.\n"
        f"Mission: {mission}\n"
        f"Current URL: {current_url}\n"
        f"Current title: {current_title}\n"
        f"Source iteration for bundle: {hint_bundle.source_iteration}\n"
        f"Candidates: {json.dumps(serialized_candidates, ensure_ascii=True)}\n\n"
        "Interactive element index:\n"
        f"{element_index_text or 'No interactive elements detected.'}\n"
    )
    system_prompt = (
        "You are a strict speculative-action validator.\n"
        "Accept only if a candidate is clearly consistent with the current screenshot and page state.\n"
        "If uncertain, reject or abstain."
    )

    try:
        decision, _, _ = generate_model_with_cost(
            prompt=user_prompt,
            model_object_type=HintValidationResult,
            system_prompt=system_prompt,
            model=model,
            reasoning_level=reasoning_level,
            image=screenshot,
            image_detail=image_detail,
        )
        if not isinstance(decision, HintValidationResult):
            return HintValidationResult(
                decision="reject",
                confidence=0.0,
                reason="Validator returned invalid output.",
                reject_reason="validator_error",
            )
        if decision.decision == "accept":
            accepted_id = str(decision.candidate_id or "").strip()
            valid_ids = {c.candidate_id for c in candidates}
            if not accepted_id or accepted_id not in valid_ids:
                return HintValidationResult(
                    decision="reject",
                    confidence=0.0,
                    reason="Validator accepted an unknown candidate_id.",
                    reject_reason="invalid_candidate_id",
                )
        return decision
    except Exception:
        return HintValidationResult(
            decision="reject",
            confidence=0.0,
            reason="Validator call failed.",
            reject_reason="validator_error",
        )


def hydrate_candidate_to_action_step(
    candidate: HintCandidate,
    *,
    budget_spent: int,
    budget_remaining: int,
    budget_total: int,
) -> ActionStep:
    """Convert a validated speculative candidate into an executable action step."""
    args = dict(candidate.function_arguments or {})
    args["budget_spent"] = int(max(0, budget_spent))
    args["budget_remaining"] = int(max(0, budget_remaining))
    args["budget_total"] = int(max(1, budget_total))

    if not str(args.get("reasoning", "")).strip():
        args["reasoning"] = (
            f"Using validated speculative hint candidate {candidate.candidate_id}."
        )
    if not str(args.get("narrative", "")).strip():
        args["narrative"] = _build_fallback_narrative(candidate.function_name, args)

    if (
        _tool_needs_element(candidate.function_name)
        and args.get("element_id") is None
        and candidate.target_signature is not None
        and candidate.target_signature.overlay_index is not None
    ):
        args["element_id"] = int(candidate.target_signature.overlay_index)

    return ActionStep.from_function_call(
        function_name=str(candidate.function_name or "").strip(),
        arguments=args,
    )
