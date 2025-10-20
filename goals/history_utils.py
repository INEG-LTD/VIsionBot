"""Shared helpers for history navigation goals"""
from __future__ import annotations

from typing import Optional, Dict, Any, List

from .base import GoalResult, GoalStatus, GoalContext
from pydantic import BaseModel, Field
from ai_utils import generate_model


def evaluate_history_move_goal(
    *,
    direction: str,
    steps: int,
    expected_url: Optional[str],
    expected_title_substr: Optional[str],
    start_index: Optional[int],
    start_url: Optional[str],
    context: GoalContext,
) -> GoalResult:
    current_url = context.current_state.url
    current_title = context.current_state.title or ""

    # Expected URL wins if provided
    if expected_url:
        url_ok = (current_url == expected_url)
        title_ok = True if not expected_title_substr else expected_title_substr.lower() in current_title.lower()
        if url_ok and title_ok:
            return GoalResult(
                status=GoalStatus.ACHIEVED,
                confidence=0.95,
                reasoning="Reached expected URL",
                evidence={
                    "current_url": current_url,
                    "expected_url": expected_url,
                    "current_title": current_title,
                    "expected_title_contains": expected_title_substr,
                },
            )
        # NL-first fallback: if title clearly matches the expected title substring, consider success with lower confidence
        if expected_title_substr and expected_title_substr.lower() in current_title.lower():
            return GoalResult(
                status=GoalStatus.ACHIEVED,
                confidence=0.75,
                reasoning="Reached page matching expected title (URL variant detected)",
                evidence={
                    "current_url": current_url,
                    "expected_url": expected_url,
                    "current_title": current_title,
                    "expected_title_contains": expected_title_substr,
                },
            )

    # Steps + baseline index
    hist = context.url_history or []
    pointer = getattr(context, 'url_pointer', None)
    steps = max(1, int(steps or 1))

    if start_index is None:
        # Fall back to pointer when possible
        start_index = pointer if pointer is not None else (len(hist) - 1)

    if direction == 'back':
        target_idx = max(0, min(start_index, len(hist) - 1) - steps)
    else:
        target_idx = min(len(hist) - 1, max(0, start_index) + steps)

    # Prefer exact pointer match when available
    if pointer is not None and 0 <= target_idx < len(hist):
        if pointer == target_idx and current_url == hist[target_idx]:
            return GoalResult(
                status=GoalStatus.ACHIEVED,
                confidence=0.9,
                reasoning=f"Reached history index {target_idx} ({direction} {steps})",
                evidence={
                    "current_url": current_url,
                    "target_index": target_idx,
                    "start_index": start_index,
                    "direction": direction,
                    "steps": steps,
                },
            )

    # Fallback URL match to target index
    if 0 <= target_idx < len(hist) and current_url == hist[target_idx]:
        return GoalResult(
            status=GoalStatus.ACHIEVED,
            confidence=0.85,
            reasoning=f"Reached target URL at history index {target_idx}",
            evidence={
                "current_url": current_url,
                "target_index": target_idx,
                "start_index": start_index,
                "direction": direction,
                "steps": steps,
            },
        )

    # Pending otherwise
    return GoalResult(
        status=GoalStatus.PENDING,
        confidence=0.5,
        reasoning="History navigation not yet at target index",
        evidence={
            "current_url": current_url,
            "target_index": target_idx,
            "start_index": start_index,
            "pointer": pointer,
            "direction": direction,
            "steps": steps,
        },
        next_actions=["Issue another navigation step or adjust target"],
    )


# ---------------------------------------------------------------------------
# Target resolution helpers (extract/resolve back/forward intents)
# ---------------------------------------------------------------------------

class BackTargetSelection(BaseModel):
    selected_index: int = Field(default=-1, description="Absolute index in provided history list; -1 if none")
    steps_back: int = Field(default=0, description="If a simple numeric back request, number of steps back")
    rationale: str = Field(default="", description="Reason for choice")


class ForwardTargetSelection(BaseModel):
    selected_index: int = Field(default=-1, description="Absolute index in provided history list; -1 if none")
    steps_forward: int = Field(default=0, description="If a simple numeric forward request, number of steps forward")
    rationale: str = Field(default="", description="Reason for choice")


def extract_back_request(description: str) -> Dict[str, Optional[str | int]]:
    text = (description or "").lower().strip()
    import re as _re
    # Explicit numeric forms
    for pat in [r"go\s+(\d+)\s+pages?\s+back", r"back\s+(\d+)\s+pages?", r"go back\s+(\d+)\s+pages?"]:
        m = _re.search(pat, text)
        if m:
            try:
                n = int(m.group(1))
                return {"mode": "steps", "steps": max(1, n), "query": None}
            except Exception:
                pass
    # Named page forms
    for pat in [r"go back to (.+)", r"back to (.+)", r"return to (.+)"]:
        m = _re.search(pat, text)
        if m:
            return {"mode": "named", "steps": None, "query": m.group(1).strip()}
    # Simple back
    if any(x in text for x in ["go back", "back", "previous page", "prev page"]):
        return {"mode": "steps", "steps": 1, "query": None}
    return {"mode": None, "steps": None, "query": None}


def resolve_back_target(
    description: str,
    url_history: List[str],
    pointer: Optional[int],
    state_history: Optional[List[Any]] = None,
    current_url: Optional[str] = None,
) -> Dict[str, Optional[int | str]]:
    """Resolve a back navigation request into concrete targets.

    Returns dict with: steps_back, target_index, expected_url, expected_title_substr
    """
    req = extract_back_request(description)

    # Titles map from state history if available
    titles: Dict[int, str] = {}
    try:
        if state_history:
            title_by_url = {}
            for st in state_history:
                if getattr(st, 'url', None):
                    title_by_url[st.url] = getattr(st, 'title', "") or ""
            for idx, url in enumerate(url_history or []):
                titles[idx] = title_by_url.get(url, "")
    except Exception:
        pass

    # If insufficient history → default to back(1)
    if not url_history or len(url_history) < 2:
        return {"steps_back": 1, "target_index": None, "expected_url": None, "expected_title_substr": None}

    # Try AI-assisted selection
    try:
        max_items = 14
        start = max(0, len(url_history) - max_items)
        lines = []
        for i in range(start, len(url_history)):
            title = titles.get(i, "")
            marker = " (current)" if pointer is not None and i == pointer else ""
            lines.append(f"{i}: {title} | {url_history[i]}{marker}")
        history_block = "\n".join(lines)

        numeric_hint = int(req.get("steps") or 0) if req.get("mode") == "steps" else 0
        named_hint = (req.get("query") or "").strip()

        system = (
            "Given the user's back-navigation request and recent history, choose a target index to go to,\n"
            "or a steps_back value. Use absolute indices from the list.\n"
            "If the request is numeric (e.g., 'back 2 pages'), prefer pointer - N if valid.\n"
            "Do not choose about:blank unless the user explicitly asks for it.\n"
            "Return only structured fields."
        )
        prompt = (
            f"User back request: {description}\n"
            f"Parsed numeric hint: {numeric_hint}\n"
            f"Parsed named hint: {named_hint}\n\n"
            f"History (absolute indices):\n{history_block}\n\n"
            "Choose the best selected_index (absolute) OR a steps_back value."
        )
        model_out = generate_model(
            prompt=prompt,
            model_object_type=BackTargetSelection,
            system_prompt=system,
        )
        selected_index = int(getattr(model_out, 'selected_index', -1) or -1)
        steps_back = int(getattr(model_out, 'steps_back', 0) or 0)
        if selected_index >= 0 and selected_index < len(url_history):
            expected_url = url_history[selected_index]
            return {
                "steps_back": max(1, (pointer - selected_index) if pointer is not None else (len(url_history) - 1 - selected_index)),
                "target_index": selected_index,
                "expected_url": expected_url,
                "expected_title_substr": titles.get(selected_index, "") or None,
            }
        if steps_back > 0:
            idx = max(0, (pointer - steps_back) if pointer is not None else (len(url_history) - 1 - steps_back))
            return {
                "steps_back": steps_back,
                "target_index": idx,
                "expected_url": url_history[idx] if 0 <= idx < len(url_history) else None,
                "expected_title_substr": titles.get(idx, "") or None,
            }
    except Exception:
        pass

    # Deterministic fallback
    if req.get("mode") == "steps" and req.get("steps"):
        n = max(1, int(req["steps"]))
        idx = max(0, (pointer - n) if pointer is not None else (len(url_history) - 1 - n))
        return {
            "steps_back": n,
            "target_index": idx,
            "expected_url": url_history[idx] if 0 <= idx < len(url_history) else None,
            "expected_title_substr": titles.get(idx, "") or None,
        }
    if req.get("mode") == "named" and req.get("query"):
        q = (req["query"] or "").lower()
        best_idx = None
        for i in range((pointer - 1) if pointer else (len(url_history) - 2), -1, -1):
            title = (titles.get(i, "") or "").lower()
            url = (url_history[i] or "").lower()
            if all(tok in (title + " " + url) for tok in q.split() if tok):
                best_idx = i
                break
        if best_idx is not None:
            return {
                "steps_back": max(1, (pointer - best_idx) if pointer is not None else (len(url_history) - 1 - best_idx)),
                "target_index": best_idx,
                "expected_url": url_history[best_idx],
                "expected_title_substr": titles.get(best_idx, "") or None,
            }
    # Default simple back(1), but avoid about:blank if possible
    default_idx = (pointer - 1) if pointer is not None else (len(url_history) - 2)
    default_idx = max(0, default_idx)
    default_url = url_history[default_idx] if 0 <= default_idx < len(url_history) else None
    if (default_url or "").startswith("about:blank") and default_idx - 1 >= 0:
        return {
            "steps_back": 2,
            "target_index": max(0, default_idx - 1),
            "expected_url": url_history[max(0, default_idx - 1)],
            "expected_title_substr": titles.get(max(0, default_idx - 1), "") or None,
        }
    return {"steps_back": 1, "target_index": default_idx, "expected_url": default_url, "expected_title_substr": titles.get(default_idx, "") or None}


def extract_forward_request(description: str) -> Dict[str, Optional[str | int]]:
    text = (description or "").lower().strip()
    import re as _re
    for pat in [r"go\s+(\d+)\s+pages?\s+forward", r"forward\s+(\d+)\s+pages?", r"go forward\s+(\d+)\s+pages?"]:
        m = _re.search(pat, text)
        if m:
            try:
                n = int(m.group(1))
                return {"mode": "steps", "steps": max(1, n), "query": None}
            except Exception:
                pass
    if any(x in text for x in ["go forward", "forward", "next page"]):
        return {"mode": "steps", "steps": 1, "query": None}
    return {"mode": None, "steps": None, "query": None}


def resolve_forward_target(
    description: str,
    url_history: List[str],
    pointer: Optional[int],
    state_history: Optional[List[Any]] = None,
    current_url: Optional[str] = None,
) -> Dict[str, Optional[int | str]]:
    """Resolve a forward navigation request into concrete targets.

    Returns dict with: steps_forward, target_index, expected_url
    """
    req = extract_forward_request(description)

    # If insufficient history
    if not url_history or len(url_history) < 2:
        return {"steps_forward": 1, "target_index": None, "expected_url": None}

    # Try AI-assisted selection (mirror back selection but simpler)
    try:
        max_items = 14
        start = max(0, len(url_history) - max_items)
        lines = []
        for i in range(start, len(url_history)):
            marker = " (current)" if pointer is not None and i == pointer else ""
            lines.append(f"{i}: {url_history[i]}{marker}")
        history_block = "\n".join(lines)

        numeric_hint = int(req.get("steps") or 0) if req.get("mode") == "steps" else 0
        system = (
            "Given the user's forward-navigation request and recent history, choose a target index to go to,\n"
            "or a steps_forward value. Use absolute indices from the list. Return structured fields."
        )
        prompt = (
            f"User forward request: {description}\n"
            f"Parsed numeric hint: {numeric_hint}\n\n"
            f"History (absolute indices):\n{history_block}\n\n"
            "Choose the best selected_index (absolute) OR a steps_forward value."
        )
        model_out = generate_model(
            prompt=prompt,
            model_object_type=ForwardTargetSelection,
            system_prompt=system,
        )
        selected_index = int(getattr(model_out, 'selected_index', -1) or -1)
        steps_forward = int(getattr(model_out, 'steps_forward', 0) or 0)
        if selected_index >= 0 and selected_index < len(url_history):
            expected_url = url_history[selected_index]
            return {
                "steps_forward": max(1, (selected_index - pointer) if pointer is not None else (selected_index - (len(url_history) - 1))),
                "target_index": selected_index,
                "expected_url": expected_url,
            }
        if steps_forward > 0:
            idx = min(len(url_history) - 1, (pointer + steps_forward) if pointer is not None else (len(url_history) - 1))
            return {
                "steps_forward": steps_forward,
                "target_index": idx,
                "expected_url": url_history[idx] if 0 <= idx < len(url_history) else None,
            }
    except Exception:
        pass

    # Deterministic fallback
    if req.get("mode") == "steps" and req.get("steps"):
        n = max(1, int(req["steps"]))
        idx = min(len(url_history) - 1, (pointer + n) if pointer is not None else (len(url_history) - 1))
        return {
            "steps_forward": n,
            "target_index": idx,
            "expected_url": url_history[idx] if 0 <= idx < len(url_history) else None,
        }

    # Default simple forward(1)
    default_idx = (pointer + 1) if pointer is not None else (len(url_history) - 1)
    default_idx = min(len(url_history) - 1, max(0, default_idx))
    default_url = url_history[default_idx] if 0 <= default_idx < len(url_history) else None
    return {"steps_forward": 1, "target_index": default_idx, "expected_url": default_url}
