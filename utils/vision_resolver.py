"""Vision-based overlay resolver utilities.

These helpers provide deterministic, vision-only scoring of overlay
elements so that fast-path actions can avoid DOM queries entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import re

from utils.semantic_targets import (
    SemanticTarget,
    build_semantic_target,
    semantic_score_element,
)


# ---------------------------------------------------------------------------
# Data structures


@dataclass(frozen=True)
class VisionResolution:
    """Best-match overlay resolution details."""

    best_index: Optional[int]
    scored: List[Tuple[int, float]]
    top_score: float
    next_score: Optional[float]


@dataclass(frozen=True)
class _InstructionProfile:
    tokens: List[str]
    hint_terms: List[str]
    semantic_target: Optional[SemanticTarget]
    mode: str
    interpretation_mode: str


# ---------------------------------------------------------------------------
# Tokenisation helpers


_STOP_WORDS = {
    "the",
    "a",
    "an",
    "to",
    "for",
    "please",
    "click",
    "tap",
    "button",
    "press",
    "open",
    "show",
    "select",
    "choice",
    "option",
    "field",
    "input",
    "type",
    "fill",
    "enter",
    "into",
    "text",
    "box",
    "and",
    "or",
    "upload",
    "file",
}


def _tokenize(instruction: str) -> List[str]:
    text = (instruction or "").lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", text) if t and len(t) > 1]
    filtered = [t for t in tokens if t not in _STOP_WORDS]
    deduped: List[str] = []
    seen = set()
    for tok in filtered:
        if tok not in seen:
            deduped.append(tok)
            seen.add(tok)
    return deduped[:20]


def _extract_hint_terms(semantic_hint: Optional[SemanticTarget]) -> List[str]:
    if not semantic_hint:
        return []
    terms: List[str] = [
        *(semantic_hint.required_terms or []),
        *(semantic_hint.primary_terms or []),
        *(semantic_hint.context_terms or []),
    ]
    normalized = [t.strip().lower() for t in terms if t]
    deduped: List[str] = []
    seen = set()
    for term in normalized:
        if term not in seen:
            deduped.append(term)
            seen.add(term)
    return deduped[:20]


def _build_instruction_profile(
    instruction: str,
    *,
    mode: str,
    interpretation_mode: str,
    semantic_hint: Optional[SemanticTarget],
) -> _InstructionProfile:
    tokens = _tokenize(instruction)
    hint = _extract_hint_terms(semantic_hint)
    sem_target = semantic_hint
    if interpretation_mode.lower() != "literal" and sem_target is None:
        sem_target = build_semantic_target(instruction)
    return _InstructionProfile(
        tokens=tokens,
        hint_terms=hint,
        semantic_target=sem_target,
        mode=mode,
        interpretation_mode=interpretation_mode.lower(),
    )


# ---------------------------------------------------------------------------
# Element feature helpers


def _get_text_field(element: Dict[str, Any], key: str) -> str:
    value = element.get(key)
    if not value:
        return ""
    return str(value)


def _combined_text(element: Dict[str, Any]) -> str:
    parts = [
        _get_text_field(element, "textContent"),
        _get_text_field(element, "ariaLabel"),
        _get_text_field(element, "description"),
        _get_text_field(element, "placeholder"),
    ]
    combined = " ".join(part for part in parts if part).lower()
    return combined


def _context_text(element: Dict[str, Any]) -> str:
    return _get_text_field(element, "contextText").lower()


def _is_clickable(element: Dict[str, Any]) -> bool:
    tag = _get_text_field(element, "tagName").lower()
    role = _get_text_field(element, "role").lower()
    etype = _get_text_field(element, "type").lower()
    if tag in {"a", "button"}:
        return True
    if role in {"button", "link", "menuitem", "tab", "checkbox", "radio"}:
        return True
    if tag == "input" and etype in {"button", "submit", "reset"}:
        return True
    return bool(element.get("isClickable"))


def _is_textual_field(element: Dict[str, Any]) -> bool:
    tag = _get_text_field(element, "tagName").lower()
    etype = _get_text_field(element, "type").lower()
    if tag in {"textarea"}:
        return True
    if tag == "input" and etype in {"text", "email", "search", "password", "url", "number"}:
        return True
    return bool(element.get("isTextField"))


def _is_select(element: Dict[str, Any]) -> bool:
    tag = _get_text_field(element, "tagName").lower()
    role = _get_text_field(element, "role").lower()
    etype = _get_text_field(element, "type").lower()
    return tag == "select" or role in {"combobox", "listbox"} or etype in {"select"}


def _is_datetime(element: Dict[str, Any]) -> bool:
    tag = _get_text_field(element, "tagName").lower()
    etype = _get_text_field(element, "type").lower()
    return tag == "input" and etype in {"date", "datetime", "datetime-local", "time"}


def _is_upload(element: Dict[str, Any]) -> bool:
    tag = _get_text_field(element, "tagName").lower()
    etype = _get_text_field(element, "type").lower()
    return tag == "input" and etype == "file"


def _normalized_area_ratio(element: Dict[str, Any]) -> float:
    coords = element.get("normalizedCoords") or element.get("box_2d") or []
    if not isinstance(coords, Sequence) or len(coords) < 4:
        return 0.0
    try:
        y1, x1, y2, x2 = coords[:4]
        height = max(float(y2) - float(y1), 0.0)
        width = max(float(x2) - float(x1), 0.0)
        return (height * width) / 1_000_000.0
    except Exception:
        return 0.0


def _detail_snapshot(element: Dict[str, Any]) -> str:
    tag = _get_text_field(element, "tagName").lower()
    role = _get_text_field(element, "role").lower()
    text = re.sub(r"\s+", " ", _combined_text(element))[:80]
    return f"tag={tag} role={role} text='{text}'"


def _normalize_coords_from_metadata(element: Dict[str, Any], page_w: int, page_h: int) -> List[int]:
    if page_w <= 0:
        page_w = 1000
    if page_h <= 0:
        page_h = 1000

    coords = element.get("normalizedCoords") or element.get("box_2d")
    if isinstance(coords, Sequence) and len(coords) >= 4:
        return [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]

    bounds = element.get("bounds") or element.get("rect")
    if isinstance(bounds, dict):
        try:
            x = float(bounds.get("x", bounds.get("left", 0)))
            y = float(bounds.get("y", bounds.get("top", 0)))
            w = float(bounds.get("width", 0))
            h = float(bounds.get("height", 0))
        except Exception:
            x = y = w = h = 0.0
        x2 = x + max(w, 1.0)
        y2 = y + max(h, 1.0)
        return [
            int(max(0, min(1000, round(y / page_h * 1000)))),
            int(max(0, min(1000, round(x / page_w * 1000)))),
            int(max(0, min(1000, round(y2 / page_h * 1000)))),
            int(max(0, min(1000, round(x2 / page_w * 1000)))),
        ]

    center = element.get("coordinates") or {}
    if isinstance(center, dict) and {"x", "y"}.issubset(center.keys()):
        try:
            cx = float(center.get("x", 0))
            cy = float(center.get("y", 0))
        except Exception:
            cx = cy = 0.0
        size = float(center.get("size", 44)) or 44.0
        half = size / 2.0
        x1 = cx - half
        y1 = cy - half
        x2 = cx + half
        y2 = cy + half
        return [
            int(max(0, min(1000, round(y1 / page_h * 1000)))),
            int(max(0, min(1000, round(x1 / page_w * 1000)))),
            int(max(0, min(1000, round(y2 / page_h * 1000)))),
            int(max(0, min(1000, round(x2 / page_w * 1000)))),
        ]

    return [0, 0, 0, 0]


def _coerce_overlay_metadata(element: Dict[str, Any], page_w: int, page_h: int) -> Dict[str, Any]:
    if not element:
        return {
            "tagName": "",
            "role": "",
            "type": "",
            "ariaLabel": "",
            "textContent": "",
            "placeholder": "",
            "name": "",
            "id": "",
            "className": "",
            "href": "",
            "normalizedCoords": [0, 0, 0, 0],
            "isClickable": False,
        }

    attrs = element.get("attributes") or {}
    tag = (element.get("tagName") or element.get("tag") or "").lower()
    if not tag:
        elem_type = element.get("elementType")
        if isinstance(elem_type, str) and "[" in elem_type:
            tag = elem_type.split("[", 1)[0].lower()
        elif isinstance(elem_type, str):
            tag = elem_type.lower()

    role = (
        element.get("role")
        or attrs.get("role")
        or ""
    )
    etype = (
        element.get("type")
        or attrs.get("type")
        or ""
    )

    overlay = {
        "tagName": tag,
        "role": role,
        "type": etype,
        "ariaLabel": element.get("ariaLabel") or attrs.get("aria-label") or "",
        "textContent": element.get("text")
        or element.get("textContent")
        or element.get("innerText")
        or attrs.get("text")
        or "",
        "placeholder": element.get("placeholder") or attrs.get("placeholder") or "",
        "name": element.get("name") or attrs.get("name") or "",
        "id": element.get("id") or attrs.get("id") or "",
        "className": element.get("className") or attrs.get("class") or "",
        "href": element.get("href") or attrs.get("href") or "",
        "normalizedCoords": _normalize_coords_from_metadata(element, page_w, page_h),
        "isClickable": bool(element.get("isClickable")),
    }
    return overlay


# ---------------------------------------------------------------------------
# Scoring


def _score_element(
    element: Dict[str, Any],
    profile: _InstructionProfile,
) -> Tuple[float, List[str], List[str]]:
    text = _combined_text(element)
    context = _context_text(element)
    score = 0.0
    token_hits: List[str] = []
    hint_hits: List[str] = []

    for token in profile.tokens:
        if token in text:
            score += 6
            token_hits.append(token)
        elif token in context:
            score += 3
            token_hits.append(token)

    for hint in profile.hint_terms:
        if hint in text:
            score += 3
            hint_hits.append(hint)
        elif hint in context:
            score += 1
            hint_hits.append(hint)

    # Area heuristics – discourage very small/very large regions
    area = _normalized_area_ratio(element)
    if area < 0.001:
        score -= 6
    elif area < 0.005:
        score -= 3
    elif area > 0.18:
        score -= 6
    elif area > 0.08:
        score -= 3
    elif area > 0.03:
        score -= 1

    mode = profile.mode
    if mode == "click":
        if _is_clickable(element):
            score += 8
        else:
            score -= 4
    elif mode in {"field", "type"}:
        if _is_textual_field(element):
            score += 8
        elif _is_clickable(element):
            score -= 2
        else:
            score -= 4
    elif mode == "select":
        if _is_select(element):
            score += 8
        elif _is_clickable(element):
            score += 2
        else:
            score -= 3
    elif mode == "datetime":
        if _is_datetime(element):
            score += 8
        elif _is_textual_field(element):
            score += 2
        else:
            score -= 4
    elif mode == "upload":
        if _is_upload(element):
            score += 10
        elif "upload" in text or "attach" in text:
            score += 4
        else:
            score -= 4

    return score, token_hits, hint_hits


def _apply_semantic_target(
    scored: Dict[int, float],
    element_map: Dict[int, Dict[str, Any]],
    profile: _InstructionProfile,
    semantic_target: Optional[SemanticTarget],
) -> None:
    if not semantic_target:
        return
    valid_hit = False
    for idx, element in element_map.items():
        sem_score = semantic_score_element(element, semantic_target)
        if sem_score is None:
            scored[idx] = scored.get(idx, 0.0) - 1000
            continue
        valid_hit = True
        scored[idx] = scored.get(idx, 0.0) + sem_score
    if not valid_hit:
        # No valid semantic matches, revert to base scores
        for idx in list(scored.keys()):
            if scored[idx] <= -1000:
                scored[idx] = -1000


def _sort_and_select(scored: Dict[int, float]) -> VisionResolution:
    filtered = [(idx, sc) for idx, sc in scored.items() if sc > -999]
    filtered.sort(key=lambda item: item[1], reverse=True)
    if not filtered:
        return VisionResolution(best_index=None, scored=[], top_score=0.0, next_score=None)

    top_idx, top_score = filtered[0]
    next_score = filtered[1][1] if len(filtered) > 1 else None

    # Require a modest margin unless the top score is strong
    if next_score is not None and top_score < max(8.0, next_score + 1.5):
        return VisionResolution(best_index=None, scored=filtered, top_score=top_score, next_score=next_score)
    if top_score < 6:
        return VisionResolution(best_index=None, scored=filtered, top_score=top_score, next_score=next_score)

    return VisionResolution(best_index=top_idx, scored=filtered, top_score=top_score, next_score=next_score)


# ---------------------------------------------------------------------------
# Public API


def resolve_overlays(
    instruction: str,
    element_data: List[Dict[str, Any]],
    *,
    mode: str,
    interpretation_mode: str = "literal",
    semantic_hint: Optional[SemanticTarget] = None,
) -> VisionResolution:
    if not element_data:
        return VisionResolution(best_index=None, scored=[], top_score=0.0, next_score=None)

    profile = _build_instruction_profile(
        instruction,
        mode=mode,
        interpretation_mode=interpretation_mode,
        semantic_hint=semantic_hint,
    )

    scored: Dict[int, float] = {}
    element_map: Dict[int, Dict[str, Any]] = {}

    for element in element_data:
        idx = element.get("index")
        if idx is None:
            continue
        base_score, _, _ = _score_element(element, profile)
        scored[idx] = base_score
        element_map[idx] = element

    if profile.interpretation_mode != "literal" and profile.semantic_target:
        _apply_semantic_target(scored, element_map, profile, profile.semantic_target)

    resolution = _sort_and_select(scored)
    if resolution.best_index is not None:
        details = [f"#{idx}:{score:.2f}" for idx, score in resolution.scored[:5]]
        print(
            f"[VisionResolver][{mode}] best={resolution.best_index} score={resolution.top_score:.2f} candidates={' '.join(details)}"
        )
    else:
        print(
            f"[VisionResolver][{mode}] no confident match. top_candidates={[(idx, round(score, 2)) for idx, score in resolution.scored[:5]]}"
        )
    return resolution


def resolve_click_from_overlays(
    description: str,
    element_data: List[Dict[str, Any]],
    *,
    interpretation_mode: str = "literal",
    semantic_hint: Optional[SemanticTarget] = None,
) -> VisionResolution:
    return resolve_overlays(
        description,
        element_data,
        mode="click",
        interpretation_mode=interpretation_mode,
        semantic_hint=semantic_hint,
    )


def resolve_field_from_overlays(
    description: str,
    element_data: List[Dict[str, Any]],
    *,
    interpretation_mode: str = "literal",
    semantic_hint: Optional[SemanticTarget] = None,
) -> VisionResolution:
    return resolve_overlays(
        description,
        element_data,
        mode="field",
        interpretation_mode=interpretation_mode,
        semantic_hint=semantic_hint,
    )


def resolve_select_from_overlays(
    description: str,
    element_data: List[Dict[str, Any]],
    *,
    interpretation_mode: str = "literal",
    semantic_hint: Optional[SemanticTarget] = None,
) -> VisionResolution:
    return resolve_overlays(
        description,
        element_data,
        mode="select",
        interpretation_mode=interpretation_mode,
        semantic_hint=semantic_hint,
    )


def resolve_datetime_from_overlays(
    description: str,
    element_data: List[Dict[str, Any]],
    *,
    interpretation_mode: str = "literal",
    semantic_hint: Optional[SemanticTarget] = None,
) -> VisionResolution:
    return resolve_overlays(
        description,
        element_data,
        mode="datetime",
        interpretation_mode=interpretation_mode,
        semantic_hint=semantic_hint,
    )


def resolve_upload_from_overlays(
    description: str,
    element_data: List[Dict[str, Any]],
    *,
    interpretation_mode: str = "literal",
    semantic_hint: Optional[SemanticTarget] = None,
) -> VisionResolution:
    return resolve_overlays(
        description,
        element_data,
        mode="upload",
        interpretation_mode=interpretation_mode,
        semantic_hint=semantic_hint,
    )


def score_element_against_instruction(
    instruction: str,
    element: Dict[str, Any],
    *,
    page_w: Optional[int] = None,
    page_h: Optional[int] = None,
    mode: str = "click",
    interpretation_mode: str = "literal",
    semantic_hint: Optional[SemanticTarget] = None,
) -> Tuple[float, Dict[str, Any]]:
    page_w = int(page_w or element.get("pageWidth") or element.get("viewportWidth") or 1000)
    page_h = int(page_h or element.get("pageHeight") or element.get("viewportHeight") or 1000)
    overlay = _coerce_overlay_metadata(element, page_w, page_h)
    profile = _build_instruction_profile(
        instruction,
        mode=mode,
        interpretation_mode=interpretation_mode,
        semantic_hint=semantic_hint,
    )
    base_score, token_hits, hint_hits = _score_element(overlay, profile)
    score = base_score
    diagnostics: Dict[str, Any] = {
        "mode": mode,
        "interpretation": interpretation_mode,
        "tokens": token_hits,
        "hint_terms": hint_hits,
        "element_snapshot": _detail_snapshot(overlay),
        "normalized_area": _normalized_area_ratio(overlay),
    }

    if profile.interpretation_mode != "literal" and profile.semantic_target:
        semantic_component = semantic_score_element(overlay, profile.semantic_target)
        if semantic_component is None:
            score = -1000.0
            diagnostics["semantic_dropped"] = True
        else:
            score += semantic_component
            diagnostics["semantic_score"] = semantic_component

    diagnostics["score"] = score
    return score, diagnostics


def rank_elements_against_instruction(
    instruction: str,
    elements: List[Dict[str, Any]],
    *,
    page_w: Optional[int] = None,
    page_h: Optional[int] = None,
    mode: str = "click",
    interpretation_mode: str = "literal",
    semantic_hint: Optional[SemanticTarget] = None,
) -> List[Tuple[int, float, Dict[str, Any]]]:
    ranked: List[Tuple[int, float, Dict[str, Any]]] = []
    for idx, element in enumerate(elements or []):
        if not element:
            continue
        score, diag = score_element_against_instruction(
            instruction,
            element,
            page_w=page_w,
            page_h=page_h,
            mode=mode,
            interpretation_mode=interpretation_mode,
            semantic_hint=semantic_hint,
        )
        diag["element_index"] = idx
        ranked.append((idx, score, diag))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def visible_from_overlays(
    instruction: str,
    element_data: List[Dict[str, Any]],
    *,
    mode: str = "click",
    interpretation_mode: str = "literal",
    semantic_hint: Optional[SemanticTarget] = None,
    min_score: float = 6.0,
) -> bool:
    resolution = resolve_overlays(
        instruction,
        element_data,
        mode=mode,
        interpretation_mode=interpretation_mode,
        semantic_hint=semantic_hint,
    )
    if not resolution.scored:
        return False
    top_score = resolution.scored[0][1]
    return top_score >= min_score


def dom_text_visible(page: Any, text: str) -> bool:
    if not text:
        return False
    query = text.strip()
    if not query:
        return False
    try:
        locator = page.get_by_text(query, exact=False)
        count = 0
        try:
            count = locator.count()
        except Exception:
            count = 0
        if count:
            return True
        script = """
        (search) => {
            const needle = (search || '').toLowerCase();
            if (!needle) return false;
            const nodes = Array.from(document.querySelectorAll('[aria-label], [placeholder], [title]'));
            return nodes.some(el => {
                const value = (el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.getAttribute('title') || '').toLowerCase();
                return value.includes(needle);
            });
        }
        """
        return bool(page.evaluate(script, query))
    except Exception:
        return False


__all__ = [
    "VisionResolution",
    "resolve_click_from_overlays",
    "resolve_field_from_overlays",
    "resolve_select_from_overlays",
    "resolve_datetime_from_overlays",
    "resolve_upload_from_overlays",
    "resolve_overlays",
    "score_element_against_instruction",
    "rank_elements_against_instruction",
    "visible_from_overlays",
    "dom_text_visible",
]
