"""
Utility helpers for describing overlay elements so we can match them across iterations.
"""
from typing import Any, Dict, Optional


def describe_overlay_element(element: Dict[str, Any]) -> str:
    """Return a canonical, human-readable description for the overlay entry."""
    if not element:
        return ""

    parts = []
    idx = element.get("index")
    tag = (element.get("tagName") or "").lower().strip() or "element"
    parts.append(f"#{idx or '?'} tag={tag}")

    role = (element.get("role") or "").lower().strip()
    if role and role != tag:
        parts.append(f"role={role}")

    elem_type = (element.get("type") or "").lower().strip()
    if elem_type and elem_type not in (tag, role):
        parts.append(f"type={elem_type}")

    placeholder = (element.get("placeholder") or "").strip()
    if placeholder:
        parts.append(f'placeholder="{placeholder[:40]}"')

    text = (element.get("text") or element.get("textContent") or element.get("description") or "").strip()
    if text:
        parts.append(f'txt="{text[:60]}"')
    else:
        aria = (element.get("ariaLabel") or element.get("aria_label") or "").strip()
        if aria:
            parts.append(f'aria="{aria[:60]}"')

    return " ".join(parts).strip()


def overlay_element_metadata(element: Dict[str, Any]) -> dict:
    """Return normalized metadata for matching overlay elements across iterations."""
    def _norm(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    return {
        "tag": _norm(element.get("tagName")),
        "role": _norm(element.get("role")),
        "type": _norm(element.get("type")),
        "name": _norm(element.get("name")),
        "placeholder": _norm(element.get("placeholder")),
        "aria": _norm(element.get("ariaLabel") or element.get("aria_label")),
        "text": _norm(element.get("text") or element.get("textContent") or element.get("description")),
    }


def metadata_matches(target: dict, element: Dict[str, Any]) -> bool:
    """Compare normalized metadata against an overlay entry for equality."""
    if not target:
        return False
    overlay_meta = overlay_element_metadata(element)
    for key, value in target.items():
        if not value:
            continue
        if overlay_meta.get(key) != value:
            return False
    return True
