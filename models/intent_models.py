"""Structured intent models for action planning."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ActionIntent(BaseModel):
    """Structured representation of a user instruction."""

    action: str = Field(description="Verb describing the action (click, type, select, upload, datetime, press, scroll)")
    raw_command: str = Field(description="Original user instruction")

    target_text: Optional[str] = Field(default=None, description="Literal text describing the target element")
    role_hint: Optional[str] = Field(default=None, description="Semantic hint for the element role (button, link, field, etc.)")
    value: Optional[str] = Field(default=None, description="Value associated with the action (text to type, option to select, file path, etc.)")
    helper_text: Optional[str] = Field(default=None, description="Additional contextual helper provided with the command")

    modifiers: Dict[str, str] = Field(default_factory=dict, description="Slot modifiers such as ordinal, region, direction")
    collection_hint: Optional[str] = Field(default=None, description="Grouping hint (list, table, menu, etc.)")
    attribute_filters: Dict[str, str] = Field(default_factory=dict, description="Attribute equals filters (e.g., aria-label)")

    def ordinal(self) -> Optional[int]:
        """Return zero-based ordinal index if available (first -> 0, last -> -1)."""
        raw = self.modifiers.get("ordinal")
        if raw is None:
            return None
        try:
            idx = int(raw)
            return idx
        except ValueError:
            return None

    def normalized_terms(self) -> List[str]:
        """Return key terms for semantic scoring."""
        terms: List[str] = []
        if self.target_text:
            terms.extend(_tokenize(self.target_text))
        if self.role_hint:
            terms.append(self.role_hint.lower())
        for value in self.attribute_filters.values():
            terms.extend(_tokenize(value))
        return list(dict.fromkeys(filter(None, terms)))

    def subject_hint(self) -> Optional[str]:
        if self.role_hint:
            return self.role_hint
        if self.target_text:
            tokens = _tokenize(self.target_text)
            if tokens:
                return tokens[0]
        return None


def _tokenize(text: str) -> List[str]:
    import re

    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t]


__all__ = ["ActionIntent"]
