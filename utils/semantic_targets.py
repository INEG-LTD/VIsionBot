"""Semantic target interpretation for natural-language UI commands."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
import re

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class SemanticTarget:
    """Structured interpretation of an element description."""

    description: str
    role: Optional[str] = None
    primary_terms: List[str] = field(default_factory=list)
    required_terms: List[str] = field(default_factory=list)
    context_terms: List[str] = field(default_factory=list)
    forbidden_terms: List[str] = field(default_factory=list)

    def all_terms(self) -> List[str]:
        terms = list(dict.fromkeys([
            *(self.required_terms or []),
            *(self.primary_terms or []),
            *(self.context_terms or []),
        ]))
        return [t for t in terms if t]

    def candidate_phrases(self) -> List[str]:
        """Generate alternative phrases for DOM fallback lookups."""
        phrases: List[str] = []
        base = self.description.strip()
        if base:
            phrases.append(base)

        tokens = _tokenize(self.description)
        if tokens:
            phrases.append(" ".join(tokens))

        for term in self.primary_terms:
            if term:
                phrases.append(term)
                if self.role:
                    phrases.append(f"{term} {self.role}")

        for term in self.required_terms:
            if term:
                phrases.append(term)
                if self.role:
                    phrases.append(f"{term} {self.role}")

        if self.role:
            phrases.append(self.role)

        return _unique(phrases)


class _InterpretationModel(BaseModel):
    """Pydantic model for LLM-generated interpretations."""

    role: Optional[str] = None
    must_terms: List[str] = Field(default_factory=list)
    preferred_terms: List[str] = Field(default_factory=list)
    context_terms: List[str] = Field(default_factory=list)
    avoid_terms: List[str] = Field(default_factory=list)


_STOPWORDS: Set[str] = {
    "click",
    "press",
    "tap",
    "double",
    "single",
    "button",
    "link",
    "field",
    "textbox",
    "input",
    "the",
    "a",
    "an",
    "to",
    "on",
    "in",
    "at",
    "of",
    "and",
    "for",
    "item",
    "element",
    "target",
    "select",
    "choose",
    "pick",
    "option",
    "between",
}


def _tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", text or "").strip().lower()
    tokens = [t for t in cleaned.split() if t and t not in _STOPWORDS]
    return tokens


def _unique(strings: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in strings:
        if not item:
            continue
        norm = item.strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            ordered.append(item.strip())
    return ordered


def _extract_role(tokens: Sequence[str]) -> Optional[str]:
    if any(t in {"button", "btn"} for t in tokens):
        return "button"
    if any(t in {"link", "anchor"} for t in tokens):
        return "link"
    if any(t in {"textbox", "input", "field"} for t in tokens):
        return "textbox"
    if any(t in {"dropdown", "combobox", "select"} for t in tokens):
        return "combobox"
    return None


def _call_semantic_model(description: str) -> Optional[_InterpretationModel]:
    try:
        from ai_utils import generate_model  # Lazy import to avoid cycles
    except Exception:
        return None

    system_prompt = (
        "You convert natural-language UI commands into structured search hints. "
        "Return concise lowercase phrases. Do not include explanations."
    )
    user_prompt = f"""
Command: "{description}"
Provide a JSON object with keys:
- role: optional element role (button, link, checkbox, textbox, etc.)
- must_terms: list of short phrases that must appear for correctness
- preferred_terms: list of phrases strongly associated with the target
- context_terms: cues that often appear near the target
- avoid_terms: words indicating the wrong element (e.g., 'remove', 'cancel', 'filter')
Use lowercase strings without punctuation. Keep lists small (<=5 items each).
"""
    try:
        interpretation = generate_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model_object_type=_InterpretationModel,
            reasoning_level="low",
        )
        return interpretation
    except Exception:
        return None


def build_semantic_target(description: str) -> Optional[SemanticTarget]:
    description = (description or "").strip()
    if not description:
        return None

    tokens = _tokenize(description)
    if not tokens:
        return None

    base_role = _extract_role(tokens)
    base_required: Set[str] = set()
    base_primary: Set[str] = set(tokens)

    interpretation = _call_semantic_model(description)

    context_terms: List[str] = []
    forbidden_terms: List[str] = []

    if interpretation:
        if interpretation.role:
            core_role = interpretation.role.strip().lower()
            base_role = core_role if core_role else base_role
        if interpretation.must_terms:
            base_required.update(interpretation.must_terms)
        if interpretation.preferred_terms:
            base_primary.update(interpretation.preferred_terms)
        context_terms = interpretation.context_terms or []
        forbidden_terms = interpretation.avoid_terms or []

    primary_terms = _unique(base_primary)
    required_terms = _unique(base_required)
    context_terms = _unique(context_terms or [])
    forbidden_terms = _unique(forbidden_terms or [])

    target = SemanticTarget(
        description=description,
        role=base_role,
        primary_terms=primary_terms,
        required_terms=required_terms,
        context_terms=context_terms,
        forbidden_terms=forbidden_terms,
    )

    if not (target.primary_terms or target.required_terms or target.context_terms):
        return None
    return target


def semantic_score_element(element: Dict[str, Any], target: SemanticTarget) -> Optional[int]:
    """Return a semantic score or None if the element violates the target."""

    def contains(text: str, phrase: str) -> bool:
        text_norm = text.lower()
        phrase_norm = phrase.lower().strip()
        if not text_norm or not phrase_norm:
            return False
        return phrase_norm in text_norm

    accessible_parts = [
        str(element.get("ariaLabel", "")),
        str(element.get("textContent", "")),
        str(element.get("description", "")),
    ]
    accessible = " ".join(p for p in accessible_parts if p).strip().lower()
    context = str(element.get("contextText", "")).strip().lower()

    combined = f"{accessible} {context}".strip()

    for forbidden in target.forbidden_terms:
        if contains(combined, forbidden):
            return None

    for term in target.required_terms:
        if term and not contains(combined, term):
            return None

    score = 0
    matched_primary = 0
    for term in target.primary_terms:
        if contains(accessible, term):
            score += 6
            matched_primary += 1
        elif contains(context, term):
            score += 3

    if target.primary_terms and target.required_terms:
        if matched_primary == 0 and all(not contains(accessible, req) for req in target.required_terms):
            return None
    elif target.primary_terms and matched_primary == 0:
        return None

    for term in target.context_terms:
        if contains(context, term):
            score += 2
        elif contains(accessible, term):
            score += 1

    element_role = (str(element.get("role")) or str(element.get("tagName")) or "").lower()
    if target.role and target.role in element_role:
        score += 4

    # Add proportional score based on term coverage
    total_terms = len(target.primary_terms) or len(target.required_terms)
    if total_terms:
        coverage = matched_primary / max(1, len(target.primary_terms))
        score += int(coverage * 4)

    return score if score > 0 else None


__all__ = ["SemanticTarget", "build_semantic_target", "semantic_score_element"]
