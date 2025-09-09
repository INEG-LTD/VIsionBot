"""
Intent parsing helpers extracted from BrowserVisionBot.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple


def parse_while_statement(goal_description: str) -> tuple[Optional[str], Optional[str]]:
    """Parse a simple while-style instruction.

    Supports forms like:
    - "while CONDITION: ACTION"
    - "do ACTION until CONDITION"
    - "ACTION until CONDITION"
    Returns (condition_text, loop_goal_description).
    """
    text = (goal_description or "").strip()
    if not text:
        return None, None

    # while X: do Y
    m = re.match(r"\s*while\s+(.+?):\s*(.+)$", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # do Y until X  OR  Y until X
    m = re.match(r"\s*(?:do\s+)?(.+?)\s+until\s+(.+)$", text, flags=re.IGNORECASE)
    if m:
        return m.group(2).strip(), m.group(1).strip()

    return None, None


def extract_click_target(goal_description: str) -> Optional[str]:
    """Extract a human-friendly click target description from a goal description."""
    goal_lower = (goal_description or "").lower().strip()
    # Patterns like: click the 'submit' button, click 'OK', click login, close modal, select X, choose X
    m = re.search(r"click(?:\s+the)?\s+'([^']+)'", goal_lower)
    if m:
        return m.group(1)
    m = re.search(r"click(?:\s+the)?\s+\"([^\"]+)\"", goal_lower)
    if m:
        return m.group(1)
    m = re.search(r"click(?:\s+the)?\s+([\w\s\-]+?)(?:\s+button|\s+link|\s+icon)?$", goal_lower)
    if m:
        return m.group(1).strip()
    # close/choose/select variants
    m = re.search(r"(?:close|select|choose)\s+(?:the\s+)?([\w\s\-]+)$", goal_lower)
    if m:
        return m.group(1).strip()
    return None


def extract_press_target(goal_description: str) -> Optional[str]:
    """Extract a key or key-combo from a description like 'press enter' or 'press ctrl+c'."""
    txt = (goal_description or "").lower()
    # Simple keys
    simple_keys = [
        "enter", "return", "tab", "escape", "esc", "space", "backspace",
        "arrowup", "arrowdown", "arrowleft", "arrowright",
    ]
    m = re.search(r"press\s+([a-z\+]+)", txt)
    if m:
        key = m.group(1).strip()
        # Normalize common synonyms
        if key == "esc":
            key = "escape"
        if key in simple_keys:
            return key
        # Key combination like ctrl+c, cmd+enter, alt+tab
        cm = re.match(r"(ctrl|cmd|alt|shift)\+([a-z]+)", key)
        if cm:
            return f"{cm.group(1)}+{cm.group(2)}"
    return None


def extract_navigation_intent(goal_description: str) -> Optional[str]:
    """Extract a navigation intent phrase from text: e.g., 'go to pricing' -> 'pricing'."""
    patterns = [
        r"go to (?:the )?(.+)",
        r"navigate to (?:the )?(.+)",
        r"open (?:the )?(.+)",
        r"visit (?:the )?(.+)",
    ]
    goal_lower = (goal_description or "").lower().strip()
    for pat in patterns:
        m = re.search(pat, goal_lower)
        if m:
            intent = m.group(1).strip()
            # Normalize trailing descriptors like "page", "section"
            intent = re.sub(r"\s+(page|section|area)$", r" \1", intent)
            return intent
    return None


# ---------------- Structured Goal Syntax Parsers ----------------

def parse_structured_if(text: str) -> Optional[tuple[str, str, Optional[str]]]:
    """Parse explicit IF/THEN[/ELSE] syntax.

    Supported forms (case-insensitive, flexible spacing):
    - "if: <condition> then: <success> else: <fail>"
    - "if <condition> then <success> else <fail>"
    - without else: "if: <condition> then: <success>"
    Returns (condition, success_action, fail_action_or_None) or None if no match.
    """
    t = (text or "").strip()
    if not t:
        return None
    # Normalize whitespace
    t1 = re.sub(r"\s+", " ", t).strip()
    # Regex with optional colons after keywords
    m = re.match(r"(?i)^if\s*: ?(.+?)\s+then\s*: ?(.+?)(?:\s+else\s*: ?(.+))?$", t1)
    if m:
        cond = m.group(1).strip()
        then = m.group(2).strip()
        els = m.group(3).strip() if m.group(3) else None
        return cond, then, els
    return None


def parse_keyword_command(text: str) -> Optional[tuple[str, str]]:
    """Parse a simple keyword command of the form "keyword: payload".

    Supported keywords (case-insensitive):
      - click, scroll, press, navigate, open, visit, form, fill, back, forward

    Returns (keyword_lower, payload_stripped) or None if not matched.
    """
    t = (text or "").strip()
    if not t:
        return None
    m = re.match(r"^\s*([a-z]+)\s*:\s*(.*)$", t, flags=re.IGNORECASE)
    if not m:
        return None
    kw = m.group(1).lower()
    payload = (m.group(2) or "").strip()
    # Normalize synonyms
    alias = {
        "open": "navigate",
        "visit": "navigate",
        "fill": "form",
    }
    kw = alias.get(kw, kw)
    if kw in {"click", "scroll", "press", "navigate", "form", "back", "forward"}:
        return kw, payload
    return None


def parse_structured_while(text: str) -> Optional[tuple[str, str]]:
    """Parse explicit WHILE/DO or DO/UNTIL syntax.

    Supported forms (case-insensitive, flexible spacing):
    - "while: <condition> do: <body>"
    - "while <condition> do <body>"
    - "do: <body> until: <condition>" (or "repeat: <body> until: <condition>")
    Returns (condition, body) or None if no match.
    """
    t = (text or "").strip()
    if not t:
        return None
    t1 = re.sub(r"\s+", " ", t).strip()
    m = re.match(r"(?i)^while\s*: ?(.+?)\s+do\s*: ?(.+)$", t1)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.match(r"(?i)^(?:do|repeat)\s*: ?(.+?)\s+until\s*: ?(.+)$", t1)
    if m:
        body = m.group(1).strip()
        cond = m.group(2).strip()
        return cond, body
    return None
