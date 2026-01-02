"""
Intent parsing helpers extracted from BrowserVisionBot.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

from models import ActionIntent


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




def parse_keyword_command(text: str) -> Optional[tuple[str, str, Optional[str]]]:
    """Parse a simple keyword command of the form "keyword: payload[: helper]".

    Supported keywords (case-insensitive):
      - click, scroll, press, navigate, open, visit, form, fill, back, forward, ref,
        type, select, upload, datetime

    Returns (keyword_lower, payload_stripped, helper_or_None) or None if not matched.
    """
    t = (text or "").strip()
    if not t:
        return None
    # Accept forms like "click: payload" or "click action: payload"
    # Allow underscores and hyphens in keywords (e.g., "defer_input", "sub-focus")
    m = re.match(r"^\s*([a-z][a-z0-9_\-]*)(?:\s+action)?\s*:\s*(.*)$", t, flags=re.IGNORECASE)
    if m:
        kw = m.group(1).lower().replace("-", "_")
        payload = (m.group(2) or "").strip()
    else:
        # Allow bare keywords for defer so commands like "defer" or "defer take over" work.
        m_defer = re.match(r"^\s*(defer(?:-input)?)(?:\s+(.*))?$", t, flags=re.IGNORECASE)
        if not m_defer:
            return None
        kw = m_defer.group(1).lower().replace("-", "_")
        payload = (m_defer.group(2) or "").strip()
    # Normalize synonyms
    alias = {
        "open": "navigate",
        "visit": "navigate",
        "fill": "form",
    }
    kw = alias.get(kw, kw)

    supported = {
        "click",
        "scroll",
        "press",
        "navigate",
        "form",
        "back",
        "forward",
        "ref",
        "type",
        "select",
        "upload",
        "datetime",
        "defer",
        "defer_input",
    }
    if kw not in supported:
        return None

    command_text = payload
    helper_text: Optional[str] = None
    if kw in {"click", "press", "scroll", "navigate", "back", "forward", "ref"}:
        command_text, helper_text = _split_command_helper(payload)

    return kw, command_text, helper_text


def parse_type_command(text: str) -> Optional[tuple[str, Optional[str]]]:
    """Parse a type command into (value, target) where target may be None.

    Supported forms:
    - "type: <value> into <target>"
    - "type: <value> in <target>"
    - "type: \"<value>\" into email"
    - "type: '<value>' into search field"
    - "type: <value>" (no target -> current focus or generic textbox)
    """
    if not text:
        return None
    t = text.strip()
    m = re.match(r"(?i)^\s*type\s*:\s*(.+)$", t)
    if not m:
        return None
    rest = m.group(1).strip()

    # Split on into/in - look for the LAST occurrence of preposition + target pattern
    # Find all matches of "into/in + target" and take the last one
    # Use greedy matching to capture full target phrase (e.g., "text input field 'What...'")
    matches = list(re.finditer(r"\s+(into|in)\s*:?\s*(.+?)(?:\s*$)", rest, flags=re.IGNORECASE))

    if matches:
        # Take the last match (closest to end)
        last_match = matches[-1]
        preposition_pos = last_match.start()

        # Check if this looks like a valid target (must contain target-like words)
        potential_target = last_match.group(2).strip()
        target_keywords = ['textarea', 'input', 'field', 'box', 'area', 'element', 'search', 'form', 'login', 'email', 'password', 'name', 'button', 'textbox']

        # Target must contain at least one target keyword to be valid
        if any(keyword in potential_target.lower() for keyword in target_keywords):
            raw_val = rest[:preposition_pos].strip()
            target = potential_target
        else:
            # Doesn't look like a field target, treat entire string as value
            raw_val = rest
            target = None
    else:
        # No preposition found
        raw_val = rest
        target = None

    # Strip quotes
    if (raw_val.startswith("'") and raw_val.endswith("'")) or (raw_val.startswith('"') and raw_val.endswith('"')):
        raw_val = raw_val[1:-1]
    return raw_val, target


def _parse_value_target(command: str, keyword: str) -> Optional[tuple[str, Optional[str]]]:
    """Generic parser for commands like: keyword: <value> [into|in|to] <target>."""
    if not command:
        return None
    t = command.strip()
    m = re.match(rf"(?i)^\s*{keyword}\s*:\s*(.+)$", t)
    if not m:
        return None
    rest = m.group(1).strip()
    m2 = re.match(r"^(.*?)(?:\s+(?:into|in|to)\s*:?\s*(.+))?$", rest, flags=re.IGNORECASE)
    if not m2:
        return None
    raw_val = (m2.group(1) or "").strip()
    target = (m2.group(2) or "").strip() or None
    if (raw_val.startswith("'") and raw_val.endswith("'")) or (raw_val.startswith('"') and raw_val.endswith('"')):
        raw_val = raw_val[1:-1]
    return raw_val, target


def parse_select_command(text: str) -> Optional[tuple[str, Optional[str]]]:
    result = _parse_value_target(text, "select")
    if not result:
        return None
    
    value, target = result
    
    # Strip common suffixes that aren't part of the actual option value
    if value:
        # Remove trailing "option", "choice", "item" etc.
        value = re.sub(r'\s+(option|choice|item|selection)$', '', value, flags=re.IGNORECASE).strip()
    
    return value, target


def parse_upload_command(text: str) -> Optional[tuple[str, Optional[str]]]:
    return _parse_value_target(text, "upload")


def parse_datetime_command(text: str) -> Optional[tuple[str, Optional[str]]]:
    return _parse_value_target(text, "datetime")






# ---------------------------------------------------------------------------
# Slot-based Action Intent parsing

_STOPWORDS = {
    "the",
    "a",
    "an",
    "to",
    "for",
    "of",
    "in",
    "on",
    "at",
    "into",
    "please",
    "button",
    "link",
    "icon",
    "item",
    "field",
    "box",
    "option",
    "menu",
    "list",
    "tab",
    "row",
    "column",
}


def parse_action_intent(command: str) -> Optional[ActionIntent]:
    """Convert a natural-language command into a structured ActionIntent."""
    if not command:
        return None
    text = command.strip()
    if not text:
        return None

    action = _detect_action(text)
    if not action:
        return None

    if action == "type":
        return _build_type_intent(text)
    if action == "select":
        return _build_value_target_intent(text, action)
    if action == "upload":
        return _build_value_target_intent(text, action)
    if action == "datetime":
        return _build_value_target_intent(text, action)
    if action == "click":
        return _build_click_intent(text)
    if action in {"press", "scroll"}:
        kw_cmd = parse_keyword_command(text)
        helper_text = kw_cmd[2] if kw_cmd and kw_cmd[0] == action else None
        return ActionIntent(action=action, raw_command=command, helper_text=helper_text)

    return None


def _detect_action(text: str) -> Optional[str]:
    lowered = text.lower().strip()
    kw_cmd = parse_keyword_command(text)
    if kw_cmd:
        return kw_cmd[0]

    for action in ("click", "tap", "press", "type", "select", "upload", "datetime", "scroll"):
        if lowered.startswith(action + " ") or lowered == action:
            if action == "tap":
                return "click"
            return action
        # Allow imperative like "please click" etc.
        if lowered.startswith(f"please {action}"):
            if action == "tap":
                return "click"
            return action
    return None


def _build_click_intent(text: str) -> ActionIntent:
    helper_text = None
    kw_cmd = parse_keyword_command(text)
    if kw_cmd and kw_cmd[0] == "click":
        payload = kw_cmd[1]
        helper_text = kw_cmd[2]
    else:
        payload = _extract_payload(text, "click")
    lowered = payload.lower()

    modifiers: Dict[str, str] = {}
    ordinal = _extract_ordinal(lowered)
    if ordinal is not None:
        modifiers["ordinal"] = str(ordinal)
        lowered = _remove_phrase(lowered, _ordinal_patterns())

    role_hint = _extract_role_hint(lowered)
    if role_hint:
        lowered = _remove_phrase(lowered, ROLE_KEYWORDS[role_hint])

    collection_hint = _extract_collection_hint(lowered)
    if collection_hint:
        lowered = _remove_phrase(lowered, COLLECTION_HINTS[collection_hint])

    target_text, attribute_filters = _extract_target_text(payload)

    if not target_text:
        tokens = [t for t in re.split(r"[^a-z0-9]+", lowered) if t and t not in _STOPWORDS]
        if tokens:
            target_text = " ".join(tokens)

    return ActionIntent(
        action="click",
        raw_command=text,
        target_text=target_text or None,
        role_hint=role_hint,
        collection_hint=collection_hint,
        modifiers=modifiers,
        attribute_filters=attribute_filters,
        helper_text=helper_text,
    )


def _build_type_intent(text: str) -> ActionIntent:
    kw_cmd = parse_keyword_command(text)
    helper_text = kw_cmd[2] if kw_cmd and kw_cmd[0] == "type" else None

    parsed = parse_type_command(text)
    value: Optional[str] = None
    target: Optional[str] = None
    if parsed:
        value, target = parsed
    else:
        # fallback manual parse e.g. "type hello in the search box"
        m = re.match(r"(?i)^\s*type\s+(?:the\s+)?(.+?)\s+(?:into|in|to)\s+(.+)$", text.strip())
        if m:
            value = m.group(1).strip().strip("'\"")
            target = m.group(2).strip()
        else:
            m = re.match(r"(?i)^\s*type\s+(.+)$", text.strip())
            if m:
                value = m.group(1).strip().strip("'\"")

    modifiers: Dict[str, str] = {}
    role_hint = "textbox"
    collection_hint = None
    attribute_filters: Dict[str, str] = {}
    target_text, attr_filters = _extract_target_text(target or "")
    if attr_filters:
        attribute_filters.update(attr_filters)

    return ActionIntent(
        action="type",
        raw_command=text,
        target_text=target_text or target,
        role_hint=role_hint,
        value=value,
        modifiers=modifiers,
        collection_hint=collection_hint,
        attribute_filters=attribute_filters,
        helper_text=helper_text,
    )


def _build_value_target_intent(text: str, action: str) -> ActionIntent:
    parser = {
        "select": parse_select_command,
        "upload": parse_upload_command,
        "datetime": parse_datetime_command,
    }[action]
    parsed = parser(text)
    value, target = parsed if parsed else (None, None)
    # Normalize empty strings to None so downstream handlers can detect missing values
    value = value or None
    target = target or None

    kw_cmd = parse_keyword_command(text)
    helper_text = kw_cmd[2] if kw_cmd and kw_cmd[0] == action else None

    role_hint = None
    if action == "select":
        role_hint = "combobox"
    elif action == "upload":
        role_hint = "textbox"
    elif action == "datetime":
        role_hint = "textbox"

    target_text, attribute_filters = _extract_target_text(target or "")

    return ActionIntent(
        action=action,
        raw_command=text,
        target_text=target_text or target,
        role_hint=role_hint,
        value=value,
        attribute_filters=attribute_filters,
        helper_text=helper_text,
    )


def _extract_payload(text: str, keyword: str) -> str:
    kw_cmd = parse_keyword_command(text)
    if kw_cmd and kw_cmd[0] == keyword:
        return kw_cmd[1]
    lowered = text.lower().lstrip()
    if lowered.startswith(keyword):
        return text[len(keyword) :].strip(" :")
    synonyms = {"tap": "click"}
    for syn, canonical in synonyms.items():
        if canonical == keyword and lowered.startswith(syn):
            return text[len(syn) :].strip(" :")
    return text


def _split_command_helper(payload: str) -> tuple[str, Optional[str]]:
    if not payload:
        return "", None

    parts = re.split(r"\s*:\s+", payload, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return parts[0].strip(), parts[1].strip()

    return payload.strip(), None


def _extract_ordinal(text: str) -> Optional[int]:
    for word, idx in ORDINAL_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return idx
    match = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", text)
    if match:
        return max(int(match.group(1)) - 1, 0)
    return None


def _ordinal_patterns():
    patterns = set(ORDINAL_WORDS.keys())
    patterns.update({"1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"})
    return patterns


def _extract_role_hint(text: str) -> Optional[str]:
    for role, keywords in ROLE_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return role
    return None


def _extract_collection_hint(text: str) -> Optional[str]:
    for collection, keywords in COLLECTION_HINTS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return collection
    return None


def _extract_target_text(text: str) -> tuple[Optional[str], Dict[str, str]]:
    attribute_filters: Dict[str, str] = {}
    if not text:
        return None, attribute_filters

    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
    if quoted:
        for q1, q2 in quoted:
            val = q1 or q2
            attribute_filters.setdefault("aria-label", val)
        return quoted[0][0] or quoted[0][1], attribute_filters

    cleaned = text.strip()
    if cleaned:
        return cleaned, attribute_filters

    return None, attribute_filters


def _remove_phrase(text: str, keywords) -> str:
    if not text:
        return text
    result = text
    for kw in keywords:
        result = re.sub(rf"\b{re.escape(kw)}\b", " ", result)
    result = re.sub(r"\s+", " ", result)
    return result.strip()









ORDINAL_WORDS: Dict[str, int] = {
    "first": 0,
    "second": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,
    "sixth": 5,
    "seventh": 6,
    "eighth": 7,
    "ninth": 8,
    "tenth": 9,
    "last": -1,
}

ROLE_KEYWORDS = {
    "button": {"button", "btn", "cta"},
    "link": {"link", "anchor"},
    "checkbox": {"checkbox", "check box"},
    "radio": {"radio", "radio button"},
    "tab": {"tab"},
    "combobox": {"dropdown", "combo", "combobox", "listbox"},
    "textbox": {"textbox", "text box", "input", "field", "search box"},
}

COLLECTION_HINTS = {
    "list": {"list", "listing"},
    "menu": {"menu"},
    "table": {"table", "grid"},
    "column": {"column", "col"},
    "row": {"row"},
    "section": {"section", "panel"},
}
