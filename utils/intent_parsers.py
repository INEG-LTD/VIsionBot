"""
Intent parsing helpers extracted from Browser.
"""
from __future__ import annotations

import re
from typing import Optional

def parse_keyword_command(text: str):
    t = re.sub(r"\s*\n\s*", " ", (text or "").strip())
    if not t:
        return None

    if ":" in t:
        kw, rest = t.split(":", 1)
        kw = re.sub(r"\s+action\s*$", "", kw, flags=re.I).strip()
        if not kw:
            return None
    else:
        kw, _, rest = t.partition(" ")

    cmd, helper = _split_command_helper(rest.strip())
    return kw.lower().replace("-", "_"), cmd, helper

def _split_command_helper(payload: str) -> tuple[str, Optional[str]]:
    if not payload:
        return "", None

    parts = re.split(r"\s*:\s+", payload, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return parts[0].strip(), parts[1].strip()

    return payload.strip(), None
