"""
Text processing utilities for the vision bot.
"""

import re
from typing import List, Optional


class TextUtils:
    """Utility class for text processing operations."""
    
    _ORD = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, 
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }

    @staticmethod
    def dedupe(seq: List[str]) -> List[str]:
        """Remove duplicate strings while preserving order."""
        seen = set()
        out = []
        for item in seq:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract URLs from text using regex."""
        return re.findall(r"https?://[^\s)]+", text or "", flags=re.I)

    @classmethod
    def to_num(cls, tok: str) -> Optional[int]:
        """Convert text tokens to numbers (including spelled out numbers)."""
        tok = tok.lower()
        if tok.isdigit():
            return int(tok)
        return cls._ORD.get(tok)

    @staticmethod
    def slugify(bits: List[str]) -> str:
        """Convert text to URL-friendly slug."""
        return "-".join(b for b in re.split(r"[\s_/]+", " ".join(bits).strip().lower()) if b)

    @staticmethod
    def looks_like_navigation(low: str) -> bool:
        """Detect if text indicates navigation intent."""
        return any(w in low for w in ["navigate to", "go to", "open ", "visit ", "view ", "go back to", "reach "])

    @staticmethod
    def looks_like_form_fill(low: str) -> bool:
        """Detect if text indicates form filling intent."""
        return any(w in low for w in ["fill", "type", "enter", "complete", "select", "choose", "upload", "attach"]) and "login" not in low

    @staticmethod
    def looks_like_login(low: str) -> bool:
        """Detect if text indicates login intent."""
        return any(w in low for w in ["log in", "login", "sign in", "authenticate"]) or all(w in low for w in ["username", "password"])

    @staticmethod
    def looks_like_submit(low: str) -> bool:
        """Detect if text indicates submission intent."""
        return any(w in low for w in ["submit", "send application", "place order", "checkout", "pay now", "finish", "confirm"])

    @staticmethod
    def guess_cta_preferences(low: str) -> List[str]:
        """Guess preferred call-to-action buttons based on context."""
        prefs = ["Next", "Continue"]
        if any(w in low for w in ["apply", "application"]):
            prefs = ["Apply", "Apply now"] + prefs
        if any(w in low for w in ["checkout", "purchase", "buy"]):
            prefs = ["Checkout", "Buy now", "Purchase"] + prefs
        if any(w in low for w in ["login", "sign in"]):
            prefs = ["Login", "Sign in", "Sign In"] + prefs
        return prefs

    @staticmethod
    def is_navigation_intent(text: str) -> bool:
        """Check if text indicates navigation intent."""
        low = text.lower()
        return TextUtils.looks_like_navigation(low)

    @staticmethod
    def quote(s: str) -> str:
        """Add quotes around a string if it doesn't already have them."""
        if not s:
            return '""'
        if s.startswith('"') and s.endswith('"'):
            return s
        return f'"{s}"'

    @staticmethod
    def canon(s: str) -> str:
        """Canonicalize text for comparison."""
        return re.sub(r'\s+', ' ', s.strip().lower())
