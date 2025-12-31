"""
Command parsing utilities for interpreting user actions.
"""
import re
from typing import Dict, Any, Optional, Tuple

def parse_action_intent(action_description: str) -> str:
    """
    Parse the intent of an action description.
    
    Args:
        action_description: The natural language description of the action
        
    Returns:
        The inferred intent (click, type, scroll, etc.)
    """
    desc = action_description.lower().strip()
    
    if desc.startswith("click") or desc.startswith("tap"):
        return "click"
    elif desc.startswith("type") or desc.startswith("enter") or desc.startswith("fill"):
        return "type"
    elif desc.startswith("scroll"):
        return "scroll"
    elif desc.startswith("press") or desc.startswith("hit"):
        return "press"
    elif desc.startswith("select") or desc.startswith("choose"):
        return "select"
    elif desc.startswith("upload") or desc.startswith("attach"):
        return "upload"
    elif desc.startswith("wait") or desc.startswith("sleep"):
        return "wait"
    elif desc.startswith("hover"):
        return "hover"
    elif desc.startswith("navigate") or desc.startswith("go to"):
        return "navigate"
        
    return "unknown"


def parse_keyword_command(command: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a keyword command (e.g., "click: button").
    
    Returns:
        Tuple of (keyword, argument) or (None, None)
    """
    match = re.match(r"^(\w+):\s*(.+)$", command)
    if match:
        return match.group(1).lower(), match.group(2).strip()
    return None, None


