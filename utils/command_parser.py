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

def parse_structured_if(command: str) -> Optional[Dict[str, Any]]:
    """
    Parse a structured IF command.
    Format: if: condition -> then_action [-> else_action]
    """
    if not command.lower().startswith("if:"):
        return None
        
    parts = command[3:].split("->")
    if len(parts) < 2:
        return None
        
    condition = parts[0].strip()
    then_action = parts[1].strip()
    else_action = parts[2].strip() if len(parts) > 2 else None
    
    return {
        "type": "if",
        "condition": condition,
        "then": then_action,
        "else": else_action
    }

def parse_structured_while(command: str) -> Optional[Dict[str, Any]]:
    """
    Parse a structured WHILE command.
    Format: while: condition -> action
    """
    if not command.lower().startswith("while:"):
        return None
        
    parts = command[6:].split("->")
    if len(parts) < 2:
        return None
        
    condition = parts[0].strip()
    action = parts[1].strip()
    
    return {
        "type": "while",
        "condition": condition,
        "action": action
    }

def parse_structured_for(command: str) -> Optional[Dict[str, Any]]:
    """
    Parse a structured FOR command.
    Format: for: item in list -> action
    """
    if not command.lower().startswith("for:"):
        return None
        
    parts = command[4:].split("->")
    if len(parts) < 2:
        return None
        
    loop_def = parts[0].strip()
    action = parts[1].strip()
    
    # Parse "item in list"
    loop_match = re.match(r"(\w+)\s+in\s+(.+)", loop_def)
    if not loop_match:
        return None
        
    variable = loop_match.group(1)
    source = loop_match.group(2)
    
    return {
        "type": "for",
        "variable": variable,
        "source": source,
        "action": action
    }

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

def parse_focus_command(command: str) -> Optional[str]:
    """Parse a focus command - Focus system removed, always returns None."""
    return None
        return command[6:].strip()
    return None

def parse_undo_command(command: str) -> bool:
    """Check if command is an undo request - Focus system removed, always returns False."""
    return False
    return command.lower().strip() == "undo"

def extract_press_target(description: str) -> Optional[str]:
    """Extract key to press from description."""
    match = re.search(r"(?:press|hit)\s+(?:the\s+)?['\"]?([a-zA-Z0-9_+\s]+)['\"]?", description.lower())
    if match:
        return match.group(1)
    return None
