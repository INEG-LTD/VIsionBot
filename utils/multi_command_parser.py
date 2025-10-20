"""
Multi-command parsing utilities for handling multiple commands within goal bodies.
"""
from __future__ import annotations

import re
from typing import List, Tuple, Optional
from ai_utils import generate_text


def parse_multiple_commands(prompt: str) -> List[str]:
    """
    Parse a single prompt that may contain multiple commands and return a list of individual commands.
    
    This function uses AI to intelligently split complex prompts into individual commands
    while preserving the intent and context of each command.
    
    Args:
        prompt: The input prompt that may contain multiple commands
        
    Returns:
        List of individual command strings
    """
    if not prompt or not prompt.strip():
        return []
    
    # First, try simple rule-based parsing for obvious cases
    simple_commands = _try_simple_parsing(prompt)
    if simple_commands:
        return simple_commands
    
    # If simple parsing fails, use AI to intelligently split the prompt
    return _ai_parse_commands(prompt)


def _try_simple_parsing(prompt: str) -> Optional[List[str]]:
    """
    Try to parse commands using simple rules for obvious cases.
    
    Returns:
        List of commands if parsing succeeds, None if AI parsing is needed
    """
    prompt = prompt.strip()
    
    # Case 1: Commands separated by newlines (common in while loops)
    if '\n' in prompt and _has_command_keywords(prompt):
        commands = [cmd.strip() for cmd in prompt.split('\n') if cmd.strip()]
        if len(commands) > 1 and all(_is_valid_command(cmd) for cmd in commands):
            return commands
    
    # Case 2: Commands separated by commas
    if ',' in prompt and _has_command_keywords(prompt):
        commands = [cmd.strip() for cmd in prompt.split(',')]
        if len(commands) > 1 and all(_is_valid_command(cmd) for cmd in commands):
            return commands
    
    # Case 3: Commands separated by "then" or "and then"
    then_patterns = [
        r'(.+?)\s+then\s+(.+)',
        r'(.+?)\s+and\s+then\s+(.+)',
        r'(.+?)\s+after\s+that\s+(.+)',
    ]
    
    for pattern in then_patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            first_cmd = match.group(1).strip()
            second_cmd = match.group(2).strip()
            if _is_valid_command(first_cmd) and _is_valid_command(second_cmd):
                return [first_cmd, second_cmd]
    
    # Case 4: Numbered steps (1. 2. 3. etc.)
    numbered_pattern = r'^\s*\d+\.\s*(.+?)(?=\s*\d+\.\s*|$)'
    matches = re.findall(numbered_pattern, prompt, re.MULTILINE | re.DOTALL)
    if len(matches) > 1 and all(_is_valid_command(cmd.strip()) for cmd in matches):
        return [cmd.strip() for cmd in matches]
    
    return None


def _has_command_keywords(prompt: str) -> bool:
    """Check if prompt contains command keywords."""
    keywords = ['click', 'scroll', 'press', 'form', 'navigate', 'type', 'fill', 'select', 'wait', 'focus', 'undofocus', 'subfocus', 'undo', 'defer']
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in keywords)


def _is_valid_command(command: str) -> bool:
    """Check if a command string looks valid."""
    command = command.strip()
    if not command:
        return False
    
    # Must contain at least one action keyword
    if not _has_command_keywords(command):
        return False
    
    # Must be reasonable length
    if len(command) < 3 or len(command) > 200:
        return False
    
    return True


def _ai_parse_commands(prompt: str) -> List[str]:
    """
    Use AI to intelligently parse multiple commands from a complex prompt.
    
    This is the fallback when simple parsing fails.
    """
    system_prompt = """
    You are a command parser for a web automation system. Your job is to split a complex prompt into individual, executable commands.

    Available command types:
    - click: [target] - Click on an element
    - scroll: [direction/amount] - Scroll the page
    - press: [key] - Press a key
    - form: [action] - Fill or interact with forms
    - navigate: [target] - Navigate to a page or section
    - type: [text] - Type text into a field
    - wait: [time] - Wait for a specified time

    Rules:
    1. Split the prompt into individual commands that can be executed sequentially
    2. Preserve the original intent and context of each command
    3. Each command should be self-contained and executable
    4. Maintain logical flow and dependencies between commands
    5. If a command references previous commands, include necessary context
    6. Return only the individual commands, one per line
    7. Do not add explanations or commentary

    Examples:
    Input: "click the login button, then fill the form with my email, then press enter"
    Output:
    click: the login button
    form: fill the form with my email
    press: enter

    Input: "click a job listing, click the close button, then continue to the next job"
    Output:
    click: a job listing
    click: the close button
    continue to the next job
    """
    
    try:
        response = generate_text(
            prompt=f"Parse these commands: {prompt}",
            system_prompt=system_prompt,
        )
        
        if not response:
            return [prompt]  # Fallback to original prompt
        
        # Split response into individual commands
        commands = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                commands.append(line)
        
        # Validate commands
        valid_commands = [cmd for cmd in commands if _is_valid_command(cmd)]
        
        if valid_commands:
            return valid_commands
        else:
            return [prompt]  # Fallback to original prompt
            
    except Exception as e:
        print(f"⚠️ Error in AI command parsing: {e}")
        return [prompt]  # Fallback to original prompt


def is_multi_command(prompt: str) -> bool:
    """
    Check if a prompt contains multiple commands.
    
    Args:
        prompt: The prompt to check
        
    Returns:
        True if the prompt contains multiple commands, False otherwise
    """
    commands = parse_multiple_commands(prompt)
    return len(commands) > 1
