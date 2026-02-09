"""
Action tools schema for OpenAI function calling.

This module defines the complete set of actions available to the agent as
OpenAI function calling tools, along with conversion utilities to maintain
backward compatibility with the keyword command format.
"""

from typing import Dict, Any, List

# ============================================================================
# ACTION TOOLS SCHEMA
# ============================================================================

ACTION_TOOLS: List[Dict[str, Any]] = [
    # ========================================================================
    # INTERACTION ACTIONS
    # ========================================================================
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click on an interactive element on the page",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_type": {
                        "type": "string",
                        "enum": [
                            "button",
                            "link",
                            "checkbox",
                            "radio",
                            "tab",
                            "icon",
                            "menu item",
                            "card",
                            "image",
                            "text"
                        ],
                        "description": "Type of element to click"
                    },
                    "description": {
                        "type": "string",
                        "description": "Clear description of the element to click (e.g., 'Sign In', 'menu icon', 'first result')"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for clicking the element"
                    },
                    "overlay_index": {
                        "type": "integer",
                        "description": "The index of the element to click"
                    },
                    "why_this_overlay_index": {
                        "type": "string",
                        "description": "Why this overlay index was chosen"
                    }
                },
                "required": ["element_type", "description", "reasoning", "overlay_index", "why_this_overlay_index"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text into an input field",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The exact text to type into the field"
                    },
                    "field_description": {
                        "type": "string",
                        "description": "Description of the input field (e.g., 'email field', 'search box', 'username input')"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for typing the text into the field"
                    }
                },
                "required": ["text", "field_description", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clear_text",
            "description": "Clear the text from an input field. Use this when you need to clear the text from the field before typing new text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field_description": {
                        "type": "string",
                        "description": "Description of the input field (e.g., 'email field', 'search box', 'username input')"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for clearing the text from the field"
                    }
                },
                "required": ["field_description", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "select_option",
            "description": "Select an option from a dropdown or select element",
            "parameters": {
                "type": "object",
                "properties": {
                    "option": {
                        "type": "string",
                        "description": "The option text to select (e.g., 'United States', 'Blue', 'Large')"
                    },
                    "dropdown_description": {
                        "type": "string",
                        "description": "Description of the dropdown/select element (e.g., 'country dropdown', 'size selector')"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for selecting the option"
                    }
                },
                "required": ["option", "dropdown_description", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "upload_file",
            "description": "Upload a file to a file input element",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path or name of the file to upload (e.g., 'resume.pdf', 'profile_pic.jpg')"
                    },
                    "target_description": {
                        "type": "string",
                        "description": "Description of the upload target (e.g., 'file upload button', 'attachment field')"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for uploading the file"
                    }
                },
                "required": ["file_path", "target_description", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_datetime",
            "description": "Set a date, time, or datetime value in a date/time picker",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "Date/time value in ISO format or natural format (e.g., '2026-01-15', '14:30', '2026-01-15T14:30')"
                    },
                    "picker_description": {
                        "type": "string",
                        "description": "Description of the date/time picker (e.g., 'departure date', 'appointment time')"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for setting the date/time"
                    }
                },
                "required": ["value", "picker_description", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "Press a keyboard key or key combination",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "enum": [
                            "Enter",
                            "Escape",
                            "Tab",
                            "ArrowDown",
                            "ArrowUp",
                            "ArrowLeft",
                            "ArrowRight",
                            "Backspace",
                            "Delete",
                            "PageDown",
                            "PageUp",
                            "Home",
                            "End",
                            "Space"
                        ],
                        "description": "The key to press"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for pressing the key"
                    }
                },
                "required": ["key", "reasoning"],
                "additionalProperties": False
            }
        }
    },

    # ========================================================================
    # NAVIGATION ACTIONS
    # ========================================================================
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Navigate to a specific URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to navigate to (must include protocol, e.g., 'https://example.com')"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for navigating to the URL"
                    }
                },
                "required": ["url", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "go_back",
            "description": "Navigate back in browser history",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "integer",
                        "description": "Number of pages to go back (default: 1)",
                        "minimum": 1,
                        "default": 1
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for going back in browser history"
                    }
                },
                "required": ["steps", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "go_forward",
            "description": "Navigate forward in browser history",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "integer",
                        "description": "Number of pages to go forward (default: 1)",
                        "minimum": 1,
                        "default": 1
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for going forward in browser history"
                    }
                },
                "required": ["steps", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scroll_page",
            "description": "Scroll the page in a specific direction",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "Direction to scroll"
                    },
                    "amount": {
                        "type": "string",
                        "enum": ["small", "medium", "large"],
                        "description": "Amount to scroll (optional, default: medium)",
                        "default": "medium"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for scrolling the page"
                    }
                },
                "required": ["direction", "reasoning"],
                "additionalProperties": False
            }
        }
    },

    # ========================================================================
    # DATA EXTRACTION & COMPLETION
    # ========================================================================
    {
        "type": "function",
        "function": {
            "name": "extract_data",
            "description": "Extract specific data from the current page and store it",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_description": {
                        "type": "string",
                        "description": "Clear description of what data to extract (e.g., 'job title and company name', 'product price and rating', 'all table rows')"
                    },
                    "format_hint": {
                        "type": "string",
                        "enum": ["text", "list", "table", "structured"],
                        "description": "Expected format of the data (optional)",
                        "default": "text"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for extracting the data"
                    }
                },
                "required": ["data_description", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember_data",
            "description": "Remember data for later use",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "The data to remember"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for remembering the data"
                    }
                },
                "required": ["data", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Mark the task as successfully completed",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was accomplished (e.g., 'Successfully logged in and navigated to dashboard', 'Extracted all job listings')"
                    },
                    "details": {
                        "type": "string",
                        "description": "Detailed explanation of the completion, including any important context or results"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for completing the task"
                    }
                },
                "required": ["summary", "details", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complete_sequence",
            "description": "Mark the sequence as successfully completed and end the sequence",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for completing the sequence and ending the sequence"
                    }
                },
                "required": ["reasoning"],
                "additionalProperties": False
            }
        }
    },

    # ========================================================================
    # COMMUNICATION & CONTROL
    # ========================================================================
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask the user a question when clarification or additional information is needed",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Clear, specific question to ask the user"
                    },
                    "context": {
                        "type": "string",
                        "description": "Why you need this information (helps user understand)",
                        "default": ""
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for asking the user the question"
                    }
                },
                "required": ["question", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "talk_to_user",
            "description": "Send a message to the user (for updates, explanations, or notifications).",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to send to the user. This should be in a conversational tone and not a command. This is purely for communication with the user and does not advance the task."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for talking to the user"
                    }
                },
                "required": ["message", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "defer_action",
            "description": "Defer an action for later execution (use when page is still loading or not ready)",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why this action needs to be deferred (e.g., 'page still loading', 'waiting for element to appear')"
                    },
                    "intended_action": {
                        "type": "string",
                        "description": "Description of the action that will be performed once ready"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for deferring the action"
                    }
                },
                "required": ["reason", "intended_action", "reasoning"],
                "additionalProperties": False
            }
        }
    },
]


# ============================================================================
# FUNCTION CALL TO KEYWORD CONVERTER
# ============================================================================


def function_call_to_keyword_action(function_name: str, arguments: Dict[str, Any]) -> str:
    """
    Convert OpenAI function call to keyword action string format.

    This maintains compatibility with existing _execute_keyword_command infrastructure.

    Args:
        function_name: Name of the function called
        arguments: Dictionary of function arguments

    Returns:
        Keyword action string (e.g., "click: button search")

    Raises:
        ValueError: If function name is unknown
    """

    if function_name == "click":
        return f"click: {arguments['element_type']} {arguments['description']}"

    elif function_name == "type_text":
        # Escape single quotes in text
        text = arguments['text']
        return f"type: '{text}' : {arguments['field_description']}"

    elif function_name == "clear_text":
        return f"clear_text: {arguments['field_description']}"

    elif function_name == "select_option":
        # Escape single quotes in option
        option = arguments['option'].replace("'", "\\'")
        return f"select: '{option}' in {arguments['dropdown_description']}"

    elif function_name == "upload_file":
        return f"upload: {arguments['file_path']} in {arguments['target_description']}"

    elif function_name == "set_datetime":
        return f"datetime: {arguments['value']} in {arguments['picker_description']}"

    elif function_name == "press_key":
        return f"press: {arguments['key']}"

    elif function_name == "open_url":
        return f"open: {arguments['url']}"

    elif function_name == "go_back":
        steps = arguments.get('steps', 1)
        return f"back: {steps}" if steps > 1 else "back"

    elif function_name == "go_forward":
        steps = arguments.get('steps', 1)
        return f"forward: {steps}" if steps > 1 else "forward"

    elif function_name == "scroll_page":
        direction = arguments['direction']
        # amount = arguments.get('amount', 'medium')
        # if amount != 'medium':
        #     return f"scroll: {direction} {amount}"
        return f"scroll: {direction}"

    elif function_name == "extract_data":
        return f"extract: {arguments['data_description']}"
    elif function_name == "remember_data":
        return f"remember: {arguments['data']}"
    elif function_name == "complete_task":
        # Use details as the main content, summary as prefix
        return f"complete: {arguments['details']}"
    elif function_name == "complete_sequence":
        return f"complete_sequence: {arguments['reasoning']}"
    elif function_name == "ask_user":
        context = arguments.get('context', '')
        question = arguments['question']
        if context:
            return f"ask: {question} (Context: {context})"
        return f"ask: {question}"

    elif function_name == "talk_to_user":
        return f"talk: {arguments['message']}"

    elif function_name == "defer_action":
        return f"defer: {arguments['reason']}"

    else:
        raise ValueError(f"Unknown function: {function_name}")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ACTION_TOOLS",
    "function_call_to_keyword_action",
]
