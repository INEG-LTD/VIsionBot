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
    # DATA EXTRACTION
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

    # ========================================================================
    # COGNITIVE TOOLS - Thinking, progress, and communication
    # ========================================================================
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Stop and think about what's happening. Use this when you need to reason through a problem, plan your next steps, or figure out why something isn't working. You must also decide what to do next via the next_action parameter. No browser action is taken.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Your internal reasoning — what you're thinking about, what you've noticed, what you plan to do next"
                    },
                    "next_action": {
                        "type": "string",
                        "enum": ["continue", "mark_progress", "done", "stuck"],
                        "description": "What to do after thinking. 'continue' = keep working with browser actions. 'mark_progress' = I just completed a unit of work, record it. 'done' = the task is fully complete, nothing left to do. 'stuck' = I can't make progress, stop."
                    }
                },
                "required": ["reasoning", "next_action"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assert_condition",
            "description": "Check whether something you expect to be true actually is. Use this after an action to verify it worked — like checking that a button changed state, a page loaded, or text appeared.",
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {
                        "type": "string",
                        "description": "What you're checking (e.g., 'The like button changed to filled/red', 'The search results page loaded')"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why you believe this condition is true or false based on what you see"
                    }
                },
                "required": ["condition", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mark_progress",
            "description": "Record that you completed a unit of work. Call this every time you finish one iteration of a repeating task (e.g., liked a post, extracted a listing, filled a form). Set done=true when you've finished everything.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "What you just accomplished (e.g., 'Liked the post about machine learning by @alice', 'Extracted job listing for Software Engineer at Google')"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this counts as progress"
                    },
                    "count": {
                        "type": "integer",
                        "description": "How many units of work this represents (default: 1)",
                        "default": 1,
                        "minimum": 1
                    },
                    "done": {
                        "type": "boolean",
                        "description": "Only relevant for open-ended tasks (target='all'). Set to true when there's nothing left to do. For numeric targets, the system auto-completes when the count is reached — keep this false.",
                        "default": False
                    }
                },
                "required": ["description", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "revise_target",
            "description": "Adjust how many times you need to do something. Use this if you discover the actual number differs from the original target (e.g., there are only 3 items when asked for 5).",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_target": {
                        "type": "integer",
                        "description": "The revised target count",
                        "minimum": 1
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the target needs to change (e.g., 'Only 3 job listings are visible on the page, not 5')"
                    }
                },
                "required": ["new_target", "reason"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flag",
            "description": "Send a heads-up to the user about something you noticed. This doesn't stop your work — it's just a notification. Use it for things like login walls, CAPTCHAs, unexpected states, or anything the user should know about.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "What you want to tell the user (e.g., 'Heads up — this site is asking me to log in before I can continue')"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why this is worth flagging"
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
            "name": "wait_for",
            "description": "Wait for something to happen on the page before continuing. Use this when you can see the page is loading, an animation is playing, or content hasn't appeared yet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {
                        "type": "string",
                        "description": "What you're waiting for (e.g., 'the search results to load', 'the spinner to disappear', 'the modal to close')"
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Maximum seconds to wait (default: 10)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 30
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why you need to wait"
                    }
                },
                "required": ["condition", "reasoning"],
                "additionalProperties": False
            }
        }
    },

    # ========================================================================
    # COMMUNICATION
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
]


# ============================================================================
# FUNCTION CALL TO KEYWORD CONVERTER
# ============================================================================


def function_call_to_keyword_action(function_name: str, arguments: Dict[str, Any]) -> str:
    """
    Convert OpenAI function call to keyword action string format.

    This maintains compatibility with existing _execute_keyword_command infrastructure.
    """

    if function_name == "click":
        return f"click: {arguments['element_type']} {arguments['description']}"

    elif function_name == "type_text":
        text = arguments['text']
        return f"type: '{text}' : {arguments['field_description']}"

    elif function_name == "clear_text":
        return f"clear_text: {arguments['field_description']}"

    elif function_name == "select_option":
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
        return f"scroll: {direction}"

    elif function_name == "extract_data":
        return f"extract: {arguments['data_description']}"

    # New cognitive tools
    elif function_name == "think":
        next_action = arguments.get('next_action', 'continue')
        return f"think: {arguments['reasoning']} | next_action={next_action}"

    elif function_name == "assert_condition":
        return f"assert: {arguments['condition']} | {arguments['reasoning']}"

    elif function_name == "mark_progress":
        count = arguments.get('count', 1)
        done = arguments.get('done', False)
        return f"mark_progress: {arguments['description']} | count={count} | done={done}"

    elif function_name == "revise_target":
        return f"revise_target: {arguments['new_target']} | {arguments['reason']}"

    elif function_name == "flag":
        return f"flag: {arguments['message']}"

    elif function_name == "wait_for":
        timeout = arguments.get('timeout_seconds', 10)
        return f"wait_for: {arguments['condition']} | timeout={timeout}"

    elif function_name == "ask_user":
        context = arguments.get('context', '')
        question = arguments['question']
        if context:
            return f"ask: {question} (Context: {context})"
        return f"ask: {question}"

    else:
        raise ValueError(f"Unknown function: {function_name}")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ACTION_TOOLS",
    "function_call_to_keyword_action",
]
