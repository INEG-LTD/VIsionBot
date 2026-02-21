"""Action tool schemas for function-calling.

This module defines the tool schemas for action execution
(click/type/think/etc.) used by the browser agent.
"""

from copy import deepcopy
from typing import Dict, Any, List, Optional

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
            "description": (
                "Click on an interactive element. Pick the element from the "
                "INTERACTIVE ELEMENTS index by its [id] number. For elements "
                "marked 'SEE CROP GALLERY', cross-reference the gallery images."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "element_id": {
                        "type": "integer",
                        "description": "The [id] number from the INTERACTIVE ELEMENTS index"
                    },
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
                            "text",
                            "input"
                        ],
                        "description": "Type of element to click"
                    },
                    "description": {
                        "type": "string",
                        "description": "Clear description of the element (e.g., 'Sign In button', 'close icon')"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for clicking the element"
                    },
                },
                "required": ["element_id", "element_type", "description", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": (
                "Type text into an input field. You may specify element_id from "
                "the INTERACTIVE ELEMENTS index to target a specific field."
            ),
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
                    },
                    "element_id": {
                        "type": "integer",
                        "description": "Optional [id] from the INTERACTIVE ELEMENTS index for the target input field"
                    },
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
            "description": (
                "Clear the text from an input field. You may specify element_id from "
                "the INTERACTIVE ELEMENTS index to target a specific field."
            ),
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
                    },
                    "element_id": {
                        "type": "integer",
                        "description": "Optional [id] from the INTERACTIVE ELEMENTS index for the target input field"
                    },
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
            "description": (
                "Select an option from a dropdown or select element. You may specify "
                "element_id from the INTERACTIVE ELEMENTS index to target a specific dropdown."
            ),
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
                    },
                    "element_id": {
                        "type": "integer",
                        "description": "Optional [id] from the INTERACTIVE ELEMENTS index for the target dropdown"
                    },
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
            "description": (
                "Scroll the page or a specific scrollable container (modal, sidebar, overflow panel, list). "
                "Three modes:\n"
                "1. Window scroll: omit element_id and scroll_to_element_id — scrolls the main page.\n"
                "2. Container scroll: provide element_id of ANY element inside the container — the system "
                "finds the nearest scrollable ancestor and scrolls it. Use this for modals, sidebars, "
                "dropdown lists, chat panels, or any overflow area.\n"
                "3. Scroll-to: provide scroll_to_element_id — brings that element into view regardless "
                "of where it is. direction and amount are ignored in this mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                        "description": "Direction to scroll. Ignored when scroll_to_element_id is set."
                    },
                    "amount": {
                        "type": "string",
                        "enum": ["small", "medium", "large"],
                        "description": "How far to scroll: small=150px, medium=400px, large=800px. Default: medium. Ignored when scroll_to_element_id is set.",
                        "default": "medium"
                    },
                    "element_id": {
                        "type": "integer",
                        "description": (
                            "ID of any element inside the scrollable container you want to scroll. "
                            "The system walks up the DOM to find the nearest scrollable ancestor and scrolls it. "
                            "Example: to scroll a modal, pass the ID of any element visible inside the modal."
                        )
                    },
                    "scroll_to_element_id": {
                        "type": "integer",
                        "description": (
                            "ID of an element to bring into view. Scrolls whatever container is needed. "
                            "When set, direction and amount are ignored."
                        )
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for the scroll action"
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
            "description": """
            Extract specific data from the current page and store it.
            This can also be used to 'read' a webpage and also simultaneously 'write' the data to the notebook.
            
            You should use this tool when you want to:
            - Read the content of the page
            - Extract specific data from the page
            - Summarize the content of the page
            - Store the data in the notebook
            - Use the data in the next actions
            - Create a summary of the page
            - Create a list of the page
            - Create a table of the page
            - Create a structured data of the page
            """,
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
            "description": (
                "Stop and think about what's happening. Decide what to do next via next_action.\n"
                "• continue = more work needed, take another user-facing action.\n"
                "• start_loop = you need to repeat an action sequence N times. Provide loop_count and loop_description. The action you just did counts as round 1.\n"
                "• advance = (loop only) current iteration is done, move to the next round.\n"
                "• end_loop = exit the loop early (before all rounds are done).\n"
                "• done = mission is fully complete.\n"
                "• stuck = current strategy failed, provide a new one in reasoning + recommended_next_step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Your internal reasoning — what you're thinking about, what you've noticed, what you plan to do next"
                    },
                    "next_action": {
                        "type": "string",
                        "enum": ["continue", "start_loop", "advance", "end_loop", "done", "stuck"],
                        "description": (
                            "What to do after thinking. "
                            "'continue' = need another user-facing action. "
                            "'start_loop' = begin a loop (requires loop_count and loop_description). "
                            "'advance' = (in-loop) current iteration done, advance to next round. "
                            "'end_loop' = exit the loop early. "
                            "'done' = mission fully complete. "
                            "'stuck' = strategy failed, provide replacement in reasoning."
                        )
                    },
                    "recommended_next_step": {
                        "type": "string",
                        "description": (
                            "Optional one-step recommendation for the very next action. "
                            "Format as a concrete action hint such as 'click: top article title'."
                        )
                    },
                    "loop_count": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Required when next_action='start_loop'. Total number of iterations (including the one already done)."
                    },
                    "loop_description": {
                        "type": "string",
                        "description": "Required when next_action='start_loop'. Describes what one iteration looks like (e.g., 'click a job listing')."
                    },
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
    # TAB MANAGEMENT
    # ========================================================================
    {
        "type": "function",
        "function": {
            "name": "switch_tab",
            "description": "Switch to a different browser tab. Check the OPEN TABS section to see available tab IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tab_id": {
                        "type": "string",
                        "description": "Tab ID to switch to (e.g., 't2')"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for switching tabs"
                    }
                },
                "required": ["tab_id", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "close_tab",
            "description": "Close a browser tab. Cannot close the last remaining tab.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tab_id": {
                        "type": "string",
                        "description": "Tab ID to close (e.g., 't2')"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for closing the tab"
                    }
                },
                "required": ["tab_id", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_tab",
            "description": "Open a new browser tab, optionally navigating to a URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to in the new tab (optional — omit for a blank tab)"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for opening a new tab"
                    }
                },
                "required": ["reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dismiss_dialog",
            "description": "Dismiss the JavaScript dialog currently blocking the page. Must be called before any other browser action can proceed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "accept": {
                        "type": "boolean",
                        "description": "true = OK/Accept, false = Cancel/Dismiss"
                    },
                    "input_text": {
                        "type": "string",
                        "description": "Text to enter (only for prompt() dialogs, ignored otherwise)"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for accepting or dismissing the dialog"
                    }
                },
                "required": ["accept", "reasoning"],
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
    {
        "type": "function",
        "function": {
            "name": "report_data",
            "description": (
                "Report textual data back to the host application via callback. "
                "Use this when the user needs intermediate or final textual output."
                "Use this tool when asked to return/give/report data/information/results/etc back to the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "string",
                        "description": "Text payload to report back to the user"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for reporting this data now"
                    }
                },
                "required": ["payload", "reasoning"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_data",
            "description": (
                "Write textual data to the local filesystem. "
                "If path is omitted, save to the default bba-data location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "Text content to write to disk"
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional target file or directory path. If omitted, default location is used."
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Optional filename to use when path is a directory or omitted"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "description": "Write mode (default: overwrite)",
                        "default": "overwrite"
                    },
                    "format_hint": {
                        "type": "string",
                        "enum": ["text", "markdown", "json", "csv"],
                        "description": "Optional format hint to choose default file extension",
                        "default": "text"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for writing this data"
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
            "name": "send_email",
            "description": "Send an email via the Resend API. Use when the user asks to email someone or to send a message by email",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line"
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body (HTML or plain text). Will be sent as HTML; use <p> tags for paragraphs if desired."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for sending this email"
                    }
                },
                "required": ["to", "subject", "body", "reasoning"],
                "additionalProperties": False
            }
        }
    },
]

# Optional memory evidence on key decision-bearing tools.
_MEMORY_EVIDENCE_PROPERTY: Dict[str, Any] = {
    "memory_evidence_ids": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Optional memory IDs that justify this action."
    },
}

_EVIDENCE_TOOLS = {"click", "type_text", "extract_data", "think"}
for _tool in ACTION_TOOLS:
    _fn = _tool.get("function", {})
    if _fn.get("name") not in _EVIDENCE_TOOLS:
        continue
    _params = _fn.get("parameters", {})
    if not isinstance(_params, dict):
        continue
    _properties = _params.setdefault("properties", {})
    _properties.update(_MEMORY_EVIDENCE_PROPERTY)

_NARRATIVE_PROPERTY: Dict[str, Any] = {
    "narrative": {
        "type": "string",
        "description": (
            "A short first-person description of what you're doing, written for a non-technical observer. "
            "No memory IDs, element numbers, internal strategy names, or technical references. "
            "E.g. 'I'm clicking the sign-in button', 'I've just submitted the search form and I'm "
            "waiting for results', 'I'm going back to the previous page to try a different link'."
        )
    }
}

_BUDGET_CONTRACT_PROPERTIES: Dict[str, Any] = {
    "budget_spent": {
        "type": "integer",
        "minimum": 0,
        "description": "Controller budget counter: actions spent so far in this mission."
    },
    "budget_remaining": {
        "type": "integer",
        "minimum": 0,
        "description": "Controller budget counter: actions remaining in this mission."
    },
    "budget_total": {
        "type": "integer",
        "minimum": 1,
        "description": "Controller budget counter: total allowed actions for this mission."
    },
}
_BUDGET_CONTRACT_FIELDS = tuple(_BUDGET_CONTRACT_PROPERTIES.keys())

for _tool in ACTION_TOOLS:
    _fn = _tool.get("function", {})
    _params = _fn.get("parameters", {})
    if not isinstance(_params, dict):
        continue
    _properties = _params.setdefault("properties", {})
    _properties.update(_NARRATIVE_PROPERTY)
    _properties.update(_BUDGET_CONTRACT_PROPERTIES)
    _required = _params.setdefault("required", [])
    if "narrative" not in _required:
        _required.append("narrative")
    if "budget_spent" not in _required:
        _required.append("budget_spent")
    if "budget_remaining" not in _required:
        _required.append("budget_remaining")
    if "budget_total" not in _required:
        _required.append("budget_total")

# Extra contract for think(next_action=stuck)
_THINK_TOOL = next(
    (t for t in ACTION_TOOLS if t.get("function", {}).get("name") == "think"),
    None,
)
if _THINK_TOOL:
    _think_props = _THINK_TOOL["function"]["parameters"].setdefault("properties", {})
    _think_props["stuck_pattern"] = {
        "type": "string",
        "enum": [
            "action_loop",
            "no_state_change",
            "failure_cluster",
            "navigation_loop",
            "element_not_found",
            "other",
        ],
        "description": "Required when next_action=stuck.",
    }

# Tools available during checkpoint mode (after a browser action).
# Only think is available — agent must decide next via think's next_action.
_ACTION_TOOLS_BY_NAME: Dict[str, Dict[str, Any]] = {
    tool["function"]["name"]: tool for tool in ACTION_TOOLS
}
NON_CHECKPOINT_TOOLS: List[Dict[str, Any]] = [
    tool for tool in ACTION_TOOLS
    if tool["function"]["name"] != "think"
]
CHECKPOINT_TOOLS: List[Dict[str, Any]] = [
    _ACTION_TOOLS_BY_NAME["think"],
]

# Tools available when a dialog is blocking the active tab.
# Only dismiss_dialog when not checkpointing; no browser actions are allowed.
DIALOG_TOOLS: List[Dict[str, Any]] = [
    tool for tool in NON_CHECKPOINT_TOOLS
    if tool["function"]["name"] == "dismiss_dialog"
]
DIALOG_CHECKPOINT_TOOLS: List[Dict[str, Any]] = [
    _ACTION_TOOLS_BY_NAME["dismiss_dialog"],
    _ACTION_TOOLS_BY_NAME["think"],
]


def get_filtered_tools(
    checkpoint_mode: bool = False,
    dialog_pending: bool = False,
    budget_constraints_enabled: bool = True,
) -> List[Dict[str, Any]]:
    """
    Get the appropriate tool list based on current state.

    Args:
        checkpoint_mode: If True, return CHECKPOINT_TOOLS instead of the normal action tools
        dialog_pending: If True, return DIALOG_TOOLS (dialog blocks everything)
        budget_constraints_enabled: If False, strip budget fields from tool schemas

    Returns:
        Filtered list of tools available to the agent
    """
    # Dialog takes highest priority — blocks all browser actions
    if dialog_pending:
        selected = DIALOG_CHECKPOINT_TOOLS if checkpoint_mode else DIALOG_TOOLS
    else:
        selected = CHECKPOINT_TOOLS if checkpoint_mode else NON_CHECKPOINT_TOOLS

    if budget_constraints_enabled:
        return selected

    stripped = deepcopy(selected)
    for tool in stripped:
        params = tool.get("function", {}).get("parameters", {})
        if not isinstance(params, dict):
            continue
        properties = params.get("properties")
        if isinstance(properties, dict):
            for field in _BUDGET_CONTRACT_FIELDS:
                properties.pop(field, None)
        required = params.get("required")
        if isinstance(required, list):
            params["required"] = [item for item in required if item not in _BUDGET_CONTRACT_FIELDS]
    return stripped


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ACTION_TOOLS",
]
