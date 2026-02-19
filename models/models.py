"""
Consolidated data models for browser-vision-bot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# CORE MODELS - Basic types and page elements
# ============================================================================

class ActionType(str, Enum):
    """Types of actions that can be performed"""
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    PRESS = "press"
    COMPLETE = "complete"
    ASK = "ask"
    HANDLE_SELECT = "handle_select"
    HANDLE_UPLOAD = "handle_upload"
    HANDLE_DATETIME = "handle_datetime"
    BACK = "back"
    FORWARD = "forward"
    OPEN = "open"


class PageSection(str, Enum):
    """Sections of a web page"""
    HEADER = "header"
    CONTENT = "content"
    SIDEBAR = "sidebar"
    MODAL = "modal"
    FOOTER = "footer"


class NotebookEntryType(str, Enum):
    """Types of entries stored in the agent notebook"""
    EXTRACTION = "extraction"
    URL_EXTRACTION = "url_extraction"


class DetectedElement(BaseModel):
    """A UI element detected in the screenshot"""
    element_label: Optional[str] = Field(default=None, description="The label of the element")
    element_type: str = Field(description="Type: button, input, link, text, select, upload, date, etc.")
    is_clickable: bool = Field(description="Can this element be clicked?")
    box_2d: List[int] = Field(description="Gemini format: [y_min, x_min, y_max, x_max] normalized 0-1000")
    section: PageSection = Field(description="Which section of the page this is in")
    field_subtype: Optional[str] = Field(default=None, description="For inputs: text, email, password, select, upload, date, etc.")
    confidence: Optional[float] = Field(default=0.5, description="Detection confidence 0.0-1.0")
    requires_special_handling: Optional[bool] = Field(default=False, description="Whether this field requires special multi-step handling")
    overlay_number: Optional[int] = Field(default=None, description="The overlay number from numbered detection system")
    is_focused: Optional[bool] = Field(default=False, description="Is this element currently focused?")
    has_visible_text: Optional[bool] = Field(
        default=False,
        description="True when the element has visible on-screen text (inner text, placeholder, or title).",
    )
    text_presence_score: Optional[int] = Field(
        default=0,
        description="Heuristic text signal strength (0-3) used to bias overlay selection toward text-bearing elements.",
    )
    css_class: Optional[str] = Field(default=None, description="CSS class attribute of the element")
    css_id: Optional[str] = Field(default=None, description="CSS id attribute of the element")
    is_done: Optional[bool] = Field(default=False, description="True when element was already interacted with during the current loop (marked via data-bvb-done DOM attribute)")



class PageElements(BaseModel):
    """A list of detected elements on the page"""
    elements: List[DetectedElement] = Field(description="The list of detected elements")


class ActionStep(BaseModel):
    """One viewport-safe action"""
    action: str
    reasoning: str | None = None
    keys_to_press: List[str] | None = None
    function_name: str | None = None
    function_arguments: dict | None = None

    def __init__(self, **data):
        super().__init__(**data)

    @classmethod
    def from_function_call(
        cls,
        function_name: str,
        arguments: dict,
    ) -> "ActionStep":
        """
        Create ActionStep from function call.

        Args:
            function_name: Name of the function called
            arguments: Dictionary of function arguments

        Returns:
            ActionStep instance with both readable action text and function call data.
        """
        action_string = cls._render_action_text(function_name, arguments)

        return cls(
            action=action_string,
            reasoning=arguments.get("reasoning"),
            function_name=function_name,
            function_arguments=arguments
        )

    @staticmethod
    def _render_action_text(function_name: str, arguments: dict) -> str:
        """Render a concise readable command string for logs/history."""
        if function_name == "click":
            element_id = arguments.get('element_id')
            eid_suffix = f" [id={element_id}]" if element_id is not None else ""
            return f"click: {arguments.get('description', '')}{eid_suffix}".strip()
        if function_name == "type_text":
            eid = arguments.get('element_id')
            suffix = f" [id={eid}]" if eid is not None else ""
            return f"type: {arguments.get('text', '')} : {arguments.get('field_description', '')}{suffix}".strip()
        if function_name == "clear_text":
            eid = arguments.get('element_id')
            suffix = f" [id={eid}]" if eid is not None else ""
            return f"clear_text: {arguments.get('field_description', '')}{suffix}".strip()
        if function_name == "select_option":
            eid = arguments.get('element_id')
            suffix = f" [id={eid}]" if eid is not None else ""
            return f"select_option: {arguments.get('option', '')} in {arguments.get('dropdown_description', '')}{suffix}".strip()
        if function_name == "upload_file":
            return f"upload_file: {arguments.get('file_path', '')} in {arguments.get('target_description', '')}".strip()
        if function_name == "set_datetime":
            return f"set_datetime: {arguments.get('value', '')} in {arguments.get('picker_description', '')}".strip()
        if function_name == "press_key":
            return f"press: {arguments.get('key', '')}".strip()
        if function_name == "open_url":
            return f"open: {arguments.get('url', '')}".strip()
        if function_name == "go_back":
            return f"go_back: {arguments.get('steps', 1)}"
        if function_name == "go_forward":
            return f"go_forward: {arguments.get('steps', 1)}"
        if function_name == "scroll_page":
            return f"scroll: {arguments.get('direction', 'down')}"
        if function_name == "extract_data":
            return f"extract: {arguments.get('data_description', '')}".strip()
        if function_name == "think":
            return (
                f"think: {arguments.get('reasoning', '')} | "
                f"next_action={arguments.get('next_action', 'continue')}"
            ).strip()
        if function_name == "assert_condition":
            return f"assert: {arguments.get('condition', '')}".strip()
        if function_name == "mark_progress":
            return (
                f"mark_progress: {arguments.get('description', '')} | "
                f"count={arguments.get('count', 1)} | done={arguments.get('done', False)}"
            ).strip()
        if function_name == "revise_target":
            return f"revise_target: {arguments.get('new_target', '')} | {arguments.get('reason', '')}".strip()
        if function_name == "flag":
            return f"flag: {arguments.get('message', '')}".strip()
        if function_name == "wait_for":
            return f"wait_for: {arguments.get('condition', '')} | timeout={arguments.get('timeout_seconds', 10)}".strip()
        if function_name == "ask_user":
            return f"ask: {arguments.get('question', '')}".strip()
        if function_name == "switch_tab":
            return f"switch_tab: {arguments.get('tab_id', '')}".strip()
        if function_name == "close_tab":
            return f"close_tab: {arguments.get('tab_id', '')}".strip()
        if function_name == "open_tab":
            return f"open_tab: {arguments.get('url', '')}".strip()
        if function_name == "dismiss_dialog":
            return (
                f"dismiss_dialog: accept={arguments.get('accept', False)}"
                + (f" | input_text={arguments.get('input_text', '')}" if arguments.get("input_text") else "")
            ).strip()
        if function_name == "send_email":
            to_str = arguments.get("to") or ""
            return f"send_email: to={to_str} | subject={arguments.get('subject', '')}".strip()
        return f"{function_name}: {arguments}"

    def _parse_action(text: str) -> tuple[str, str]:
        """Parse action text into (command, body). Raises ValueError if invalid."""
        text = text.strip()
        if not text:
            raise ValueError("action cannot be empty")

        # Strip @ URL suffix if present (e.g., "click button @ https://...")
        if " @ " in text:
            text = text.split(" @ ")[0].strip()

        if ":" in text:
            cmd, body = text.split(":", 1)
            return cmd.strip().lower(), body.strip()

        # Try to parse "command args" format
        match = re.match(r"^(\w+)\s+(.+)$", text)
        if match:
            cmd, body = match.groups()

            # Clean up body - handle various formats
            # "- on X" -> "X"
            body = re.sub(r"^-\s+on\s+", "", body, flags=re.IGNORECASE)
            # "on X" -> "X"
            if cmd.lower() == "click" and body.lower().startswith("on "):
                body = body[3:]

            return cmd.lower(), body.strip()

        # Single word command (defer, stop, forward, back)
        if text.isalpha():
            return text.lower(), ""

        raise ValueError("action must be 'command: target' or 'command target'")

    def _normalize_body(cmd: str, body: str) -> str:
        """Normalize the body based on command type."""
        if cmd == "click":
            if not body:
                raise ValueError("click requires a target")
            body = body[3:].strip() if body.lower().startswith("on ") else body
            # Add element type hint if missing
            if not re.search(r"\b(button|link|tab|checkbox|radio|option|div|input|icon|item)\b", body, re.I):
                if not body.lower().endswith("link"):
                    body = f"{body} link"
            return body.strip()

        if cmd == "type":
            if not body:
                raise ValueError("type requires text and target")
            # Accept "text : field" or "text in/into field" format
            if " : " in body:
                parts = body.split(" : ", 1)
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    return re.sub(r"\s+", " ", body).strip()
            if re.search(r"\b(in|into)\b", body, re.I):
                return re.sub(r"\s+", " ", body).strip()
            raise ValueError("type must use 'text : field' or 'text in field'")

        if cmd == "press":
            key = body.strip()
            if not key or " " in key:
                raise ValueError("press requires a single key")
            return key

        if cmd == "scroll":
            # Strip quotes and whitespace from direction
            direction = body.strip().strip("'\"").lower()
            if direction not in {"up", "down", "left", "right"}:
                raise ValueError("scroll must be up, down, left, or right")
            return direction

        if cmd in {"back", "forward"}:
            return body if body and body.isdigit() else "1"

        if cmd in {"extract", "interceptor", "form", "select", "upload", "datetime", "open",
                "handle_datetime"}:
            if not body:
                raise ValueError(f"{cmd} requires additional detail")
            return body

        # Commands with optional body
        return body


class ActionPlan(BaseModel):
    """A sequential plan of actions that can be executed before the viewport changes."""
    steps: List[ActionStep]
    reasoning: str
    confidence: float
    expected_outcome: str


class VisionPlan(BaseModel):
    """Plan generated by the vision model"""
    detected_elements: PageElements = Field(description="All detected UI elements")
    action_steps: List[ActionStep] = Field(description="Steps to execute")
    reasoning: str = Field(description="Why this plan was chosen")
    confidence: float = Field(description="Overall confidence in this plan", ge=0.0, le=1.0)


class Goal(BaseModel):
    """Goal definition"""
    target_url_contains: List[str] = Field(default_factory=list, description="URL should contain these strings")
    target_page_text: List[str] = Field(default_factory=list, description="Page should contain this text")
    form_should_be_filled: bool = Field(default=False, description="All required form fields should be filled")


class PageInfo(BaseModel):
    """Information about the current page state"""
    width: int = Field(description="Current page width in pixels")
    height: int = Field(description="Current page height in pixels")
    scroll_x: int = Field(description="Current scroll X position")
    scroll_y: int = Field(description="Current scroll Y position")
    url: str = Field(description="Current page URL")
    title: str = Field(description="Current page title")
    dpr: float = Field(description="Device pixel ratio")
    ss_pixel_w: int = Field(description="Screenshot pixel width")
    ss_pixel_h: int = Field(description="Screenshot pixel height")
    css_scale: float = Field(description="CSS scale")
    doc_width: int = Field(description="Document width in pixels")
    doc_height: int = Field(description="Document height in pixels")


class FailedAction(BaseModel):
    """Represents a failed action attempt with context about where and what was attempted"""
    action: str = Field(description="The action command that was attempted (e.g., 'click: Submit button')")
    overlay_index: Optional[int] = Field(default=None, description="The overlay index that was used for this action")
    url: str = Field(description="The page URL where the action was attempted")
    page_title: Optional[str] = Field(default=None, description="The page title where the action was attempted")
    timestamp: Optional[float] = Field(default=None, description="When the action failed (Unix timestamp)")


# ============================================================================
# INTENT MODELS - Structured action intents
# ============================================================================

class ActionIntent(BaseModel):
    """Structured representation of a user instruction"""

    action: str = Field(description="Verb describing the action (click, type, select, upload, datetime, press, scroll)")
    raw_command: str = Field(description="Original user instruction")

    target_text: Optional[str] = Field(default=None, description="Literal text describing the target element")
    role_hint: Optional[str] = Field(default=None, description="Semantic hint for the element role (button, link, field, etc.)")
    value: Optional[str] = Field(default=None, description="Value associated with the action (text to type, option to select, file path, etc.)")
    helper_text: Optional[str] = Field(default=None, description="Additional contextual helper provided with the command")

    modifiers: Dict[str, str] = Field(default_factory=dict, description="Slot modifiers such as ordinal, region, direction")
    collection_hint: Optional[str] = Field(default=None, description="Grouping hint (list, table, menu, etc.)")
    attribute_filters: Dict[str, str] = Field(default_factory=dict, description="Attribute equals filters (e.g., aria-label)")

    def ordinal(self) -> Optional[int]:
        """Return zero-based ordinal index if available (first -> 0, last -> -1)"""
        raw = self.modifiers.get("ordinal")
        if raw is None:
            return None
        try:
            idx = int(raw)
            return idx
        except ValueError:
            return None

    def normalized_terms(self) -> List[str]:
        """Return key terms for semantic scoring"""
        terms: List[str] = []
        if self.target_text:
            terms.extend(_tokenize(self.target_text))
        if self.role_hint:
            terms.append(self.role_hint.lower())
        for value in self.attribute_filters.values():
            terms.extend(_tokenize(value))
        return list(dict.fromkeys(filter(None, terms)))

    def subject_hint(self) -> Optional[str]:
        if self.role_hint:
            return self.role_hint
        if self.target_text:
            tokens = _tokenize(self.target_text)
            if tokens:
                return tokens[0]
        return None


def _tokenize(text: str) -> List[str]:
    import re
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t]
