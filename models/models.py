"""
Consolidated data models for browser-vision-bot.

This module contains all core data models organized by category:
- Core models: ActionType, PageSection, DetectedElement, etc.
- Task models: Task, MissionPlan
- Intent models: ActionIntent
"""
from __future__ import annotations

from ast import Str
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

        Converts function call to keyword action string format for backward
        compatibility with existing executor infrastructure.

        Args:
            function_name: Name of the function called
            arguments: Dictionary of function arguments
            reasoning: Optional reasoning for the action

        Returns:
            ActionStep instance with both keyword and function call data
        """
        from agent.action_tools import function_call_to_keyword_action

        # Convert to keyword format for backward compatibility
        action_string = function_call_to_keyword_action(function_name, arguments)

        return cls(
            action=action_string,
            function_name=function_name,
            function_arguments=arguments
        )

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
# TASK MODELS - Unified task system with targets
# ============================================================================

class TaskStatus(str, Enum):
    """Status of task execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskCompletionStatus(str, Enum):
    """How a task finished"""
    COMPLETED = "completed"
    PARTIAL = "partial"
    STUCK = "stuck"
    BLOCKED = "blocked"


@dataclass
class Task:
    """A unified task with a target. target=1 for single actions, target=N for repetitive, target='all' for indefinite."""
    goal: str
    target: Union[int, str] = 1  # 1, 5, "all"
    progress: int = 0
    history: List[str] = field(default_factory=list)
    task_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    completion_status: Optional[TaskCompletionStatus] = None
    created_at: float = 0.0
    completed_at: Optional[float] = None


@dataclass
class MissionPlan:
    """Complete plan of tasks for executing a mission"""
    tasks: List[Task] = field(default_factory=list)
    current_task_index: int = 0

    def get_current_task(self) -> Optional[Task]:
        """Get the current task being executed"""
        if 0 <= self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        return None

    def get_completed_tasks(self) -> List[Task]:
        """Get all completed tasks"""
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]

    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks"""
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def all_tasks_completed(self) -> bool:
        """Check if all tasks have been completed"""
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks)

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Find a task by its ID"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None


class TaskDefinition(BaseModel):
    """Task definition from the mission planner"""
    task: str
    target: Union[int, str] = 1  # 1, 5, "all"


class MissionPlannerOutput(BaseModel):
    """Output from Mission Planner decomposition"""
    tasks: list[TaskDefinition]


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
