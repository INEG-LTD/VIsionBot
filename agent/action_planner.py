"""
Action Planner - Determines the next viewport-aware plan based on current state.

Step 2: LLM-based action planning that builds a sequence of actions you can execute
before the viewport changes, relying only on what is visible.
"""

from typing import Optional, List, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import re

from core.session import SessionTracker
from agent.notebook import Notebook
from agent.agent_context import EnvironmentState
from lib.ai import (
    generate_model,
    generate_action_with_tools,
    ReasoningLevel,
    get_default_agent_model,
    get_default_agent_reasoning_level,
)
from models.models import ActionPlan, ActionStep, FailedAction, NotebookEntryType, PageElements
from utils.event_logger import get_event_logger

class ActionPlanner:
    """
    Determines a viewport-safe action plan based on:
    - Current viewport (what's visible on screen)
    - Environment state (interactions, scroll position, etc.)
    - User prompt (what we're trying to achieve)

    This is reactive - it determines an ordered list of actions that can be executed
    without waiting for new UI elements to appear.
    """
    
    def __init__(
        self,
        user_prompt: str,
        session_tracker: SessionTracker,
        base_knowledge: Optional[List[str]] = None,
        *,
        model_name: Optional[str] = None,
        reasoning_level: Union[ReasoningLevel, str, None] = None,
        image_detail: str = "high",
        interaction_summary_limit: Optional[int] = None,
        include_visible_text_in_agent_context: bool = False,
        max_actions_per_plan: int = 6,
        extraction_schema: Optional[Dict[str, Any]] = None,
        current_sequence_task: Optional[str] = None,
        current_sequential_iteration: int = 0,
        current_iteration: int = 0,
    ):
        self.user_prompt = user_prompt
        self.base_knowledge = base_knowledge or []
        self.model_name = model_name or get_default_agent_model()
        if reasoning_level is None:
            reasoning = ReasoningLevel.coerce(get_default_agent_reasoning_level())
        else:
            reasoning = ReasoningLevel.coerce(reasoning_level)
        self.reasoning_level: ReasoningLevel = reasoning
        self.image_detail = image_detail
        self._system_prompt_cache: dict[str, str] = {}  # Cache system prompts
        self.interaction_summary_limit = interaction_summary_limit
        self.include_visible_text_in_agent_context = include_visible_text_in_agent_context
        self.session_tracker: SessionTracker = session_tracker
        self.max_actions_per_plan = max_actions_per_plan
        self.extraction_schema = extraction_schema
        self.current_sequential_iteration = current_sequential_iteration
        self.current_sequence_task = current_sequence_task
        self.current_iteration = current_iteration
   
    def get_next_actions_with_function_calling(
        self,
        environment_state: EnvironmentState,
        screenshot: bytes,
        notebook: Notebook,
        element_data: PageElements,
    ) -> tuple[Optional[list[ActionStep]], Optional[str]]:
        """
        Generate next action using function calling (alternative to full plan generation).

        This uses OpenAI function calling to generate a single validated action
        with strict parameter schemas.

        Args:
            environment_state: Current environment state
            screenshot: Current screenshot (viewport only)
            notebook: Agent's notebook with previously extracted data

        Returns:
            Tuple of (list[ActionStep], error_message). list[ActionStep] is None if generation failed.
        """
        from agent.action_tools import ACTION_TOOLS

        try:
            user_prompt = f"""
            You are currently trying to: {self.user_prompt}

            Based on the screenshot and context, what is the best next action to accomplish this task?
            """
            system_prompt = self._build_function_calling_system_prompt(
                self.session_tracker,
                environment_state,
                notebook,
                element_data,
            )
            
            # Generate action using function calling
            result = generate_action_with_tools(
                prompt=user_prompt,
                tools=ACTION_TOOLS,
                system_prompt=system_prompt,
                image=screenshot,
                image_detail=self.image_detail,
                model=self.model_name,
                reasoning_level=self.reasoning_level,
                tool_choice="required"
            )

            actions = []
            for action in result:
                # Check if function was called
                if not action["function_name"]:
                    return None, "Model did not call any function"
                
                get_event_logger().system_debug(f"Function name: {action['function_name']}")
                get_event_logger().system_debug(f"Arguments: {action['arguments']}")
                
                # Create ActionStep from function call
                action_step = ActionStep.from_function_call(
                    function_name=action["function_name"],
                    arguments=action["arguments"],
                )
                get_event_logger().command_generated(command=action_step)

                actions.append(action_step)

            return actions, None

        except Exception as e:
            get_event_logger().system_error(f"Error generating action with function calling: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error generating action: {e}"

    def _build_function_calling_system_prompt(
        self,
        session_tracker: SessionTracker,
        state: EnvironmentState,
        notebook: Notebook,
        element_data: PageElements,
    ) -> str:
        """Build system prompt for function calling action generation."""

        # Build base knowledge section if provided
        base_knowledge_section = ""
        if self.base_knowledge:
            base_knowledge_section = "\n\nCUSTOM RULES:\n"
            for i, knowledge in enumerate(self.base_knowledge, 1):
                base_knowledge_section += f"{i}. {knowledge}\n"

        # Get history and navigation info
        history_block = self._get_history_block()
        nav_summary = self._summarize_navigation_history(
            getattr(state, "url_history", []),
            getattr(state, "url_pointer", None)
        )

        # Build sequential task section ONLY if actually in a sequence
        sequence_info = ""
        if self.current_sequence_task is not None:
            get_event_logger().system_debug("Building sequential task prompt section")
            sequence_info = f"""
═══════════════════════════════════════════════════════════════
🔄 SEQUENTIAL TASK MODE (PRIORITY CONTEXT)
═══════════════════════════════════════════════════════════════

You are executing a multi-turn sequential task.

Main task: {self.current_sequence_task}
Current turn: {self.current_sequential_iteration}

HOW SEQUENTIAL TASKS WORK:
1. Each turn requires specific action(s)
2. After completing actions for your turn → call complete_task
3. After completing the FINAL turn → call complete_sequence

EXAMPLES:
• "click a button 3 times":
  Turn 1: Click button → complete_task
  Turn 2: Click button → complete_task
  Turn 3: Click button → complete_task → complete_sequence

• "type 'hello' in field A and field B for 3 turns":
  Turn 1: Type in A → Type in B → complete_task
  Turn 2: Type in A → Type in B → complete_task
  Turn 3: Type in A → Type in B → complete_task → complete_sequence

CRITICAL RULES:
✓ Check "Turn {self.current_sequential_iteration} actions so far" below
✓ Count ONLY current turn actions (ignore previous turns)
✓ If required action(s) succeeded this turn → call complete_task
✓ DO NOT repeat actions endlessly (if it worked once, you're done)
✓ After final turn → call complete_sequence

"""
            # Add previous turn results
            for notebook_entry in notebook.to_list():
                if notebook_entry.type == NotebookEntryType.SEQUENTIAL_SUBTASK_RESULT:
                    prev_turn_history = self._get_history_block(
                        sequential_iteration=notebook_entry.subtask_turn,
                        just_data=True
                    )
                    sequence_info += f"""
Turn {notebook_entry.subtask_turn} completed:
{prev_turn_history}
"""

            # Add current turn progress
            current_turn_history = self._get_history_block(
                sequential_iteration=self.current_sequential_iteration,
                just_data=True
            )
            sequence_info += f"""
Turn {self.current_sequential_iteration} actions so far:
{current_turn_history if current_turn_history else "No actions yet this turn."}

═══════════════════════════════════════════════════════════════
"""

        # Build overlay list
        candidate_lines: list[str] = []
        for elem in element_data.elements:
            idx = elem.overlay_number
            elem_type = elem.element_type or "unknown"
            subtype = elem.field_subtype or ""
            label = elem.element_label or ""
            focused = elem.is_focused

            # Compact format: Overlay {idx} type={type} [subtype={subtype}] label={label} focused={bool}
            overlay_desc = f"Overlay {idx} type={elem_type}"
            if subtype:
                overlay_desc += f" subtype={subtype}"
            if label:
                overlay_desc += f" label=\"{label}\""
            overlay_desc += f" focused={focused}"

            candidate_lines.append(overlay_desc)

        overlays = "\n".join(candidate_lines) if candidate_lines else "No interactive elements found."

        return f"""You are a browser automation agent. Your job: determine the next action to accomplish the user's task.

{sequence_info}

═══════════════════════════════════════════════════════════════
WHAT YOU'VE DONE SO FAR
═══════════════════════════════════════════════════════════════
{history_block if history_block else "No actions yet."}

Navigation history:
{nav_summary}

═══════════════════════════════════════════════════════════════
AVAILABLE ELEMENTS (Overlays)
═══════════════════════════════════════════════════════════════
{overlays}

Format: Overlay <#> type=<button|input|link|...> [subtype=<text|email|...>] label="<text>" focused=<true|false>

FOCUS STATE EXPLAINED:
• focused=true → Element has keyboard focus
• If you need to type and input is focused=true → Just call type_text (don't click first)
• Clicking an already-focused element usually does nothing
• Don't click elements just to focus them if they're already focused

═══════════════════════════════════════════════════════════════
WHEN TO CALL complete_task
═══════════════════════════════════════════════════════════════
Call complete_task when:
✓ User's request is fulfilled (e.g., "click login" → you clicked login successfully)
✓ Data extraction done and stored in notebook
✓ Navigation completed successfully
✓ Form filled and submitted successfully

Do NOT call complete_task when:
✗ An action just failed
✗ You're waiting for a page to load (use defer_action instead)
✗ You're stuck and need help (use ask_user instead)
✗ You're in the middle of a multi-step process

{self._format_notebook(notebook)}

{self._format_extraction_schema() if self.extraction_schema else ""}

═══════════════════════════════════════════════════════════════
FUNCTIONS AVAILABLE
═══════════════════════════════════════════════════════════════
• click - Click an element (provide overlay index in overlay_index parameter)
• type_text - Type text into an input
• clear_text - Clear text from an input field
• select_option - Select option from dropdown
• upload_file - Upload a file
• set_datetime - Set date/time in picker
• press_key - Press keyboard key (Enter, Tab, Escape, etc.)
• scroll_page - Scroll up/down
• open_url - Navigate to URL
• go_back / go_forward - Browser navigation
• extract_data - Extract and store data in notebook
• remember_data - Store information for later use
• complete_task - Mark current task as complete
• complete_sequence - End sequential task (call after final turn)
• ask_user - Ask user for clarification
• talk_to_user - Send message to user (doesn't advance task)
• defer_action - Wait for page to load/element to appear

═══════════════════════════════════════════════════════════════
ERROR RECOVERY ESCALATION
═══════════════════════════════════════════════════════════════
After 1 failed attempt → Try different element or approach
After 2 failed attempts → Try scrolling or navigation
After 3 failed attempts → Use ask_user to get help
NEVER retry the exact same action that just failed

═══════════════════════════════════════════════════════════════
SPECIAL CASES
═══════════════════════════════════════════════════════════════
• Modal blocking page → Close modal first
• No elements found → Try scrolling
• CAPTCHA appears → Use ask_user
• Page still loading → Use defer_action
• Element list changed after scroll → This is normal

═══════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════
1. Only act on elements visible in screenshot
2. Be specific in element descriptions
3. type_text REPLACES content (doesn't append)
4. Check focused=true before clicking to focus
5. Don't repeat failed actions
6. For sequential tasks: complete_task after each turn, complete_sequence after final turn
{base_knowledge_section}

Choose the next action to take.
"""

    def _get_history_block(self, sequential_iteration: Optional[int] = None, just_data: bool = False) -> str:
        if not self.session_tracker:
            return ""
        return self.session_tracker.history_block(limit=self.interaction_summary_limit, sequential_iteration=sequential_iteration, just_data=just_data)

    def _format_notebook(self, notebook: Notebook) -> str:
        """Format notebook entries for inclusion in the prompt."""

        entries = notebook.to_list()
        if not entries:
            return ""

        notebook_str = ""
        for i, entry in enumerate(entries, 1):
            notebook_str += f"{i}. {entry.task} → {entry.data}\n"

        return f"""
═══════════════════════════════════════════════════════════════
NOTEBOOK (Your Stored Data)
═══════════════════════════════════════════════════════════════
{notebook_str}
• Use extract_data or remember_data to store information
• Check notebook to determine if extraction tasks are complete
• Partial matches OK (extracting "Jan 1, 1990" satisfies "extract birth date")
"""

    def _format_extraction_schema(self) -> str:
        """Format extraction schema for inclusion in the prompt."""
        if not self.extraction_schema:
            return ""

        import json

        required_fields = self.extraction_schema.get("required", [])
        properties = self.extraction_schema.get("properties", {})

        field_lines = []
        for field in required_fields:
            field_info = properties.get(field, {})
            description = field_info.get("description", "")
            field_lines.append(f"  • {field}: {description}")

        example = json.dumps({field: "..." for field in required_fields}, indent=2)

        return f"""
═══════════════════════════════════════════════════════════════
EXTRACTION SCHEMA (Use Exact Field Names)
═══════════════════════════════════════════════════════════════
Required fields:
{chr(10).join(field_lines)}

Example correct format:
{example}

⚠️ Use these EXACT field names when calling extract_data
"""

    def _summarize_navigation_history(self, url_history: List[str], url_pointer: Optional[int]) -> str:
        """Provide a concise navigation summary for the prompt."""
        if not url_history:
            return "No navigation history recorded yet."

        total = len(url_history)
        pointer = url_pointer if url_pointer is not None and 0 <= url_pointer < total else total - 1
        pointer = max(0, pointer)

        prev_url = url_history[pointer - 1] if pointer > 0 else None
        next_url = url_history[pointer + 1] if pointer < total - 1 else None

        start_idx = max(0, total - 3)  # Reduced for speed
        lines = []
        for idx in range(start_idx, total):
            marker = " (current)" if idx == pointer else ""
            lines.append(f"{idx}: {url_history[idx]}{marker}")

        history_block = "\n    ".join(lines)
        prev_line = f"Previous page (back target): {prev_url}" if prev_url else "Previous page (back target): none"
        next_line = f"Next page (forward target): {next_url}" if next_url else "Next page (forward target): none"

        return (
            f"Total pages visited: {total}\n"
            f"Current history index: {pointer}\n"
            f"{prev_line}\n"
            f"{next_line}\n"
            f"Recent history (oldest → newest):\n    {history_block}"
        )
