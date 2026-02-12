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
        task_target: Union[int, str] = 1,
        task_progress: int = 0,
        task_history: Optional[List[str]] = None,
        force_think: bool = False,
        current_iteration: int = 0,
        browser_actions_in_round: int = 0,
        checkpoint_mode: bool = False,
        suppress_mark_progress: bool = False,
        active_strategy: Optional[str] = None,
        last_action_summary: Optional[str] = None,
        tab_bar: Optional[str] = None,
        dialog_notice: Optional[str] = None,
        tab_events: Optional[List[str]] = None,
        dialog_pending: bool = False,
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
        self.task_target = task_target
        self.task_progress = task_progress
        self.task_history = task_history or []
        self.force_think = force_think
        self.current_iteration = current_iteration
        self.browser_actions_in_round = browser_actions_in_round
        self.checkpoint_mode = checkpoint_mode
        self.suppress_mark_progress = suppress_mark_progress
        self.active_strategy = active_strategy
        self.last_action_summary = last_action_summary
        self.tab_bar = tab_bar
        self.dialog_notice = dialog_notice
        self.tab_events = tab_events or []
        self.dialog_pending = dialog_pending

    def _build_reflection_block(self) -> str:
        """Build the reflection block for the user prompt.

        Contains:
        - ACTIVE STRATEGY: persistent reasoning from the last think(continue), shown every turn until cleared
        - LAST ACTION: what the agent just did and the result
        - TAB EVENTS: tab opens/closes/dialog events since last turn
        """
        parts = []

        if self.active_strategy:
            parts.append(f"ACTIVE STRATEGY:\n{self.active_strategy}\n")

        if self.last_action_summary:
            parts.append(f"LAST ACTION:\n{self.last_action_summary}\n")

        if self.tab_events:
            events_str = "\n".join(f"- {e}" for e in self.tab_events)
            parts.append(f"TAB EVENTS:\n{events_str}\n")

        if not parts:
            return ""

        return "\n".join(parts) + "\n"

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
        from agent.action_tools import get_filtered_tools

        try:
            # Build reflection block (active strategy + last action + tab events)
            reflection = self._build_reflection_block()

            # Dialog notice goes at the very top of user prompt (highest attention)
            dialog_prefix = ""
            if self.dialog_notice:
                dialog_prefix = f"{self.dialog_notice}\n"

            if self.checkpoint_mode:
                user_prompt = f"""{dialog_prefix}{reflection}Task: {self.user_prompt}
Progress so far: {self.task_progress}/{self.task_target} recorded. Your last action is NOT yet counted.

If your last action completed a unit of work, call mark_progress to record it.
If you need more actions before this counts as progress, call think with next_action=continue.
"""
            else:
                user_prompt = f"""{dialog_prefix}{reflection}You are currently trying to: {self.user_prompt}

Based on the screenshot, what is the best next action?
"""

            # Get filtered tools based on current state
            tools = get_filtered_tools(
                suppress_mark_progress=self.suppress_mark_progress,
                checkpoint_mode=self.checkpoint_mode,
                dialog_pending=self.dialog_pending,
            )

            system_prompt = self._build_function_calling_system_prompt(
                self.session_tracker,
                environment_state,
                notebook,
                element_data,
            )

            # Generate action using function calling
            result = generate_action_with_tools(
                prompt=user_prompt,
                tools=tools,
                system_prompt=system_prompt,
                image=screenshot,
                image_detail=self.image_detail,
                model=self.model_name,
                reasoning_level=self.reasoning_level,
                tool_choice="required",
                parallel_tool_calls=self.max_actions_per_plan > 1,
            )

            actions = []
            # Limit to max_actions_per_plan to prevent excessive batching
            get_event_logger().system_debug(f"LLM returned {len(result)} tool calls, limiting to {self.max_actions_per_plan}")
            limited_result = result[:self.max_actions_per_plan]

            for action in limited_result:
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

        # Build progress context
        progress_info = ""
        if self.task_target != 1 or self.task_progress > 0 or self.task_history:
            # Format target display
            target_display = self.task_target if isinstance(self.task_target, str) else f"{self.task_target}"
            numeric_target = self.task_target if isinstance(self.task_target, int) else None
            current_round = self.task_progress + 1

            if self.task_progress > 0 and self.task_history:
                # Mid-task: per-iteration reframing
                last_completed = self.task_history[-1]
                is_round_start = self.browser_actions_in_round == 0

                history_lines = ""
                for i, item in enumerate(self.task_history, 1):
                    history_lines += f"  {i}. {item} ✓\n"

                if numeric_target:
                    remaining = numeric_target - self.task_progress

                    # Only show the "do it again" nudge at the start of a new round
                    round_start_hint = (
                        f'You just finished: "{last_completed}"\n'
                        f'What you see on screen is from your previous round — do it again.\n'
                    ) if is_round_start else ""

                    progress_info = f"""
═══════════════════════════════════════════════════════════════
ROUND {current_round} OF {target_display}
═══════════════════════════════════════════════════════════════

Task: {self.user_prompt}

Completed:
{history_lines}
{round_start_hint}{remaining} round{"s" if remaining != 1 else ""} left.

• After each round, call mark_progress to record what you did
• mark_progress only counts after real browser actions — you must actually do the work each time

"""
                else:
                    # Open-ended target ("all")
                    round_start_hint = (
                        f'You just finished: "{last_completed}"\n'
                        f'What you see on screen may include results of your previous work.\n'
                    ) if is_round_start else ""

                    progress_info = f"""
═══════════════════════════════════════════════════════════════
PROGRESS ({self.task_progress} done so far)
═══════════════════════════════════════════════════════════════

Task: {self.user_prompt}

Completed:
{history_lines}
{round_start_hint}Keep going — look for more to do.

• After each unit of work, call mark_progress to record it
• When there's nothing left to do, call mark_progress with done=true

"""
            else:
                # First iteration: no progress yet
                if numeric_target:
                    progress_info = f"""
═══════════════════════════════════════════════════════════════
ROUND 1 OF {target_display}
═══════════════════════════════════════════════════════════════

Task: {self.user_prompt}

No rounds completed yet — get started.

• After each round, call mark_progress to record what you did
• mark_progress only counts after real browser actions — you must actually do the work each time

"""
                else:
                    progress_info = f"""
═══════════════════════════════════════════════════════════════
TASK PROGRESS
═══════════════════════════════════════════════════════════════

Task: {self.user_prompt}
Progress: 0/{target_display}

No progress yet — get started.

• After each unit of work, call mark_progress to record it
• When there's nothing left to do, call mark_progress with done=true

"""

        # Add forced think prompt if stuck
        forced_think_prompt = ""
        if self.force_think:
            forced_think_prompt = """
═══════════════════════════════════════════════════════════════
⚠️ STUCK DETECTION - THINK FIRST
═══════════════════════════════════════════════════════════════

You've been working for a while without making progress.

Before doing anything else, call think() and reason about:
• What's going wrong? Am I stuck in a loop?
• Is there a different approach I should try?
• Should I scroll to find more content?
• Should I stop and report what I've accomplished so far?

After thinking, either:
• Try a different approach
• Call mark_progress with done=true if you're truly stuck

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

        # Build tab bar section (only shown when 2+ tabs are open)
        tab_section = ""
        if self.tab_bar:
            tab_section = f"""
═══════════════════════════════════════════════════════════════
OPEN TABS
═══════════════════════════════════════════════════════════════
{self.tab_bar}

"""

        return f"""You are controlling a web browser. You can see the current page as a screenshot.
Look at what's on screen, decide what to do, and do it — just like a person would.

{forced_think_prompt}

{progress_info}
{tab_section}
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

FOCUS STATE:
• focused=true → Element already has keyboard focus
• If you need to type and input is focused=true → Just call type_text (don't click first)
• Clicking an already-focused element usually does nothing

{self._format_notebook(notebook)}

═══════════════════════════════════════════════════════════════
AVAILABLE FUNCTIONS
═══════════════════════════════════════════════════════════════

BROWSER ACTIONS:
• click - Click an element
• type_text - Type text into an input
• clear_text - Clear text from an input field
• select_option - Select option from dropdown
• upload_file - Upload a file
• set_datetime - Set date/time in picker
• press_key - Press keyboard key (Enter, Tab, Escape, etc.)
• scroll_page - Scroll up/down
• open_url - Navigate to URL
• go_back / go_forward - Browser navigation

DATA & COMMUNICATION:
• extract_data - Extract and store data in notebook
• ask_user - Ask user for clarification
• flag - Send non-blocking notification to user

TAB MANAGEMENT:
• switch_tab - Switch to a different browser tab
• close_tab - Close a browser tab
• open_tab - Open a new tab (optionally with URL)
• dismiss_dialog - Accept or dismiss a JavaScript dialog

COGNITIVE ACTIONS:
• think - Stop and reason about your situation (no browser action)
• assert_condition - Check if something is true from the screenshot
• mark_progress - Record completion of a unit of work
• revise_target - Adjust your target mid-execution
• wait_for - Wait for a condition with timeout

═══════════════════════════════════════════════════════════════
ERROR RECOVERY
═══════════════════════════════════════════════════════════════
If something isn't working:
• Try a different element or approach
• Scroll if you can't find what you need
• Use think() to reason about what's wrong
• Use flag() to notify the user of issues
• Don't repeat the exact same failed action

═══════════════════════════════════════════════════════════════
GUIDELINES
═══════════════════════════════════════════════════════════════
1. Only act on elements visible in screenshot
2. Be specific in element descriptions
3. type_text REPLACES content (doesn't append)
4. Check focused=true before clicking to focus
5. Don't repeat failed actions
6. Use natural language when marking progress
{base_knowledge_section}

Choose the next action to take.
"""

    def _get_history_block(self, just_data: bool = False) -> str:
        if not self.session_tracker:
            return ""
        return self.session_tracker.history_block(limit=self.interaction_summary_limit, just_data=just_data)

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
• Use extract_data to store information
• Check notebook to see what you've already collected
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
