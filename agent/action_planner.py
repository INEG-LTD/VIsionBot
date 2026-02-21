"""
Action Planner - Determines the next viewport-aware plan based on current state.

Step 2: LLM-based action planning that builds a sequence of actions you can execute
before the viewport changes, relying only on what is visible.
"""

from typing import Optional, List, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import re

from agent.memory import NarrativeMemory
from agent.notebook import Notebook
from agent.agent_context import EnvironmentState
from agent.prompts import (
    DecisionContext,
    SHARED_CONTRADICTION_GATE,
    get_memory_developer_policy,
    render_decision_context,
)
from lib.ai import (
    generate_model,
    generate_action_with_tools,
    ReasoningLevel,
    get_default_agent_model,
    get_default_agent_reasoning_level,
)
from models.models import ActionPlan, ActionStep, FailedAction, NotebookEntryType, PageElements
from utils.event_logger import get_event_logger


def strip_targeting_data(step: str) -> str:
    """Remove element_ids from a recommended step string.

    The LLM should only see the action intent (e.g. 'click: Martin bot avatar')
    and must determine fresh targeting from the current element index.
    Stale IDs cause the LLM to copy them verbatim instead of looking.
    """
    # Remove element_id patterns: "[id=123]" or "element_id=123"
    step = re.sub(r'\s*\[id=\d+\]', '', step)
    step = re.sub(r'\s*element_id\s*=\s*\d+', '', step)
    # Clean up trailing whitespace / colons
    step = re.sub(r'\s*:\s*$', '', step).strip()
    return step or "continue with the recommended action"


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
        memory_store: NarrativeMemory,
        base_knowledge: Optional[List[str]] = None,
        *,
        model_name: Optional[str] = None,
        reasoning_level: Union[ReasoningLevel, str, None] = None,
        image_detail: str = "high",
        interaction_summary_limit: Optional[int] = None,
        include_visible_text_in_agent_context: bool = False,
        max_actions_per_plan: int = 6,
        current_iteration: int = 0,
        user_facing_actions_in_round: int = 0,
        checkpoint_mode: bool = False,
        active_strategy: Optional[str] = None,
        last_action_summary: Optional[str] = None,
        tab_bar: Optional[str] = None,
        dialog_notice: Optional[str] = None,
        tab_events: Optional[List[str]] = None,
        dialog_pending: bool = False,
        recommended_next_step: Optional[str] = None,
        recommended_next_step_source_id: Optional[str] = None,
        decision_context: Optional[DecisionContext] = None,
        element_index_text: Optional[str] = None,
        gallery_images: Optional[List[bytes]] = None,
        user_hints: Optional[List[str]] = None,
        # Loop state
        in_loop: bool = False,
        loop_round: int = 0,
        loop_count: Optional[int] = None,
        loop_description: str = "",
        recent_actions: Optional[List[str]] = None,
        iterations_remaining: int = 0,
        max_iterations: int = 0,
        budget_spent: int = 0,
        budget_phase: str = "normal",
        low_budget_mode: bool = False,
        budget_constraints_enabled: bool = True,
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
        self._system_prompt_cache: dict[str, str] = {}
        self.interaction_summary_limit = interaction_summary_limit
        self.include_visible_text_in_agent_context = include_visible_text_in_agent_context
        self.memory_store: NarrativeMemory = memory_store
        self.max_actions_per_plan = max_actions_per_plan
        self.current_iteration = current_iteration
        self.user_facing_actions_in_round = user_facing_actions_in_round
        self.checkpoint_mode = checkpoint_mode
        self.active_strategy = active_strategy
        self.last_action_summary = last_action_summary
        self.tab_bar = tab_bar
        self.dialog_notice = dialog_notice
        self.tab_events = tab_events or []
        self.dialog_pending = dialog_pending
        self.recommended_next_step = recommended_next_step
        self.recommended_next_step_source_id = recommended_next_step_source_id
        self.decision_context = decision_context
        self.element_index_text = element_index_text
        self.gallery_images = gallery_images
        self.user_hints = [hint.strip() for hint in (user_hints or []) if isinstance(hint, str) and hint.strip()]
        # Loop state
        self.in_loop = in_loop
        self.loop_round = loop_round
        self.loop_count = loop_count
        self.loop_description = loop_description
        self.recent_actions = recent_actions or []
        self.iterations_remaining = iterations_remaining
        self.max_iterations = max_iterations
        self.budget_spent = max(0, int(budget_spent or 0))
        self.budget_phase = str(budget_phase or "normal").strip().lower() or "normal"
        self.low_budget_mode = bool(low_budget_mode)
        self.budget_constraints_enabled = bool(budget_constraints_enabled)

    def _build_reflection_block(self) -> str:
        """Build the reflection block for the user prompt.

        Contains:
        - LOOP STATUS: current loop round/count when in a loop
        - ACTIVE STRATEGY: persistent reasoning from the last think(continue)
        - LAST ACTION: what the agent just did and the result
        - RECENT ACTIONS: compact log of last ~10 actions
        - TAB EVENTS: tab opens/closes/dialog events since the last iteration
        """
        parts = []

        # Budget / iteration budget awareness
        if self.max_iterations > 0:
            mode = "ON" if self.low_budget_mode else "OFF"
            parts.append(
                "BUDGET STATE:\n"
                f"- spent={self.budget_spent}\n"
                f"- remaining={self.iterations_remaining}\n"
                f"- total={self.max_iterations}\n"
                f"- phase={self.budget_phase}\n"
                f"- low_budget_mode={mode}\n"
            )
            if self.budget_constraints_enabled:
                parts.append(
                    "BUDGET CONTRACT (MANDATORY IN REASONING):\n"
                    '- Include "State: ..."\n'
                    '- Include "Budget: spent=X, remaining=Y, total=Z"\n'
                    '- Include "Why: ..."\n'
                )
                if self.iterations_remaining <= 5:
                    parts.append("You are near budget exhaustion — prioritize completion-oriented actions.\n")

        # Loop framing
        if self.in_loop and self.loop_count:
            remaining = max(0, self.loop_count - self.loop_round + 1)
            parts.append(
                f"═══ LOOP {self.loop_round} OF {self.loop_count}: {self.loop_description} ═══\n"
                f"Elements marked [DONE] in the index have already been handled.\n"
                f"{remaining} round{'s' if remaining != 1 else ''} remaining.\n"
                f"After completing this round, call think(next_action=advance).\n"
            )

        if self.active_strategy:
            parts.append(f"ACTIVE STRATEGY:\n{self.active_strategy}\n")

        if self.last_action_summary:
            parts.append(f"LAST ACTION:\n{self.last_action_summary}\n")

        if self.user_hints:
            hints_str = "\n".join(f"- {h}" for h in self.user_hints)
            parts.append(f"HINTS FROM USER:\n{hints_str}\n")

        if self.recommended_next_step:
            source = (
                f" (source: {self.recommended_next_step_source_id})"
                if self.recommended_next_step_source_id
                else ""
            )
            cleaned_step = self._strip_targeting_data(self.recommended_next_step)
            parts.append(
                f"RECOMMENDED NEXT STEP (ONE SHOT){source}:\n{cleaned_step}\n"
            )

        if self.checkpoint_mode and self.user_facing_actions_in_round > 0:
            parts.append(
                "CHECKPOINT:\n"
                "You just performed a user-facing action. Decide what to do next via think().\n"
            )

        # Recent actions log
        if self.recent_actions:
            actions_str = "\n".join(
                f"  {i+1}. {a}" for i, a in enumerate(self.recent_actions)
            )
            parts.append(f"RECENT ACTIONS:\n{actions_str}\n")

        if self.tab_events:
            events_str = "\n".join(f"- {e}" for e in self.tab_events)
            parts.append(f"TAB EVENTS:\n{events_str}\n")

        if not parts:
            return ""

        return "\n".join(parts) + "\n"

    @staticmethod
    def _strip_targeting_data(step: str) -> str:
        """Delegate to module-level helper."""
        return strip_targeting_data(step)

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

            continuation_mode = ""
            if self.active_strategy:
                continuation_mode = """
ACTIVE STRATEGY CONTINUATION MODE
You already have an ACTIVE STRATEGY. Continue it.

Rules:
1. Do not create a new plan unless stuck or page changed enough.
2. The screenshot is the source of truth. If ACTIVE STRATEGY conflicts with what's visible, update immediately.
3. Do not restate the full plan.
4. If you call think with next_action=continue, only provide a brief first-person status + next step.
5. If your recent attempts didn't create visible progress, call think(next_action=stuck).
6. You should keep working after next_action=stuck — it's a strategy switch, not completion.
7. For every non-think tool call, reasoning must explain how the action advances the ACTIVE STRATEGY.
"""

            decision_context_block = (
                render_decision_context(self.decision_context)
                if self.decision_context is not None
                else (
                    f"Planning iteration: unknown\n"
                    f"Action iteration: {self.current_iteration}\n"
                    f"Mission: {environment_state.user_prompt}\n"
                    f"Current page: {environment_state.current_url} — {environment_state.page_title}\n"
                    f"Budget: spent={self.budget_spent}, remaining={self.iterations_remaining}, total={self.max_iterations}\n"
                    f"Budget phase: {self.budget_phase}\n"
                    f"Low-budget mode: {'on' if self.low_budget_mode else 'off'}"
                )
            )

            if self.checkpoint_mode:
                user_prompt = f"""{dialog_prefix}{reflection}{continuation_mode}Mission: {self.user_prompt}
Decision context:
{decision_context_block}

CHECKPOINT — choose via think():
• think(next_action=continue) — more work needed for the mission
• think(next_action=start_loop, loop_count=N, loop_description="...") — need to repeat an action N times
• think(next_action=advance) — (in loop) current iteration done, move to next round
• think(next_action=done) — mission is fully complete
• think(next_action=stuck) — strategy failed, provide new one

{SHARED_CONTRADICTION_GATE}
"""
            else:
                user_prompt = f"""{dialog_prefix}{reflection}{continuation_mode}Mission: {self.user_prompt}

Decision context:
{decision_context_block}

Based on the screenshot, what is the best next action?

{SHARED_CONTRADICTION_GATE}
"""

            # Get filtered tools based on current state
            tools = get_filtered_tools(
                checkpoint_mode=self.checkpoint_mode,
                dialog_pending=self.dialog_pending,
                budget_constraints_enabled=self.budget_constraints_enabled,
            )

            system_prompt = self._build_function_calling_system_prompt(
                environment_state,
                notebook,
                element_data,
            )

            # Build image arguments
            # Element index mode: clean screenshot + gallery pages
            # Other modes: just the screenshot
            image_arg = screenshot
            multi_image_arg = None
            if self.gallery_images:
                multi_image_arg = [screenshot] + self.gallery_images
                image_arg = None

            # Generate action using function calling
            result = generate_action_with_tools(
                prompt=user_prompt,
                tools=tools,
                system_prompt=system_prompt,
                developer_prompt=get_memory_developer_policy(self.budget_constraints_enabled),
                image=image_arg,
                multi_image=multi_image_arg,
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
                actions.append(action_step)

            return actions, None

        except Exception as e:
            get_event_logger().system_error(f"Error generating action with function calling: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error generating action: {e}"

    def _build_function_calling_system_prompt(
        self,
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

        user_hints_section = ""
        if self.user_hints:
            user_hints_section = "\n\nHINTS FROM USER:\n"
            for hint in self.user_hints:
                user_hints_section += f"- {hint}\n"

        # Get history and navigation info
        memory_narrative_block = self._get_memory_narrative_block()
        memory_index_block = self._get_memory_entry_index()
        executed_action_ledger = self.memory_store.get_executed_action_ledger(n=20)
        stuck_hint_lines = self.memory_store.get_stuck_pattern_hints()
        nav_summary = self._summarize_navigation_history(
            getattr(state, "url_history", []),
            getattr(state, "url_pointer", None)
        )

        # Build loop context (only shown when in a loop)
        progress_info = ""

        forced_think_prompt = ""

        # Build tab bar section (only shown when 2+ tabs are open)
        tab_section = ""
        if self.tab_bar:
            tab_section = f"""
═══════════════════════════════════════════════════════════════
OPEN TABS
═══════════════════════════════════════════════════════════════
{self.tab_bar}

"""

        stuck_hints = ", ".join(stuck_hint_lines) if stuck_hint_lines else "none right now"

        # Build element index section
        gallery_note = ""
        if self.gallery_images:
            n = len(self.gallery_images)
            gallery_note = (
                f"\nYou are also shown {n} CROP GALLERY image{'s' if n > 1 else ''} "
                f"after the screenshot. Elements marked 'SEE CROP GALLERY' in the "
                f"index have visual crops in these gallery images."
            )
        budget_constraints_status = "enabled" if self.budget_constraints_enabled else "disabled"
        if self.budget_constraints_enabled:
            budget_reasoning_contract = """12. Every tool-call reasoning must include:
    - State: current task/page state
    - Budget: numeric spent/remaining/total
    - Why: why this is the most efficient next action
13. Every tool call arguments object must include:
    - budget_spent
    - budget_remaining
    - budget_total
    and these values must exactly match the Budget status shown above."""
        else:
            budget_reasoning_contract = """12. Every tool-call reasoning should include:
    - State: current task/page state
    - Why: why this is the most efficient next action
13. Budget fields (budget_spent, budget_remaining, budget_total) are optional when budget constraints are disabled."""
        opening_instruction = (
            "You are controlling a web browser. You can see the current page "
            "as a screenshot." + gallery_note + "\n"
            "Use the INTERACTIVE ELEMENTS index below to identify elements by their [id] number."
        )
        element_section = f"""═══════════════════════════════════════════════════════════════
INTERACTIVE ELEMENTS
═══════════════════════════════════════════════════════════════
{self.element_index_text or "No interactive elements detected."}

HOW TO CLICK:
• Pick the [id] number from the list above and pass it as element_id.
• For elements marked 'SEE CROP GALLERY', look at the gallery images to
  visually identify the element before clicking.
• The system clicks the element directly by its DOM reference — no coordinate
  estimation needed.

FOCUS STATE:
• focused → Element already has keyboard focus
• If you need to type and input is focused → Just call type_text (don't click first)"""

        return f"""{opening_instruction}

{forced_think_prompt}

{progress_info}
{tab_section}
═══════════════════════════════════════════════════════════════
WHAT YOU'VE DONE SO FAR
═══════════════════════════════════════════════════════════════
{memory_narrative_block if memory_narrative_block else "No actions yet."}
Executed action ledger (facts only):
{executed_action_ledger}

Mission: {self.user_prompt}

Budget status:
- spent={self.budget_spent}
- remaining={self.iterations_remaining}
- total={self.max_iterations}
- phase={self.budget_phase}
- low_budget_mode={"on" if self.low_budget_mode else "off"}
- constraints={budget_constraints_status}

Potential stuck patterns from memory scan: {stuck_hints}

Memory ID ledger (for citing any prior memory entry):
{memory_index_block}

Navigation history:
{nav_summary}

{element_section}

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
• scroll_page - Scroll the page or a specific container
  Examples:
    scroll_page(direction="down")                              ← scroll main page
    scroll_page(direction="down", element_id=42)               ← scroll the modal/sidebar/panel containing element 42
    scroll_page(direction="up", element_id=17)                 ← scroll up inside a sidebar (pass any element inside it)
    scroll_page(direction="down", scroll_to_element_id=55)     ← bring element 55 into view
  When to use element_id: any time you see a modal, drawer, dropdown list, chat panel, or overflow
  area — pick any element visible inside it and pass its id. The system finds the scrollable container.
• open_url - Navigate to URL
• go_back / go_forward - Browser navigation

DATA & COMMUNICATION:
• extract_data - Extract and store data in notebook
• ask_user - Ask user for clarification
• report_data - Send textual data back to the host callback
• write_data - Write text data to disk
• send_email - Send an email via the Resend API
• flag - Send non-blocking notification to user

TAB MANAGEMENT:
• switch_tab - Switch to a different browser tab
• close_tab - Close a browser tab
• open_tab - Open a new tab (optionally with URL)
• dismiss_dialog - Accept or dismiss a JavaScript dialog

COGNITIVE ACTIONS:
• think - Stop and reason; decide next via next_action param
  - continue: keep working
  - start_loop: begin repeating an action (provide loop_count, loop_description)
  - advance: (in loop) current iteration done, move to next round
  - end_loop: exit loop early
  - done: mission complete
  - stuck: switch strategy
• assert_condition - Check if something is true from the screenshot
• wait_for - Wait for a condition with timeout

═══════════════════════════════════════════════════════════════
LOOPS
═══════════════════════════════════════════════════════════════
When you need to repeat an action for multiple targets:
1. Do the first action normally.
2. At the checkpoint, call think(next_action="start_loop", loop_count=N, loop_description="...").
   N includes the action you already did (round 1).
3. Each subsequent round: do the action, then think(next_action="advance").
4. Elements you've already interacted with show [DONE] in the element index.
5. The loop auto-completes after the last round, or use think(next_action="end_loop") to exit early.

═══════════════════════════════════════════════════════════════
ERROR RECOVERY
═══════════════════════════════════════════════════════════════
If something isn't working:
• Try a different element or approach
• Scroll if you can't find what you need
• Use think() to reason about what's wrong
• Use flag() to notify the user of issues
• Don't repeat the exact same failed action

STUCK STRATEGY SWITCH RULE:
• If my recent attempts did not create visible progress, I should not keep the same strategy
• I should call think(next_action=stuck)
• In that call:
  - reasoning should be natural first-person language for my new ACTIVE STRATEGY
  - recommended_next_step should be one concrete immediate action

{SHARED_CONTRADICTION_GATE}

═══════════════════════════════════════════════════════════════
GUIDELINES
═══════════════════════════════════════════════════════════════
1. Only act on elements visible in the screenshot — click by their visual position
2. Be specific in element descriptions
3. type_text REPLACES content (doesn't append)
4. Check focused=true before clicking to focus
5. Don't repeat failed actions
6. When ACTIVE STRATEGY is present, think(next_action=continue) should be a brief natural first-person status + immediate next step (no full re-plan)
7. If what you planned conflicts with the current screenshot, follow the screenshot and adjust plan
8. Stick exactly to the mission. Do not add extra steps, verification, confirmations, or sub-tasks that the mission did not ask for. If the mission is simple, your strategy should be simple.
9. When setting an ACTIVE STRATEGY, keep it minimal — only describe what is directly required to complete the mission as stated. Do not elaborate beyond that.
10. When ACTIVE STRATEGY is present, each non-think tool call reasoning should explicitly state
    how that action advances the ACTIVE STRATEGY
11. Reference relevant memory entries (mem_XXXXXX) in your reasoning. If a RECOMMENDED NEXT STEP is present, follow it or explain why you're deviating.
{budget_reasoning_contract}
{base_knowledge_section}
{user_hints_section}

Choose the next action to take.
"""

    def _get_memory_narrative_block(self, just_data: bool = False) -> str:
        if not self.memory_store:
            return ""
        limit = self.interaction_summary_limit or 20
        narrative = self.memory_store.get_narrative(n=limit)
        if just_data:
            return narrative
        return f"Recent memory narrative:\n{narrative}"

    def _get_memory_entry_index(self, max_lines: int = 120) -> str:
        if not self.memory_store or not self.memory_store.entries:
            return "No memory entries yet."

        entries = self.memory_store.entries
        total = len(entries)

        if total <= max_lines:
            selected = entries
        else:
            head = max_lines // 3
            tail = max_lines // 3
            middle = max_lines - head - tail
            mid_start = max((total // 2) - (middle // 2), 0)
            selected = entries[:head] + entries[mid_start: mid_start + middle] + entries[-tail:]

        lines: List[str] = []
        for entry in selected:
            lines.append(
                f"[{entry.memory_id}] {entry.entry_kind} | {entry.action_type} -> {entry.outcome}"
            )

        if total > len(selected):
            lines.append(f"... {total - len(selected)} additional memory entries omitted for prompt size ...")

        return "\n".join(lines)

    def _format_notebook(self, notebook: Notebook) -> str:
        """Format notebook entries for inclusion in the prompt."""

        entries = notebook.to_list()
        if not entries:
            return ""

        notebook_str = ""
        for i, entry in enumerate(entries, 1):
            notebook_str += f"{i}. {entry.description} → {entry.data}\n"

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

        recent_urls_block = "\n    ".join(lines)
        prev_line = f"Previous page (back target): {prev_url}" if prev_url else "Previous page (back target): none"
        next_line = f"Next page (forward target): {next_url}" if next_url else "Next page (forward target): none"

        return (
            f"Total pages visited: {total}\n"
            f"Current history index: {pointer}\n"
            f"{prev_line}\n"
            f"{next_line}\n"
            f"Recent history (oldest → newest):\n    {recent_urls_block}"
        )
