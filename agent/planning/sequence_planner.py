"""
Sequence Planner - Manages sequential task iteration and progression.

This module provides the SequencePlanner class that operates within sequential tasks
to determine next steps or end the sequence based on current state and history.
"""

from typing import Optional, List, Dict, Any, Union
import json

from models.models import (
    Sequence,
    SequenceDecision,
    TurnResult,
)
from core.config import SequentialTaskConfig
from agent.agent_context import EnvironmentState
from agent.notebook import Notebook
from lib.ai import (
    generate_model,
    ReasoningLevel,
    get_default_agent_model,
    get_default_agent_reasoning_level,
)


class SequencePlanner:
    """
    Operates within sequential tasks to determine next steps.
    Has access to sequence history and current page state.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        reasoning_level: Optional[ReasoningLevel] = None,
        config: Optional[SequentialTaskConfig] = None,
    ):
        """
        Initialize the SequencePlanner.

        Args:
            model_name: Model to use for planning (default: agent model, or config override)
            reasoning_level: Reasoning level for planning (default: agent reasoning level, or config override)
            config: Sequential task configuration
        """
        self.config = config or SequentialTaskConfig()

        # Use config overrides if provided, otherwise fall back to defaults
        self.model_name = (
            self.config.sequence_planner_model
            or model_name
            or get_default_agent_model()
        )
        self.reasoning_level = (
            self.config.sequence_planner_reasoning_level
            or reasoning_level
            or get_default_agent_reasoning_level()
        )

    def decide_next_action(
        self,
        sequential_task: Sequence,
        environment_state: EnvironmentState,
        screenshot: bytes,
        overlay_data: Optional[List[Dict[str, Any]]] = None,
        notebook: Optional[Union[Notebook, List[Dict[str, Any]]]] = None,
    ) -> SequenceDecision:
        """
        Decides what to do next within a sequential task.

        Args:
            sequential_task: The sequential task being executed
            environment_state: Current browser/agent state
            screenshot: Current page screenshot
            overlay_data: Current page overlays (interactive elements)
            notebook: Agent's extracted data (Notebook or list of dicts).

        Returns:
            SequenceDecision with either a new task or end signal

        Raises:
            Exception: If LLM call fails or returns invalid decision
        """
        # Check if we should end based on hard limits
        if self.should_end_sequence(sequential_task):
            return SequenceDecision(
                decision="end_sequence",
                reasoning=self._get_end_reason(sequential_task),
                completion_reason=self._get_end_reason(sequential_task),
            )

        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_sequence_planner_prompt(
            sequential_task,
            environment_state,
            overlay_data,
            notebook,
        )

        # Call LLM for decision
        try:
            decision: SequenceDecision = generate_model(
                prompt=user_prompt,
                model_object_type=SequenceDecision,
                system_prompt=system_prompt,
                image=screenshot,
                image_detail="low",  # Use low detail for faster iteration
                model=self.model_name,
                reasoning_level=self.reasoning_level,
            )
            return decision
        except Exception as e:
            # If LLM call fails, default to ending sequence with error
            return SequenceDecision(
                decision="end_sequence",
                reasoning=f"Failed to generate next action: {str(e)}",
                completion_reason=f"Error in Sequence Planner: {str(e)}",
            )

    def should_retry_iteration(
        self,
        sequential_task: Sequence,
        last_error: Optional[str] = None,
    ) -> bool:
        """
        Determines if current iteration should be retried.

        Args:
            sequential_task: The sequential task being executed
            last_error: Optional error message from last attempt

        Returns:
            True if turn_attempts < max_attempts_per_iteration
        """
        return (
            sequential_task.state.turn_attempts
            < self.config.max_attempts_per_iteration
        )

    def should_end_sequence(
        self,
        sequential_task: Sequence,
    ) -> bool:
        """
        Checks if sequence should end based on completion strategy.

        Evaluates:
        - Total iterations vs max_total_iterations
        - Success count vs target (for strict mode)
        - Success rate vs threshold (for threshold mode)
        - fail_fast + any failures

        Args:
            sequential_task: The sequential task being executed

        Returns:
            True if sequence should end (safety limits or completion criteria met)
        """
        state = sequential_task.state

        # Safety limit: max total iterations
        if state.current_turn >= self.config.max_total_iterations:
            return True

        # Fail fast: end on first failure
        if self.config.fail_fast and state.total_failure_count > 0:
            return True

        # If we have a target count, check completion strategies
        if sequential_task.target_count is not None:
            target = sequential_task.target_count

            # Check if we've attempted all target iterations
            total_completed = state.total_success_count + state.total_failure_count
            attempted_all = total_completed >= target

            if self.config.completion_strategy == "strict":
                # Only end if we have exactly target successes
                return state.total_success_count >= target

            elif self.config.completion_strategy == "best_effort":
                # End after attempting all iterations
                return attempted_all

            elif self.config.completion_strategy == "threshold":
                # End if we've attempted all AND met threshold
                if attempted_all:
                    success_rate = (
                        state.total_success_count / total_completed
                        if total_completed > 0
                        else 0.0
                    )
                    return success_rate >= self.config.success_threshold
                # Or end if we've reached target successes
                return state.total_success_count >= target

        # For indefinite sequences (no target_count), don't end here
        # Let the LLM decide based on page state
        return False

    def _get_end_reason(self, sequential_task: Sequence) -> str:
        """
        Generates human-readable reason for ending sequence.

        Args:
            sequential_task: The sequential task being executed

        Returns:
            Explanation of why sequence is ending
        """
        state = sequential_task.state

        if state.current_turn >= self.config.max_total_iterations:
            return f"Maximum iteration limit reached ({self.config.max_total_iterations})"

        if self.config.fail_fast and state.total_failure_count > 0:
            return f"Fail-fast enabled: sequence ended after {state.total_failure_count} failure(s)"

        if sequential_task.target_count is not None:
            target = sequential_task.target_count

            if self.config.completion_strategy == "strict":
                if state.total_success_count >= target:
                    return f"Target reached: {state.total_success_count}/{target} successful iterations"

            elif self.config.completion_strategy == "best_effort":
                total_completed = state.total_success_count + state.total_failure_count
                return f"All iterations attempted: {state.total_success_count} successes, {state.total_failure_count} failures out of {target} target"

            elif self.config.completion_strategy == "threshold":
                total_completed = state.total_success_count + state.total_failure_count
                success_rate = (
                    state.total_success_count / total_completed
                    if total_completed > 0
                    else 0.0
                )
                return f"Threshold met: {success_rate:.1%} success rate ({state.total_success_count}/{total_completed}) meets {self.config.success_threshold:.1%} threshold"

        return "Sequence completion criteria met"

    def _build_system_prompt(self) -> str:
        """
        Builds the system prompt for Sequence Planner.

        Returns:
            System prompt string
        """
        return """You are a Sequence Planner for a browser automation agent executing a sequential task.

Your role is to decide what should happen next in the sequence.

You have exactly TWO decision options:

1. GENERATE NEXT TASK
   - Create a new, specific Normal Task for the current iteration
   - The task must be:
     * Immediately executable (grounded in current page state)
     * Specific (not abstract - reference actual visible elements)
     * Aligned with the sequential task
     * Informed by iteration history (don't repeat failed approaches)

   Example: "Extract the company name from the 3rd job listing card currently visible"
   (NOT: "Extract company name" - too vague)

2. END SEQUENCE
   - Signal that the sequential task is complete
   - Reasons to end:
     * Completion condition is satisfied
     * No more data available to process
     * Unrecoverable error state
     * Task has been achieved

DECISION CRITERIA:

When to GENERATE NEXT TASK:
- Completion condition not yet met
- More data/elements are available to process
- Current iteration has remaining retry attempts (if previous task failed)
- Task still requires more work

When to END SEQUENCE:
- Completion condition is satisfied (e.g., collected enough items)
- No more relevant elements visible on page
- Task has been achieved
- Unrecoverable error state (same error repeated many times)

IMPORTANT GUIDELINES:
- Always look at the screenshot FIRST before generating a task
- Reference specific, visible elements in your generated tasks
- If a previous attempt failed, try a different approach (different selector, different strategy)
- Don't generate duplicate tasks that were just attempted
- Consider the completion condition when deciding to end
- If you see the same elements/state as previous iterations, you may be stuck - consider ending

CRITICAL RULE: Generated tasks must ALWAYS match the sequential task
- If the task is "Extract from N items", EVERY task must be an extraction task
- Example: "Extract job title from 5th listing" (correct)
- Example: "Scroll down to see 5th listing" (WRONG - this is NOT an extraction task)
- The mini-loop will handle scrolling, clicking, and navigation automatically
- Your job is to specify WHAT to extract/accomplish, not HOW to navigate
- NEVER generate pure navigation tasks (scroll, click next page) - the agent will handle that

CRITICAL: AVOIDING DUPLICATE EXTRACTIONS
When extracting from multiple similar elements (job listings, products, items, etc.):
1. Use the current iteration number to determine which element to target
   - Iteration 0 → Extract from the 1st element
   - Iteration 1 → Extract from the 2nd element
   - Iteration 2 → Extract from the 3rd element
   - And so on...

2. Check the "Extracted items" list in the iteration history
   - If you see duplicate identifiers (same job title, company name, product name, etc.)
   - You are targeting the WRONG element - increment to the next one

3. Be EXPLICIT about element position in your generated task
   - Good: "Click the 3rd job listing card and extract the company name"
   - Bad: "Click the first job listing and extract the company name" (when current_turn > 0)
   - Good: "Extract product price from the 5th product card visible"
   - Bad: "Extract product price from the first product card" (when current_turn > 0)

4. If the extracted data matches a previous iteration, you have NOT advanced
   - The Sequence Planner must ensure each iteration targets a DIFFERENT element
   - Use positional references: "the Nth element" where N = current_turn + 1

OUTPUT FORMAT:

Your response must be valid JSON with this structure:
{
  "decision": "generate_task" | "end_sequence",
  "reasoning": "Explain your decision",
  "next_task": "Specific task instruction (if generate_task)",
  "completion_reason": "Why ending (if end_sequence)"
}
"""

    def _build_sequence_planner_prompt(
        self,
        sequential_task: Sequence,
        environment_state: EnvironmentState,
        overlay_data: Optional[List[Dict[str, Any]]],
        notebook: Optional[Union[Notebook, List[Dict[str, Any]]]],
    ) -> str:
        """
        Builds the user prompt for Sequence Planner decision.

        Includes:
        - Sequential task and completion condition
        - Current iteration state (index, attempts)
        - Completed iteration history (with results)
        - Current page context
        - Agent's notebook (extracted data; Notebook or list of dicts)
        - Two decision options

        Args:
            sequential_task: The sequential task being executed
            environment_state: Current browser/agent state
            overlay_data: Current page overlays
            notebook: Agent's extracted data (Notebook or list of dicts).

        Returns:
            User prompt string
        """
        state = sequential_task.state

        # Format iteration history
        history_section = self._format_iteration_history(sequential_task)

        # Format overlay summary
        overlay_section = self._format_overlay_summary(overlay_data)

        # Format notebook
        notebook_section = self._format_notebook(notebook)

        # Calculate which element should be targeted for this iteration
        target_element_number = state.current_turn + 1

        # Build prompt
        prompt = f"""SEQUENTIAL TASK:
Task: {sequential_task.goal}
Completion Condition: {sequential_task.completion_condition}
Target Count: {sequential_task.target_count if sequential_task.target_count is not None else "Unknown (indefinite)"}

CURRENT STATE:
- Current Iteration: {state.current_turn}
- **TARGET ELEMENT FOR THIS ITERATION: The {target_element_number}{self._ordinal_suffix(target_element_number)} element**
- Attempts for this iteration: {state.turn_attempts} / {self.config.max_attempts_per_iteration}
- Total Successes: {state.total_success_count}
- Total Failures: {state.total_failure_count}
- Total Iterations Completed: {len(state.completed_turns)}

{history_section}

CURRENT PAGE STATE:
- URL: {environment_state.current_url}
- Page Title: {environment_state.page_title}
{overlay_section}
- [See screenshot for visual context]

{notebook_section}

DECISION REQUIRED:

Based on the sequential task, completion condition, iteration history, and current page state:

1. Should you generate a new Normal Task for the current iteration?
   - If yes, specify the exact task (grounded in visible elements)
   - IMPORTANT: For iteration {state.current_turn}, you should target the **{target_element_number}{self._ordinal_suffix(target_element_number)}** element
   - Check the extracted items list above - avoid extracting duplicate data
   - Consider: Are there more items to process? Can you see the next element?

2. Should you end the sequential task?
   - If yes, explain why the sequence is complete
   - Consider: Is the task satisfied? Are there no more items? Are we stuck?

Remember:
- Look at the screenshot to see what's actually visible
- Reference specific elements in your generated tasks (e.g., "the {target_element_number}{self._ordinal_suffix(target_element_number)} job listing card")
- If previous attempts failed, try a different approach
- Navigation tasks (scroll, click next page) are valid
- Don't repeat the exact same task that just failed
- **CRITICAL: For iteration {state.current_turn}, target the {target_element_number}{self._ordinal_suffix(target_element_number)} element, NOT the 1st element**

Now make your decision."""

        return prompt

    def _ordinal_suffix(self, n: int) -> str:
        """
        Returns the ordinal suffix for a number (st, nd, rd, th).

        Args:
            n: The number

        Returns:
            Ordinal suffix string
        """
        if 10 <= n % 100 <= 20:
            return "th"
        else:
            return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

    def _format_iteration_history(self, sequential_task: Sequence) -> str:
        """
        Formats iteration history for prompt using summary-based approach.

        Shows:
        1. Summary of ALL iterations (successes, failures, extracted items)
        2. Detailed view of recent iterations (last 5, not 10)

        This prevents context window issues where Sequence Planner "forgets"
        early iterations when processing large sequences (e.g., 50+ items).

        Args:
            sequential_task: The sequential task being executed

        Returns:
            Formatted history string
        """
        state = sequential_task.state

        if not state.completed_turns:
            return "ITERATION HISTORY:\n(None yet - this is the first iteration)"

        if not self.config.include_iteration_history:
            return "ITERATION HISTORY:\n(History not included - configured to exclude)"

        # Build summary of ALL iterations
        summary_lines = ["ITERATION HISTORY SUMMARY:"]
        summary_lines.append(f"- Total iterations completed: {len(state.completed_turns)}")
        summary_lines.append(f"- Successful iterations: {state.total_success_count}")
        summary_lines.append(f"- Failed iterations: {state.total_failure_count}")

        # Extract all successful results to prevent duplicates
        all_extracted_items = []
        seen_items = set()
        duplicate_count = 0

        for iter_result in state.completed_turns:
            if iter_result.status == "success" and iter_result.result:
                # Try to extract meaningful identifiers from result
                result = iter_result.result
                if isinstance(result, dict):
                    # Extract key fields like titles, names, IDs
                    identifier = None
                    for key in ["job_title", "title", "name", "id", "url"]:
                        if key in result:
                            identifier = result[key]
                            break
                    if identifier:
                        item_str = str(identifier)[:50]  # Limit length
                        all_extracted_items.append(item_str)
                        # Check for duplicates
                        if item_str in seen_items:
                            duplicate_count += 1
                        else:
                            seen_items.add(item_str)
                elif isinstance(result, str):
                    item_str = result[:50]
                    all_extracted_items.append(item_str)
                    if item_str in seen_items:
                        duplicate_count += 1
                    else:
                        seen_items.add(item_str)

        if all_extracted_items:
            summary_lines.append(f"\nExtracted items (to avoid duplicates):")
            # Show unique items first
            unique_items = list(seen_items)[:20]
            for idx, item in enumerate(unique_items, 1):
                summary_lines.append(f"  {idx}. {item}")

            if len(unique_items) < len(all_extracted_items):
                summary_lines.append(f"  ... and {len(all_extracted_items) - len(unique_items)} more items")

            # Warn about duplicates
            if duplicate_count > 0:
                summary_lines.append(f"\n⚠️ WARNING: {duplicate_count} duplicate extraction(s) detected!")
                summary_lines.append(f"   You have extracted {len(all_extracted_items)} items total, but only {len(seen_items)} are unique.")
                summary_lines.append(f"   Make sure to target a DIFFERENT element for each iteration!")

        # Build detailed view of RECENT iterations (last 5, not 10)
        max_recent = 5  # Reduced from 10 to save context
        recent_iterations = state.completed_turns[-max_recent:]

        recent_lines = ["\n\nRECENT ITERATIONS (detailed):"]
        for iter_result in recent_iterations:
            status_emoji = "✓" if iter_result.status == "success" else "✗"
            recent_lines.append(
                f"\nTurn {iter_result.turn}: {status_emoji} {iter_result.status.upper()}"
            )
            recent_lines.append(f"  Attempts: {iter_result.attempts}")

            if iter_result.result:
                result_str = str(iter_result.result)
                if len(result_str) > 100:
                    result_str = result_str[:100] + "..."
                recent_lines.append(f"  Result: {result_str}")

            if iter_result.error:
                error_str = str(iter_result.error)
                if len(error_str) > 100:
                    error_str = error_str[:100] + "..."
                recent_lines.append(f"  Error: {error_str}")

        if len(state.completed_turns) > max_recent:
            omitted = len(state.completed_turns) - max_recent
            recent_lines.insert(
                1, f"(Showing {max_recent} most recent, {omitted} older iterations in summary above)"
            )

        return "\n".join(summary_lines + recent_lines)

    def _format_overlay_summary(
        self, overlay_data: Optional[List[Dict[str, Any]]]
    ) -> str:
        """
        Formats overlay data summary for prompt.

        Args:
            overlay_data: Current page overlays

        Returns:
            Formatted overlay summary
        """
        if not overlay_data:
            return "- Overlays: (No overlay data available)"

        # Count by type
        type_counts: Dict[str, int] = {}
        for overlay in overlay_data:
            element_type = overlay.get("type", "unknown")
            type_counts[element_type] = type_counts.get(element_type, 0) + 1

        # Format summary
        summary_parts = [
            f"{count} {elem_type}(s)"
            for elem_type, count in sorted(type_counts.items())
        ]

        return f"- Overlays: {len(overlay_data)} interactive elements ({', '.join(summary_parts[:5])})"

    def _format_notebook(self, notebook: Optional[Union[Notebook, List[Dict[str, Any]]]]) -> str:
        """
        Formats agent's notebook for prompt.

        Args:
            notebook: Agent's extracted data (Notebook or list of dicts).

        Returns:
            Formatted notebook string
        """
        entries = notebook.to_list() if isinstance(notebook, Notebook) else notebook
        if not entries:
            return "AGENT'S NOTEBOOK (previously extracted data):\n(Empty - no data extracted yet)"

        notebook_lines = ["AGENT'S NOTEBOOK (previously extracted data):"]

        # Show recent entries (last 5)
        recent_entries = entries[-5:]
        for i, entry in enumerate(recent_entries, 1):
            entry_str = json.dumps(entry, indent=2)
            if len(entry_str) > 200:
                entry_str = entry_str[:200] + "..."
            notebook_lines.append(f"\nEntry {i}:")
            notebook_lines.append(entry_str)

        if len(entries) > 5:
            omitted = len(entries) - 5
            notebook_lines.insert(
                1, f"(Showing 5 most recent entries, {omitted} older entries omitted)"
            )

        return "\n".join(notebook_lines)
