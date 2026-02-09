"""
Sequence Planner - Manages sequential task iteration and progression.

This module provides the SequencePlanner class that operates within sequential tasks
to determine next steps or end the sequence based on current state and history.
"""

from typing import Optional, List, Dict, Any, Union
import json

from pydantic import BaseModel, Field

from models import PageElements
from models.models import (
    Sequence,
    SequenceDecision,
    SequenceDecisionType,
)
from core.config import SequentialTaskConfig, CompletionStrategy
from agent.agent_context import EnvironmentState
from agent.notebook import Notebook
from lib.ai import (
    generate_model,
    ReasoningLevel,
    generate_text,
    get_default_agent_model,
    get_default_agent_reasoning_level,
)

class SequenceFormat(BaseModel):
    """
    The format of a sequential task.
    """
    task_count_prefix: str = Field(description="The prefix of the task count.")
    task_count_suffix: str = Field(description="The suffix of the task count.")

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

    def decide_sequence_task_format(self, sequential_task: Sequence) -> SequenceFormat:
        """
        Decides the format of the sequence task.
        """
        user_prompt = f"""
        Sequential task: {sequential_task.task}
        """
        
        system_prompt = """
        Your task is to decide the format of a sequential task.
        A sequential task is a task that is to be completed over a sequence of steps.
        Examples:
        - Extract the name and company of 2 job listings
        - Type 'abc' 3 times
        - Extract the url from the first 5 job listings,
        etc.
        
        The format should be structured as follows:
        Example:
        - "Extract the name and company of 2 job listings"
        - Your response should be: "Extract the name and company of the job listing"
        
        - "Type 'abc' 3 times"
        - Your response should be: "Type 'abc'"
        
        - "Extract the url from the first 5 job listings"
        - Your response should be: "Extract the url from the job listing"
        
        The task count that would be between the prefix and suffix will be an ordinal number eg 1st, 2nd, 3rd, 4th, 5th, etc.
        """
        
        generated_sequence_format = generate_text(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model=self.model_name,
            reasoning_level=self.reasoning_level,
        )
        
        return generated_sequence_format

    def should_end_sequence(self, sequential_task: Sequence) -> tuple[bool, Optional[str]]:
        """Return (should_end, reason). Reason is None when should_end is False."""
        s = sequential_task.state
        c = self.config

        if s.current_turn > c.max_total_iterations:
            return True, f"Maximum iteration limit reached ({c.max_total_iterations})"
        if c.fail_fast and s.total_failure_count:
            return True, f"Fail-fast enabled: sequence ended after {s.total_failure_count} failure(s)"

        target = sequential_task.target_count
        if target == -1:
            return False, None

        succ = s.total_success_count
        done = succ + s.total_failure_count
        attempted_all = done >= target

        if c.completion_strategy == CompletionStrategy.STRICT:
            return (True, f"Target reached: {succ}/{target} successful iterations") if succ >= target else (False, None)

        if c.completion_strategy == CompletionStrategy.BEST_EFFORT:
            return (
                True,
                f"All iterations attempted: {succ} successes, {s.total_failure_count} failures out of {target} target",
            ) if attempted_all else (False, None)

        # THRESHOLD
        if succ >= target:
            return True, f"Target reached: {succ}/{target} successful iterations"
        if not attempted_all:
            return False, None
        rate = (succ / done) if done else 0.0
        return (
            True,
            f"Threshold met: {rate:.1%} success rate ({succ}/{done}) meets {c.success_threshold:.1%} threshold",
        ) if rate >= c.success_threshold else (False, None)

    def _build_system_prompt(self, notebook: Notebook) -> str:
        """
        Builds the system prompt for Sequence Planner.

        Returns:
            System prompt string
        """
        return f"""You are a Sequence Planner for a browser automation agent executing a sequential task.

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


{self._format_notebook(notebook)}

DECISION CRITERIA:

When to GENERATE NEXT TASK:
- Completion condition not yet met
- More data/elements are available to process
- Task still requires more work

When to END SEQUENCE:
- Completion condition is satisfied (e.g., collected enough items)
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
- Your job is to specify WHAT to extract/accomplish, not HOW to navigate
- NEVER generate pure navigation tasks (scroll, click next page)

CRITICAL: AVOIDING DUPLICATE EXTRACTIONS
When extracting from multiple similar elements (job listings, products, items, etc.):
1. Use the current iteration number to determine which element to target
   - Iteration 1 → Extract from the 1st element
   - Iteration 2 → Extract from the 2nd element
   - Iteration 3 → Extract from the 3rd element
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
   - Use positional references: "the Nth element" where N = current_turn

5. If the target count is -1 it means you should continue generating tasks until what was requested is complete.
    Example: "Extract all job listings on the page" (assuming there are 8 job listings on the page) means you should continue generating tasks until all job listings are extracted.
    - Good: You should continue to generate the next task until you've seen that there are no more job listings to extract.
    "Extract the 1st job listing"
    "Extract the 2nd job listing"
    "Extract the 3rd job listing"
    ...
    "Extract the 8th job listing"
    
    - Bad: You should always specify what the next task is to do even if the target count is -1.
    "Extract the job listing"
    "Extract the 9th job listing"
    
OUTPUT FORMAT:

Your response must be valid JSON with this structure:
{{
  "decision": "generate_task" | "end_sequence",
  "reasoning": "Explain your decision",
  "next_task": "Specific task instruction (if decision is generate_task)",
  "completion_reason": "Why ending (if decision is end_sequence)"
}}
"""

    def _build_sequence_planner_prompt(
        self,
        sequential_task: Sequence,
        environment_state: EnvironmentState,
        notebook: Notebook,
    ) -> str:
        """
        Builds the user prompt for Sequence Planner decision.

        Includes:
        - Sequential task and completion condition
        - Current iteration state (index, attempts)
        - Completed iteration history (with results)
        - Current page context
        - Agent's notebook (extracted data)
        - Two decision options

        Args:
            sequential_task: The sequential task being executed
            environment_state: Current browser/agent state
            notebook: Agent's extracted data (Notebook).

        Returns:
            User prompt string
        """
        state = sequential_task.state

        # Format iteration history
        history_section = self._format_iteration_history(sequential_task)

        # Format notebook
        notebook_section = self._format_notebook(notebook)

        # Calculate which element should be targeted for this iteration
        target_element_number = state.current_turn

        # Build prompt
        prompt = f"""SEQUENTIAL TASK:
                    Task: {sequential_task.task}
                    Target Count: {sequential_task.target_count if sequential_task.target_count != -1 else "-1 (indefinite)"}
                    
                    CURRENT STATE:
                    - Current Iteration: {state.current_turn}
                    - **TARGET ELEMENT FOR THIS ITERATION: The {target_element_number}{self._ordinal_suffix(target_element_number)} element**
                    - Total Successes: {state.total_success_count}
                    - Total Failures: {state.total_failure_count}
                    - Total Iterations Completed: {len(state.completed_turns)}

                    {history_section}

                    CURRENT PAGE STATE:
                    - URL: {environment_state.current_url}
                    - Page Title: {environment_state.page_title}
                    
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
                    - Don't repeat the exact same task that just failed
                    - **CRITICAL: For iteration {state.current_turn}, target the {target_element_number}{self._ordinal_suffix(target_element_number)} element**

                    Now make your decision.
"""

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
        if not self.config.include_iteration_history:
            return ""

        turns = sequential_task.state.completed_turns
        if not turns:
            return ""

        return "\n".join(
            f"Turn {t.turn}: {(t.task_attempted or '(no task)').strip()}: "
            f"{'SUCCEEDED' if t.status == 'success' else 'FAILED'}"
            for t in turns
        )

    def _format_notebook(self, notebook: Notebook) -> str:
        """Format notebook entries for inclusion in the prompt."""
        
        notebook_str = ""
        for i, entry in enumerate(notebook.to_list(), 1):
            # Only include notebook extraction entries
            notebook_str += f"{i}.This is the task and what I was asked to extract: {entry.task} -> This is what I extracted: {entry.data}\n"

        prompt = f"""
                Notebook:
                    As an agent, you have access to a notebook where you can store anything you want.
                    The notebook is also used to remember things that you might need to complete the task.
                    The notebook is completely controlled by you and you are to use it to store data that the user has asked you to extract or that you would like to remember.
                    Storing data in the notebook is completely optional, except when requested by the user and you are to use it at your discretion.
                    The notebook is an array list of entries. Each entry is a dictionary item that corresponds to an extraction that was performed by you.
                    
                    Here is the notebook:
                    The notebook is a list of entries. Each entry is a dictionary item that corresponds to an extraction that was performed by you for a task.
                    Each entry is formatted as follows:
                        <index of entry>. This is the task and what I was asked to extract: <the task and what I was asked to extract> -> This is what I extracted: <the data that was extracted>
                    {notebook_str}
                    """
        
        return prompt