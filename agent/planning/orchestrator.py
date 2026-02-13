"""
Task Orchestrator - Decomposes user requests into structured task lists.

This module provides the TaskOrchestrator class that analyzes user requests
and breaks them down into ordered sequences of Normal and Sequential tasks.
"""

import time
from models import TaskStatus
from utils import page_utils
from utils.debug_print import dprint, PrintMode
from utils.event_logger import get_event_logger
from typing import List, Optional, Dict, Any, Callable, Tuple
import re
import uuid

from models.models import (
    MissionPlan,
    Task,
    TaskDefinition,
    MissionPlannerOutput,
)
from lib.ai import (
    generate_model,
    ReasoningLevel,
    get_default_agent_model,
    get_default_agent_reasoning_level,
)


class TaskOrchestrator:
    """
    Decomposes user requests into structured task lists.
    Identifies which parts require sequential processing.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        reasoning_level: Optional[ReasoningLevel] = None,
    ):
        """
        Initialize the TaskOrchestrator.

        Args:
            model_name: Model to use for decomposition (default: agent model)
            reasoning_level: Reasoning level for decomposition (default: agent reasoning level)
        """
        self.model_name = model_name or get_default_agent_model()
        self.reasoning_level = reasoning_level or get_default_agent_reasoning_level()

    def decompose_mission(
        self,
        user_mission: str,
        initial_context: Optional[Dict[str, Any]] = None,
        screenshot: Optional[bytes] = None,
    ) -> MissionPlan:
        context = initial_context or {}
        feedback_history = []

        event_logger = get_event_logger()
        try:
            event_logger.task_decompose_start(user_mission)
        except Exception:
            pass

        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt_text = self._build_user_prompt(
            user_mission,
            context,
            feedback_history
        )

        # Call LLM for structured output
        output: MissionPlannerOutput = None
        try:
            output = generate_model(
                prompt=user_prompt_text,
                model_object_type=MissionPlannerOutput,
                system_prompt=system_prompt,
                model=self.model_name,
                reasoning_level=self.reasoning_level,
                image=screenshot
            )
            
            if output is not None:
                event_logger.task_decompose_complete(output.tasks, len(output.tasks))
            else:
                event_logger.task_decompose_fail(error="Failed to generate task decomposition")
                return MissionPlan(tasks=[])

        except Exception as e:
            # If we failed to get output, return an empty task list
            if output is None:
                event_logger.task_decompose_fail(error=f"Failed to generate task decomposition: {str(e)}")
                return MissionPlan(tasks=[])

        # Convert output to MissionPlan with proper task IDs
        task_list = self._convert_to_task_list(output)

        return task_list

    def _build_system_prompt(self) -> str:
        return """You are a Task Orchestrator for a browser automation agent.

Break the user's request into an ordered list of tasks. Each task has:
- **task**: Clear, actionable instruction the agent can execute
- **target**: 1 (default), N (repeat N times), or "all" (until exhausted)
- **start_hint**: One-line orientation — where to look, what the target element looks like

PRINCIPLES:
1. **Stick to what the user asked.** Don't add tasks they didn't request.
2. **Use common sense about context.** If the user is already on amazon.com and says "search for wireless mouse", don't add a "navigate to Amazon" task.
3. **Right granularity.** "Search for wireless mouse" is one task, not "click search bar" + "type wireless mouse" + "press Enter". But "search for X and add the first result to cart" is two tasks.
4. **Actionable language.** Use verbs the agent supports: click, type, scroll, press, open, back, forward, extract. Don't use "identify", "review", "verify".

TARGET PATTERNS:
- "first 5 posts" → target=5
- "all products" → target="all"
- No quantifier → target=1
"""

    def _build_user_prompt(
        self,
        user_prompt: str,
        context: Dict[str, Any],
        feedback_history: list[Dict[str, Any]],
    ) -> str:
        """
        Builds the user prompt for task decomposition.

        Args:
            user_prompt: User's request
            context: Current context (URL, page info, etc.)
            feedback_history: Previous validation feedback (if any)
        """
        # Build context section
        context_section = ""
        if context:
            context_lines = []
            if "url" in context:
                context_lines.append(f"- URL: {context['url']}")
            if "page_title" in context:
                context_lines.append(f"- Page Title: {context['page_title']}")
            if "page_description" in context:
                context_lines.append(f"- Page Description: {context['page_description']}")

            if context_lines:
                context_section = "\n\nCURRENT CONTEXT:\n" + "\n".join(context_lines)

        # Build feedback section
        feedback_section = ""
        if feedback_history:
            feedback_section = "\n\nPREVIOUS ATTEMPTS AND FEEDBACK:\n"
            for entry in feedback_history:
                feedback_section += f"\nAttempt {entry['attempt']} (Generated {entry['task_count']} tasks):\n"
                feedback_section += f"User Feedback: {entry['feedback']}\n"
            feedback_section += "\nPlease address the feedback above in your new decomposition.\n"

        # Build examples section
        examples_section = """

EXAMPLES:

Example 1 — Navigation + repetitive action:
Request: "Go to LinkedIn and like the first 5 posts"
{
  "tasks": [
    {
      "task": "Navigate to linkedin.com",
      "target": 1,
      "start_hint": "Type linkedin.com in the address bar or use open_url"
    },
    {
      "task": "Like a post in the feed",
      "target": 5,
      "start_hint": "The Like button is a thumbs-up icon below each post. Scroll between posts."
    }
  ]
}

Example 2 — Simple search (already on the site):
Context: User is on amazon.com
Request: "Search for wireless mouse"
{
  "tasks": [
    {
      "task": "Search for 'wireless mouse' using the search bar",
      "target": 1,
      "start_hint": "The search bar is at the top of the page"
    }
  ]
}
Note: No "navigate to Amazon" task — user is already there.

Example 3 — Extract all:
Request: "Extract all product prices from this page"
{
  "tasks": [
    {
      "task": "Extract product prices from listings",
      "target": "all",
      "start_hint": "Prices are typically near each product title, prefixed with $"
    }
  ]
}
"""

        return f"""USER REQUEST:
{user_prompt}{context_section}{feedback_section}{examples_section}

Now analyze the user request and generate the task decomposition in the specified JSON format."""

    def _convert_to_task_list(self, tasks: MissionPlannerOutput) -> MissionPlan:
        """
        Converts MissionPlannerOutput to MissionPlan with proper task IDs.

        Args:
            tasks: The orchestrator output with task definitions

        Returns:
            MissionPlan with tasks that have unique IDs
        """
        task_list: List[Task] = []

        for task_def in tasks.tasks:
            task_id = f"task_{uuid.uuid4().hex[:8]}"

            # Create unified Task with target
            new_task = Task(
                goal=task_def.task,
                target=task_def.target,
                start_hint=task_def.start_hint,
                task_id=task_id,
                status=TaskStatus.PENDING,
                created_at=time.time(),
                completed_at=None,
                progress=0,
                history=[],
            )

            task_list.append(new_task)

        return MissionPlan(tasks=task_list)
