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
        return """
            You are a Task Orchestrator providing instructions to a browser automation agent.

            Your role is to analyze a user's request and break it down into an ordered list of tasks.

            Each task has two components:
            1. **task** (string): What to do - a clear, actionable instruction
            2. **target** (int or "all"): How many times to do it
               - target=1: Single action (default)
               - target=N: Repeat N times (e.g., target=5 means do it 5 times)
               - target="all": Keep doing it until exhausted/no more items

            EXAMPLES:

            Single action (target=1):
            - "Click the login button" → target=1
            - "Navigate to linkedin.com" → target=1
            - "Type 'hello' in the search box" → target=1

            Repetitive action (target=N):
            - "Type 'hello' 3 times" → target=3
            - "Like the first 5 posts" → target=5
            - "Extract company names from 10 job listings" → target=10

            Indefinite action (target="all"):
            - "Extract all product prices" → target="all"
            - "Click every 'Save' button on the page" → target="all"
            - "Scroll until the end of the page" → target="all"

            DETECTION PATTERNS:
            Look for these patterns to determine target:
            - Numbers: "3 times", "first 5", "top 10" → extract the number
            - Quantifiers: "all", "every", "each" → target="all"
            - No quantifier: default → target=1

            IMPORTANT RULES:
            - Ensure tasks are in logical execution order
            - Each task should be atomic and specific
            - Use plain English that maps to supported commands:
              * click, type, scroll, press, open, back, forward, extract
            - DO NOT use unsupported verbs like "identify", "summarize", "review"
            - Consider dependencies (e.g., navigate before extracting data)

            Example breakdown:
            User: "Go to LinkedIn and like the first 3 posts"
            →
            Task 1: "Navigate to linkedin.com" (target=1)
            Task 2: "Like posts" (target=3)
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

Example 1: Repetitive task with count
Request: "Type 'hello' 3 times"
Response:
{
  "tasks": [
    {
      "task": "Type 'hello'",
      "target": 3
    }
  ],
  "reasoning": "User wants to type 'hello' exactly 3 times. Simple repetitive action.",
  "confidence": 0.95
}

Example 2: Complex multi-step with repetition
Request: "Go to LinkedIn and like the first 5 posts"
Response:
{
  "tasks": [
    {
      "task": "Navigate to linkedin.com",
      "target": 1
    },
    {
      "task": "Like posts",
      "target": 5
    }
  ],
  "reasoning": "Two tasks: (1) navigate once (target=1), (2) like posts 5 times (target=5).",
  "confidence": 0.9
}

Example 3: Indefinite repetition
Request: "Extract all product prices from this page"
Response:
{
  "tasks": [
    {
      "task": "Extract product prices",
      "target": "all"
    }
  ],
  "reasoning": "Extract prices until exhausted. Using target='all' since count is unknown.",
  "confidence": 0.85
}

Example 4: No repetition needed
Request: "Click the login button and enter my credentials"
Response:
{
  "tasks": [
    {
      "task": "Click the login button",
      "target": 1
    },
    {
      "task": "Enter username and password in the login form",
      "target": 1
    }
  ],
  "reasoning": "Two single actions, each done once.",
  "confidence": 0.95
}

Example 5: Use supported commands
Request: "Open the 5th article and give me a summary"
Good Response:
{
  "tasks": [
    {
      "task": "Click the 5th article link",
      "target": 1
    },
    {
      "task": "Extract article content and key summary points",
      "target": 1
    }
  ],
  "reasoning": "Using 'click' and 'extract' - both supported commands."
}
Bad Response (DO NOT DO THIS):
- "Identify the 5th article link" (use 'click' instead)
- "Summarize the article" (use 'extract' instead)
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
                task_id=task_id,
                status=TaskStatus.PENDING,
                created_at=time.time(),
                completed_at=None,
                progress=0,
                history=[],
            )

            task_list.append(new_task)

        return MissionPlan(tasks=task_list)