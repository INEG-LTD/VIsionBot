"""
Task Orchestrator - Decomposes user requests into structured task lists.

This module provides the TaskOrchestrator class that analyzes user requests
and breaks them down into ordered sequences of Normal and Sequential tasks.
"""

from ctypes import Union
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
    SequenceBlueprint,
    SequenceState,
    Task,
    Sequence,
    TaskType,
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

            TASK TYPES:

            1. NORMAL TASK:
                - A single, direct instruction that can be executed immediately
                - Examples: "Click the login button", "Type 'hello' in the search box", "Extract the page title"
                - Should be specific and actionable
                - Use for navigation, single interactions, one-time extractions

            2. SEQUENTIAL TASK:
                - A high-level goal that requires iteration or repeated actions
                - Must include:
                    * A clear goal (what to accomplish)
                    * A completion condition (when it's done)
                    * Optional target_count (if known upfront, e.g., "first 5 items")
                - Examples:
                    * Goal: "Extract company names from job listings"
                    Completion: "when 5 company names have been extracted OR no more listings visible"
                    Target Count: 5
                    * Goal: "Apply to all visible job postings"
                    Completion: "when all 'Apply' buttons on the page have been clicked"
                    Target Count: null (unknown upfront)
                - Use for data collection loops, applying actions to multiple items, repetitive tasks

                DETECTION PATTERNS FOR SEQUENTIAL TASKS:
                    Look for these patterns in user requests:
                    - Numbers indicating quantity: "first 5", "top 10", "all items", "3 companies"
                    - Iteration indicators: "for each", "every", "all", "each of the"
                    - Collection verbs: "gather", "extract all", "collect from", "save all"
                    - Repetition verbs: "apply to multiple", "click all", "save each"
                    - Range patterns: "from 1 to 10", "the first N", "until X"


                For Sequential Tasks, you MUST provide:
                    * For definite sequences: "when 5 companies extracted"
                    * For indefinite sequences: "when no more listings visible" or "when reached end of page"
                    * Can combine conditions: "when 10 items collected OR no more items available"

            IMPORTANT:
                - You are not allowed to use an if statement because if a task is not immediately executable, it is up to the agent to figure out what to do next.
                - Ensure tasks are in logical execution order
                - Because each task is atomic, always be specific about each term in the task and state what the task is to do.
                    Example: There's a task to extract the url from a Videographer job listing.
                    The task should be: "Extract the url from the Videographer job listing" and not "Extract the url from the job listing" as the latter does not specify what task is to be done precisely.
                - Each task should move toward completing the user's request
                - Consider dependencies (e.g., must navigate before extracting data)
                - The insturctions you return should be in plain english but should map semantically to the supported agent commands. These are the only supported verbs:
                    - click (taps, opens, selects)
                    - type (enter text/credentials/search terms)
                    - scroll (reveal more content)
                    - press (keyboard shortcuts/enter)
                    - open (go to a URL)
                    - back (go back in history)
                    - forward (go forward in history)
                    - extract (extract data from page)
                    
                    So an example of a task could be: 
                        - "Navigate to the login page"
                        - "Type in the username and password"
                        - "Click the login button"
                        - "Search for '...' in the search box"
                        - "Press Enter to search"
                        - etc.
                - Do NOT create tasks with unsupported verbs like "identify", "summarize", "review" as standalone instructions.
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

Example 1: Simple sequence
Request: "Find the first 3 iOS developer jobs on this page and save the company names"
Response:
{
  "tasks": [
    {
      "type": "sequential",
      "description": "Extract company names from first 3 iOS developer job listings",
      "goal": "Extract company names from job listings",
      "completion_condition": "when 3 company names extracted successfully",
      "target_count": 3
    },
    {
      "type": "normal",
      "description": "Save the extracted company names to file",
      "instruction": "Save the extracted company names to a file"
    }
  ],
  "reasoning": "The request has two main parts: (1) extract company names from 3 jobs (sequential task with definite count), (2) save the results (normal task). The sequential task will handle iteration through the job listings.",
  "confidence": 0.95
}

Example 2: Complex multi-step
Request: "Go to LinkedIn, search for 'Python developer remote', and apply to the first 5 jobs"
Response:
{
  "tasks": [
    {
      "type": "normal",
      "description": "Navigate to LinkedIn",
      "instruction": "Navigate to linkedin.com"
    },
    {
      "type": "normal",
      "description": "Search for 'Python developer remote'",
      "instruction": "Search for 'Python developer remote' in the job search box"
    },
    {
      "type": "normal",
      "description": "Submit the search",
      "instruction": "Click the search button or press Enter to search"
    },
    {
      "type": "sequential",
      "description": "Apply to first 5 job listings",
      "goal": "Submit applications to job listings",
      "completion_condition": "when 5 job applications submitted OR no more 'Easy Apply' buttons visible",
      "target_count": 5
    }
  ],
  "reasoning": "Request breaks into 4 logical steps: (1) navigate to site, (2) enter search terms, (3) initiate search, (4) iteratively apply to jobs. First 3 are normal single actions. Last is sequential with target count of 5.",
  "confidence": 0.9
}

Example 3: Indefinite sequence
Request: "Extract all product prices from this page"
Response:
{
  "tasks": [
    {
      "type": "sequential",
      "description": "Extract all product prices visible on the page",
      "goal": "Extract product prices",
      "completion_condition": "when no more product price elements are visible on the page",
      "target_count": -1
    }
  ],
  "reasoning": "Single sequential task with indefinite iteration - we don't know how many products exist upfront. The task will continue until no more prices are found.",
  "confidence": 0.85
}

Example 4: No sequential needed
Request: "Click the login button and enter my credentials"
Response:
{
  "tasks": [
    {
      "type": "normal",
      "description": "Click the login button",
      "instruction": "Click the login button"
    },
    {
      "type": "normal",
      "description": "Enter credentials in the login form",
      "instruction": "Enter username and password in the login form"
    }
  ],
  "reasoning": "This is a simple two-step process with no iteration needed. Both steps are single actions.",
  "confidence": 0.95
}

Example 5: Enforce supported commands
Request: "Click and open the 5th article webpage and give me a summary of the article"
Good Response:
{
  "tasks": [
    {
      "type": "normal",
      "description": "Open the 5th article link",
      "instruction": "click: 5th article link"
    },
    {
      "type": "normal",
      "description": "Extract the article content and summary points",
      "instruction": "extract: article content and key summary points"
    }
  ],
  "reasoning": "Only supported commands are used: 'click:' to open the article, and 'extract:' to gather and summarize content in one step. No unsupported verbs like 'identify' or 'summarize' as standalone tasks."
}
Bad Response (DO NOT DO THIS):
- "Identify the 5th article link"
- "Summarize the article"
- Any instruction without a supported leading verb (click, type, scroll, press, navigate/back/forward, wait, defer, extract).
Explanation: 'identify' and 'summarize' are not agent commands; use 'click:' and 'extract:' instead.
"""

        return f"""USER REQUEST:
{user_prompt}{context_section}{feedback_section}{examples_section}

Now analyze the user request and generate the task decomposition in the specified JSON format."""

    def _convert_to_task_list(self, tasks: MissionPlannerOutput) -> MissionPlan:
        """
        Converts MissionPlannerOutput to MissionPlan with proper task IDs.

        Args:
            output: The orchestrator output with task definitions

        Returns:
            MissionPlan with tasks that have unique IDs
        """

        task_list: List[Union[Task, Sequence]] = []

        for task in tasks.tasks:
            task_id = f"task_{uuid.uuid4().hex[:8]}"

            if task.type == TaskType.NORMAL:
                new_task = Task(goal=task.task, task_id=task_id, status=TaskStatus.PENDING, created_at=time.time(), completed_at=None)
            elif task.type == TaskType.SEQUENTIAL:
                new_task = self._generate_sequential_task(task.task, task_id)
                
            task_list.append(new_task)
        return MissionPlan(tasks=task_list)

    def _generate_sequential_task(self, task_prompt: str, task_id: str) -> Sequence:
        system_prompt = """
        Your role is to generate a sequential task based on the user's prompt.
        
        A sequential task is a task that requires multiple turns to complete.
        Examples:
            - Extract the name and company of 2 job listings
            
            Expected result: {{
                "target_count": 2,
                "state": {{
                    "current_turn": 1,
                    "completed_turns": [],
                    "total_success_count": 0,
                    "total_failure_count": 0
                }},
                "current_subtask": None,
                "extraction_schema": {{
                    "name": "string",
                    "company": "string"
                }},
                "task": "Extract the name and company of 2 job listings",
            }}
        """
        
        user_prompt = f"""
        User prompt: {task_prompt}
        """
        
        generated_sequential_task = generate_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model_object_type=SequenceBlueprint,
            model=self.model_name,
            reasoning_level=self.reasoning_level,
        )
        
        if generated_sequential_task is None:
            get_event_logger().sequential_task_generation_error(error="Failed to generate sequential task")
            dummy_sequence = Sequence(
                target_count=-1,
                extraction_schema={},
                task=task_prompt,
                task_id=task_id,
                created_at=time.time(),
                completed_at=None,
                state=SequenceState(current_turn=1, completed_turns=[], total_success_count=0, total_failure_count=0),
            )
            return dummy_sequence
        
        sequence = Sequence(
            target_count=generated_sequential_task.target_count,
            extraction_schema={},
            task=generated_sequential_task.task,
            task_id=task_id,
            created_at=time.time(),
            completed_at=None,
            state=SequenceState(current_turn=1, completed_turns=[], total_success_count=0, total_failure_count=0),
        )
        
        return sequence