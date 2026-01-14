"""
Task Orchestrator - Decomposes user requests into structured task lists.

This module provides the TaskOrchestrator class that analyzes user requests
and breaks them down into ordered sequences of Normal and Sequential tasks.
"""

from typing import Optional, Dict, Any, Callable, Tuple
import uuid

from models.task_models import (
    TaskList,
    NormalTask,
    SequentialTask,
    TaskType,
    TaskOrchestratorOutput,
)
from ai_utils import (
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

    def decompose_user_request(
        self,
        user_prompt: str,
        initial_context: Optional[Dict[str, Any]] = None,
        validation_callback: Optional[Callable[[TaskList], Tuple[bool, str]]] = None,
        max_validation_attempts: int = 3,
    ) -> TaskList:
        """
        Analyzes user request and generates a TaskList with Normal and Sequential tasks.

        Args:
            user_prompt: The user's original request
            initial_context: Optional context (URL, page state, etc.)
            validation_callback: Optional callback for user validation of task plan.
                                Should return (approved: bool, feedback: str).
                                If approved=False, the orchestrator will regenerate with feedback.
            max_validation_attempts: Maximum regeneration attempts if validation fails

        Returns:
            TaskList with ordered tasks

        Raises:
            ValueError: If unable to generate valid task decomposition after max attempts
        """
        context = initial_context or {}
        feedback_history = []

        for attempt in range(max_validation_attempts):
            # Build prompts
            system_prompt = self._build_system_prompt()
            user_prompt_text = self._build_user_prompt(
                user_prompt,
                context,
                feedback_history
            )

            # Call LLM for structured output
            try:
                output = generate_model(
                    prompt=user_prompt_text,
                    model_object_type=TaskOrchestratorOutput,
                    system_prompt=system_prompt,
                    model=self.model_name,
                    reasoning_level=self.reasoning_level,
                )

                # Handle case where generate_model returns a string (parsing failed)
                if isinstance(output, str):
                    # Try to parse it ourselves
                    import json
                    import re

                    # Strip markdown code blocks if present
                    cleaned = output.strip()
                    if cleaned.startswith("```"):
                        # Remove ```json or ``` at start and ``` at end
                        cleaned = re.sub(r'^```(?:json)?\s*\n', '', cleaned)
                        cleaned = re.sub(r'\n```\s*$', '', cleaned)

                    # Parse JSON
                    data = json.loads(cleaned)
                    output = TaskOrchestratorOutput(**data)

                # Ensure output is the correct type
                if not isinstance(output, TaskOrchestratorOutput):
                    raise ValueError(f"Expected TaskOrchestratorOutput, got {type(output)}")

            except Exception as e:
                if attempt == max_validation_attempts - 1:
                    raise ValueError(f"Failed to generate task decomposition: {e}") from e
                continue

            # Convert output to TaskList with proper task IDs
            task_list = self._convert_to_task_list(output)

            # If no validation callback, return immediately
            if validation_callback is None:
                return task_list

            # Call validation callback
            try:
                approved, feedback = validation_callback(task_list)
                if approved:
                    return task_list

                # Add feedback for next attempt
                feedback_history.append({
                    "attempt": attempt + 1,
                    "feedback": feedback,
                    "task_count": len(task_list.tasks),
                })
            except Exception as e:
                # If validation callback fails, just return the task list
                print(f"⚠️ Validation callback error: {e}")
                return task_list

        # Max attempts reached
        raise ValueError(
            f"Failed to generate approved task decomposition after {max_validation_attempts} attempts"
        )

    def _build_system_prompt(self) -> str:
        """
        Builds the system prompt for task decomposition.

        Instructs LLM to:
        - Identify logical task boundaries
        - Detect iteration/sequential patterns (numbers, "for each", etc.)
        - Create Normal tasks for single actions
        - Create Sequential tasks for loops/iterations
        - Define clear completion conditions for Sequential tasks
        """
        return """You are a Task Orchestrator for a browser automation agent.

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

TASK DECOMPOSITION RULES:

1. Break down complex requests into logical steps
2. Maintain execution order (tasks should flow naturally)
3. Keep tasks focused and atomic
4. Sequential tasks should be reserved for actual iteration/repetition
5. Normal tasks should be specific and immediately actionable
6. Include setup tasks (navigation, search) before data collection
7. Include cleanup/saving tasks after data collection if needed
8. Consider page structure (may need to navigate, filter, or search first)

SEQUENTIAL TASK GUIDELINES:

For Sequential Tasks, you MUST provide:
- goal: Clear statement of what to accomplish (e.g., "Extract company names")
- completion_condition: Natural language description of when complete
  * For definite sequences: "when 5 companies extracted"
  * For indefinite sequences: "when no more listings visible" or "when reached end of page"
  * Can combine conditions: "when 10 items collected OR no more items available"
- target_count: Integer if known upfront (e.g., 5), null if unknown

IMPORTANT:
- Keep tasks focused and atomic
- Sequential tasks handle the iteration logic - don't create separate Normal tasks for each iteration
- Normal tasks should be actionable without additional decomposition
- Ensure tasks are in logical execution order
- Each task should move toward completing the user's request
- Consider dependencies (e.g., must navigate before extracting data)

OUTPUT FORMAT:

Your response will be parsed as JSON. Return a structured object with:
- tasks: Array of task objects (either Normal or Sequential)
- reasoning: Explanation of how the request was decomposed
- confidence: Your confidence in this decomposition (0.0-1.0)

Each Normal Task should have:
- type: "normal"
- description: Natural language description
- instruction: Specific instruction for the agent
- depends_on: null (or task_id if it depends on previous task results)

Each Sequential Task should have:
- type: "sequential"
- description: Natural language description
- goal: What to accomplish
- completion_condition: When it's complete
- target_count: Integer or null

IMPORTANT: Do NOT include these internal fields in your response:
- task_id (generated automatically)
- status (managed by system)
- state (managed by system)
- results (managed by system)
- current_subtask (managed by system)
- started_at, completed_at (managed by system)
- error (managed by system)

Only include the fields listed above for each task type. The system will handle all internal state management.
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
      "target_count": null
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
"""

        return f"""USER REQUEST:
{user_prompt}{context_section}{feedback_section}{examples_section}

Now analyze the user request and generate the task decomposition in the specified JSON format."""

    def _infer_extraction_schema(self, goal: str) -> Optional[Dict[str, Any]]:
        """
        Infer extraction schema from natural language goal using LLM.

        Parses goals like:
        - "Extract job title and company name"
        - "Get price, description, and availability"
        - "Collect email addresses and phone numbers"

        Args:
            goal: The sequential task goal

        Returns:
            JSON schema dict or None if not an extraction task
        """
        import json

        goal_lower = goal.lower()

        # Check if this is an extraction task
        extraction_verbs = ["extract", "get", "collect", "gather", "scrape", "retrieve"]
        if not any(verb in goal_lower for verb in extraction_verbs):
            return None

        # Use LLM to extract field names
        prompt = f"""Given this task goal, identify the data fields that should be extracted.

Task Goal: "{goal}"

Respond with a JSON list of field names in snake_case format.

Examples:
Goal: "Extract job title and company name from listings"
Response: ["job_title", "company_name"]

Goal: "Get the price, product description, and availability status"
Response: ["price", "product_description", "availability_status"]

Goal: "Collect email addresses and phone numbers"
Response: ["email_address", "phone_number"]

Rules:
- Use snake_case (lowercase with underscores)
- Remove articles (the, a, an)
- Be specific (e.g., "company_name" not just "company")
- Only include fields explicitly mentioned in the goal

Now analyze the task goal above and respond with ONLY the JSON list, nothing else."""

        try:
            # Use haiku for fast, cheap inference
            response = generate_model(
                system_prompt="You extract structured field names from task descriptions. Respond only with valid JSON.",
                user_prompt=prompt,
                reasoning_level=ReasoningLevel.NONE,
            )

            # Parse the response
            response_text = response.strip()

            # Handle markdown code blocks
            if response_text.startswith("```"):
                # Extract JSON from code block
                lines = response_text.split("\n")
                json_lines = [l for l in lines if l and not l.startswith("```")]
                response_text = "\n".join(json_lines)

            fields = json.loads(response_text)

            if not isinstance(fields, list) or not fields:
                return None

            # Validate all fields are strings
            normalized_fields = [str(f).strip() for f in fields if f]

            if not normalized_fields:
                return None

            # Generate simple JSON schema
            return {
                "type": "object",
                "required": normalized_fields,
                "properties": {
                    field: {"type": "string", "description": f"The {field.replace('_', ' ')}"}
                    for field in normalized_fields
                }
            }

        except Exception as e:
            # If LLM fails, return None (no schema)
            print(f"[Warning] Failed to infer extraction schema: {e}")
            return None

    def _convert_to_task_list(self, output: TaskOrchestratorOutput) -> TaskList:
        """
        Converts TaskOrchestratorOutput to TaskList with proper task IDs.

        Args:
            output: The orchestrator output with task definitions

        Returns:
            TaskList with tasks that have unique IDs
        """
        from models.task_models import SequentialState

        tasks_with_ids = []

        for task in output.tasks:
            # Generate unique task ID if not set or empty
            if not task.task_id:
                task.task_id = f"task_{uuid.uuid4().hex[:8]}"

            if task.type == TaskType.NORMAL:
                # It's already a NormalTask
                tasks_with_ids.append(task)
            elif task.type == TaskType.SEQUENTIAL:
                # It's already a SequentialTask

                # IMPORTANT: Ensure state is initialized (LLM might return state=None)
                if task.state is None:
                    task.state = SequentialState()

                # Ensure results list is initialized
                if task.results is None:
                    task.results = []

                # Infer extraction schema if this is an extraction task
                if task.extraction_schema is None:
                    schema = self._infer_extraction_schema(task.goal)
                    if schema:
                        task.extraction_schema = schema
                        try:
                            fields = list(schema.get("properties", {}).keys())
                            print(f"🔍 Inferred extraction schema for sequential task: {fields}")
                        except Exception:
                            pass

                tasks_with_ids.append(task)

        return TaskList(tasks=tasks_with_ids)
