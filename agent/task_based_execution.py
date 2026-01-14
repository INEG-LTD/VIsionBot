"""
Task-Based Execution Extension for AgentController.

This module provides task-based execution capabilities that integrate with
the existing AgentController reactive loop.
"""

from typing import Optional, Dict, Any, List, Tuple, Callable, Union
import time

from models.task_models import (
    TaskList,
    NormalTask,
    SequentialTask,
    TaskStatus,
    TaskType,
    IterationResult,
)
from agent.task_orchestrator import TaskOrchestrator
from agent.bridge_planner import BridgePlanner
from agent.completion_contract import EnvironmentState
from bot_config import SequentialTaskConfig
from action_result import ActionResult
from agent.task_result_retrieval import TaskResultRetriever, TaskResultAccessor


class TaskBasedExecutionMixin:
    """
    Mixin class that adds task-based execution capabilities to AgentController.

    This mixin provides:
    - Task orchestration (decomposing user requests into tasks)
    - Normal task execution (single actions)
    - Sequential task execution (iteration loops with Bridge Planner)
    - Task state management and result tracking
    """

    def _initialize_task_system(
        self,
        sequential_task_config: Optional[SequentialTaskConfig] = None,
    ) -> None:
        """
        Initialize task-based execution system.

        Args:
            sequential_task_config: Configuration for sequential task execution
        """
        self.sequential_task_config = sequential_task_config or SequentialTaskConfig()

        # Task orchestrator for decomposition
        self.task_orchestrator = TaskOrchestrator(
            model_name=self.agent_model_name,
            reasoning_level=self.agent_reasoning_level,
        )

        # Bridge planner for sequential task iteration
        self.bridge_planner = BridgePlanner(
            model_name=self.agent_model_name,
            reasoning_level=self.agent_reasoning_level,
            config=self.sequential_task_config,
        )

        # Result retriever for natural language access to task results
        self.task_result_retriever = TaskResultRetriever(
            model_name=self.agent_model_name,
            use_llm_matching=True,
        )

        # Current task list (set during execution)
        self.task_list: Optional[TaskList] = None

        # Result accessor (created when task_list is set)
        self.task_result_accessor: Optional[TaskResultAccessor] = None

        # Flag to enable task-based execution
        self.use_task_based_execution: bool = False

    def _run_task_based_execution(
        self,
        user_prompt: str,
        agent_context: Optional[Any] = None,
    ):
        """
        Execute a task using the task-based execution system.

        This is the main entry point for task-based execution. It:
        1. Decomposes the user request into tasks
        2. Executes each task sequentially
        3. Returns a TaskResult

        Args:
            user_prompt: User's high-level request
            agent_context: Optional agent context

        Returns:
            TaskResult indicating success or failure
        """
        from .task_result import TaskResult

        try:
            self.event_logger.system_info("Using task-based execution")
        except Exception:
            pass

        # Decompose user request into tasks
        try:
            task_list = self._decompose_user_request_into_tasks(
                user_prompt=user_prompt,
                initial_context=None,
                validation_callback=None,
            )
        except Exception as e:
            try:
                self.event_logger.system_error(f"Task decomposition failed: {e}")
            except Exception:
                pass
            return TaskResult(
                success=False,
                confidence=1.0,
                reasoning=f"Task decomposition failed: {str(e)}",
                evidence=self._build_evidence() if hasattr(self, '_build_evidence') else {}
            )

        # Execute task list
        try:
            success = self._execute_task_list(task_list, user_prompt)
        except Exception as e:
            try:
                self.event_logger.system_error(f"Task execution failed: {e}")
            except Exception:
                pass
            return TaskResult(
                success=False,
                confidence=1.0,
                reasoning=f"Task execution failed: {str(e)}",
                evidence=self._build_evidence() if hasattr(self, '_build_evidence') else {}
            )

        # Build result
        if success:
            # Gather all results from completed tasks
            completed_tasks = task_list.get_completed_tasks()

            # Calculate confidence based on task completion
            total_tasks = len(task_list.tasks)
            completed_count = len(completed_tasks)
            confidence = completed_count / total_tasks if total_tasks > 0 else 1.0

            reasoning = f"Successfully completed {completed_count}/{total_tasks} tasks"

            try:
                self.event_logger.system_info(reasoning)
            except Exception:
                pass

            return TaskResult(
                success=True,
                confidence=confidence,
                reasoning=reasoning,
                evidence=self._build_evidence() if hasattr(self, '_build_evidence') else {}
            )
        else:
            # Task execution failed
            completed_tasks = task_list.get_completed_tasks()
            total_tasks = len(task_list.tasks)
            completed_count = len(completed_tasks)

            reasoning = f"Task execution failed after completing {completed_count}/{total_tasks} tasks"

            try:
                self.event_logger.system_error(reasoning)
            except Exception:
                pass

            return TaskResult(
                success=False,
                confidence=0.0,
                reasoning=reasoning,
                evidence=self._build_evidence() if hasattr(self, '_build_evidence') else {}
            )

    def _decompose_user_request_into_tasks(
        self,
        user_prompt: str,
        initial_context: Optional[Dict[str, Any]] = None,
        validation_callback: Optional[Callable[[TaskList], Tuple[bool, str]]] = None,
    ) -> TaskList:
        """
        Decompose user request into a TaskList.

        Args:
            user_prompt: User's request
            initial_context: Optional context (URL, page title, etc.)
            validation_callback: Optional callback for user validation

        Returns:
            TaskList with Normal and Sequential tasks
        """
        # Build initial context
        context = initial_context or {}
        try:
            context["url"] = self.bot.page.url
            context["page_title"] = self.bot.page.title()
        except Exception:
            pass

        # Call task orchestrator
        task_list = self.task_orchestrator.decompose_user_request(
            user_prompt=user_prompt,
            initial_context=context,
            validation_callback=validation_callback,
            max_validation_attempts=3,
        )

        # Log task decomposition
        try:
            self.event_logger.system_info(f"Task decomposition: {len(task_list.tasks)} tasks generated")
            for i, task in enumerate(task_list.tasks, 1):
                task_type = "Sequential" if task.type == TaskType.SEQUENTIAL else "Normal"
                self.event_logger.system_debug(f"  Task {i} ({task_type}): {task.description}")
        except Exception:
            pass

        return task_list

    def _execute_task_list(
        self,
        task_list: TaskList,
        user_prompt: str,
    ) -> bool:
        """
        Execute all tasks in the task list.

        Args:
            task_list: List of tasks to execute
            user_prompt: Original user prompt

        Returns:
            True if all tasks completed successfully, False otherwise
        """
        self.task_list = task_list

        # Create result accessor for this task list
        self.task_result_accessor = TaskResultAccessor(
            task_list=self.task_list,
            retriever=self.task_result_retriever,
        )

        # Execute tasks sequentially
        while self.task_list.get_current_task() is not None:
            current_task = self.task_list.get_current_task()

            # Log task start
            try:
                task_num = self.task_list.current_task_index + 1
                total_tasks = len(self.task_list.tasks)
                self.event_logger.system_info(f"Executing task {task_num}/{total_tasks}: {current_task.description}")
            except Exception:
                pass

            # Execute based on task type
            if isinstance(current_task, NormalTask):
                success = self._execute_normal_task(current_task, user_prompt)
            elif isinstance(current_task, SequentialTask):
                success = self._execute_sequential_task(current_task, user_prompt)
            else:
                # Unknown task type
                try:
                    self.event_logger.system_error(f"Unknown task type: {type(current_task)}")
                except Exception:
                    pass
                success = False

            # Update task status
            if success:
                current_task.status = TaskStatus.COMPLETED
                current_task.completed_at = time.time()
            else:
                current_task.status = TaskStatus.FAILED
                current_task.completed_at = time.time()

                # If a task fails, decide whether to continue or stop
                # For now, we'll stop on first failure
                try:
                    self.event_logger.system_error(f"Task failed: {current_task.description}")
                except Exception:
                    pass
                return False

            # Move to next task
            self.task_list.advance_to_next_task()

        # All tasks completed successfully
        return True

    def _prepare_task_context(
        self,
        task: Union[NormalTask, SequentialTask],
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare context for task execution, including access to previous task results.

        Args:
            task: The task being executed

        Returns:
            Context dict with previous results if available
        """
        if not task.depends_on or not self.task_result_accessor:
            return None

        # Get the dependent task
        dependent_task = self.task_list.get_task_by_id(task.depends_on)

        if not dependent_task or dependent_task.status != TaskStatus.COMPLETED:
            return None

        # Extract results from dependent task
        results = self.task_result_retriever.extract_results_from_tasks([dependent_task])

        if not results:
            return None

        return {
            "previous_task": {
                "id": dependent_task.task_id,
                "description": dependent_task.description,
                "results": results,
            }
        }

    def _is_optional_task(self, task_instruction: str) -> bool:
        """
        Detect if a task is optional based on language patterns.

        Args:
            task_instruction: The task instruction to analyze

        Returns:
            True if task contains optional language patterns
        """
        optional_patterns = [
            "if present",
            "if available",
            "if exists",
            "if visible",
            "if found",
            "if any",
            "if there is",
            "if there are",
            "optional",
            # Additional grammatical variations
            "is present",
            "is available",
            "is visible",
            "is found",
            "are present",
            "are available",
            "are visible",
            "are found",
        ]

        instruction_lower = task_instruction.lower()
        detected = any(pattern in instruction_lower for pattern in optional_patterns)

        # Debug logging
        try:
            if detected:
                matching = [p for p in optional_patterns if p in instruction_lower]
                self.event_logger.system_debug(
                    f"[Optional detection] Task is optional (matched: {matching}): {task_instruction[:80]}"
                )
        except Exception:
            pass

        return detected

    def _validate_extraction_schema(
        self,
        extracted_data: Any,
        schema: Dict[str, Any],
        action: str
    ) -> Dict[str, Any]:
        """
        Validate extracted data against the expected schema.

        Args:
            extracted_data: The data extracted by the agent
            schema: The JSON schema to validate against
            action: The extraction action (for logging)

        Returns:
            Dict with keys:
                - valid: bool indicating if validation passed
                - error: str with error message if validation failed
                - expected_fields: list of expected field names
                - actual_fields: list of actual field names
        """
        expected_fields = schema.get("required", [])

        # Handle different data formats
        if isinstance(extracted_data, dict):
            # Single extraction - filter out metadata fields (start with _)
            actual_fields = [k for k in extracted_data.keys() if not k.startswith('_')]
        elif isinstance(extracted_data, list) and len(extracted_data) > 0:
            # List of extractions - check first item
            if isinstance(extracted_data[0], dict):
                actual_fields = [k for k in extracted_data[0].keys() if not k.startswith('_')]
            else:
                actual_fields = []
        else:
            actual_fields = []

        # Check if all required fields are present
        missing_fields = [f for f in expected_fields if f not in actual_fields]
        # Only report extra fields that are NOT metadata (don't start with _)
        extra_fields = [f for f in actual_fields if f not in expected_fields]

        if missing_fields or extra_fields:
            error_parts = []
            if missing_fields:
                error_parts.append(f"Missing fields: {missing_fields}")
            if extra_fields:
                error_parts.append(f"Unexpected fields: {extra_fields}")

            return {
                "valid": False,
                "error": "; ".join(error_parts),
                "expected_fields": expected_fields,
                "actual_fields": actual_fields,
            }

        return {
            "valid": True,
            "error": None,
            "expected_fields": expected_fields,
            "actual_fields": actual_fields,
        }

    def _execute_normal_task(
        self,
        task: NormalTask,
        user_prompt: str,
    ) -> bool:
        """
        Execute a Normal Task using the existing ReactiveGoalDeterminer.

        Args:
            task: The normal task to execute
            user_prompt: Original user prompt (for context)

        Returns:
            True if task completed successfully, False otherwise
        """
        task.status = TaskStatus.IN_PROGRESS

        # Check if task is optional based on language
        is_optional = self._is_optional_task(task.instruction)

        # Prepare task context with previous results
        task_context = self._prepare_task_context(task)

        # Build enhanced task instruction with context
        task_prompt = task.instruction

        if task_context and "previous_task" in task_context:
            prev = task_context["previous_task"]
            task_prompt = f"""{task.instruction}

CONTEXT - Previous Task Results:
Task: {prev['description']}
Results: {prev['results']}

Use the results above to complete your task."""

        # Execute using mini reactive loop for this specific task
        result = self._run_reactive_loop_for_task(
            task_instruction=task_prompt,
            original_prompt=user_prompt,
            max_iterations=20,  # Limit iterations per task
            extraction_schema=getattr(task, 'extraction_schema', None),
        )

        # Store result
        task.result = {
            "success": result.success,
            "reasoning": result.reasoning,
            "data": result.evidence if hasattr(result, "evidence") else None,
        }
        task.error = None if result.success else result.reasoning

        # For optional tasks, treat "not found" or "element no longer present" as success
        if not result.success and is_optional:
            # Check if this is a "no action possible" scenario (element not found/removed)
            reasoning_lower = result.reasoning.lower() if result.reasoning else ""
            is_not_found_scenario = any(
                phrase in reasoning_lower
                for phrase in ["no action could be determined", "element no longer present", "not visible", "not found"]
            )

            if is_not_found_scenario:
                try:
                    self.event_logger.system_info(
                        f"✓ Optional task completed (element not present/removed): {task.instruction}"
                    )
                except Exception:
                    pass
                # Update task result to reflect success
                task.result["success"] = True
                task.error = None
                return True  # Treat as success since it's optional

            # Other failure reason (not related to element visibility)
            try:
                self.event_logger.system_warning(
                    f"Optional task failed for unexpected reason: {result.reasoning}"
                )
            except Exception:
                pass
            return True  # Still treat as success since it's optional, but log as warning

        # Add task results to notebook for subsequent tasks
        if result.success and task.result.get("data"):
            # Add to notebook so next tasks can reference this task's results
            if hasattr(self, "notebook") and self.notebook is not None:
                self.notebook.append({
                    "source": "task",
                    "task_id": task.task_id,
                    "description": task.description,
                    "data": task.result["data"],
                    "type": "normal_task_result"
                })
                try:
                    self.event_logger.system_debug(
                        f"Added task results to notebook: {task.description}"
                    )
                except Exception:
                    pass

        return result.success

    def _execute_sequential_task(
        self,
        task: SequentialTask,
        user_prompt: str,
    ) -> bool:
        """
        Execute a Sequential Task using the Bridge Planner.

        Args:
            task: The sequential task to execute
            user_prompt: Original user prompt (for context)

        Returns:
            True if task completed successfully (based on completion strategy)
        """
        task.status = TaskStatus.IN_PROGRESS

        try:
            self.event_logger.system_info(f"Starting sequential task: {task.goal}")
            if task.target_count:
                self.event_logger.system_info(f"Target count: {task.target_count}")
            self.event_logger.system_info(f"Completion condition: {task.completion_condition}")
        except Exception:
            pass

        # Track if sequence ended due to error
        ended_with_error = False
        error_reason = None

        # Sequential execution loop
        while True:
            # Check if we should end based on hard limits
            if self.bridge_planner.should_end_sequence(task):
                reason = self.bridge_planner._get_end_reason(task)
                try:
                    self.event_logger.system_info(f"Sequential task ending: {reason}")
                except Exception:
                    pass
                break

            # Capture current state
            snapshot = self._capture_snapshot(full_page=False)
            environment_state = EnvironmentState(
                browser_state=snapshot,
                interaction_history=self.bot.session_tracker.interaction_history,
                user_prompt=user_prompt,
                task_start_url=self.task_start_url,
                task_start_time=self.task_start_time,
                current_url=snapshot.url,
                page_title=snapshot.title,
                visible_text=snapshot.visible_text,
                url_history=self.bot.session_tracker.url_history.copy() if self.bot.session_tracker.url_history else [],
                url_pointer=getattr(self.bot.session_tracker, "url_pointer", None)
            )

            # Get overlay data
            try:
                page_info = self.bot.page_utils.get_page_info()
                overlay_data, _, _ = self.bot._collect_overlay_data(user_prompt, page_info)
            except Exception:
                overlay_data = None

            # Get Bridge Planner decision
            decision = self.bridge_planner.decide_next_action(
                sequential_task=task,
                environment_state=environment_state,
                screenshot=snapshot.screenshot,
                overlay_data=overlay_data,
                notebook=self.notebook,
            )

            # Log decision
            try:
                self.event_logger.system_debug(f"Bridge Planner decision: {decision.decision}")
                self.event_logger.system_debug(f"Reasoning: {decision.reasoning}")
            except Exception:
                pass

            # Handle decision
            if decision.decision == "end_sequence":
                try:
                    self.event_logger.system_info(f"Bridge Planner ending sequence: {decision.completion_reason}")
                except Exception:
                    pass

                # Check if this is an error-based ending
                if decision.completion_reason and any(
                    keyword in decision.completion_reason.lower()
                    for keyword in ["error", "failed", "exception", "internal server error"]
                ):
                    ended_with_error = True
                    error_reason = decision.completion_reason
                    try:
                        self.event_logger.system_error(
                            f"Sequential task terminated due to error: {error_reason}"
                        )
                    except Exception:
                        pass

                break

            elif decision.decision == "generate_task":
                # Execute the generated task
                generated_task_instruction = decision.next_task
                task.current_subtask = generated_task_instruction

                try:
                    iteration_num = task.state.current_iteration
                    attempt_num = task.state.iteration_attempts + 1
                    self.event_logger.system_info(
                        f"Iteration {iteration_num}, Attempt {attempt_num}: {generated_task_instruction}"
                    )
                except Exception:
                    pass

                # Execute the generated task (using existing ReactiveGoalDeterminer)
                result = self._execute_generated_subtask(
                    generated_task_instruction,
                    task,
                    environment_state,
                    snapshot.screenshot,
                    overlay_data,
                )

                # Handle result
                if result.success:
                    # Record successful iteration
                    self._record_iteration_success(task, result)

                    # Move to next iteration
                    task.state.current_iteration += 1
                    task.state.iteration_attempts = 0
                else:
                    # Task failed, increment attempt counter
                    task.state.iteration_attempts += 1

                    # Check if we should retry
                    if self.bridge_planner.should_retry_iteration(task, result.error):
                        try:
                            remaining = self.sequential_task_config.max_attempts_per_iteration - task.state.iteration_attempts
                            self.event_logger.system_warning(
                                f"🔄 Iteration {task.state.current_iteration}, Attempt {task.state.iteration_attempts} failed: {result.error}"
                            )
                            self.event_logger.system_info(
                                f"   Retrying iteration {task.state.current_iteration}... ({remaining} attempts remaining)"
                            )
                        except Exception:
                            pass
                    else:
                        # Max retries reached, record failure and move to next iteration
                        self._record_iteration_failure(task, result)

                        try:
                            self.event_logger.system_error(
                                f"Iteration {task.state.current_iteration} failed after {task.state.iteration_attempts} attempts"
                            )
                        except Exception:
                            pass

                        # Move to next iteration
                        task.state.current_iteration += 1
                        task.state.iteration_attempts = 0

            # Safety check: prevent infinite loops
            if task.state.current_iteration >= self.sequential_task_config.max_total_iterations:
                try:
                    self.event_logger.system_warning("Maximum iteration limit reached, ending sequential task")
                except Exception:
                    pass
                break

        # Check if sequence ended with error
        if ended_with_error:
            try:
                self.event_logger.system_error(
                    f"Sequential task FAILED due to error: {error_reason}"
                )
                self.event_logger.system_info(
                    f"Completed {task.state.total_success_count} iterations before error"
                )
            except Exception:
                pass

            # Mark task as failed and return False
            task.status = TaskStatus.FAILED
            return False

        # Log completion summary
        try:
            self.event_logger.system_info(
                f"Sequential task complete: {task.state.total_success_count} successes, "
                f"{task.state.total_failure_count} failures out of "
                f"{task.state.current_iteration} iterations"
            )
        except Exception:
            pass

        # Determine success based on completion strategy
        if self.sequential_task_config.completion_strategy == "strict":
            if task.target_count:
                return task.state.total_success_count >= task.target_count
            else:
                # For indefinite sequences, success if we have any successes
                return task.state.total_success_count > 0

        elif self.sequential_task_config.completion_strategy == "best_effort":
            # Always succeed (we did our best)
            return True

        elif self.sequential_task_config.completion_strategy == "threshold":
            total = task.state.total_success_count + task.state.total_failure_count
            if total == 0:
                return False
            success_rate = task.state.total_success_count / total
            return success_rate >= self.sequential_task_config.success_threshold

        return False

    def _execute_generated_subtask(
        self,
        task_instruction: str,
        sequential_task: SequentialTask,
        environment_state: EnvironmentState,
        screenshot: bytes,
        overlay_data: Optional[List[Dict[str, Any]]],
    ) -> ActionResult:
        """
        Execute a generated subtask within a sequential task.

        This method runs a mini reactive loop for the specific subtask instruction.
        The subtask is complete when the agent issues a complete: command.

        Args:
            task_instruction: The specific instruction to execute
            sequential_task: Parent sequential task
            environment_state: Current environment state
            screenshot: Current screenshot
            overlay_data: Current overlay data

        Returns:
            ActionResult indicating success/failure
        """
        # Track this task attempt
        iteration_idx = sequential_task.state.current_iteration

        # Find or create current iteration tracking
        current_iter = None
        for ir in sequential_task.state.completed_iterations:
            if ir.iteration == iteration_idx and ir.status == "in_progress":
                current_iter = ir
                break

        # Track task attempt (will be added to iteration result when completed)
        if current_iter and task_instruction not in current_iter.tasks_attempted:
            current_iter.tasks_attempted.append(task_instruction)

        # Execute using mini reactive loop
        result = self._run_reactive_loop_for_task(
            task_instruction=task_instruction,
            original_prompt=sequential_task.goal,
            max_iterations=10,  # Limit iterations per subtask
            extraction_schema=getattr(sequential_task, 'extraction_schema', None),
        )

        # Convert TaskResult to ActionResult
        return ActionResult(
            success=result.success,
            message=result.reasoning,
            data=result.evidence if hasattr(result, "evidence") else {},
            error=result.reasoning if not result.success else None,
        )

    def _run_reactive_loop_for_task(
        self,
        task_instruction: str,
        original_prompt: str,
        max_iterations: int = 20,
        extraction_schema: Optional[Dict[str, Any]] = None,
    ) -> "TaskResult":
        """
        Run a mini reactive loop for a specific task instruction.

        This executes actions until the agent issues a complete: command or max iterations reached.

        Args:
            task_instruction: The specific instruction to execute
            original_prompt: Original user prompt (for context)
            max_iterations: Maximum iterations for this task

        Returns:
            TaskResult indicating success/failure
        """
        from .task_result import TaskResult
        from agent.reactive_goal_determiner import ReactiveGoalDeterminer

        # Track task-specific failed/ineffective actions
        task_failed_actions = []
        task_ineffective_actions = []

        # Track repeated failures for stuck detection
        repeated_failure_count = {}  # action -> count

        for iteration in range(max_iterations):
            # Debug: Log iteration start
            try:
                is_opt = self._is_optional_task(task_instruction)
                self.event_logger.system_debug(
                    f"[Mini-loop] Starting iteration {iteration}/{max_iterations-1} for task: '{task_instruction}' (optional={is_opt})"
                )
            except Exception:
                pass

            # Capture current state
            try:
                snapshot = self._capture_snapshot(full_page=False)
            except Exception as e:
                return TaskResult(
                    success=False,
                    confidence=0.0,
                    reasoning=f"Failed to capture snapshot: {str(e)}",
                    evidence={"error": str(e)},
                )

            # Build environment state
            environment_state = EnvironmentState(
                browser_state=snapshot,
                interaction_history=self.bot.session_tracker.interaction_history,
                user_prompt=original_prompt,
                task_start_url=self.task_start_url,
                task_start_time=self.task_start_time,
                current_url=snapshot.url,
                page_title=snapshot.title,
                visible_text=snapshot.visible_text,
                url_history=self.bot.session_tracker.url_history.copy() if self.bot.session_tracker.url_history else [],
                url_pointer=getattr(self.bot.session_tracker, "url_pointer", None)
            )

            # Get overlay data
            try:
                page_info = self.bot.page_utils.get_page_info()
                overlay_data, _, _ = self.bot._collect_overlay_data(task_instruction, page_info)
            except Exception:
                overlay_data = None

            # Create goal determiner for this task
            goal_determiner = ReactiveGoalDeterminer(
                task_instruction,
                base_knowledge=self.base_knowledge if hasattr(self, "base_knowledge") else [],
                model_name=self.agent_model_name,
                reasoning_level=self.agent_reasoning_level,
                image_detail=self.image_detail if hasattr(self, "image_detail") else "low",
                interaction_summary_limit=None,
                include_overlays_in_agent_context=self.include_overlays_in_agent_context if hasattr(self, "include_overlays_in_agent_context") else True,
                include_visible_text_in_agent_context=self.include_visible_text_in_agent_context if hasattr(self, "include_visible_text_in_agent_context") else False,
                history_manager=self.bot.history_manager if hasattr(self.bot, "history_manager") else None,
                max_actions_per_plan=self.max_actions_per_plan if hasattr(self, "max_actions_per_plan") else 6,
                extraction_schema=extraction_schema,
            )

            # Determine next action
            try:
                action_plan = goal_determiner.determine_action_plan(
                    environment_state=environment_state,
                    screenshot=snapshot.screenshot,
                    failed_actions=task_failed_actions,
                    ineffective_actions=task_ineffective_actions,
                    overlay_data=overlay_data,
                    notebook=self.notebook if hasattr(self, "notebook") else [],
                )
                # Debug logging
                try:
                    self.event_logger.system_debug(
                        f"[Mini-loop iter {iteration}] action_plan returned: {action_plan is not None}, "
                        f"has steps: {action_plan.steps if action_plan else 'N/A'}"
                    )
                except Exception:
                    pass
            except Exception as e:
                # Debug logging
                try:
                    self.event_logger.system_debug(
                        f"[Mini-loop iter {iteration}] Exception in determine_action_plan: {str(e)}"
                    )
                except Exception:
                    pass
                # Check if this is an optional task and we've already had success
                is_optional = self._is_optional_task(task_instruction)
                if is_optional and iteration > 0:
                    # Optional task, element likely no longer present after successful action
                    try:
                        self.event_logger.system_info(
                            f"Optional task completed: Element no longer present after {iteration} action(s)"
                        )
                    except Exception:
                        pass

                    return TaskResult(
                        success=True,
                        confidence=1.0,
                        reasoning=f"Optional task completed: Element no longer present after action",
                        evidence={"iterations": iteration + 1, "actions_tried": iteration + 1},
                    )

                return TaskResult(
                    success=False,
                    confidence=0.0,
                    reasoning=f"Failed to determine action: {str(e)}",
                    evidence={"error": str(e)},
                )

            if not action_plan or not action_plan.steps:
                # Agent returned empty steps instead of using "complete:" command
                # This is a prompt adherence issue - the agent should ALWAYS return at least one step
                # Either: (1) "complete: <reason>" when done, or (2) an action to perform

                # Get agent's reasoning about why it returned empty steps
                agent_reasoning = ""
                if action_plan and hasattr(action_plan, 'reasoning'):
                    agent_reasoning = action_plan.reasoning or ""

                # Log the prompt adherence issue
                try:
                    self.event_logger.system_warning(
                        f"⚠️ Agent returned empty steps (should use 'complete:' instead). "
                        f"Reasoning: {agent_reasoning[:150]}"
                    )
                except Exception:
                    pass

                # Check if agent indicates task is naturally complete
                reasoning_lower = agent_reasoning.lower()
                completion_indicators = [
                    "no action is needed",
                    "no action needed",
                    "nothing to do",
                    "not present",
                    "not found",
                    "not visible",
                    "no such",
                    "no longer",
                    "already done",
                    "proceed",
                    "continue",
                    "condition",
                    "conditional"
                ]

                agent_indicates_done = any(indicator in reasoning_lower for indicator in completion_indicators)

                if agent_indicates_done or iteration > 0:
                    # Agent is saying task is done OR we had prior successful actions
                    # Interpret empty steps as implicit "complete:" (fallback for prompt non-compliance)
                    try:
                        self.event_logger.system_info(
                            f"✓ Task completed (implicit): {agent_reasoning or 'No further actions needed'}"
                        )
                    except Exception:
                        pass

                    return TaskResult(
                        success=True,
                        confidence=1.0,
                        reasoning=agent_reasoning or "Task completed, no further actions needed",
                        evidence={"iterations": iteration + 1, "actions_tried": iteration + 1},
                    )

                # First iteration, no clear completion signal - likely stuck/error
                try:
                    self.event_logger.system_error(
                        f"❌ No actions determined on first iteration. Agent should have used 'complete:' or provided action. "
                        f"Reasoning: {agent_reasoning[:100]}"
                    )
                except Exception:
                    pass

                return TaskResult(
                    success=False,
                    confidence=0.0,
                    reasoning=f"No action could be determined for task. Agent reasoning: {agent_reasoning}",
                    evidence={"task_instruction": task_instruction},
                )

            # Execute ALL steps in the action plan sequentially
            plan_failed = False
            last_successful_action = None

            for step_idx, action_step in enumerate(action_plan.steps):
                current_action = action_step.action

                # Check for complete: command
                if current_action and current_action.lower().startswith("complete:"):
                    completion_reasoning = current_action.split(":", 1)[1].strip() if ":" in current_action else "Task completed"

                    # Log mini-loop completion
                    if iteration > 0 or step_idx > 0:
                        try:
                            self.event_logger.system_debug(
                                f"   ✓ Task completed after trying {iteration + 1} action(s) in mini-loop"
                            )
                        except Exception:
                            pass

                    return TaskResult(
                        success=True,
                        confidence=1.0,
                        reasoning=completion_reasoning,
                        evidence={"iterations": iteration + 1, "actions_tried": iteration + 1},
                    )

                # Check for ask: command (agent asking for help when stuck)
                if current_action and current_action.lower().startswith("ask:"):
                    question = current_action.split(":", 1)[1].strip() if ":" in current_action else "Need assistance"
                    try:
                        self.event_logger.system_info(f"Agent asking for help: {question}")
                    except Exception:
                        pass

                    # For now, return failure to indicate human intervention needed
                    # In the future, this could pause and wait for user input
                    return TaskResult(
                        success=False,
                        confidence=0.0,
                        reasoning=f"Agent requested assistance: {question}",
                        evidence={"question": question, "iterations": iteration + 1},
                    )

                # Log action attempt (if not first step in plan and not first iteration)
                if step_idx > 0:
                    try:
                        self.event_logger.system_debug(
                            f"   → Executing step {step_idx + 1}/{len(action_plan.steps)}: {current_action[:60]}"
                        )
                    except Exception:
                        pass
                elif iteration > 0:
                    try:
                        self.event_logger.system_debug(
                            f"   → Mini-loop iteration {iteration + 1}: Trying alternative approach after previous action failed"
                        )
                    except Exception:
                        pass

                # Execute the action
                try:
                    result = self.bot.act(current_action)

                    # Track ineffective/failed actions
                    if not result.success:
                        task_failed_actions.append(current_action)

                        # Log failure
                        try:
                            self.event_logger.system_debug(
                                f"   ✗ Action failed: {result.error or result.message}"
                            )
                        except Exception:
                            pass

                        # Track repeated failures for stuck detection
                        normalized_action = current_action.lower().strip()
                        repeated_failure_count[normalized_action] = repeated_failure_count.get(normalized_action, 0) + 1

                        # Check if stuck (same action failed 3+ times)
                        if repeated_failure_count[normalized_action] >= 3:
                            try:
                                self.event_logger.system_warning(
                                    f"Stuck: Action '{current_action}' failed {repeated_failure_count[normalized_action]} times"
                                )
                            except Exception:
                                pass

                            # If agent hasn't issued ask: command by now, fail early
                            # Check if current plan has ask: in any step
                            has_ask = any(
                                step.action.lower().startswith("ask:")
                                for step in action_plan.steps
                            )

                            if not has_ask:
                                # Agent is stuck and not asking for help - fail early
                                return TaskResult(
                                    success=False,
                                    confidence=0.0,
                                    reasoning=f"Task stuck: Action '{current_action}' failed {repeated_failure_count[normalized_action]} times without resolution",
                                    evidence={
                                        "stuck_action": current_action,
                                        "failure_count": repeated_failure_count[normalized_action],
                                        "iterations": iteration + 1,
                                    },
                                )
                            # else: Agent is asking for help in this plan, let it execute

                        # Step failed, break out of step loop to get new plan
                        plan_failed = True
                        break

                    elif not result.metadata.get("page_changed", False):
                        task_ineffective_actions.append(current_action)

                    # Track last successful action
                    if result.success:
                        last_successful_action = current_action

                    # Auto-complete for PURE extraction tasks (Option C: Multi-Action Detection)
                    # Only auto-complete if the task instruction contains ONLY extraction verbs
                    if result.success and current_action.lower().startswith("extract:"):
                        # Validate extraction against schema if available
                        if extraction_schema and hasattr(result, 'data') and result.data:
                            validation_result = self._validate_extraction_schema(
                                result.data,
                                extraction_schema,
                                current_action
                            )
                            if not validation_result["valid"]:
                                # Log validation failure but don't fail the task
                                # The agent can try again in the next iteration
                                try:
                                    self.event_logger.system_warning(
                                        f"⚠️ Extraction schema validation failed: {validation_result['error']}"
                                    )
                                    self.event_logger.system_info(
                                        f"   Expected fields: {validation_result['expected_fields']}"
                                    )
                                    self.event_logger.system_info(
                                        f"   Actual fields: {validation_result['actual_fields']}"
                                    )
                                except Exception:
                                    pass
                                # Mark as failed so agent can retry with correct schema
                                task_failed_actions.append(current_action)
                                plan_failed = True
                                break
                            else:
                                # Validation passed
                                try:
                                    self.event_logger.system_info(
                                        f"✓ Extraction schema validation passed"
                                    )
                                except Exception:
                                    pass

                        # Define all possible action verbs
                        action_verbs = [
                            "click", "type", "select", "scroll", "press", "navigate",
                            "back", "forward", "wait", "defer", "upload", "datetime",
                            "form", "mini_goal"
                        ]

                        # Check if task has any non-extraction action verbs
                        task_lower = task_instruction.lower()
                        has_other_actions = any(verb in task_lower for verb in action_verbs)

                        # Debug logging for extraction auto-complete decision
                        try:
                            self.event_logger.system_debug(
                                f"[Extraction auto-complete check] "
                                f"has_other_actions={has_other_actions}, "
                                f"task='{task_instruction[:80]}...'"
                            )
                        except Exception:
                            pass

                        # If this is a pure extraction task (no other actions), auto-complete
                        if not has_other_actions:
                            # Log mini-loop completion
                            try:
                                self.event_logger.system_info(
                                    f"✓ Pure extraction task auto-completed: {current_action}"
                                )
                            except Exception:
                                pass

                            # Extraction succeeded, task is complete
                            return TaskResult(
                                success=True,
                                confidence=1.0,
                                reasoning=f"Extraction completed successfully: {current_action}",
                                evidence={
                                    "iterations": iteration + 1,
                                    "actions_tried": iteration + 1,
                                    "extracted_data": result.data if hasattr(result, "data") else None,
                                },
                            )
                        else:
                            # Multi-action task, agent must issue complete: command
                            try:
                                self.event_logger.system_debug(
                                    f"[Extraction] Multi-action task detected, waiting for agent to issue complete: command"
                                )
                            except Exception:
                                pass
                        # else: Multi-action task, let agent issue complete: command

                except Exception as e:
                    task_failed_actions.append(current_action)
                    plan_failed = True
                    break

            # End of step loop - if we get here, all steps in plan were executed
            # Continue to next mini-loop iteration to get a new plan

        # Max iterations reached without completion
        return TaskResult(
            success=False,
            confidence=0.0,
            reasoning=f"Task did not complete within {max_iterations} iterations",
            evidence={"max_iterations": max_iterations, "task_instruction": task_instruction},
        )

    def _record_iteration_success(
        self,
        sequential_task: SequentialTask,
        result: ActionResult,
    ) -> None:
        """
        Record a successful iteration in sequential task.

        Args:
            sequential_task: The sequential task
            result: The successful action result
        """
        iteration_idx = sequential_task.state.current_iteration
        attempts = sequential_task.state.iteration_attempts + 1

        # Create iteration result
        iter_result = IterationResult(
            iteration=iteration_idx,
            status="success",
            attempts=attempts,
            result=result.data,
            tasks_attempted=[sequential_task.current_subtask] if sequential_task.current_subtask else [],
        )

        # Add to completed iterations
        sequential_task.state.completed_iterations.append(iter_result)
        sequential_task.state.total_success_count += 1

        # Add result to task results
        sequential_task.results.append(result.data)

        # Log success
        try:
            # Check if we have actions_tried info from mini-loop (in result.data from ActionResult)
            actions_tried = None
            if hasattr(result, 'data') and isinstance(result.data, dict):
                actions_tried = result.data.get("actions_tried")

            if actions_tried and actions_tried > 1:
                self.event_logger.system_info(
                    f"ℹ️ ✓ Iteration {iteration_idx} successful after {attempts} attempt(s) "
                    f"({actions_tried} actions tried in mini-loop)"
                )
            else:
                self.event_logger.system_info(f"ℹ️ ✓ Iteration {iteration_idx} successful after {attempts} attempt(s)")
        except Exception:
            pass

    def _record_iteration_failure(
        self,
        sequential_task: SequentialTask,
        result: ActionResult,
    ) -> None:
        """
        Record a failed iteration in sequential task.

        Args:
            sequential_task: The sequential task
            result: The failed action result
        """
        iteration_idx = sequential_task.state.current_iteration
        attempts = sequential_task.state.iteration_attempts

        # Create iteration result
        iter_result = IterationResult(
            iteration=iteration_idx,
            status="failed",
            attempts=attempts,
            error=result.error,
            tasks_attempted=[sequential_task.current_subtask] if sequential_task.current_subtask else [],
        )

        # Add to completed iterations
        sequential_task.state.completed_iterations.append(iter_result)
        sequential_task.state.total_failure_count += 1

        # Add None to results to preserve indexing
        sequential_task.results.append(None)

        # Log failure
        try:
            self.event_logger.system_error(f"✗ Iteration {iteration_idx} failed after {attempts} attempt(s)")
        except Exception:
            pass
