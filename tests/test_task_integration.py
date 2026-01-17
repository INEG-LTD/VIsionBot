"""
Integration tests for task system end-to-end flows.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from models.models import (
    TaskList,
    NormalTask,
    SequentialTask,
    TaskStatus,
    TaskType,
    IterationResult,
)
from core.config import SequentialTaskConfig
from agent.subagent.retrieval import TaskResultRetriever, TaskResultAccessor


class TestEndToEndSequentialExecution:
    """Test end-to-end sequential task execution scenarios"""

    def test_successful_sequential_completion(self, sample_sequential_task, default_sequential_config):
        """Test successful completion of sequential task"""
        from agent.planning.bridge import BridgePlanner

        task = sample_sequential_task
        planner = BridgePlanner(config=default_sequential_config)

        # Simulate successful iterations
        for i in range(5):
            # Record success
            task.state.completed_iterations.append(
                IterationResult(
                    iteration=i,
                    status="success",
                    attempts=1,
                    result={"company": f"Company{i}"},
                )
            )
            task.state.total_success_count += 1
            task.results.append({"company": f"Company{i}"})

        task.state.current_iteration = 5

        # Check if should end
        should_end = planner.should_end_sequence(task)

        assert should_end is True
        assert task.state.total_success_count == 5
        assert task.state.total_failure_count == 0
        assert len(task.results) == 5

    def test_sequential_with_retries(self, sample_sequential_task, default_sequential_config):
        """Test sequential task with retry attempts"""
        from agent.planning.bridge import BridgePlanner

        task = sample_sequential_task
        planner = BridgePlanner(config=default_sequential_config)

        # Iteration 0: Success on first try
        task.state.iteration_attempts = 0
        assert planner.should_retry_iteration(task) is True

        task.state.completed_iterations.append(
            IterationResult(iteration=0, status="success", attempts=1, result={"company": "Apple"})
        )
        task.state.total_success_count += 1
        task.results.append({"company": "Apple"})

        # Move to iteration 1
        task.state.current_iteration = 1
        task.state.iteration_attempts = 0

        # Iteration 1: Fail twice, succeed on third try
        for attempt in range(2):
            task.state.iteration_attempts = attempt + 1
            assert planner.should_retry_iteration(task, "error") is True

        # Third attempt succeeds
        task.state.iteration_attempts = 2
        task.state.completed_iterations.append(
            IterationResult(iteration=1, status="success", attempts=3, result={"company": "Google"})
        )
        task.state.total_success_count += 1
        task.results.append({"company": "Google"})

        assert task.state.total_success_count == 2
        assert len(task.results) == 2

    def test_sequential_partial_completion(self, sample_sequential_task, default_sequential_config):
        """Test sequential task with partial success"""
        from agent.planning.bridge import BridgePlanner

        task = sample_sequential_task
        planner = BridgePlanner(config=default_sequential_config)

        # 3 successes, 2 failures
        for i in range(3):
            task.state.completed_iterations.append(
                IterationResult(iteration=i, status="success", attempts=1, result={"company": f"Co{i}"})
            )
            task.state.total_success_count += 1
            task.results.append({"company": f"Co{i}"})

        for i in range(3, 5):
            task.state.completed_iterations.append(
                IterationResult(iteration=i, status="failed", attempts=3, error="Not found")
            )
            task.state.total_failure_count += 1
            task.results.append(None)

        task.state.current_iteration = 5

        # With best_effort, should complete
        should_end = planner.should_end_sequence(task)
        assert should_end is True

        # Verify results
        assert task.state.total_success_count == 3
        assert task.state.total_failure_count == 2
        assert len(task.results) == 5
        assert task.results.count(None) == 2


class TestTaskDependencyFlow:
    """Test task dependency and result flow"""

    def test_dependent_task_access(self, completed_task_list):
        """Test that dependent task can access previous results"""
        # Get the sequential task (task_002)
        sequential_task = completed_task_list.tasks[1]
        assert isinstance(sequential_task, SequentialTask)

        # Get the dependent task (task_003)
        dependent_task = completed_task_list.tasks[2]
        assert isinstance(dependent_task, NormalTask)
        assert dependent_task.depends_on == "task_002"

        # Verify dependency link works
        dependency = completed_task_list.get_task_by_id(dependent_task.depends_on)
        assert dependency == sequential_task

        # Extract results from dependency
        retriever = TaskResultRetriever()
        results = retriever.extract_results_from_tasks([dependency])

        # Should have 4 results (filtering out the None)
        assert len(results) == 4
        assert {"company": "Apple Inc."} in results

    def test_result_flow_through_multiple_tasks(self):
        """Test result flow through multiple dependent tasks"""
        # Task 1: Extract data
        task1 = SequentialTask(
            task_id="extract_task",
            description="Extract companies",
            goal="Extract company names",
            completion_condition="when done",
            status=TaskStatus.COMPLETED,
        )
        task1.results = ["Apple", "Google", "Meta"]

        # Task 2: Process data (depends on task1)
        task2 = NormalTask(
            task_id="process_task",
            description="Count companies",
            instruction="Count extracted companies",
            depends_on="extract_task",
            status=TaskStatus.COMPLETED,
        )
        task2.result = {"count": 3}

        # Task 3: Save data (depends on task2)
        task3 = NormalTask(
            task_id="save_task",
            description="Save count",
            instruction="Save company count",
            depends_on="process_task",
            status=TaskStatus.COMPLETED,
        )
        task3.result = {"file": "count.txt"}

        task_list = TaskList(tasks=[task1, task2, task3])

        # Verify dependency chain
        assert task2.depends_on == "extract_task"
        assert task3.depends_on == "process_task"

        # Each task can access its dependency
        dep1 = task_list.get_task_by_id(task2.depends_on)
        assert dep1 == task1

        dep2 = task_list.get_task_by_id(task3.depends_on)
        assert dep2 == task2


class TestResultRetrievalIntegration:
    """Test result retrieval integration with task execution"""

    def test_retrieve_results_after_execution(self, completed_task_list):
        """Test retrieving results after task execution"""
        accessor = TaskResultAccessor(
            task_list=completed_task_list,
            retriever=TaskResultRetriever(use_llm_matching=False),
        )

        # Query for company names
        results = accessor.get_results("company names", return_first_only=False)

        assert results is not None
        assert len(results) == 4  # 4 successful extractions
        assert {"company": "Apple Inc."} in results
        assert {"company": "Google LLC"} in results

    def test_retrieve_all_results(self, completed_task_list):
        """Test retrieving all results from completed tasks"""
        accessor = TaskResultAccessor(
            task_list=completed_task_list,
            retriever=TaskResultRetriever(),
        )

        all_results = accessor.get_all_results()

        # Should have: 1 from task1 + 4 from task2 + 1 from task3 = 6
        assert len(all_results) == 6

    def test_no_results_for_incomplete_tasks(self):
        """Test that incomplete tasks don't return results"""
        task = NormalTask(
            task_id="task_001",
            description="Extract data",
            instruction="Extract company name",
            status=TaskStatus.IN_PROGRESS,  # Not complete
        )

        task_list = TaskList(tasks=[task])
        accessor = TaskResultAccessor(task_list=task_list)

        # Should have no completed tasks
        completed = task_list.get_completed_tasks()
        assert len(completed) == 0

        # Should have no results
        all_results = accessor.get_all_results()
        assert len(all_results) == 0


class TestCompletionStrategyScenarios:
    """Test different completion strategy scenarios"""

    def test_strict_strategy_scenario(self):
        """Test strict completion strategy in realistic scenario"""
        from agent.planning.bridge import BridgePlanner

        config = SequentialTaskConfig(
            completion_strategy="strict",
            max_total_iterations=10,
        )
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract exactly 5 companies",
            goal="Extract company names",
            completion_condition="when exactly 5 extracted successfully",
            target_count=5,
        )

        # Scenario: 3 successes, 2 failures after 5 iterations
        task.state.total_success_count = 3
        task.state.total_failure_count = 2
        task.state.current_iteration = 5

        # Strict mode: should NOT end (need 5 successes)
        assert planner.should_end_sequence(task) is False

        # Continue and get 2 more successes
        task.state.total_success_count = 5
        task.state.current_iteration = 7

        # Now should end (have 5 successes)
        assert planner.should_end_sequence(task) is True

    def test_threshold_strategy_scenario(self):
        """Test threshold completion strategy"""
        from agent.planning.bridge import BridgePlanner

        config = SequentialTaskConfig(
            completion_strategy="threshold",
            success_threshold=0.7,
            max_total_iterations=10,
        )
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract companies with 70% success rate",
            goal="Extract company names",
            completion_condition="when 70% success rate achieved",
            target_count=10,
        )

        # Scenario: 7 successes, 3 failures (70% rate)
        task.state.total_success_count = 7
        task.state.total_failure_count = 3
        task.state.current_iteration = 10

        # Should end: 7/10 = 70% >= 70% threshold
        assert planner.should_end_sequence(task) is True

        # Scenario 2: 6 successes, 4 failures (60% rate)
        task.state.total_success_count = 6
        task.state.total_failure_count = 4

        # Should NOT end: 6/10 = 60% < 70%
        assert planner.should_end_sequence(task) is False


class TestErrorRecoveryScenarios:
    """Test error recovery scenarios"""

    def test_max_retries_then_continue(self):
        """Test that task continues after max retries exhausted"""
        from agent.planning.bridge import BridgePlanner

        config = SequentialTaskConfig(
            max_attempts_per_iteration=3,
            fail_fast=False,
        )
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract with retries",
            goal="Extract companies",
            completion_condition="when 5 extracted",
            target_count=5,
        )

        # Iteration 0: Fail 3 times
        task.state.current_iteration = 0
        task.state.iteration_attempts = 3

        # Should not retry (at limit)
        assert planner.should_retry_iteration(task, "error") is False

        # Record failure and move on
        task.state.completed_iterations.append(
            IterationResult(iteration=0, status="failed", attempts=3, error="Failed")
        )
        task.state.total_failure_count += 1
        task.results.append(None)

        # Move to iteration 1
        task.state.current_iteration = 1
        task.state.iteration_attempts = 0

        # Should continue (fail_fast=False)
        assert planner.should_end_sequence(task) is False

    def test_fail_fast_on_error(self):
        """Test fail fast behavior on error"""
        from agent.planning.bridge import BridgePlanner

        config = SequentialTaskConfig(
            max_attempts_per_iteration=3,
            fail_fast=True,
        )
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract with fail fast",
            goal="Extract companies",
            completion_condition="when 5 extracted",
            target_count=5,
        )

        # Record 1 failure
        task.state.total_failure_count = 1
        task.state.current_iteration = 1

        # Should end immediately (fail_fast=True)
        assert planner.should_end_sequence(task) is True


class TestTaskListManagement:
    """Test task list management during execution"""

    def test_task_list_progression(self, sample_task_list):
        """Test progression through task list"""
        task_list = sample_task_list

        # Start at task 0
        assert task_list.current_task_index == 0
        current = task_list.get_current_task()
        assert current.task_id == "task_001"

        # Complete task 1
        current.status = TaskStatus.COMPLETED
        task_list.advance_to_next_task()

        # Now at task 2 (sequential)
        assert task_list.current_task_index == 1
        current = task_list.get_current_task()
        assert current.task_id == "task_002"
        assert isinstance(current, SequentialTask)

        # Complete task 2
        current.status = TaskStatus.COMPLETED
        task_list.advance_to_next_task()

        # Now at task 3 (dependent)
        assert task_list.current_task_index == 2
        current = task_list.get_current_task()
        assert current.task_id == "task_003"
        assert current.depends_on == "task_002"

        # Complete task 3
        current.status = TaskStatus.COMPLETED
        has_more = task_list.advance_to_next_task()

        # No more tasks
        assert has_more is False
        assert task_list.all_tasks_completed() is True

    def test_partial_task_list_completion(self):
        """Test task list with some completed, some pending"""
        task1 = NormalTask(task_id="t1", description="Task 1", instruction="Do 1")
        task1.status = TaskStatus.COMPLETED

        task2 = NormalTask(task_id="t2", description="Task 2", instruction="Do 2")
        task2.status = TaskStatus.IN_PROGRESS

        task3 = NormalTask(task_id="t3", description="Task 3", instruction="Do 3")
        task3.status = TaskStatus.PENDING

        task_list = TaskList(tasks=[task1, task2, task3])

        # Check completion status
        assert task_list.all_tasks_completed() is False

        completed = task_list.get_completed_tasks()
        assert len(completed) == 1

        pending = task_list.get_pending_tasks()
        assert len(pending) == 1
