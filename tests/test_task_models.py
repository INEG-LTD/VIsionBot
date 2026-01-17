"""
Unit tests for task models.
"""

import pytest
import time
from models.models import (
    TaskType,
    TaskStatus,
    BaseTask,
    NormalTask,
    SequentialTask,
    TaskList,
    IterationResult,
    SequentialState,
    BridgePlannerDecision,
    TaskOrchestratorOutput,
)


class TestTaskEnums:
    """Test task enums"""

    def test_task_type_values(self):
        """Test TaskType enum values"""
        assert TaskType.NORMAL == "normal"
        assert TaskType.SEQUENTIAL == "sequential"

    def test_task_status_values(self):
        """Test TaskStatus enum values"""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"


class TestNormalTask:
    """Test NormalTask model"""

    def test_create_normal_task(self):
        """Test creating a normal task"""
        task = NormalTask(
            task_id="task_001",
            description="Click the login button",
            instruction="Click the login button on the page",
        )

        assert task.task_id == "task_001"
        assert task.type == TaskType.NORMAL
        assert task.description == "Click the login button"
        assert task.instruction == "Click the login button on the page"
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.error is None
        assert task.depends_on is None

    def test_normal_task_with_dependency(self):
        """Test normal task with dependency"""
        task = NormalTask(
            task_id="task_002",
            description="Save results",
            instruction="Save the extracted data to file",
            depends_on="task_001",
        )

        assert task.depends_on == "task_001"

    def test_normal_task_result_storage(self):
        """Test storing result in normal task"""
        task = NormalTask(
            task_id="task_001",
            description="Extract data",
            instruction="Extract company name",
        )

        task.result = {"company": "Apple Inc."}
        task.status = TaskStatus.COMPLETED

        assert task.result == {"company": "Apple Inc."}
        assert task.status == TaskStatus.COMPLETED


class TestSequentialTask:
    """Test SequentialTask model"""

    def test_create_sequential_task(self):
        """Test creating a sequential task"""
        task = SequentialTask(
            task_id="task_001",
            description="Extract company names",
            goal="Extract company names from job listings",
            completion_condition="when 5 company names extracted",
            target_count=5,
        )

        assert task.task_id == "task_001"
        assert task.type == TaskType.SEQUENTIAL
        assert task.goal == "Extract company names from job listings"
        assert task.completion_condition == "when 5 company names extracted"
        assert task.target_count == 5
        assert task.status == TaskStatus.PENDING
        assert len(task.results) == 0
        assert task.state.current_iteration == 0
        assert task.state.total_success_count == 0
        assert task.state.total_failure_count == 0

    def test_sequential_task_indefinite(self):
        """Test sequential task with no target count"""
        task = SequentialTask(
            task_id="task_001",
            description="Extract all visible companies",
            goal="Extract all company names visible on page",
            completion_condition="when no more companies visible",
            target_count=None,
        )

        assert task.target_count is None

    def test_sequential_state_tracking(self):
        """Test sequential state updates"""
        task = SequentialTask(
            task_id="task_001",
            description="Extract companies",
            goal="Extract company names",
            completion_condition="when 3 extracted",
            target_count=3,
        )

        # Simulate successful iteration
        task.state.current_iteration = 0
        task.state.iteration_attempts = 2
        task.results.append({"company": "Apple"})

        iter_result = IterationResult(
            iteration=0,
            status="success",
            attempts=2,
            result={"company": "Apple"},
            tasks_attempted=["Extract from listing 0"],
        )
        task.state.completed_iterations.append(iter_result)
        task.state.total_success_count += 1

        assert task.state.total_success_count == 1
        assert len(task.state.completed_iterations) == 1
        assert len(task.results) == 1

        # Move to next iteration
        task.state.current_iteration = 1
        task.state.iteration_attempts = 0

        assert task.state.current_iteration == 1
        assert task.state.iteration_attempts == 0


class TestIterationResult:
    """Test IterationResult model"""

    def test_create_success_iteration(self):
        """Test creating successful iteration result"""
        result = IterationResult(
            iteration=0,
            status="success",
            attempts=1,
            result={"company": "Apple Inc."},
            tasks_attempted=["Extract company name from listing 0"],
        )

        assert result.iteration == 0
        assert result.status == "success"
        assert result.attempts == 1
        assert result.result == {"company": "Apple Inc."}
        assert len(result.tasks_attempted) == 1

    def test_create_failed_iteration(self):
        """Test creating failed iteration result"""
        result = IterationResult(
            iteration=1,
            status="failed",
            attempts=3,
            error="Element not found",
            tasks_attempted=[
                "Extract from listing 1",
                "Extract using alt selector",
                "Extract using text match",
            ],
        )

        assert result.iteration == 1
        assert result.status == "failed"
        assert result.attempts == 3
        assert result.error == "Element not found"
        assert len(result.tasks_attempted) == 3


class TestTaskList:
    """Test TaskList model"""

    def test_create_task_list(self):
        """Test creating a task list"""
        task1 = NormalTask(
            task_id="task_001",
            description="Navigate",
            instruction="Go to linkedin.com",
        )
        task2 = SequentialTask(
            task_id="task_002",
            description="Extract data",
            goal="Extract company names",
            completion_condition="when 5 extracted",
            target_count=5,
        )

        task_list = TaskList(tasks=[task1, task2])

        assert len(task_list.tasks) == 2
        assert task_list.current_task_index == 0

    def test_get_current_task(self):
        """Test getting current task"""
        task1 = NormalTask(task_id="task_001", description="Task 1", instruction="Do task 1")
        task2 = NormalTask(task_id="task_002", description="Task 2", instruction="Do task 2")

        task_list = TaskList(tasks=[task1, task2])

        current = task_list.get_current_task()
        assert current == task1
        assert current.task_id == "task_001"

    def test_advance_to_next_task(self):
        """Test advancing to next task"""
        task1 = NormalTask(task_id="task_001", description="Task 1", instruction="Do task 1")
        task2 = NormalTask(task_id="task_002", description="Task 2", instruction="Do task 2")

        task_list = TaskList(tasks=[task1, task2])

        assert task_list.current_task_index == 0

        has_more = task_list.advance_to_next_task()
        assert has_more is True
        assert task_list.current_task_index == 1

        has_more = task_list.advance_to_next_task()
        assert has_more is False
        assert task_list.current_task_index == 2

    def test_get_completed_tasks(self):
        """Test getting completed tasks"""
        task1 = NormalTask(task_id="task_001", description="Task 1", instruction="Do task 1")
        task1.status = TaskStatus.COMPLETED

        task2 = NormalTask(task_id="task_002", description="Task 2", instruction="Do task 2")
        task2.status = TaskStatus.IN_PROGRESS

        task3 = NormalTask(task_id="task_003", description="Task 3", instruction="Do task 3")
        task3.status = TaskStatus.COMPLETED

        task_list = TaskList(tasks=[task1, task2, task3])

        completed = task_list.get_completed_tasks()
        assert len(completed) == 2
        assert task1 in completed
        assert task3 in completed

    def test_get_pending_tasks(self):
        """Test getting pending tasks"""
        task1 = NormalTask(task_id="task_001", description="Task 1", instruction="Do task 1")
        task1.status = TaskStatus.COMPLETED

        task2 = NormalTask(task_id="task_002", description="Task 2", instruction="Do task 2")
        task2.status = TaskStatus.PENDING

        task3 = NormalTask(task_id="task_003", description="Task 3", instruction="Do task 3")
        task3.status = TaskStatus.PENDING

        task_list = TaskList(tasks=[task1, task2, task3])

        pending = task_list.get_pending_tasks()
        assert len(pending) == 2
        assert task2 in pending
        assert task3 in pending

    def test_all_tasks_completed(self):
        """Test checking if all tasks completed"""
        task1 = NormalTask(task_id="task_001", description="Task 1", instruction="Do task 1")
        task1.status = TaskStatus.COMPLETED

        task2 = NormalTask(task_id="task_002", description="Task 2", instruction="Do task 2")
        task2.status = TaskStatus.COMPLETED

        task_list = TaskList(tasks=[task1, task2])

        assert task_list.all_tasks_completed() is True

        task2.status = TaskStatus.PENDING
        assert task_list.all_tasks_completed() is False

    def test_get_task_by_id(self):
        """Test getting task by ID"""
        task1 = NormalTask(task_id="task_001", description="Task 1", instruction="Do task 1")
        task2 = NormalTask(task_id="task_002", description="Task 2", instruction="Do task 2")

        task_list = TaskList(tasks=[task1, task2])

        found = task_list.get_task_by_id("task_002")
        assert found == task2
        assert found.task_id == "task_002"

        not_found = task_list.get_task_by_id("task_999")
        assert not_found is None


class TestBridgePlannerDecision:
    """Test BridgePlannerDecision model"""

    def test_generate_task_decision(self):
        """Test generate task decision"""
        decision = BridgePlannerDecision(
            decision="generate_task",
            reasoning="More items to process",
            next_task="Extract company name from listing #3",
        )

        assert decision.decision == "generate_task"
        assert decision.next_task == "Extract company name from listing #3"
        assert decision.completion_reason is None

    def test_end_sequence_decision(self):
        """Test end sequence decision"""
        decision = BridgePlannerDecision(
            decision="end_sequence",
            reasoning="Target count reached",
            completion_reason="Successfully extracted 5 company names",
        )

        assert decision.decision == "end_sequence"
        assert decision.completion_reason == "Successfully extracted 5 company names"
        assert decision.next_task is None


class TestTaskOrchestratorOutput:
    """Test TaskOrchestratorOutput model"""

    def test_orchestrator_output(self):
        """Test task orchestrator output"""
        task1 = NormalTask(
            task_id="task_001",
            description="Navigate",
            instruction="Go to site",
        )
        task2 = SequentialTask(
            task_id="task_002",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when done",
        )

        output = TaskOrchestratorOutput(
            tasks=[task1, task2],
            reasoning="Decomposed into navigation and extraction tasks",
            confidence=0.9,
        )

        assert len(output.tasks) == 2
        assert output.reasoning == "Decomposed into navigation and extraction tasks"
        assert output.confidence == 0.9
