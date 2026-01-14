"""
Pytest configuration and shared fixtures for task system tests.
"""

import pytest
from unittest.mock import Mock, MagicMock
from models.task_models import (
    TaskList,
    NormalTask,
    SequentialTask,
    TaskStatus,
    IterationResult,
)
from bot_config import SequentialTaskConfig


@pytest.fixture
def sample_normal_task():
    """Create a sample normal task"""
    return NormalTask(
        task_id="task_001",
        description="Navigate to website",
        instruction="Go to linkedin.com",
    )


@pytest.fixture
def sample_sequential_task():
    """Create a sample sequential task"""
    return SequentialTask(
        task_id="task_002",
        description="Extract company names",
        goal="Extract company names from job listings",
        completion_condition="when 5 company names extracted",
        target_count=5,
    )


@pytest.fixture
def sample_task_list(sample_normal_task, sample_sequential_task):
    """Create a sample task list"""
    save_task = NormalTask(
        task_id="task_003",
        description="Save results",
        instruction="Save company names to file",
        depends_on="task_002",
    )

    return TaskList(
        tasks=[sample_normal_task, sample_sequential_task, save_task]
    )


@pytest.fixture
def completed_task_list():
    """Create a task list with completed tasks"""
    task1 = NormalTask(
        task_id="task_001",
        description="Navigate to LinkedIn",
        instruction="Go to linkedin.com",
        status=TaskStatus.COMPLETED,
    )
    task1.result = {"url": "https://linkedin.com"}

    task2 = SequentialTask(
        task_id="task_002",
        description="Extract company names",
        goal="Extract company names",
        completion_condition="when 5 extracted",
        target_count=5,
        status=TaskStatus.COMPLETED,
    )
    task2.results = [
        {"company": "Apple Inc."},
        {"company": "Google LLC"},
        {"company": "Meta Platforms"},
        None,  # Failed iteration
        {"company": "Amazon"},
    ]
    task2.state.total_success_count = 4
    task2.state.total_failure_count = 1

    task3 = NormalTask(
        task_id="task_003",
        description="Save to file",
        instruction="Save companies to file",
        status=TaskStatus.COMPLETED,
        depends_on="task_002",
    )
    task3.result = {"file": "companies.txt", "count": 4}

    return TaskList(tasks=[task1, task2, task3])


@pytest.fixture
def default_sequential_config():
    """Create default sequential task config"""
    return SequentialTaskConfig(
        max_attempts_per_iteration=3,
        completion_strategy="best_effort",
        success_threshold=0.6,
        max_total_iterations=50,
        fail_fast=False,
    )


@pytest.fixture
def strict_sequential_config():
    """Create strict sequential task config"""
    return SequentialTaskConfig(
        max_attempts_per_iteration=3,
        completion_strategy="strict",
        max_total_iterations=50,
    )


@pytest.fixture
def threshold_sequential_config():
    """Create threshold sequential task config"""
    return SequentialTaskConfig(
        max_attempts_per_iteration=5,
        completion_strategy="threshold",
        success_threshold=0.8,
        max_total_iterations=100,
    )


@pytest.fixture
def mock_environment_state():
    """Create mock environment state"""
    from agent.completion_contract import EnvironmentState

    mock_state = Mock(spec=EnvironmentState)
    mock_state.current_url = "https://example.com"
    mock_state.page_title = "Example Page"
    mock_state.browser_state = Mock()
    mock_state.browser_state.url = "https://example.com"
    mock_state.browser_state.title = "Example Page"
    mock_state.interaction_history = []

    return mock_state


@pytest.fixture
def mock_screenshot():
    """Create mock screenshot bytes"""
    return b"fake_screenshot_data"


@pytest.fixture
def mock_overlay_data():
    """Create mock overlay data"""
    return [
        {"type": "button", "text": "Submit", "index": 1},
        {"type": "link", "text": "Home", "index": 2},
        {"type": "input", "placeholder": "Search", "index": 3},
    ]


@pytest.fixture
def sample_iteration_results():
    """Create sample iteration results"""
    return [
        IterationResult(
            iteration=0,
            status="success",
            attempts=1,
            result={"company": "Apple Inc."},
            tasks_attempted=["Extract from listing 0"],
        ),
        IterationResult(
            iteration=1,
            status="failed",
            attempts=3,
            error="Element not found",
            tasks_attempted=[
                "Extract from listing 1",
                "Try alternative selector",
                "Try text match",
            ],
        ),
        IterationResult(
            iteration=2,
            status="success",
            attempts=1,
            result={"company": "Google LLC"},
            tasks_attempted=["Extract from listing 2"],
        ),
    ]
