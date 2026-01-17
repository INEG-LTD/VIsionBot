"""
Unit tests for Bridge Planner.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from models.models import (
    SequentialTask,
    SequentialState,
    IterationResult,
    BridgePlannerDecision,
)
from core.config import SequentialTaskConfig
from agent.planning.bridge import BridgePlanner


class TestBridgePlannerRetryLogic:
    """Test Bridge Planner retry logic"""

    def test_should_retry_iteration_under_limit(self):
        """Test that retry is allowed when under limit"""
        config = SequentialTaskConfig(max_attempts_per_iteration=3)
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when 5 extracted",
            target_count=5,
        )
        task.state.iteration_attempts = 2

        should_retry = planner.should_retry_iteration(task, "some error")

        assert should_retry is True

    def test_should_not_retry_at_limit(self):
        """Test that retry is not allowed when at limit"""
        config = SequentialTaskConfig(max_attempts_per_iteration=3)
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when 5 extracted",
            target_count=5,
        )
        task.state.iteration_attempts = 3

        should_retry = planner.should_retry_iteration(task, "some error")

        assert should_retry is False


class TestBridgePlannerCompletionStrategies:
    """Test Bridge Planner completion strategies"""

    def test_strict_strategy_not_complete(self):
        """Test strict strategy when target not reached"""
        config = SequentialTaskConfig(
            completion_strategy="strict",
            max_total_iterations=50,
        )
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when 5 extracted",
            target_count=5,
        )
        task.state.total_success_count = 3
        task.state.total_failure_count = 0
        task.state.current_iteration = 3

        should_end = planner.should_end_sequence(task)

        assert should_end is False

    def test_strict_strategy_complete(self):
        """Test strict strategy when target reached"""
        config = SequentialTaskConfig(
            completion_strategy="strict",
            max_total_iterations=50,
        )
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when 5 extracted",
            target_count=5,
        )
        task.state.total_success_count = 5
        task.state.total_failure_count = 0
        task.state.current_iteration = 5

        should_end = planner.should_end_sequence(task)

        assert should_end is True

    def test_best_effort_strategy_not_complete(self):
        """Test best effort strategy when not all attempted"""
        config = SequentialTaskConfig(
            completion_strategy="best_effort",
            max_total_iterations=50,
        )
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when 5 attempted",
            target_count=5,
        )
        task.state.total_success_count = 2
        task.state.total_failure_count = 1
        task.state.current_iteration = 3

        should_end = planner.should_end_sequence(task)

        # Not all 5 attempted yet
        assert should_end is False

    def test_best_effort_strategy_complete(self):
        """Test best effort strategy when all attempted"""
        config = SequentialTaskConfig(
            completion_strategy="best_effort",
            max_total_iterations=50,
        )
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when 5 attempted",
            target_count=5,
        )
        task.state.total_success_count = 3
        task.state.total_failure_count = 2
        task.state.current_iteration = 5

        should_end = planner.should_end_sequence(task)

        # All 5 attempted (3 success + 2 failure)
        assert should_end is True

    def test_threshold_strategy_meets_threshold(self):
        """Test threshold strategy when threshold met"""
        config = SequentialTaskConfig(
            completion_strategy="threshold",
            success_threshold=0.6,
            max_total_iterations=50,
        )
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when threshold met",
            target_count=5,
        )
        task.state.total_success_count = 4
        task.state.total_failure_count = 1
        task.state.current_iteration = 5

        should_end = planner.should_end_sequence(task)

        # 4/5 = 0.8 >= 0.6 threshold, and all 5 attempted
        assert should_end is True

    def test_threshold_strategy_below_threshold(self):
        """Test threshold strategy when below threshold"""
        config = SequentialTaskConfig(
            completion_strategy="threshold",
            success_threshold=0.8,
            max_total_iterations=50,
        )
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when threshold met",
            target_count=5,
        )
        task.state.total_success_count = 3
        task.state.total_failure_count = 2
        task.state.current_iteration = 5

        should_end = planner.should_end_sequence(task)

        # 3/5 = 0.6 < 0.8 threshold
        assert should_end is False

    def test_indefinite_sequence_no_target(self):
        """Test indefinite sequence with no target count"""
        config = SequentialTaskConfig(
            completion_strategy="best_effort",
            max_total_iterations=50,
        )
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract all visible",
            goal="Extract all companies",
            completion_condition="when no more visible",
            target_count=None,  # Indefinite
        )
        task.state.total_success_count = 5
        task.state.current_iteration = 5

        should_end = planner.should_end_sequence(task)

        # Should not end automatically (no target), let LLM decide
        assert should_end is False


class TestBridgePlannerSafetyLimits:
    """Test Bridge Planner safety limits"""

    def test_max_iterations_reached(self):
        """Test that sequence ends at max iterations"""
        config = SequentialTaskConfig(
            max_total_iterations=10,
            completion_strategy="best_effort",
        )
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when done",
            target_count=50,  # More than max_total_iterations
        )
        task.state.current_iteration = 10
        task.state.total_success_count = 5

        should_end = planner.should_end_sequence(task)

        assert should_end is True

    def test_fail_fast_enabled(self):
        """Test fail fast mode"""
        config = SequentialTaskConfig(
            fail_fast=True,
            max_total_iterations=50,
        )
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when 5 extracted",
            target_count=5,
        )
        task.state.total_success_count = 2
        task.state.total_failure_count = 1  # Has a failure
        task.state.current_iteration = 3

        should_end = planner.should_end_sequence(task)

        # Should end due to fail_fast + failure
        assert should_end is True

    def test_fail_fast_disabled(self):
        """Test fail fast disabled"""
        config = SequentialTaskConfig(
            fail_fast=False,
            max_total_iterations=50,
        )
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when 5 extracted",
            target_count=5,
        )
        task.state.total_success_count = 2
        task.state.total_failure_count = 1
        task.state.current_iteration = 3

        should_end = planner.should_end_sequence(task)

        # Should continue despite failure
        assert should_end is False


class TestBridgePlannerEndReason:
    """Test Bridge Planner end reason generation"""

    def test_end_reason_max_iterations(self):
        """Test end reason for max iterations"""
        config = SequentialTaskConfig(max_total_iterations=10)
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when done",
        )
        task.state.current_iteration = 10

        reason = planner._get_end_reason(task)

        assert "10" in reason
        assert "limit" in reason.lower()

    def test_end_reason_fail_fast(self):
        """Test end reason for fail fast"""
        config = SequentialTaskConfig(fail_fast=True)
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when done",
        )
        task.state.total_failure_count = 1

        reason = planner._get_end_reason(task)

        assert "fail" in reason.lower()
        assert "1" in reason

    def test_end_reason_strict_complete(self):
        """Test end reason for strict completion"""
        config = SequentialTaskConfig(completion_strategy="strict")
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when 5 extracted",
            target_count=5,
        )
        task.state.total_success_count = 5

        reason = planner._get_end_reason(task)

        assert "5" in reason
        assert "success" in reason.lower()


class TestIterationHistoryFormatting:
    """Test iteration history formatting"""

    def test_format_iteration_history_empty(self):
        """Test formatting empty iteration history"""
        config = SequentialTaskConfig(include_iteration_history=True)
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when done",
        )

        formatted = planner._format_iteration_history(task)

        assert "None yet" in formatted or "first iteration" in formatted.lower()

    def test_format_iteration_history_with_results(self):
        """Test formatting iteration history with results"""
        config = SequentialTaskConfig(
            include_iteration_history=True,
            max_history_in_prompt=10,
        )
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when done",
        )

        # Add some iterations
        task.state.completed_iterations.append(
            IterationResult(
                iteration=0,
                status="success",
                attempts=1,
                result={"company": "Apple"},
                tasks_attempted=["Extract from listing 0"],
            )
        )
        task.state.completed_iterations.append(
            IterationResult(
                iteration=1,
                status="failed",
                attempts=3,
                error="Element not found",
                tasks_attempted=["Extract from listing 1", "Retry with alt selector"],
            )
        )

        formatted = planner._format_iteration_history(task)

        assert "Iteration 0" in formatted
        assert "SUCCESS" in formatted
        assert "Apple" in formatted
        assert "Iteration 1" in formatted
        assert "FAILED" in formatted

    def test_format_iteration_history_limit(self):
        """Test iteration history respects max limit"""
        config = SequentialTaskConfig(
            include_iteration_history=True,
            max_history_in_prompt=2,
        )
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when done",
        )

        # Add 5 iterations
        for i in range(5):
            task.state.completed_iterations.append(
                IterationResult(
                    iteration=i,
                    status="success",
                    attempts=1,
                    result={"company": f"Company{i}"},
                    tasks_attempted=[f"Extract {i}"],
                )
            )

        formatted = planner._format_iteration_history(task)

        # Should only show last 2 iterations
        assert "Iteration 3" in formatted
        assert "Iteration 4" in formatted
        assert "omitted" in formatted.lower()

    def test_format_iteration_history_disabled(self):
        """Test iteration history when disabled"""
        config = SequentialTaskConfig(include_iteration_history=False)
        planner = BridgePlanner(config=config)

        task = SequentialTask(
            task_id="task_001",
            description="Extract data",
            goal="Extract companies",
            completion_condition="when done",
        )

        # Add iterations
        task.state.completed_iterations.append(
            IterationResult(
                iteration=0,
                status="success",
                attempts=1,
                result={"company": "Apple"},
            )
        )

        formatted = planner._format_iteration_history(task)

        # Should indicate history not included
        assert "not included" in formatted.lower() or "configured to exclude" in formatted.lower()


class TestOverlaySummaryFormatting:
    """Test overlay summary formatting"""

    def test_format_overlay_summary_none(self):
        """Test formatting when no overlay data"""
        planner = BridgePlanner()

        formatted = planner._format_overlay_summary(None)

        assert "No overlay" in formatted or "not available" in formatted.lower()

    def test_format_overlay_summary_with_data(self):
        """Test formatting overlay summary with data"""
        planner = BridgePlanner()

        overlay_data = [
            {"type": "button", "text": "Submit"},
            {"type": "button", "text": "Cancel"},
            {"type": "link", "text": "Home"},
            {"type": "input", "placeholder": "Search"},
        ]

        formatted = planner._format_overlay_summary(overlay_data)

        assert "4" in formatted
        assert "button" in formatted
        assert "link" in formatted
        assert "input" in formatted


class TestNotebookFormatting:
    """Test notebook formatting"""

    def test_format_notebook_empty(self):
        """Test formatting empty notebook"""
        planner = BridgePlanner()

        formatted = planner._format_notebook(None)

        assert "Empty" in formatted or "no data" in formatted.lower()

    def test_format_notebook_with_entries(self):
        """Test formatting notebook with entries"""
        planner = BridgePlanner()

        notebook = [
            {"type": "extraction", "data": {"company": "Apple"}},
            {"type": "extraction", "data": {"company": "Google"}},
            {"type": "url", "url": "https://example.com"},
        ]

        formatted = planner._format_notebook(notebook)

        assert "Apple" in formatted
        assert "Google" in formatted
        assert "example.com" in formatted

    def test_format_notebook_shows_recent(self):
        """Test that notebook shows only recent entries"""
        planner = BridgePlanner()

        # Create 10 entries
        notebook = [
            {"entry": i, "data": f"data{i}"}
            for i in range(10)
        ]

        formatted = planner._format_notebook(notebook)

        # Should show last 5 entries
        assert "data9" in formatted
        assert "data8" in formatted
        assert "data7" in formatted
        assert "data6" in formatted
        assert "data5" in formatted

        # Should indicate omitted entries
        assert "omitted" in formatted.lower() or "5 most recent" in formatted.lower()
