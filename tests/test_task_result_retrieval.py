"""
Unit tests for task result retrieval system.
"""

import pytest
from models.task_models import (
    TaskList,
    NormalTask,
    SequentialTask,
    TaskStatus,
)
from agent.task_result_retrieval import (
    TaskResultRetriever,
    TaskResultAccessor,
    TaskMatchResult,
)


class TestTaskResultRetriever:
    """Test TaskResultRetriever"""

    def test_extract_keywords(self):
        """Test keyword extraction"""
        retriever = TaskResultRetriever(use_llm_matching=False)

        keywords = retriever._extract_keywords("Extract company names from the job listings")

        # Should extract meaningful words, exclude stop words
        assert "extract" in keywords
        assert "company" in keywords
        assert "names" in keywords
        assert "job" in keywords
        assert "listings" in keywords

        # Stop words should be excluded
        assert "the" not in keywords
        assert "from" not in keywords

    def test_keyword_based_matching(self):
        """Test keyword-based task matching"""
        retriever = TaskResultRetriever(use_llm_matching=False)

        # Create tasks
        task1 = NormalTask(
            task_id="task_001",
            description="Navigate to LinkedIn",
            instruction="Go to linkedin.com",
            status=TaskStatus.COMPLETED,
        )
        task1.result = {"url": "https://linkedin.com"}

        task2 = SequentialTask(
            task_id="task_002",
            description="Extract company names from job listings",
            goal="Extract company names",
            completion_condition="when 5 extracted",
            status=TaskStatus.COMPLETED,
        )
        task2.results = ["Apple", "Google", "Meta"]

        task3 = NormalTask(
            task_id="task_003",
            description="Save salary data to file",
            instruction="Save salaries to file",
            status=TaskStatus.COMPLETED,
        )
        task3.result = {"file": "salaries.txt"}

        tasks = [task1, task2, task3]

        # Match for company names
        matches = retriever._keyword_based_matching(
            query="the company names we extracted",
            tasks=tasks,
        )

        # task2 should match best (has "company", "names", "extract")
        assert len(matches) > 0
        assert matches[0] == task2

    def test_extract_results_from_normal_task(self):
        """Test extracting results from normal task"""
        retriever = TaskResultRetriever()

        task = NormalTask(
            task_id="task_001",
            description="Extract data",
            instruction="Extract company name",
            status=TaskStatus.COMPLETED,
        )
        task.result = {"company": "Apple Inc."}

        results = retriever.extract_results_from_tasks([task])

        assert len(results) == 1
        assert results[0] == {"company": "Apple Inc."}

    def test_extract_results_from_sequential_task(self):
        """Test extracting results from sequential task"""
        retriever = TaskResultRetriever()

        task = SequentialTask(
            task_id="task_001",
            description="Extract companies",
            goal="Extract company names",
            completion_condition="when 5 extracted",
            status=TaskStatus.COMPLETED,
        )
        task.results = [
            {"company": "Apple"},
            {"company": "Google"},
            None,  # Failed iteration
            {"company": "Meta"},
        ]

        results = retriever.extract_results_from_tasks([task])

        # Should extract valid results, filter None
        assert len(results) == 3
        assert {"company": "Apple"} in results
        assert {"company": "Google"} in results
        assert {"company": "Meta"} in results

    def test_extract_results_no_results(self):
        """Test extracting from tasks with no results"""
        retriever = TaskResultRetriever()

        task = NormalTask(
            task_id="task_001",
            description="Navigate",
            instruction="Go to site",
            status=TaskStatus.COMPLETED,
        )
        # No result set

        results = retriever.extract_results_from_tasks([task])

        assert len(results) == 0


class TestTaskResultAccessor:
    """Test TaskResultAccessor"""

    def setup_method(self):
        """Setup test fixtures"""
        # Create sample task list
        self.task1 = NormalTask(
            task_id="task_001",
            description="Navigate to LinkedIn",
            instruction="Go to linkedin.com",
            status=TaskStatus.COMPLETED,
        )
        self.task1.result = {"url": "https://linkedin.com"}

        self.task2 = SequentialTask(
            task_id="task_002",
            description="Extract company names from job listings",
            goal="Extract company names",
            completion_condition="when 5 extracted",
            target_count=5,
            status=TaskStatus.COMPLETED,
        )
        self.task2.results = [
            {"company": "Apple Inc."},
            {"company": "Google LLC"},
            {"company": "Meta Platforms"},
            {"company": "Amazon"},
            {"company": "Netflix"},
        ]

        self.task3 = NormalTask(
            task_id="task_003",
            description="Save company names to file",
            instruction="Save to companies.txt",
            status=TaskStatus.COMPLETED,
        )
        self.task3.result = {"file": "companies.txt", "count": 5}

        self.task_list = TaskList(tasks=[self.task1, self.task2, self.task3])

    def test_get_results_keyword_matching(self):
        """Test getting results with keyword matching"""
        retriever = TaskResultRetriever(use_llm_matching=False)
        accessor = TaskResultAccessor(
            task_list=self.task_list,
            retriever=retriever,
        )

        # Query for company names
        results = accessor.get_results("company names")

        # Should return first result from task2
        assert results is not None
        assert results == {"company": "Apple Inc."}

    def test_get_results_return_all(self):
        """Test getting all matching results"""
        retriever = TaskResultRetriever(use_llm_matching=False)
        accessor = TaskResultAccessor(
            task_list=self.task_list,
            retriever=retriever,
        )

        # Query for company names, return all
        results = accessor.get_results("company names", return_first_only=False)

        # Should return all results from task2
        assert results is not None
        assert len(results) == 5

    def test_get_results_no_match(self):
        """Test getting results with no match"""
        retriever = TaskResultRetriever(use_llm_matching=False)
        accessor = TaskResultAccessor(
            task_list=self.task_list,
            retriever=retriever,
        )

        # Query for something that doesn't match
        results = accessor.get_results("salary information")

        assert results is None

    def test_get_all_results(self):
        """Test getting all results from all tasks"""
        retriever = TaskResultRetriever(use_llm_matching=False)
        accessor = TaskResultAccessor(
            task_list=self.task_list,
            retriever=retriever,
        )

        all_results = accessor.get_all_results()

        # Should have results from all 3 tasks
        # task1: 1 result, task2: 5 results, task3: 1 result = 7 total
        assert len(all_results) == 7


class TestKeywordExtraction:
    """Test keyword extraction edge cases"""

    def test_empty_string(self):
        """Test keyword extraction from empty string"""
        retriever = TaskResultRetriever()
        keywords = retriever._extract_keywords("")
        assert len(keywords) == 0

    def test_only_stop_words(self):
        """Test keyword extraction from only stop words"""
        retriever = TaskResultRetriever()
        keywords = retriever._extract_keywords("the and or but")
        assert len(keywords) == 0

    def test_special_characters(self):
        """Test keyword extraction with special characters"""
        retriever = TaskResultRetriever()
        keywords = retriever._extract_keywords("Extract company-names from job_listings!")

        assert "extract" in keywords
        assert "company" in keywords
        assert "names" in keywords
        assert "job" in keywords
        assert "listings" in keywords

    def test_case_insensitivity(self):
        """Test that keyword extraction is case-insensitive"""
        retriever = TaskResultRetriever()
        keywords1 = retriever._extract_keywords("Extract Company Names")
        keywords2 = retriever._extract_keywords("extract company names")

        assert keywords1 == keywords2


class TestTaskDependencies:
    """Test task dependency handling"""

    def test_task_with_dependency(self):
        """Test task that depends on another task"""
        task1 = SequentialTask(
            task_id="task_001",
            description="Extract companies",
            goal="Extract company names",
            completion_condition="when 5 extracted",
            status=TaskStatus.COMPLETED,
        )
        task1.results = ["Apple", "Google", "Meta"]

        task2 = NormalTask(
            task_id="task_002",
            description="Save companies",
            instruction="Save to file",
            depends_on="task_001",
            status=TaskStatus.PENDING,
        )

        task_list = TaskList(tasks=[task1, task2])

        # Get the dependent task
        dependent = task_list.get_task_by_id(task2.depends_on)

        assert dependent == task1
        assert dependent.status == TaskStatus.COMPLETED

        # Extract results from dependent task
        retriever = TaskResultRetriever()
        results = retriever.extract_results_from_tasks([dependent])

        assert len(results) == 3
        assert "Apple" in results


class TestFormatTasksForPrompt:
    """Test formatting tasks for LLM prompt"""

    def test_format_normal_task(self):
        """Test formatting normal task"""
        retriever = TaskResultRetriever()

        task_summaries = [
            {
                "index": 0,
                "id": "task_001",
                "type": "normal",
                "description": "Navigate to site",
                "instruction": "Go to linkedin.com",
                "has_result": True,
            }
        ]

        formatted = retriever._format_tasks_for_prompt(task_summaries)

        assert "task_001" in formatted
        assert "Navigate to site" in formatted
        assert "Normal Task" in formatted
        assert "Go to linkedin.com" in formatted

    def test_format_sequential_task(self):
        """Test formatting sequential task"""
        retriever = TaskResultRetriever()

        task_summaries = [
            {
                "index": 0,
                "id": "task_001",
                "type": "sequential",
                "description": "Extract companies",
                "goal": "Extract company names",
                "result_count": 5,
            }
        ]

        formatted = retriever._format_tasks_for_prompt(task_summaries)

        assert "task_001" in formatted
        assert "Extract companies" in formatted
        assert "Sequential Task" in formatted
        assert "Extract company names" in formatted
        assert "5" in formatted
