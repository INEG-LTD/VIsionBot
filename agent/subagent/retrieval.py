"""
Task Result Retrieval - Natural language-based result access for tasks.

This module provides functionality to retrieve results from previously completed
tasks using natural language matching.
"""

from typing import Optional, List, Dict, Any, Union
import re

from models.models import TaskList, NormalTask, SequentialTask, TaskType
from lib.ai import generate_model
from pydantic import BaseModel, Field
from utils.debug_print import dprint, PrintMode


class TaskMatchResult(BaseModel):
    """Result of matching a query to tasks"""
    task_id: str = Field(description="ID of the matching task")
    task_description: str = Field(description="Description of the task")
    confidence: float = Field(description="Confidence in the match (0.0-1.0)", ge=0.0, le=1.0)
    reasoning: str = Field(description="Why this task matches the query")


class TaskResultRetriever:
    """
    Retrieves results from completed tasks using natural language queries.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_llm_matching: bool = True,
    ):
        """
        Initialize the task result retriever.

        Args:
            model_name: Model to use for matching (default: agent model)
            use_llm_matching: Whether to use LLM for semantic matching (vs keyword only)
        """
        self.model_name = model_name
        self.use_llm_matching = use_llm_matching

    def retrieve_results_for_query(
        self,
        query: str,
        task_list: TaskList,
        top_k: int = 1,
    ) -> List[Union[NormalTask, SequentialTask]]:
        """
        Retrieve tasks whose results match the natural language query.

        Args:
            query: Natural language query (e.g., "the company names extracted earlier")
            task_list: Task list to search
            top_k: Number of top matching tasks to return

        Returns:
            List of matching tasks, ordered by relevance
        """
        # Get completed tasks only
        completed_tasks = task_list.get_completed_tasks()

        if not completed_tasks:
            return []

        # Try keyword-based matching first (fast)
        keyword_matches = self._keyword_based_matching(query, completed_tasks)

        if keyword_matches and not self.use_llm_matching:
            return keyword_matches[:top_k]

        # If LLM matching enabled, use it for better semantic understanding
        if self.use_llm_matching and len(completed_tasks) > 0:
            llm_matches = self._llm_based_matching(query, completed_tasks, top_k)
            if llm_matches:
                return llm_matches

        # Fall back to keyword matches
        return keyword_matches[:top_k] if keyword_matches else []

    def extract_results_from_tasks(
        self,
        tasks: List[Union[NormalTask, SequentialTask]],
    ) -> List[Any]:
        """
        Extract result data from a list of tasks.

        Args:
            tasks: List of tasks

        Returns:
            List of results from the tasks
        """
        results = []

        for task in tasks:
            if isinstance(task, NormalTask):
                if task.result:
                    results.append(task.result)
            elif isinstance(task, SequentialTask):
                # For sequential tasks, return the accumulated results
                if task.results:
                    # Filter out None values (failed iterations)
                    valid_results = [r for r in task.results if r is not None]
                    if valid_results:
                        results.extend(valid_results)

        return results

    def _keyword_based_matching(
        self,
        query: str,
        tasks: List[Union[NormalTask, SequentialTask]],
    ) -> List[Union[NormalTask, SequentialTask]]:
        """
        Match tasks based on keyword overlap.

        Args:
            query: Query string
            tasks: Tasks to search

        Returns:
            List of matching tasks, ordered by keyword overlap
        """
        # Extract keywords from query
        query_keywords = self._extract_keywords(query)

        if not query_keywords:
            return []

        # Score each task
        scored_tasks = []

        for task in tasks:
            # Build searchable text from task
            searchable_text = f"{task.description} {task.type}"

            if isinstance(task, NormalTask):
                searchable_text += f" {task.instruction}"
            elif isinstance(task, SequentialTask):
                searchable_text += f" {task.goal}"

            # Extract keywords from task
            task_keywords = self._extract_keywords(searchable_text)

            # Calculate overlap
            overlap = len(query_keywords & task_keywords)

            if overlap > 0:
                # Calculate score (normalized by query keyword count)
                score = overlap / len(query_keywords)
                scored_tasks.append((task, score))

        # Sort by score descending
        scored_tasks.sort(key=lambda x: x[1], reverse=True)

        return [task for task, score in scored_tasks]

    def _llm_based_matching(
        self,
        query: str,
        tasks: List[Union[NormalTask, SequentialTask]],
        top_k: int,
    ) -> List[Union[NormalTask, SequentialTask]]:
        """
        Use LLM to match query to tasks semantically.

        Args:
            query: Query string
            tasks: Tasks to search
            top_k: Number of top matches to return

        Returns:
            List of matching tasks ordered by relevance
        """
        # Build task summaries for LLM
        task_summaries = []
        for i, task in enumerate(tasks):
            summary = {
                "index": i,
                "id": task.task_id,
                "type": task.type.value,
                "description": task.description,
            }

            if isinstance(task, NormalTask):
                summary["instruction"] = task.instruction
                summary["has_result"] = task.result is not None
            elif isinstance(task, SequentialTask):
                summary["goal"] = task.goal
                summary["result_count"] = len([r for r in task.results if r is not None])

            task_summaries.append(summary)

        # Create LLM prompt
        system_prompt = """You are a task matching assistant.

Your job is to match a natural language query to previously completed tasks.

The user will provide:
1. A query (e.g., "the company names extracted earlier")
2. A list of completed tasks with their descriptions

You should identify which task(s) best match the query based on semantic similarity.

Consider:
- What data the task collected
- The purpose of the task
- Keywords and concepts in both query and task descriptions
- Whether the task has results

Return the top matching tasks with confidence scores."""

        user_prompt = f"""Query: "{query}"

Completed Tasks:
{self._format_tasks_for_prompt(task_summaries)}

Which task(s) best match this query? Return up to {top_k} matches."""

        # Call LLM for structured output
        try:
            from pydantic import BaseModel, Field
            from typing import List

            class TaskMatches(BaseModel):
                matches: List[TaskMatchResult] = Field(
                    description=f"Top {top_k} matching tasks ordered by relevance"
                )

            matches_result: TaskMatches = generate_model(
                prompt=user_prompt,
                model_object_type=TaskMatches,
                system_prompt=system_prompt,
                model=self.model_name,
            )

            # Map matches back to tasks
            matched_tasks = []
            for match in matches_result.matches:
                for task in tasks:
                    if task.task_id == match.task_id:
                        matched_tasks.append(task)
                        break

            return matched_tasks

        except Exception as e:
            # LLM matching failed, return empty
            dprint(f"⚠️ LLM-based task matching failed: {e}")
            return []

    def _extract_keywords(self, text: str) -> set:
        """
        Extract meaningful keywords from text.

        Args:
            text: Text to extract keywords from

        Returns:
            Set of keywords (lowercased, alphanumeric)
        """
        # Lowercase
        text = text.lower()

        # Extract words (alphanumeric only)
        words = re.findall(r'\b\w+\b', text)

        # Filter out common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "should", "could", "may", "might", "can", "must", "this",
            "that", "these", "those", "it", "its", "i", "you", "he", "she", "we",
            "they", "what", "which", "who", "when", "where", "why", "how"
        }

        # Return non-stop-words
        return {word for word in words if word not in stop_words and len(word) > 2}

    def _format_tasks_for_prompt(self, task_summaries: List[Dict[str, Any]]) -> str:
        """
        Format task summaries for LLM prompt.

        Args:
            task_summaries: List of task summary dicts

        Returns:
            Formatted string
        """
        lines = []

        for summary in task_summaries:
            task_type = summary["type"]
            desc = summary["description"]
            task_id = summary["id"]

            if task_type == "normal":
                has_result = "✓" if summary.get("has_result") else "✗"
                lines.append(f"- Task {summary['index']} ({task_id}): {desc}")
                lines.append(f"  Type: Normal Task")
                lines.append(f"  Instruction: {summary.get('instruction', 'N/A')}")
                lines.append(f"  Has Result: {has_result}")

            elif task_type == "sequential":
                result_count = summary.get("result_count", 0)
                lines.append(f"- Task {summary['index']} ({task_id}): {desc}")
                lines.append(f"  Type: Sequential Task")
                lines.append(f"  Goal: {summary.get('goal', 'N/A')}")
                lines.append(f"  Results Collected: {result_count}")

            lines.append("")  # Blank line between tasks

        return "\n".join(lines)


class TaskResultAccessor:
    """
    High-level accessor for task results with automatic retrieval.
    """

    def __init__(
        self,
        task_list: TaskList,
        retriever: Optional[TaskResultRetriever] = None,
    ):
        """
        Initialize the result accessor.

        Args:
            task_list: Task list to access
            retriever: Result retriever (creates default if None)
        """
        self.task_list = task_list
        self.retriever = retriever or TaskResultRetriever()

    def get_results(
        self,
        query: str,
        return_first_only: bool = True,
    ) -> Optional[Any]:
        """
        Get results matching a natural language query.

        Args:
            query: Natural language query
            return_first_only: If True, return only the first match's results

        Returns:
            Results from matching task(s), or None if no match
        """
        # Retrieve matching tasks
        matching_tasks = self.retriever.retrieve_results_for_query(
            query=query,
            task_list=self.task_list,
            top_k=1 if return_first_only else 5,
        )

        if not matching_tasks:
            return None

        # Extract results
        results = self.retriever.extract_results_from_tasks(matching_tasks)

        if not results:
            return None

        if return_first_only:
            return results[0] if results else None

        return results

    def get_all_results(self) -> List[Any]:
        """
        Get all results from all completed tasks.

        Returns:
            List of all results
        """
        completed_tasks = self.task_list.get_completed_tasks()
        return self.retriever.extract_results_from_tasks(completed_tasks)
