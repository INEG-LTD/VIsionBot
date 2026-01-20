"""
Agent Results - Consolidated result types for mission, task, and turn execution.

This module contains all result types used throughout the agent system:
- TaskResult: Result of a single task execution
- MissionResult: Result of a complete mission (returned to user)
- TurnDecision: Decision made at each agent turn
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


@dataclass
class TaskResult:
    """Result of a single task execution"""
    success: bool
    reasoning: str = ""
    confidence: float = 0.0
    evidence: Optional[Dict[str, Any]] = None

    @property
    def status(self) -> str:
        """Status string for compatibility"""
        return "achieved" if self.success else "failed"


@dataclass
class MissionResult:
    """
    Result object returned from mission execution (execute_mission).

    Contains:
    - success: Whether the mission completed successfully
    - extracted_data: Dictionary of extracted data (key: extraction prompt, value: extracted result)
    - orchestration: Task orchestration metadata
    - reasoning: Explanation of the result
    - confidence: Confidence score (0.0-1.0)
    - task_result: Original TaskResult for advanced access
    """
    success: bool
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    orchestration: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.0
    task_result: Optional[TaskResult] = None

    def __init__(
        self,
        task_result: TaskResult,
        extracted_data: Optional[Dict[str, Any]] = None,
        orchestration: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize MissionResult from TaskResult.

        Args:
            task_result: The TaskResult from agent execution
            extracted_data: Optional dictionary of extracted data
            orchestration: Optional orchestration metadata
        """
        self.success = task_result.success
        self.extracted_data = extracted_data or {}
        self.orchestration = orchestration or {}
        self.reasoning = task_result.reasoning
        self.confidence = task_result.confidence
        self.task_result = task_result

        if task_result.evidence:
            evidence = task_result.evidence
            if "extracted_data" in evidence:
                self.extracted_data.update(evidence["extracted_data"])
            if "orchestration" in evidence and not self.orchestration:
                self.orchestration = evidence["orchestration"]

    def get(self, key: str, default: Any = None) -> Any:
        """Get extracted data by key, with optional default"""
        return self.extracted_data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access: result['product price']"""
        return self.extracted_data[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists: 'product price' in result"""
        return key in self.extracted_data

    def keys(self):
        """Get all extraction prompt keys"""
        return self.extracted_data.keys()

    def values(self):
        """Get all extracted values"""
        return self.extracted_data.values()

    def items(self):
        """Get all (key, value) pairs"""
        return self.extracted_data.items()


class TurnDecision(BaseModel):
    """
    Decision made at each agent turn: completion status and next action.

    This combines two related decisions into one efficient call:
    1. Is the task complete? (yes/no)
    2. What's the next action? (if not complete)
    """
    model_config = ConfigDict(extra="forbid")

    is_complete: bool = Field(description="True if the task is complete, False otherwise")
    next_action: Optional[Any] = Field(
        default=None,
        description="The next action to take. Only required if is_complete is False."
    )
