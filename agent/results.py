"""
Agent Results - Consolidated result types for mission, task, and turn execution.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union

from models.models import TaskCompletionStatus


@dataclass
class TaskResult:
    """Result of a single task execution"""
    success: bool
    reasoning: str = ""
    confidence: float = 0.0
    evidence: Optional[Dict[str, Any]] = None
    completion_status: Optional[TaskCompletionStatus] = None
    progress: int = 0
    target: Union[int, str] = 1
    history: Optional[List[str]] = None

    @property
    def status(self) -> str:
        """Status string for compatibility"""
        return "achieved" if self.success else "failed"


@dataclass
class MissionResult:
    success: bool
    task_results: List[TaskResult]
    reasoning: str = ""

    def __init__(self):
        self.success = False
        self.task_results = []
        self.reasoning = ""
