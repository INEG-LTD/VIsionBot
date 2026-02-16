"""
Agent Results - Consolidated result types for mission, planning iteration, and task execution.
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

    @property
    def status(self) -> str:
        """Status string for compatibility"""
        return "achieved" if self.success else "failed"


@dataclass
class MissionResult:
    success: bool
    task_results: List[TaskResult]
    reasoning: str = ""
    partial: bool = False
    final_answer_draft: str = ""

    def __init__(self):
        self.success = False
        self.task_results = []
        self.reasoning = ""
        self.partial = False
        self.final_answer_draft = ""


# Backward-compatibility alias for legacy imports at package root.
AgentResult = MissionResult
