"""
Agent Results - Consolidated result types for mission, task, and turn execution.

This module contains all result types used throughout the agent system:
- TaskResult: Result of a single task execution
- MissionResult: Result of a complete mission (returned to user)
- TurnDecision: Decision made at each agent turn
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


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
    success: bool
    task_results: List[TaskResult]
    reasoning: str = ""

    def __init__(self):
        """
        Initialize MissionResult from TaskResult.

        Args:
            task_result: The TaskResult from agent execution
            extracted_data: Optional dictionary of extracted data
            orchestration: Optional orchestration metadata
        """
        self.success = False
        self.task_results = []
        self.reasoning = ""