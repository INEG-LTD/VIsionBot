"""
Task Result - Simple result type for task execution (replaces GoalResult).

This provides a simple result structure without goal evaluation logic.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


@dataclass
class TaskResult:
    """Simple result type for task execution"""
    success: bool
    reasoning: str = ""
    confidence: float = 0.0
    evidence: Optional[Dict[str, Any]] = None
    
    @property
    def status(self) -> str:
        """Status string for compatibility"""
        return "achieved" if self.success else "failed"

