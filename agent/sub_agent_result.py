"""
Sub-agent execution result data structures.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import time


@dataclass
class SubAgentResult:
    """
    Captures the outcome of a sub-agent execution.
    """
    agent_id: str
    tab_id: str
    instruction: str
    success: bool
    status: str
    confidence: float
    reasoning: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    completed_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["duration"] = max(0.0, self.completed_at - self.started_at)
        return data

