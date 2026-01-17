"""
Agent Results - Consolidated result types for mission, task, and turn execution.

This module contains all result types used throughout the agent system:
- TaskResult: Result of a single task execution
- MissionResult: Result of a complete mission (returned to user)
- SubAgentResult: Result of a sub-agent execution
- TurnDecision: Decision made at each agent turn
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
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
    - sub_agent_results: Results from any sub-agents
    - orchestration: Task orchestration metadata
    - reasoning: Explanation of the result
    - confidence: Confidence score (0.0-1.0)
    - task_result: Original TaskResult for advanced access
    """
    success: bool
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    sub_agent_results: List[Dict[str, Any]] = field(default_factory=list)
    orchestration: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.0
    task_result: Optional[TaskResult] = None

    def __init__(
        self,
        task_result: TaskResult,
        extracted_data: Optional[Dict[str, Any]] = None,
        sub_agent_results: Optional[List[Dict[str, Any]]] = None,
        orchestration: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize MissionResult from TaskResult.

        Args:
            task_result: The TaskResult from agent execution
            extracted_data: Optional dictionary of extracted data
            sub_agent_results: Optional list of sub-agent results
            orchestration: Optional orchestration metadata
        """
        self.success = task_result.success
        self.extracted_data = extracted_data or {}
        self.sub_agent_results = sub_agent_results or []
        self.orchestration = orchestration or {}
        self.reasoning = task_result.reasoning
        self.confidence = task_result.confidence
        self.task_result = task_result

        if task_result.evidence:
            evidence = task_result.evidence
            if "extracted_data" in evidence:
                self.extracted_data.update(evidence["extracted_data"])
            if "sub_agents" in evidence and not self.sub_agent_results:
                self.sub_agent_results = evidence["sub_agents"]
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


@dataclass
class SubAgentResult:
    """Result of a sub-agent execution"""
    agent_id: str
    tab_id: str
    instruction: str
    success: bool
    status: str
    confidence: float
    reasoning: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: float = field(default_factory=lambda: 0.0)
    completed_at: float = field(default_factory=lambda: 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["duration"] = max(0.0, self.completed_at - self.started_at)
        return data


class TurnDecision(BaseModel):
    """
    Decision made at each agent turn: completion status, sub-agent policy, and next action.

    This combines three related decisions into one efficient call:
    1. Is the task complete? (yes/no)
    2. Should sub-agents be used? (yes/no)
    3. What's the next action? (if not complete)
    """
    model_config = ConfigDict(extra="forbid")

    is_complete: bool = Field(description="True if the task is complete, False otherwise")
    needs_sub_agents: bool = Field(description="True if sub-agents should be used, False otherwise")
    next_action: Optional[Any] = Field(
        default=None,
        description="The next action to take. Only required if is_complete is False."
    )
