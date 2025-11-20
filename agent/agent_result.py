"""
Agent Result - Structured return type for agentic mode execution.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from .task_result import TaskResult


@dataclass
class AgentResult:
    """
    Result object returned from agentic mode execution.
    
    Contains:
    - success: Whether the task completed successfully
    - extracted_data: Dictionary of extracted data (key: extraction prompt, value: extracted result)
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
        Initialize AgentResult from TaskResult.
        
        Args:
            task_result: The TaskResult from agent execution
            extracted_data: Optional dictionary of extracted data
        """
        self.success = task_result.success
        self.extracted_data = extracted_data or {}
        self.sub_agent_results = sub_agent_results or []
        self.orchestration = orchestration or {}
        self.reasoning = task_result.reasoning
        self.confidence = task_result.confidence
        self.task_result = task_result
        
        # Also extract any extracted_data from task_result evidence if present
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


