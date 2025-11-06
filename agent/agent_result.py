"""
Agent Result - Structured return type for agentic mode execution.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from goals.base import GoalResult, GoalStatus


@dataclass
class AgentResult:
    """
    Result object returned from agentic mode execution.
    
    Contains:
    - success: Whether the task completed successfully
    - extracted_data: Dictionary of extracted data (key: extraction prompt, value: extracted result)
    - reasoning: Explanation of the result
    - confidence: Confidence score (0.0-1.0)
    - goal_result: Original GoalResult for advanced access
    """
    success: bool
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.0
    goal_result: Optional[GoalResult] = None
    
    def __init__(self, goal_result: GoalResult, extracted_data: Optional[Dict[str, Any]] = None):
        """
        Initialize AgentResult from GoalResult.
        
        Args:
            goal_result: The GoalResult from agent execution
            extracted_data: Optional dictionary of extracted data
        """
        self.success = goal_result.status == GoalStatus.ACHIEVED
        self.extracted_data = extracted_data or {}
        self.reasoning = goal_result.reasoning
        self.confidence = goal_result.confidence
        self.goal_result = goal_result
        
        # Also extract any extracted_data from goal_result evidence if present
        if goal_result.evidence and "extracted_data" in goal_result.evidence:
            self.extracted_data.update(goal_result.evidence["extracted_data"])
    
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


