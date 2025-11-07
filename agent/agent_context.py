"""
AgentContext - Context information passed between agents.

Phase 3: Sub-Agent Infrastructure
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import time
import uuid


@dataclass
class AgentContext:
    """
    Context information for an agent, including parent-child relationships.
    
    Attributes:
        agent_id: Unique identifier for this agent
        parent_agent_id: ID of parent agent (None for main agent)
        tab_id: Tab ID this agent is working on
        instruction: Task/instruction for this agent
        created_at: Timestamp when agent was created
        status: Current status (pending, running, completed, failed)
        result: Result from agent execution (if completed)
        metadata: Additional context/metadata
    """
    agent_id: str
    tab_id: str
    instruction: str
    parent_agent_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and initialize agent context"""
        if not self.agent_id:
            raise ValueError("agent_id is required")
        if not self.tab_id:
            raise ValueError("tab_id is required")
        if not self.instruction:
            raise ValueError("instruction is required")
        if self.created_at == 0:
            self.created_at = time.time()
    
    def mark_running(self) -> None:
        """Mark agent as running"""
        self.status = "running"
    
    def mark_completed(self, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark agent as completed with optional result"""
        self.status = "completed"
        self.result = result
    
    def mark_failed(self, error: Optional[str] = None) -> None:
        """Mark agent as failed with optional error message"""
        self.status = "failed"
        self.result = {"error": error} if error else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_id": self.agent_id,
            "tab_id": self.tab_id,
            "instruction": self.instruction,
            "parent_agent_id": self.parent_agent_id,
            "created_at": self.created_at,
            "status": self.status,
            "result": self.result,
            "metadata": self.metadata
        }
    
    @classmethod
    def create_main_agent(cls, tab_id: str, instruction: str) -> "AgentContext":
        """Create context for main agent"""
        return cls(
            agent_id=f"main_{uuid.uuid4().hex[:12]}",
            tab_id=tab_id,
            instruction=instruction,
            parent_agent_id=None
        )
    
    @classmethod
    def create_sub_agent(
        cls,
        parent_agent_id: str,
        tab_id: str,
        instruction: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "AgentContext":
        """Create context for sub-agent"""
        return cls(
            agent_id=f"sub_{uuid.uuid4().hex[:12]}",
            tab_id=tab_id,
            instruction=instruction,
            parent_agent_id=parent_agent_id,
            metadata=metadata or {}
        )

