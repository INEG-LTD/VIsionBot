"""
Agent Iteration Decision - Single LLM call for completion, sub-agent policy, and next action.

This combines three related decisions into one efficient call made at each agent iteration:
1. Is the task complete? (yes/no)
2. Should sub-agents be used? (yes/no)  
3. What's the next action? (if not complete)
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

from agent.reactive_goal_determiner import ActionPlan


class AgentIterationDecision(BaseModel):
    """Decision made at each agent iteration: completion status, sub-agent policy, and next action"""
    model_config = ConfigDict(extra="forbid")
    
    # Completion check
    is_complete: bool = Field(description="True if the task is complete, False otherwise")
    
    # Sub-agent policy check
    needs_sub_agents: bool = Field(description="True if sub-agents should be used, False otherwise")
    
    # Next action (only needed if task is not complete)
    next_action: Optional[ActionPlan] = Field(
        default=None,
        description="The next action to take. Only required if is_complete is False. If is_complete is True, this can be None."
    )
