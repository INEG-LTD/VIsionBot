"""
Agent Results - Result types for mission execution.
"""
from dataclasses import dataclass


@dataclass
class MissionResult:
    success: bool = False
    reasoning: str = ""
    narrative: str = ""
    partial: bool = False
    final_answer_draft: str = ""
    total_iterations: int = 0
    total_actions: int = 0
    final_url: str = ""
    duration_s: float = 0.0
    total_cost_usd: float = 0.0
    budget_total: int = 0
    budget_spent: int = 0
    budget_remaining: int = 0
    budget_phase: str = "normal"

    def __bool__(self) -> bool:
        return self.success


# Backward-compatibility alias for legacy imports at package root.
AgentResult = MissionResult
