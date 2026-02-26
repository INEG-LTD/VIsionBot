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
    mission_ms: float = 0.0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    image_count: int = 0
    retry_count: int = 0
    failure_code: str = ""
    failure_stage: str = ""
    avg_iteration_ms: float = 0.0
    p95_iteration_ms: float = 0.0
    avg_llm_ms: float = 0.0
    avg_tool_ms: float = 0.0
    avg_tokens_in: float = 0.0
    avg_tokens_out: float = 0.0
    avg_images_per_call: float = 0.0
    retries_per_mission: float = 0.0

    def __bool__(self) -> bool:
        return self.success


# Backward-compatibility alias for legacy imports at package root.
AgentResult = MissionResult
