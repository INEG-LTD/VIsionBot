"""
Agent Results - Result types for mission execution.
"""
from dataclasses import dataclass


@dataclass
class MissionResult:
    success: bool = False
    reasoning: str = ""
    partial: bool = False
    final_answer_draft: str = ""


# Backward-compatibility alias for legacy imports at package root.
AgentResult = MissionResult
