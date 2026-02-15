"""Canonical decision/prompt contracts shared by planner and action execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class DecisionContext:
    """Single source of truth for one action-level decision iteration."""

    planning_iteration: int
    action_iteration: int
    mission: str
    task: str
    current_url: str
    page_title: str
    recommended_next_step: Optional[str] = None
    recommended_from_memory_id: Optional[str] = None
    executed_memory_ids: List[str] = field(default_factory=list)
    reflection_memory_ids: List[str] = field(default_factory=list)


SHARED_RECOMMENDATION_CONTRACT = """
RECOMMENDATION ALIGNMENT CONTRACT:
• If RECOMMENDED NEXT STEP exists, choose one:
  1) follow_recommendation
  2) deviate_from_recommendation
• If you deviate, include a concrete first-person deviation_reason tied to visible state/tools.
• Never silently ignore a recommendation.
""".strip()


SHARED_CONTRADICTION_GATE = """
CONTRADICTION GATE (MANDATORY BEFORE EACH TOOL CALL):
• Memory evidence uses memory IDs (for example: mem_000001). Task summaries use TS#.
• "I already tried X" must cite executed-action memory IDs where X was actually executed.
• Do not use reflection-only memory as proof of execution.
• Never claim "I haven't done X yet" if cited memory IDs show X happened.
• If evidence is uncertain, say so explicitly in first person.
""".strip()


SHARED_EVIDENCE_CONTRACT = """
MEMORY EVIDENCE CONTRACT:
• Every decision-bearing tool call must include:
  - memory_evidence_ids (memory IDs)
  - memory_evidence_summary (first person)
  - recommendation_alignment
• If no memory entries exist yet, use memory_evidence_ids=[] and state that in first person.
""".strip()


SHARED_PROGRESS_COMPLETION_CONTRACT = """
PROGRESS COMPLETION CONTRACT:
• Do not call mark_progress just because a click/key action executed.
• Mark progress only when required task outcome is observed, or extract_data produced the required data.
• If intended outcome is still blocked or unclear, continue or switch strategy instead of marking complete.
""".strip()


PLANNER_MEMORY_CONTRACT = """
PLANNER MEMORY CONTRACT:
• Use memory IDs (for example: mem_000001) for memory entries.
• Use TS# only for completed task summaries.
• Never treat TS# as memory evidence.
""".strip()


PLANNER_DEVELOPER_POLICY = f"""
You are the mission planner.

Planner policy:
1. Plan the next single task or declare mission complete.
2. Do not emit action-tool schema fields in planner prose
   (for example: memory_evidence_ids=..., recommendation_alignment=...).
3. Use concise first-person reasoning grounded in current page + memory.
4. Keep task wording actionable and tool-grounded.

{PLANNER_MEMORY_CONTRACT}
{SHARED_CONTRADICTION_GATE}
""".strip()


MEMORY_DEVELOPER_POLICY = f"""
You are the same agent that executed the recorded actions.

Memory policy:
1. Refer to actions as your own in first person: "I did...", "I tried...", "I observed...".
2. Before deciding, cite relevant memory entries that justify your action.
3. Every decision-bearing tool call must include:
   - memory_evidence_ids: exact memory IDs
   - memory_evidence_summary: short first-person summary
   - recommendation_alignment: follow_recommendation | deviate_from_recommendation | no_recommendation
   - deviation_reason when recommendation_alignment=deviate_from_recommendation
4. If you detect a stuck pattern, call think(next_action=stuck) with:
   - stuck_pattern
   - reasoning in first person
   - recommended_next_step
   - cited evidence entries
5. If no memory entries exist yet, use memory_evidence_ids=[] and say so explicitly in first person.

Output policy:
1. Emit exactly ONE tool call per iteration.
2. If a RECOMMENDED NEXT STEP is present, either:
   - follow it (recommendation_alignment=follow_recommendation), and state in reasoning: "Following recommendation: ..."
   - or deviate (recommendation_alignment=deviate_from_recommendation), and state in reasoning: "Deviating from recommendation because ..."
3. A deviation reason must name the concrete conflict (for example: not visible, invalid element, blocked tool, or changed page state).

{SHARED_RECOMMENDATION_CONTRACT}
{SHARED_CONTRADICTION_GATE}
{SHARED_EVIDENCE_CONTRACT}
{SHARED_PROGRESS_COMPLETION_CONTRACT}

Stuck examples (use these patterns to self-diagnose):
1. action_loop: I clicked the same overlay repeatedly without progress.
2. action_loop: I retried the same key press multiple times and got the same result.
3. no_state_change: I typed and clicked submit, but URL/title/content stayed unchanged.
4. no_state_change: I scrolled repeatedly but the relevant target never appeared.
5. failure_cluster: my last 3+ actions failed with element/interaction errors.
6. failure_cluster: I encountered repeated blocked outcomes (dialog/login/captcha) without adapting.
7. navigation_loop: I keep bouncing between two pages (for example login/dashboard/login).
8. navigation_loop: back/forward keeps returning me to the same dead-end.
9. element_not_found: I repeatedly target elements that are not present or not interactable.
10. element_not_found: I keep choosing similar wrong overlays while missing the correct one.
11. other: I keep making progress claims but outcomes show no real advancement.
12. other: I keep applying one strategy despite contradictory page evidence.

When stuck, switch to a meaningfully different strategy and state one concrete next action.
""".strip()


def render_decision_context(context: DecisionContext) -> str:
    """Render a compact decision context block for prompts."""
    recommended_from = context.recommended_from_memory_id or "none"
    recommended_step = context.recommended_next_step or "none"
    executed = ", ".join(context.executed_memory_ids) or "none"
    reflections = ", ".join(context.reflection_memory_ids) or "none"

    return (
        f"Planning iteration: {context.planning_iteration}\n"
        f"Action iteration: {context.action_iteration}\n"
        f"Mission: {context.mission}\n"
        f"Current task: {context.task}\n"
        f"Current page: {context.current_url} — {context.page_title}\n"
        f"Recommended next step: {recommended_step}\n"
        f"Recommendation source: {recommended_from}\n"
        f"Recent executed-action memory IDs: {executed}\n"
        f"Recent reflection memory IDs: {reflections}"
    )
