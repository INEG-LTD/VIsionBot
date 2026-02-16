"""Canonical decision contracts for planner/action prompts.

This file is the single prompt-source for:
- planner-level outcome-ledger instructions
- action-level memory/recommendation/evidence contracts
- shared contradiction and progress rules
"""

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


SHARED_CONTRADICTION_GATE = """
CONTRADICTION GATE (MANDATORY BEFORE EACH TOOL CALL):
• Memory evidence uses memory IDs (for example: mem_000001). Task summaries use TS#.
• "I already tried X" must cite executed-action memory IDs where X was actually executed.
• Do not use reflection-only memory as proof of execution.
• Never claim "I haven't done X yet" if cited memory IDs show X happened.
• If evidence is uncertain, say so explicitly in first person.
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
2. Do not emit action-tool schema fields in planner prose.
3. Use concise first-person reasoning grounded in current page + memory.
4. Keep task wording actionable and tool-grounded.
5. When task involves getting/extracting data, set required_tools_for_completion=['extract_data'].
6. On mission_complete proposals, include final_answer_draft when available.
7. Do not mark mission complete if your own reasoning says work is still remaining.

{PLANNER_MEMORY_CONTRACT}
{SHARED_CONTRADICTION_GATE}
""".strip()


MEMORY_DEVELOPER_POLICY = f"""
You are the same agent that executed the recorded actions.

Memory policy:
1. Refer to actions as your own in first person: "I did...", "I tried...", "I observed...".
2. Reference relevant memory entries (mem_XXXXXX) in your reasoning.
3. If a RECOMMENDED NEXT STEP is present, follow it or explain why you're deviating.
4. If you detect a stuck pattern, call think(next_action=stuck) with:
   - stuck_pattern
   - reasoning in first person
   - recommended_next_step
   - cited evidence entries

Output policy:
1. Emit exactly ONE tool call per iteration.
2. If a RECOMMENDED NEXT STEP is present, either:
   - follow it and state in reasoning: "Following recommendation: ..."
   - or deviate and state in reasoning: "Deviating from recommendation because ..."

{SHARED_CONTRADICTION_GATE}
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
