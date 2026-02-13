# Project Memory

This file stores durable collaboration context for future Codex sessions in this repo.

## Thread Memory (2026-02-13)

- Product direction: build a general-purpose browser-use agent.
- Constraint: proposed solutions should be agent-native, not page-specific heuristics.
- Constraint for this iteration: prompt-only changes (no controller/executor logic changes).
- Goal discussed: reduce repetitive `think` restatements and carry strategy continuity between iterations.
- Implemented change: added `ACTIVE STRATEGY CONTINUATION MODE` prompt instructions in `agent/action_planner.py`.
- Prompt behavior added:
  - when `active_strategy` exists, continue current strategy instead of inventing a new one;
  - `think(next_action=continue)` should be delta-only;
  - prefer `mark_progress` when unit of work is complete;
  - use `stuck` rather than inventing a new strategy when there is no useful delta.

