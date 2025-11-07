# Agent System Overview

This document consolidates the implementation details for the BrowserVision agent, covering how the major components collaborate to execute complex, multi-tab automation workflows.

## 1. Top-Level Architecture

- **`vision_bot.py`** orchestrates the end-to-end automation flow: launching or attaching to Playwright, handling `act()` goals, and coordinating `agentic_mode()` runs.
- **`agent/agent_controller.py`** implements the Step 1 reactive agent loop. Each iteration captures browser state, checks completion, gathers history, and determines the next command to run.
- **`action_executor.py`** executes low-level interactions (click, type, press, scroll) with fallbacks, retry logic, and DOM-change tracking.
- **`goals/`** contains goal evaluators (ClickGoal, TypeGoal, ExtractGoal, etc.) that determine whether a requested action has succeeded based on the `GoalContext`.
- **`ai_utils.py`** centralizes LLM + vision calls for planning, element analysis, extraction, and completion checks.

### Agentic Mode Flow

1. Capture environment snapshot (`BrowserState`).
2. `CompletionContract` evaluates whether the overall task is already complete.
3. `ReactiveGoalDeterminer` inspects the current screenshot and interaction history to pick the immediate next command.
4. If the command references a known failure/ineffective action, it is rejected and a new decision is requested.
5. Execute the command (via `act()` or direct extraction) and record the outcome.
6. Track failed/ineffective actions, extracted data, orchestration events, and retry limits.
7. Repeat until the goal is achieved, a defer command is issued, or iteration limits are reached.

## 2. Action Resolution & Execution

- **Target resolution**: `_resolve_click_target()` blends DOM queries with optional vision hints to locate the correct element, prioritizing context-provided selectors and recorded history (`action_executor.py`).
- **Click strategy**: `_execute_click()` first attempts a precise mouse click. If the same selector is retried on an unchanged DOM, it automatically switches to a programmatic click.
- **Keyboard input**: `_parse_and_press_keys()` normalizes comma-separated input (`"arrow_down,enter"` → sequential presses) and standardizes arrow key names.
- **Failed vs. ineffective actions**: `AgentController` tracks both (command failed vs. command succeeded but page unchanged). These lists feed back into the prompts so the LLM avoids repeating them.
- **Context safety**: `ContextGuard`, `FocusManager`, and handler utilities expose `set_page()` methods to keep every component aligned when tabs change.

## 3. Reactive Decision Making

- `agent/reactive_goal_determiner.py` builds a detailed prompt emphasizing:
  - **Extraction priority** (`extract:` commands whenever the user wants data).
  - **Command catalogue** including `click`, `type`, `press`, `scroll`, `select`, `form`, `upload`, and the newly documented `defer` handoff.
  - **Element specificity**: click/type commands must name the element type.
  - **Base knowledge**: user-supplied rules injected into the system and action prompts.
  - **Failed/ineffective history**: clearly enumerated so the LLM steers away from repeats.

### Base Knowledge Propagation

1. `agentic_mode(base_knowledge=...)` passes the rules into `AgentController`.
2. `GoalMonitor` copies the rules into every `GoalContext`.
3. Goal evaluators (ClickGoal, FormGoal, NavigationGoal, etc.) include the base knowledge block in their own LLM prompts so evaluation criteria align with the customized behavior.

## 4. Extraction Pipeline

- `BrowserVisionBot.extract()` captures screenshots (viewport, full-page, or element scope) and uses `generate_text` or `generate_model` to retrieve text, JSON, or structured outputs (`Pydantic`).
- Confidence gating and retry logic protect against low-quality responses.
- Interaction history records each extraction (`InteractionType.EXTRACT`) including prompt, success, data, and errors.
- `extract_batch()` iterates a list of prompts; `extract_multi_field()` builds field-specific requests from a common description.
- `AgentController` detects `extract:` commands (or natural-language extraction hints) and routes them through `bot.extract()`, storing aggregated results in `AgentResult.extracted_data` and task evidence.
- The `CompletionContract` explicitly checks that extraction goals are satisfied before declaring success.

## 5. Tab Management & Decisions

### Tab Manager (`tab_management/tab_manager.py`)

- Registers Playwright `Page` objects with unique `TabInfo` records (URL, title, purpose, timestamps, metadata).
- Tracks the active tab, supports `switch_to_tab`, `close_tab`, metadata updates, and orphan cleanup.
- Integrates with `BrowserVisionBot` via `start()`, `_attach_new_page_listener()`, and `switch_to_page()` to keep every component’s page reference synchronized and caches invalidated.
- Maintains chronological URL history and a pointer for the current position so the agent can issue `back:`/`forward:` commands without reconstructing URLs manually.

### Tab Decision Engine (`tab_management/tab_decision_engine.py`)

- Builds an LLM prompt summarizing the current tab, all other tabs, and the user goal.
- Returns a structured `TabDecision` (`action`, `target_tab_id`, `reasoning`, `confidence`, `should_take_action`).
- Possible actions: `SWITCH`, `CLOSE`, `CONTINUE`, `SPAWN_SUB_AGENT`.
- `AgentController` checks for decisions each iteration; `_execute_tab_decision()` runs them (bringing tabs to front, closing, or invoking sub-agent logic).

## 6. Sub-Agent Infrastructure

- **`AgentContext`** (`agent/agent_context.py`) tracks agent identity, parent-child links, status, result payloads, timestamps, metadata, and exposes helper constructors (`create_main_agent`, `create_sub_agent`).
- **`SubAgentController`** manages the lifecycle of child agents:
  - `spawn_sub_agent()` verifies tab availability, updates tab metadata, and instantiates a child `AgentController`.
  - `execute_sub_agent()` switches to the child tab, runs `agentic_mode`, records success/failure, and switches back.
  - Supports listing, querying, result retrieval, execution history, and cleanup of completed agents.
- **Propagation**: Sub-agents inherit base knowledge and share the same Bot instance, but operate on isolated tabs.

## 7. Orchestration & Results

- The main `AgentController` logs orchestration events (agent start/end, tab decisions, sub-agent lifecycle) into `self.orchestration_events`.
- Completed sub-agent outcomes are drained via `sub_agent_controller.pop_completed_results()` and merged into `self.sub_agent_results`.
- `_build_evidence()` compiles a comprehensive report including extracted data, sub-agent summaries, tab snapshots, and orchestration events.
- `AgentResult` wraps the final `GoalResult`, exposing `success`, `confidence`, `reasoning`, aggregated `extracted_data`, `sub_agent_results`, and `orchestration` details.

## 8. Supporting Infrastructure

- **Deduplication**: `interaction_deduper` tracks previous interactions; dedup commands (enable/disable) can be triggered via `act()`.
- **Focus & Vision Utilities**: `FocusManager`, `ElementAnalyzer`, `vision_resolver`, and `overlay_manager` collaborate to describe elements, maintain focus contexts, and provide consistent vision hints.
- **Handlers**: Specialized modules for selects, uploads, and datetime pickers expose `set_page()` interfaces to remain in sync after tab switches.
- **Documentation**: `TAB_DETECTION.md`, `ACTION_QUEUE.md`, `COMMAND_LEDGER.md`, and related files provide deep dives into ancillary systems like new-tab detection, deferred execution, and command auditing.

## 9. Key Interaction Patterns

- **Defer command**: When the user requests manual control (captcha, MFA, human decision), the reactive prompt now advertises `defer:` commands. Runtime support exists in `vision_bot.py` (`DeferGoal` & `TimedSleepGoal`).
- **Plan truncation**: In agent mode, plan generation is limited to a single step and auto-scroll is disabled to keep the loop reactive.
- **State resets**: Cache invalidation (`_cached_screenshot_with_overlays`, `_cached_dom_signature`, etc.) occurs whenever DOM signature changes or tabs switch, preventing stale context.
- **Fallbacks**: Navigation commands are converted to clicks, and pressing keys defaults to brief commands (e.g., `press: Enter`).
- **History-aware navigation**: The reactive prompt summarizes recent URLs and highlights previous/next pages so the agent can choose `back:` or `forward:` when returning to a prior step is faster than searching again.

## 10. Extending the System

To add new capabilities:

1. **New command**: Update `ReactiveGoalDeterminer` prompts, ensure `AgentController` understands how to execute it, and extend goal evaluation if needed.
2. **New goal logic**: Implement a `Goal` subclass under `goals/`, register it with `vision_bot.py` goal factory, and propagate base knowledge/evidence requirements.
3. **Additional orchestration**: Hook into `AgentController`’s orchestration logging to track custom lifecycle events or data flows.
4. **Custom extraction schemas**: Pass bespoke Pydantic models into `bot.extract(..., output_format="structured", model_schema=MyModel)`; results integrate automatically into evidence and `AgentResult`.

## 11. References

- Core modules:
  - `vision_bot.py`
  - `agent/agent_controller.py`
  - `agent/reactive_goal_determiner.py`
  - `action_executor.py`
  - `tab_management/tab_manager.py`
  - `tab_management/tab_decision_engine.py`
  - `agent/sub_agent_controller.py`
  - `goals/`
- Key documentation: `TAB_DETECTION.md`, `ACTION_QUEUE.md`, `COMMAND_LEDGER.md`, `ADVANCED_LEDGER_FEATURES.md`, `POST_ACTION_HOOKS.md`, `IF_GOAL_VISION.md`, `SCROLL_TRACKING.md`.

This single reference replaces the phase-by-phase implementation documents and should serve as the canonical description of how the BrowserVision agent operates today.


