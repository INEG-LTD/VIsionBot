"""
Action Planner - Determines the next viewport-aware plan based on current state.

Step 2: LLM-based action planning that builds a sequence of actions you can execute
before the viewport changes, relying only on what is visible.
"""

from typing import Optional, List, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import re
import time
import json

from agent.memory import NarrativeMemory
from agent.notebook import Notebook
from agent.agent_context import EnvironmentState
from agent.prompts import (
    DecisionContext,
    SHARED_CONTRADICTION_GATE,
    get_memory_developer_policy,
    render_decision_context,
)
from lib.ai import (
    generate_action_with_tools,
    ReasoningLevel,
    get_default_agent_model,
    get_default_agent_reasoning_level,
)
from models.models import ActionPlan, ActionStep, FailedAction, NotebookEntryType, PageElements
from agent.speculative_hints import HintBundle, HintCandidate, HintTargetSignature
from utils.event_logger import get_event_logger

# Compact hint envelope key map (used in next_hint_json):
# - s: status ("c" candidate | "n" none)
# - f: function_name
# - a: function_arguments
# - oi: overlay_index
# - c: confidence
# - r: reason
# - id: candidate_id (optional)
_NEXT_HINT_COMPACT_KEYS: Dict[str, str] = {
    "s": "status",
    "f": "function_name",
    "a": "function_arguments",
    "oi": "overlay_index",
    "c": "confidence",
    "r": "reason",
    "id": "candidate_id",
}

_NOTEBOOK_ENTRY_DESCRIPTION_MAX_CHARS = 100
_NOTEBOOK_ENTRY_DATA_MAX_CHARS = 240


def strip_targeting_data(step: str) -> str:
    """Remove element_ids from a recommended step string.

    The LLM should only see the action intent (e.g. 'click: Martin bot avatar')
    and must determine fresh targeting from the current element index.
    Stale IDs cause the LLM to copy them verbatim instead of looking.
    """
    # Remove element_id patterns: "[id=123]" or "element_id=123"
    step = re.sub(r'\s*\[id=\d+\]', '', step)
    step = re.sub(r'\s*element_id\s*=\s*\d+', '', step)
    # Clean up trailing whitespace / colons
    step = re.sub(r'\s*:\s*$', '', step).strip()
    return step or "continue with the recommended action"


class ActionPlanner:
    """
    Determines a viewport-safe action plan based on:
    - Current viewport (what's visible on screen)
    - Environment state (interactions, scroll position, etc.)
    - User prompt (what we're trying to achieve)

    This is reactive - it determines an ordered list of actions that can be executed
    without waiting for new UI elements to appear.
    """
    
    def __init__(
        self,
        user_prompt: str,
        memory_store: NarrativeMemory,
        base_knowledge: Optional[List[str]] = None,
        *,
        model_name: Optional[str] = None,
        reasoning_level: Union[ReasoningLevel, str, None] = None,
        image_detail: str = "high",
        include_visible_text_in_agent_context: bool = False,
        max_actions_per_plan: int = 6,
        current_iteration: int = 0,
        user_facing_actions_in_round: int = 0,
        previous_response_id: Optional[str] = None,
        tool_call_outputs: Optional[List[dict]] = None,
        notebook_last_sent_index: int = 0,
        memory_narrative_n: Optional[int] = None,
        last_action_summary: Optional[str] = None,
        tab_bar: Optional[str] = None,
        dialog_notice: Optional[str] = None,
        tab_events: Optional[List[str]] = None,
        dialog_pending: bool = False,
        recommended_next_step: Optional[str] = None,
        recommended_next_step_source_id: Optional[str] = None,
        decision_context: Optional[DecisionContext] = None,
        element_index_text: Optional[str] = None,
        gallery_images: Optional[List[bytes]] = None,
        user_hints: Optional[List[str]] = None,
        policy_constraints_block: Optional[str] = None,
        # Loop state
        in_loop: bool = False,
        loop_round: int = 0,
        loop_count: Optional[int] = None,
        loop_description: str = "",
        recent_actions: Optional[List[str]] = None,
        iterations_remaining: int = 0,
        max_iterations: int = 0,
        budget_spent: int = 0,
        budget_phase: str = "normal",
        low_budget_mode: bool = False,
        budget_constraints_enabled: bool = True,
        tool_registry: Optional[Any] = None,
        effect_policy_engine: Optional[Any] = None,
    ):
        self.user_prompt = user_prompt
        self.base_knowledge = base_knowledge or []
        self.model_name = model_name or get_default_agent_model()
        if reasoning_level is None:
            reasoning = ReasoningLevel.coerce(get_default_agent_reasoning_level())
        else:
            reasoning = ReasoningLevel.coerce(reasoning_level)
        self.reasoning_level: ReasoningLevel = reasoning
        self.image_detail = image_detail
        self._system_prompt_cache: dict[str, str] = {}
        self.include_visible_text_in_agent_context = include_visible_text_in_agent_context
        self.memory_store: NarrativeMemory = memory_store
        self.max_actions_per_plan = max_actions_per_plan
        self.current_iteration = current_iteration
        self.user_facing_actions_in_round = user_facing_actions_in_round
        self.previous_response_id = previous_response_id
        self.tool_call_outputs = tool_call_outputs
        self.notebook_last_sent_index = max(0, int(notebook_last_sent_index or 0))
        self.memory_narrative_n = memory_narrative_n
        self.last_response_id: Optional[str] = None
        self.last_tool_call_ids: List[str] = []
        self.last_action_summary = last_action_summary
        self.tab_bar = tab_bar
        self.dialog_notice = dialog_notice
        self.tab_events = tab_events or []
        self.dialog_pending = dialog_pending
        self.recommended_next_step = recommended_next_step
        self.recommended_next_step_source_id = recommended_next_step_source_id
        self.decision_context = decision_context
        self.element_index_text = element_index_text
        self.gallery_images = gallery_images
        self.user_hints = [hint.strip() for hint in (user_hints or []) if isinstance(hint, str) and hint.strip()]
        self.policy_constraints_block = (policy_constraints_block or "").strip()
        # Loop state
        self.in_loop = in_loop
        self.loop_round = loop_round
        self.loop_count = loop_count
        self.loop_description = loop_description
        self.recent_actions = recent_actions or []
        self.iterations_remaining = iterations_remaining
        self.max_iterations = max_iterations
        self.budget_spent = max(0, int(budget_spent or 0))
        self.budget_phase = str(budget_phase or "normal").strip().lower() or "normal"
        self.low_budget_mode = bool(low_budget_mode)
        self.budget_constraints_enabled = bool(budget_constraints_enabled)
        self.tool_registry = tool_registry
        self.effect_policy_engine = effect_policy_engine
        self.last_call_telemetry: dict[str, Any] = {}
        self.last_failure_code: Optional[str] = None
        self.last_failure_stage: Optional[str] = None
        self.last_hint_bundle: Optional[HintBundle] = None
        self.last_hint_status: str = "missing"
        self.last_hint_reason: str = ""
        self.last_hint_confidence: float = 0.0

    @staticmethod
    def _parse_next_hint_envelope(raw_hint_json: Any) -> tuple[Optional[HintCandidate], str, str, float]:
        """Parse planner-embedded next hint envelope from a JSON string."""
        if not isinstance(raw_hint_json, str):
            return None, "missing", "field_missing", 0.0
        text = raw_hint_json.strip()
        if not text:
            return None, "none", "empty_hint", 0.0
        try:
            payload = json.loads(text)
        except Exception:
            return None, "invalid", "invalid_json", 0.0
        if not isinstance(payload, dict):
            return None, "invalid", "invalid_payload", 0.0

        status = str(payload.get("s", payload.get("status", "")) or "").strip().lower()
        if status in {"c", "candidate"}:
            status = "candidate"
        elif status in {"n", "none"}:
            status = "none"
        elif status not in {"candidate", "none"}:
            if str(payload.get("f", payload.get("function_name", "")) or "").strip():
                # Backward compatibility with pre-envelope hints.
                status = "candidate"
            else:
                status = "none"

        raw_confidence = payload.get("c", payload.get("confidence", 0.0))
        try:
            confidence = float(raw_confidence)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        reason = str(payload.get("r", payload.get("reason", "")) or "").strip()
        if status == "none":
            return None, "none", reason or "unspecified", confidence

        function_name = str(payload.get("f", payload.get("function_name", "")) or "").strip()
        if not function_name:
            return None, "invalid", "missing_function_name", confidence
        function_arguments = payload.get("a", payload.get("function_arguments"))
        if not isinstance(function_arguments, dict):
            function_arguments = {}

        overlay_index = payload.get("oi", payload.get("overlay_index"))
        target_signature: Optional[HintTargetSignature] = None
        if isinstance(overlay_index, bool):
            overlay_index = None
        if overlay_index is not None:
            try:
                target_signature = HintTargetSignature(overlay_index=int(overlay_index))
            except Exception:
                target_signature = None

        candidate_id = str(
            payload.get("id", payload.get("candidate_id", "planner_hint_1")) or ""
        ).strip() or "planner_hint_1"
        return (
            HintCandidate(
                candidate_id=candidate_id,
                function_name=function_name,
                function_arguments=function_arguments,
                target_signature=target_signature,
                confidence=confidence,
                reason=reason,
            ),
            "candidate",
            reason,
            confidence,
        )

    def _build_reflection_block(self) -> str:
        """Build the reflection block for the user prompt.

        Contains:
        - LOOP STATUS: current loop round/count when in a loop
        - LAST ACTION: what the agent just did and the result
        - RECENT ACTIONS: compact log of recent actions
        - TAB EVENTS: tab opens/closes/dialog events since the last iteration
        """
        parts = []

        # Budget / iteration budget awareness
        if self.max_iterations > 0:
            mode = "ON" if self.low_budget_mode else "OFF"
            parts.append(
                "BUDGET STATE:\n"
                f"- spent={self.budget_spent}\n"
                f"- remaining={self.iterations_remaining}\n"
                f"- total={self.max_iterations}\n"
                f"- phase={self.budget_phase}\n"
                f"- low_budget_mode={mode}\n"
            )
            if self.budget_constraints_enabled:
                # parts.append(
                #     "BUDGET CONTRACT (MANDATORY IN REASONING):\n"
                #     '- Include "State: ..."\n'
                #     '- Include "Budget: spent=X, remaining=Y, total=Z"\n'
                #     '- Include "Why: ..."\n'
                # )
                if self.iterations_remaining <= 5:
                    parts.append("You are near budget exhaustion — prioritize completion-oriented actions.\n")

        # Loop framing
        if self.in_loop and self.loop_count:
            remaining = max(0, self.loop_count - self.loop_round + 1)
            parts.append(
                f"═══ LOOP {self.loop_round} OF {self.loop_count}: {self.loop_description} ═══\n"
                f"Elements marked [DONE] in the index have already been handled.\n"
                f"{remaining} round{'s' if remaining != 1 else ''} remaining.\n"
                f"After completing this round, call think(next_action=advance).\n"
            )

        if self.last_action_summary:
            parts.append(f"LAST ACTION:\n{self.last_action_summary}\n")

        if self.user_hints:
            hints_str = "\n".join(f"- {h}" for h in self.user_hints)
            parts.append(f"HINTS FROM USER:\n{hints_str}\n")

        if self.recommended_next_step:
            source = (
                f" (source: {self.recommended_next_step_source_id})"
                if self.recommended_next_step_source_id
                else ""
            )
            cleaned_step = self._strip_targeting_data(self.recommended_next_step)
            parts.append(
                f"RECOMMENDED NEXT STEP (ONE SHOT){source}:\n{cleaned_step}\n"
            )

        # Recent actions log
        if self.recent_actions:
            actions_str = "\n".join(
                f"  {i+1}. {a}" for i, a in enumerate(self.recent_actions)
            )
            parts.append(f"RECENT ACTIONS:\n{actions_str}\n")

        if self.tab_events:
            events_str = "\n".join(f"- {e}" for e in self.tab_events)
            parts.append(f"TAB EVENTS:\n{events_str}\n")

        if not parts:
            return ""

        return "\n".join(parts) + "\n"

    @staticmethod
    def _strip_targeting_data(step: str) -> str:
        """Delegate to module-level helper."""
        return strip_targeting_data(step)

    def get_next_actions_with_function_calling(
        self,
        environment_state: EnvironmentState,
        screenshot: bytes,
        notebook: Notebook,
        element_data: PageElements,
    ) -> tuple[Optional[list[ActionStep]], Optional[str]]:
        """
        Generate next action using function calling (alternative to full plan generation).

        This uses OpenAI function calling to generate a single validated action
        with strict parameter schemas.

        Args:
            environment_state: Current environment state
            screenshot: Current screenshot (viewport only)
            notebook: Agent's notebook with previously extracted data

        Returns:
            Tuple of (list[ActionStep], error_message). list[ActionStep] is None if generation failed.
        """
        self.last_call_telemetry = {}
        self.last_failure_code = None
        self.last_failure_stage = None
        self.last_hint_bundle = None
        self.last_hint_status = "missing"
        self.last_hint_reason = ""
        self.last_hint_confidence = 0.0

        try:
            # Build reflection block (last action + tab events)
            reflection = self._build_reflection_block()

            # Dialog notice goes at the very top of user prompt (highest attention)
            dialog_prefix = ""
            if self.dialog_notice:
                dialog_prefix = f"{self.dialog_notice}\n"

            decision_context_block = (
                render_decision_context(self.decision_context)
                if self.decision_context is not None
                else (
                    f"Planning iteration: unknown\n"
                    f"Action iteration: {self.current_iteration}\n"
                    f"Mission: {environment_state.user_prompt}\n"
                    f"Current page: {environment_state.current_url} — {environment_state.page_title}\n"
                    f"Budget: spent={self.budget_spent}, remaining={self.iterations_remaining}, total={self.max_iterations}\n"
                    f"Budget phase: {self.budget_phase}\n"
                    f"Low-budget mode: {'on' if self.low_budget_mode else 'off'}"
                )
            )

            # On iteration 2+ (previous_response_id set): send only the current page
            # state and fresh element index. The model recalls history, ledger, and
            # narrative from its KV cache via previous_response_id.
            # On iteration 1: send the full dynamic context as normal.
            if self.previous_response_id:
                dynamic_context = self._build_function_calling_delta_context(
                    environment_state,
                    notebook,
                    element_data,
                )
            else:
                dynamic_context = self._build_function_calling_dynamic_context(
                    environment_state,
                    notebook,
                    element_data,
                )

            user_prompt = f"""{dynamic_context}

{dialog_prefix}{reflection}Mission: {self.user_prompt}

Decision context:
{decision_context_block}

Based on the screenshot, decision context, and the mission, what is the best next action?

{SHARED_CONTRADICTION_GATE}
"""

            # Get filtered tools from registry based on policy + current state.
            if self.tool_registry is None:
                self.last_failure_code = "tool_registry_missing"
                self.last_failure_stage = "planner_tool_selection"
                return None, "Tool registry is not configured."
            tools = self.tool_registry.get_planner_schemas(
                dialog_pending=self.dialog_pending,
                budget_enabled=self.budget_constraints_enabled,
                policy_engine=self.effect_policy_engine,
            )
            if not tools:
                self.last_failure_code = "tool_policy_filtered_empty"
                self.last_failure_stage = "planner_tool_selection"
                return None, "No tools available for the active tool policy in this mode."

            # Static only — identical every iteration → cache fires from iteration 2 onward
            system_prompt = self._build_function_calling_static_prompt()

            # Build image arguments
            # Element index mode: clean screenshot + gallery pages
            # Other modes: just the screenshot
            image_arg = screenshot
            multi_image_arg = None
            if self.gallery_images:
                multi_image_arg = [screenshot] + self.gallery_images
                image_arg = None

            developer_prompt = get_memory_developer_policy(self.budget_constraints_enabled)
            planner_model = self.model_name

            # Generate action using function calling
            call_started_at = time.perf_counter()
            result = generate_action_with_tools(
                prompt=user_prompt,
                tools=tools,
                system_prompt=system_prompt,
                developer_prompt=developer_prompt,
                image=image_arg,
                multi_image=multi_image_arg,
                image_detail=self.image_detail,
                model=planner_model,
                reasoning_level=self.reasoning_level,
                tool_choice="required",
                parallel_tool_calls=self.max_actions_per_plan > 1,
                previous_response_id=self.previous_response_id,
                tool_call_outputs=self.tool_call_outputs,
            )
            llm_latency_ms = (time.perf_counter() - call_started_at) * 1000.0
            # Capture response_id and call_ids for the next iteration's chain
            if result:
                self.last_response_id = result[0].get("response_id")
                self.last_tool_call_ids = [
                    r["call_id"] for r in result if r.get("call_id")
                ]

            actions = []
            embedded_hint_candidate: Optional[HintCandidate] = None
            # Limit to max_actions_per_plan to prevent excessive batching
            get_event_logger().system_debug(f"LLM returned {len(result)} tool calls, limiting to {self.max_actions_per_plan}")
            limited_result = result[:self.max_actions_per_plan]

            for action in limited_result:
                # Check if function was called
                if not action["function_name"]:
                    return None, "Model did not call any function"

                get_event_logger().system_debug(f"Function name: {action['function_name']}")
                get_event_logger().system_debug(f"Arguments: {action['arguments']}")

                action_arguments = action["arguments"] if isinstance(action.get("arguments"), dict) else {}
                next_hint_raw = action_arguments.get("next_hint_json")
                if self.last_hint_status == "missing":
                    parsed_candidate, parsed_status, parsed_reason, parsed_conf = self._parse_next_hint_envelope(next_hint_raw)
                    self.last_hint_status = parsed_status
                    self.last_hint_reason = parsed_reason
                    self.last_hint_confidence = float(parsed_conf or 0.0)
                    embedded_hint_candidate = parsed_candidate

                # Keep execution payload clean (hint metadata is planner-side only).
                if isinstance(action_arguments, dict) and "next_hint_json" in action_arguments:
                    action_arguments = dict(action_arguments)
                    action_arguments.pop("next_hint_json", None)

                # Create ActionStep from function call
                action_step = ActionStep.from_function_call(
                    function_name=action["function_name"],
                    arguments=action_arguments,
                )
                actions.append(action_step)

            if embedded_hint_candidate is not None:
                self.last_hint_bundle = HintBundle(
                    source_iteration=int(self.current_iteration or 0),
                    source_url=str(environment_state.current_url or ""),
                    source_title=str(environment_state.page_title or ""),
                    candidates=[embedded_hint_candidate],
                )
            else:
                self.last_hint_bundle = None

            usage = {}
            if result and isinstance(result[0], dict):
                usage = result[0].get("usage") or {}
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
            image_count = 1 + len(self.gallery_images or [])
            self.last_call_telemetry = {
                "llm_latency_ms": llm_latency_ms,
                "tokens_in": input_tokens,
                "tokens_out": output_tokens,
                "tokens_total": total_tokens,
                "image_count": image_count,
                "planner_retries": 0,
                "model": planner_model,
                "hint_status": self.last_hint_status,
                "hint_reason": self.last_hint_reason,
                "hint_confidence": self.last_hint_confidence,
            }
            return actions, None

        except Exception as e:
            self.last_failure_code = "planner_generation_failed"
            self.last_failure_stage = "planner_model_call"
            get_event_logger().system_error(f"Error generating action with function calling: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error generating action: {e}"

    def _build_function_calling_static_prompt(self) -> str:
        """Build static planner instructions for function-calling planning."""
        # Build base knowledge section if provided
        base_knowledge_section = ""
        if self.base_knowledge:
            base_knowledge_section = "\n\nCUSTOM RULES:\n"
            for i, knowledge in enumerate(self.base_knowledge, 1):
                base_knowledge_section += f"{i}. {knowledge}\n"

        budget_constraints_status = "enabled" if self.budget_constraints_enabled else "disabled"
        if self.budget_constraints_enabled:
            budget_reasoning_contract = """
11. Every tool call arguments object must include:
    - budget_spent
    - budget_remaining
    - budget_total
    and these values must exactly match the Budget status shown above."""
        else:
            budget_reasoning_contract = """
11. Budget fields (budget_spent, budget_remaining, budget_total) are optional when budget constraints are disabled."""

        policy_section = ""
        if self.policy_constraints_block:
            policy_section = f"""
═══════════════════════════════════════════════════════════════
POLICY CONSTRAINTS
═══════════════════════════════════════════════════════════════
{self.policy_constraints_block}
"""
        next_hint_key_map = ", ".join(
            f"{compact_key}={full_name}"
            for compact_key, full_name in _NEXT_HINT_COMPACT_KEYS.items()
        )

        return f"""
You are controlling a web browser.
Use the dynamic context to understand current state and choose the best next action.

Tool schemas are provided as function definitions. Categories:
• Browser: click, type_text, clear_text, select_option, upload_file, set_datetime, press_key, scroll_down, scroll_up, scroll_container, scroll_to_element, open_url, go_back, go_forward
• Data/Comm: extract_data, ask_user, report_data, write_data, send_email, bash, read_file, find_files, read_clipboard, flag
• Tabs: switch_tab, close_tab, open_tab, dismiss_dialog
• Cognitive: think (next_action: continue|start_loop|advance|end_loop|done|stuck), assert_condition, wait_for

Scroll rules:
• scroll_down / scroll_up: only for the main page (no element_id).
• scroll_container: for modals/sidebars/lists; pass element_id inside that container.
• scroll_to_element: bring a specific [id] into view.

{policy_section}

═══════════════════════════════════════════════════════════════
LOOPS
═══════════════════════════════════════════════════════════════
When you need to repeat an action for multiple targets:
1. Do the first action normally.
2. At the checkpoint, call think(next_action="start_loop", loop_count=N, loop_description="...").
   N includes the action you already did (round 1).
3. Each subsequent round: do the action, then think(next_action="advance").
4. Elements you've already interacted with show [DONE] in the element index.
5. The loop auto-completes after the last round, or use think(next_action="end_loop") to exit early.

═══════════════════════════════════════════════════════════════
ERROR RECOVERY
═══════════════════════════════════════════════════════════════
If something isn't working:
• Try a different element or approach
• Scroll if you can't find what you need
• Use think() to reason about what's wrong
• Use flag() to notify the user of issues
• Don't repeat the exact same failed action

STUCK STRATEGY SWITCH RULE:
• If my recent attempts did not create visible progress, I should not keep the same strategy
• I should call think(next_action=stuck)
• In that call:
  - reasoning should be natural first-person language for a meaningfully different approach
  - recommended_next_step should be one concrete immediate action

{SHARED_CONTRADICTION_GATE}

═══════════════════════════════════════════════════════════════
GUIDELINES
═══════════════════════════════════════════════════════════════
1. Only act on elements visible in the screenshot — click by their visual position
2. Be specific in element descriptions
3. type_text REPLACES content (doesn't append)
4. Check focused=true before clicking to focus
5. Don't repeat failed actions
6. think(next_action=continue) should be a brief natural first-person status + immediate next step (no full re-plan)
7. If what you planned conflicts with the current screenshot, follow the screenshot and adjust plan
8. Stick exactly to the mission. Do not add extra steps, verification, confirmations, or sub-tasks that the mission did not ask for.
9. If you call think(next_action=stuck), propose a meaningfully different approach and one concrete immediate action.
10. Reference relevant memory entries (mem_XXXXXX) in your reasoning. If a RECOMMENDED NEXT STEP is present, follow it or explain why you're deviating.
11. `next_hint_json` is REQUIRED on every tool call and must always be valid JSON.
    Compact key map (must follow exactly): {next_hint_key_map}
    Status values: "c" for candidate, "n" for none.
    Use exactly one envelope:
    - Candidate envelope:
      {{"s":"c","f":"...","a":{{...}},"oi":123,"c":0.0-1.0,"r":"...","id":"optional_id"}}
    - None envelope (when uncertain):
      {{"s":"n","r":"why uncertain","c":0.0}}
{budget_reasoning_contract}
{base_knowledge_section}
"""

    def _build_function_calling_dynamic_context(
        self,
        state: EnvironmentState,
        notebook: Notebook,
        element_data: PageElements,
    ) -> str:
        """Build dynamic planner context that changes every iteration."""

        user_hints_section = ""
        if self.user_hints:
            user_hints_section = "\n\nHINTS FROM USER:\n"
            for hint in self.user_hints:
                user_hints_section += f"- {hint}\n"

        # Get history and navigation info
        memory_narrative_block = self._get_memory_narrative_block()
        memory_index_block = self._get_memory_entry_index()
        executed_action_ledger = self.memory_store.get_executed_action_ledger(
            n=max(1, len(self.memory_store.entries))
        )
        recent_user_answers_block = self._get_recent_user_answers_block()
        stuck_hint_lines = self.memory_store.get_stuck_pattern_hints()
        nav_summary = self._summarize_navigation_history(
            getattr(state, "url_history", []),
            getattr(state, "url_pointer", None)
        )

        # Build loop context (only shown when in a loop)
        progress_info = ""

        forced_think_prompt = ""

        # Build tab bar section (only shown when 2+ tabs are open)
        tab_section = ""
        if self.tab_bar:
            tab_section = f"""
═══════════════════════════════════════════════════════════════
OPEN TABS
═══════════════════════════════════════════════════════════════
{self.tab_bar}

"""

        stuck_hints = ", ".join(stuck_hint_lines) if stuck_hint_lines else "none right now"

        # Build element index section
        gallery_note = ""
        if self.gallery_images:
            n = len(self.gallery_images)
            gallery_note = (
                f"\nYou are also shown {n} CROP GALLERY image{'s' if n > 1 else ''} "
                f"after the screenshot. Elements marked 'SEE CROP GALLERY' in the "
                f"index have visual crops in these gallery images."
            )

        opening_instruction = (
            "You are controlling a web browser. You can see the current page "
            "as a screenshot." + gallery_note + "\n"
            "Use the INTERACTIVE ELEMENTS index below to identify elements by their [id] number."
        )
        element_section = f"""═══════════════════════════════════════════════════════════════
INTERACTIVE ELEMENTS
═══════════════════════════════════════════════════════════════
{self.element_index_text or "No interactive elements detected."}

HOW TO CLICK:
• Pick the [id] number from the list above and pass it as element_id.
• For elements marked 'SEE CROP GALLERY', look at the gallery images to
  visually identify the element before clicking.
• The system clicks the element directly by its DOM reference — no coordinate
  estimation needed.

FOCUS STATE:
• focused → Element already has keyboard focus
• If you need to type and input is focused → Just call type_text (don't click first)"""

        return f"""
═══════════════════════════════════════════════════════════════
DYNAMIC CONTEXT
═══════════════════════════════════════════════════════════════
{opening_instruction}

{forced_think_prompt}

{progress_info}
{tab_section}
═══════════════════════════════════════════════════════════════
WHAT YOU'VE DONE SO FAR
═══════════════════════════════════════════════════════════════
{memory_narrative_block if memory_narrative_block else "No actions yet."}
Executed action ledger (facts only):
{executed_action_ledger}

Mission: {self.user_prompt}

{recent_user_answers_block}

Potential stuck patterns from memory scan: {stuck_hints}

Memory ID ledger (for citing any prior memory entry):
{memory_index_block}

Navigation history:
{nav_summary}

{element_section}

{self._format_notebook(notebook)}

{user_hints_section}

Choose the next action to take.
"""

    def _build_function_calling_delta_context(
        self,
        state: EnvironmentState,
        notebook: Notebook,
        element_data: PageElements,
    ) -> str:
        """Minimal context for iterations 2+ when previous_response_id is active.

        The model already has the full history (narrative, action ledger, navigation
        history, memory index, Q&A pairs) in its KV cache from prior iterations.
        We only send what is genuinely new: the current page URL/title, the fresh
        element index, and anything session-scoped that may have changed (tabs,
        notebook, user hints).
        """
        # Gallery note (new each iteration if crop gallery is present)
        gallery_note = ""
        if self.gallery_images:
            n = len(self.gallery_images)
            gallery_note = (
                f"You are also shown {n} CROP GALLERY image{'s' if n > 1 else ''} "
                f"after the screenshot. Elements marked 'SEE CROP GALLERY' in the "
                f"index have visual crops in these gallery images.\n"
            )

        # Tab bar — only when 2+ tabs are open
        tab_section = ""
        if self.tab_bar:
            tab_section = f"""
═══════════════════════════════════════════════════════════════
OPEN TABS
═══════════════════════════════════════════════════════════════
{self.tab_bar}
"""

        # User hints — always include so the agent doesn't miss new guidance
        user_hints_section = ""
        if self.user_hints:
            hints_str = "\n".join(f"- {h}" for h in self.user_hints)
            user_hints_section = f"\nHINTS FROM USER:\n{hints_str}\n"

        # Element index — always fresh (page may have changed)
        element_section = f"""═══════════════════════════════════════════════════════════════
INTERACTIVE ELEMENTS
═══════════════════════════════════════════════════════════════
{self.element_index_text or "No interactive elements detected."}

HOW TO CLICK:
• Pick the [id] number from the list above and pass it as element_id.
• For elements marked 'SEE CROP GALLERY', look at the gallery images to
  visually identify the element before clicking.
• The system clicks the element directly by its DOM reference — no coordinate
  estimation needed.

FOCUS STATE:
• focused → Element already has keyboard focus
• If you need to type and input is focused → Just call type_text (don't click first)"""

        # Notebook — include if non-empty (agent needs to see what it has collected)
        notebook_section = self._format_notebook(notebook)

        return f"""
═══════════════════════════════════════════════════════════════
CURRENT PAGE
═══════════════════════════════════════════════════════════════
URL: {state.current_url}
Title: {state.page_title}
{gallery_note}
(Prior context — mission history, action ledger, navigation history, memory entries — is available from your previous conversation turns.)
{tab_section}{user_hints_section}
{element_section}

{notebook_section}

Choose the next action to take.
"""

    def _build_function_calling_system_prompt(
        self,
        state: EnvironmentState,
        notebook: Notebook,
        element_data: PageElements,
    ) -> str:
        """Backward-compatible full prompt builder for HTTP path."""
        static_prompt = self._build_function_calling_static_prompt()
        dynamic_prompt = self._build_function_calling_dynamic_context(
            state, notebook, element_data
        )
        return f"{static_prompt}\n\n{dynamic_prompt}"

    def _get_memory_narrative_block(self, just_data: bool = False) -> str:
        if not self.memory_store:
            return ""
        n = self.memory_narrative_n if self.memory_narrative_n is not None else max(1, len(self.memory_store.entries))
        narrative = self.memory_store.get_narrative(n=n)
        if just_data:
            return narrative
        return f"Recent memory narrative:\n{narrative}"

    def _get_recent_user_answers_block(self) -> str:
        """Format recent ask_user answers so planner can leverage them directly."""
        if not self.memory_store:
            return (
                "═══════════════════════════════════════════════════════════════\n"
                "RECENT USER ANSWERS\n"
                "═══════════════════════════════════════════════════════════════\n"
                "No user answers recorded yet."
            )

        total_pairs = len(self.memory_store.question_answer_pairs)
        pairs = self.memory_store.get_recent_question_answers(n=total_pairs)
        if not pairs:
            return (
                "═══════════════════════════════════════════════════════════════\n"
                "RECENT USER ANSWERS\n"
                "═══════════════════════════════════════════════════════════════\n"
                "No user answers recorded yet."
            )

        lines: List[str] = []
        for idx, pair in enumerate(pairs, start=1):
            question = str(pair.get("question", "") or "").strip() or "Unknown question"
            answer = str(pair.get("answer", "") or "").strip() or "(empty answer)"
            lines.append(f"{idx}. Q: {question}")
            lines.append(f"   A: {answer}")

        lines.append(
            "Use these as authoritative user input. Avoid re-asking the same question unless needed."
        )
        body = "\n".join(lines)
        return (
            "═══════════════════════════════════════════════════════════════\n"
            "RECENT USER ANSWERS\n"
            "═══════════════════════════════════════════════════════════════\n"
            f"{body}"
        )

    def _get_memory_entry_index(self) -> str:
        if not self.memory_store or not self.memory_store.entries:
            return "No memory entries yet."

        lines: List[str] = []
        for entry in self.memory_store.entries:
            lines.append(
                f"[{entry.memory_id}] {entry.entry_kind} | {entry.action_type} -> {entry.outcome}"
            )

        return "\n".join(lines)

    @staticmethod
    def _truncate_compact_text(value: Any, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) <= max_chars:
            return text
        return f"{text[:max(0, max_chars - 3)]}..."

    def _format_notebook_data_preview(self, data: Any) -> str:
        if isinstance(data, (dict, list)):
            try:
                serialized = json.dumps(data, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            except Exception:
                serialized = str(data)
        else:
            serialized = str(data)
        return self._truncate_compact_text(serialized, _NOTEBOOK_ENTRY_DATA_MAX_CHARS)

    def _format_notebook(self, notebook: Notebook) -> str:
        """Format notebook entries for inclusion in the prompt."""

        entries = notebook.to_list()
        if not entries:
            return ""

        start_index = 0
        heading = "NOTEBOOK (Your Stored Data)"
        summary = ""
        if self.previous_response_id:
            # Delta mode: prior notebook entries are already in the Responses API chain context.
            start_index = min(self.notebook_last_sent_index, len(entries))
            if start_index >= len(entries):
                return ""
            heading = "NOTEBOOK (New Since Last Turn)"
            summary = (
                f"Showing {len(entries) - start_index} new entries. "
                f"{start_index} earlier entries are already in prior context.\n"
            )

        lines: List[str] = []
        for i, entry in enumerate(entries[start_index:], start=start_index + 1):
            description = self._truncate_compact_text(
                getattr(entry, "description", ""),
                _NOTEBOOK_ENTRY_DESCRIPTION_MAX_CHARS,
            )
            data_preview = self._format_notebook_data_preview(getattr(entry, "data", ""))
            lines.append(f"{i}. {description} -> {data_preview}")

        notebook_str = "\n".join(lines)
        return f"""
═══════════════════════════════════════════════════════════════
{heading}
═══════════════════════════════════════════════════════════════
{summary}{notebook_str}
• Use extract_data to store information
• Check notebook to see what you've already collected
"""

    def _summarize_navigation_history(self, url_history: List[str], url_pointer: Optional[int]) -> str:
        """Provide a concise navigation summary for the prompt."""
        if not url_history:
            return "No navigation history recorded yet."

        total = len(url_history)
        pointer = url_pointer if url_pointer is not None and 0 <= url_pointer < total else total - 1
        pointer = max(0, pointer)

        prev_url = url_history[pointer - 1] if pointer > 0 else None
        next_url = url_history[pointer + 1] if pointer < total - 1 else None

        lines = []
        for idx in range(total):
            marker = " (current)" if idx == pointer else ""
            lines.append(f"{idx}: {url_history[idx]}{marker}")

        recent_urls_block = "\n    ".join(lines)
        prev_line = f"Previous page (back target): {prev_url}" if prev_url else "Previous page (back target): none"
        next_line = f"Next page (forward target): {next_url}" if next_url else "Next page (forward target): none"

        return (
            f"Total pages visited: {total}\n"
            f"Current history index: {pointer}\n"
            f"{prev_line}\n"
            f"{next_line}\n"
            f"Recent history (oldest → newest):\n    {recent_urls_block}"
        )
