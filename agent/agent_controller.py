"""Agent controller — mission execution with inline loops.

Architecture overview:
- Mission goes directly to the execution loop (no planner).
- Agent uses browser actions to accomplish the mission.
- When repetition is needed, the agent declares a loop inline via think(start_loop).
- Loop rounds are advanced with think(advance) and exited with think(end_loop) or think(done).
"""

import os
import json
import time
import threading
import re
import uuid
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Callable, Union, Type
import hashlib
import copy

from browser.dom import build_page_elements
from browser.annotate import build_element_index, build_crop_gallery
from core.browser import ExecutionTimer
from core.executor.base import Executor
from core.agent_workspace import AgentWorkspace, AgentWorkspaceManager
from core.sandbox_policy import SandboxPolicyEngine
from agent.memory import InteractionType, MemoryEntryKind, MemoryState, NarrativeMemory
from models import PageElements, PageInfo
from models.models import ActionStep, FailedAction, set_action_text_renderer
from agent.results import MissionResult
from agent.agent_context import EnvironmentState
from agent.events import (
    AgentEvent,
    EventDefinition,
    EventResult,
    coerce_emit_events,
    normalize_event_definitions,
    validate_event_payload,
)
from agent.notebook import Notebook
from agent.speculative_hints import (
    HintBundle,
    HintCandidate,
    HintValidationResult,
    filter_candidates_deterministic,
    hydrate_candidate_to_action_step,
    set_tools_requiring_element,
    validate_hints,
)
from agent.prompts import (
    DecisionContext,
)
from agent.skills import (
    SkillMeta,
    discover_skills,
    format_active_skill,
    format_skills_catalog,
    load_skill_body,
)
from agent.tools import create_default_registry
from agent.tooling import (
    Effect,
    EffectPolicyEngine,
    ThinkControl,
    ThinkNextAction,
    ToolContext,
    ToolEngine,
    ToolOutcome,
    ToolOutput,
)
from utils.debug_print import dprint
from lib.ai import (
    ReasoningLevel,
    set_default_model,
    set_default_reasoning_level,
    set_default_agent_model,
    set_default_agent_reasoning_level,
)
from pydantic import BaseModel

from utils.event_logger import set_event_logger
from utils.screenshot_store import ScreenshotStore, ScreenshotMeta

from .interceptor_manager import InterceptorManager, Interceptor, InterceptorMode, InterceptorContext
from core.config import Config
from execution.result import ActionResult
from utils import PageUtils
from core.browser import Browser

# Type alias for user question callback (ask: command handler)
# Callback receives: question, context, options, multi_select, yes_no.
UserQuestionCallback = Callable[[str, dict, List[str], bool, bool], str]
# Type alias for reported text callback (report_data: command handler)
# Callback receives: payload (str), context (dict)
DataReportCallback = Callable[[str, dict], None]
EventCallback = Callable[[AgentEvent], Any]


@dataclass
class ExecutionState:
    """Mutable execution state for the mission loop."""
    total_actions: int = 0
    actions_since_progress: int = 0
    user_facing_actions_since_progress: int = 0
    last_action_summary: Optional[str] = None
    last_response_id: Optional[str] = None
    last_tool_call_ids: List[str] = field(default_factory=list)
    notebook_entries_sent: int = 0
    failed_elements: List[FailedAction] = field(default_factory=list)
    validation_failures: int = 0
    # Loop state
    in_loop: bool = False
    loop_count: Optional[int] = None
    loop_round: int = 0
    loop_description: str = ""
    # Recent action log (compact summaries)
    recent_actions: List[str] = field(default_factory=list)
    # Mission-scoped planner scratchpad notes: (note_id, note_text)
    agent_notes: List[Tuple[int, str]] = field(default_factory=list)
    agent_note_counter: int = 0
    # Skill context
    active_skill_name: Optional[str] = None
    active_skill_body: Optional[str] = None
    # Budget telemetry
    budget_total: int = 0
    budget_spent: int = 0
    budget_remaining: int = 0
    budget_phase: str = "normal"
    low_budget_mode: bool = False
    budget_constraints_enabled: bool = True
    planning_batch_limit: int = 0
    # Aggregated telemetry
    iteration_ms_samples: List[float] = field(default_factory=list)
    llm_latency_ms_samples: List[float] = field(default_factory=list)
    tool_latency_ms_samples: List[float] = field(default_factory=list)
    navigation_latency_ms_samples: List[float] = field(default_factory=list)
    tokens_in_total: int = 0
    tokens_out_total: int = 0
    image_count_total: int = 0
    llm_call_count: int = 0
    tool_call_count: int = 0
    retry_count: int = 0
    # Last-iteration telemetry values
    iteration_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    navigation_latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    image_count: int = 0
    tool_calls: int = 0
    retries: int = 0
    # Failure tagging
    failure_code: Optional[str] = None
    failure_stage: Optional[str] = None


def merge_agent_notes(state: ExecutionState, raw_notes: List[str]) -> None:
    """Append planner scratchpad notes to state with stable incremental IDs."""
    for raw_note in raw_notes:
        note_text = str(raw_note or "").strip()
        if not note_text:
            continue
        state.agent_note_counter += 1
        state.agent_notes.append((state.agent_note_counter, note_text))


def clear_planner_response_chain(state: ExecutionState) -> None:
    """Drop cached Responses API chain state so the next planner call is fresh."""
    state.last_response_id = None
    state.last_tool_call_ids = []
    state.notebook_entries_sent = 0


def is_context_length_exceeded_error(error_text: Optional[str]) -> bool:
    text = str(error_text or "").strip().casefold()
    if not text:
        return False
    return (
        "context_length_exceeded" in text
        or "exceeds the context window" in text
        or "input exceeds the context window" in text
        or "maximum context length" in text
    )


@dataclass
class ScreenshotPreparation:
    """Result of _build_element_index_and_start_gallery()."""
    screenshot_bytes: bytes
    gallery_images: Optional[List[bytes]] = None
    element_index_text: Optional[str] = None
    text_rich_count: int = 0
    text_poor_count: int = 0


@dataclass
class PlannerCallResult:
    """Result payload from one planner model invocation."""

    actions: Optional[List[ActionStep]] = None
    error: Optional[str] = None
    response_id: Optional[str] = None
    tool_call_ids: List[str] = field(default_factory=list)
    planner_elapsed_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    image_count: int = 0
    retries: int = 0
    failure_code: Optional[str] = None
    failure_stage: Optional[str] = None
    exception: Optional[Exception] = None
    hint_bundle: Optional[HintBundle] = None
    hint_status: Optional[str] = None
    hint_reason: Optional[str] = None
    hint_confidence: float = 0.0


@dataclass
class SpeculativeResolution:
    """Result payload for planner/validator arbitration."""

    actions: Optional[List[ActionStep]] = None
    error: Optional[str] = None
    planner: PlannerCallResult = field(default_factory=PlannerCallResult)
    hint_path: str = "planner_only"
    hint_candidate_id: Optional[str] = None
    hint_validation_ms: float = 0.0
    hint_confidence: float = 0.0
    hint_reject_reason: Optional[str] = None


@dataclass
class ToolRuntimeAdapter:
    """Adapter exposed to ToolContext for delegated built-in execution."""

    agent: "Agent"
    detected_elements: PageElements
    page_info: PageInfo
    environment_state: EnvironmentState
    current_iteration: int

    def execute_builtin_tool(self, tool_name: str, args_model: Any) -> ToolOutcome:
        args = (
            args_model.model_dump(mode="python")
            if hasattr(args_model, "model_dump")
            else dict(args_model or {})
        )
        args_dict = args if isinstance(args, dict) else {}
        action_step = ActionStep.from_function_call(tool_name, args_dict)

        if tool_name in self.agent._controller_tool_names():
            return self.agent._execute_controller_tool(
                tool_name=tool_name,
                action_step=action_step,
                action_args=args_dict,
            )

        observe_warning: Optional[str] = None
        if tool_name == "open_url":
            url = str(args_dict.get("url", "")).strip()
            url_decision = self.agent.sandbox_policy.check_url(url)
            if not url_decision.allowed:
                warning = f"open_url blocked by sandbox: {url_decision.reason}"
                self.agent.event_logger.system_warning(warning)
                if self.agent.sandbox_policy.enforce:
                    before_state = self.agent.memory_store._capture_current_state()
                    self.agent._record_controller_action(
                        action_type=InteractionType.NAVIGATION.value,
                        action_step=action_step,
                        success=False,
                        error_message=warning,
                        action_params={
                            "operation": "open_url",
                            "url": url,
                            "sandbox_blocked": True,
                        },
                        before_state=before_state,
                        after_state=self.agent.memory_store._capture_current_state(),
                    )
                    return ToolOutcome(
                        output=ToolOutput(
                            success=False,
                            summary=f"open_url FAILED: {url_decision.reason}",
                            error=warning,
                        )
                    )
                observe_warning = warning

        result = self.agent.action_executor.execute_via_adapter(
            function_name=tool_name,
            function_arguments=args_dict,
            detected_elements=self.detected_elements,
            page_info=self.page_info,
            environment_state=self.environment_state,
            base_knowledge=self.agent.base_knowledge,
            current_iteration=self.current_iteration,
        )
        success = bool(getattr(result, "success", False))
        summary = (
            self.agent._build_action_summary(action_step, "success" if success else "failed")
            if action_step is not None
            else (getattr(result, "message", "") or "")
        )
        if observe_warning:
            summary = f"{summary} | sandbox(observe): {observe_warning}" if summary else observe_warning
        error = str(getattr(result, "error", "") or "").strip() or None
        data = getattr(result, "data", None)
        return ToolOutcome(
            output=ToolOutput(
                success=success,
                summary=summary,
                error=error,
                data=data if isinstance(data, dict) else None,
            )
        )


"""
Agent Controller - Mission Execution

Runs missions through the execution loop with inline loops for repetition.
"""

DEFAULT_RESEND_FROM_EMAIL = "Agent <agent@updates.thebrowseragentcompany.com>"
DECISION_CONTEXT_EXECUTED_ID_WINDOW = 8
DECISION_CONTEXT_REFLECTION_ID_WINDOW = 6


def _load_dotenv_if_available() -> None:
    """Load local .env values when python-dotenv is available."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    try:
        load_dotenv(override=False)
    except Exception:
        return


def _resolve_resend_from_email() -> str:
    configured = str(os.environ.get("RESEND_FROM_EMAIL", "")).strip()
    return DEFAULT_RESEND_FROM_EMAIL


def _parse_email_recipients(raw_to: Any) -> List[str]:
    if raw_to is None:
        return []
    if isinstance(raw_to, (list, tuple, set)):
        pieces = [str(item).strip() for item in raw_to]
    else:
        pieces = [piece.strip() for piece in re.split(r"[;,]", str(raw_to))]
    recipients: List[str] = []
    seen: set[str] = set()
    for piece in pieces:
        if not piece:
            continue
        key = piece.lower()
        if key in seen:
            continue
        seen.add(key)
        recipients.append(piece)
    return recipients


def _format_resend_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    details: List[str] = []
    code = getattr(exc, "code", None)
    if code not in (None, ""):
        details.append(f"code={code}")
    error_type = str(getattr(exc, "error_type", "")).strip()
    if error_type:
        details.append(f"type={error_type}")
    suggested_action = str(getattr(exc, "suggested_action", "")).strip()
    if suggested_action:
        details.append(f"suggested_action={suggested_action}")
    if details:
        return f"{message} ({'; '.join(details)})"
    return message

class Agent:
    """
    Agent controller running mission execution.
    """
    
    def __init__(
        self,
        config: Config,
        base_knowledge: Optional[List[str]] = None,
        clarification_callback: Optional[Any] = None,
        # User question callback for ask: command
        user_question_callback: Optional[UserQuestionCallback] = None,
        # Agent talk callback for talk: command
        agent_talk_callback: Optional[Callable[[str], None]] = None,
        # Completion callback for complete: command
        completion_callback: Optional[Callable[[str], None]] = None,
        # Data report callback for report_data: command
        data_report_callback: Optional[DataReportCallback] = None,
        # Agent event callback for cross-tool milestone events.
        event_callback: Optional[EventCallback] = None,
        # Allowed event definitions exposed to planner/runtime.
        event_definitions: Optional[List[EventDefinition]] = None,
        # Callback to request a hint when the agent declares itself stuck
        on_stuck_callback: Optional[Callable[[str, int], Optional[str]]] = None,
        # Optional existing agent id to reuse/load its workspace.
        agent_id: Optional[str] = None,
    ):
        _load_dotenv_if_available()
        self.config = config
        self.mission_result = MissionResult()

        # Set global print mode based on config (affects all dprint calls and API debug logging)
        from utils.debug_print import set_print_mode, PrintMode
        set_print_mode(PrintMode.DEBUG if config.debug.debug_mode else PrintMode.NORMAL)

        # Access event logger from agent
        from utils.event_logger import EventLogger
        self.event_logger = EventLogger(
            debug_mode=config.debug.debug_mode,
            show_overlay_candidates=config.debug.show_overlay_candidates,
            show_llm_costs=config.debug.show_llm_costs,
            suppress_policy_debug_logs=bool(getattr(config.debug, "suppress_policy_debug_logs", False)),
            suppress_live_telemetry_terminal_logs=bool(
                getattr(config.debug, "suppress_live_telemetry_terminal_logs", False)
            ),
        )
        set_event_logger(self.event_logger)  # Set as global
        
        self.iteration_delay = 0.5
        self.mission_start_url: Optional[str] = None
        self.mission_start_time: Optional[float] = None
        self.base_knowledge = base_knowledge or []  # Base knowledge rules that guide agent behavior
        self.notebook: Notebook = Notebook()
        self._mission_tracker: Dict[str, Dict[str, Any]] = {}
        # Completion / evaluation behavior
        self.clarification_callback = clarification_callback
        self._user_inputs: List[Dict[str, Any]] = []
        self._temp_user_inputs: List[Dict[str, Any]] = []  # Single-use suggestions
        self._original_user_prompt: str = ""

        # Store user question callback for ask: command
        self.user_question_callback = user_question_callback
        self.agent_talk_callback = agent_talk_callback
        self._last_ask_iteration: int = -2  # Track last iteration where ask: was answered (to prevent consecutive asks)

        # Store completion callback for complete: command
        self.completion_callback = completion_callback
        self.data_report_callback = data_report_callback
        self.event_callback = event_callback
        normalized_event_definitions, event_lookup = normalize_event_definitions(event_definitions)
        self.event_definitions: List[EventDefinition] = normalized_event_definitions
        self.event_definition_map: Dict[str, EventDefinition] = event_lookup
        self._accepted_agent_event_counts: Dict[str, int] = {}
        self._reset_agent_event_tracking()
        self.event_callback_timeout_seconds: float = float(
            max(0.0, float(getattr(self.config.execution, "agent_events_callback_timeout_seconds", 0.0) or 0.0))
        )
        self.event_callback_response_max_chars: int = int(
            max(0, int(getattr(self.config.execution, "agent_events_callback_response_max_chars", 0) or 0))
        )
        self.on_stuck_callback = on_stuck_callback
        self._screenshot_counter = 0  # Counter for naming screenshots

        self.agent_model_name: str = config.model.agent_model
        self.agent_reasoning_level: ReasoningLevel = config.model.agent_reasoning_level
        self.command_model_name: str = self.config.model.command_model
        self.command_reasoning_level: ReasoningLevel = self.config.model.command_reasoning_level
        self.image_detail: str = config.model.image_detail
        self.wait_for_load_state: str = str(self.config.execution.wait_for_load_state or "networkidle")
        self.wait_for_load_timeout_ms: int = int(self.config.execution.wait_for_load_timeout_ms or 0)
        self.tool_registry = create_default_registry()
        set_action_text_renderer(self.tool_registry.render_action_text)
        self.effect_policy = EffectPolicyEngine(
            preset=self.config.execution.tool_policy.preset,
            mode=self.config.execution.tool_policy.mode,
            event_logger=self.event_logger,
        )
        self.tool_engine = ToolEngine(
            registry=self.tool_registry,
            policy_engine=self.effect_policy,
            event_logger=self.event_logger,
        )
        self.policy_visible_tool_names: List[str] = []
        self._cached_snapshot: Optional[MemoryState] = None
        self._cached_snapshot_fingerprint: Optional[str] = None
        self._cached_page_info: Optional[PageInfo] = None
        self._cached_detected_elements: Optional[PageElements] = None
        set_tools_requiring_element(self.tool_registry.tools_requiring_element())
        set_default_model(self.command_model_name)
        set_default_reasoning_level(self.command_reasoning_level)
        set_default_agent_model(self.agent_model_name)
        set_default_agent_reasoning_level(self.agent_reasoning_level)
        
        # Pause functionality: Allows pausing agent execution between actions
        self._paused = False
        self._pause_lock = threading.Lock()  # Thread-safe access to pause state
        self._pause_event = threading.Event()  # Event to block execution when paused
        self._pause_event.set()  # Initially not paused (event is set = not blocking)
        self._pause_message = "Paused"
        self._cancel_event = threading.Event()
        self._hints_lock = threading.Lock()
        self._pending_hints: List[str] = []
        self._gallery_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gallery")
        self._speculative_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="speculative")
        self._speculative_lock = threading.Lock()
        self._planner_cached_hint_bundle: Optional[HintBundle] = None

        self.interceptor_stack: List[Dict[str, Any]] = []  # Stack of active interceptors

        # Execution timer for tracking mission, iteration, and action timings
        self.execution_timer = ExecutionTimer()
        self.agent_id: Optional[str] = (str(agent_id).strip() if agent_id else None)
        self._loaded_resume_run_id: Optional[str] = None
        self._loaded_resume_mission: str = ""
        self._resume_checkpoint_loaded: bool = False

        # Per-agent storage workspace.
        self.workspace_manager = AgentWorkspaceManager(
            base_dir=self.config.storage.base_dir,
            event_logger=self.event_logger,
        )
        self.agent_workspace: AgentWorkspace = self.workspace_manager.create_agent(
            persistence_mode=self.config.storage.default_persistence_mode,
            agent_id=self.agent_id,
        )
        # Mirror resolved workspace id (new or loaded) for callers.
        self.agent_id = self.agent_workspace.agent_id
        self._active_run_id: Optional[str] = None
        self._active_run_open: bool = False
        self._run_event_log_callback: Optional[Callable[[Any], None]] = None
        self._run_event_log_handle = None

        # Route storage defaults into this agent workspace.
        self._apply_workspace_paths()

        self.show_llm_costs = config.debug.show_llm_costs
        _show_overlay_candidates = config.debug.show_overlay_candidates
        self.save_screenshots = config.debug.save_screenshots
        self.screenshot_dir = config.debug.screenshot_dir

        self._current_iteration = 0
        self.execution_state: Optional[ExecutionState] = None
        self.available_skills: List[SkillMeta] = []
        self.available_skills_catalog: str = ""
        self._available_skills_by_key: Dict[str, SkillMeta] = {}

        # Initialize execution system
        self._extraction_model_cache: Dict[tuple[str, ...], Type[BaseModel]] = {}
        self.screenshot_store = ScreenshotStore(
            max_in_memory_items=self.config.debug.screenshot_stream_in_memory_items,
            max_in_memory_mb=self.config.debug.screenshot_stream_in_memory_mb,
            persist_to_disk=self.config.debug.screenshot_stream_persist_to_disk,
            disk_dir=self.config.debug.screenshot_stream_dir,
            max_disk_files=self.config.debug.screenshot_stream_max_disk_files,
        )
        self.sandbox_policy = SandboxPolicyEngine(
            config=self.config,
            workspace_root=self.agent_workspace.workspace_root,
            event_logger=self.event_logger,
        )

    def __enter__(self) -> 'Agent':
        """
        Context manager entry point. Automatically calls start().
        
        Example:
            >>> with Agent(browser, config) as agent:
            ...     agent.execute_mission("Click the button")
            # Automatically calls end() on exit
        """
        self._start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.event_logger.system_info("Stopping agent")
        # Clean up browser if it exists
        if self.browser:
            try:
                self.browser.end()
                self.event_logger.system_info("Browser stopped")
            except Exception:
                pass
        self.event_logger.system_info("Agent stopped")

    def register_tool(self, fn: Callable[..., Any]) -> None:
        """Register a custom tool into this agent instance registry."""
        self.tool_registry.register(fn)
        set_tools_requiring_element(self.tool_registry.tools_requiring_element())
        self._refresh_policy_visible_tool_names()

    @staticmethod
    def _normalize_skill_key(value: str) -> str:
        return re.sub(r"[\s_-]+", "-", str(value or "").strip().casefold())

    def _resolve_skill_directories(self) -> list[str]:
        """Resolve configured skill directories relative to workspace root."""
        configured = list(getattr(getattr(self.config, "skills", None), "skills_dirs", []) or [])
        if not configured:
            configured = ["agent_skills"]
        workspace_root = self.agent_workspace.workspace_root
        resolved: list[str] = []
        for raw in configured:
            text = str(raw or "").strip()
            if not text:
                continue
            candidate = Path(text).expanduser()
            if not candidate.is_absolute():
                candidate = (workspace_root / candidate).resolve()
            else:
                candidate = candidate.resolve()
            resolved.append(str(candidate))
        return resolved

    def _discover_available_skills(self) -> None:
        self.available_skills = []
        self.available_skills_catalog = ""
        self._available_skills_by_key = {}

        if not bool(getattr(getattr(self.config, "skills", None), "enabled", True)):
            return

        skill_dirs = self._resolve_skill_directories()
        started = time.perf_counter()
        try:
            discovered = discover_skills(skill_dirs)
            self.available_skills = discovered
            self.available_skills_catalog = format_skills_catalog(discovered) if discovered else ""

            for skill in discovered:
                key_name = self._normalize_skill_key(skill.name)
                if key_name and key_name not in self._available_skills_by_key:
                    self._available_skills_by_key[key_name] = skill
                key_dir = self._normalize_skill_key(skill.path.name)
                if key_dir and key_dir not in self._available_skills_by_key:
                    self._available_skills_by_key[key_dir] = skill

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.event_logger.skills_discovery_completed(
                directories=skill_dirs,
                discovered_count=len(discovered),
                catalog_chars=len(self.available_skills_catalog),
                duration_ms=elapsed_ms,
            )
            self.event_logger.system_info(
                "Skills discovery complete",
                skills_enabled=True,
                discovered_count=len(discovered),
                directories=skill_dirs,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.available_skills = []
            self.available_skills_catalog = ""
            self._available_skills_by_key = {}
            self.event_logger.skills_discovery_failed(
                directories=skill_dirs,
                error=str(e),
                failure_code="discovery_error",
                duration_ms=elapsed_ms,
            )
            raise

    def _find_skill_by_name(self, raw_name: str) -> Optional[SkillMeta]:
        key = self._normalize_skill_key(raw_name)
        if not key:
            return None
        return self._available_skills_by_key.get(key)

    def _build_active_skill_context(self, state: ExecutionState) -> Optional[str]:
        skill_name = str(getattr(state, "active_skill_name", "") or "").strip()
        skill_body = str(getattr(state, "active_skill_body", "") or "").strip()
        if not skill_name or not skill_body:
            return None
        skill = self._find_skill_by_name(skill_name)
        if skill is None:
            synthetic_skill = SkillMeta(
                name=skill_name,
                description="Restored active skill",
                path=self.agent_workspace.workspace_root,
            )
            return format_active_skill(synthetic_skill, skill_body)
        return format_active_skill(skill, skill_body)

    def _apply_think_control(
        self,
        *,
        control: ThinkControl,
        state: ExecutionState,
        append_recent_action: Callable[[str], None],
        exit_loop: Callable[[], None],
    ) -> tuple[Optional[str], bool]:
        """Apply think-only control semantics. Returns (mission_done_reasoning, should_replan)."""
        next_action = control.next_action

        if next_action == ThinkNextAction.START_LOOP:
            requested_count = int(control.loop_count or 1)
            count, clamped = self._clamp_loop_count_to_budget(
                requested_loop_count=requested_count,
                budget_remaining=(state.budget_remaining if state.budget_constraints_enabled else 0),
            )
            description = str(control.loop_description or "").strip() or "loop"
            state.in_loop = True
            state.loop_count = count
            state.loop_round = 2
            state.loop_description = description
            state.user_facing_actions_since_progress = 0
            summary = f"Loop started: {description} (round 2 of {count})"
            if clamped:
                summary += f" [clamped to budget remaining={state.budget_remaining}]"
            state.last_action_summary = summary
            append_recent_action(f"[LOOP START] {description} — {count} total rounds")
            self.event_logger.loop_state_changed(
                change="start",
                loop_round=state.loop_round,
                loop_count=state.loop_count,
                loop_description=state.loop_description,
            )
            return None, True

        if next_action == ThinkNextAction.ADVANCE:
            if not state.in_loop:
                state.last_action_summary = "advance ignored — not in a loop"
                return None, True
            if state.user_facing_actions_since_progress == 0:
                state.last_action_summary = (
                    "advance BLOCKED: No user-facing actions since last advance. Do a user-facing action first."
                )
                return None, True
            state.loop_round += 1
            state.user_facing_actions_since_progress = 0
            if state.loop_count and state.loop_round > state.loop_count:
                state.last_action_summary = f"Loop complete — all {state.loop_count} rounds done"
                append_recent_action(f"[LOOP COMPLETE] {state.loop_count} rounds done")
                self.event_logger.loop_state_changed(
                    change="end",
                    loop_round=state.loop_count,
                    loop_count=state.loop_count,
                    loop_description=state.loop_description,
                )
                exit_loop()
                return None, True
            remaining = state.loop_count - state.loop_round + 1 if state.loop_count else "?"
            state.last_action_summary = (
                f"Advanced to loop round {state.loop_round} of {state.loop_count} ({remaining} remaining)"
            )
            append_recent_action(f"[ADVANCE] Round {state.loop_round} of {state.loop_count}")
            self.event_logger.loop_state_changed(
                change="advance",
                loop_round=state.loop_round,
                loop_count=state.loop_count,
                loop_description=state.loop_description,
            )
            return None, True

        if next_action == ThinkNextAction.END_LOOP:
            if not state.in_loop:
                state.last_action_summary = "end_loop ignored — not in a loop"
                return None, True
            state.last_action_summary = f"Loop ended early at round {state.loop_round} of {state.loop_count}"
            append_recent_action("[LOOP END] early exit")
            self.event_logger.loop_state_changed(
                change="end_early",
                loop_round=state.loop_round,
                loop_count=state.loop_count,
                loop_description=state.loop_description,
            )
            exit_loop()
            return None, True

        if next_action == ThinkNextAction.STUCK:
            replacement_strategy = str(control.hint_message or "").strip() or "Trying a different strategy."
            state.last_action_summary = f"Strategy switch (stuck): \"{replacement_strategy}\""
            append_recent_action("[STUCK] Strategy switch")
            with self._hints_lock:
                self._pending_hints.append(replacement_strategy)
            if self.on_stuck_callback:
                try:
                    hint = self.on_stuck_callback(replacement_strategy, self._current_iteration)
                    if hint:
                        with self._hints_lock:
                            self._pending_hints.append(hint.strip())
                except Exception as e:
                    self.event_logger.system_warning(f"on_stuck_callback failed: {e}")
            return None, True

        if next_action == ThinkNextAction.DONE:
            missing_required = self._missing_required_agent_events()
            if missing_required:
                missing_text = ", ".join(sorted(missing_required))
                state.last_action_summary = (
                    f"done BLOCKED: missing required Agent Events: {missing_text}"
                )
                with self._hints_lock:
                    self._pending_hints.append(
                        "Mission completion blocked by required Agent Events. "
                        f"Emit required events first: {missing_text}."
                    )
                self.event_logger.system_warning(
                    "Mission done blocked by required Agent Events",
                    missing_required=missing_required,
                )
                return None, True
            reasoning = str(control.done_reasoning or "").strip() or "Mission complete"
            state.last_action_summary = f"Mission complete: {reasoning}"
            return reasoning, False

        # CONTINUE (or any future non-terminal action): no controller-side mutation needed.
        return None, False

    @staticmethod
    def _controller_tool_names() -> set[str]:
        return {
            "switch_tab",
            "close_tab",
            "open_tab",
            "dismiss_dialog",
            "send_email",
            "bash",
            "read_file",
            "find_files",
            "read_clipboard",
            "activate_skill",
        }

    def _execute_controller_tool(
        self,
        *,
        tool_name: str,
        action_step: ActionStep,
        action_args: Dict[str, Any],
    ) -> ToolOutcome:
        if tool_name == "switch_tab":
            return self._tool_switch_tab(action_step=action_step, action_args=action_args)
        if tool_name == "close_tab":
            return self._tool_close_tab(action_step=action_step, action_args=action_args)
        if tool_name == "open_tab":
            return self._tool_open_tab(action_step=action_step, action_args=action_args)
        if tool_name == "dismiss_dialog":
            return self._tool_dismiss_dialog(action_step=action_step, action_args=action_args)
        if tool_name == "send_email":
            return self._tool_send_email(action_step=action_step, action_args=action_args)
        if tool_name == "bash":
            return self._tool_bash(action_step=action_step, action_args=action_args)
        if tool_name == "read_file":
            return self._tool_read_file(action_step=action_step, action_args=action_args)
        if tool_name == "find_files":
            return self._tool_find_files(action_step=action_step, action_args=action_args)
        if tool_name == "read_clipboard":
            return self._tool_read_clipboard(action_step=action_step)
        if tool_name == "activate_skill":
            return self._tool_activate_skill(action_step=action_step, action_args=action_args)
        return ToolOutcome(
            output=ToolOutput(
                success=False,
                summary=f"Unsupported controller tool: {tool_name}",
                error=f"Unsupported controller tool: {tool_name}",
            )
        )

    def _tool_switch_tab(self, *, action_step: ActionStep, action_args: Dict[str, Any]) -> ToolOutcome:
        before_state = self.memory_store._capture_current_state()
        action_success = False
        action_error: Optional[str] = None
        tab_id = str(action_args.get("tab_id", "")).strip()
        summary = "switch_tab FAILED: Tab management not available"
        if self.tab_manager:
            try:
                new_page = self.tab_manager.switch_to(tab_id)
                self.action_executor.set_page(new_page)
                title = ""
                try:
                    title = new_page.title()
                except Exception:
                    pass
                summary = f"Switched to tab [{tab_id}]: \"{title}\""
                policy_allowed, policy_warning, current_url = self._enforce_current_page_policy(
                    source="switch_tab"
                )
                if not policy_allowed:
                    action_error = f"{policy_warning} ({current_url})"
                    summary = f"switch_tab FAILED: {action_error}"
                else:
                    if policy_warning:
                        summary += f" | sandbox(observe): {policy_warning}"
                    action_success = True
            except ValueError as e:
                summary = f"switch_tab FAILED: {e}"
                action_error = str(e)
        else:
            action_error = "Tab management not available"

        self._record_controller_action(
            action_type=InteractionType.NAVIGATION.value,
            action_step=action_step,
            success=action_success,
            error_message=action_error,
            action_params={"operation": "switch_tab", "tab_id": tab_id},
            before_state=before_state,
            after_state=self.memory_store._capture_current_state(),
        )
        return ToolOutcome(
            output=ToolOutput(success=action_success, summary=summary, error=action_error),
        )

    def _tool_close_tab(self, *, action_step: ActionStep, action_args: Dict[str, Any]) -> ToolOutcome:
        before_state = self.memory_store._capture_current_state()
        action_success = False
        action_error: Optional[str] = None
        tab_id = str(action_args.get("tab_id", "")).strip()
        summary = "close_tab FAILED: Tab management not available"
        if self.tab_manager:
            try:
                new_page = self.tab_manager.close_tab(tab_id)
                self.action_executor.set_page(new_page)
                active = self.tab_manager.get_active()
                active_id = active.id if active else "?"
                summary = f"Closed tab [{tab_id}]. Now on tab [{active_id}]"
                policy_allowed, policy_warning, current_url = self._enforce_current_page_policy(
                    source="close_tab"
                )
                if not policy_allowed:
                    action_error = f"{policy_warning} ({current_url})"
                    summary = f"close_tab FAILED: {action_error}"
                else:
                    if policy_warning:
                        summary += f" | sandbox(observe): {policy_warning}"
                    action_success = True
            except ValueError as e:
                summary = f"close_tab FAILED: {e}"
                action_error = str(e)
        else:
            action_error = "Tab management not available"

        self._record_controller_action(
            action_type=InteractionType.NAVIGATION.value,
            action_step=action_step,
            success=action_success,
            error_message=action_error,
            action_params={"operation": "close_tab", "tab_id": tab_id},
            before_state=before_state,
            after_state=self.memory_store._capture_current_state(),
        )
        return ToolOutcome(
            output=ToolOutput(success=action_success, summary=summary, error=action_error),
        )

    def _tool_open_tab(self, *, action_step: ActionStep, action_args: Dict[str, Any]) -> ToolOutcome:
        before_state = self.memory_store._capture_current_state()
        action_success = False
        action_error: Optional[str] = None
        url = str(action_args.get("url", "")).strip() or None
        observe_warning: Optional[str] = None
        summary = "open_tab FAILED: Tab management not available"
        if url:
            url_decision = self.sandbox_policy.check_url(url)
            if not url_decision.allowed:
                warning = f"open_tab blocked by sandbox: {url_decision.reason}"
                self.event_logger.system_warning(warning)
                if self.sandbox_policy.enforce:
                    action_error = warning
                    summary = f"open_tab FAILED: {url_decision.reason}"
                else:
                    observe_warning = warning

        if action_error is None and self.tab_manager:
            try:
                new_page = self.tab_manager.open_tab(url)
                self.action_executor.set_page(new_page)
                active = self.tab_manager.get_active()
                active_id = active.id if active else "?"
                summary = f"Opened new tab [{active_id}]"
                if url:
                    summary += f" at {url}"
                if observe_warning:
                    summary += f" | sandbox(observe): {observe_warning}"
                policy_allowed, policy_warning, current_url = self._enforce_current_page_policy(
                    source="open_tab"
                )
                if not policy_allowed:
                    action_error = f"{policy_warning} ({current_url})"
                    summary = f"open_tab FAILED: {action_error}"
                else:
                    if policy_warning:
                        summary += f" | sandbox(observe): {policy_warning}"
                    action_success = True
            except Exception as e:
                summary = f"open_tab FAILED: {e}"
                action_error = str(e)
        elif action_error is None:
            action_error = "Tab management not available"

        self._record_controller_action(
            action_type=InteractionType.NAVIGATION.value,
            action_step=action_step,
            success=action_success,
            error_message=action_error,
            action_params={"operation": "open_tab", "url": url},
            before_state=before_state,
            after_state=self.memory_store._capture_current_state(),
        )
        return ToolOutcome(
            output=ToolOutput(success=action_success, summary=summary, error=action_error),
        )

    def _tool_dismiss_dialog(self, *, action_step: ActionStep, action_args: Dict[str, Any]) -> ToolOutcome:
        before_state = self.memory_store._capture_current_state()
        action_success = False
        action_error: Optional[str] = None
        accept = bool(action_args.get("accept", False))
        input_text_raw = action_args.get("input_text")
        input_text = str(input_text_raw).strip() if input_text_raw not in (None, "") else None
        if self.tab_manager and self.tab_manager.pending_dialog:
            self.tab_manager.dismiss_dialog(accept, input_text)
            action_word = "accepted" if accept else "dismissed"
            summary = f"Dialog {action_word}"
            action_success = True
        else:
            summary = "dismiss_dialog: No dialog pending"
            action_error = "No dialog pending"

        self._record_controller_action(
            action_type="dismiss_dialog",
            action_step=action_step,
            success=action_success,
            error_message=action_error,
            action_params={"accept": accept, "input_text": input_text},
            before_state=before_state,
            after_state=self.memory_store._capture_current_state(),
        )
        return ToolOutcome(
            output=ToolOutput(success=action_success, summary=summary, error=action_error),
        )

    def _tool_send_email(self, *, action_step: ActionStep, action_args: Dict[str, Any]) -> ToolOutcome:
        before_state = self.memory_store._capture_current_state()
        action_success = False
        action_error: Optional[str] = None
        duplicate_of: Optional[str] = None
        message_id: Optional[str] = None
        summary = "send_email FAILED"

        raw_to = action_args.get("to")
        to_list = _parse_email_recipients(raw_to)
        subject = str(action_args.get("subject", "")).strip()
        body = str(action_args.get("body", "")).strip()
        body_preview = body if len(body) <= 200 else f"{body[:197]}..."
        body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest() if body else ""
        effective_from_email = _resolve_resend_from_email()

        canonical_to = sorted({email.lower() for email in to_list})
        signature_source = f"{'|'.join(canonical_to)}\n{subject.lower()}\n{body}"
        email_signature = hashlib.sha256(signature_source.encode("utf-8")).hexdigest()

        for entry in reversed(self.memory_store.entries):
            if entry.action_type != "send_email":
                continue
            if entry.outcome not in {"success", "no_change"}:
                continue
            if str(entry.action_params.get("email_signature", "")).strip() == email_signature:
                duplicate_of = entry.memory_id
                action_success = True
                summary = (
                    f"send_email skipped: identical email already sent ({duplicate_of}). "
                    "If the mission was only this email, call think(next_action=done)."
                )
                break

        try:
            if duplicate_of is None:
                api_key = str(os.environ.get("RESEND_API_KEY", "")).strip()
                if not api_key:
                    action_error = "RESEND_API_KEY is not set"
                    summary = f"send_email FAILED: {action_error}"
                elif "@" not in effective_from_email:
                    action_error = (
                        "RESEND_FROM_EMAIL is invalid; expected an email address "
                        "or display-name format like 'Team <team@example.com>'"
                    )
                    summary = f"send_email FAILED: {action_error}"
                elif not to_list:
                    action_error = "No recipients (to) provided"
                    summary = f"send_email FAILED: {action_error}"
                elif not subject:
                    action_error = "Subject is required"
                    summary = f"send_email FAILED: {action_error}"
                elif not body:
                    action_error = "Body is required"
                    summary = f"send_email FAILED: {action_error}"
                else:
                    try:
                        import resend
                    except ImportError:
                        action_error = (
                            "resend package is not installed in the active Python environment"
                        )
                        summary = f"send_email FAILED: {action_error}"
                    else:
                        resend.api_key = api_key
                        params = {
                            "from": effective_from_email,
                            "to": to_list,
                            "subject": subject,
                            "html": body,
                        }
                        send_result = resend.Emails.send(params)
                        raw_message_id = (
                            send_result.get("id")
                            if isinstance(send_result, dict)
                            else getattr(send_result, "id", None)
                        )
                        if raw_message_id:
                            message_id = str(raw_message_id).strip()
                        summary = (
                            f"Email sent to {', '.join(to_list)}: \"{subject}\" "
                            f"| body=\"{body_preview}\""
                        )
                        if message_id:
                            summary += f" | message_id={message_id}"
                        action_success = True
        except Exception as e:
            action_error = _format_resend_error(e)
            summary = f"send_email FAILED: {action_error}"

        self._record_controller_action(
            action_type="send_email",
            action_step=action_step,
            success=action_success,
            error_message=action_error,
            action_params={
                "operation": "send_email",
                "to": to_list,
                "subject": subject,
                "body_preview": body_preview,
                "body_hash": body_hash,
                "from_email": effective_from_email,
                "email_signature": email_signature,
                "duplicate_of": duplicate_of,
                "message_id": message_id,
            },
            before_state=before_state,
            after_state=self.memory_store._capture_current_state(),
        )
        if action_success:
            with self._hints_lock:
                self._pending_hints.append(
                    "You already sent the requested email. Do not send it again. "
                    "If the mission is complete, call think(next_action=done)."
                )
        return ToolOutcome(
            output=ToolOutput(
                success=action_success,
                summary=summary,
                error=action_error,
                data={
                    "to": to_list,
                    "subject": subject,
                    "message_id": message_id,
                    "duplicate_of": duplicate_of,
                },
            ),
        )

    def _tool_bash(self, *, action_step: ActionStep, action_args: Dict[str, Any]) -> ToolOutcome:
        import subprocess

        before_state = self.memory_store._capture_current_state()
        command = str(action_args.get("command", "")).strip()
        command_timeout = self.sandbox_policy.command_timeout_seconds()
        action_success = False
        action_error: Optional[str] = None
        observe_warning: Optional[str] = None
        summary = "bash FAILED: No command provided"

        if not command:
            action_error = "No command provided"
        else:
            command_decision = self.sandbox_policy.check_command(command)
            if not command_decision.allowed:
                warning = f"bash blocked by sandbox: {command_decision.reason}"
                self.event_logger.system_warning(warning)
                if self.sandbox_policy.enforce:
                    action_error = warning
                    summary = f"bash FAILED: {command_decision.reason}"
                else:
                    observe_warning = warning

            if action_error is None:
                try:
                    proc = subprocess.run(
                        ["bash", "-lc", command],
                        capture_output=True,
                        text=True,
                        timeout=command_timeout,
                    )
                    stdout = (proc.stdout or "").rstrip()
                    stderr = (proc.stderr or "").rstrip()
                    exit_code = proc.returncode

                    parts = [f"bash: `{command}`", f"exit_code={exit_code}"]
                    if stdout:
                        preview = stdout if len(stdout) <= 2000 else f"{stdout[:2000]}\n... (truncated)"
                        parts.append(f"stdout:\n{preview}")
                    else:
                        parts.append("stdout: (no output)")
                    if stderr:
                        parts.append(f"stderr: {stderr[:500]}")
                    summary = "\n".join(parts)
                    if observe_warning:
                        summary += f"\nsandbox(observe): {observe_warning}"
                    action_success = True
                except subprocess.TimeoutExpired:
                    action_error = f"Command timed out after {command_timeout}s"
                    summary = f"bash FAILED: {action_error}"
                except Exception as e:
                    action_error = str(e)
                    summary = f"bash FAILED: {action_error}"

        self._record_controller_action(
            action_type="bash",
            action_step=action_step,
            success=action_success,
            error_message=action_error,
            action_params={"command": command},
            before_state=before_state,
            after_state=self.memory_store._capture_current_state(),
        )
        return ToolOutcome(
            output=ToolOutput(
                success=action_success,
                summary=summary,
                error=action_error,
                data={"operation": "bash", "command": command},
            ),
        )

    def _tool_read_file(self, *, action_step: ActionStep, action_args: Dict[str, Any]) -> ToolOutcome:
        from pathlib import Path

        before_state = self.memory_store._capture_current_state()
        path_arg = str(action_args.get("path", "")).strip()
        start_line_raw = action_args.get("start_line")
        end_line_raw = action_args.get("end_line")
        start_line: Optional[int] = None
        end_line: Optional[int] = None
        action_success = False
        action_error: Optional[str] = None
        observe_warning: Optional[str] = None
        summary = "read_file FAILED: No path provided"

        if start_line_raw is not None:
            try:
                start_line = max(1, int(start_line_raw))
            except (TypeError, ValueError):
                start_line = None
        if end_line_raw is not None:
            try:
                end_line = max(1, int(end_line_raw))
            except (TypeError, ValueError):
                end_line = None

        if not path_arg:
            action_error = "No path provided"
        else:
            try:
                resolved = Path(path_arg).expanduser().resolve()
                path_decision = self.sandbox_policy.check_path(resolved, operation="read")
                if not path_decision.allowed:
                    warning = f"read_file blocked by sandbox: {path_decision.reason}"
                    self.event_logger.system_warning(warning)
                    if self.sandbox_policy.enforce:
                        action_error = warning
                        summary = f"read_file FAILED: {path_decision.reason}"
                    else:
                        observe_warning = warning
                if action_error is not None:
                    raise RuntimeError(action_error)

                lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
                total_lines = len(lines)

                if start_line and end_line and start_line > end_line:
                    action_error = f"Invalid line range: start_line={start_line} > end_line={end_line}"
                    summary = f"read_file FAILED: {action_error}"
                else:
                    start_idx = (start_line - 1) if start_line else 0
                    end_idx = end_line if end_line else total_lines
                    start_idx = min(max(start_idx, 0), total_lines)
                    end_idx = min(max(end_idx, start_idx), total_lines)
                    content = "\n".join(lines[start_idx:end_idx])
                    if len(content) > 4000:
                        content = f"{content[:4000]}\n... (truncated)"

                    if start_line or end_line:
                        range_note = f" (lines {start_idx + 1}-{end_idx} of {total_lines})"
                    else:
                        range_note = f" ({total_lines} lines)"

                    summary = f"read_file: {resolved}{range_note}\n{content}"
                    if observe_warning:
                        summary += f"\nsandbox(observe): {observe_warning}"
                    action_success = True
            except FileNotFoundError:
                action_error = f"File not found: {path_arg}"
                summary = f"read_file FAILED: {action_error}"
            except Exception as e:
                action_error = str(e)
                summary = f"read_file FAILED: {action_error}"

        self._record_controller_action(
            action_type="read_file",
            action_step=action_step,
            success=action_success,
            error_message=action_error,
            action_params={
                "path": path_arg,
                "start_line": start_line,
                "end_line": end_line,
            },
            before_state=before_state,
            after_state=self.memory_store._capture_current_state(),
        )
        return ToolOutcome(
            output=ToolOutput(
                success=action_success,
                summary=summary,
                error=action_error,
                data={
                    "operation": "read_file",
                    "path": path_arg,
                    "start_line": start_line,
                    "end_line": end_line,
                },
            ),
        )

    def _tool_find_files(self, *, action_step: ActionStep, action_args: Dict[str, Any]) -> ToolOutcome:
        from pathlib import Path

        before_state = self.memory_store._capture_current_state()
        pattern = str(action_args.get("pattern", "")).strip()
        directory = str(action_args.get("directory", "~")).strip() or "~"
        recursive = bool(action_args.get("recursive", True))
        action_success = False
        action_error: Optional[str] = None
        observe_warning: Optional[str] = None
        summary = "find_files FAILED: No pattern provided"

        if not pattern:
            action_error = "No pattern provided"
        else:
            try:
                root = Path(directory).expanduser().resolve()
                path_decision = self.sandbox_policy.check_path(root, operation="find")
                if not path_decision.allowed:
                    warning = f"find_files blocked by sandbox: {path_decision.reason}"
                    self.event_logger.system_warning(warning)
                    if self.sandbox_policy.enforce:
                        action_error = warning
                        summary = f"find_files FAILED: {path_decision.reason}"
                    else:
                        observe_warning = warning
                if action_error is not None:
                    raise RuntimeError(action_error)

                glob_fn = root.rglob if recursive else root.glob
                max_results = 50
                found_matches: List[str] = []
                for path in glob_fn(pattern):
                    found_matches.append(str(path))
                    if len(found_matches) > max_results:
                        break

                has_more = len(found_matches) > max_results
                matches = sorted(found_matches[:max_results])
                if matches:
                    listing = "\n".join(matches)
                    note = (
                        f" (showing first {len(matches)}; more matches exist)"
                        if has_more
                        else f" ({len(matches)} found)"
                    )
                    summary = f"find_files: `{pattern}` in {root}{note}\n{listing}"
                else:
                    summary = f"find_files: `{pattern}` in {root} - no matches found"
                if observe_warning:
                    summary += f"\nsandbox(observe): {observe_warning}"
                action_success = True
            except Exception as e:
                action_error = str(e)
                summary = f"find_files FAILED: {action_error}"

        self._record_controller_action(
            action_type="find_files",
            action_step=action_step,
            success=action_success,
            error_message=action_error,
            action_params={
                "pattern": pattern,
                "directory": directory,
                "recursive": recursive,
            },
            before_state=before_state,
            after_state=self.memory_store._capture_current_state(),
        )
        return ToolOutcome(
            output=ToolOutput(
                success=action_success,
                summary=summary,
                error=action_error,
                data={
                    "operation": "find_files",
                    "pattern": pattern,
                    "directory": directory,
                },
            ),
        )

    def _tool_read_clipboard(self, *, action_step: ActionStep) -> ToolOutcome:
        import subprocess
        import sys

        before_state = self.memory_store._capture_current_state()
        action_success = False
        action_error: Optional[str] = None
        content = ""
        observe_warning: Optional[str] = None
        summary = "read_clipboard FAILED"

        clipboard_decision = self.sandbox_policy.check_clipboard_read()
        if not clipboard_decision.allowed:
            warning = f"read_clipboard blocked by sandbox: {clipboard_decision.reason}"
            self.event_logger.system_warning(warning)
            if self.sandbox_policy.enforce:
                action_error = warning
                summary = f"read_clipboard FAILED: {clipboard_decision.reason}"
            else:
                observe_warning = warning

        try:
            if action_error is not None:
                raise RuntimeError(action_error)
            if sys.platform == "darwin":
                proc = subprocess.run(
                    ["pbpaste"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if proc.returncode != 0:
                    stderr = (proc.stderr or "").strip() or "unknown error"
                    raise RuntimeError(f"pbpaste failed: {stderr}")
                content = proc.stdout
            elif sys.platform.startswith("linux"):
                any_tool_found = False
                for cmd in (
                    ["xclip", "-selection", "clipboard", "-o"],
                    ["xsel", "--clipboard", "--output"],
                ):
                    try:
                        proc = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        any_tool_found = True
                        if proc.returncode == 0:
                            content = proc.stdout
                            break
                    except FileNotFoundError:
                        continue
                else:
                    msg = (
                        "Clipboard read failed (xclip/xsel returned non-zero)"
                        if any_tool_found
                        else "No clipboard tool found (install xclip or xsel)"
                    )
                    raise RuntimeError(msg)
            elif sys.platform == "win32":
                proc = subprocess.run(
                    ["powershell", "-command", "Get-Clipboard"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if proc.returncode != 0:
                    stderr = (proc.stderr or "").strip() or "unknown error"
                    raise RuntimeError(f"Get-Clipboard failed: {stderr}")
                content = proc.stdout
            else:
                raise RuntimeError(f"Unsupported platform: {sys.platform}")

            content = content.rstrip()
            if content:
                preview = content if len(content) <= 1000 else f"{content[:1000]}\n... (truncated)"
                summary = f"read_clipboard: {len(content)} chars\n{preview}"
            else:
                summary = "read_clipboard: clipboard is empty"
            if observe_warning:
                summary += f"\nsandbox(observe): {observe_warning}"
            action_success = True
        except Exception as e:
            action_error = str(e)
            summary = f"read_clipboard FAILED: {action_error}"

        self._record_controller_action(
            action_type="read_clipboard",
            action_step=action_step,
            success=action_success,
            error_message=action_error,
            action_params={"content_length": len(content)},
            before_state=before_state,
            after_state=self.memory_store._capture_current_state(),
        )
        return ToolOutcome(
            output=ToolOutput(
                success=action_success,
                summary=summary,
                error=action_error,
                data={"operation": "read_clipboard"},
            ),
        )

    def _tool_activate_skill(self, *, action_step: ActionStep, action_args: Dict[str, Any]) -> ToolOutcome:
        before_state = self.memory_store._capture_current_state()
        requested_name = str(action_args.get("skill_name", "")).strip()
        reasoning_text = str(action_args.get("reasoning", "") or "").strip()
        action_success = False
        action_error: Optional[str] = None
        observe_warning: Optional[str] = None
        summary = "activate_skill FAILED: No skill name provided"
        failure_code = "missing_name"

        activated_skill_name: Optional[str] = None
        activated_skill_path: Optional[str] = None
        body_char_count = 0
        load_ms = 0.0
        active_before = (
            str(getattr(self.execution_state, "active_skill_name", "") or "").strip()
            if self.execution_state is not None
            else ""
        )
        max_body_chars = max(
            0,
            int(getattr(getattr(self.config, "skills", None), "max_body_chars", 12000) or 0),
        )
        self.event_logger.skill_activation_requested(
            requested_skill_name=requested_name or "(missing)",
            iteration=int(self._current_iteration or 0),
            action_id=None,
            reasoning_present=bool(reasoning_text),
        )

        if not requested_name:
            action_error = "No skill name provided"
        else:
            skill = self._find_skill_by_name(requested_name)
            if skill is None:
                failure_code = "not_found"
                available = [item.name for item in self.available_skills]
                if available:
                    preview = ", ".join(available[:10])
                    extra = "..." if len(available) > 10 else ""
                    action_error = f"Unknown skill '{requested_name}'. Available: {preview}{extra}"
                else:
                    action_error = f"Unknown skill '{requested_name}'. No skills discovered."
                summary = f"activate_skill FAILED: {action_error}"
            else:
                skill_file = skill.path / "SKILL.md"
                path_decision = self.sandbox_policy.check_path(skill_file, operation="read")
                self.event_logger.skill_resource_accessed(
                    skill_name=skill.name,
                    relative_path="SKILL.md",
                    absolute_path=str(skill_file),
                    allowed=bool(path_decision.allowed),
                    sandbox_reason=str(path_decision.reason or ""),
                    operation="activate_skill",
                )
                if not path_decision.allowed:
                    failure_code = "sandbox_blocked"
                    warning = f"activate_skill blocked by sandbox: {path_decision.reason}"
                    self.event_logger.system_warning(warning)
                    if self.sandbox_policy.enforce:
                        action_error = warning
                        summary = f"activate_skill FAILED: {path_decision.reason}"
                    else:
                        observe_warning = warning
                if action_error is None:
                    try:
                        load_started = time.perf_counter()
                        body, body_stats = load_skill_body(
                            skill,
                            max_chars=(max_body_chars or None),
                            return_stats=True,
                        )
                        load_ms = (time.perf_counter() - load_started) * 1000.0
                        body_char_count = len(body)
                        original_chars = int(body_stats.get("original_chars", body_char_count) or body_char_count)
                        retained_chars = int(body_stats.get("retained_chars", body_char_count) or body_char_count)
                        was_truncated = bool(body_stats.get("truncated", False))
                        if self.execution_state is not None:
                            self.execution_state.active_skill_name = skill.name
                            self.execution_state.active_skill_body = body
                        activated_skill_name = skill.name
                        activated_skill_path = str(skill.path)
                        summary = f"Activated skill '{skill.name}' from {skill.path}"
                        if was_truncated:
                            summary += f" (body capped at {max_body_chars} chars)"
                        if observe_warning:
                            summary += f" | sandbox(observe): {observe_warning}"
                        self.event_logger.skill_activation_succeeded(
                            skill_name=skill.name,
                            skill_path=str(skill.path),
                            body_chars_loaded=body_char_count,
                            max_body_chars=max_body_chars,
                            truncated=was_truncated,
                            load_ms=load_ms,
                            iteration=int(self._current_iteration or 0),
                        )
                        if was_truncated:
                            self.event_logger.skill_context_truncated(
                                skill_name=skill.name,
                                original_chars=original_chars,
                                cap_chars=max_body_chars,
                                retained_chars=retained_chars,
                                iteration=int(self._current_iteration or 0),
                            )
                        if active_before and active_before != skill.name:
                            self.event_logger.skill_switched(
                                from_skill=active_before,
                                to_skill=skill.name,
                                iteration=int(self._current_iteration or 0),
                            )
                        action_success = True
                    except Exception as e:
                        failure_code = "read_error"
                        action_error = str(e)
                        summary = f"activate_skill FAILED: {action_error}"

        if not action_success:
            self.event_logger.skill_activation_failed(
                requested_skill_name=requested_name or "(missing)",
                failure_code=failure_code,
                error=str(action_error or "unknown error"),
                available_skills_count=len(self.available_skills),
                iteration=int(self._current_iteration or 0),
            )

        self._record_controller_action(
            action_type="activate_skill",
            action_step=action_step,
            success=action_success,
            error_message=action_error,
            action_params={
                "requested_skill_name": requested_name,
                "activated_skill_name": activated_skill_name,
                "skill_path": activated_skill_path,
                "max_body_chars": max_body_chars,
                "load_ms": round(load_ms, 3),
            },
            before_state=before_state,
            after_state=self.memory_store._capture_current_state(),
        )
        return ToolOutcome(
            output=ToolOutput(
                success=action_success,
                summary=summary,
                error=action_error,
                data={
                    "operation": "activate_skill",
                    "requested_skill_name": requested_name,
                    "activated_skill_name": activated_skill_name,
                    "skill_path": activated_skill_path,
                    "body_chars": body_char_count,
                    "max_body_chars": max_body_chars,
                    "load_ms": load_ms,
                },
            ),
        )

    def pause(self, message: str = "Paused") -> None:
        with self._pause_lock:
            self._paused = True
            self._pause_message = message
            self._pause_event.clear()

    def resume(self) -> None:
        with self._pause_lock:
            self._paused = False
            self._pause_event.set()

    def cancel(self) -> None:
        self._cancel_event.set()
        # Unblock wait() if currently paused so cancellation can be observed immediately.
        with self._pause_lock:
            self._paused = False
            self._pause_event.set()

    def inject_hint(self, text: str) -> None:
        hint = (text or "").strip()
        if not hint:
            return
        with self._hints_lock:
            self._pending_hints.append(hint)

    def get_state_snapshot(self) -> Dict[str, Any]:
        with self._hints_lock:
            pending_hints = list(self._pending_hints)
        with self._pause_lock:
            paused = self._paused
            pause_message = self._pause_message

        return {
            "paused": paused,
            "pause_message": pause_message,
            "cancel_requested": self._cancel_event.is_set(),
            "resume_loaded": self.has_loaded_checkpoint,
            "loaded_run_id": self.loaded_resume_run_id,
            "current_iteration": self._current_iteration,
            "pending_hints": pending_hints,
            "execution_state": copy.deepcopy(self.execution_state),
            "mission_result": copy.deepcopy(self.mission_result),
            "llm_total_cost_usd": self.event_logger.total_cost_usd,
            "llm_total_tokens": self.event_logger.total_tokens,
        }

    def get_screenshot_bytes(self, screenshot_id: str) -> Optional[bytes]:
        return self.screenshot_store.get_bytes(screenshot_id)

    def get_screenshot_meta(self, screenshot_id: str) -> Optional[ScreenshotMeta]:
        return self.screenshot_store.get_meta(screenshot_id)

    def list_recent_screenshots(self, limit: int = 20) -> List[ScreenshotMeta]:
        return self.screenshot_store.list_recent(limit)

    def get_latest_screenshot_bytes(self) -> Optional[bytes]:
        return self.screenshot_store.get_latest_bytes()

    def get_latest_screenshot_meta(self) -> Optional[ScreenshotMeta]:
        return self.screenshot_store.get_latest_meta()

    def clear_screenshot_cache(self) -> None:
        self.screenshot_store.clear()

    @property
    def has_loaded_checkpoint(self) -> bool:
        """True when `load_checkpoint()` has prepared a resumable mission state."""
        return bool(self._resume_checkpoint_loaded and self._loaded_resume_mission)

    @property
    def loaded_resume_run_id(self) -> str:
        """Run id currently loaded for resume (empty when none)."""
        return str(self._loaded_resume_run_id or "")

    @property
    def loaded_resume_mission(self) -> str:
        """Mission text currently loaded for resume (empty when none)."""
        return str(self._loaded_resume_mission or "")

    @staticmethod
    def _serialize_failed_actions(failed_elements: Optional[List[FailedAction]]) -> List[Dict[str, Any]]:
        serialized: List[Dict[str, Any]] = []
        for item in (failed_elements or []):
            try:
                if hasattr(item, "model_dump"):
                    serialized.append(item.model_dump(mode="python"))
                elif isinstance(item, dict):
                    serialized.append(dict(item))
            except Exception:
                continue
        return serialized

    @staticmethod
    def _deserialize_failed_actions(raw_items: Any) -> List[FailedAction]:
        restored: List[FailedAction] = []
        if not isinstance(raw_items, list):
            return restored
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                if hasattr(FailedAction, "model_validate"):
                    restored.append(FailedAction.model_validate(item))
                else:  # pragma: no cover - legacy fallback
                    restored.append(FailedAction.parse_obj(item))
            except Exception:
                continue
        return restored

    @classmethod
    def _execution_state_from_payload(cls, payload: Any) -> ExecutionState:
        source = payload if isinstance(payload, dict) else {}
        allowed = set(ExecutionState.__dataclass_fields__.keys())
        clean: Dict[str, Any] = {
            key: value
            for key, value in source.items()
            if key in allowed and key != "failed_elements"
        }
        clean["failed_elements"] = cls._deserialize_failed_actions(source.get("failed_elements", []))
        try:
            return ExecutionState(**clean)
        except Exception:
            return ExecutionState()

    @staticmethod
    def _execution_state_to_payload(state: Optional[ExecutionState]) -> Dict[str, Any]:
        if state is None:
            return {}
        payload = copy.deepcopy(getattr(state, "__dict__", {}))
        payload["failed_elements"] = Agent._serialize_failed_actions(
            getattr(state, "failed_elements", None)
        )
        return payload

    @staticmethod
    def _mission_result_from_payload(payload: Any) -> MissionResult:
        source = payload if isinstance(payload, dict) else {}
        allowed = set(MissionResult.__dataclass_fields__.keys())
        clean = {key: value for key, value in source.items() if key in allowed}
        try:
            return MissionResult(**clean)
        except Exception:
            return MissionResult()

    @staticmethod
    def _mission_result_to_payload(result: MissionResult) -> Dict[str, Any]:
        try:
            return asdict(result)
        except Exception:
            return {}

    def _build_resume_checkpoint_payload(
        self,
        *,
        mission: str,
        state: Optional[ExecutionState],
        status: str,
    ) -> Dict[str, Any]:
        with self._hints_lock:
            pending_hints = list(self._pending_hints)
        with self._pause_lock:
            paused = bool(self._paused)
            pause_message = str(self._pause_message or "Paused")
        return {
            "schema_version": 1,
            "captured_at": time.time(),
            "status": str(status or "running"),
            "agent_id": str(self.agent_workspace.agent_id or ""),
            "run_id": str(self._active_run_id or ""),
            "mission": str(mission or ""),
            "current_iteration": int(self._current_iteration or 0),
            "mission_start_url": str(self.mission_start_url or ""),
            "mission_start_time": float(self.mission_start_time or 0.0),
            "execution_state": self._execution_state_to_payload(state),
            "mission_result": self._mission_result_to_payload(self.mission_result),
            "memory": self.memory_store.to_payload() if hasattr(self, "memory_store") else {},
            "llm_totals": {
                "total_cost_usd": float(self.event_logger.total_cost_usd or 0.0),
                "total_tokens": int(self.event_logger.total_tokens or 0),
            },
            "pause": {
                "paused": paused,
                "message": pause_message,
            },
            "cancel_requested": bool(self._cancel_event.is_set()),
            "pending_hints": pending_hints,
        }

    def _persist_resume_checkpoint(
        self,
        *,
        mission: str,
        state: Optional[ExecutionState],
        status: str = "running",
    ) -> None:
        """
        Best-effort checkpoint persistence.

        We intentionally write at iteration boundaries so crash recovery resumes from
        the last completed iteration.
        """
        if not self.agent_workspace.current_run_root:
            return
        payload = self._build_resume_checkpoint_payload(
            mission=mission,
            state=state,
            status=status,
        )
        wrote = self.workspace_manager.write_run_checkpoint(self.agent_workspace, payload)
        if not wrote:
            self.event_logger.system_warning(
                "Failed to persist mission checkpoint",
                run_id=self._active_run_id,
                iteration=self._current_iteration,
            )

    def load_checkpoint(
        self,
        *,
        run_id: Optional[str] = None,
        prefer_active: bool = True,
    ) -> Tuple[bool, str]:
        """
        Load a persisted mission checkpoint into memory.

        After loading, the agent is paused and ready for `resume_loaded_mission()`.
        """
        if not getattr(self, "started", False):
            return False, "Agent must be started before loading a checkpoint."
        if not hasattr(self, "memory_store"):
            return False, "Agent memory is not initialized."

        resolved_run_id = self.workspace_manager.resolve_resume_run_id(
            self.agent_workspace,
            requested_run_id=run_id,
            prefer_active=prefer_active,
        )
        if not resolved_run_id:
            return False, "No runs found for this agent."

        run_root = self.workspace_manager.attach_existing_run(
            self.agent_workspace,
            run_id=resolved_run_id,
        )
        if run_root is None:
            return False, f"Run '{resolved_run_id}' was not found."

        payload = self.workspace_manager.read_run_checkpoint(
            self.agent_workspace,
            run_id=resolved_run_id,
        )
        if not payload:
            return False, f"Run '{resolved_run_id}' has no checkpoint yet."

        mission = str(payload.get("mission", "") or "").strip()
        if not mission:
            return False, f"Checkpoint for run '{resolved_run_id}' is missing mission text."

        self._current_iteration = int(payload.get("current_iteration", 0) or 0)
        self.mission_start_url = str(payload.get("mission_start_url", "") or "")
        self.mission_start_time = float(payload.get("mission_start_time", 0.0) or 0.0) or None
        self.execution_state = self._execution_state_from_payload(payload.get("execution_state", {}))
        self.mission_result = self._mission_result_from_payload(payload.get("mission_result", {}))

        memory_payload = payload.get("memory", {})
        if isinstance(memory_payload, dict):
            self.memory_store.load_payload(memory_payload)
        else:
            self.memory_store.start_mission(mission)

        llm_totals = payload.get("llm_totals", {}) if isinstance(payload.get("llm_totals", {}), dict) else {}
        self.event_logger.set_usage_totals(
            total_cost_usd=float(llm_totals.get("total_cost_usd", 0.0) or 0.0),
            total_tokens=int(llm_totals.get("total_tokens", 0) or 0),
        )

        with self._hints_lock:
            self._pending_hints = [
                str(item).strip()
                for item in (payload.get("pending_hints", []) or [])
                if str(item).strip()
            ]
        with self._pause_lock:
            self._paused = True
            self._pause_message = "Loaded from checkpoint. Press Resume to continue."
            self._pause_event.clear()
        self._cancel_event.clear()

        # Clear per-page caches to avoid stale derived state after restore.
        self._cached_snapshot = None
        self._cached_snapshot_fingerprint = None
        self._cached_page_info = None
        self._cached_detected_elements = None
        self._reset_agent_event_tracking()
        self._reset_speculative_state()

        self._loaded_resume_run_id = resolved_run_id
        self._loaded_resume_mission = mission
        self._resume_checkpoint_loaded = True
        self._active_run_id = resolved_run_id
        self._active_run_open = False

        self.event_logger.system_info(
            "Checkpoint loaded",
            agent_id=self.agent_workspace.agent_id,
            run_id=resolved_run_id,
            current_iteration=self._current_iteration,
        )
        return True, (
            f"Loaded {self.agent_workspace.agent_id} / {resolved_run_id} at "
            f"iteration {self._current_iteration}."
        )

    def resume_loaded_mission(self) -> MissionResult:
        """Continue a mission from a loaded checkpoint."""
        if not self.has_loaded_checkpoint:
            return MissionResult(
                success=False,
                reasoning="No checkpoint is loaded.",
                narrative="No checkpoint is loaded.",
            )

        mission = str(self._loaded_resume_mission or "").strip()
        if not mission:
            return MissionResult(
                success=False,
                reasoning="Loaded checkpoint has no mission.",
                narrative="Loaded checkpoint has no mission.",
            )
        run_id = str(self._loaded_resume_run_id or "").strip()
        if not run_id:
            return MissionResult(
                success=False,
                reasoning="Loaded checkpoint has no run id.",
                narrative="Loaded checkpoint has no run id.",
            )

        # Register pre-configured interceptors before execution (same flow as execute_mission()).
        for interceptor_data in self.interceptor_stack:
            self.register_interceptor(
                trigger=interceptor_data["trigger"],
                mode=interceptor_data["mode"],
                handler=interceptor_data["handler"],
            )

        try:
            run_root = self.workspace_manager.attach_existing_run(
                self.agent_workspace,
                run_id=run_id,
            )
            if run_root is None:
                return MissionResult(
                    success=False,
                    reasoning=f"Run '{run_id}' no longer exists.",
                    narrative="Run is missing.",
                )

            self._detach_run_event_log_sink()
            self._active_run_id = run_id
            self._active_run_open = True
            self._attach_run_event_log_sink()
            if self.config.sandbox.audit.enabled:
                self.sandbox_policy.set_audit_log_path(self.agent_workspace.sandbox_audit_path)
                self.effect_policy.set_audit_log_path(self.agent_workspace.sandbox_audit_path)
            else:
                self.sandbox_policy.set_audit_log_path(None)
                self.effect_policy.set_audit_log_path(None)

            self._cancel_event.clear()
            with self._pause_lock:
                self._paused = False
                self._pause_event.set()
            if self.mission_start_time is None:
                self.mission_start_time = time.time()
            self.memory_store.start_mission(mission)
            self.execution_timer.start_mission()
            self.event_logger.system_info(
                "Resuming mission from checkpoint",
                agent_id=self.agent_workspace.agent_id,
                run_id=run_id,
                current_iteration=self._current_iteration,
            )
            try:
                self._discover_available_skills()
            except Exception as e:
                self.available_skills = []
                self.available_skills_catalog = ""
                self._available_skills_by_key = {}
                self.event_logger.system_warning(f"Skill discovery failed: {e}")

            resumed_state = copy.deepcopy(self.execution_state) if self.execution_state else None
            self._reset_speculative_state()
            mission_result = self._run_execution_loop(
                mission,
                existing_state=resumed_state,
            )
            if self.execution_timer.mission_start_time is not None:
                self.execution_timer.end_mission()
            self.mission_result = mission_result
        finally:
            # Best-effort guard so per-run sink never leaks across missions.
            self._detach_run_event_log_sink()
            self.sandbox_policy.set_audit_log_path(None)
            self.effect_policy.set_audit_log_path(None)

        self.event_logger.agent_complete(self.mission_result.success, self.mission_result.reasoning)
        self._resume_checkpoint_loaded = False
        return self.mission_result

    def _apply_workspace_paths(self) -> None:
        """Route default storage paths into this agent's workspace."""
        ws = self.agent_workspace
        self.config.browser.user_data_dir = str(ws.browser_profile_dir)
        self.config.browser.downloads_path = str(ws.browser_downloads_dir)
        self.config.debug.screenshot_dir = str(ws.screenshots_dir)
        self.config.debug.screenshot_stream_dir = str(ws.stream_screenshots_dir)

        for path in (
            ws.workspace_root,
            ws.written_data_dir,
            ws.browser_profile_dir,
            ws.browser_downloads_dir,
            ws.screenshots_dir,
            ws.stream_screenshots_dir,
            ws.runs_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _list_workspace_files(self, max_files: int = 50) -> list:
        """List files in the workspace root (non-recursive). Returns filenames only."""
        try:
            root = self.agent_workspace.workspace_root
            if not root.is_dir():
                return []
            files = sorted(
                (p.name for p in root.iterdir() if p.is_file()),
            )
            return files[:max_files]
        except Exception:
            return []

    def _attach_run_event_log_sink(self) -> None:
        """Stream event logger output into run-scoped JSONL."""
        self._detach_run_event_log_sink()
        event_log_path = self.agent_workspace.run_event_log_path
        if not event_log_path:
            return
        try:
            event_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._run_event_log_handle = event_log_path.open("a", encoding="utf-8")
        except Exception:
            self._run_event_log_handle = None
            return

        def _sink(event: Any) -> None:
            try:
                if hasattr(event, "to_dict"):
                    payload = event.to_dict()
                else:
                    payload = {"event": str(event)}
                handle = self._run_event_log_handle
                if handle is None:
                    return
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")
                handle.flush()
            except Exception:
                pass

        try:
            self.event_logger.register_callback(_sink)
            self._run_event_log_callback = _sink
        except Exception:
            self._run_event_log_callback = None
            handle = self._run_event_log_handle
            self._run_event_log_handle = None
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass

    def _detach_run_event_log_sink(self) -> None:
        callback = self._run_event_log_callback
        self._run_event_log_callback = None
        if not callback:
            handle = self._run_event_log_handle
            self._run_event_log_handle = None
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
            return
        try:
            self.event_logger.unregister_callback(callback)
        except Exception:
            pass
        handle = self._run_event_log_handle
        self._run_event_log_handle = None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def _current_page_url(self) -> str:
        """Best-effort current page URL from active browser page."""
        try:
            if self.browser and self.browser.page:
                return str(self.browser.page.url or "").strip()
        except Exception:
            pass
        return ""

    def _enforce_current_page_policy(self, *, source: str) -> Tuple[bool, Optional[str], str]:
        """
        Validate current page URL against sandbox policy.

        Returns:
            (allowed, warning_message_or_none, current_url)
        """
        current_url = self._current_page_url()
        if not current_url:
            return True, None, current_url

        decision = self.sandbox_policy.check_url(current_url)
        if decision.allowed:
            return True, None, current_url

        warning = f"{source} landed on disallowed URL: {decision.reason}"
        self.event_logger.system_warning(
            warning,
            source=source,
            current_url=current_url,
            sandbox_preset=self.sandbox_policy.preset,
            sandbox_mode=self.sandbox_policy.mode,
        )
        if self.sandbox_policy.enforce:
            return False, warning, current_url
        return True, warning, current_url

    def _start(self) -> None:
        """Start the agent"""
        # Create browser without context manager - we'll manage its lifecycle
        # through the Agent's context manager instead
        self.browser = Browser(config=self.config)
        self.browser.started = True
        
        # Initialize components
        self.memory_store: NarrativeMemory = NarrativeMemory(self.browser)
        self.memory_store.set_base_knowledge(self.base_knowledge)
        
        self.page_utils = PageUtils(self.browser.page)

        # Initialize action executor
        self.action_executor: Executor = Executor(
            self.browser,
            self.memory_store,
            self.notebook,
            self.page_utils,
            user_question_callback=self.user_question_callback,
            agent_talk_callback=self.agent_talk_callback,
            data_report_callback=self.data_report_callback,
            workspace_paths={
                "written_data_dir": str(self.agent_workspace.written_data_dir),
                "workspace_root": str(self.agent_workspace.workspace_root),
            },
            upload_mode=getattr(self.config.execution, "upload_mode", "auto"),
            force_workspace_write_data=bool(
                getattr(self.config.execution, "force_workspace_write_data", False)
            ),
            sandbox_policy=self.sandbox_policy,
        )
        
        # State tracking
        self.current_attempt = 0
        self.last_screenshot_hash = None
        self.last_dom_signature = None
        
        # Screenshot caching for performance optimization
        self._cached_clean_screenshot = None
        self._cached_dom_signature = None
        
        self.interceptor_manager = InterceptorManager(self.browser)

        # Tab manager for multi-tab support
        self.tab_manager = self.browser.tab_manager

        # Emit navigation events for main-frame URL changes.
        try:
            def _on_frame_navigated(frame):
                try:
                    if self.browser and self.browser.page and frame == self.browser.page.main_frame:
                        self.event_logger.browser_navigation(url=frame.url)
                except Exception:
                    pass

            if hasattr(self.browser.page, "on"):
                self.browser.page.on("framenavigated", _on_frame_navigated)
        except Exception:
            pass

        # Plan generator for AI planning prompts
        self.started = True
      
    def execute_mission(
        self,
        user_prompt: str,
        starting_url: str = "",
        base_knowledge: Optional[List[str]] = None,
    ) -> MissionResult:
        previous_base_knowledge = list(self.base_knowledge)
        if base_knowledge is not None:
            self.base_knowledge = [
                str(item).strip()
                for item in base_knowledge
                if str(item).strip()
            ]
            self.memory_store.set_base_knowledge(self.base_knowledge)

        if starting_url != "" and starting_url != "about:blank":
            self.browser.page.goto(starting_url)
        # Register all pre-registered interceptors with the new controller
        for interceptor_data in self.interceptor_stack:
            self.register_interceptor(
                trigger=interceptor_data["trigger"],
                mode=interceptor_data["mode"],
                handler=interceptor_data["handler"]
            )
        
        # Run the mission
        try:
            mission_result = self._run_mission(user_prompt)
        finally:
            # Best-effort guard so per-run sink never leaks across missions.
            self._detach_run_event_log_sink()
            self.sandbox_policy.set_audit_log_path(None)
            self.effect_policy.set_audit_log_path(None)
            if base_knowledge is not None:
                self.base_knowledge = previous_base_knowledge
                self.memory_store.set_base_knowledge(self.base_knowledge)

        self.event_logger.agent_complete(mission_result.success, mission_result.reasoning)
        

        self.mission_result = mission_result
        return mission_result
    
    def register_interceptor(
        self,
        trigger: Interceptor,
        mode: InterceptorMode,
        handler: Optional[Callable[[InterceptorContext], None]] = None
    ):
        """Register an interceptor trigger and handler."""
        self.interceptor_manager.register_interceptor(trigger, mode, handler)

    def _handle_interceptor_trigger(self, entry: Dict[str, Any], action: Optional[str] = None, action_step: Optional[Any] = None) -> bool:
        """Process a triggered interceptor"""
        if len(self.interceptor_stack) >= self.interceptor_manager.recursion_limit:
            self.event_logger.system_warning(
                f"Interceptor recursion limit reached ({self.interceptor_manager.recursion_limit})"
            )
            return False

        # Add instruction if missing
        if "instruction" not in entry:
            entry["instruction"] = f"Interact with: {action}"
        
        self.interceptor_stack.append(entry)
        self.event_logger.system_info(f"🎯 Interceptor Active: {entry['instruction']}")

        if entry["mode"] != InterceptorMode.SCRIPTED:
            self.interceptor_stack.pop()
            self.event_logger.system_warning("Unsupported interceptor mode (scripted-only)")
            return False
        try:
            self.interceptor_manager.execute_scripted(entry, self, action_step, action)
            self.interceptor_stack.pop()
            self.event_logger.system_info("✅ Scripted Interceptor Complete")
            return True
        except Exception as e:
            self.interceptor_stack.pop()
            self.event_logger.system_error(f"❌ Scripted Interceptor Failed: {e}")
            return False

    def _warm_prompt_cache_background(self) -> None:
        """Fire a background thread to pre-warm the OpenAI prompt cache.

        Builds the exact static system prompt and developer prompt that the
        first iteration will use, then sends them to OpenAI with a minimal
        user message and max_output_tokens=1.  By the time the agent finishes
        the initial page load and DOM capture (~500-800ms), the cache is
        already warm — so iteration 1 gets a cache hit instead of a cold miss.
        """
        import threading
        from agent.action_planner import ActionPlanner
        from agent.prompts import get_memory_developer_policy

        budget_enabled = bool(
            getattr(self.config.execution, "budget_constraints_enabled", True)
        )

        # Build policy block (mission-constant) so the static prompt matches.
        policy_block = None
        policy_parts: List[str] = []
        if getattr(self.config, "sandbox", None) and getattr(self.config.sandbox, "prompt", None):
            if self.config.sandbox.prompt.include_policy_block:
                try:
                    policy_parts.append(self.sandbox_policy.render_prompt_policy_block())
                except Exception:
                    pass
        try:
            policy_parts.append(self.effect_policy.render_prompt_policy_block())
        except Exception:
            pass
        if policy_parts:
            policy_block = "\n\n".join(part for part in policy_parts if str(part or "").strip())

        try:
            planner = ActionPlanner(
                user_prompt="",
                memory_store=self.memory_store,
                model_name=self.config.model.agent_model,
                budget_constraints_enabled=budget_enabled,
                base_knowledge=self.base_knowledge,
                policy_constraints_block=policy_block,
                workspace_files=self._list_workspace_files(),
            )
            static_prompt = planner._build_function_calling_static_prompt()
        except Exception:
            return

        # Developer prompt also contributes to the cached prefix.
        try:
            developer_prompt = get_memory_developer_policy(budget_enabled)
        except Exception:
            developer_prompt = None

        model = self.config.model.agent_model

        def _warm() -> None:
            try:
                from openai import OpenAI
                import os
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                messages = [
                    {"role": "system", "content": static_prompt},
                ]
                if developer_prompt:
                    messages.append({"role": "developer", "content": developer_prompt})
                messages.append({"role": "user", "content": "."})
                client.responses.create(
                    model=model,
                    input=messages,
                    max_output_tokens=1,
                )
            except Exception:
                pass  # Best-effort; never block the mission

        threading.Thread(target=_warm, daemon=True, name="prompt-cache-warmer").start()

    def _maybe_wait_for_iteration_load(self, reason: str = "iteration") -> None:
        if not self.config.execution.wait_for_load_before_iteration:
            return
        try:
            if getattr(self.browser, "page", None) and not self.browser.page.is_closed():
                try:
                    self.event_logger.system_debug(
                        f"Waiting for page load before {reason} (state={self.wait_for_load_state})"
                    )
                except Exception:
                    pass
                self.browser.wait_for_load(
                    timeout=self.wait_for_load_timeout_ms,
                    state=self.wait_for_load_state,
                )
                self._wait_for_dom_stable(timeout_ms=min(self.wait_for_load_timeout_ms, 600))
        except Exception:
            # Best-effort wait; do not block on load wait errors.
            pass

    def _wait_for_dom_stable(self, timeout_ms: int = 600, sample_interval_ms: int = 80) -> None:
        """Best-effort DOM stability wait to reduce no-op iterations."""
        page = getattr(self.browser, "page", None)
        if page is None:
            return
        deadline = time.monotonic() + max(0.0, float(timeout_ms) / 1000.0)
        previous_sig: Optional[str] = None
        stable_samples = 0
        while time.monotonic() < deadline:
            try:
                current_sig = str(
                    page.evaluate(
                        """() => {
                            const body = document.body;
                            const txt = body ? (body.innerText || "") : "";
                            const len = txt.length;
                            const ready = document.readyState || "unknown";
                            return `${ready}:${len}:${window.scrollX || 0}:${window.scrollY || 0}`;
                        }"""
                    )
                    or ""
                )
            except Exception:
                return
            if current_sig == previous_sig and current_sig:
                stable_samples += 1
                if stable_samples >= 2:
                    return
            else:
                stable_samples = 0
                previous_sig = current_sig
            time.sleep(max(0.01, float(sample_interval_ms) / 1000.0))
    
    def _run_mission(self, user_mission: str) -> MissionResult:
        """
        Execute a mission directly.

        The agent executes browser actions to accomplish the mission.
        When repetition is needed, it declares loops inline via think(start_loop).

        Args:
            user_mission: User's high-level request
        Returns:
            MissionResult indicating success or failure
        """
        self._mission_tracker = {}
        self._original_user_mission = user_mission
        self._loaded_resume_run_id = None
        self._loaded_resume_mission = ""
        self._resume_checkpoint_loaded = False
        self._current_iteration = 0
        self.execution_state = None
        self._cached_snapshot = None
        self._cached_snapshot_fingerprint = None
        self._cached_page_info = None
        self._cached_detected_elements = None
        self._reset_speculative_state()
        self._cancel_event.clear()
        with self._pause_lock:
            self._paused = False
            self._pause_message = "Paused"
            self._pause_event.set()
        self._detach_run_event_log_sink()
        try:
            self._active_run_id = self.workspace_manager.start_run(
                self.agent_workspace,
                mission=user_mission,
            )
            self._active_run_open = True
            self._attach_run_event_log_sink()
            if self.config.sandbox.audit.enabled:
                self.sandbox_policy.set_audit_log_path(self.agent_workspace.sandbox_audit_path)
                self.effect_policy.set_audit_log_path(self.agent_workspace.sandbox_audit_path)
            else:
                self.sandbox_policy.set_audit_log_path(None)
                self.effect_policy.set_audit_log_path(None)
        except Exception as e:
            self._active_run_id = None
            self._active_run_open = False
            self._detach_run_event_log_sink()
            self.sandbox_policy.set_audit_log_path(None)
            self.effect_policy.set_audit_log_path(None)
            self.event_logger.system_warning(f"Failed to initialize run workspace: {e}")
        self.event_logger.agent_start(user_mission)
        try:
            self._discover_available_skills()
        except Exception as e:
            self.available_skills = []
            self.available_skills_catalog = ""
            self._available_skills_by_key = {}
            self.event_logger.system_warning(f"Skill discovery failed: {e}")

        # Initialize tracking
        try:
            self.mission_start_url = self.browser.page.url
        except Exception:
            self.mission_start_url = "unknown"
        self.mission_start_time = time.time()

        self.execution_timer.start_mission()

        if self.base_knowledge:
            self.memory_store.set_base_knowledge(self.base_knowledge)
        self.memory_store.start_mission(user_mission)
        # Persist an initial checkpoint so a crash before iteration 1 is still resumable.
        self._persist_resume_checkpoint(
            mission=user_mission,
            state=self.execution_state,
            status="running",
        )

        # Check if starting from a blank page
        if self.browser.page.url.startswith("about:blank"):
            self.event_logger.agent_error("Page is on initial blank page.")
            if self.execution_timer.mission_start_time is not None:
                self.execution_timer.end_mission()
            self.mission_result = self._build_mission_result(
                success=False,
                reasoning="Page is blank",
                narrative="Page is blank",
                state=self.execution_state,
            )
            return self.mission_result

        # Execute the mission directly
        self.mission_result = self._run_execution_loop(
            user_mission,
        )

        if self.execution_timer.mission_start_time is not None:
            self.execution_timer.end_mission()

        return self.mission_result

    def _build_mission_result(
        self,
        *,
        success: bool,
        reasoning: str,
        narrative: str,
        state: Optional[ExecutionState],
    ) -> MissionResult:
        duration_s = 0.0
        if self.mission_start_time is not None:
            duration_s = max(0.0, time.time() - self.mission_start_time)

        event_count = 0
        try:
            history = self.event_logger.get_event_history()
            if self.mission_start_time is not None:
                event_count = sum(
                    1 for event in history
                    if float(getattr(event, "timestamp", 0.0) or 0.0) >= float(self.mission_start_time)
                )
            else:
                event_count = len(history)
        except Exception:
            event_count = 0

        final_url = ""
        try:
            final_url = self.browser.page.url if self.browser and self.browser.page else ""
        except Exception:
            final_url = ""

        avg_iteration_ms = self._mean(getattr(state, "iteration_ms_samples", []) if state else [])
        p95_iteration_ms = self._p95(getattr(state, "iteration_ms_samples", []) if state else [])
        avg_llm_ms = self._mean(getattr(state, "llm_latency_ms_samples", []) if state else [])
        avg_tool_ms = self._mean(getattr(state, "tool_latency_ms_samples", []) if state else [])
        llm_calls = int(getattr(state, "llm_call_count", 0) or 0)
        tokens_in_total = int(getattr(state, "tokens_in_total", 0) or 0)
        tokens_out_total = int(getattr(state, "tokens_out_total", 0) or 0)
        image_count_total = int(getattr(state, "image_count_total", 0) or 0)
        tool_call_count = int(getattr(state, "tool_call_count", 0) or 0)
        retry_count = int(getattr(state, "retry_count", 0) or 0)
        avg_tokens_in = float(tokens_in_total) / float(llm_calls) if llm_calls > 0 else 0.0
        avg_tokens_out = float(tokens_out_total) / float(llm_calls) if llm_calls > 0 else 0.0
        avg_images_per_call = float(image_count_total) / float(llm_calls) if llm_calls > 0 else 0.0
        retries_per_mission = float(retry_count) / max(1.0, float(self._current_iteration or 1))
        failure_code = str(getattr(state, "failure_code", "") or "")
        failure_stage = str(getattr(state, "failure_stage", "") or "")

        result = MissionResult(
            success=success,
            reasoning=reasoning,
            narrative=narrative,
            total_iterations=self._current_iteration,
            total_actions=int(getattr(state, "total_actions", 0) or 0),
            final_url=final_url,
            duration_s=duration_s,
            total_cost_usd=self.event_logger.total_cost_usd,
            budget_total=int(getattr(state, "budget_total", 0) or 0),
            budget_spent=int(getattr(state, "budget_spent", 0) or 0),
            budget_remaining=int(getattr(state, "budget_remaining", 0) or 0),
            budget_phase=str(getattr(state, "budget_phase", "normal") or "normal"),
            mission_ms=duration_s * 1000.0,
            tool_calls=tool_call_count,
            tokens_in=tokens_in_total,
            tokens_out=tokens_out_total,
            image_count=image_count_total,
            retry_count=retry_count,
            failure_code=failure_code,
            failure_stage=failure_stage,
            avg_iteration_ms=avg_iteration_ms,
            p95_iteration_ms=p95_iteration_ms,
            avg_llm_ms=avg_llm_ms,
            avg_tool_ms=avg_tool_ms,
            avg_tokens_in=avg_tokens_in,
            avg_tokens_out=avg_tokens_out,
            avg_images_per_call=avg_images_per_call,
            retries_per_mission=retries_per_mission,
        )
        self._persist_resume_checkpoint(
            mission=self.memory_store.current_mission if hasattr(self, "memory_store") else "",
            state=state,
            status="success" if bool(success) else "failed",
        )
        if bool(getattr(self.config.debug, "telemetry_final_summary_enabled", True)):
            self.event_logger.system_info(
                "Final telemetry summary",
                mission_ms=round(result.mission_ms, 3),
                iterations=int(result.total_iterations or 0),
                tool_calls=int(result.tool_calls or 0),
                tokens_in=int(result.tokens_in or 0),
                tokens_out=int(result.tokens_out or 0),
                avg_iteration_ms=round(result.avg_iteration_ms, 3),
                p95_iteration_ms=round(result.p95_iteration_ms, 3),
                avg_llm_ms=round(result.avg_llm_ms, 3),
                avg_tool_ms=round(result.avg_tool_ms, 3),
                avg_tokens_in=round(result.avg_tokens_in, 3),
                avg_tokens_out=round(result.avg_tokens_out, 3),
                avg_images_per_call=round(result.avg_images_per_call, 3),
                retries_per_mission=round(result.retries_per_mission, 3),
                failure_code=result.failure_code or None,
                failure_stage=result.failure_stage or None,
            )
        if self._active_run_open:
            try:
                self.workspace_manager.finish_run(
                    self.agent_workspace,
                    success=bool(success),
                    reasoning=reasoning,
                    total_actions=int(getattr(state, "total_actions", 0) or 0),
                    total_iterations=int(self._current_iteration or 0),
                    final_url=final_url,
                    duration_s=duration_s,
                    event_count=event_count,
                )
            except Exception as e:
                self.event_logger.system_warning(f"Failed to finalize run workspace: {e}")
            finally:
                self._active_run_open = False
                self._detach_run_event_log_sink()
                self.sandbox_policy.set_audit_log_path(None)
                self.effect_policy.set_audit_log_path(None)
        else:
            self._detach_run_event_log_sink()
            self.sandbox_policy.set_audit_log_path(None)
            self.effect_policy.set_audit_log_path(None)

        return result

    @staticmethod
    def _compute_budget_phase(remaining: int, total: int) -> str:
        """Derive a simple budget phase from remaining vs total actions."""
        total_i = max(1, int(total or 1))
        remaining_i = max(0, int(remaining or 0))
        critical_threshold = max(1, int(total_i * 0.03))
        caution_threshold = max(3, int(total_i * 0.10))
        if caution_threshold <= critical_threshold:
            caution_threshold = critical_threshold + 1
        if remaining_i <= critical_threshold:
            return "critical"
        if remaining_i <= caution_threshold:
            return "caution"
        return "normal"

    @staticmethod
    def _clamp_loop_count_to_budget(requested_loop_count: Any, budget_remaining: int) -> Tuple[int, bool]:
        """Clamp loop count to remaining budget. Returns (effective_count, clamped)."""
        try:
            count = int(requested_loop_count)
        except (TypeError, ValueError):
            count = 1
        count = max(1, count)
        remaining_i = max(0, int(budget_remaining or 0))
        if remaining_i > 0 and count > remaining_i:
            return remaining_i, True
        return count, False

    @staticmethod
    def _mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _p95(values: List[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(v) for v in values)
        idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95))))
        return float(ordered[idx])

    def _resolve_policy_visible_tool_names(self) -> tuple[str, list[str]]:
        """Resolve active effect policy into runtime-allowed tool names."""
        policy_id = (
            f"effect:{self.config.execution.tool_policy.preset.value}:"
            f"{self.config.execution.tool_policy.mode.value}"
        )
        allowed: list[str] = []
        for spec in self.tool_registry.iter_specs():
            decision = self.effect_policy.evaluate(spec.manifest, phase="planner", record=False)
            if decision.allowed or not self.effect_policy.enforce:
                allowed.append(spec.manifest.name)
        return policy_id, allowed

    def _refresh_policy_visible_tool_names(self) -> None:
        """Load active effect policy into controller runtime state."""
        _, allowed = self._resolve_policy_visible_tool_names()
        self.policy_visible_tool_names = [
            str(name).strip()
            for name in (allowed or [])
            if str(name).strip()
        ]

    @staticmethod
    def _record_failure(state: ExecutionState, *, code: str, stage: str) -> None:
        state.failure_code = str(code or "").strip() or None
        state.failure_stage = str(stage or "").strip() or None

    def _emit_live_telemetry(self, state: ExecutionState) -> None:
        """Emit factual rolling telemetry per iteration."""
        if not bool(getattr(self.config.debug, "telemetry_live_enabled", True)):
            return
        avg_iteration = self._mean(state.iteration_ms_samples)
        avg_llm = self._mean(state.llm_latency_ms_samples)
        avg_tool = self._mean(state.tool_latency_ms_samples)
        avg_tokens_in = (
            float(state.tokens_in_total) / float(state.llm_call_count)
            if state.llm_call_count > 0
            else 0.0
        )
        avg_tokens_out = (
            float(state.tokens_out_total) / float(state.llm_call_count)
            if state.llm_call_count > 0
            else 0.0
        )
        avg_images = (
            float(state.image_count_total) / float(state.llm_call_count)
            if state.llm_call_count > 0
            else 0.0
        )
        retries_per_mission = (
            float(state.retry_count) / max(1.0, float(self._current_iteration or 1))
        )
        self.event_logger.live_telemetry(
            avg_iteration_ms=round(avg_iteration, 3),
            p95_iteration_ms=round(self._p95(state.iteration_ms_samples), 3),
            avg_llm_ms=round(avg_llm, 3),
            avg_tool_ms=round(avg_tool, 3),
            avg_tokens_in=round(avg_tokens_in, 3),
            avg_tokens_out=round(avg_tokens_out, 3),
            avg_images_per_call=round(avg_images, 3),
            retries_per_mission=round(retries_per_mission, 3),
            mission_ms=round(sum(state.iteration_ms_samples), 3),
        )




    def _capture_snapshot(self, full_page: bool = False) -> MemoryState:
        """
        Capture current browser state snapshot.
        
        Args:
            full_page: If True, capture full page screenshot (for exploration mode)
                      If False, capture viewport only (normal mode)
        """
        snapshot = self.memory_store._capture_current_state()
        
        # _capture_current_state already captures viewport screenshot. Re-capture only when
        # caller explicitly requests full-page or when screenshot is missing.
        if full_page or snapshot.screenshot is None:
            try:
                fmt = (self.config.browser.screenshot_format or "jpeg").lower()
                shot_kwargs: dict = {"type": fmt}
                if fmt == "jpeg":
                    shot_kwargs["quality"] = int(self.config.browser.screenshot_quality or 80)
                if full_page:
                    snapshot.screenshot = self.browser.page.screenshot(full_page=True, **shot_kwargs)
                    dprint("📸 Using full-page screenshot for exploration mode")
                else:
                    snapshot.screenshot = self.browser.page.screenshot(full_page=False, **shot_kwargs)
            except Exception as e:
                dprint(f"⚠️ Failed to capture screenshot: {e}")
                snapshot.screenshot = None

        # Save screenshot for debugging if enabled
        if self.save_screenshots and snapshot.screenshot:
            try:
                from pathlib import Path
                from datetime import datetime

                # Create directory if it doesn't exist
                screenshot_path = Path(self.screenshot_dir)
                screenshot_path.mkdir(parents=True, exist_ok=True)

                # Generate filename with iteration and timestamp
                self._screenshot_counter += 1
                timestamp = datetime.now().strftime("%H%M%S")
                filename = f"iter{self._current_iteration:03d}_snap{self._screenshot_counter:03d}_{timestamp}.png"
                filepath = screenshot_path / filename

                # Save the screenshot
                with open(filepath, "wb") as f:
                    f.write(snapshot.screenshot)
                dprint(f"📸 Saved screenshot: {filepath}")
            except Exception as e:
                dprint(f"⚠️ Failed to save screenshot: {e}")

        if snapshot.screenshot and self.config.debug.stream_screenshots:
            try:
                screenshot_meta = self.screenshot_store.put(
                    snapshot.screenshot,
                    iteration=self._current_iteration,
                    url=getattr(snapshot, "url", "") or "",
                    title=getattr(snapshot, "title", "") or "",
                )
                setattr(snapshot, "screenshot_id", screenshot_meta.screenshot_id)
                self.event_logger.screenshot_captured(
                    screenshot_id=screenshot_meta.screenshot_id,
                    iteration=screenshot_meta.iteration,
                    byte_size=screenshot_meta.byte_size,
                    sha256=screenshot_meta.sha256,
                    path=screenshot_meta.path,
                    in_memory=screenshot_meta.in_memory,
                    url=screenshot_meta.url,
                )
            except Exception as e:
                self.event_logger.system_warning(f"Failed to stream screenshot metadata: {e}")

        # Compute screenshot hash for change detection
        screenshot_data = getattr(snapshot, "screenshot", None)
        screenshot_hash = None
        if screenshot_data:
            try:
                screenshot_hash = hashlib.md5(screenshot_data).hexdigest()
            except Exception:
                screenshot_hash = None
        setattr(snapshot, "screenshot_hash", screenshot_hash)
        
        return snapshot

    def _get_latest_recommended_next_step(self) -> Tuple[Optional[str], Optional[str]]:
        """Return newest recommended step plus source memory ID."""
        return self.memory_store.get_latest_recommended_next_step()

    def _record_controller_action(
        self,
        *,
        action_type: str,
        action_step: Any,
        success: bool,
        error_message: Optional[str] = None,
        action_params: Optional[Dict[str, Any]] = None,
        before_state: Optional[MemoryState] = None,
        after_state: Optional[MemoryState] = None,
    ) -> None:
        """Record controller-handled actions into narrative memory."""
        try:
            args = getattr(action_step, "function_arguments", {}) or {}
            params = dict(action_params or {})

            evidence_ids = args.get("memory_evidence_ids")
            if isinstance(evidence_ids, list):
                normalized_ids = self.memory_store.normalize_memory_ids(evidence_ids)
                if normalized_ids:
                    params["memory_evidence_ids"] = normalized_ids

            if getattr(action_step, "function_name", None):
                params["tool"] = action_step.function_name

            if before_state is None:
                before_state = self.memory_store._capture_current_state()
            if after_state is None:
                after_state = self.memory_store._capture_current_state()

            self.memory_store.record_action(
                action_type=action_type,
                action_params=params,
                reasoning=args.get("reasoning") or getattr(action_step, "reasoning", None),
                before_state=before_state,
                after_state=after_state,
                success=success,
                error_message=error_message,
                mission=self.memory_store.current_mission,
                reference_memory_ids=normalized_ids if isinstance(evidence_ids, list) else [],
            )
        except Exception:
            pass

    def _reset_agent_event_tracking(self) -> None:
        self._accepted_agent_event_counts = {
            name: 0
            for name in self.event_definition_map.keys()
        }

    def _mark_agent_event_accepted(self, event_name: str) -> None:
        current = int(self._accepted_agent_event_counts.get(event_name, 0) or 0)
        self._accepted_agent_event_counts[event_name] = current + 1

    def _missing_required_agent_events(self) -> List[str]:
        missing: List[str] = []
        for definition in self.event_definitions:
            if not definition.required:
                continue
            if int(self._accepted_agent_event_counts.get(definition.name, 0) or 0) <= 0:
                missing.append(definition.name)
        return missing

    def _event_once_per_mission_already_accepted(self, event_name: str) -> bool:
        return int(self._accepted_agent_event_counts.get(event_name, 0) or 0) > 0

    def _build_agent_events_status(self) -> dict[str, Any]:
        if not self.event_definitions:
            return {
                "enabled": False,
                "can_complete_mission": True,
                "completion_blockers": [],
                "required_events": {},
                "accepted_counts": {},
                "ack_required_events": [],
            }
        missing_required = self._missing_required_agent_events()
        completion_blockers: List[str] = []
        if missing_required:
            completion_blockers.append(
                "missing_required_events: " + ", ".join(sorted(missing_required))
            )
        can_complete = not completion_blockers
        required_states: Dict[str, str] = {}
        for definition in self.event_definitions:
            if not definition.required:
                continue
            required_states[definition.name] = (
                "accepted"
                if int(self._accepted_agent_event_counts.get(definition.name, 0) or 0) > 0
                else "missing"
            )
        ack_required = [
            definition.name
            for definition in self.event_definitions
            if definition.require_callback_ack
        ]
        return {
            "enabled": True,
            "can_complete_mission": can_complete,
            "completion_blockers": completion_blockers,
            "required_events": required_states,
            "accepted_counts": dict(self._accepted_agent_event_counts),
            "ack_required_events": ack_required,
        }

    @staticmethod
    def _extract_callback_ack(response: Any) -> tuple[Optional[bool], str]:
        if not isinstance(response, dict):
            return None, ""
        ack_value = response.get("ack")
        reason = str(response.get("reason", "") or "").strip()
        if isinstance(ack_value, bool):
            return ack_value, reason
        return None, reason

    def _collect_pending_agent_events(
        self,
        *,
        action_step: ActionStep,
        action_args: Dict[str, Any],
    ) -> List[dict[str, Any]]:
        merged: List[dict[str, Any]] = []
        existing = getattr(action_step, "pending_events", []) or []
        if isinstance(existing, list):
            merged.extend(coerce_emit_events(existing))
        raw_emit = action_args.pop("emit_events", None) if isinstance(action_args, dict) else None
        merged.extend(coerce_emit_events(raw_emit))
        action_step.function_arguments = dict(action_args or {})
        action_step.pending_events = list(merged)
        return merged

    def _dispatch_agent_events(
        self,
        *,
        pending_events: List[dict[str, Any]],
        function_name: str,
        action_id: str,
        action_success: bool,
    ) -> List[EventResult]:
        results: List[EventResult] = []
        if not pending_events:
            return results

        if not action_success:
            for item in pending_events:
                name = str(item.get("name", "") or "").strip() or "unknown_event"
                event_id = f"{action_id}:{name}:{uuid.uuid4().hex[:8]}"
                error = "tool action failed before event dispatch"
                self.event_logger.agent_event_callback_error(
                    event_id=event_id,
                    action_id=action_id,
                    name=name,
                    error=error,
                    tool=function_name,
                    iteration=self._current_iteration,
                )
                results.append(EventResult(event_id=event_id, name=name, delivered=False, error=error))
            return results

        if self.browser and self.browser.page:
            try:
                current_url = str(self.browser.page.url or "")
            except Exception:
                current_url = ""
            try:
                page_title = str(self.browser.page.title() or "")
            except Exception:
                page_title = ""
        else:
            current_url = ""
            page_title = ""

        for idx, item in enumerate(pending_events):
            name = str(item.get("name", "") or "").strip()
            function_name_norm = str(function_name or "").strip().lower()
            payload = item.get("data")
            data = payload if isinstance(payload, dict) else {}
            event_id = f"{action_id}:{idx}:{uuid.uuid4().hex[:8]}"
            if not name:
                error = "event name is empty"
                self.event_logger.agent_event_callback_error(
                    event_id=event_id,
                    action_id=action_id,
                    name="",
                    error=error,
                    tool=function_name,
                    iteration=self._current_iteration,
                )
                results.append(EventResult(event_id=event_id, name="", delivered=False, error=error))
                continue

            definition = self.event_definition_map.get(name)
            if definition is None:
                error = f"event '{name}' is not defined"
                self.event_logger.agent_event_callback_error(
                    event_id=event_id,
                    action_id=action_id,
                    name=name,
                    error=error,
                    tool=function_name,
                    iteration=self._current_iteration,
                )
                results.append(EventResult(event_id=event_id, name=name, delivered=False, error=error))
                continue

            allowed_tools = list(definition.allowed_tools or [])
            if allowed_tools and function_name_norm not in allowed_tools:
                error = (
                    f"event '{name}' is not allowed for tool '{function_name}'. "
                    f"Allowed tools: {', '.join(allowed_tools)}"
                )
                self.event_logger.agent_event_callback_error(
                    event_id=event_id,
                    action_id=action_id,
                    name=name,
                    error=error,
                    tool=function_name,
                    iteration=self._current_iteration,
                )
                results.append(EventResult(event_id=event_id, name=name, delivered=False, error=error))
                continue

            if definition.once_per_mission and self._event_once_per_mission_already_accepted(name):
                error = f"event '{name}' is once_per_mission and was already accepted"
                self.event_logger.agent_event_callback_error(
                    event_id=event_id,
                    action_id=action_id,
                    name=name,
                    error=error,
                    tool=function_name,
                    iteration=self._current_iteration,
                )
                results.append(EventResult(event_id=event_id, name=name, delivered=False, error=error))
                continue

            valid_payload, reason = validate_event_payload(definition, data)
            if not valid_payload:
                error = reason or "event payload validation failed"
                self.event_logger.agent_event_callback_error(
                    event_id=event_id,
                    action_id=action_id,
                    name=name,
                    error=error,
                    tool=function_name,
                    iteration=self._current_iteration,
                )
                results.append(EventResult(event_id=event_id, name=name, delivered=False, error=error))
                continue

            event = AgentEvent(
                event_id=event_id,
                action_id=action_id,
                name=name,
                data=data,
                context={
                    "url": current_url,
                    "page_title": page_title,
                    "iteration": int(self._current_iteration or 0),
                    "tool": function_name,
                },
            )
            self.event_logger.agent_event_emitted(
                event_id=event.event_id,
                action_id=event.action_id,
                name=event.name,
                tool=function_name,
                iteration=self._current_iteration,
            )
            callback_ok, callback_response, callback_error = self._invoke_event_callback(event)
            accepted = True
            failure_error = ""

            if definition.require_callback_ack:
                if not callback_ok:
                    accepted = False
                    failure_error = callback_error or "callback ack required but callback failed"
                else:
                    ack_value, ack_reason = self._extract_callback_ack(callback_response)
                    if ack_value is not True:
                        accepted = False
                        if ack_value is False:
                            failure_error = ack_reason or "callback rejected event (ack=false)"
                        else:
                            failure_error = (
                                ack_reason
                                or "callback ack required but response missing {'ack': true}"
                            )

            if callback_ok:
                self.event_logger.agent_event_callback_success(
                    event_id=event.event_id,
                    action_id=event.action_id,
                    name=event.name,
                    tool=function_name,
                    iteration=self._current_iteration,
                    has_response=callback_response is not None,
                )
            elif callback_error != "event callback not configured":
                self.event_logger.agent_event_callback_error(
                    event_id=event.event_id,
                    action_id=event.action_id,
                    name=event.name,
                    error=callback_error,
                    tool=function_name,
                    iteration=self._current_iteration,
                )

            if not accepted:
                self.event_logger.agent_event_callback_error(
                    event_id=event.event_id,
                    action_id=event.action_id,
                    name=event.name,
                    error=failure_error,
                    tool=function_name,
                    iteration=self._current_iteration,
                )
                results.append(
                    EventResult(
                        event_id=event.event_id,
                        name=event.name,
                        delivered=False,
                        error=failure_error,
                    )
                )
                continue

            self._mark_agent_event_accepted(event.name)
            results.append(
                EventResult(
                    event_id=event.event_id,
                    name=event.name,
                    delivered=True,
                    response=callback_response if callback_ok else None,
                    error=None,
                )
            )
        return results

    def _invoke_event_callback(self, event: AgentEvent) -> tuple[bool, Any, str]:
        if self.event_callback is None:
            return False, None, "event callback not configured"

        timeout_s = float(self.event_callback_timeout_seconds or 0.0)
        if timeout_s <= 0.0:
            try:
                return True, self.event_callback(event), ""
            except Exception as exc:
                return False, None, str(exc) or exc.__class__.__name__

        done = threading.Event()
        holder: Dict[str, Any] = {}

        def _runner() -> None:
            try:
                holder["response"] = self.event_callback(event)
            except Exception as exc:
                holder["error"] = str(exc) or exc.__class__.__name__
            finally:
                done.set()

        thread = threading.Thread(target=_runner, daemon=True, name="agent-event-callback")
        thread.start()
        completed = done.wait(timeout_s)
        if not completed:
            return False, None, f"event callback timeout after {timeout_s:.2f}s"
        if "error" in holder:
            return False, None, str(holder.get("error", "event callback failed"))
        return True, holder.get("response"), ""

    def _summarize_event_results(
        self,
        event_results: List[EventResult],
    ) -> tuple[str, List[str]]:
        if not event_results:
            return "", []
        fragments: List[str] = []
        hints: List[str] = []
        for result in event_results:
            if result.delivered:
                if result.response is None:
                    fragments.append(f"{result.name}=emitted")
                    continue
                response_text = self._format_event_callback_response(result.response)
                if response_text:
                    fragments.append(f"{result.name}={response_text}")
                    hints.append(
                        f'Event callback for "{result.name}" returned: {response_text}. '
                        "Use this signal in your next actions."
                    )
            else:
                err = str(result.error or "callback_failed")
                fragments.append(f"{result.name}=error({err})")
        return "; ".join(fragments[:4]), hints

    def _format_event_callback_response(self, response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, (dict, list, str, int, float, bool)):
            try:
                text = (
                    json.dumps(response, ensure_ascii=True, sort_keys=True)
                    if not isinstance(response, str)
                    else response
                )
            except Exception:
                text = str(response)
        else:
            text = str(response)
        text = text.strip()
        if not text:
            return ""
        max_chars = int(self.event_callback_response_max_chars or 0)
        if max_chars > 0 and len(text) > max_chars:
            return text[: max(0, max_chars - 3)] + "..."
        return text

    @staticmethod
    def _build_action_summary(action_step, result_str: str) -> str:
        """Build a natural-language summary of an action for the reflection block.

        Uses the action_step's function_name and arguments to produce a brief,
        first-person description of what happened.
        """
        fn = getattr(action_step, "function_name", None)
        args = getattr(action_step, "function_arguments", {}) or {}

        if fn == "click":
            elem = args.get("element_type", "element")
            desc = args.get("description", "")
            return f"You clicked the {elem} \"{desc}\". Result: {result_str}."
        elif fn == "type_text":
            text = args.get("text", "")
            field = args.get("field_description", "input field")
            return f"You typed \"{text}\" into {field}. Result: {result_str}."
        elif fn == "clear_text":
            field = args.get("field_description", "input field")
            return f"You cleared the text in {field}. Result: {result_str}."
        elif fn == "select_option":
            option = args.get("option", "")
            dropdown = args.get("dropdown_description", "dropdown")
            return f"You selected \"{option}\" in {dropdown}. Result: {result_str}."
        elif fn == "scroll_down":
            return f"You scrolled down. Result: {result_str}."
        elif fn == "scroll_up":
            return f"You scrolled up. Result: {result_str}."
        elif fn == "scroll_container":
            direction = args.get("direction", "down")
            return f"You scrolled a container {direction}. Result: {result_str}."
        elif fn == "scroll_to_element":
            element_id = args.get("element_id")
            return f"You scrolled to element [id={element_id}]. Result: {result_str}."
        elif fn == "press_key":
            key = args.get("key", "")
            return f"You pressed {key}. Result: {result_str}."
        elif fn == "open_url":
            url = args.get("url", "")
            return f"You navigated to {url}. Result: {result_str}."
        elif fn == "go_back":
            return f"You went back in browser history. Result: {result_str}."
        elif fn == "go_forward":
            return f"You went forward in browser history. Result: {result_str}."
        elif fn == "upload_file":
            file_path = args.get("file_path", "")
            return f"You uploaded file \"{file_path}\". Result: {result_str}."
        elif fn == "extract_data":
            desc = args.get("data_description", "data")
            return f"You extracted: \"{desc}\". Result: {result_str}."
        elif fn == "report_data":
            payload = args.get("payload", "")
            return f"You reported data to the user: \"{payload}\". Result: {result_str}."
        elif fn == "write_data":
            resolved = args.get("path", "") or args.get("file_name", "") or "default location"
            return f"You wrote data to {resolved}. Result: {result_str}."
        elif fn == "wait_for":
            condition = args.get("condition", "")
            return f"You waited for: \"{condition}\". Result: {result_str}."
        elif fn == "send_email":
            to = args.get("to", "")
            subject = args.get("subject", "")
            body = args.get("body", "")
            return f"You sent an email to {to} with the subject \"{subject}\" and the body \"{body}\". Result: {result_str}."
        else:
            # Fallback: use the action string
            action_str = getattr(action_step, "action", str(action_step))
            return f"You performed: {action_str}. Result: {result_str}."

    def _build_element_index_and_start_gallery(
        self,
        snapshot,
        detected_elements: PageElements,
    ) -> ScreenshotPreparation:
        """Build element index and kick off gallery generation in background.

        Returns a ScreenshotPreparation immediately with element index text
        and counts populated.  If text-poor elements exist, gallery generation
        runs in ``_gallery_executor`` — call ``_collect_gallery()`` on the
        returned prep object before passing it to the planner.
        """
        screenshot = snapshot.screenshot

        if not screenshot:
            return ScreenshotPreparation(screenshot_bytes=screenshot)

        result = build_element_index(
            detected_elements.elements,
            max_elements=self.config.elements.max_index_elements,
            viewport_only=True,
        )

        prep = ScreenshotPreparation(
            screenshot_bytes=screenshot,
            element_index_text=result.index_text,
            text_rich_count=result.text_rich_count,
            text_poor_count=result.text_poor_count,
        )

        # Start gallery generation in background thread so the main thread
        # can build context (memory, decision, tabs, etc.) in parallel.
        if result.text_poor_elements:
            crops_per = self.config.elements.crops_per_gallery
            prep._gallery_future = self._gallery_executor.submit(
                build_crop_gallery,
                screenshot,
                result.text_poor_elements,
                crops_per,
            )

        return prep

    def _collect_gallery(self, prep: ScreenshotPreparation) -> None:
        """Block until gallery generation completes and attach results.

        Also saves debug screenshots when ``save_screenshots`` is enabled.
        """
        future = getattr(prep, "_gallery_future", None)
        if future is not None:
            try:
                gallery_images = future.result()
            except Exception:
                gallery_images = None
            prep.gallery_images = gallery_images
            if gallery_images:
                dprint(f"📸 Built {len(gallery_images)} gallery page(s)")
            delattr(prep, "_gallery_future")

        if self.save_screenshots and prep.screenshot_bytes:
            try:
                from pathlib import Path
                from datetime import datetime
                ss_dir = Path(self.screenshot_dir)
                ss_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%H%M%S")
                clean_path = str(ss_dir / f"iter{self._current_iteration:03d}_clean_{ts}.png")
                with open(clean_path, "wb") as f:
                    f.write(prep.screenshot_bytes)
                if prep.gallery_images:
                    for gi_idx, gi_bytes in enumerate(prep.gallery_images):
                        gp = str(ss_dir / f"iter{self._current_iteration:03d}_gallery{gi_idx + 1}_{ts}.png")
                        with open(gp, "wb") as f:
                            f.write(gi_bytes)
                dprint(f"📸 Saved clean + {len(prep.gallery_images or [])} gallery screenshot(s)")
            except Exception as e:
                dprint(f"⚠️ Could not save debug screenshots: {e}")

    def _reset_speculative_state(self) -> None:
        """Cancel and clear speculative-runtime state."""
        with self._speculative_lock:
            self._planner_cached_hint_bundle = None

    def _cache_planner_hint_bundle(self, hint_bundle: Optional[HintBundle]) -> None:
        """Store planner-embedded hint bundle for the next iteration."""
        if not bool(getattr(self.config.execution, "speculative_hints_enabled", False)):
            with self._speculative_lock:
                self._planner_cached_hint_bundle = None
            return
        if not isinstance(hint_bundle, HintBundle):
            with self._speculative_lock:
                self._planner_cached_hint_bundle = None
            return

        candidates = list(hint_bundle.candidates or [])
        with self._speculative_lock:
            self._planner_cached_hint_bundle = hint_bundle if candidates else None

        if candidates:
            self.event_logger.system_debug(
                "Planner-embedded hint cached",
                source_iteration=int(hint_bundle.source_iteration or 0),
                candidate_count=len(candidates),
            )

    def _consume_or_discard_cached_planner_hint(
        self,
        *,
        expected_source_iteration: int,
    ) -> Optional[HintBundle]:
        """Consume cached planner hint if fresh for this iteration."""
        with self._speculative_lock:
            hint_bundle = self._planner_cached_hint_bundle
            self._planner_cached_hint_bundle = None

        if hint_bundle is None:
            return None
        if int(hint_bundle.source_iteration or 0) != int(expected_source_iteration or 0):
            self.event_logger.system_debug(
                "Discarded stale planner-embedded hint",
                expected_source_iteration=expected_source_iteration,
                hint_source_iteration=int(hint_bundle.source_iteration or 0),
            )
            return None
        self.event_logger.system_debug(
            "Planner-embedded hint consumed",
            source_iteration=int(hint_bundle.source_iteration or 0),
            candidate_count=len(hint_bundle.candidates or []),
        )
        return hint_bundle

    def _invoke_action_planner_once(
        self,
        *,
        action_planner: Any,
        environment_state: EnvironmentState,
        screenshot: bytes,
        notebook: Notebook,
        detected_elements: PageElements,
    ) -> PlannerCallResult:
        """Invoke planner once and normalize telemetry/errors."""
        outcome = PlannerCallResult()
        started_at = time.perf_counter()
        try:
            actions, error = action_planner.get_next_actions_with_function_calling(
                environment_state=environment_state,
                screenshot=screenshot,
                notebook=notebook,
                element_data=detected_elements,
            )
            outcome.actions = actions
            outcome.error = error
            planner_stats = dict(getattr(action_planner, "last_call_telemetry", {}) or {})
            outcome.llm_latency_ms = float(planner_stats.get("llm_latency_ms", 0.0) or 0.0)
            outcome.tokens_in = int(planner_stats.get("tokens_in", 0) or 0)
            outcome.tokens_out = int(planner_stats.get("tokens_out", 0) or 0)
            outcome.image_count = int(planner_stats.get("image_count", 0) or 0)
            outcome.retries = int(planner_stats.get("planner_retries", 0) or 0)
            outcome.response_id = str(getattr(action_planner, "last_response_id", "") or "") or None
            outcome.tool_call_ids = list(getattr(action_planner, "last_tool_call_ids", []) or [])
            outcome.failure_code = str(getattr(action_planner, "last_failure_code", "") or "") or None
            outcome.failure_stage = str(getattr(action_planner, "last_failure_stage", "") or "") or None
            maybe_hint_bundle = getattr(action_planner, "last_hint_bundle", None)
            outcome.hint_bundle = maybe_hint_bundle if isinstance(maybe_hint_bundle, HintBundle) else None
            outcome.hint_status = str(getattr(action_planner, "last_hint_status", "") or "") or None
            outcome.hint_reason = str(getattr(action_planner, "last_hint_reason", "") or "") or None
            try:
                outcome.hint_confidence = float(getattr(action_planner, "last_hint_confidence", 0.0) or 0.0)
            except Exception:
                outcome.hint_confidence = 0.0
        except Exception as e:
            outcome.exception = e
        finally:
            outcome.planner_elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        return outcome

    def _resolve_actions_with_speculative_validation(
        self,
        *,
        action_planner: Any,
        environment_state: EnvironmentState,
        screenshot: bytes,
        notebook: Notebook,
        detected_elements: PageElements,
        snapshot: MemoryState,
        element_index_text: str,
        state: ExecutionState,
        hint_bundle: Optional[HintBundle],
        dialog_pending: bool,
    ) -> SpeculativeResolution:
        """Resolve actions via planner fallback and optional speculative validator fast-path."""
        planner_result = PlannerCallResult()
        result = SpeculativeResolution(
            planner=planner_result,
            hint_path="planner_only",
        )

        def _run_planner_once() -> PlannerCallResult:
            return self._invoke_action_planner_once(
                action_planner=action_planner,
                environment_state=environment_state,
                screenshot=screenshot,
                notebook=notebook,
                detected_elements=detected_elements,
            )

        speculative_enabled = bool(getattr(self.config.execution, "speculative_hints_enabled", False))
        if not speculative_enabled or hint_bundle is None:
            planner_result = _run_planner_once()
            if planner_result.exception is not None:
                raise planner_result.exception
            result.planner = planner_result
            result.actions = planner_result.actions
            result.error = planner_result.error
            result.hint_path = "planner_only" if not speculative_enabled else "planner_no_hint"
            return result

        min_confidence = float(
            getattr(self.config.execution, "speculative_hints_min_confidence", 0.75) or 0.75
        )
        filtered_candidates = filter_candidates_deterministic(
            hint_bundle=hint_bundle,
            current_url=str(getattr(snapshot, "url", "") or ""),
            current_title=str(getattr(snapshot, "title", "") or ""),
            detected_elements=detected_elements,
            policy_visible_tool_names=self.policy_visible_tool_names,
            dialog_pending=dialog_pending,
            in_loop=bool(state.in_loop),
            min_confidence=min_confidence,
        )
        if not filtered_candidates:
            planner_result = _run_planner_once()
            if planner_result.exception is not None:
                raise planner_result.exception
            result.planner = planner_result
            result.actions = planner_result.actions
            result.error = planner_result.error
            result.hint_path = "hint_prefilter_reject"
            result.hint_reject_reason = "target_missing"
            return result

        planner_embedded_candidates = [
            c for c in filtered_candidates
            if str(c.candidate_id or "").strip().startswith("planner_hint_")
        ]
        if planner_embedded_candidates:
            best_candidate = max(
                planner_embedded_candidates,
                key=lambda c: float(c.confidence or 0.0),
            )
            try:
                hinted_step = hydrate_candidate_to_action_step(
                    best_candidate,
                    budget_spent=state.budget_spent,
                    budget_remaining=state.budget_remaining,
                    budget_total=state.budget_total,
                )
                accepted = SpeculativeResolution(
                    actions=[hinted_step],
                    error=None,
                    planner=PlannerCallResult(),
                    hint_path="hint_accept_planner_embedded",
                    hint_candidate_id=str(best_candidate.candidate_id or "").strip() or None,
                    hint_validation_ms=0.0,
                    hint_confidence=float(best_candidate.confidence or 0.0),
                    hint_reject_reason=None,
                )
                self.event_logger.system_info(
                    "Speculative arbitration winner",
                    winner="hint_accept_planner_embedded",
                    hint_candidate_id=accepted.hint_candidate_id,
                    hint_confidence=accepted.hint_confidence,
                    hint_validation_ms=0.0,
                )
                return accepted
            except Exception:
                planner_result = _run_planner_once()
                if planner_result.exception is not None:
                    raise planner_result.exception
                result.planner = planner_result
                result.actions = planner_result.actions
                result.error = planner_result.error
                result.hint_path = "hint_reject_planner_used"
                result.hint_reject_reason = "invalid_candidate_id"
                return result

        validator_model = str(self.config.model.command_model or "").strip()
        validator_reasoning = self.config.model.command_reasoning_level
        candidate_by_id: Dict[str, HintCandidate] = {
            c.candidate_id: c for c in filtered_candidates if str(c.candidate_id).strip()
        }

        def _run_validator_once() -> Tuple[HintValidationResult, float]:
            validator_started_at = time.perf_counter()
            decision = validate_hints(
                mission=environment_state.user_prompt,
                current_url=str(getattr(snapshot, "url", "") or ""),
                current_title=str(getattr(snapshot, "title", "") or ""),
                hint_bundle=hint_bundle,
                filtered_candidates=filtered_candidates,
                screenshot=screenshot,
                element_index_text=element_index_text,
                model=validator_model,
                reasoning_level=validator_reasoning,
                image_detail="low",
            )
            return decision, (time.perf_counter() - validator_started_at) * 1000.0

        validator_future = self._speculative_executor.submit(_run_validator_once)
        planner_future: Optional[Future] = None
        validator_decision: Optional[HintValidationResult] = None
        validator_elapsed_ms = 0.0

        planner_future = self._speculative_executor.submit(_run_planner_once)

        def _maybe_accept_hint(decision: HintValidationResult, elapsed_ms: float) -> Optional[SpeculativeResolution]:
            accept_conf = float(decision.confidence or 0.0)
            decision_id = str(decision.candidate_id or "").strip()
            if decision.decision != "accept":
                return None
            if accept_conf < min_confidence:
                return None
            candidate = candidate_by_id.get(decision_id)
            if candidate is None:
                return None
            try:
                hinted_step = hydrate_candidate_to_action_step(
                    candidate,
                    budget_spent=state.budget_spent,
                    budget_remaining=state.budget_remaining,
                    budget_total=state.budget_total,
                )
            except Exception:
                return None
            accepted = SpeculativeResolution(
                actions=[hinted_step],
                error=None,
                planner=PlannerCallResult(),
                hint_path="hint_accept",
                hint_candidate_id=decision_id,
                hint_validation_ms=float(elapsed_ms),
                hint_confidence=accept_conf,
                hint_reject_reason=None,
            )
            return accepted

        if validator_decision is not None:
            accepted = _maybe_accept_hint(validator_decision, validator_elapsed_ms)
            if accepted is not None:
                if planner_future is not None:
                    try:
                        planner_future.cancel()
                    except Exception:
                        pass
                self.event_logger.system_info(
                    "Speculative arbitration winner",
                    winner="hint_accept",
                    hint_candidate_id=accepted.hint_candidate_id,
                    hint_confidence=accepted.hint_confidence,
                    hint_validation_ms=round(accepted.hint_validation_ms, 3),
                )
                return accepted

        planner_outcome: Optional[PlannerCallResult] = None
        while True:
            wait_futures: List[Future] = []
            if planner_future is not None:
                wait_futures.append(planner_future)
            if validator_future is not None and validator_decision is None:
                wait_futures.append(validator_future)
            if not wait_futures:
                break

            done, _ = wait(wait_futures, return_when=FIRST_COMPLETED)

            if validator_future is not None and validator_decision is None and validator_future in done:
                try:
                    validator_decision, validator_elapsed_ms = validator_future.result()
                except Exception:
                    validator_decision = HintValidationResult(
                        decision="reject",
                        confidence=0.0,
                        reason="Validator future failed.",
                        reject_reason="validator_error",
                    )
                accepted = _maybe_accept_hint(validator_decision, validator_elapsed_ms)
                if accepted is not None:
                    if planner_future is not None:
                        try:
                            planner_future.cancel()
                        except Exception:
                            pass
                    self.event_logger.system_info(
                        "Speculative arbitration winner",
                        winner="hint_accept",
                        hint_candidate_id=accepted.hint_candidate_id,
                        hint_confidence=accepted.hint_confidence,
                        hint_validation_ms=round(accepted.hint_validation_ms, 3),
                    )
                    return accepted

            if planner_future is not None and planner_future in done:
                planner_outcome = planner_future.result()
                if planner_outcome.exception is not None:
                    raise planner_outcome.exception
                break

        if planner_outcome is None:
            planner_outcome = _run_planner_once()
            if planner_outcome.exception is not None:
                raise planner_outcome.exception

        hint_path = "planner_won"
        reject_reason = None
        hint_confidence = 0.0
        hint_validation_ms = 0.0
        if validator_decision is not None:
            hint_confidence = float(validator_decision.confidence or 0.0)
            hint_validation_ms = float(validator_elapsed_ms or 0.0)
            if validator_decision.decision == "reject":
                hint_path = "hint_reject_planner_used"
                reject_reason = str(validator_decision.reject_reason or "state_conflict")
            elif validator_decision.decision == "abstain":
                hint_path = "hint_reject_planner_used"
                reject_reason = str(validator_decision.reject_reason or "abstain")
            elif validator_decision.decision == "accept" and hint_confidence < min_confidence:
                hint_path = "hint_reject_planner_used"
                reject_reason = "low_confidence"
            elif validator_decision.decision == "accept":
                hint_path = "hint_reject_planner_used"
                reject_reason = "invalid_candidate_id"

        self.event_logger.system_info(
            "Speculative arbitration winner",
            winner=hint_path,
            hint_confidence=round(hint_confidence, 3),
            hint_reject_reason=reject_reason,
            hint_validation_ms=round(hint_validation_ms, 3),
        )

        return SpeculativeResolution(
            actions=planner_outcome.actions,
            error=planner_outcome.error,
            planner=planner_outcome,
            hint_path=hint_path,
            hint_candidate_id=None,
            hint_validation_ms=hint_validation_ms,
            hint_confidence=hint_confidence,
            hint_reject_reason=reject_reason,
        )

    def _run_execution_loop(
        self,
        mission: str,
        *,
        existing_state: Optional[ExecutionState] = None,
    ) -> MissionResult:
        """
        Unified mission execution loop.

        The agent uses browser actions to accomplish the mission directly.
        When repetition is needed, the agent declares loops inline via think(start_loop).

        Args:
            mission: The mission string to execute

        Returns:
            MissionResult with success/failure and reasoning
        """
        from agent.action_planner import ActionPlanner

        max_actions = self.config.execution.max_actions_per_mission
        self._refresh_policy_visible_tool_names()
        active_policy_id, _ = self._resolve_policy_visible_tool_names()
        profile_max_steps = max(1, int(self.config.execution.max_actions_per_plan or 1))
        profile_timeout_s = 45.0

        if existing_state is None:
            state = ExecutionState(
                budget_constraints_enabled=bool(self.config.execution.budget_constraints_enabled),
            )
        else:
            # Resume path: continue from the loaded state snapshot.
            state = copy.deepcopy(existing_state)
            state.budget_constraints_enabled = bool(self.config.execution.budget_constraints_enabled)
        state.budget_total = max_actions
        state.planning_batch_limit = min(
            int(self.config.execution.max_actions_per_plan or 1),
            profile_max_steps,
        )
        self.execution_state = state
        self.memory_store.start_mission(mission)
        self.event_logger.system_info(
            "Tool effect policy active",
            tool_policy_id=active_policy_id,
            tool_policy_preset=self.effect_policy.preset.value,
            tool_policy_mode=self.effect_policy.mode.value,
            visible_tools_count=len(self.policy_visible_tool_names or []),
            max_steps=profile_max_steps,
            timeout_s=profile_timeout_s,
        )

        # Pre-warm the OpenAI prompt cache so iteration 1 gets a cache hit.
        self._warm_prompt_cache_background()

        def _append_recent_action(summary: str) -> None:
            state.recent_actions.append(summary)

        def _refresh_budget_state() -> None:
            state.budget_spent = max(0, int(state.total_actions or 0))
            state.budget_total = max(1, int(max_actions or 1))
            state.budget_remaining = max(0, state.budget_total - state.budget_spent)
            state.budget_phase = self._compute_budget_phase(
                remaining=state.budget_remaining,
                total=state.budget_total,
            )
            state.low_budget_mode = state.budget_phase in {"caution", "critical"}
            state.planning_batch_limit = (
                1
                if (state.budget_constraints_enabled and state.low_budget_mode)
                else min(int(self.config.execution.max_actions_per_plan or 1), profile_max_steps)
            )

        def _exit_loop() -> None:
            state.in_loop = False
            state.loop_count = None
            state.loop_round = 0
            state.loop_description = ""
            try:
                self.action_executor.clear_done_markers()
            except Exception:
                pass

        stall_soft_timeout_s = float(
            max(0.0, getattr(self.config.execution, "iteration_stall_soft_timeout_s", 0.0) or 0.0)
        )
        stall_hard_timeout_s = float(
            max(0.0, getattr(self.config.execution, "iteration_stall_hard_timeout_s", 0.0) or 0.0)
        )
        if (
            stall_hard_timeout_s > 0.0
            and stall_soft_timeout_s > 0.0
            and stall_hard_timeout_s < stall_soft_timeout_s
        ):
            self.event_logger.system_warning(
                "Iteration hard timeout was lower than soft timeout; clamping hard timeout to soft timeout",
                soft_timeout_seconds=stall_soft_timeout_s,
                hard_timeout_seconds=stall_hard_timeout_s,
            )
            stall_hard_timeout_s = stall_soft_timeout_s
        if stall_soft_timeout_s > 0.0 or stall_hard_timeout_s > 0.0:
            self.event_logger.system_info(
                "Iteration stall watchdog enabled",
                soft_timeout_seconds=stall_soft_timeout_s,
                hard_timeout_seconds=stall_hard_timeout_s,
            )
        watchdog_state: Dict[str, Any] = {
            "iteration": 0,
            "started_at_monotonic": 0.0,
            "result_count": 0,
            "soft_triggered": False,
            "hard_triggered": False,
            "hard_reason": "",
            "last_stage": "idle",
        }
        watchdog_lock = threading.Lock()
        watchdog_stop_event = threading.Event()
        watchdog_thread: Optional[threading.Thread] = None
        watchdog_poll_interval_s = 0.25
        watchdog_suppressed_tools = {"ask_user", "extract_data"}

        def _is_watchdog_suppressed_stage(stage: str) -> bool:
            stage_value = str(stage or "").strip().lower()
            if not stage_value.startswith("before_action:"):
                return False
            tool_name = stage_value.split(":", 1)[1].strip()
            return tool_name in watchdog_suppressed_tools

        class _IterationHardTimeout(Exception):
            """Raised when an iteration exceeds hard watchdog timeout before first result."""
            pass

        def _emit_soft_watchdog_warning(
            *,
            iteration: int,
            stage: str,
            elapsed_s: float,
            source: str,
            invoke_stuck_callback: bool,
        ) -> None:
            self.event_logger.system_warning(
                "Iteration appears stalled: soft timeout reached before first action result",
                iteration=iteration,
                stage=stage,
                elapsed_seconds=round(elapsed_s, 3),
                soft_timeout_seconds=stall_soft_timeout_s,
                source=source,
            )
            with self._hints_lock:
                self._pending_hints.append(
                    "This iteration appears stalled. Switch strategy, avoid repeating the same attempt, "
                    "and produce a concrete result."
                )

            if invoke_stuck_callback and self.on_stuck_callback:
                try:
                    hint = self.on_stuck_callback(
                        (
                            "Iteration stalled: no action result produced after "
                            f"{elapsed_s:.1f}s."
                        ),
                        iteration,
                    )
                    if hint:
                        with self._hints_lock:
                            self._pending_hints.append(hint.strip())
                except Exception as e:
                    self.event_logger.system_warning(f"on_stuck_callback failed: {e}")

        def _watchdog_poll_loop() -> None:
            while not watchdog_stop_event.wait(watchdog_poll_interval_s):
                if stall_soft_timeout_s <= 0.0 and stall_hard_timeout_s <= 0.0:
                    continue

                with watchdog_lock:
                    started_at_monotonic = float(watchdog_state.get("started_at_monotonic", 0.0) or 0.0)
                    result_count = int(watchdog_state.get("result_count", 0) or 0)
                    soft_triggered = bool(watchdog_state.get("soft_triggered", False))
                    hard_triggered = bool(watchdog_state.get("hard_triggered", False))
                    iteration = int(watchdog_state.get("iteration", 0) or 0)
                    stage = str(watchdog_state.get("last_stage", "watchdog_poll") or "watchdog_poll")

                if started_at_monotonic <= 0.0 or result_count > 0:
                    continue

                elapsed_s = max(0.0, time.monotonic() - started_at_monotonic)
                if _is_watchdog_suppressed_stage(stage):
                    continue

                if stall_soft_timeout_s > 0.0 and (not soft_triggered) and elapsed_s >= stall_soft_timeout_s:
                    should_emit_soft = False
                    with watchdog_lock:
                        if (
                            int(watchdog_state.get("result_count", 0) or 0) <= 0
                            and not bool(watchdog_state.get("soft_triggered", False))
                        ):
                            watchdog_state["soft_triggered"] = True
                            should_emit_soft = True
                            iteration = int(watchdog_state.get("iteration", 0) or 0)
                            stage = str(watchdog_state.get("last_stage", stage) or stage)
                    if should_emit_soft:
                        _emit_soft_watchdog_warning(
                            iteration=iteration,
                            stage=stage,
                            elapsed_s=elapsed_s,
                            source="realtime_watchdog",
                            invoke_stuck_callback=False,
                        )

                if stall_hard_timeout_s > 0.0 and (not hard_triggered) and elapsed_s >= stall_hard_timeout_s:
                    hard_reason = (
                        "Iteration hard timeout reached before first action result "
                        f"({elapsed_s:.1f}s >= {stall_hard_timeout_s:.1f}s)."
                    )
                    should_mark_hard = False
                    with watchdog_lock:
                        if (
                            int(watchdog_state.get("result_count", 0) or 0) <= 0
                            and not bool(watchdog_state.get("hard_triggered", False))
                        ):
                            watchdog_state["hard_triggered"] = True
                            watchdog_state["hard_reason"] = hard_reason
                            should_mark_hard = True
                            iteration = int(watchdog_state.get("iteration", 0) or 0)
                            stage = str(watchdog_state.get("last_stage", stage) or stage)
                    if should_mark_hard:
                        self.event_logger.system_warning(
                            "Iteration hard timeout reached before first action result",
                            iteration=iteration,
                            stage=stage,
                            elapsed_seconds=round(elapsed_s, 3),
                            hard_timeout_seconds=stall_hard_timeout_s,
                            source="realtime_watchdog",
                        )

        def _ensure_watchdog_thread() -> None:
            nonlocal watchdog_thread
            if stall_soft_timeout_s <= 0.0 and stall_hard_timeout_s <= 0.0:
                return
            if watchdog_thread is not None and watchdog_thread.is_alive():
                return
            watchdog_stop_event.clear()
            watchdog_thread = threading.Thread(
                target=_watchdog_poll_loop,
                name="iteration-watchdog",
                daemon=True,
            )
            watchdog_thread.start()

        def _stop_watchdog_thread() -> None:
            nonlocal watchdog_thread
            watchdog_stop_event.set()
            current_thread = threading.current_thread()
            thread_ref = watchdog_thread
            watchdog_thread = None
            if thread_ref is not None and thread_ref.is_alive() and thread_ref is not current_thread:
                thread_ref.join(timeout=1.0)

        def _start_iteration_watchdog(iteration: int) -> None:
            with watchdog_lock:
                watchdog_state["iteration"] = int(iteration or 0)
                watchdog_state["started_at_monotonic"] = time.monotonic()
                watchdog_state["result_count"] = 0
                watchdog_state["soft_triggered"] = False
                watchdog_state["hard_triggered"] = False
                watchdog_state["hard_reason"] = ""
                watchdog_state["last_stage"] = "iteration_start"
            _ensure_watchdog_thread()

        def _check_iteration_watchdog(stage: str) -> None:
            if stall_soft_timeout_s <= 0.0 and stall_hard_timeout_s <= 0.0:
                return
            with watchdog_lock:
                watchdog_state["last_stage"] = stage
                hard_triggered = bool(watchdog_state.get("hard_triggered", False))
                hard_reason = str(watchdog_state.get("hard_reason", "") or "")
                started_at_monotonic = float(watchdog_state.get("started_at_monotonic", 0.0) or 0.0)
                result_count = int(watchdog_state.get("result_count", 0) or 0)
                soft_triggered = bool(watchdog_state.get("soft_triggered", False))
                iteration = int(watchdog_state.get("iteration", 0) or 0)

            if hard_triggered:
                raise _IterationHardTimeout(
                    hard_reason.strip()
                    or "Iteration hard timeout reached before first action result."
                )
            if started_at_monotonic <= 0.0:
                return
            if result_count > 0:
                return

            elapsed_s = max(0.0, time.monotonic() - started_at_monotonic)
            if _is_watchdog_suppressed_stage(stage):
                return

            if (
                stall_soft_timeout_s > 0.0
                and not soft_triggered
                and elapsed_s >= stall_soft_timeout_s
            ):
                should_emit_soft = False
                with watchdog_lock:
                    if (
                        int(watchdog_state.get("result_count", 0) or 0) <= 0
                        and not bool(watchdog_state.get("soft_triggered", False))
                    ):
                        watchdog_state["soft_triggered"] = True
                        should_emit_soft = True
                        iteration = int(watchdog_state.get("iteration", iteration) or iteration)
                if should_emit_soft:
                    _emit_soft_watchdog_warning(
                        iteration=iteration,
                        stage=stage,
                        elapsed_s=elapsed_s,
                        source="checkpoint",
                        invoke_stuck_callback=True,
                    )

            if stall_hard_timeout_s > 0.0 and elapsed_s >= stall_hard_timeout_s:
                hard_reason = (
                    "Iteration hard timeout reached before first action result "
                    f"({elapsed_s:.1f}s >= {stall_hard_timeout_s:.1f}s)."
                )
                with watchdog_lock:
                    if (
                        int(watchdog_state.get("result_count", 0) or 0) <= 0
                        and not bool(watchdog_state.get("hard_triggered", False))
                    ):
                        watchdog_state["hard_triggered"] = True
                        watchdog_state["hard_reason"] = hard_reason
                raise _IterationHardTimeout(
                    hard_reason
                )

        def _mission_result(*, success: bool, reasoning: str, narrative: str, state: ExecutionState) -> MissionResult:
            _stop_watchdog_thread()
            self._reset_speculative_state()
            if not success and not state.failure_code:
                self._record_failure(
                    state,
                    code="mission_failed",
                    stage="execution_loop",
                )
            self._persist_resume_checkpoint(
                mission=mission,
                state=state,
                status="success" if success else "failed",
            )
            return self._build_mission_result(
                success=success,
                reasoning=reasoning,
                narrative=narrative,
                state=state,
            )

        def _record_action_result(
            action_step: Any,
            *,
            success: bool,
            result_str: str,
            summary: Optional[str] = None,
            error: Optional[str] = None,
            extra: Optional[dict[str, Any]] = None,
        ) -> None:
            with watchdog_lock:
                watchdog_state["result_count"] = int(watchdog_state.get("result_count", 0) or 0) + 1
            return

        _refresh_budget_state()

        while state.total_actions < max_actions:
            if self._cancel_event.is_set():
                return _mission_result(
                    success=False,
                    reasoning="Mission cancelled",
                    narrative="Mission cancelled",
                    state=state,
                )

            self._pause_event.wait()
            if self._cancel_event.is_set():
                return _mission_result(
                    success=False,
                    reasoning="Mission cancelled",
                    narrative="Mission cancelled",
                    state=state,
                )

            iteration_started_at = time.time()
            stage_timings_ms: Dict[str, float] = {
                "load_wait_ms": 0.0,
                "dom_read_ms": 0.0,
                "screenshot_ms": 0.0,
                "plan_ms": 0.0,
            }
            iteration_tool_latency_ms = 0.0
            iteration_navigation_latency_ms = 0.0
            iteration_tool_calls = 0
            iteration_llm_latency_ms = 0.0
            iteration_tokens_in = 0
            iteration_tokens_out = 0
            iteration_image_count = 0
            iteration_retries = 0
            state.total_actions += 1
            _refresh_budget_state()
            self._current_iteration += 1
            self.execution_state = state
            _start_iteration_watchdog(self._current_iteration)
            self.event_logger.iteration_start(
                iteration=self._current_iteration,
                max_iterations=max_actions,
                mission=mission,
                budget_spent=state.budget_spent,
                budget_remaining=state.budget_remaining,
                budget_phase=state.budget_phase,
                low_budget_mode=state.low_budget_mode,
            )
            iteration_hint_path = "disabled"
            iteration_hint_candidate_id: Optional[str] = None
            iteration_hint_validation_ms = 0.0
            iteration_hint_confidence = 0.0
            iteration_hint_reject_reason: Optional[str] = None
            iteration_planner_hint_status: Optional[str] = None
            iteration_planner_hint_reason: Optional[str] = None
            iteration_planner_hint_confidence = 0.0
            hint_bundle: Optional[HintBundle] = None

            try:
                _check_iteration_watchdog(stage="iteration_start")
                try:
                    load_wait_started = time.perf_counter()
                    self._maybe_wait_for_iteration_load(reason="iteration")
                    stage_timings_ms["load_wait_ms"] = (time.perf_counter() - load_wait_started) * 1000.0
                    _check_iteration_watchdog(stage="before_snapshot")
                    snapshot = self._capture_snapshot(full_page=False)
                    snapshot_url = str(getattr(snapshot, "url", "") or "").strip()
                    if snapshot_url:
                        current_url_decision = self.sandbox_policy.check_url(snapshot_url)
                        if not current_url_decision.allowed:
                            warning = f"Current page blocked by sandbox: {current_url_decision.reason}"
                            self.event_logger.system_warning(
                                warning,
                                current_url=snapshot_url,
                                sandbox_preset=self.sandbox_policy.preset,
                                sandbox_mode=self.sandbox_policy.mode,
                            )
                            if self.sandbox_policy.enforce:
                                state.last_action_summary = f"sandbox FAILED: {warning} ({snapshot_url})"
                                _append_recent_action(state.last_action_summary)
                                return _mission_result(
                                    success=False,
                                    reasoning=f"{warning} ({snapshot_url})",
                                    narrative="Sandbox blocked disallowed current page",
                                    state=state,
                                )
                            state.last_action_summary = f"sandbox(observe): {warning} ({snapshot_url})"
                            _append_recent_action(state.last_action_summary)
                    dom_read_started = time.perf_counter()
                    snapshot_hash = str(getattr(snapshot, "screenshot_hash", "") or "")
                    snapshot_fingerprint = "|".join(
                        [
                            snapshot_url,
                            str(getattr(snapshot, "title", "") or ""),
                            str(getattr(snapshot, "scroll_x", 0) or 0),
                            str(getattr(snapshot, "scroll_y", 0) or 0),
                            snapshot_hash,
                        ]
                    )
                    can_reuse_dom = (
                        (not state.in_loop)
                        and bool(snapshot_fingerprint)
                        and snapshot_fingerprint == self._cached_snapshot_fingerprint
                        and self._cached_page_info is not None
                        and self._cached_detected_elements is not None
                    )
                    if can_reuse_dom:
                        page_info = self._cached_page_info
                        detected_elements = self._cached_detected_elements
                        self.event_logger.system_debug(
                            "Reused cached DOM/page state for iteration",
                            iteration=self._current_iteration,
                        )
                    else:
                        page_info = self.page_utils.get_page_info()
                        detected_elements = build_page_elements(self.browser.page, page_info)
                        self._cached_snapshot = snapshot
                        self._cached_snapshot_fingerprint = snapshot_fingerprint
                        self._cached_page_info = page_info
                        self._cached_detected_elements = detected_elements
                    stage_timings_ms["dom_read_ms"] = (time.perf_counter() - dom_read_started) * 1000.0
                except _IterationHardTimeout:
                    raise
                except Exception as e:
                    self._record_failure(
                        state,
                        code="state_capture_failed",
                        stage="iteration_state_capture",
                    )
                    return _mission_result(
                        success=False,
                        reasoning=f"Failed to capture state: {str(e)}",
                        state=state,
                        narrative="Failed to capture state",
                    )

                # --- Phase 1: Build element index + start gallery in background ---
                screenshot_started = time.perf_counter()
                prep = self._build_element_index_and_start_gallery(snapshot, detected_elements)
                annotated_screenshot_bytes = prep.screenshot_bytes
                element_index_text = prep.element_index_text

                self.event_logger.element_capture(
                    total=len(getattr(detected_elements, "elements", []) or []),
                    text_rich=prep.text_rich_count,
                    text_poor=prep.text_poor_count,
                )

                # --- Phase 2: Build context while gallery generates in background ---
                memory_entries = list(self.memory_store.entries)
                executed_ids = [
                    entry.memory_id
                    for entry in memory_entries
                    if entry.entry_kind == MemoryEntryKind.EXECUTED_ACTION.value
                ]
                reflection_ids = [
                    entry.memory_id
                    for entry in memory_entries
                    if entry.entry_kind == MemoryEntryKind.REFLECTION.value
                ]
                recent_executed_ids = executed_ids[-DECISION_CONTEXT_EXECUTED_ID_WINDOW:]
                recent_reflection_ids = reflection_ids[-DECISION_CONTEXT_REFLECTION_ID_WINDOW:]
                recommended_step, recommended_step_source_id = self._get_latest_recommended_next_step()

                decision_context = DecisionContext(
                    action_iteration=state.total_actions,
                    mission=mission,
                    current_url=snapshot.url,
                    page_title=snapshot.title,
                    recommended_next_step=recommended_step,
                    recommended_from_memory_id=recommended_step_source_id,
                    executed_memory_ids=recent_executed_ids,
                    executed_memory_older_count=max(0, len(executed_ids) - len(recent_executed_ids)),
                    reflection_memory_ids=recent_reflection_ids,
                    reflection_memory_older_count=max(0, len(reflection_ids) - len(recent_reflection_ids)),
                    budget_spent=state.budget_spent,
                    budget_remaining=state.budget_remaining,
                    budget_total=state.budget_total,
                    budget_phase=state.budget_phase,
                    low_budget_mode=state.low_budget_mode,
                )

                _narrative_n = max(1, int(
                    len(memory_entries)
                    * float(self.config.execution.memory_narrative_recent_percent or 1.0)
                ))
                environment_state = EnvironmentState(
                    browser_state=snapshot,
                    memory_narrative=self.memory_store.get_narrative(n=_narrative_n),
                    user_prompt=mission,
                    mission_start_url=self.mission_start_url,
                    mission_start_time=self.mission_start_time,
                    current_url=snapshot.url,
                    page_title=snapshot.title,
                    url_history=self.memory_store.url_history.copy(),
                    url_pointer=self.memory_store.url_pointer,
                    decision_context=decision_context,
                )

                tab_bar = None
                dialog_notice = None
                tab_events = []
                dialog_pending = False
                if self.tab_manager:
                    self.tab_manager.refresh_metadata()
                    tab_bar = self.tab_manager.build_tab_bar()
                    dialog_notice = self.tab_manager.build_dialog_notice()
                    tab_events = self.tab_manager.drain_events()
                    dialog_pending = self.tab_manager.has_pending_dialog_on_active()

                with self._hints_lock:
                    pending_hints = list(self._pending_hints)
                    self._pending_hints.clear()

                policy_constraints_block = None
                policy_parts: List[str] = []
                if self.config.sandbox.prompt.include_policy_block:
                    policy_parts.append(self.sandbox_policy.render_prompt_policy_block())
                policy_parts.append(self.effect_policy.render_prompt_policy_block())
                if policy_parts:
                    policy_constraints_block = "\n\n".join(part for part in policy_parts if str(part or "").strip())
                if (
                    self.config.debug.debug_mode
                    and state.budget_constraints_enabled
                    and state.low_budget_mode
                ):
                    self.event_logger.system_debug(
                        "Low-budget mode active; using single-action planning",
                        budget_phase=state.budget_phase,
                        budget_remaining=state.budget_remaining,
                        planning_batch_limit=state.planning_batch_limit,
                    )
                # Build tool outputs for any pending function calls from the last response.
                # The Responses API requires outputs for all function calls before new input.
                # Only chain (use previous_response_id) when we have the call IDs to satisfy it;
                # if extraction ever failed, fall back to a fresh non-chained call.
                pending_tool_outputs: Optional[list] = None
                effective_response_id: Optional[str] = None
                planner_response_chaining_enabled = bool(
                    getattr(self.config.execution, "use_previous_response_id", True)
                )
                if (
                    planner_response_chaining_enabled
                    and state.last_response_id
                    and state.last_tool_call_ids
                ):
                    pending_tool_outputs = [
                        {"type": "function_call_output", "call_id": cid, "output": "ok"}
                        for cid in state.last_tool_call_ids
                    ]
                    effective_response_id = state.last_response_id

                # --- Phase 3: Collect gallery result (blocks if still running) ---
                self._collect_gallery(prep)
                gallery_images = prep.gallery_images
                # Adaptive: use high detail only when many visual/crop elements present
                adaptive_image_detail = "high" if len(gallery_images or []) > 5 else "low"
                stage_timings_ms["screenshot_ms"] = (time.perf_counter() - screenshot_started) * 1000.0
                active_skill_context = self._build_active_skill_context(state)

                action_planner = ActionPlanner(
                    mission,
                    self.memory_store,
                    base_knowledge=self.base_knowledge,
                    model_name=self.agent_model_name,
                    reasoning_level=self.agent_reasoning_level,
                    image_detail=adaptive_image_detail,
                    max_actions_per_plan=state.planning_batch_limit,
                    previous_response_id=effective_response_id,
                    tool_call_outputs=pending_tool_outputs,
                    notebook_last_sent_index=(
                        state.notebook_entries_sent if effective_response_id else 0
                    ),
                    memory_narrative_n=_narrative_n,
                    last_action_summary=state.last_action_summary,
                    tab_bar=tab_bar,
                    dialog_notice=dialog_notice,
                    tab_events=tab_events,
                    dialog_pending=dialog_pending,
                    recommended_next_step=recommended_step,
                    recommended_next_step_source_id=recommended_step_source_id,
                    decision_context=decision_context,
                    element_index_text=element_index_text,
                    gallery_images=gallery_images,
                    current_iteration=self._current_iteration,
                    user_facing_actions_in_round=state.user_facing_actions_since_progress,
                    user_hints=pending_hints,
                    policy_constraints_block=policy_constraints_block,
                    in_loop=state.in_loop,
                    loop_round=state.loop_round,
                    loop_count=state.loop_count,
                    loop_description=state.loop_description,
                    recent_actions=state.recent_actions,
                    agent_notes=state.agent_notes,
                    iterations_remaining=state.budget_remaining,
                    max_iterations=state.budget_total,
                    budget_spent=state.budget_spent,
                    budget_phase=state.budget_phase,
                    low_budget_mode=state.low_budget_mode,
                    budget_constraints_enabled=state.budget_constraints_enabled,
                    tool_registry=self.tool_registry,
                    effect_policy_engine=self.effect_policy,
                    event_definitions=self.event_definitions,
                    agent_events_status=self._build_agent_events_status(),
                    available_skills_metadata=self.available_skills_catalog,
                    active_skill_context=active_skill_context,
                    active_skill_name=state.active_skill_name,
                    workspace_files=self._list_workspace_files(),
                )

                actions_list: Optional[list] = None
                error: Optional[str] = None
                # Consume planner-embedded hint for this iteration (if available).
                hint_bundle = self._consume_or_discard_cached_planner_hint(
                    expected_source_iteration=max(0, self._current_iteration - 1),
                )
                _check_iteration_watchdog(stage="before_planner_call")
                try:
                    resolution = self._resolve_actions_with_speculative_validation(
                        action_planner=action_planner,
                        environment_state=environment_state,
                        screenshot=annotated_screenshot_bytes,
                        notebook=self.notebook,
                        detected_elements=detected_elements,
                        snapshot=snapshot,
                        element_index_text=element_index_text,
                        state=state,
                        hint_bundle=hint_bundle,
                        dialog_pending=dialog_pending,
                    )
                    planner_outcome = resolution.planner
                    actions_list = resolution.actions
                    error = resolution.error

                    stage_timings_ms["plan_ms"] += float(planner_outcome.planner_elapsed_ms or 0.0)
                    iteration_llm_latency_ms += float(planner_outcome.llm_latency_ms or 0.0)
                    iteration_tokens_in += int(planner_outcome.tokens_in or 0)
                    iteration_tokens_out += int(planner_outcome.tokens_out or 0)
                    iteration_image_count += int(planner_outcome.image_count or 0)
                    iteration_retries += int(planner_outcome.retries or 0)

                    iteration_hint_path = str(resolution.hint_path or "planner_only")
                    iteration_hint_candidate_id = (
                        str(resolution.hint_candidate_id or "").strip() or None
                    )
                    iteration_hint_validation_ms = float(resolution.hint_validation_ms or 0.0)
                    iteration_hint_confidence = float(resolution.hint_confidence or 0.0)
                    iteration_hint_reject_reason = (
                        str(resolution.hint_reject_reason or "").strip() or None
                    )
                    iteration_planner_hint_status = (
                        str(getattr(planner_outcome, "hint_status", "") or "").strip() or None
                    )
                    iteration_planner_hint_reason = (
                        str(getattr(planner_outcome, "hint_reason", "") or "").strip() or None
                    )
                    iteration_planner_hint_confidence = float(
                        getattr(planner_outcome, "hint_confidence", 0.0) or 0.0
                    )

                    if planner_response_chaining_enabled and planner_outcome.response_id:
                        state.last_response_id = planner_outcome.response_id
                        state.last_tool_call_ids = list(planner_outcome.tool_call_ids or [])
                        state.notebook_entries_sent = len(self.notebook)
                    elif (
                        str(iteration_hint_path or "").startswith("hint_accept")
                        or not planner_response_chaining_enabled
                    ):
                        # Keep chain state coherent when planner output is bypassed.
                        clear_planner_response_chain(state)
                    self._cache_planner_hint_bundle(planner_outcome.hint_bundle)
                    _check_iteration_watchdog(stage="after_planner_call")
                except _IterationHardTimeout:
                    raise
                except Exception as e:
                    self._record_failure(
                        state,
                        code="planner_call_exception",
                        stage="planner_model_call",
                    )
                    return _mission_result(
                        success=False,
                        reasoning=f"Error: {str(e)}",
                        state=state,
                        narrative="Failed to get next actions",
                    )

                if not actions_list:
                    planner_failure_code = str(getattr(planner_outcome, "failure_code", "") or "").strip()
                    planner_failure_stage = str(getattr(planner_outcome, "failure_stage", "") or "").strip()
                    if planner_failure_code:
                        self._record_failure(
                            state,
                            code=planner_failure_code,
                            stage=planner_failure_stage or "planner_generation",
                        )
                    if planner_failure_code == "tool_policy_filtered_empty":
                        return _mission_result(
                            success=False,
                            reasoning=error or "No tools are enabled for the active effect policy.",
                            state=state,
                            narrative="Active effect policy produced an empty planner tool set for this mode.",
                        )
                    if (
                        planner_failure_code == "planner_generation_failed"
                        and is_context_length_exceeded_error(error)
                    ):
                        had_chain_state = bool(state.last_response_id or state.last_tool_call_ids)
                        clear_planner_response_chain(state)
                        if had_chain_state:
                            self.event_logger.system_warning(
                                "Planner context overflow detected; cleared response chain and retrying fresh",
                                iteration=self._current_iteration,
                            )
                            state.last_action_summary = (
                                "Planner context overflow detected. Cleared response chain and retrying with a fresh prompt."
                            )
                            continue
                    state.validation_failures += 1
                    if state.validation_failures <= self.config.execution.validation_failure_escalation_limit:
                        state.last_action_summary = f"Action validation issue: {error or 'No action generated'}. Retrying."
                        continue
                    self._record_failure(
                        state,
                        code="action_validation_failure",
                        stage="planner_validation",
                    )
                    return _mission_result(
                        success=False,
                        reasoning=f"Repeated action validation failures: {error or 'No action generated'}",
                        state=state,
                        narrative="Repeated action validation failures",
                    )

                if not str(iteration_hint_path or "").startswith("hint_accept"):
                    merge_agent_notes(state, action_planner.new_notes)

                for action_step in actions_list:
                    if self._cancel_event.is_set():
                        return _mission_result(
                            success=False,
                            reasoning="Mission cancelled",
                            narrative="Mission cancelled",
                            state=state,
                        )
                    self._pause_event.wait()
                    if self._cancel_event.is_set():
                        return _mission_result(
                            success=False,
                            reasoning="Mission cancelled",
                            narrative="Mission cancelled",
                            state=state,
                        )

                    function_name = (getattr(action_step, "function_name", None) or "").strip()
                    action_args = dict(getattr(action_step, "function_arguments", {}) or {})
                    current_action = getattr(action_step, "action", "") or function_name
                    reasoning = action_args.get("reasoning", "")
                    narrative = action_args.get("narrative", "")
                    pending_events = self._collect_pending_agent_events(
                        action_step=action_step,
                        action_args=action_args,
                    )
                    iteration_tool_calls += 1
                    action_id = f"it{self._current_iteration}_a{iteration_tool_calls}"
                    _check_iteration_watchdog(stage=f"before_action:{function_name or 'unknown'}")
                    if self.config.debug.debug_mode:
                        self.event_logger.system_debug(
                            "Budget telemetry",
                            tool=function_name,
                            budget_phase=state.budget_phase,
                            budget_remaining=state.budget_remaining,
                            budget_spent=state.budget_spent,
                            budget_total=state.budget_total,
                        )

                    self.event_logger.action_determined(
                        action=current_action,
                        reasoning=reasoning,
                        narrative=narrative,
                        tool=function_name,
                        budget_phase=state.budget_phase,
                        budget_remaining=state.budget_remaining,
                        budget_spent=state.budget_spent,
                        budget_total=state.budget_total,
                        in_loop=state.in_loop,
                        loop_round=state.loop_round if state.in_loop else None,
                        loop_count=state.loop_count if state.in_loop else None,
                    )

                    if not function_name:
                        tool_error = "Missing function_name on action step."
                        self._record_failure(
                            state,
                            code="invalid_action_step",
                            stage="runtime_action_validation",
                        )
                        state.last_action_summary = tool_error
                        _append_recent_action(tool_error)
                        return _mission_result(
                            success=False,
                            reasoning=tool_error,
                            narrative="Planner produced an invalid action step.",
                            state=state,
                        )

                    tool_spec = self.tool_registry.get(function_name)
                    if tool_spec is None:
                        tool_error = f"Tool '{function_name}' is not registered."
                        self._record_failure(
                            state,
                            code="unknown_tool",
                            stage="runtime_tool_lookup",
                        )
                        state.last_action_summary = tool_error
                        _append_recent_action(tool_error)
                        return _mission_result(
                            success=False,
                            reasoning=tool_error,
                            narrative="Planner called an unregistered tool.",
                            state=state,
                        )

                    runtime_adapter = ToolRuntimeAdapter(
                        agent=self,
                        detected_elements=detected_elements,
                        page_info=page_info,
                        environment_state=environment_state,
                        current_iteration=self._current_iteration,
                    )
                    tool_ctx = ToolContext(
                        page=self.browser.page if self.browser else None,
                        elements=detected_elements,
                        page_info=page_info,
                        environment_state=environment_state,
                        memory_store=self.memory_store,
                        event_logger=self.event_logger,
                        sandbox_policy=self.sandbox_policy,
                        action_step=action_step,
                        runtime_state=runtime_adapter,
                    )
                    tool_started_at = time.perf_counter()
                    outcome = self.tool_engine.execute(function_name, action_args, tool_ctx)
                    duration_ms = (time.perf_counter() - tool_started_at) * 1000.0
                    iteration_tool_latency_ms += max(0.0, duration_ms)
                    if Effect.NAVIGATE_WEB in tool_spec.manifest.effects:
                        iteration_navigation_latency_ms += max(0.0, duration_ms)

                    state.actions_since_progress += 1
                    result_success = bool(getattr(outcome.output, "success", False))
                    result_error = str(getattr(outcome.output, "error", "") or "").strip() or None
                    result_data = (
                        outcome.output.data
                        if isinstance(getattr(outcome.output, "data", None), dict)
                        else {}
                    )

                    policy_observe_warning = ""
                    post_nav_sensitive_functions = {
                        "click",
                        "press_key",
                        "open_url",
                        "go_back",
                        "go_forward",
                    }
                    nav_break = False
                    if result_success and function_name in post_nav_sensitive_functions:
                        try:
                            current_url = self.browser.page.url if self.browser and self.browser.page else ""
                        except Exception:
                            current_url = ""
                        post_nav_decision = self.sandbox_policy.check_url(current_url)
                        if not post_nav_decision.allowed:
                            warning = f"Post-navigation URL blocked by sandbox: {post_nav_decision.reason}"
                            self.event_logger.system_warning(warning)
                            if self.sandbox_policy.enforce:
                                result_success = False
                                result_error = warning
                            else:
                                policy_observe_warning = warning
                        if current_url and current_url != snapshot_url:
                            nav_break = True

                    mission_done_reasoning: Optional[str] = None
                    should_replan_after_control = False
                    event_summary_text = ""
                    event_hint_lines: List[str] = []
                    if function_name == "think" and result_success and outcome.control is None:
                        control_error = "think must return a control payload."
                        self._record_failure(
                            state,
                            code="missing_think_control",
                            stage="runtime_tool_control",
                        )
                        state.last_action_summary = control_error
                        _append_recent_action(state.last_action_summary)
                        return _mission_result(
                            success=False,
                            reasoning=control_error,
                            narrative="Think tool did not provide control instructions.",
                            state=state,
                        )
                    if outcome.control is not None:
                        if function_name != "think":
                            control_error = (
                                f"Tool '{function_name}' returned control payload, but only think may control mission flow."
                            )
                            self._record_failure(
                                state,
                                code="invalid_control_payload",
                                stage="runtime_tool_control",
                            )
                            state.last_action_summary = control_error
                            _append_recent_action(state.last_action_summary)
                            return _mission_result(
                                success=False,
                                reasoning=control_error,
                                narrative="Non-think tool attempted to control mission flow.",
                                state=state,
                            )
                        if result_success:
                            mission_done_reasoning, should_replan_after_control = self._apply_think_control(
                                control=outcome.control,
                                state=state,
                                append_recent_action=_append_recent_action,
                                exit_loop=_exit_loop,
                            )

                    if pending_events:
                        event_results = self._dispatch_agent_events(
                            pending_events=pending_events,
                            function_name=function_name,
                            action_id=action_id,
                            action_success=result_success,
                        )
                        event_summary_text, event_hint_lines = self._summarize_event_results(event_results)
                        if event_hint_lines:
                            with self._hints_lock:
                                self._pending_hints.extend(event_hint_lines)

                    result_str = "success" if result_success else "failed"
                    summary = str(getattr(outcome.output, "summary", "") or "").strip()
                    if not summary:
                        summary = (
                            f"{narrative} ({result_str})"
                            if narrative
                            else self._build_action_summary(action_step, result_str)
                        )
                    if (
                        function_name == "think"
                        and outcome.control is not None
                        and outcome.control.next_action != ThinkNextAction.CONTINUE
                    ):
                        control_summary = str(state.last_action_summary or "").strip()
                        if control_summary:
                            summary = control_summary
                    if policy_observe_warning:
                        summary = f"{summary} | sandbox(observe): {policy_observe_warning}" if summary else policy_observe_warning
                    if result_error:
                        summary = f"{summary} | {result_error}" if summary else result_error
                    if event_summary_text:
                        summary = f"{summary} | events: {event_summary_text}" if summary else f"events: {event_summary_text}"
                    state.last_action_summary = summary
                    _append_recent_action(summary)

                    if function_name == "ask_user" and result_success:
                        ask_question = str(action_args.get("question", "")).strip()
                        ask_answer = ""
                        ask_status = ""
                        if isinstance(result_data, dict):
                            ask_answer = str(result_data.get("answer", "") or "").strip()
                            ask_status = str(result_data.get("status", "") or "").strip().lower()
                            if not ask_question:
                                ask_question = str(result_data.get("question", "") or "").strip()
                        if ask_status == "skipped":
                            ask_answer = "(user skipped)"
                            if isinstance(result_data, dict):
                                result_data["answer"] = ask_answer
                        elif not ask_answer:
                            recent_pairs = self.memory_store.get_recent_question_answers(n=1)
                            if recent_pairs:
                                ask_answer = str(recent_pairs[0].get("answer", "") or "").strip()
                                if not ask_question:
                                    ask_question = str(recent_pairs[0].get("question", "") or "").strip()
                        hint = ""
                        if ask_status == "skipped":
                            if ask_question:
                                hint = (
                                    f'User skipped the question: "{ask_question}". '
                                    "Do not ask again; proceed with best judgment."
                                )
                            else:
                                hint = "User skipped the question. Do not ask again; proceed with best judgment."
                        elif ask_answer:
                            if ask_question:
                                hint = (
                                    f'User answer received for "{ask_question}": "{ask_answer}". '
                                    "Use this answer to continue and avoid asking the same question again."
                                )
                            else:
                                hint = (
                                    f'User answer received: "{ask_answer}". '
                                    "Use this answer to continue and avoid re-asking it."
                                )
                        if hint:
                            with self._hints_lock:
                                self._pending_hints.append(hint)

                    self.event_logger.action_complete(
                        tool=function_name,
                        narrative=narrative,
                        success=result_success,
                        result_str=result_str,
                        duration_ms=duration_ms,
                        iteration=self._current_iteration,
                    )
                    _record_action_result(
                        action_step,
                        success=result_success,
                        result_str=result_str,
                        summary=state.last_action_summary,
                        error=result_error,
                        extra={"data": result_data},
                    )

                    if result_success and str(getattr(tool_spec.manifest.progress_policy, "value", "")) == "USER_FACING":
                        if function_name == "report_data":
                            reported = bool((result_data or {}).get("reported", True))
                            if reported:
                                state.user_facing_actions_since_progress += 1
                        elif function_name == "send_email":
                            duplicate_of = (result_data or {}).get("duplicate_of")
                            if not duplicate_of:
                                state.user_facing_actions_since_progress += 1
                        else:
                            state.user_facing_actions_since_progress += 1

                        if state.in_loop:
                            overlay_index = action_args.get("element_id")
                            if overlay_index is not None:
                                try:
                                    self.action_executor.mark_element_done(int(overlay_index))
                                except (TypeError, ValueError):
                                    pass

                    if result_success and self.tab_manager:
                        active_tab = self.tab_manager.get_active()
                        if active_tab and self.browser.page is not active_tab.page:
                            self.action_executor.set_page(active_tab.page)

                    if result_success and mission_done_reasoning:
                        if state.in_loop:
                            self.event_logger.loop_state_changed(
                                change="end",
                                loop_round=state.loop_round,
                                loop_count=state.loop_count,
                                loop_description=state.loop_description,
                            )
                            _exit_loop()
                        return _mission_result(
                            success=True,
                            reasoning=mission_done_reasoning or "Mission complete",
                            narrative=narrative or state.last_action_summary or "Mission complete",
                            state=state,
                        )

                    if not result_success:
                        # If policy engine blocks at runtime, fail mission immediately.
                        if result_error and (
                            "Effects denied by preset" in result_error
                            or "Effects not allowed in preset" in result_error
                            or "blocked by effect policy" in (state.last_action_summary or "")
                        ):
                            self._record_failure(
                                state,
                                code="tool_disallowed_by_policy",
                                stage="runtime_tool_enforcement",
                            )
                            return _mission_result(
                                success=False,
                                reasoning=state.last_action_summary or result_error,
                                narrative="Runtime effect policy blocked planner action.",
                                state=state,
                            )
                        try:
                            overlay_index = action_args.get("element_id")
                            if overlay_index is None:
                                overlay_index = action_args.get("overlay_index")
                            failed_action = FailedAction(
                                action=current_action,
                                overlay_index=overlay_index,
                                url=page_info.url if page_info else "",
                                page_title=page_info.title if page_info else None,
                                timestamp=time.time(),
                            )
                            state.failed_elements.append(failed_action)
                        except Exception:
                            pass
                        break

                    if should_replan_after_control:
                        break
                    if nav_break:
                        break
                    continue

                state.validation_failures = 0

            except _IterationHardTimeout as timeout_exc:
                timeout_reason = str(timeout_exc).strip() or "Iteration hard timeout reached."
                state.last_action_summary = timeout_reason
                _append_recent_action(f"[ITERATION TIMEOUT] {timeout_reason}")
                self.event_logger.system_warning(
                    "Iteration hard timeout triggered; forcing replan on next iteration",
                    iteration=self._current_iteration,
                    timeout_reason=timeout_reason,
                    soft_timeout_seconds=stall_soft_timeout_s,
                    hard_timeout_seconds=stall_hard_timeout_s,
                )
                continue
            except Exception as e:
                self._record_failure(
                    state,
                    code="execution_loop_exception",
                    stage="iteration_action_execution",
                )
                return _mission_result(
                    success=False,
                    reasoning=f"Error: {str(e)}",
                    narrative="Error",
                    state=state,
                )
            finally:
                iteration_duration_ms = (time.time() - iteration_started_at) * 1000.0
                state.iteration_ms = float(iteration_duration_ms)
                state.llm_latency_ms = float(iteration_llm_latency_ms)
                state.tool_latency_ms = float(iteration_tool_latency_ms)
                state.navigation_latency_ms = float(iteration_navigation_latency_ms)
                state.tokens_in = int(iteration_tokens_in)
                state.tokens_out = int(iteration_tokens_out)
                state.image_count = int(iteration_image_count)
                state.tool_calls = int(iteration_tool_calls)
                state.retries = int(iteration_retries)

                state.iteration_ms_samples.append(state.iteration_ms)
                state.llm_latency_ms_samples.append(state.llm_latency_ms)
                state.tool_latency_ms_samples.append(state.tool_latency_ms)
                state.navigation_latency_ms_samples.append(state.navigation_latency_ms)
                state.tokens_in_total += state.tokens_in
                state.tokens_out_total += state.tokens_out
                state.image_count_total += state.image_count
                state.tool_call_count += state.tool_calls
                state.retry_count += state.retries
                if stage_timings_ms["plan_ms"] > 0.0:
                    state.llm_call_count += 1

                avg_iteration_ms = self._mean(state.iteration_ms_samples)
                avg_llm_ms = self._mean(state.llm_latency_ms_samples)
                avg_tool_ms = self._mean(state.tool_latency_ms_samples)
                avg_tokens_in = (
                    float(state.tokens_in_total) / float(state.llm_call_count)
                    if state.llm_call_count > 0
                    else 0.0
                )
                avg_tokens_out = (
                    float(state.tokens_out_total) / float(state.llm_call_count)
                    if state.llm_call_count > 0
                    else 0.0
                )
                avg_images = (
                    float(state.image_count_total) / float(state.llm_call_count)
                    if state.llm_call_count > 0
                    else 0.0
                )
                retries_per_mission = (
                    float(state.retry_count) / max(1.0, float(self._current_iteration or 1))
                )
                self.event_logger.iteration_complete(
                    iteration=self._current_iteration,
                    duration_ms=iteration_duration_ms,
                    iteration_ms=round(state.iteration_ms, 3),
                    mission_ms=round(sum(state.iteration_ms_samples), 3),
                    llm_latency_ms=round(state.llm_latency_ms, 3),
                    tool_latency_ms=round(state.tool_latency_ms, 3),
                    navigation_latency_ms=round(state.navigation_latency_ms, 3),
                    tokens_in=state.tokens_in,
                    tokens_out=state.tokens_out,
                    image_count=state.image_count,
                    tool_calls=state.tool_calls,
                    retries=state.retries,
                    avg_iteration_ms=round(avg_iteration_ms, 3),
                    p95_iteration_ms=round(self._p95(state.iteration_ms_samples), 3),
                    avg_llm_ms=round(avg_llm_ms, 3),
                    avg_tool_ms=round(avg_tool_ms, 3),
                    avg_tokens_in=round(avg_tokens_in, 3),
                    avg_tokens_out=round(avg_tokens_out, 3),
                    avg_images_per_call=round(avg_images, 3),
                    retries_per_mission=round(retries_per_mission, 3),
                    hint_path=iteration_hint_path,
                    hint_candidate_id=iteration_hint_candidate_id,
                    hint_validation_ms=round(iteration_hint_validation_ms, 3),
                    hint_confidence=round(iteration_hint_confidence, 3),
                    hint_reject_reason=iteration_hint_reject_reason,
                    planner_hint_status=iteration_planner_hint_status,
                    planner_hint_reason=iteration_planner_hint_reason,
                    planner_hint_confidence=round(iteration_planner_hint_confidence, 3),
                    stage_plan_ms=round(stage_timings_ms["plan_ms"], 3),
                    stage_dom_read_ms=round(stage_timings_ms["dom_read_ms"], 3),
                    stage_screenshot_ms=round(stage_timings_ms["screenshot_ms"], 3),
                    stage_load_wait_ms=round(stage_timings_ms["load_wait_ms"], 3),
                )
                self._emit_live_telemetry(state)
                self._persist_resume_checkpoint(
                    mission=mission,
                    state=state,
                    status="running",
                )

        self._record_failure(
            state,
            code="max_actions_reached",
            stage="execution_budget_exhaustion",
        )
        return _mission_result(
            success=False,
            reasoning=f"Max actions ({max_actions}) reached without completion",
            narrative="Max actions reached without completion",
            state=state,
        )
