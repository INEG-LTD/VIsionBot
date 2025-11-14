import json
import time
from typing import Optional, List, Dict, Any, Tuple, Set
from enum import Enum
import re
import hashlib
from urllib.parse import urlparse

from goals.base import BrowserState, GoalResult, GoalStatus
from agent.completion_contract import CompletionContract, EnvironmentState, CompletionEvaluation
from agent.reactive_goal_determiner import ReactiveGoalDeterminer
from agent.agent_context import AgentContext
from agent.sub_agent_controller import SubAgentController
from agent.sub_agent_result import SubAgentResult
from tab_management import TabDecisionEngine, TabAction
from ai_utils import (
    generate_model,
    ReasoningLevel,
    get_default_agent_model,
    get_default_agent_reasoning_level,
)
from pydantic import BaseModel, ConfigDict

_REQUIREMENT_KEYWORD_MAP = {
    "lede": ["lede", "introduction", "intro", "opening paragraph"],
    "infobox": ["infobox"],
    "milestones": ["milestone", "timeline", "chronology", "history"],
    "founding": ["founding", "founded", "established", "origin", "formation"],
    "leadership": ["leadership", "executive", "ceo", "chairman", "president", "founder"],
    "products": ["product", "service", "achievement", "mission", "program"],
    "controversies": ["controversy", "controversies", "criticism", "lawsuit", "scandal", "challenge"],
    "summary": ["summary", "summaries", "report", "overview"],
    "bullet_list": ["bullet", "list", "milestones", "timeline", "chronology"],
    "sources": ["source", "citation", "url", "link", "reference", "section title"],
    "revision": ["revision", "last updated", "last-edited", "last edited", "edit history"],
}

class _SubAgentPolicyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy_level: str
    policy_score: float
    rationale: str


class _SubAgentTaskPlan(BaseModel):
    name: str
    instruction: str
    suggested_url: Optional[str] = None
    expected_output: Optional[str] = None


class _ParallelWorkPlan(BaseModel):
    main_agent_focus: str
    sub_tasks: List[_SubAgentTaskPlan]
    integration_notes: Optional[str] = None


class _RetargetDecision(BaseModel):
    action: Optional[str]
    rationale: str


class SubAgentPolicyLevel(Enum):
    """Adaptive utilization policy levels for sub-agent usage."""
    SINGLE_THREADED = "single_threaded"
    PARALLELIZED = "parallelized"


"""
Agent Controller - Step 2: LLM-Based Completion + Reactive Goal Determination

Implements agentic mode with:
- Reactive loop: observe → check completion (LLM) → determine goal → act → repeat
- LLM-based completion evaluation (CompletionContract)
- EnvironmentState for full context
- Reactive goal determination (what to do now)
"""

class AgentController:
    """
    Basic reactive agent controller.
    
    Step 1 implementation: Minimal viable agent that:
    1. Observes browser state
    2. Checks simple completion criteria
    3. Generates actions using existing PlanGenerator
    4. Executes actions using existing ActionExecutor
    5. Repeats until done or max iterations
    """
    
    def __init__(
        self,
        bot,
        track_ineffective_actions: bool = True,
        base_knowledge: Optional[List[str]] = None,
        allow_partial_completion: bool = False
    ):
        """
        Initialize agent controller.
        
        Args:
            bot: BrowserVisionBot instance to control
            track_ineffective_actions: If True, track and avoid repeating actions that didn't yield page changes.
                                       Default: True (recommended for better performance)
            base_knowledge: Optional list of knowledge rules/instructions that guide the agent's behavior.
                           Example: ["just press enter after you've typed a search term into a search field"]
        """
        self.bot = bot
        self.max_iterations = 50
        self.iteration_delay = 0.5
        self.task_start_url: Optional[str] = None
        self.task_start_time: Optional[float] = None
        self.exploration_retry_max = 2  # Max retries when exploration mode returns invalid command
        self.allow_non_clickable_clicks = True  # Allow clicking non-clickable elements (configurable)
        self.track_ineffective_actions = track_ineffective_actions  # Track actions that didn't yield page changes
        self.detect_ineffective_actions = track_ineffective_actions
        self.base_knowledge = base_knowledge or []  # Base knowledge rules that guide agent behavior
        self.failed_actions: List[str] = []  # Track actions that failed AND didn't yield any change
        self.ineffective_actions: List[str] = []  # Track actions that succeeded BUT didn't yield any change
        self.extracted_data: Dict[str, Any] = {}  # Store extracted data (key: extraction prompt, value: result)
        self.sub_agent_results: List[SubAgentResult] = []  # Store completed sub-agent results
        self.orchestration_events: List[Dict[str, Any]] = []  # Track orchestration events for reporting
        self.sub_agent_policy_level: SubAgentPolicyLevel = SubAgentPolicyLevel.SINGLE_THREADED
        self.sub_agent_policy_rationale: str = "Default adaptive policy (single-threaded)."
        self._sub_agent_policy_override: Optional[SubAgentPolicyLevel] = None
        self._sub_agent_policy_score: float = 0.0
        self.sub_agent_spawn_count: int = 0
        self._spawns_this_iteration: int = 0
        self._current_iteration: int = 0
        self._parallel_plan_done: bool = False
        self._parallel_plan: Optional[_ParallelWorkPlan] = None
        self._main_agent_focus: Optional[str] = None
        self._integration_notes: Optional[str] = None
        self._extraction_failures: Dict[str, int] = {}
        self._completed_extractions: Set[str] = set()
        self._extraction_prompt_map: Dict[str, str] = {}
        self._retarget_attempts: Dict[str, int] = {}
        self._queued_action: Optional[str] = None
        self._queued_action_reason: Optional[str] = None
        self._task_tracker: Dict[str, Dict[str, Any]] = {}
        self.allow_partial_completion = allow_partial_completion
        self._user_inputs: List[Dict[str, Any]] = []
        self._requirement_flags: Dict[str, bool] = {}
        self._original_user_prompt: str = ""
        self._primary_output_tasks_initialized: bool = False
        self._last_action_summary: Optional[Dict[str, Any]] = None

        self.agent_model_name: str = getattr(bot, "agent_model_name", get_default_agent_model())
        agent_reasoning = getattr(bot, "agent_reasoning_level", None)
        if agent_reasoning is None:
            agent_reasoning = ReasoningLevel.coerce(get_default_agent_reasoning_level())
        else:
            agent_reasoning = ReasoningLevel.coerce(agent_reasoning)
        self.agent_reasoning_level: ReasoningLevel = agent_reasoning
        
        # Initialize TabDecisionEngine if TabManager is available
        self.tab_decision_engine: Optional[TabDecisionEngine] = None
        if hasattr(bot, 'tab_manager') and bot.tab_manager:
            try:
                self.tab_decision_engine = TabDecisionEngine(
                    bot.tab_manager,
                    model_name=self.agent_model_name,
                    reasoning_level=self.agent_reasoning_level,
                )
            except Exception as e:
                print(f"⚠️ Failed to initialize TabDecisionEngine: {e}")
                self.tab_decision_engine = None
        
        # Phase 3: Sub-agent support
        self.agent_context: Optional[AgentContext] = None
        self.sub_agent_controller: Optional[SubAgentController] = None
    
    def run_agentic_mode(self, user_prompt: str, agent_context: Optional[AgentContext] = None) -> GoalResult:
        """
        Run basic agentic mode.
        
        Args:
            user_prompt: User's high-level request
            agent_context: Optional agent context (for sub-agents)
            
        Returns:
            GoalResult indicating success or failure
        """
        # Reset per-run state
        role = "Main agent"
        if agent_context and agent_context.parent_agent_id:
            role = f"Sub-agent helper (task: {agent_context.instruction})"
        print(f"🤖 Active role: {role}")

        self.sub_agent_results = []
        self.orchestration_events = []
        self.sub_agent_policy_level = SubAgentPolicyLevel.SINGLE_THREADED
        self.sub_agent_policy_rationale = "Default adaptive policy (single-threaded)."
        self._sub_agent_policy_override = None
        self._sub_agent_policy_score = 0.0
        self.sub_agent_spawn_count = 0
        self._spawns_this_iteration = 0
        self._current_iteration = 0
        self._task_tracker = {}
        self._original_user_prompt = user_prompt
        self._primary_output_tasks_initialized = False
        self._initialize_requirement_flags(user_prompt)
        self._ensure_primary_output_tasks(user_prompt)
        
        # Phase 3: Set agent context
        if agent_context:
            self.agent_context = agent_context
            # Initialize sub-agent controller if this is main agent
            if agent_context.parent_agent_id is None:
                if not self.sub_agent_controller:
                    self.sub_agent_controller = SubAgentController(
                        self.bot,
                        agent_context,
                        controller_factory=self._spawn_child_controller,
                        track_ineffective_actions=self.detect_ineffective_actions,
                        allow_partial_completion=self.allow_partial_completion
                    )
        else:
            # Create main agent context if not provided
            if self.bot.tab_manager:
                current_tab = self.bot.tab_manager.get_active_tab()
                if current_tab:
                    self.agent_context = AgentContext.create_main_agent(
                        tab_id=current_tab.tab_id,
                        instruction=user_prompt
                    )
                    if not self.sub_agent_controller:
                        self.sub_agent_controller = SubAgentController(
                            self.bot,
                            self.agent_context,
                            controller_factory=self._spawn_child_controller,
                            track_ineffective_actions=self.detect_ineffective_actions,
                            allow_partial_completion=self.allow_partial_completion
                        )
        
        agent_type = "Sub-agent" if (self.agent_context and self.agent_context.parent_agent_id) else "Main agent"
        print(f"🤖 Starting agentic mode ({agent_type}): {user_prompt}")
        print(f"   Max iterations: {self.max_iterations}")

        self._log_event(
            "agent_start",
            agent_type=agent_type,
            prompt=user_prompt,
            tab_id=self.agent_context.tab_id if self.agent_context else None,
        )
        
        # Initialize task tracking
        # Ensure we're using the current page (may have been switched)
        try:
            self.task_start_url = self.bot.page.url
        except Exception:
            self.task_start_url = "unknown"
        self.task_start_time = time.time()
        
        # Set base knowledge on goal monitor for goal evaluation
        if self.base_knowledge:
            self.bot.goal_monitor.set_base_knowledge(self.base_knowledge)
        
        # Create completion contract (Step 2: LLM-based)
        completion_contract = CompletionContract(
            user_prompt,
            allow_partial_completion=self.allow_partial_completion,
            model_name=self.agent_model_name,
            reasoning_level=self.agent_reasoning_level,
        )
        
        if not self.bot.started:
            print("❌ Bot not started. Call bot.start() first.")
            self._log_event("agent_complete", status="failed", reason="bot_not_started")
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=1.0,
                reasoning="Bot not started",
                evidence=self._build_evidence()
            )
        
        if self.bot.page.url.startswith("about:blank"):
            print("❌ Page is on initial blank page.")
            self._log_event("agent_complete", status="failed", reason="blank_page")
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=1.0,
                reasoning="Page is blank",
                evidence=self._build_evidence()
            )
        
        # Main reactive loop
        for iteration in range(self.max_iterations):
            print(f"\n{'='*60}")
            print(f"🔄 Iteration {iteration + 1}/{self.max_iterations}")
            print(f"{'='*60}")
            
            self._log_event("iteration_start", iteration=iteration + 1)
            self._current_iteration = iteration + 1
            self._spawns_this_iteration = 0

            # Drain any completed sub-agent results before continuing
            self._drain_sub_agent_results()
            
            # 1. Observe: Capture browser state (start with viewport)
            snapshot = self._capture_snapshot(full_page=False)
            print(f"📍 Current URL: {snapshot.url}")
            print(f"📄 Page title: {snapshot.title}")
            
            # 2. LLM-based completion check
            environment_state = EnvironmentState(
                browser_state=snapshot,
                interaction_history=self.bot.goal_monitor.interaction_history,
                user_prompt=user_prompt,
                task_start_url=self.task_start_url,
                task_start_time=self.task_start_time,
                current_url=snapshot.url,
                page_title=snapshot.title,
                visible_text=snapshot.visible_text,
                url_history=self.bot.goal_monitor.url_history.copy() if self.bot.goal_monitor.url_history else [],
                url_pointer=getattr(self.bot.goal_monitor, "url_pointer", None)
            )
            
            is_complete, completion_reasoning, evaluation = completion_contract.evaluate(
                environment_state,
                screenshot=snapshot.screenshot
            )
            
            if is_complete:
                print(f"✅ Task complete: {completion_reasoning}")
                print(f"   Confidence: {evaluation.confidence:.2f}")
                if evaluation.evidence:
                    print(f"   Evidence: {evaluation.evidence}")
                evidence_dict: Dict[str, Any] = {}
                if evaluation.evidence:
                    try:
                        import json
                        evidence_dict = json.loads(evaluation.evidence) if isinstance(evaluation.evidence, str) else evaluation.evidence
                    except Exception:
                        evidence_dict = {"evidence": evaluation.evidence}
                
                evidence_dict = self._build_evidence(evidence_dict)
                self._log_event(
                    "agent_complete",
                    status="achieved",
                    confidence=evaluation.confidence,
                    reasoning=completion_reasoning,
                )
                return GoalResult(
                    status=GoalStatus.ACHIEVED,
                    confidence=evaluation.confidence,
                    reasoning=completion_reasoning,
                    evidence=evidence_dict
                )
            else:
                print(f"🔄 Task not complete: {completion_reasoning}")
            
            # Update adaptive sub-agent utilization policy based on latest context
            self._update_sub_agent_policy(
                user_prompt=user_prompt,
                completion_reasoning=completion_reasoning
            )
            
            if (
                self.sub_agent_policy_level == SubAgentPolicyLevel.PARALLELIZED
                and not self._parallel_plan_done
                and (not self.agent_context or self.agent_context.parent_agent_id is None)
            ):
                self._orchestrate_parallel_work(user_prompt)
            # 2.5. Check tab management decisions (Phase 2)
            if self.tab_decision_engine and self.bot.tab_manager:
                current_tab = self.bot.tab_manager.get_active_tab()
                if current_tab:
                    try:
                        # Make tab decision
                        tab_decision = self.tab_decision_engine.make_decision(
                            current_tab_id=current_tab.tab_id,
                            user_prompt=user_prompt,
                            current_action=None,  # Will be determined next
                            task_context={
                                "iteration": iteration + 1,
                                "max_iterations": self.max_iterations,
                                    "completion_reasoning": completion_reasoning,
                                    "sub_agent_policy": self.sub_agent_policy_level.value,
                                    "sub_agent_policy_score": round(self._sub_agent_policy_score, 2),
                                    "sub_agent_policy_reason": self.sub_agent_policy_rationale,
                                    "sub_agent_policy_override": self._sub_agent_policy_override.value if self._sub_agent_policy_override else None,
                                    "sub_agent_spawn_count": self.sub_agent_spawn_count,
                                    "sub_agent_spawns_this_iteration": self._spawns_this_iteration,
                            }
                        )
                        
                        # Execute decision if needed
                        if tab_decision.should_take_action:
                            executed = self._execute_tab_decision(tab_decision)
                            if executed:
                                # Tab switch/close happened, continue to next iteration
                                print("🔄 Tab action executed, continuing to next iteration...")
                                time.sleep(self.iteration_delay)
                                continue
                    except Exception as e:
                        print(f"⚠️ Error in tab decision making: {e}")
            
            # 3. Determine next action - this will tell us if exploration is needed
            current_action: Optional[str] = None
            needs_exploration = False

            if self._queued_action:
                current_action = self._queued_action
                print(f"🎯 Using queued action: {current_action}")
                if self._queued_action_reason:
                    print(f"   Reason: {self._queued_action_reason}")
                self._log_event(
                    "queued_action_consumed",
                    action=current_action,
                    reason=self._queued_action_reason,
                )
                self._queued_action = None
                self._queued_action_reason = None

            if current_action is None:
                # Create reactive goal determiner (Step 2: determines next action from viewport)
                dynamic_prompt = self._build_current_task_prompt(user_prompt)
                goal_determiner = ReactiveGoalDeterminer(
                    dynamic_prompt,
                    base_knowledge=self.base_knowledge,
                    model_name=self.agent_model_name,
                    reasoning_level=self.agent_reasoning_level,
                )
                
                print("🔍 Determining next action based on current viewport...")
                if self.track_ineffective_actions:
                    if self.failed_actions:
                        print(f"   ⚠️ Previously failed actions (failed + no change): {', '.join(self.failed_actions)}")
                    if self.ineffective_actions:
                        print(f"   ⚠️ Previously ineffective actions (succeeded but no change): {', '.join(self.ineffective_actions)}")
                
                current_action, needs_exploration = goal_determiner.determine_next_action(
                    environment_state,
                    screenshot=snapshot.screenshot,
                    is_exploring=False,  # Start with viewport mode
                    failed_actions=self.failed_actions if self.track_ineffective_actions else [],  # Pass failed actions only if tracking enabled
                    ineffective_actions=self.ineffective_actions if self.track_ineffective_actions else []  # Pass ineffective actions only if tracking enabled
                )
                
                # Safeguard: Filter out actions that exactly match failed or ineffective actions (only if tracking enabled)
                if (
                    self.track_ineffective_actions
                    and current_action
                    and current_action.lower().startswith("click:")
                    and (current_action in self.failed_actions or current_action in self.ineffective_actions)
                ):
                    print(f"⚠️ Generated action matches a failed/ineffective action, forcing None: {current_action}")
                    current_action = None
            
            # 4. If no action determined, trigger exploration mode
            # Exploration mode should only be used when we cannot determine any action
            # Only trigger exploration when current_action is None (no valid action found)
            if current_action is None:
                # Identify what we're looking for
                remaining_tasks = goal_determiner._identify_remaining_tasks(environment_state.interaction_history)
                if remaining_tasks:
                    print(f"🔍 Element not visible in viewport - looking for: {remaining_tasks}")
                    print("   Switching to full-page screenshot for exploration")
                else:
                    print("🔍 Element not visible in viewport - switching to full-page screenshot for exploration")
                
                # Capture both viewport and full-page for comparison
                viewport_snapshot = snapshot  # Keep the viewport snapshot
                snapshot = self._capture_snapshot(full_page=True)
                environment_state = EnvironmentState(
                    browser_state=snapshot,
                    interaction_history=self.bot.goal_monitor.interaction_history,
                    user_prompt=user_prompt,
                    task_start_url=self.task_start_url,
                    task_start_time=self.task_start_time,
                    current_url=snapshot.url,
                    page_title=snapshot.title,
                    visible_text=snapshot.visible_text,
                    url_history=self.bot.goal_monitor.url_history.copy() if self.bot.goal_monitor.url_history else [],
                    url_pointer=getattr(self.bot.goal_monitor, "url_pointer", None)
                )
                
                # Retry logic for exploration mode
                exploration_retry_count = 0
                current_action = None
                
                while exploration_retry_count <= self.exploration_retry_max:
                    if exploration_retry_count > 0:
                        print(f"🔍 Retry {exploration_retry_count}/{self.exploration_retry_max} - Re-determining scroll direction...")
                    else:
                        print("🔍 Re-determining scroll direction with full-page screenshot...")
                    
                    print(f"   Current scroll position: Y={snapshot.scroll_y}, Viewport height: {snapshot.page_height}")
                    current_action, needs_exploration = goal_determiner.determine_next_action(
                        environment_state,
                        screenshot=snapshot.screenshot,
                        is_exploring=True,  # Now in exploration mode
                        viewport_snapshot=viewport_snapshot,  # Pass viewport for comparison
                        failed_actions=self.failed_actions if self.track_ineffective_actions else [],  # Pass failed actions only if tracking enabled
                        ineffective_actions=self.ineffective_actions if self.track_ineffective_actions else []  # Pass ineffective actions only if tracking enabled
                    )
                    
                    # Safeguard: Filter out actions that exactly match failed/ineffective actions (except scroll commands in exploration mode, only if tracking enabled)
                    if (
                        self.track_ineffective_actions
                        and current_action
                        and current_action.lower().startswith("click:")
                        and (current_action in self.failed_actions or current_action in self.ineffective_actions)
                        and not current_action.startswith("scroll:")
                    ):
                        print(f"⚠️ Generated action matches a failed/ineffective action: {current_action}")
                        current_action = None
                    
                    # Validate that exploration mode only returns scroll commands
                    if current_action and current_action.startswith("scroll:"):
                        # Valid scroll command - exit retry loop
                        break
                    else:
                        exploration_retry_count += 1
                        if exploration_retry_count <= self.exploration_retry_max:
                            print(f"⚠️ Exploration mode returned invalid command: {current_action}")
                            print(f"   Retrying ({exploration_retry_count}/{self.exploration_retry_max})...")
                            # Small delay before retry
                            time.sleep(0.2)
                        else:
                            # Max retries reached - use deterministic fallback
                            print(f"⚠️ Exploration mode returned invalid command after {self.exploration_retry_max} retries: {current_action}")
                            print("   Using deterministic scroll direction based on element position...")
                            current_action = self._determine_scroll_direction_from_position(
                                viewport_snapshot, snapshot, remaining_tasks
                            )
            
            # Fallback if determiner fails
            if not current_action:
                current_action = self._determine_next_action(user_prompt, snapshot)
            
            if not current_action:
                print("⚠️ Cannot determine next action")
                if iteration == self.max_iterations - 1:
                    self._log_event(
                        "agent_complete",
                        status="failed",
                        reason="no_action",
                        iterations=iteration + 1,
                    )
                    return GoalResult(
                        status=GoalStatus.FAILED,
                        confidence=0.5,
                        reasoning="Could not determine next action",
                        evidence=self._build_evidence({"iterations": iteration + 1})
                    )
                time.sleep(self.iteration_delay)
                continue
            
            print(f"🎯 Next action: {current_action}")
            
            # 4. Check if this is an extraction command or detect extraction needs from natural language
            extraction_prompt = self._detect_extraction_need(current_action, user_prompt)
            
            if extraction_prompt:
                self._update_requirement_flags_from_text(extraction_prompt)
                subject = self._infer_extraction_subject(extraction_prompt)
                normalized_key_source = subject if subject else extraction_prompt
                normalized_key = self._normalize_extraction_prompt(normalized_key_source)
                canonical_prompt = self._build_comprehensive_extraction_prompt(
                    extraction_prompt,
                    subject,
                    self._original_user_prompt or user_prompt,
                    snapshot.url if snapshot else None
                )
                self._extraction_prompt_map[normalized_key] = canonical_prompt
                task_description = self._build_extraction_task_description(subject, extraction_prompt)
                self._register_task(normalized_key, task_description, "extraction")
                print(f"📊 Extraction detected: {canonical_prompt}")
                already_completed = normalized_key in self._completed_extractions
                task_entry = self._task_tracker.get(normalized_key)
                if task_entry and not already_completed:
                    task_entry["status"] = "running"
                    task_entry["updated_at"] = time.time()
                
                if already_completed:
                    print(f"ℹ️ Extraction skipped (already completed): {canonical_prompt}")
                    if task_entry:
                        task_entry["status"] = "completed"
                    self._activate_primary_output_task("Required Wikipedia fields already captured.")
                    time.sleep(self.iteration_delay)
                    continue
                if self._should_skip_extraction(normalized_key):
                    print(f"⚠️ Extraction skipped after repeated failures: {canonical_prompt}")
                    time.sleep(self.iteration_delay)
                    continue
                
                try:
                    # Call extract() directly - simpler and more efficient
                    result = self.bot.extract(
                        prompt=canonical_prompt,
                        output_format="json",
                        scope="page"
                    )
                    self._record_extraction_success(normalized_key)
                    self._completed_extractions.add(normalized_key)
                    
                    self._extraction_prompt_map[normalized_key] = canonical_prompt
                    
                    # Store extracted data with the canonical prompt as key
                    self.extracted_data[canonical_prompt] = result
                    self._mark_task_completed(
                        normalized_key,
                        {
                            "type": "extraction",
                            "fields": list(result.keys()),
                        }
                    )
                    self._activate_primary_output_task("Extraction completed successfully.")
                    print(f"✅ Extraction completed: {result}")
                    
                    # Continue to next iteration (extraction is complete)
                    time.sleep(self.iteration_delay)
                    continue
                except Exception as e:
                    print(f"⚠️ Extraction failed: {e}")
                    import traceback
                    traceback.print_exc()
                    self._record_extraction_failure(normalized_key)
                    self._mark_task_failed(normalized_key, str(e))
                    
                    retarget_decision = self._retarget_after_extraction_failure(
                        prompt_key=normalized_key,
                        extraction_prompt=canonical_prompt,
                        snapshot=snapshot,
                        environment_state=environment_state,
                        user_prompt=user_prompt
                    )
                    if retarget_decision and retarget_decision.action:
                        action_lower = retarget_decision.action.lower()
                        if action_lower != "none":
                            self._queued_action = retarget_decision.action
                            self._queued_action_reason = retarget_decision.rationale
                            print(f"🔁 Retargeting after extraction failure → {retarget_decision.action}")
                            print(f"   Rationale: {retarget_decision.rationale}")
                            self._log_event(
                                "retarget_suggested",
                                action=retarget_decision.action,
                                rationale=retarget_decision.rationale,
                                prompt=canonical_prompt,
                                failures=self._extraction_failures.get(normalized_key, 0),
                            )
                        else:
                            print(f"ℹ️ Retarget assessment: no navigation change needed ({retarget_decision.rationale})")
                    
                    # Continue to next iteration anyway
                    time.sleep(self.iteration_delay)
                    continue
            
            # 4.1. Handle internal control commands (e.g., policy overrides)
            if self._handle_internal_command(current_action, user_prompt):
                time.sleep(self.iteration_delay)
                continue
            
            # 4. Filter out navigation commands - convert to click commands instead
            current_action = self._filter_navigation_commands(current_action)
            
            # 5. Parse action to extract parameters for act()
            act_params = self._parse_action_for_act_params(current_action, user_prompt)
            
            # Log parameters being passed to act()
            print("📋 act() parameters:")
            print(f"   goal_description: {act_params['goal_description']}")
            print(f"   additional_context: {act_params['additional_context']}")
            print(f"   interpretation_mode: {act_params['interpretation_mode']}")
            print(f"   target_context_guard: {act_params['target_context_guard']}")
            print(f"   modifier: {act_params['modifier']}")
            print("   max_attempts: 5")
            print("   max_retries: 1")
            print(f"   allow_non_clickable_clicks: {self.allow_non_clickable_clicks}")
            
            # Capture page state BEFORE action
            state_before = self._get_page_state()
            
            # 6. Use existing act() function - it handles goal creation, planning, execution
            # This leverages all the existing infrastructure (goal creation, plan generation, etc.)
            try:
                success = self.bot.act(
                    goal_description=act_params["goal_description"],
                    additional_context=act_params["additional_context"],
                    interpretation_mode=act_params["interpretation_mode"],
                    target_context_guard=act_params["target_context_guard"],
                    modifier=act_params["modifier"],
                    max_attempts=5,  # Allow 5 attempts per action for better reliability
                    max_retries=1,
                    allow_non_clickable_clicks=self.allow_non_clickable_clicks  # Pass configurable setting
                )
                
                # Capture page state AFTER action
                state_after = self._get_page_state()
                defer_input_consumer = getattr(self.bot, "consume_last_defer_input", None)
                if callable(defer_input_consumer):
                    defer_payload = defer_input_consumer()
                    if defer_payload:
                        self._handle_defer_input(defer_payload, current_action)
                
                # Check if action yielded any change
                page_changed = True
                if self.detect_ineffective_actions:
                    page_changed = self._page_state_changed(state_before, state_after)
                is_click_action = current_action.lower().startswith("click:")
                
                # Only track ineffective actions if the feature is enabled
                if self.track_ineffective_actions:
                    if success:
                        if page_changed:
                            # Successful command that changed the page - clear both failed and ineffective actions
                            if self.failed_actions or self.ineffective_actions:
                                total = len(self.failed_actions) + len(self.ineffective_actions)
                                print(f"   ✅ Command succeeded and changed page - clearing {total} ineffective action(s) from memory")
                                self.failed_actions.clear()
                                self.ineffective_actions.clear()
                        else:
                            # Successful command but no page change - add to ineffective actions
                            if is_click_action and self.detect_ineffective_actions:
                                print(f"⚠️ Action succeeded but did not yield any change: {current_action}")
                                print(f"   URL before: {state_before['url']}")
                                print(f"   URL after: {state_after['url']}")
                                print("   DOM signature unchanged")
                                # Add to ineffective actions list (nudge to try something different)
                                if current_action not in self.ineffective_actions:
                                    self.ineffective_actions.append(current_action)
                                    print("   📝 Added to ineffective actions list (will try different approach in future iterations)")
                            else:
                                if self.detect_ineffective_actions:
                                    print(f"ℹ️ Non-click action succeeded without visible change: {current_action}")
                    else:
                        # Failed command - check if it yielded any change
                        if not page_changed:
                            print(f"⚠️ Action failed and did not yield any change: {current_action}")
                            print(f"   URL before: {state_before['url']}")
                            print(f"   URL after: {state_after['url']}")
                            print("   DOM signature unchanged")
                            # Add to failed actions list (avoid trying this again)
                            if current_action not in self.failed_actions:
                                self.failed_actions.append(current_action)
                                print("   📝 Added to failed actions list (will avoid in future iterations)")
                
                self._record_action_outcome(
                    current_action,
                    success,
                    state_before,
                    state_after,
                    page_changed,
                )
                
                if success:
                    print(f"✅ Action completed: {current_action}")
                    self._log_event(
                        "action_completed",
                        action=current_action,
                        success=True,
                        url_after=state_after["url"],
                    )
                    # Check if overall task is now complete (using LLM evaluation)
                    snapshot_after = self._capture_snapshot()
                    env_state_after = EnvironmentState(
                        browser_state=snapshot_after,
                        interaction_history=self.bot.goal_monitor.interaction_history,
                        user_prompt=user_prompt,
                        task_start_url=self.task_start_url,
                        task_start_time=self.task_start_time,
                        current_url=snapshot_after.url,
                        page_title=snapshot_after.title,
                        visible_text=snapshot_after.visible_text,
                        url_history=self.bot.goal_monitor.url_history.copy() if self.bot.goal_monitor.url_history else [],
                        url_pointer=getattr(self.bot.goal_monitor, "url_pointer", None)
                    )
                    is_complete, completion_reasoning, eval_after = completion_contract.evaluate(
                        env_state_after,
                        screenshot=snapshot_after.screenshot
                    )
                    if is_complete:
                        # Convert evidence string to dict if needed
                        evidence_dict: Dict[str, Any] = {}
                        if eval_after.evidence:
                            try:
                                import json
                                evidence_dict = json.loads(eval_after.evidence) if isinstance(eval_after.evidence, str) else eval_after.evidence
                            except (json.JSONDecodeError, TypeError, ValueError):
                                evidence_dict = {"evidence": eval_after.evidence}
                        
                        evidence_dict = self._build_evidence(evidence_dict)
                        self._log_event(
                            "agent_complete",
                            status="achieved",
                            confidence=eval_after.confidence,
                            reasoning=completion_reasoning,
                        )
                        return GoalResult(
                            status=GoalStatus.ACHIEVED,
                            confidence=eval_after.confidence,
                            reasoning=completion_reasoning,
                            evidence=evidence_dict
                        )
                else:
                    print(f"⚠️ Action failed: {current_action}")
                    self._log_event(
                        "action_completed",
                        action=current_action,
                        success=False,
                        url_after=state_after["url"],
                    )
                    # Continue to next iteration (will try again with fresh state)
                    
            except Exception as e:
                print(f"⚠️ Error executing action: {e}")
                import traceback
                traceback.print_exc()
            
            # Small delay between iterations
            time.sleep(self.iteration_delay)
        
        # Drain any remaining sub-agent results before reporting failure
        self._drain_sub_agent_results()
        self._log_event(
            "agent_complete",
            status="failed",
            reason="max_iterations",
            iterations=self.max_iterations,
        )
        # Max iterations reached
        print(f"❌ Max iterations ({self.max_iterations}) reached")
        return GoalResult(
            status=GoalStatus.FAILED,
            confidence=0.5,
            reasoning=f"Max iterations ({self.max_iterations}) reached without completion",
            evidence=self._build_evidence({"max_iterations": self.max_iterations})
        )
    
    def _capture_snapshot(self, full_page: bool = False) -> BrowserState:
        """
        Capture current browser state snapshot.
        
        Args:
            full_page: If True, capture full page screenshot (for exploration mode)
                      If False, capture viewport only (normal mode)
        """
        snapshot = self.bot.goal_monitor._capture_current_state()
        
        # Override screenshot if full_page is requested
        if full_page:
            try:
                snapshot.screenshot = self.bot.page.screenshot(full_page=True)
                print("📸 Using full-page screenshot for exploration mode")
            except Exception as e:
                print(f"⚠️ Failed to capture full-page screenshot: {e}")
                # Fall back to viewport screenshot
        
        # Compute screenshot hash for change detection
        screenshot_data = getattr(snapshot, "screenshot", None)
        screenshot_hash = None
        if screenshot_data:
            try:
                screenshot_hash = hashlib.md5(screenshot_data).hexdigest()
            except Exception:
                screenshot_hash = None
        setattr(snapshot, "screenshot_hash", screenshot_hash)
        if hasattr(self.bot, "page"):
            try:
                setattr(self.bot.page, "_last_screenshot_hash", screenshot_hash)
            except Exception:
                pass
        
        return snapshot
    
    def _determine_next_action(
        self, 
        user_prompt: str, 
        snapshot: BrowserState,
        completion_evaluation: Optional[CompletionEvaluation] = None
    ) -> Optional[str]:
        """
        Determine what action needs to be done next.
        
        Step 2: Enhanced with state awareness - can use completion evaluation hints.
        Later steps will do full LLM-based reactive goal determination.
        
        Args:
            user_prompt: Original user request
            snapshot: Current browser state (viewport snapshot)
            completion_evaluation: Optional completion evaluation with remaining_steps hints
            
        Returns:
            Specific action description (e.g., "type: John Doe in name field")
            or None if cannot determine
        """
        # Step 2: Use completion evaluation hints if available
        if completion_evaluation and completion_evaluation.remaining_steps:
            # Use the first remaining step as the next action
            next_step = completion_evaluation.remaining_steps[0]
            print(f"🎯 Using suggested next step: {next_step}")
            
            # Special handling: if the step suggests scrolling, prioritize it
            # This ensures we scroll when elements aren't visible
            if next_step.lower().startswith("scroll:"):
                print("📍 Scrolling to reveal more content")
                return next_step
            
            return next_step
        
        # Fallback: use prompt directly (Step 2 still basic for goal determination)
        return user_prompt
    
    def _url_matches_target(self, url: Optional[str], target: str) -> bool:
        if not url or not target:
            return False
        url_lower = url.lower()
        target_lower = target.lower().strip()
        if not target_lower:
            return False

        if target_lower in url_lower:
            return True

        try:
            parsed = urlparse(url_lower)
            domain = parsed.netloc
            if domain and target_lower in domain:
                return True
            path = parsed.path
            if path and target_lower in path:
                return True
        except Exception:
            pass

        return False

    def _history_steps_to_target(self, history: List[str], pointer: int, target: str, direction: str) -> Optional[int]:
        if not history:
            return None

        if direction == "back":
            indices = range(pointer - 1, -1, -1)
        else:
            indices = range(pointer + 1, len(history))

        steps = 0
        for idx in indices:
            steps += 1
            if self._url_matches_target(history[idx], target):
                return steps
        return None

    def _filter_navigation_commands(self, action_command: str) -> str:
        """
        Filter navigate commands so they are only used when no simpler navigation path exists.
        
        Args:
            action_command: The action command to filter
            
        Returns:
            Potentially transformed action command
        """
        if action_command.lower().startswith("navigate:"):
            print(f"ℹ️ Navigation command preserved: {action_command}")
            return action_command
        
        return action_command
    
    def _handle_internal_command(self, current_action: Optional[str], user_prompt: str) -> bool:
        """
        Handle controller-level internal commands that should not be forwarded to act().
        
        Returns True if the command was handled (iteration should continue), False otherwise.
        """
        if not current_action:
            return False
        
        command = current_action.strip()
        command_lower = command.lower()
        
        if command_lower.startswith("subagents:"):
            directive = command.split(":", 1)[1].strip() if ":" in command else ""
            if not directive:
                print("⚠️ Sub-agent override command provided without a directive. "
                      "Use one of: single, parallel, reset.")
                self._log_event(
                    "sub_agent_policy_override_requested",
                    directive=None,
                    recognized=False,
                    source="command",
                    user_prompt_excerpt=user_prompt[:120]
                )
                return True
            
            handled = self._apply_sub_agent_override(directive)
            self._log_event(
                "sub_agent_policy_override_requested",
                directive=directive,
                recognized=handled,
                source="command",
                user_prompt_excerpt=user_prompt[:120]
            )
            if handled:
                print("🛠️ Sub-agent utilization override updated.")
            else:
                print(f"⚠️ Unknown sub-agent override directive: '{directive}'. "
                      "Valid options: single, parallel, aggressive, max, reset.")
            return True
        
        return False
    
    @staticmethod
    def _policy_display_name(level: SubAgentPolicyLevel) -> str:
        """Return human-readable label for a policy level."""
        return level.value.replace("_", " ").title()
    
    def _apply_sub_agent_override(self, directive: str) -> bool:
        """Apply a user-specified override for the sub-agent utilization policy."""
        directive_normalized = directive.strip().lower()
        reset_keywords = {"reset", "clear", "default", "auto", "adaptive"}
        override_map = {
            "single": SubAgentPolicyLevel.SINGLE_THREADED,
            "single-threaded": SubAgentPolicyLevel.SINGLE_THREADED,
            "solo": SubAgentPolicyLevel.SINGLE_THREADED,
            "off": SubAgentPolicyLevel.SINGLE_THREADED,
            "conservative": SubAgentPolicyLevel.SINGLE_THREADED,
            "parallel": SubAgentPolicyLevel.PARALLELIZED,
            "parallelized": SubAgentPolicyLevel.PARALLELIZED,
            "aggressive": SubAgentPolicyLevel.PARALLELIZED,
            "max": SubAgentPolicyLevel.PARALLELIZED,
        }
        score_map = {
            SubAgentPolicyLevel.SINGLE_THREADED: 0.0,
            SubAgentPolicyLevel.PARALLELIZED: 1.0,
        }
        
        if directive_normalized in reset_keywords:
            was_overridden = self._sub_agent_policy_override is not None
            self._sub_agent_policy_override = None
            self.sub_agent_policy_level = SubAgentPolicyLevel.SINGLE_THREADED
            self._sub_agent_policy_score = 0.0
            self.sub_agent_policy_rationale = "Override cleared; returning to adaptive policy (single-threaded baseline)."
            if was_overridden:
                print("🔄 Sub-agent utilization override cleared. Adaptive policy re-enabled.")
            else:
                print("ℹ️ Sub-agent utilization is already using adaptive policy.")
            return True
        
        if directive_normalized in override_map:
            level = override_map[directive_normalized]
            self._sub_agent_policy_override = level
            self.sub_agent_policy_level = level
            self._sub_agent_policy_score = score_map[level]
            self.sub_agent_policy_rationale = (
                f"Override active: forced to {self._policy_display_name(level)}."
            )
            print(f"✅ Sub-agent utilization override -> {self._policy_display_name(level)}")
            return True
        
        return False
    
    def _detect_extraction_need(self, current_action: Optional[str], user_prompt: str) -> Optional[str]:
        """
        Detect if extraction is needed from current action ONLY.
        
        We should only extract if the current action explicitly indicates extraction.
        Do NOT extract based on user prompt if the action is something else (like click).
        
        Args:
            current_action: The current action command (e.g., "extract: product price")
            user_prompt: The original user prompt (not used for detection, only for context)
        
        Returns:
            Extraction prompt if extraction is needed, None otherwise
        """
        if not current_action:
            return None
        
        # Check for explicit "extract:" command
        if current_action.startswith("extract:"):
            return current_action.replace("extract:", "").strip()
        
        # Check if current action indicates extraction (even if not explicitly "extract:")
        action_lower = current_action.lower()
        extraction_keywords = ["extract", "get", "find", "note", "collect", "gather", "retrieve", "pull", "fetch"]
        
        # Only check if the action itself contains extraction keywords
        # Don't extract if the action is clearly something else (like "click", "type", "scroll")
        action_type_keywords = [
            "click",
            "type",
            "press",
            "scroll",
            "select",
            "navigate",
            "upload",
            "back",
            "forward",
            "form",
            "defer",
            "datetime",
            "subagents"
        ]
        
        # If action starts with a non-extraction action type, don't extract
        for action_type in action_type_keywords:
            if action_lower.startswith(f"{action_type}:"):
                return None  # This is clearly a different action, not extraction
        
        # If action contains extraction keywords, extract the description
        for keyword in extraction_keywords:
            if keyword in action_lower:
                # Try to extract what needs to be extracted
                # Pattern: "extract/get/find: <description>"
                pattern = rf"{keyword}:\s*(.+?)(?:\s|$)"
                match = re.search(pattern, action_lower)
                if match:
                    return match.group(1).strip()
        
        # Don't fall back to checking user prompt - only use current_action
        return None

    @staticmethod
    def _normalize_extraction_prompt(prompt: str) -> str:
        """
        Normalize extraction prompt text so equivalent requests share the same key.
        """
        return " ".join(prompt.lower().strip().split())

    @staticmethod
    def _normalize_task_name(name: str) -> str:
        return " ".join(name.lower().strip().split())

    def _initialize_requirement_flags(self, user_prompt: str) -> None:
        """
        Seed requirement flags from the user's original prompt.
        """
        self._requirement_flags = {}
        lowered = user_prompt.lower()
        for key, keywords in _REQUIREMENT_KEYWORD_MAP.items():
            self._requirement_flags[key] = any(keyword in lowered for keyword in keywords)

    def _update_requirement_flags_from_text(self, text: str) -> None:
        """
        Strengthen requirement flags if new instructions mention additional deliverables.
        """
        if not text:
            return
        lowered = text.lower()
        for key, keywords in _REQUIREMENT_KEYWORD_MAP.items():
            if self._requirement_flags.get(key):
                continue
            if any(keyword in lowered for keyword in keywords):
                self._requirement_flags[key] = True

    def _infer_extraction_subject(self, extraction_prompt: str) -> Optional[str]:
        """
        Attempt to infer which subject/entity the extraction prompt targets.
        """
        if not extraction_prompt:
            return None
        match = re.search(r"from the ([^.,\\n]+?)(?: wikipedia| article| page| site)", extraction_prompt, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"for (.+?)(?: page| article| wikipedia)", extraction_prompt, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _build_extraction_task_description(self, subject: Optional[str], extraction_prompt: str) -> str:
        """
        Produce a concise description for the task tracker.
        """
        subject_label = subject or "the target subject"
        sanitized_prompt = extraction_prompt.strip().replace("\n", " ")
        return f"Extract the data requested for {subject_label}: {sanitized_prompt}"

    def _build_comprehensive_extraction_prompt(
        self,
        extraction_prompt: str,
        subject: Optional[str],
        user_prompt: str,
        current_url: Optional[str] = None
    ) -> str:
        """
        Wrap the extraction prompt with light guidance while keeping scope tight.
        """
        scope_note = f"Focus on the subject '{subject}'." if subject else "Focus on the primary subject of this page."
        lines = [
            "Extract exactly the information described below.",
            "Use the entire page content (not just the visible viewport) and avoid adding extra categories.",
            scope_note,
            "Return JSON that mirrors the requested fields only.",
        ]
        if current_url:
            lines.append(f"Current page URL: {current_url}")
        lines.extend([
            "",
            "Requested extraction:",
            extraction_prompt.strip(),
            "",
            "Reference (original user prompt):",
            user_prompt.strip()
        ])
        return "\n".join(lines)

    def _extract_requested_subjects(self, user_prompt: str) -> List[str]:
        """
        Heuristic extraction of distinct subjects mentioned in the user prompt.
        """
        lower = user_prompt.lower()
        alias_pairs = [
            ("elon musk", "Elon Musk"),
            ("space x", "SpaceX"),
            ("spacex", "SpaceX"),
            ("tesla stock", "Tesla Stock"),
            ("tesla, inc.", "Tesla, Inc."),
            ("tesla inc.", "Tesla, Inc."),
            ("tesla inc", "Tesla, Inc."),
            ("tesla", "Tesla, Inc."),
            ("yahoo finance", "Yahoo Finance"),
            ("google docs", "Google Docs"),
        ]

        found: List[Tuple[int, str]] = []
        for alias, canonical in alias_pairs:
            for match in re.finditer(re.escape(alias), lower):
                found.append((match.start(), canonical))

        found.sort(key=lambda item: item[0])
        subjects: List[str] = []
        for _, canonical in found:
            if canonical and canonical not in subjects:
                subjects.append(canonical)
        return subjects

    def _build_manual_parallel_plan(
        self,
        subjects: List[str],
        user_prompt: str
    ) -> Optional[_ParallelWorkPlan]:
        """
        Construct a deterministic parallel plan that mirrors the requested sources.
        """
        if not subjects:
            return None

        lower_prompt = user_prompt.lower()
        sub_tasks: List[_SubAgentTaskPlan] = []
        for subject in subjects:
            subject_lower = subject.lower()
            if subject_lower == "google docs":
                # Leave Google Docs work to the main agent.
                continue

            if subject_lower == "yahoo finance":
                sub_tasks.append(
                    _SubAgentTaskPlan(
                        name="Tesla stock snapshot",
                        instruction=(
                            "Navigate to https://finance.yahoo.com/quote/TSLA and capture the Tesla stock "
                            "performance metrics the user requested (price, change, market summary, key stats). "
                            "Stay focused on the Yahoo Finance data only."
                        ),
                        suggested_url="https://finance.yahoo.com/quote/TSLA",
                        expected_output="Structured Yahoo Finance data for Tesla stock."
                    )
                )
                continue

            slug = re.sub(r"[^\w]+", "_", subject).strip("_")
            suggested_url = None
            if "wikipedia" in lower_prompt:
                suggested_url = f"https://en.wikipedia.org/wiki/{slug}"

            sub_tasks.append(
                _SubAgentTaskPlan(
                    name=f"Wikipedia – {subject}",
                    instruction=(
                        f"Open the Wikipedia page for {subject} and gather only the information requested in the user prompt. "
                        "Do not add unrelated details."
                    ),
                    suggested_url=suggested_url,
                    expected_output=f"Structured Wikipedia data for {subject}."
                )
            )

        if not sub_tasks:
            return None

        main_focus = "Integrate the sub-agent outputs and compose the final write-up using Google Docs."
        integration_notes = "Combine the extracted data from each sub-task to fulfill the original request."
        return _ParallelWorkPlan(
            main_agent_focus=main_focus,
            sub_tasks=sub_tasks,
            integration_notes=integration_notes
        )

    def _handle_defer_input(self, payload: Dict[str, Any], action_command: str) -> None:
        """
        Record user-provided input collected via a defer-input command.
        """
        if not payload:
            return
        response = payload.get("response")
        if response is None:
            return
        prompt = payload.get("prompt", "")
        timestamp = payload.get("timestamp", time.time())
        entry = {
            "prompt": prompt,
            "response": response,
            "timestamp": timestamp,
            "action": action_command,
            "page_url": payload.get("page_url"),
            "page_title": payload.get("page_title"),
        }
        self._user_inputs.append(entry)
        print(f"📝 Captured user input from defer: {response}")
        self._log_event(
            "defer_input_received",
            prompt=prompt,
            response=response,
            action=action_command,
            timestamp=timestamp,
            page_url=entry.get("page_url"),
            page_title=entry.get("page_title"),
        )

    def _ensure_primary_output_tasks(self, user_prompt: str) -> None:
        """
        Register downstream deliverable tasks (e.g., Google Docs report) so the determiner
        focuses on them once prerequisite extractions are satisfied.
        """
        if self._primary_output_tasks_initialized:
            return

        lowered = (user_prompt or "").lower()
        tasks_added = False

        if any(keyword in lowered for keyword in ["google doc", "google docs", "google document"]):
            self._register_task(
                "google_docs_report",
                "Create a Google Docs report using the extracted Wikipedia basics (open docs.google.com, start a document, insert and organize the captured data).",
                "document_output",
            )
            entry = self._task_tracker.get("google_docs_report")
            if entry:
                details = entry.setdefault("details", {})
                details["requires_extractions"] = True
                details["ready"] = False
            tasks_added = True

        # Prevent redundant initialization checks even if no tasks were added.
        if tasks_added or True:
            self._primary_output_tasks_initialized = True

    def _activate_primary_output_task(self, reason: Optional[str] = None) -> None:
        """
        Flag the Google Docs output task as ready so the goal determiner pivots away from extraction.
        """
        entry = self._task_tracker.get("google_docs_report")
        if not entry:
            return
        details = entry.setdefault("details", {})
        already_ready = details.get("ready", False)
        details["ready"] = True
        if reason:
            details["ready_reason"] = reason
        entry["updated_at"] = time.time()
        if entry["status"] == "pending" and not already_ready:
            print("📝 Google Docs task unlocked: extraction prerequisites satisfied.")

    def _register_task(self, task_id: str, description: str, task_type: str) -> None:
        entry = self._task_tracker.get(task_id)
        if not entry:
            self._task_tracker[task_id] = {
                "description": description,
                "type": task_type,
                "status": "pending",
                "attempts": 0,
                "last_error": None,
                "details": {},
                "updated_at": time.time(),
            }
        else:
            if description and description != entry.get("description"):
                entry["description"] = description
            entry.setdefault("type", task_type)
        self._update_requirement_flags_from_text(description)

    def _mark_task_completed(self, task_id: str, details: Optional[Dict[str, Any]] = None) -> None:
        entry = self._task_tracker.get(task_id)
        if not entry:
            return
        entry["status"] = "completed"
        entry["updated_at"] = time.time()
        entry["last_error"] = None
        if details is not None:
            entry["details"] = details

    def _mark_task_failed(self, task_id: str, error: Optional[str] = None) -> None:
        entry = self._task_tracker.get(task_id)
        if not entry:
            return
        entry["status"] = "pending"
        entry["attempts"] = entry.get("attempts", 0) + 1
        entry["updated_at"] = time.time()
        if error:
            entry["last_error"] = error

    def _record_action_outcome(
        self,
        action: str,
        success: bool,
        state_before: Optional[Dict[str, Any]],
        state_after: Optional[Dict[str, Any]],
        page_changed: bool,
    ) -> None:
        """
        Store a lightweight summary of the last executed action so the determiner
        can see recent context and blockers can react accordingly.
        """
        summary = {
            "action": action,
            "success": success,
            "page_changed": page_changed,
            "url_before": (state_before or {}).get("url"),
            "url_after": (state_after or {}).get("url"),
            "timestamp": time.time(),
        }
        if not success:
            summary["note"] = "Action execution failed."
        elif not page_changed:
            summary["note"] = "No visible page change detected."
        self._last_action_summary = summary
        self._update_task_blockers(summary)

    def _update_task_blockers(self, summary: Dict[str, Any]) -> None:
        """
        Annotate downstream tasks with blocker notes (e.g., authentication walls).
        """
        url_after = (summary.get("url_after") or "").strip()
        if not url_after:
            return

        parsed = urlparse(url_after)
        host = (parsed.netloc or "").lower()
        entry = self._task_tracker.get("google_docs_report")
        if not entry:
            return

        details = entry.setdefault("details", {})
        blockers: Dict[str, Any] = details.setdefault("blockers", {})
        blocker_key = "google_auth_wall"
        blocker_message = "Google Docs is gated by a sign-in screen. Use defer-input or manual authentication."
        blocker_added = False
        blocker_removed = False

        if "accounts.google.com" in host or "signin" in host:
            blocker = blockers.get(blocker_key)
            if not blocker:
                blockers[blocker_key] = {
                    "message": blocker_message,
                    "url": url_after,
                    "timestamp": summary.get("timestamp"),
                }
                blocker_added = True
            entry["last_error"] = blocker_message
            details.setdefault("ready", False)
        else:
            if blockers.pop(blocker_key, None) is not None:
                blocker_removed = True
            if entry.get("last_error") == blocker_message:
                entry["last_error"] = None

        if blocker_added or blocker_removed:
            entry["updated_at"] = time.time()

    def _build_current_task_prompt(self, original_prompt: str) -> str:
        if not self._task_tracker:
            return original_prompt

        pending_items = [
            (task_id, data)
            for task_id, data in self._task_tracker.items()
            if data.get("status") != "completed"
        ]
        completed_items = [
            (task_id, data)
            for task_id, data in self._task_tracker.items()
            if data.get("status") == "completed"
        ]

        lines = [original_prompt.strip(), "", "---- Task Progress ----"]

        if completed_items:
            lines.append("Completed objectives:")
            for _, data in completed_items:
                lines.append(f"- ✅ {data.get('description')}")
        else:
            lines.append("Completed objectives: (none yet)")

        if pending_items:
            lines.append("")
            lines.append("Current focus (pending):")
            for task_id, data in pending_items:
                status = data.get("status", "pending")
                attempts = data.get("attempts", 0)
                descriptor = data.get("description")
                status_note = " (in progress)" if status == "running" else ""
                attempt_note = f" [retries: {attempts}]" if attempts else ""
                lines.append(f"- ⏳ {descriptor}{status_note}{attempt_note}")
                last_error = data.get("last_error")
                if last_error:
                    lines.append(f"    ↳ last error: {last_error}")
                details = data.get("details") or {}
                if task_id == "google_docs_report" and details.get("ready"):
                    reason = details.get("ready_reason", "Extraction prerequisites satisfied.")
                    lines.append(f"    ↳ Next step: open Google Docs and compose the report. ({reason})")
                blockers = (details.get("blockers") or {})
                if blockers:
                    for blocker in blockers.values():
                        message = blocker.get("message", "Blocked")
                        lines.append(f"    ↳ Blocked: {message}")
                        blocker_url = blocker.get("url")
                        if blocker_url:
                            lines.append(f"       url: {blocker_url}")
            lines.append("")
            lines.append("Tip: If the required content is on a different site, use `navigate: <url>` to open it directly.")
        if self._user_inputs:
            lines.append("")
            lines.append("User-provided clarifications (most recent first):")
            for entry in reversed(self._user_inputs[-3:]):
                prompt = (entry.get("prompt") or "").strip()
                response = (entry.get("response") or "").strip()
                if prompt:
                    lines.append(f"- {prompt}: {response}")
                else:
                    lines.append(f"- {response}")

        if self._last_action_summary:
            summary = self._last_action_summary
            lines.append("")
            lines.append("Recent action summary:")
            action = summary.get("action", "Unknown action")
            result_word = "succeeded" if summary.get("success") else "failed"
            change_note = "changed the page" if summary.get("page_changed") else "no visible change"
            lines.append(f"- {action} → {result_word} ({change_note}).")
            url_after = summary.get("url_after")
            if url_after:
                lines.append(f"  URL after: {url_after}")
            note = summary.get("note")
            if note:
                lines.append(f"  Note: {note}")

        return "\n".join(lines)

    def _should_skip_extraction(self, prompt_key: str) -> bool:
        failures = self._extraction_failures.get(prompt_key, 0)
        return failures >= 2

    def _record_extraction_failure(self, prompt_key: str) -> None:
        self._extraction_failures[prompt_key] = self._extraction_failures.get(prompt_key, 0) + 1

    def _record_extraction_success(self, prompt_key: str) -> None:
        if prompt_key in self._extraction_failures:
            del self._extraction_failures[prompt_key]
        if prompt_key in self._retarget_attempts:
            del self._retarget_attempts[prompt_key]

    def _retarget_after_extraction_failure(
        self,
        prompt_key: str,
        extraction_prompt: str,
        snapshot: BrowserState,
        environment_state: EnvironmentState,
        user_prompt: str,
    ) -> Optional[_RetargetDecision]:
        """
        Use an LLM to determine if we should retarget the task after repeated extraction failures.
        """
        attempts = self._retarget_attempts.get(prompt_key, 0)
        if attempts >= 2:
            return None
        self._retarget_attempts[prompt_key] = attempts + 1

        visible_text = snapshot.visible_text or ""
        snippet = visible_text[:1200]

        history_summary = ""
        if environment_state.url_history:
            history_summary = " → ".join(environment_state.url_history[-5:])

        prompt = (
            "The agent attempted to extract structured data but the extraction failed.\n"
            "Decide whether the current page contains the requested information, and if not, provide a single retargeting command.\n\n"
            f"User task: {user_prompt}\n"
            f"Extraction request: {extraction_prompt}\n"
            f"Current URL: {snapshot.url}\n"
            f"Page title: {snapshot.title}\n"
            f"Recent URL history: {history_summary or 'N/A'}\n"
            f"Visible text snippet (truncated):\n{snippet}\n\n"
            "If the page is irrelevant or missing the requested subject, output a command such as:\n"
            "- navigate: <url>\n"
            "- search: <query>\n"
            "- click: <selector or link text>\n"
            "Return 'none' if the page is correct and the agent should try extraction again without navigating.\n"
            "Avoid scroll actions, and prefer precise navigation targets when suggesting URLs."
        )

        try:
            decision = generate_model(
                prompt=prompt,
                system_prompt=(
                    "You help a web-browsing automation agent recover from failed data extractions. "
                    "Evaluate the current page and produce the best single next action. "
                    "Respond with a short command (or 'none') and a concise rationale."
                ),
                model_object_type=_RetargetDecision,
                model=self.agent_model_name,
                reasoning_level=self.agent_reasoning_level,
            )
            return decision
        except Exception as e:
            print(f"⚠️ Failed to obtain retarget decision: {e}")
            return None
    
    def _parse_action_for_act_params(
        self,
        action_command: str,
        user_prompt: str
    ) -> dict:
        """
        Parse action command to extract parameters for act() function.
        
        Extracts:
        - Ordinal information (first, second, third, etc.) → modifier
        - Collection hints (article, button, link, etc.) → additional_context
        - Interpretation mode (semantic for complex targets, literal for simple)
        - Target context guard (for filtering elements)
        
        Args:
            action_command: The action command (e.g., "click: first article")
            user_prompt: The original user prompt for context
            
        Returns:
            Dictionary with act() parameters
        """
        import re
        from utils.intent_parsers import ORDINAL_WORDS
        
        params = {
            "goal_description": action_command,
            "additional_context": "",
            "interpretation_mode": None,  # Let bot decide
            "target_context_guard": None,
            "modifier": None,
        }
        
        # Extract ordinal information
        action_lower = action_command.lower()
        ordinal_word = None
        ordinal_index = None
        
        # Check for ordinal words
        for word, idx in ORDINAL_WORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", action_lower):
                ordinal_word = word
                ordinal_index = idx
                break
        
        # Also check for numeric ordinals (1st, 2nd, etc.)
        if ordinal_index is None:
            match = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", action_lower)
            if match:
                ordinal_index = max(int(match.group(1)) - 1, 0)
                ordinal_word = f"{ordinal_index + 1}"
        
        # If ordinal found, add to modifier
        if ordinal_index is not None:
            # Format: "first" -> ["ordinal:0"], "second" -> ["ordinal:1"]
            params["modifier"] = [f"ordinal:{ordinal_index}"]
            
            # Add to additional_context for better planning
            params["additional_context"] = f"Target is the {ordinal_word} element in the list/collection. "
        
        # Extract collection hints (article, button, link, etc.)
        collection_patterns = {
            "article": r"\barticle\b",
            "button": r"\bbutton\b",
            "link": r"\blink\b",
            "item": r"\bitem\b",
            "entry": r"\bentry\b",
            "row": r"\brow\b",
        }
        
        found_collections = []
        for collection, pattern in collection_patterns.items():
            if re.search(pattern, action_lower):
                found_collections.append(collection)
        
        if found_collections:
            collection_context = f"Looking for a {', '.join(found_collections)}. "
            params["additional_context"] += collection_context
        
        # Always use semantic mode for better understanding
        params["interpretation_mode"] = "semantic"
        
        # Add target context guard for ordinal selection
        # This helps the plan generator filter to the correct ordinal position
        if ordinal_index is not None:
            # Guard: element must be at the specified position in a list/collection
            params["target_context_guard"] = f"Element must be the {ordinal_word} in the list/collection"
        
        # NOTE: We do NOT add the original user prompt to additional_context
        # The plan generator should focus ONLY on the immediate goal at hand,
        # not the overall task. This prevents the plan generator from trying to
        # accomplish multiple goals at once (e.g., searching for "hacker news"
        # when the goal is just "click: Google Search")
        
        return params
    
    def _determine_scroll_direction_from_position(
        self,
        viewport_snapshot: BrowserState,
        full_page_snapshot: BrowserState,
        target_description: Optional[str]
    ) -> str:
        """
        Deterministically determine scroll direction by detecting elements and comparing positions.
        
        This method:
        1. Detects all form elements on the page
        2. Finds the target element (e.g., "email field")
        3. Gets its absolute Y position (scroll_y + element.rect.top)
        4. Compares with current viewport bounds
        5. Returns "scroll: up" or "scroll: down"
        """
        try:
            # Get current scroll position and viewport bounds
            current_scroll_y = full_page_snapshot.scroll_y
            viewport_height = full_page_snapshot.page_height
            viewport_top = current_scroll_y
            viewport_bottom = current_scroll_y + viewport_height
            
            # Detect all form elements on the page
            page = self.bot.page
            if not page:
                # Fallback to simple heuristic
                return "scroll: down" if current_scroll_y == 0 else "scroll: up"
            
            # Get all interactive elements with their positions
            # This includes: form fields, buttons, links, and other clickable elements
            js_code = """
            (function() {
                const elements = [];
                
                // Get all potentially interactive elements
                const selectors = [
                    'input, select, textarea',  // Form fields
                    'button, [role="button"]',  // Buttons
                    'a[href]',                  // Links
                    '[onclick]',                // Elements with onclick handlers
                    '[tabindex]:not([tabindex="-1"])'  // Focusable elements
                ];
                
                const allElements = new Set();
                selectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => allElements.add(el));
                });
                
                allElements.forEach((element) => {
                    const rect = element.getBoundingClientRect();
                    
                    // Skip hidden elements
                    if (rect.width === 0 || rect.height === 0 || 
                        getComputedStyle(element).display === 'none' ||
                        getComputedStyle(element).visibility === 'hidden') {
                        return;
                    }
                    
                    // Calculate absolute Y position (viewport-relative + scroll offset)
                    const absoluteY = window.scrollY + rect.top;
                    const centerY = absoluteY + (rect.height / 2);
                    
                    // Get text content for matching
                    const textContent = element.textContent?.trim() || '';
                    const innerText = element.innerText?.trim() || '';
                    
                    elements.push({
                        tagName: element.tagName.toLowerCase(),
                        type: element.type || '',
                        name: element.name || '',
                        id: element.id || '',
                        className: element.className || '',
                        placeholder: element.placeholder || '',
                        label: element.labels?.[0]?.textContent || '',
                        textContent: textContent,
                        innerText: innerText,
                        href: element.href || '',
                        role: element.getAttribute('role') || '',
                        ariaLabel: element.getAttribute('aria-label') || '',
                        absoluteY: absoluteY,
                        centerY: centerY,
                        rect: {
                            top: rect.top,
                            left: rect.left,
                            height: rect.height,
                            width: rect.width
                        }
                    });
                });
                
                return elements;
            })();
            """
            
            elements = page.evaluate(js_code)
            if not elements:
                # Fallback if no elements found
                return "scroll: down" if current_scroll_y == 0 else "scroll: up"
            
            # Find target element based on description
            # This matches against various element attributes (name, id, type, text, href, etc.)
            target_element = None
            if target_description:
                target_lower = target_description.lower()
                
                # Extract keywords from target description
                keywords = []
                if 'email' in target_lower or 'mail' in target_lower:
                    keywords.extend(['email', 'mail'])
                if 'name' in target_lower:
                    keywords.append('name')
                if 'password' in target_lower or 'pass' in target_lower:
                    keywords.extend(['password', 'pass'])
                if 'button' in target_lower:
                    keywords.append('button')
                if 'link' in target_lower:
                    keywords.append('link')
                if 'submit' in target_lower:
                    keywords.extend(['submit', 'button'])
                if 'click' in target_lower:
                    # For click actions, try to extract what to click
                    if 'hacker' in target_lower and 'news' in target_lower:
                        keywords.extend(['hacker', 'news', 'ycombinator'])
                
                # Score each element based on how well it matches
                best_match = None
                best_score = 0
                
                for element in elements:
                    score = 0
                    
                    # Check various attributes
                    searchable_text = ' '.join([
                        element.get('name', ''),
                        element.get('id', ''),
                        element.get('type', ''),
                        element.get('textContent', ''),
                        element.get('innerText', ''),
                        element.get('placeholder', ''),
                        element.get('label', ''),
                        element.get('ariaLabel', ''),
                        element.get('href', ''),
                        element.get('className', '')
                    ]).lower()
                    
                    # Score based on keyword matches
                    for keyword in keywords:
                        if keyword in searchable_text:
                            score += 1
                            # Higher weight for exact matches in key fields
                            if keyword in element.get('name', '').lower():
                                score += 2
                            if keyword in element.get('id', '').lower():
                                score += 2
                            if keyword in element.get('type', '').lower():
                                score += 2
                    
                    # Bonus for exact type matches
                    if 'email' in keywords and element.get('type') == 'email':
                        score += 3
                    if 'password' in keywords and element.get('type') == 'password':
                        score += 3
                    
                    # Bonus for button/link tag matches
                    if 'button' in keywords and element.get('tagName') == 'button':
                        score += 2
                    if 'link' in keywords and element.get('tagName') == 'a':
                        score += 2
                    
                    if score > best_score:
                        best_score = score
                        best_match = element
                
                if best_match and best_score > 0:
                    target_element = best_match
                    print(f"   🎯 Matched target element (score: {best_score}): {best_match.get('tagName', 'unknown')} - {best_match.get('name', '') or best_match.get('id', '') or best_match.get('textContent', '')[:50]}")
            
            # If target element found, compare position
            if target_element:
                target_y = target_element['centerY']  # Use center Y for better accuracy
                
                print(f"   📍 Target element found at absolute Y: {target_y}")
                print(f"   📍 Viewport bounds: Y={viewport_top} to Y={viewport_bottom}")
                
                if target_y < viewport_top:
                    print("   ⬆️ Target is above viewport → scroll: up")
                    return "scroll: up"
                elif target_y > viewport_bottom:
                    print("   ⬇️ Target is below viewport → scroll: down")
                    return "scroll: down"
                else:
                    # Target is in viewport - shouldn't happen in exploration mode
                    print("   ✅ Target is in viewport (should not be in exploration mode)")
                    return "scroll: down"  # Default
            
            # Fallback: if we can't find target, use simple heuristic
            print("   ⚠️ Target element not found, using heuristic")
            if current_scroll_y > 0:
                return "scroll: up"  # Already scrolled, try up first
            else:
                return "scroll: down"  # At top, scroll down
                
        except Exception as e:
            print(f"⚠️ Error determining scroll direction: {e}")
            # Final fallback
            if full_page_snapshot.scroll_y > 0:
                return "scroll: up"
            else:
                return "scroll: down"
    
    def _is_in_exploration_mode(self, recent_actions) -> bool:
        """
        Detect if agent is in exploration mode (searching for missing elements).
        
        Exploration mode is triggered when:
        - Recent actions are primarily scrolling
        - Multiple scroll actions in a row suggest we're looking for something
        """
        if not recent_actions:
            return False
        
        # Count scroll actions in recent history
        scroll_count = sum(
            1 for action in recent_actions
            if action.interaction_type.value == "scroll"
        )
        
        # If 2+ scrolls in last 3 actions, we're exploring
        if scroll_count >= 2:
            return True
        
        # Also check if last action was scroll (actively searching)
        if recent_actions and recent_actions[-1].interaction_type.value == "scroll":
            # Check if we've scrolled multiple times total
            total_scrolls = sum(
                1 for action in self.bot.goal_monitor.interaction_history
                if action.interaction_type.value == "scroll"
            )
            # If we've scrolled 3+ times total, likely exploring
            if total_scrolls >= 3:
                return True
        
        return False
    
    def _simple_completion_check(
        self,
        user_prompt: str,
        snapshot: BrowserState
    ) -> tuple[bool, str]:
        """
        Simple fallback completion check (kept for compatibility).
        Step 2 uses LLM-based completion contract instead.
        
        Simple rule-based completion check.
        
        Checks:
        - URL matches if prompt mentions navigation
        - Page text contains expected keywords
        - URL patterns match expected destinations
        
        Returns:
            (is_complete, reasoning)
        """
        prompt_lower = user_prompt.lower()
        url_lower = snapshot.url.lower()
        title_lower = (snapshot.title or "").lower()
        text_lower = (snapshot.visible_text or "").lower()
        
        # Check for navigation completion
        nav_keywords = ["navigate", "go to", "visit", "open", "goto"]
        if any(keyword in prompt_lower for keyword in nav_keywords):
            # Try to extract target URL/domain from prompt
            # Simple pattern: "go to example.com" or "navigate to https://example.com"
            url_patterns = re.findall(
                r'(?:https?://)?([a-z0-9\-]+(?:\.[a-z0-9\-]+)+)',
                prompt_lower
            )
            if url_patterns:
                target_domain = url_patterns[0]
                if target_domain in url_lower:
                    return True, f"Navigated to target domain: {target_domain}"
            
            # Check for common navigation success indicators
            if "error" not in text_lower and "404" not in url_lower:
                # If we're on a different page than before, might be success
                # (This is basic - could be improved)
                pass
        
        # Check for completion keywords in page content
        completion_keywords = [
            "success", "complete", "done", "submitted", "confirmed",
            "thank you", "application received", "sent successfully"
        ]
        for keyword in completion_keywords:
            if keyword in text_lower or keyword in title_lower:
                return True, f"Completion keyword found: '{keyword}'"
        
        # Check for error indicators (task likely not complete)
        error_keywords = ["error", "failed", "invalid", "not found", "404"]
        for keyword in error_keywords:
            if keyword in text_lower or keyword in title_lower:
                # Don't mark complete if errors are present
                return False, f"Error indicator found: '{keyword}'"
        
        # Default: not complete
        return False, "Completion criteria not met"
    
    def _get_page_state(self) -> dict:
        """
        Get current page state for change detection.
        
        Returns:
            Dictionary with url and dom_signature
        """
        try:
            url = self.bot.page.url
            # Get element count as a simple DOM change indicator
            element_count = self.bot.page.evaluate("() => document.querySelectorAll('*').length")
            sig_src = f"{url}|{element_count}"
            dom_signature = hashlib.md5(sig_src.encode("utf-8")).hexdigest()
            screenshot_hash = getattr(self.bot.page, "_last_screenshot_hash", None)
            return {
                "url": url,
                "dom_signature": dom_signature,
                "screenshot_hash": screenshot_hash
            }
        except Exception:
            # Fallback to just URL
            try:
                url = self.bot.page.url
                dom_signature = hashlib.md5(url.encode("utf-8")).hexdigest()
                screenshot_hash = getattr(self.bot.page, "_last_screenshot_hash", None)
                return {
                    "url": url,
                    "dom_signature": dom_signature,
                    "screenshot_hash": screenshot_hash
                }
            except Exception:
                return {
                    "url": "",
                    "dom_signature": "",
                    "screenshot_hash": None
                }

    def _spawn_child_controller(
        self,
        bot,
        *,
        base_knowledge: Optional[List[str]] = None,
        track_ineffective_actions: Optional[bool] = None,
        allow_partial_completion: Optional[bool] = None
    ) -> "AgentController":
        """
        Factory method passed to SubAgentController to ensure sub-agents inherit configuration.
        """
        track = self.detect_ineffective_actions if track_ineffective_actions is None else track_ineffective_actions
        partial = self.allow_partial_completion if allow_partial_completion is None else allow_partial_completion
        return AgentController(
            bot=bot,
            track_ineffective_actions=track,
            base_knowledge=base_knowledge,
            allow_partial_completion=partial
        )
    
    def _drain_sub_agent_results(self) -> None:
        """
        Retrieve any newly completed sub-agent results and integrate them.
        """
        if not self.sub_agent_controller:
            return
        results = self.sub_agent_controller.pop_completed_results()
        if not results:
            return
        
        for result in results:
            self.sub_agent_results.append(result)
            status_icon = "✅" if result.success else "⚠️"
            duration = max(0.0, result.completed_at - result.started_at)
            print(f"{status_icon} Sub-agent [{result.agent_id}] ({result.instruction}) -> {result.status} "
                  f"(confidence={result.confidence:.2f}, duration={duration:.2f}s)")
            
            # Merge extracted data into main agent store if provided
            evidence = result.evidence or {}
            extracted = evidence.get("extracted_data")
            if isinstance(extracted, dict):
                for key, value in extracted.items():
                    if key not in self.extracted_data:
                        self.extracted_data[key] = value
            self._log_event(
                "sub_agent_result_recorded",
                agent_id=result.agent_id,
                success=result.success,
                status=result.status,
            )

    def _build_evidence(self, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Construct evidence dictionaries including extracted data and sub-agent results.
        """
        evidence: Dict[str, Any] = {}
        if base:
            evidence.update(base)
        if self.extracted_data and "extracted_data" not in evidence:
            evidence["extracted_data"] = self.extracted_data.copy()
        if self.sub_agent_results:
            evidence["sub_agents"] = [result.to_dict() for result in self.sub_agent_results]
        orchestration = {
            "events": list(self.orchestration_events),
            "tabs": self._build_tab_summary(),
        }
        # Only include sub-agent summaries if not already present
        if self.sub_agent_results:
            orchestration["sub_agents"] = [result.to_dict() for result in self.sub_agent_results]
        evidence["orchestration"] = orchestration
        evidence["sub_agent_policy"] = {
            "level": self.sub_agent_policy_level.value,
            "score": round(self._sub_agent_policy_score, 2),
            "rationale": self.sub_agent_policy_rationale,
            "override": self._sub_agent_policy_override.value if self._sub_agent_policy_override else None,
            "spawn_count": self.sub_agent_spawn_count,
        }
        if self._parallel_plan:
            evidence["parallel_plan"] = {
                "main_agent_focus": self._main_agent_focus,
                "integration_notes": self._integration_notes,
                "sub_tasks": [task.dict() for task in self._parallel_plan.sub_tasks],
            }
        if self._task_tracker:
            evidence["task_progress"] = {
                task_id: {
                    "description": data.get("description"),
                    "type": data.get("type"),
                    "status": data.get("status"),
                    "attempts": data.get("attempts"),
                    "last_error": data.get("last_error"),
                    "updated_at": data.get("updated_at"),
                    "details": data.get("details"),
                }
                for task_id, data in self._task_tracker.items()
            }
        return evidence

    def _build_tab_summary(self) -> List[Dict[str, Any]]:
        tab_manager = getattr(self.bot, "tab_manager", None)
        if not tab_manager:
            return []
        try:
            tabs = tab_manager.list_tabs()
        except Exception:
            return []
        summary = []
        for tab in tabs:
            if hasattr(tab, "to_dict"):
                summary.append(tab.to_dict())
            else:
                summary.append({"tab_id": getattr(tab, "tab_id", None)})
        return summary

    def _page_state_changed(self, state_before: dict, state_after: dict) -> bool:
        """
        Check if page state changed after an action.
        
        Args:
            state_before: Page state before action
            state_after: Page state after action
            
        Returns:
            True if page state changed, False otherwise
        """
        # Check if URL changed
        if state_before.get("url") != state_after.get("url"):
            return True
        
        # Check if DOM signature changed
        if state_before.get("dom_signature") != state_after.get("dom_signature"):
            return True
        
        # Check if screenshot hash changed (if available)
        if state_before.get("screenshot_hash") != state_after.get("screenshot_hash"):
            return True
        
        return False
    
    def _update_sub_agent_policy(self, user_prompt: str, completion_reasoning: Optional[str]) -> None:
        """
        Evaluate and update the adaptive sub-agent utilization policy.
        """
        new_level, new_score, new_rationale = self._compute_sub_agent_policy(
            user_prompt=user_prompt,
            completion_reasoning=completion_reasoning or ""
        )
        
        level_changed = new_level != self.sub_agent_policy_level
        score_changed = abs(new_score - self._sub_agent_policy_score) >= 0.1
        rationale_changed = new_rationale != self.sub_agent_policy_rationale
        
        self.sub_agent_policy_level = new_level
        self._sub_agent_policy_score = new_score
        self.sub_agent_policy_rationale = new_rationale
        
        if level_changed or score_changed or rationale_changed:
            print(f"📊 Sub-agent utilization policy → {self._policy_display_name(new_level)} "
                  f"(score {new_score:.2f})")
            print(f"   Reason: {new_rationale}")
            self._log_event(
                "sub_agent_policy_update",
                policy=new_level.value,
                score=new_score,
                rationale=new_rationale,
                override=self._sub_agent_policy_override.value if self._sub_agent_policy_override else None
            )
    
    def _compute_sub_agent_policy(
        self,
        user_prompt: str,
        completion_reasoning: str
    ) -> Tuple[SubAgentPolicyLevel, float, str]:
        """
        Compute the appropriate sub-agent utilization policy score and level using an LLM.
        """
        if not self.sub_agent_controller or not getattr(self.bot, "tab_manager", None):
            print("📊 Sub-agent policy decision: controller or tab manager unavailable (forcing single-threaded).")
            return (
                SubAgentPolicyLevel.SINGLE_THREADED,
                0.0,
                "Sub-agent controller or tab manager unavailable."
            )
        
        if self._sub_agent_policy_override:
            level = self._sub_agent_policy_override
            score_override = {
                SubAgentPolicyLevel.SINGLE_THREADED: 0.0,
                SubAgentPolicyLevel.PARALLELIZED: 1.0
            }[level]
            print(f"📊 Sub-agent policy override active → {self._policy_display_name(level)} (score {score_override:.2f}).")
            return (
                level,
                score_override,
                f"Override active: forced to {self._policy_display_name(level)}."
            )
        
        policy_spec, score, rationale = self._query_sub_agent_policy_llm(
            user_prompt=user_prompt,
            completion_reasoning=completion_reasoning
        )
        
        if not isinstance(policy_spec, str):
            policy_spec = SubAgentPolicyLevel.SINGLE_THREADED.value
        
        policy_spec_normalized = policy_spec.strip().lower()
        level_map = {
            "single": SubAgentPolicyLevel.SINGLE_THREADED,
            "single_threaded": SubAgentPolicyLevel.SINGLE_THREADED,
            "single-threaded": SubAgentPolicyLevel.SINGLE_THREADED,
            "assistive": SubAgentPolicyLevel.PARALLELIZED,
            "balanced": SubAgentPolicyLevel.PARALLELIZED,
            "default": SubAgentPolicyLevel.SINGLE_THREADED,
            "parallel": SubAgentPolicyLevel.PARALLELIZED,
            "parallelized": SubAgentPolicyLevel.PARALLELIZED,
            "aggressive": SubAgentPolicyLevel.PARALLELIZED,
        }
        policy_level = level_map.get(policy_spec_normalized, SubAgentPolicyLevel.SINGLE_THREADED)
        
        score_clamped = max(0.0, min(1.0, score if isinstance(score, (int, float)) else (1.0 if policy_level == SubAgentPolicyLevel.PARALLELIZED else 0.0)))
        rationale_text = rationale or "LLM policy evaluation."
        
        print(f"📊 Sub-agent policy (LLM) → {self._policy_display_name(policy_level)} (score {score_clamped:.2f})")
        print(f"   Reason: {rationale_text}")
        
        return policy_level, score_clamped, rationale_text
    
    def _query_sub_agent_policy_llm(
        self,
        user_prompt: str,
        completion_reasoning: str
    ) -> Tuple[str, float, str]:
        """
        Ask an LLM to evaluate whether sub-agents should be used and at what intensity.
        """
        policy_prompt = self._build_sub_agent_policy_prompt(
            user_prompt=user_prompt,
            completion_reasoning=completion_reasoning
        )
        try:
            result = generate_model(
                prompt=policy_prompt,
                model_object_type=_SubAgentPolicyResponse,
                system_prompt=self._build_sub_agent_policy_system_prompt(),
                model=self.agent_model_name,
                reasoning_level=self.agent_reasoning_level,
            )
        except Exception as e:
            print(f"⚠️ Failed to evaluate sub-agent policy via LLM: {e}")
            return SubAgentPolicyLevel.SINGLE_THREADED.value, 0.0, "Fallback adaptive policy (LLM error)."
        
        return result.policy_level, result.policy_score, result.rationale
    
    @staticmethod
    def _build_sub_agent_policy_system_prompt() -> str:
        return """You help a browser automation supervisor decide whether to delegate work to sub-agents.

Consider task urgency, need for parallel research, blockers, focus requirements, and historical success of sub-agents.
Return a concise policy recommendation."""
    
    def _build_sub_agent_policy_prompt(
        self,
        user_prompt: str,
        completion_reasoning: str
    ) -> str:
        active_sub_agents = 0
        if self.sub_agent_controller:
            active_sub_agents = sum(
                1 for ctx in self.sub_agent_controller.sub_agents.values()
                if ctx.status in {"pending", "running"}
            )
        
        history = self.sub_agent_controller.get_execution_history() if self.sub_agent_controller else []
        successes = sum(1 for result in history if getattr(result, "success", False))
        failures = len(history) - successes
        
        tab_manager = getattr(self.bot, "tab_manager", None)
        active_tab_url = ""
        if tab_manager and tab_manager.get_active_tab():
            try:
                active_tab_url = tab_manager.get_active_tab().url
            except Exception:
                active_tab_url = ""
        
        prompt_flags = {
            "multi_tab_emphasis": "spread across multiple tabs" if "multiple tabs" in user_prompt.lower() else "unspecified",
            "urgency": "yes" if any(word in user_prompt.lower() for word in ["quick", "fast", "urgent", "asap"]) else "no"
        }
        
        return f"""
SUB-AGENT UTILIZATION DECISION
==============================

USER PROMPT:
{user_prompt}

COMPLETION REASONING:
{completion_reasoning or "N/A"}

ACTIVE TAB URL:
{active_tab_url or "unknown"}

CURRENT POLICY:
- Override: {self._sub_agent_policy_override.value if self._sub_agent_policy_override else "None"}
- Previous level: {self.sub_agent_policy_level.value}
- Spawn count this run: {self.sub_agent_spawn_count}
- Spawns this iteration: {self._spawns_this_iteration}
- Active sub-agents: {active_sub_agents}
- Recent sub-agent successes: {successes}
- Recent sub-agent failures: {failures}

TASK CONTEXT FLAGS:
- Multi-tab emphasis: {prompt_flags["multi_tab_emphasis"]}
- Urgency mentioned: {prompt_flags["urgency"]}

DECISION GUIDELINES:
- single_threaded: keep work on the main agent; avoid spawning sub-agents.
- parallelized: aggressively use sub-agents; create dedicated tabs for parallel research or extraction.

Provide:
1. policy_level (single_threaded or parallelized)
2. policy_score (0.0-1.0 representing intensity of delegation)
3. rationale (one or two sentences)
"""

    def _normalize_suggested_url(self, url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        parts = url.strip().split()
        for part in parts:
            candidate = part.strip()
            if candidate.lower().startswith(("http://", "https://")):
                return candidate
        return None

    def _run_orchestrated_task(self, task: _SubAgentTaskPlan, plan_label: str) -> None:
        if not self.sub_agent_controller or not getattr(self.bot, "tab_manager", None):
            return
        if self.agent_context and self.agent_context.parent_agent_id and plan_label != "parallel_plan":
            print(f"ℹ️ {plan_label}: sub-agents do not orchestrate additional helpers (task '{task.name}' skipped).")
            return

        task_key = self._normalize_task_name(task.name)
        self._register_task(task_key, task.instruction or task.name, plan_label)
        task_entry = self._task_tracker.get(task_key)
        if task_entry:
            task_entry["status"] = "running"
            task_entry["updated_at"] = time.time()

        metadata = {
            "orchestrated": True,
            "plan": plan_label,
            "task_name": task.name,
            "expected_output": task.expected_output,
        }

        normalized_url = self._normalize_suggested_url(task.suggested_url)
        if task.suggested_url and not normalized_url:
            print(f"⚠️ {plan_label}: suggested URL '{task.suggested_url}' is invalid; skipping automatic navigation.")

        new_tab_id = self.bot.tab_manager.open_new_tab(
            purpose=task.name,
            url=normalized_url,
            metadata=metadata
        )
        if not new_tab_id:
            print(f"⚠️ {plan_label}: failed to open tab for task '{task.name}'.")
            self._log_event(f"{plan_label}_tab_failed", task=task.dict())
            self._mark_task_failed(task_key, "Unable to open browser tab for sub-task.")
            return

        sub_agent_id = self.sub_agent_controller.spawn_sub_agent(
            tab_id=new_tab_id,
            instruction=task.instruction,
            metadata=metadata
        )
        if not sub_agent_id:
            print(f"⚠️ {plan_label}: failed to spawn sub-agent for task '{task.name}'.")
            self._log_event(
                f"{plan_label}_spawn_failed",
                task=task.dict(),
                tab_id=new_tab_id
            )
            self._mark_task_failed(task_key, "Failed to spawn sub-agent for sub-task.")
            return

        if not (self.agent_context and self.agent_context.parent_agent_id):
            self.sub_agent_spawn_count += 1
            self._spawns_this_iteration += 1

        self._log_event(
            f"{plan_label}_task_spawned",
            task=task.dict(),
            tab_id=new_tab_id,
            agent_id=sub_agent_id
        )
        print(f"🤝 {plan_label}: running sub-agent '{task.name}' → {task.instruction}")

        result = self.sub_agent_controller.execute_sub_agent(sub_agent_id)
        self._log_event(
            f"{plan_label}_task_completed",
            task=task.dict(),
            tab_id=new_tab_id,
            agent_id=sub_agent_id,
            result=result
        )
        success = False
        confidence = None
        reasoning = None
        error_reason = None
        if isinstance(result, dict):
            success = bool(result.get("success"))
            confidence = result.get("confidence")
            reasoning = result.get("reasoning")
            error_reason = result.get("error")
        else:
            success = bool(getattr(result, "success", False))
            confidence = getattr(result, "confidence", None)
            reasoning = getattr(result, "reasoning", None)
            error_reason = getattr(result, "error", None)

        if success:
            self._mark_task_completed(
                task_key,
                {
                    "type": plan_label,
                    "success": True,
                    "confidence": confidence,
                    "reasoning": reasoning,
                }
            )
        else:
            self._mark_task_failed(task_key, error_reason or "Sub-agent task did not complete successfully.")
        self._drain_sub_agent_results()
    
    def _orchestrate_parallel_work(self, user_prompt: str) -> None:
        if self._parallel_plan_done:
            return
        if not self.sub_agent_controller or not getattr(self.bot, "tab_manager", None):
            self._parallel_plan_done = True
            return
        if self.agent_context and self.agent_context.parent_agent_id:
            self._parallel_plan_done = True
            return
        
        plan = self._create_parallel_plan(user_prompt)
        if not plan or not plan.sub_tasks:
            print("⚠️ Parallel work plan could not be generated. Continuing without orchestration.")
            self._parallel_plan_done = True
            return
        
        self._parallel_plan = plan
        self._main_agent_focus = plan.main_agent_focus
        self._integration_notes = plan.integration_notes
        self._log_event("parallel_plan_created", plan=plan.dict())
        for sub_task in plan.sub_tasks:
            task_key = self._normalize_task_name(sub_task.name)
            self._register_task(task_key, sub_task.instruction or sub_task.name, "parallel_task")
        if plan.main_agent_focus:
            focus_key = self._normalize_task_name(f"focus:{plan.main_agent_focus}")
            self._register_task(focus_key, plan.main_agent_focus, "integration")
        elif self._integration_notes:
            notes_key = self._normalize_task_name(f"notes:{self._integration_notes}")
            self._register_task(notes_key, self._integration_notes, "integration")
        
        if plan.integration_notes:
            print(f"🧩 Parallel integration notes: {plan.integration_notes}")
        if plan.main_agent_focus:
            print(f"🧭 Main agent focus set to: {plan.main_agent_focus}")
        elif self._main_agent_focus:
            print(f"🧭 Main agent continuing focus: {self._main_agent_focus}")
        
        original_tab_id = self.agent_context.tab_id if self.agent_context else None
        
        for task in plan.sub_tasks:
            self._run_orchestrated_task(task, "parallel_plan")
        
        if original_tab_id and self.bot.tab_manager:
            main_tab = self.bot.tab_manager.get_tab_info(original_tab_id)
            if main_tab:
                self.bot.tab_manager.switch_to_tab(original_tab_id)
                self.bot.switch_to_page(main_tab.page)
        
        self._parallel_plan_done = True
    
    def _create_parallel_plan(self, user_prompt: str) -> Optional[_ParallelWorkPlan]:
        subjects = self._extract_requested_subjects(user_prompt)
        manual_plan = self._build_manual_parallel_plan(subjects, user_prompt)
        if manual_plan:
            return manual_plan

        prompt = self._build_parallel_plan_prompt(user_prompt)
        try:
            plan = generate_model(
                prompt=prompt,
                model_object_type=_ParallelWorkPlan,
                system_prompt=self._build_parallel_plan_system_prompt(),
                model=self.agent_model_name,
                reasoning_level=self.agent_reasoning_level,
            )
            return plan
        except Exception as e:
            print(f"⚠️ Failed to generate parallel work plan: {e}")
            return None
    
    @staticmethod
    def _build_parallel_plan_system_prompt() -> str:
        return """You are the coordinator for a multi-agent browser automation system.
Break the user's high-level goal into:
- A concise focus for the main agent (the coordinator)
- 2-4 sub-agent tasks, each with clear instructions, an optional starting URL, and an expected output description.
Ensure sub-tasks are independent and collectively satisfy the overall request.
Create one sub-task per distinct source or deliverable; do not bundle multiple independent targets into a single task.
Return structured data in the provided schema only."""
    
    def _build_parallel_plan_prompt(self, user_prompt: str) -> str:
        tab_summary = self._build_tab_summary()
        tab_summary_text = json.dumps(tab_summary, indent=2) if tab_summary else "[]"
        base_rules = "\n".join(self.base_knowledge) if self.base_knowledge else "None provided."
        recent_results = [
            result.to_dict() for result in self.sub_agent_results[-5:]
        ] if self.sub_agent_results else []
        recent_results_text = json.dumps(recent_results, indent=2) if recent_results else "[]"
        
        return f"""
PARALLEL WORK PLANNING
======================

USER PROMPT:
{user_prompt}

CURRENT POLICY LEVEL: {self.sub_agent_policy_level.value} (score {self._sub_agent_policy_score:.2f})
OVERRIDE: {self._sub_agent_policy_override.value if self._sub_agent_policy_override else "None"}

KNOWN TABS:
{tab_summary_text}

RECENT SUB-AGENT RESULTS (if any):
{recent_results_text}

BASE KNOWLEDGE RULES:
{base_rules}

Provide a plan that:
- Gives 'main_agent_focus' (the primary coordinator's responsibility).
- Lists 'sub_tasks' (each with name, instruction, optional suggested_url, and expected_output). Each sub-task should target exactly one independent source or deliverable (e.g., one website, document, or dataset).
- Optionally adds 'integration_notes' explaining how the outputs should be combined.
- If the user mentions multiple sources (e.g., several websites or datasets), create a separate sub-task for each source so they can run in parallel or sequentially without overlap.
"""
    
    def _can_spawn_sub_agent(self) -> Tuple[bool, str]:
        """
        Determine if a sub-agent can be spawned under the current policy and budgets.
        """
        if not self.sub_agent_controller:
            return False, "Sub-agent controller not initialized."
        
        if self.sub_agent_policy_level == SubAgentPolicyLevel.SINGLE_THREADED:
            return False, "Policy set to single-threaded."
        
        return True, ""
    
    def _execute_tab_decision(self, decision) -> bool:
        """
        Execute a tab management decision.
        
        Args:
            decision: TabDecision object with action to take
        
        Returns:
            True if action was executed, False otherwise
        """
        if not self.bot.tab_manager:
            return False
        
        try:
            if decision.action == TabAction.SWITCH:
                if not decision.target_tab_id:
                    print("⚠️ Tab decision: SWITCH but no target_tab_id provided")
                    return False
                
                target_tab = self.bot.tab_manager.get_tab_info(decision.target_tab_id)
                if not target_tab:
                    print(f"⚠️ Tab decision: SWITCH to {decision.target_tab_id} but tab not found")
                    return False
                
                print(f"🔀 Tab Decision: Switching to tab {decision.target_tab_id} ({target_tab.purpose})")
                print(f"   Reasoning: {decision.reasoning}")
                print(f"   Confidence: {decision.confidence:.2f}")
                
                # Switch tab
                if self.bot.tab_manager.switch_to_tab(decision.target_tab_id):
                    # Switch bot to the new page
                    self.bot.switch_to_page(target_tab.page)
                    return True
                else:
                    print(f"⚠️ Failed to switch to tab {decision.target_tab_id}")
                    return False
            
            elif decision.action == TabAction.CLOSE:
                if not decision.target_tab_id:
                    print("⚠️ Tab decision: CLOSE but no target_tab_id provided")
                    return False
                
                target_tab = self.bot.tab_manager.get_tab_info(decision.target_tab_id)
                if not target_tab:
                    print(f"⚠️ Tab decision: CLOSE {decision.target_tab_id} but tab not found")
                    return False
                
                print(f"🗑️ Tab Decision: Closing tab {decision.target_tab_id} ({target_tab.purpose})")
                print(f"   Reasoning: {decision.reasoning}")
                print(f"   Confidence: {decision.confidence:.2f}")
                
                # Find another tab to switch to if closing active tab
                current_tab = self.bot.tab_manager.get_active_tab()
                switch_to = None
                if current_tab and current_tab.tab_id == decision.target_tab_id:
                    # Closing active tab, find another
                    other_tabs = [t for t in self.bot.tab_manager.list_tabs() if t.tab_id != decision.target_tab_id]
                    if other_tabs:
                        switch_to = other_tabs[0].tab_id
                
                # Close tab
                if self.bot.tab_manager.close_tab(decision.target_tab_id, switch_to=switch_to):
                    # If we switched to another tab, update bot
                    if switch_to:
                        new_tab = self.bot.tab_manager.get_tab_info(switch_to)
                        if new_tab:
                            self.bot.switch_to_page(new_tab.page)
                    return True
                else:
                    print(f"⚠️ Failed to close tab {decision.target_tab_id}")
                    return False
            
            elif decision.action == TabAction.CONTINUE:
                # No action needed, just log
                print("✅ Tab Decision: Continue on current tab")
                print(f"   Reasoning: {decision.reasoning}")
                return False  # Don't skip iteration, just continue normally
            
            elif decision.action == TabAction.SPAWN_SUB_AGENT:
                # Phase 3: Spawn sub-agent for another tab
                if not decision.target_tab_id:
                    print("⚠️ Tab decision: SPAWN_SUB_AGENT but no target_tab_id provided")
                    return False
                
                if not self.sub_agent_controller:
                    print("⚠️ Cannot spawn sub-agent: SubAgentController not initialized")
                    return False
                
                policy_allowed, policy_reason = self._can_spawn_sub_agent()
                if not policy_allowed:
                    print(f"🚫 Sub-agent spawn skipped ({policy_reason})")
                    self._log_event(
                        "sub_agent_spawn_blocked",
                        policy=self.sub_agent_policy_level.value,
                        reason=policy_reason,
                        override=self._sub_agent_policy_override.value if self._sub_agent_policy_override else None,
                        target_tab_id=decision.target_tab_id,
                        iteration=self._current_iteration,
                        spawn_count=self.sub_agent_spawn_count
                    )
                    return False
                
                target_tab: Optional[Any] = None
                created_new_tab = False
                
                if decision.target_tab_id:
                    target_tab = self.bot.tab_manager.get_tab_info(decision.target_tab_id)
                    if not target_tab:
                        print(f"⚠️ Tab decision: SPAWN_SUB_AGENT for {decision.target_tab_id} but tab not found")
                        return False
                else:
                    purpose = decision.target_purpose or "Sub-agent task"
                    metadata = {
                        "spawned_by_decision": True,
                        "decision_reasoning": decision.reasoning,
                    }
                    if self._sub_agent_policy_override:
                        metadata["policy_override"] = self._sub_agent_policy_override.value
                    new_tab_id = self.bot.tab_manager.open_new_tab(
                        purpose=purpose,
                        url=decision.target_url,
                        metadata=metadata
                    )
                    if not new_tab_id:
                        print("⚠️ Failed to create new tab for sub-agent.")
                        return False
                    decision.target_tab_id = new_tab_id
                    created_new_tab = True
                    target_tab = self.bot.tab_manager.get_tab_info(new_tab_id)
                    if not target_tab:
                        print(f"⚠️ Failed to retrieve newly created tab {new_tab_id} for sub-agent.")
                        return False
                
                print(f"🤖 Tab Decision: Spawning sub-agent for tab {decision.target_tab_id} ({target_tab.purpose})")
                print(f"   Reasoning: {decision.reasoning}")
                print(f"   Confidence: {decision.confidence:.2f}")
                if created_new_tab:
                    print(f"   🆕 Created new tab {decision.target_tab_id} (URL: {target_tab.url}) for the sub-agent.")
                
                # Extract instruction from reasoning or use a default
                # The LLM should provide instruction in metadata or we derive from reasoning
                instruction = decision.reasoning  # Default: use reasoning as instruction
                if decision.target_tab_id in self.bot.tab_manager.tabs:
                    tab_metadata = self.bot.tab_manager.tabs[decision.target_tab_id].metadata
                    if "sub_agent_instruction" in tab_metadata:
                        instruction = tab_metadata["sub_agent_instruction"]
                    else:
                        tab_metadata["sub_agent_instruction"] = instruction
                
                # Spawn sub-agent
                sub_agent_id = self.sub_agent_controller.spawn_sub_agent(
                    tab_id=decision.target_tab_id,
                    instruction=instruction,
                    metadata={"spawned_by_decision": True, "reasoning": decision.reasoning}
                )
                
                if sub_agent_id:
                    print(f"   ✅ Sub-agent spawned: {sub_agent_id}")
                    self._log_event(
                        "sub_agent_spawned",
                        agent_id=sub_agent_id,
                        instruction=instruction,
                        tab_id=decision.target_tab_id,
                    )
                    self.sub_agent_spawn_count += 1
                    self._spawns_this_iteration += 1
                    # Execute sub-agent immediately
                    result = self.sub_agent_controller.execute_sub_agent(sub_agent_id)
                    if result.get("success"):
                        print("   ✅ Sub-agent completed successfully")
                        self._log_event(
                            "sub_agent_execution",
                            agent_id=sub_agent_id,
                            success=True,
                            status=result.get("status"),
                        )
                    else:
                        print(f"   ⚠️ Sub-agent failed: {result.get('error', 'Unknown error')}")
                        self._log_event(
                            "sub_agent_execution",
                            agent_id=sub_agent_id,
                            success=False,
                            status=result.get("status"),
                            error=result.get("error"),
                        )
                    
                    # Pull results into controller state
                    self._drain_sub_agent_results()
                    
                    # Switch back to main agent's tab
                    if self.agent_context:
                        main_tab = self.bot.tab_manager.get_tab_info(self.agent_context.tab_id)
                        if main_tab:
                            self.bot.tab_manager.switch_to_tab(self.agent_context.tab_id)
                            self.bot.switch_to_page(main_tab.page)
                    
                    return True
                else:
                    print("   ⚠️ Failed to spawn sub-agent")
                    return False
            
            else:
                print(f"⚠️ Unknown tab action: {decision.action}")
                return False
                
        except Exception as e:
            print(f"⚠️ Error executing tab decision: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_sub_agent_results(self) -> List[SubAgentResult]:
        """Return completed sub-agent results for this controller run."""
        return list(self.sub_agent_results)

    def _log_event(self, event_type: str, **data: Any) -> None:
        event = {
            "type": event_type,
            "timestamp": time.time(),
        }
        if data:
            event.update(data)
        self.orchestration_events.append(event)

