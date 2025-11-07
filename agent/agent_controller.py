"""
Agent Controller - Step 2: LLM-Based Completion + Reactive Goal Determination

Implements agentic mode with:
- Reactive loop: observe → check completion (LLM) → determine goal → act → repeat
- LLM-based completion evaluation (CompletionContract)
- EnvironmentState for full context
- Reactive goal determination (what to do now)
"""

import time
from typing import Optional, List, Dict, Any
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
    
    def __init__(self, bot, track_ineffective_actions: bool = True, base_knowledge: Optional[List[str]] = None):
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
        self.base_knowledge = base_knowledge or []  # Base knowledge rules that guide agent behavior
        self.failed_actions: List[str] = []  # Track actions that failed AND didn't yield any change
        self.ineffective_actions: List[str] = []  # Track actions that succeeded BUT didn't yield any change
        self.extracted_data: Dict[str, Any] = {}  # Store extracted data (key: extraction prompt, value: result)
        self.sub_agent_results: List[SubAgentResult] = []  # Store completed sub-agent results
        self.orchestration_events: List[Dict[str, Any]] = []  # Track orchestration events for reporting
        
        # Initialize TabDecisionEngine if TabManager is available
        self.tab_decision_engine: Optional[TabDecisionEngine] = None
        if hasattr(bot, 'tab_manager') and bot.tab_manager:
            try:
                self.tab_decision_engine = TabDecisionEngine(bot.tab_manager)
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
        self.sub_agent_results = []
        self.orchestration_events = []
        
        # Phase 3: Set agent context
        if agent_context:
            self.agent_context = agent_context
            # Initialize sub-agent controller if this is main agent
            if agent_context.parent_agent_id is None:
                if not self.sub_agent_controller:
                    self.sub_agent_controller = SubAgentController(self.bot, agent_context)
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
                        self.sub_agent_controller = SubAgentController(self.bot, self.agent_context)
        
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
        completion_contract = CompletionContract(user_prompt)
        
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
                                "completion_reasoning": completion_reasoning
                            }
                        )
                        
                        # Execute decision if needed
                        if tab_decision.should_take_action:
                            executed = self._execute_tab_decision(tab_decision)
                            if executed:
                                # Tab switch/close happened, continue to next iteration
                                print(f"🔄 Tab action executed, continuing to next iteration...")
                                time.sleep(self.iteration_delay)
                                continue
                    except Exception as e:
                        print(f"⚠️ Error in tab decision making: {e}")
            
            # 3. Determine next action - this will tell us if exploration is needed
            # Create reactive goal determiner (Step 2: determines next action from viewport)
            goal_determiner = ReactiveGoalDeterminer(user_prompt, base_knowledge=self.base_knowledge)
            
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
            if self.track_ineffective_actions and current_action and (current_action in self.failed_actions or current_action in self.ineffective_actions):
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
                    if self.track_ineffective_actions and current_action and (current_action in self.failed_actions or current_action in self.ineffective_actions) and not current_action.startswith("scroll:"):
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
                print(f"📊 Extraction detected: {extraction_prompt}")
                
                try:
                    # Call extract() directly - simpler and more efficient
                    result = self.bot.extract(
                        prompt=extraction_prompt,
                        output_format="json",
                        scope="viewport"
                    )
                    
                    # Store extracted data with the prompt as key
                    self.extracted_data[extraction_prompt] = result
                    print(f"✅ Extraction completed: {result}")
                    
                    # Continue to next iteration (extraction is complete)
                    time.sleep(self.iteration_delay)
                    continue
                except Exception as e:
                    print(f"⚠️ Extraction failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue to next iteration anyway
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
                
                # Check if action yielded any change
                page_changed = self._page_state_changed(state_before, state_after)
                
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
                            print(f"⚠️ Action succeeded but did not yield any change: {current_action}")
                            print(f"   URL before: {state_before['url']}")
                            print(f"   URL after: {state_after['url']}")
                            print(f"   DOM signature unchanged")
                            # Add to ineffective actions list (nudge to try something different)
                            if current_action not in self.ineffective_actions:
                                self.ineffective_actions.append(current_action)
                                print(f"   📝 Added to ineffective actions list (will try different approach in future iterations)")
                    else:
                        # Failed command - check if it yielded any change
                        if not page_changed:
                            print(f"⚠️ Action failed and did not yield any change: {current_action}")
                            print(f"   URL before: {state_before['url']}")
                            print(f"   URL after: {state_after['url']}")
                            print(f"   DOM signature unchanged")
                            # Add to failed actions list (avoid trying this again)
                            if current_action not in self.failed_actions:
                                self.failed_actions.append(current_action)
                                print(f"   📝 Added to failed actions list (will avoid in future iterations)")
                
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
            target = action_command.split(":", 1)[1].strip() if ":" in action_command else ""
            history = self.bot.goal_monitor.url_history if hasattr(self.bot, "goal_monitor") else []
            pointer = getattr(self.bot.goal_monitor, "url_pointer", len(history) - 1) if hasattr(self.bot, "goal_monitor") else -1

            if history and pointer is not None and pointer >= 0:
                back_steps = self._history_steps_to_target(history, pointer, target, "back")
                if back_steps:
                    new_action = f"back: {back_steps}"
                    print(f"ℹ️ Converted navigate command to back navigation: {new_action} (target: {target})")
                    return new_action

                forward_steps = self._history_steps_to_target(history, pointer, target, "forward")
                if forward_steps:
                    new_action = f"forward: {forward_steps}"
                    print(f"ℹ️ Converted navigate command to forward navigation: {new_action} (target: {target})")
                    return new_action

            print(f"ℹ️ Navigation command preserved (no history shortcut available): {action_command}")
            return action_command
        
        return action_command
    
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
            "datetime"
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
            return {
                "url": url,
                "dom_signature": dom_signature
            }
        except Exception:
            # Fallback to just URL
            try:
                url = self.bot.page.url
                dom_signature = hashlib.md5(url.encode("utf-8")).hexdigest()
                return {
                    "url": url,
                    "dom_signature": dom_signature
                }
            except Exception:
                return {
                    "url": "",
                    "dom_signature": ""
                }
    
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
        
        return False
    
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
                    print(f"⚠️ Tab decision: SWITCH but no target_tab_id provided")
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
                    print(f"⚠️ Tab decision: CLOSE but no target_tab_id provided")
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
                print(f"✅ Tab Decision: Continue on current tab")
                print(f"   Reasoning: {decision.reasoning}")
                return False  # Don't skip iteration, just continue normally
            
            elif decision.action == TabAction.SPAWN_SUB_AGENT:
                # Phase 3: Spawn sub-agent for another tab
                if not decision.target_tab_id:
                    print(f"⚠️ Tab decision: SPAWN_SUB_AGENT but no target_tab_id provided")
                    return False
                
                if not self.sub_agent_controller:
                    print(f"⚠️ Cannot spawn sub-agent: SubAgentController not initialized")
                    return False
                
                target_tab = self.bot.tab_manager.get_tab_info(decision.target_tab_id)
                if not target_tab:
                    print(f"⚠️ Tab decision: SPAWN_SUB_AGENT for {decision.target_tab_id} but tab not found")
                    return False
                
                print(f"🤖 Tab Decision: Spawning sub-agent for tab {decision.target_tab_id} ({target_tab.purpose})")
                print(f"   Reasoning: {decision.reasoning}")
                print(f"   Confidence: {decision.confidence:.2f}")
                
                # Extract instruction from reasoning or use a default
                # The LLM should provide instruction in metadata or we derive from reasoning
                instruction = decision.reasoning  # Default: use reasoning as instruction
                if decision.target_tab_id in self.bot.tab_manager.tabs:
                    tab_metadata = self.bot.tab_manager.tabs[decision.target_tab_id].metadata
                    if "sub_agent_instruction" in tab_metadata:
                        instruction = tab_metadata["sub_agent_instruction"]
                
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
                    # Execute sub-agent immediately
                    result = self.sub_agent_controller.execute_sub_agent(sub_agent_id)
                    if result.get("success"):
                        print(f"   ✅ Sub-agent completed successfully")
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
                    print(f"   ⚠️ Failed to spawn sub-agent")
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

