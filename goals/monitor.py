"""
Goal Monitor - Central coordinator for goal monitoring and evaluation.
"""
import time
from typing import Any, Dict, List

from playwright.sync_api import Page

from .base import (
    BaseGoal, BrowserState, GoalContext, GoalResult, GoalStatus,
    Interaction, InteractionType, EvaluationTiming
)
from .element_analyzer import ElementAnalyzer


class GoalMonitor:
    """
    Central coordinator for goal monitoring and evaluation.
    
    This class integrates deeply with the browser automation to track
    all interactions and state changes, providing comprehensive context
    to goals for evaluation.
    """
    
    def __init__(self, page: Page):
        self.page = page
        self.user_prompt = ""
        self.active_goal: BaseGoal = None
        self.interaction_history: List[Interaction] = []
        self.state_history: List[BrowserState] = []
        self.url_history: List[str] = []
        self.url_pointer: int = 0
        self.session_start_time = time.time()
        self.element_analyzer = ElementAnalyzer(page)
        
        # Initialize with current state
        self._capture_initial_state()
    
    def set_user_prompt(self, user_prompt: str) -> None:
        """Set the user prompt"""
        self.user_prompt = user_prompt
    
    def add_goal(self, goal: BaseGoal) -> None:
        """Add a goal to be monitored"""
        goal.start_monitoring()
        
        # Set element analyzer for goals that need it
        if hasattr(goal, 'set_element_analyzer'):
            goal.set_element_analyzer(self.element_analyzer)
        
        # Set goal monitor reference for goals that need it (like WhileGoal with command queues)
        if hasattr(goal, 'set_goal_monitor'):
            goal.set_goal_monitor(self)
        
        self.active_goal = goal
        print(f"[GoalMonitor] Added goal: {goal}")
    
    def remove_goal(self, goal: BaseGoal) -> None:
        """Remove a goal from monitoring"""
        if goal == self.active_goal:
            goal.stop_monitoring()
            self.active_goal = None
            print(f"[GoalMonitor] Removed goal: {goal}")
    
    def record_planned_interaction(self, interaction_type: InteractionType, **kwargs) -> GoalResult:
        """
        Record a planned interaction BEFORE it happens and evaluate goals based on their timing preferences.
        
        Returns:
            Dict of goal evaluations that occurred before the interaction
        """
        pre_interaction_results = None
        
        if interaction_type == InteractionType.CLICK:
            coordinates = kwargs.get('coordinates')
            if coordinates:
                # Add screenshot to kwargs for pre-interaction evaluation
                try:
                    screenshot = self._capture_current_state().screenshot
                    kwargs['screenshot'] = screenshot
                except Exception:
                    pass
                
                # Create planned interaction data
                planned_interaction_data = {
                    'interaction_type': interaction_type,
                    'coordinates': coordinates,
                    'screenshot': kwargs.get('screenshot'),
                    **kwargs
                }
                
                # Evaluate goals that want BEFORE or BOTH timing
                if self.active_goal.EVALUATION_TIMING in (EvaluationTiming.BEFORE, EvaluationTiming.BOTH):
                    try:
                        # Create context with planned interaction data
                        context = self._build_goal_context()
                        context.planned_interaction = planned_interaction_data
                        
                        # Evaluate the goal
                        result = self.active_goal.evaluate(context)
                        pre_interaction_results = result
                        self.active_goal._last_evaluation = result
                        print(f"[GoalMonitor] Pre-interaction evaluation: {self.active_goal} -> {result.status}")
                        
                        # If goal is achieved before interaction, note it
                        if result.status == GoalStatus.ACHIEVED:
                            print(f"[GoalMonitor] Goal achieved before interaction: {self.active_goal}")
                            
                    except Exception as e:
                        print(f"[GoalMonitor] Error in pre-interaction evaluation for {self.active_goal}: {e}")
        
        elif interaction_type == InteractionType.PRESS:
            keys_to_press = kwargs.get('keys_to_press')
            if keys_to_press:
                # Create planned interaction data
                planned_interaction_data = {
                    'interaction_type': interaction_type,
                    'keys_to_press': keys_to_press,
                    **kwargs
                }
                
                # Evaluate goals that want BEFORE or BOTH timing
                if self.active_goal.EVALUATION_TIMING in (EvaluationTiming.BEFORE, EvaluationTiming.BOTH):
                    try:
                        # Create context with planned interaction data
                        context = self._build_goal_context()
                        context.planned_interaction = planned_interaction_data
                        
                        # Evaluate the goal
                        result = self.active_goal.evaluate(context)
                        pre_interaction_results = result
                        self.active_goal._last_evaluation = result
                        print(f"[GoalMonitor] Pre-interaction evaluation: {self.active_goal} -> {result.status}")
                        
                        # If goal is achieved before interaction, note it
                        if result.status == GoalStatus.ACHIEVED:
                            print(f"[GoalMonitor] Goal achieved before interaction: {self.active_goal}")
                            
                    except Exception as e:
                        print(f"[GoalMonitor] Error in pre-interaction evaluation for {self.active_goal}: {e}")
        
        elif interaction_type == InteractionType.SCROLL:
            target_x = kwargs.get('target_x')
            target_y = kwargs.get('target_y')
            scroll_direction = kwargs.get('scroll_direction')
            if target_x is not None and target_y is not None and scroll_direction:
                # Create planned interaction data
                planned_interaction_data = {
                    'interaction_type': interaction_type,
                    'target_x': target_x,
                    'target_y': target_y,
                    'scroll_direction': scroll_direction,
                    **kwargs
                }
                
                # Evaluate goals that want BEFORE or BOTH timing
                if self.active_goal.EVALUATION_TIMING in (EvaluationTiming.BEFORE, EvaluationTiming.BOTH):
                    try:
                        # Create context with planned interaction data
                        context = self._build_goal_context()
                        context.planned_interaction = planned_interaction_data
                        
                        # Evaluate the goal
                        result = self.active_goal.evaluate(context)
                        pre_interaction_results = result
                        self.active_goal._last_evaluation = result
                        print(f"[GoalMonitor] Pre-interaction evaluation: {self.active_goal} -> {result.status}")
                        
                        # If goal is achieved before interaction, note it
                        if result.status == GoalStatus.ACHIEVED:
                            print(f"[GoalMonitor] Goal achieved before interaction: {self.active_goal}")
                            
                    except Exception as e:
                        print(f"[GoalMonitor] Error in pre-interaction evaluation for {self.active_goal}: {e}")
        
        elif interaction_type in (InteractionType.SELECT, InteractionType.TYPE, InteractionType.DATETIME):
            coordinates = kwargs.get('coordinates')
            if coordinates:
                # Add screenshot to kwargs for pre-interaction evaluation
                try:
                    screenshot = self._capture_current_state().screenshot
                    kwargs['screenshot'] = screenshot
                except Exception:
                    pass
                
                # Create planned interaction data
                planned_interaction_data = {
                    'interaction_type': interaction_type,
                    'coordinates': coordinates,
                    'screenshot': kwargs.get('screenshot'),
                    'target_description': kwargs.get('target_description', ''),
                    **kwargs
                }
                
                # Evaluate goals that want BEFORE or BOTH timing
                if self.active_goal.EVALUATION_TIMING in (EvaluationTiming.BEFORE, EvaluationTiming.BOTH):
                    try:
                        # Create context with planned interaction data
                        context = self._build_goal_context()
                        context.planned_interaction = planned_interaction_data
                        
                        # Evaluate the goal
                        result = self.active_goal.evaluate(context)
                        pre_interaction_results = result
                        self.active_goal._last_evaluation = result
                        print(f"[GoalMonitor] Pre-interaction evaluation: {self.active_goal} -> {result.status}")
                        
                        # If goal is achieved before interaction, note it
                        if result.status == GoalStatus.ACHIEVED:
                            print(f"[GoalMonitor] Goal achieved before interaction: {self.active_goal}")
                            
                    except Exception as e:
                        print(f"[GoalMonitor] Error in pre-interaction evaluation for {self.active_goal}: {e}")
        
        return pre_interaction_results
    
    def check_for_retry_request(self) -> BaseGoal:
        """
        Check if any goals have requested a retry.
        
        Returns:
            List of goals that have requested retries
        """
        retry_goal = None
        if self.active_goal:
            if self.active_goal.retry_requested:
                retry_goal = self.active_goal
        return retry_goal
    
    def reset_retry_request(self) -> None:
        """Reset retry requests for all goals"""
        if self.active_goal:
            self.active_goal.reset_retry_state()
    
    def clear_all_goals(self) -> None:
        """Clear all active goals"""
        if self.active_goal:
            self.remove_goal(self.active_goal)
    
    def record_interaction(self, interaction_type: InteractionType, **kwargs) -> None:
        """
        Record an interaction that has occurred.
        This should be called AFTER the interaction completes.
        """
        before_state = self._capture_current_state()
        
        interaction = Interaction(
            timestamp=time.time(),
            interaction_type=interaction_type,
            coordinates=kwargs.get('coordinates'),
            target_element_info=kwargs.get('target_element_info'),
            text_input=kwargs.get('text_input'),
            keys_pressed=kwargs.get('keys_pressed'),
            scroll_direction=kwargs.get('scroll_direction'),
            scroll_axis=kwargs.get('scroll_axis'),
            target_x=kwargs.get('target_x'),
            target_y=kwargs.get('target_y'),
            before_state=before_state,
            success=kwargs.get('success', True),
            error_message=kwargs.get('error_message')
        )
        
        # Capture state after interaction (with small delay for page updates)
        time.sleep(0.1)
        interaction.after_state = self._capture_current_state()
        
        self.interaction_history.append(interaction)
        
        # Notify goals
        if self.active_goal:
            self.active_goal.on_interaction(interaction)
        
        # Evaluate goals that want AFTER or BOTH timing
        self._evaluate_post_interaction_goals(interaction_type)
        
        # Check for state changes
        if len(self.state_history) > 0:
            old_state = self.state_history[-1]
            new_state = interaction.after_state
            if self._is_significant_state_change(old_state, new_state):
                if self.active_goal:
                    self.active_goal.on_state_change(old_state, new_state)
        
        # Maintain URL history pointer based on the latest state
        try:
            current_url = interaction.after_state.url if interaction.after_state else (self.page.url if self.page else "")
            if not self.url_history or self.url_history[-1] != current_url:
                self.url_history.append(current_url)
            self.url_pointer = len(self.url_history) - 1
        except Exception:
            pass

        print(f"[GoalMonitor] Recorded {interaction_type} interaction")
    
    def _evaluate_post_interaction_goals(self, interaction_type: InteractionType) -> None:
        """Evaluate goals that want AFTER or BOTH timing after an interaction"""
        if self.active_goal:
            if self.active_goal.EVALUATION_TIMING in (EvaluationTiming.AFTER, EvaluationTiming.BOTH):
                try:
                    context = self._build_goal_context()
                    result = self.active_goal.evaluate(context)
                    self.active_goal._last_evaluation = result
                    print(f"[GoalMonitor] Post-interaction evaluation: {self.active_goal} -> {result.status} -> {result.reasoning}")
                    
                    # Check if this goal requested a retry during evaluation
                    if self.active_goal.retry_requested:
                        print(f"[GoalMonitor] Goal {self.active_goal} requested retry during post-interaction evaluation")
                        
                except Exception as e:
                    print(f"[GoalMonitor] Error in post-interaction evaluation for {self.active_goal}: {e}")
    
    def evaluate_goal(self) -> GoalResult:
        """
        Evaluate active goal and return its current status.
        This respects each goal's evaluation timing preferences.
        """
        result = None
        
        if self.active_goal:
            # For CONTINUOUS goals, always evaluate (don't use cache)
            if self.active_goal.EVALUATION_TIMING == EvaluationTiming.CONTINUOUS:
                try:
                    context = self._build_goal_context()
                    result = self.active_goal.evaluate(context)
                    self.active_goal._last_evaluation = result
                    
                    # Check if this goal requested a retry during evaluation
                    if self.active_goal.retry_requested:
                        print(f"[GoalMonitor] Goal {self.active_goal} requested retry during continuous evaluation")
                        
                except Exception as e:
                    error_result = GoalResult(
                        status=GoalStatus.UNKNOWN,
                        confidence=0.0,
                        reasoning=f"Error evaluating goal: {e}",
                        evidence={"error": str(e)}
                    )
                    result = error_result
            # Use the last evaluation if it exists for non-CONTINUOUS goals
            elif self.active_goal._last_evaluation:
                result = self.active_goal._last_evaluation
            else:
                # For BEFORE/AFTER goals, return pending if no evaluation yet
                result = GoalResult(
                    status=GoalStatus.PENDING,
                    confidence=1.0,
                    reasoning=f"Goal with {self.active_goal.EVALUATION_TIMING} timing waiting for appropriate evaluation trigger"
                )
        
        return result
    
    def _capture_initial_state(self) -> None:
        """Capture the initial browser state"""
        initial_state = self._capture_current_state()
        self.state_history.append(initial_state)
        self.url_history.append(initial_state.url)
        self.url_pointer = len(self.url_history) - 1
    
    def _capture_current_state(self) -> BrowserState:
        """Capture current browser state"""
        try:
            viewport = self.page.viewport_size
            scroll_info = self.page.evaluate("() => ({scrollX: window.scrollX, scrollY: window.scrollY})")
            
            return BrowserState(
                timestamp=time.time(),
                url=self.page.url,
                title=self.page.title(),
                page_width=viewport["width"],
                page_height=viewport["height"],
                scroll_x=scroll_info["scrollX"],
                scroll_y=scroll_info["scrollY"],
                screenshot=self.page.screenshot(full_page=False),
                visible_text=self.page.evaluate("document.body.innerText")[:1000]  # Truncate for performance
            )
        except Exception:
            # Fallback state
            return BrowserState(
                timestamp=time.time(),
                url=self.page.url if self.page else "unknown",
                title="",
                page_width=1280,
                page_height=800,
                scroll_x=0,
                scroll_y=0
            )
    
    def _is_significant_state_change(self, old_state: BrowserState, new_state: BrowserState) -> bool:
        """Determine if a state change is significant enough to notify goals"""
        return (
            old_state.url != new_state.url or
            abs(old_state.scroll_x - new_state.scroll_x) > 50 or
            abs(old_state.scroll_y - new_state.scroll_y) > 50 or
            old_state.title != new_state.title
        )
    
    def _build_goal_context(self) -> GoalContext:
        """Build comprehensive context for goal evaluation"""
        current_state = self._capture_current_state()
        self.state_history.append(current_state)
        
        # Update URL history if changed
        if not self.url_history or self.url_history[-1] != current_state.url:
            self.url_history.append(current_state.url)
        self.url_pointer = len(self.url_history) - 1
        
        context = GoalContext(
            initial_state=self.state_history[0] if self.state_history else current_state,
            current_state=current_state,
            all_interactions=self.interaction_history.copy(),
            url_history=self.url_history.copy(),
            page_changes=self.state_history.copy(),
            session_duration=time.time() - self.session_start_time,
            page_reference=self.page  # Provide page access for advanced goals
        )
        # Attach pointer dynamically for consumers that expect it
        try:
            setattr(context, 'url_pointer', self.url_pointer)
        except Exception:
            pass
        return context
    
    def is_goal_achieved(self, goal: BaseGoal) -> bool:
        """Check if a specific goal is achieved"""
        if goal != self.active_goal:
            return False
        
        # Use the last evaluation if it exists
        if goal._last_evaluation:
            return goal._last_evaluation.status == GoalStatus.ACHIEVED
        
        # For goals that haven't been evaluated yet, evaluate now
        try:
            context = self._build_goal_context()
            result = goal.evaluate(context)
            goal._last_evaluation = result
            return result.status == GoalStatus.ACHIEVED
        except Exception:
            return False
    
    def is_goal_failed(self, goal: BaseGoal) -> bool:
        """Check if a specific goal has failed"""
        if goal != self.active_goal:
            return False
        
        # Use the last evaluation if it exists
        if goal._last_evaluation:
            return goal._last_evaluation.status == GoalStatus.FAILED
        
        # For goals that haven't been evaluated yet, evaluate now
        try:
            context = self._build_goal_context()
            result = goal.evaluate(context)
            goal._last_evaluation = result
            return result.status == GoalStatus.FAILED
        except Exception:
            return False

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of all goal statuses"""
        result = self.evaluate_goal()
        
        summary = {
            "achieved": 1 if result.status == GoalStatus.ACHIEVED else 0,
            "pending": 1 if result.status == GoalStatus.PENDING else 0,
            "failed": 1 if result.status == GoalStatus.FAILED else 0,
            "unknown": 1 if result.status == GoalStatus.UNKNOWN else 0,
            "interactions_count": 1 if self.active_goal else 0,
            "session_duration": time.time() - self.session_start_time,
            "goal": result
        }
        
        return summary
