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
        self.active_goals: List[BaseGoal] = []
        self.interaction_history: List[Interaction] = []
        self.state_history: List[BrowserState] = []
        self.url_history: List[str] = []
        self.session_start_time = time.time()
        self.element_analyzer = ElementAnalyzer(page)
        
        # Initialize with current state
        self._capture_initial_state()
    
    def add_goal(self, goal: BaseGoal) -> None:
        """Add a goal to be monitored"""
        goal.start_monitoring()
        
        # Set element analyzer for goals that need it
        if hasattr(goal, 'set_element_analyzer'):
            goal.set_element_analyzer(self.element_analyzer)
        
        self.active_goals.append(goal)
        print(f"[GoalMonitor] Added goal: {goal}")
    
    def remove_goal(self, goal: BaseGoal) -> None:
        """Remove a goal from monitoring"""
        if goal in self.active_goals:
            goal.stop_monitoring()
            self.active_goals.remove(goal)
            print(f"[GoalMonitor] Removed goal: {goal}")
    
    def record_planned_interaction(self, interaction_type: InteractionType, **kwargs) -> Dict[str, GoalResult]:
        """
        Record a planned interaction BEFORE it happens and evaluate goals based on their timing preferences.
        
        Returns:
            Dict of goal evaluations that occurred before the interaction
        """
        pre_interaction_results = {}
        
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
                for goal in self.active_goals:
                    if goal.EVALUATION_TIMING in (EvaluationTiming.BEFORE, EvaluationTiming.BOTH):
                        try:
                            # Create context with planned interaction data
                            context = self._build_goal_context()
                            context.planned_interaction = planned_interaction_data
                            
                            # Evaluate the goal
                            result = goal.evaluate(context)
                            pre_interaction_results[str(goal)] = result
                            goal._last_evaluation = result
                            print(f"[GoalMonitor] Pre-interaction evaluation: {goal} -> {result.status}")
                            
                            # If goal is achieved before interaction, note it
                            if result.status == GoalStatus.ACHIEVED:
                                print(f"[GoalMonitor] Goal achieved before interaction: {goal}")
                                
                        except Exception as e:
                            print(f"[GoalMonitor] Error in pre-interaction evaluation for {goal}: {e}")
        
        return pre_interaction_results
    
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
            before_state=before_state,
            success=kwargs.get('success', True),
            error_message=kwargs.get('error_message')
        )
        
        # Capture state after interaction (with small delay for page updates)
        time.sleep(0.1)
        interaction.after_state = self._capture_current_state()
        
        self.interaction_history.append(interaction)
        
        # Notify goals
        for goal in self.active_goals:
            goal.on_interaction(interaction)
        
        # Evaluate goals that want AFTER or BOTH timing
        self._evaluate_post_interaction_goals(interaction_type)
        
        # Check for state changes
        if len(self.state_history) > 0:
            old_state = self.state_history[-1]
            new_state = interaction.after_state
            if self._is_significant_state_change(old_state, new_state):
                for goal in self.active_goals:
                    goal.on_state_change(old_state, new_state)
        
        print(f"[GoalMonitor] Recorded {interaction_type} interaction")
    
    def _evaluate_post_interaction_goals(self, interaction_type: InteractionType) -> None:
        """Evaluate goals that want AFTER or BOTH timing after an interaction"""
        for goal in self.active_goals:
            if goal.EVALUATION_TIMING in (EvaluationTiming.AFTER, EvaluationTiming.BOTH):
                try:
                    context = self._build_goal_context()
                    result = goal.evaluate(context)
                    goal._last_evaluation = result
                    print(f"[GoalMonitor] Post-interaction evaluation: {goal} -> {result.status}")
                except Exception as e:
                    print(f"[GoalMonitor] Error in post-interaction evaluation for {goal}: {e}")
    
    def evaluate_goals(self) -> Dict[str, GoalResult]:
        """
        Evaluate all active goals and return their current status.
        This respects each goal's evaluation timing preferences.
        """
        results = {}
        
        for goal in self.active_goals:
            # Use the last evaluation if it exists, otherwise evaluate now
            if goal._last_evaluation:
                results[str(goal)] = goal._last_evaluation
            else:
                # For CONTINUOUS goals, evaluate now
                if goal.EVALUATION_TIMING == EvaluationTiming.CONTINUOUS:
                    try:
                        context = self._build_goal_context()
                        result = goal.evaluate(context)
                        goal._last_evaluation = result
                        results[str(goal)] = result
                    except Exception as e:
                        error_result = GoalResult(
                            status=GoalStatus.UNKNOWN,
                            confidence=0.0,
                            reasoning=f"Error evaluating goal: {e}",
                            evidence={"error": str(e)}
                        )
                        results[str(goal)] = error_result
                else:
                    # For BEFORE/AFTER goals, return pending if no evaluation yet
                    results[str(goal)] = GoalResult(
                        status=GoalStatus.PENDING,
                        confidence=1.0,
                        reasoning=f"Goal with {goal.EVALUATION_TIMING} timing waiting for appropriate evaluation trigger"
                    )
        
        return results
    
    def _capture_initial_state(self) -> None:
        """Capture the initial browser state"""
        initial_state = self._capture_current_state()
        self.state_history.append(initial_state)
        self.url_history.append(initial_state.url)
    
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
        
        return GoalContext(
            initial_state=self.state_history[0] if self.state_history else current_state,
            current_state=current_state,
            all_interactions=self.interaction_history.copy(),
            url_history=self.url_history.copy(),
            page_changes=self.state_history.copy(),
            session_duration=time.time() - self.session_start_time,
            page_reference=self.page  # Provide page access for advanced goals
        )
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of all goal statuses"""
        results = self.evaluate_goals()
        
        summary = {
            "total_goals": len(self.active_goals),
            "achieved": sum(1 for r in results.values() if r.status == GoalStatus.ACHIEVED),
            "pending": sum(1 for r in results.values() if r.status == GoalStatus.PENDING),
            "failed": sum(1 for r in results.values() if r.status == GoalStatus.FAILED),
            "unknown": sum(1 for r in results.values() if r.status == GoalStatus.UNKNOWN),
            "interactions_count": len(self.interaction_history),
            "session_duration": time.time() - self.session_start_time,
            "goals": results
        }
        
        return summary
