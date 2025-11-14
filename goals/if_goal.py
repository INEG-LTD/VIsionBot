"""
IfGoal implementation for conditional goal execution.

Now supports two distinct evaluation routes:
- "see": Vision-based condition evaluation using screenshots
- "page": Page-based condition evaluation using DOM/state detection
"""
from __future__ import annotations

from typing import Optional, Dict, Any, Union
from .base import ConditionalGoal, Condition, BaseGoal, GoalResult, GoalStatus, create_environment_condition

# Type alias for JSON expressions
JsonExpr = Union[Dict[str, Any], list, str, int, float, bool, None]

class IfGoal(ConditionalGoal):
    """
    A conditional goal that works like an if statement.
    
    If the condition is true, runs the success goal.
    If the condition is false, runs the fail goal.
    
    Enhanced to support multi-command success/fail actions through CommandQueue.
    
    This is the most basic form of conditional goal execution,
    providing simple branching logic based on condition evaluation.
    
    Example:
        ```python
        # Create a condition
        condition = is_weekday_condition()
        
        # Create sub-goals
        success_goal = SomeGoal("Do weekday action")
        fail_goal = SomeGoal("Do weekend action")
        
        # Create the if goal
        if_goal = IfGoal(condition, success_goal, fail_goal)
        
        # Use like any other goal
        result = if_goal.evaluate(context)
        ```
    """
    
    def __init__(self, condition_text: str, success_goal: BaseGoal,
                 fail_goal: Optional[BaseGoal], route: str, description: Optional[str] = None,
                 max_retries: int = 3, **kwargs):
        """
        Initialize an IfGoal with route-based condition evaluation.
        
        Args:
            condition_text: Natural language condition to evaluate
            success_goal: Goal to run if condition is True
            fail_goal: Goal to run if condition is False
            route: Route for condition evaluation ("see" or "page")
            description: Optional custom description. If None, auto-generates from condition and goals
            max_retries: Maximum number of retries allowed
            **kwargs: Additional arguments passed to BaseGoal
        """
        if route not in ["see", "page"]:
            raise ValueError(f"Invalid route '{route}'. Must be 'see' or 'page'")
        
        # Create condition based on route
        condition = self._create_condition(condition_text, route)
        
        if description is None:
            fail_desc = fail_goal.description if fail_goal else "(no fail action)"
            route_info = f"[{route.upper()}]"
            description = f"{route_info} If {condition_text} then {success_goal.description} else {fail_desc}"
        
        super().__init__(description, condition, success_goal, fail_goal, max_retries, **kwargs)
        
        # Store route and condition text for caching
        self.route = route
        self.condition_text = condition_text
        self._condition_cache: Dict[str, Any] = {}  # Cache for converted conditions
        self._screenshot_cache: Dict[str, bytes] = {}  # Cache for screenshots

    def set_goal_monitor(self, goal_monitor) -> None:
        """Set the goal monitor reference."""
        self.goal_monitor = goal_monitor

    def _get_cached_screenshot(self, context) -> bytes:
        """Cache screenshots to avoid repeated generation"""
        cache_key = f"screenshot::{context.current_state.url}_{context.current_state.timestamp}"
        if cache_key not in self._screenshot_cache:
            page = context.page_reference
            if page:
                self._screenshot_cache[cache_key] = page.screenshot(type="jpeg", quality=50, full_page=False)
            else:
                return None
        return self._screenshot_cache[cache_key]

    def _handle_condition_error(self, route: str, error: Exception) -> bool:
        """Centralized error handling for both routes"""
        print(f"[IfGoal] {route.title()} condition error: {error}")
        return False

    def _create_condition(self, condition_text: str, route: str) -> Condition:
        """Create condition based on specified route"""
        if route == "see":
            return self._create_vision_condition(condition_text)
        elif route == "page":
            return self._create_page_condition(condition_text)
        else:
            raise ValueError(f"Invalid route: {route}")

    def _create_vision_condition(self, condition_text: str) -> Condition:
        """Create vision-based condition using screenshots and natural language"""
        def evaluator(context) -> bool:
            try:
                page = context.page_reference
                if not page:
                    return False
                
                # Use cached screenshot
                screenshot = self._get_cached_screenshot(context)
                if not screenshot:
                    return False
                
                # Single AI call combining question conversion and vision analysis
                from ai_utils import answer_question_with_vision
                result = answer_question_with_vision(
                    f"Is {condition_text}? Answer yes or no.", 
                    screenshot
                )
                
                print(f"🔍 DEBUG: Vision condition result: {result}")
                
                # Parse result
                answer = str(result or "").lower().strip()
                return answer in ['yes', 'true', '1']
                
            except Exception as e:
                return self._handle_condition_error("vision", e)
        
        return create_environment_condition(
            f"Vision condition: {condition_text}",
            evaluator
        )

    def _create_page_condition(self, condition_text: str) -> Condition:
        """Create page-based condition with simplified evaluation"""
        def evaluator(context) -> bool:
            try:
                page = context.page_reference
                if not page:
                    return False
                
                # Use simple text-based evaluation instead of complex condition engine
                return self._evaluate_simple_condition(condition_text, page)
                
            except Exception as e:
                return self._handle_condition_error("page", e)
        
        return create_environment_condition(
            f"Page condition: {condition_text}",
            evaluator
        )

    def _evaluate_simple_condition(self, condition_text: str, page) -> bool:
        """Simple text-based condition evaluation without complex condition engine"""
        try:
            # Get page content for text-based evaluation
            page_content = page.content()
            page_title = page.title()
            page_url = page.url
            
            # Convert condition to lowercase for matching
            condition_lower = condition_text.lower()
            
            # Simple text-based checks
            if "visible" in condition_lower or "appears" in condition_lower:
                # Check if text appears on page
                search_text = condition_text.replace("visible", "").replace("appears", "").strip()
                return search_text.lower() in page_content.lower()
            
            elif "contains" in condition_lower or "has" in condition_lower:
                # Check if page contains specific text
                search_text = condition_text.replace("contains", "").replace("has", "").strip()
                return search_text.lower() in page_content.lower()
            
            elif "url" in condition_lower:
                # Check URL-based conditions
                if "login" in condition_lower:
                    return "login" in page_url.lower()
                elif "dashboard" in condition_lower:
                    return "dashboard" in page_url.lower()
                else:
                    # Generic URL check
                    search_text = condition_text.replace("url", "").strip()
                    return search_text.lower() in page_url.lower()
            
            elif "title" in condition_lower:
                # Check title-based conditions
                search_text = condition_text.replace("title", "").strip()
                return search_text.lower() in page_title.lower()
            
            else:
                # Default: check if condition text appears anywhere on page
                return condition_text.lower() in page_content.lower()
                
        except Exception as e:
            print(f"[IfGoal] Simple condition evaluation error: {e}")
            return False

    
    def evaluate(self, context) -> GoalResult:
        """
        Evaluate the condition and run the appropriate sub-goal.
        
        This method:
        1. Evaluates the condition using the provided evaluator function
        2. Selects either the success_goal or fail_goal based on the result
        3. Executes the selected sub-goal (with multi-command support) and returns its result
        
        Args:
            context: GoalContext containing browser state and interaction history
            
        Returns:
            GoalResult from the active sub-goal
        """
        try:
            # Evaluate the condition once
            condition_result = self.condition.evaluator(context)
            self._last_condition_result = condition_result
            
            print("🔍 DEBUG: IfGoal evaluation:")
            print(f"   📋 Condition: {self.condition.description}")
            print(f"   🎯 Condition result: {condition_result}")
            
            # Determine which sub-goal to use
            active_goal = self.success_goal if condition_result else self.fail_goal
            self._current_sub_goal = active_goal

            if active_goal:
                print(f"   🎯 Selected sub-goal: {active_goal.__class__.__name__} - '{active_goal.description}'")
            else:
                print("   🎯 No sub-goal to execute for this branch")

            # Determine what command to execute based on condition
            command_to_execute = active_goal.description if active_goal else None
            
            if command_to_execute:
                # Return pending result with the command to execute
                print(f"🔀 Executing reference command for {'success' if condition_result else 'fail'} action: {command_to_execute}")
                return GoalResult(
                    status=GoalStatus.PENDING,
                    confidence=1.0,
                    reasoning=f"Condition {'TRUE' if condition_result else 'FALSE'}: executing {command_to_execute}",
                    evidence={
                        "condition_type": self.condition.condition_type.value,
                        "condition_result": condition_result,
                        "active_goal": command_to_execute,
                        "route": self.route,
                    },
                    next_actions=[command_to_execute]
                )
            else:
                # No command to execute
                status = GoalStatus.ACHIEVED
                reasoning = (
                    "Condition TRUE: no success action defined"
                    if condition_result
                    else "Condition FALSE: no fail action defined"
                )
                
                return GoalResult(
                    status=status,
                    confidence=1,
                    reasoning=reasoning,
                    evidence={
                        "condition_type": self.condition.condition_type.value,
                        "condition_result": condition_result,
                        "active_goal": None,
                        "route": self.route,
                    }
                )
            
        except Exception as e:
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.0,
                reasoning=f"Error evaluating IfGoal: {str(e)}",
                evidence={"error": str(e), "condition_type": self.condition.condition_type.value, "route": self.route}
            )

    
    def get_condition_summary(self) -> str:
        """
        Get a summary of the condition and its current state.
        
        Returns:
            String summary of the condition
        """
        status = "TRUE" if self.last_condition_result else "FALSE" if self.last_condition_result is not None else "UNEVALUATED"
        return f"Condition: {self.condition.description} (Status: {status})"
    
    def get_active_goal_info(self) -> str:
        """
        Get information about the currently active sub-goal.
        
        Returns:
            String description of the active sub-goal
        """
        if self.current_sub_goal:
            return f"Active Goal: {self.current_sub_goal.description}"
        return "No active sub-goal"

    def get_description(self, context) -> str:
        """Generate description including condition and command queue status"""
        condition_status = "TRUE" if self._last_condition_result else "FALSE" if self._last_condition_result is not None else "UNEVALUATED"
        
        # Get command queue status
        success_queue_status = "N/A"
        fail_queue_status = "N/A"
        
        if self.success_command_queue:
            progress = self.success_command_queue.get_progress()
            success_queue_status = f"{progress['completed_commands']}/{progress['total_commands']} commands"
        
        if self.fail_command_queue:
            progress = self.fail_command_queue.get_progress()
            fail_queue_status = f"{progress['completed_commands']}/{progress['total_commands']} commands"
        
        return (
            f"Goal Type: If Goal\n"
            f"Condition: {self.condition.description}\n"
            f"Condition Status: {condition_status}\n"
            f"Success Action: {self.success_goal.description}\n"
            f"Success Queue: {success_queue_status}\n"
            f"Fail Action: {self.fail_goal.description}\n"
            f"Fail Queue: {fail_queue_status}\n"
            f"Planning Guidance: Execute the appropriate action based on condition result."
        )

    def start_monitoring(self) -> None:
        """Start monitoring the IfGoal"""
        super().start_monitoring()

    def stop_monitoring(self) -> None:
        """Stop monitoring the IfGoal and cleanup command queues"""
        super().stop_monitoring()
        
        # Cancel command queues if running
        if self.success_command_queue:
            self.success_command_queue.cancel()
        if self.fail_command_queue:
            self.fail_command_queue.cancel()

    def on_interaction(self, interaction) -> None:
        """Handle interactions for the IfGoal"""
        # Let the command queues handle interactions
        if self.success_command_queue and self.success_command_queue.is_running():
            # The individual goals in the queue will handle their own interactions
            pass
        if self.fail_command_queue and self.fail_command_queue.is_running():
            # The individual goals in the queue will handle their own interactions
            pass

    def on_state_change(self, old_state, new_state) -> None:
        """Handle state changes for the IfGoal"""
        # Let the command queues handle state changes
        if self.success_command_queue and self.success_command_queue.is_running():
            # The individual goals in the queue will handle their own state changes
            pass
        if self.fail_command_queue and self.fail_command_queue.is_running():
            # The individual goals in the queue will handle their own state changes
            pass
